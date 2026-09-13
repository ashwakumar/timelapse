"""Load saved baselines for GUI, benchmark, or single-window CLI inference."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from training.baseline_features import (
    ACTIVE_SENSORS,
    extract_chronos_representations,
    extract_tabular_features,
)

MODEL_NAMES = {"Chronos + Ridge", "Text-only LM"}


class BaselinePredictor:
    def __init__(self, name: str, baseline_dir: str = "models/baselines", device: str = "cpu"):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown baseline: {name}")
        root = Path(baseline_dir)
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(
                "Saved baselines missing. Run: uv run python -m training.train_baselines"
            )
        self.metadata = json.loads(manifest.read_text())
        if (
            self.metadata.get("schema_version") != 1
            or self.metadata.get("sensors") != ACTIVE_SENSORS
        ):
            raise ValueError("Incompatible baseline manifest; prepare the baselines again")
        self.name, self.device = name, device
        if name == "Chronos + Ridge":
            from chronos import ChronosPipeline

            self.model = ChronosPipeline.from_pretrained(
                str(root / "chronos"), device_map=device, local_files_only=True
            )
            self.model.model.eval()
            with np.load(root / "chronos_ridge.npz", allow_pickle=False) as weights:
                self.coef = weights["coef"].copy()
                self.intercept = weights["intercept"].copy()
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                root / "text_model", local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                root / "text_model", local_files_only=True
            )
            cast(torch.nn.Module, self.model).to(device)
            self.model.eval()

    def predict(self, records: list[dict], batch_size: int = 16) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not records:
            return np.array([], dtype=float)
        if self.name == "Chronos + Ridge":
            features, _ = extract_chronos_representations(
                self.model, records, batch_size, self.device
            )
            predictions = features @ self.coef + self.intercept
        else:
            predictions = text_predictions(records, self.model, self.tokenizer, self.device)
        return np.clip(np.asarray(predictions, dtype=float), 0, None)


def text_predictions(records: list[dict], model: Any, tokenizer: Any, device: str) -> np.ndarray:
    """Run a frozen text-only LM on sensor summaries without exposing test labels."""
    predictions = []
    for i, record in enumerate(records):
        features, _ = extract_tabular_features([record])
        rows = features[0].reshape(len(ACTIVE_SENSORS), 5)
        summary = "\n".join(
            f"{sensor}: " + ", ".join(f"{v:.4g}" for v in row)
            for sensor, row in zip(ACTIVE_SENSORS, rows, strict=True)
        )
        prompt = (
            "Estimate turbofan remaining useful life in cycles from these sensor summaries. "
            "Columns: mean, standard deviation, minimum, maximum, final minus initial.\n"
            f"{summary}\nReply with only one nonnegative number."
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if not isinstance(answer, str):
            raise TypeError("Expected one decoded text response")
        answer = answer.strip()
        match = re.fullmatch(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", answer)
        predictions.append(float(answer) if match else float("nan"))
        if (i + 1) % 100 == 0:
            print(f"Text-only LM: {i + 1}/{len(records)} windows", flush=True)
    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--baseline-dir", default="models/baselines")
    parser.add_argument("--data-path", default="data/processed/windows.jsonl")
    parser.add_argument("--unit", type=int, required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with open(args.data_path) as handle:
        record = next(
            (
                r
                for line in handle
                if line.strip()
                for r in [json.loads(line)]
                if r["unit_number"] == args.unit and r["cycle"] == args.cycle
            ),
            None,
        )
    if record is None:
        parser.error("No window matches that engine and cycle")
    predictor = BaselinePredictor(args.model, args.baseline_dir, args.device)
    prediction = float(predictor.predict([record])[0])
    print(
        json.dumps(
            {
                "model": args.model,
                "unit": args.unit,
                "cycle": args.cycle,
                "predicted_rul": prediction if np.isfinite(prediction) else None,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
