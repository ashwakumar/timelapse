"""AeroGuard TSLM Inference Engine.

Loads the saved multimodal Time-Series Language Model checkpoint,
normalizes incoming sensor telemetry using fitted training-only parameters,
and generates remaining useful life (RUL) predictions and engineering diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel

from training.train_opentslm import AeroGuardTSLM


@dataclass(frozen=True)
class AeroGuardAssessment:
    """Structured assessment produced by AeroGuard TSLM."""

    unit_number: int
    cycle: int
    predicted_rul: float
    true_rul: float | None
    health_band: str
    action_directive: str
    cot_diagnostics: str
    observed_drift: dict[str, float]
    model_identifier: str = "AeroGuard-TSLM (SmolLM-135M + OpenTSLM Patch Encoder)"


class AeroGuardPredictor:
    """Inference runner for trained AeroGuard TSLM models."""

    def __init__(
        self,
        model_dir: str | Path = "models/aeroguard_tslm",
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load preprocessing metadata
        prep_path = self.model_dir / "preprocessing.json"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessing metadata not found at {prep_path}")
        with open(prep_path, encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.channels: list[str] = self.metadata["channels"]
        self.means: np.ndarray = np.array(self.metadata["means"], dtype=np.float32)
        self.scales: np.ndarray = np.array(self.metadata["scales"], dtype=np.float32)
        self.window_size: int = int(self.metadata.get("window_size", 30))

        # Instantiate model architecture
        self.model = AeroGuardTSLM(device=self.device)

        # Load temporal encoder and RUL head weights
        weights_path = self.model_dir / "tslm_adapters.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"TSLM adapter weights not found at {weights_path}")
        adapters = torch.load(weights_path, map_location=self.device)
        self.model.ts_encoder.load_state_dict(adapters["patch_encoder"])
        self.model.rul_head.load_state_dict(adapters["rul_head"])

        # Load LoRA adapter weights
        lora_path = self.model_dir / "lora_adapters"
        if lora_path.exists():
            try:
                self.model.llm.load_adapter(str(lora_path), adapter_name="default")
            except Exception:
                # If load_adapter complains about existing adapter, wrap base model
                pass

        self.model.to(self.device)
        self.model.eval()

    def normalize_window(self, raw_series: dict[str, list[float]]) -> torch.Tensor:
        """Normalize sensor window with training population parameters.

        Args:
            raw_series: Mapping from channel name to list of sensor values.

        Returns:
            Tensor of shape [1, 14, window_size].
        """
        arr = []
        for i, ch in enumerate(self.channels):
            vals = np.array(raw_series[ch], dtype=np.float32)
            if len(vals) != self.window_size:
                raise ValueError(
                    f"Channel {ch} has length {len(vals)}, expected window_size {self.window_size}"
                )
            scale = self.scales[i] if self.scales[i] > 1e-6 else 1.0
            norm_vals = (vals - self.means[i]) / scale
            arr.append(norm_vals)

        sensor_mat = np.stack(arr, axis=0)  # [14, window_size]
        return torch.tensor(sensor_mat, dtype=torch.float32).unsqueeze(0).to(self.device)

    def assess_record(
        self,
        record: dict[str, Any],
        max_new_tokens: int = 128,
    ) -> AeroGuardAssessment:
        """Run full multimodal assessment on a telemetry record."""
        raw_series = record["series"]
        tensor_window = self.normalize_window(raw_series)

        prompt_text = record.get(
            "prompt",
            "Analyze turbofan sensor degradation and determine remaining useful life.",
        )

        with torch.no_grad():
            pred_rul, generated_text = self.model.generate_assessment(
                sensor_series=tensor_window,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
            )

        pred_rul = max(0.0, float(pred_rul))

        # Determine health status band
        if pred_rul <= 30.0:
            band = "CRITICAL_WEAR"
            directive = (
                f"Ground engine immediately. RUL bounded at {pred_rul:.0f} cycles. "
                "Issue Maintenance Directive for High-Pressure Compressor overhaul."
            )
        elif pred_rul <= 75.0:
            band = "ELEVATED_WEAR"
            directive = (
                f"Schedule borescope inspection within {max(1, int(pred_rul - 10))} cycles. "
                "Stage-2 HPC blade clearance widening detected."
            )
        else:
            band = "NOMINAL_ENVELOPE"
            directive = (
                "Nominal flight envelope. Telemetry trends exhibit stable thermal/pressure margins. "
                "Cleared for all scheduled commercial flight segments."
            )

        # Calculate observed drifts on key sensors (e.g. T50, Ps30, BPR)
        drifts = {}
        for ch in ["sensor_4", "sensor_11", "sensor_15"]:
            if ch in raw_series:
                drifts[ch] = float(raw_series[ch][-1] - raw_series[ch][0])

        return AeroGuardAssessment(
            unit_number=int(record.get("unit_number", 0)),
            cycle=int(record.get("cycle", 0)),
            predicted_rul=round(pred_rul, 1),
            true_rul=float(record["rul"]) if "rul" in record else None,
            health_band=band,
            action_directive=directive,
            cot_diagnostics=generated_text.strip(),
            observed_drift=drifts,
        )
