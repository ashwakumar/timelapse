#!/usr/bin/env python3
"""Exploratory VitalDB feature screen for fifteen-minute hypotension onset.

This is a development-set screen, not a clinical validation. It ranks
20-second history summaries by held-out-patient permutation importance for an
onset within 15 minutes, and draws a Spearman correlation heatmap for the
twenty highest-ranked summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/cache/matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import vitaldb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from tqdm import tqdm


TARGET_TRACK = "Solar8000/ART_MBP"
TRACKS = {
    "MAP": TARGET_TRACK,
    "HR": "Solar8000/HR",
    "SpO2": "Solar8000/PLETH_SPO2",
    "EtCO2": "Solar8000/ETCO2",
    "RR_CO2": "Solar8000/RR_CO2",
    "FiO2": "Solar8000/FIO2",
    "FeO2": "Solar8000/FEO2",
    "InCO2": "Solar8000/INCO2",
    "Temperature": "Solar8000/BT",
    "ST_II": "Solar8000/ST_II",
    "MAC": "Primus/MAC",
    "ExpSevo": "Primus/EXP_SEVO",
    "ExpDes": "Primus/EXP_DES",
    "PIP": "Primus/PIP_MBAR",
    "PEEP": "Primus/PEEP_MBAR",
    "MeanAirwayPressure": "Primus/MAWP_MBAR",
    "PlateauPressure": "Primus/PPLAT_MBAR",
    "MinuteVolume": "Primus/MV",
    "TidalVolume": "Primus/TV",
    "Compliance": "Primus/COMPLIANCE",
    "VentLeak": "Primus/VENT_LEAK",
    "SetRespRate": "Primus/SET_RR_IPPV",
    "SetTidalVolume": "Primus/SET_TV_L",
    "BIS": "BIS/BIS",
    "BIS_SQI": "BIS/SQI",
    "BIS_EMG": "BIS/EMG",
    "BIS_SR": "BIS/SR",
    "BIS_SEF": "BIS/SEF",
    "RemiRate": "Orchestra/RFTN20_RATE",
    "RemiEffectSite": "Orchestra/RFTN20_CE",
    "PropofolRate": "Orchestra/PPF20_RATE",
    "PropofolEffectSite": "Orchestra/PPF20_CE",
}

STATIC_COLUMNS = {
    "Age": "age",
    "Height": "height",
    "Weight": "weight",
    "BMI": "bmi",
    "ASA": "asa",
    "EmergencyOperation": "emop",
}

PLAUSIBLE_RANGES = {
    "MAP": (20, 200),
    "HR": (20, 220),
    "SpO2": (50, 100),
    "EtCO2": (5, 100),
    "RR_CO2": (1, 80),
    "FiO2": (10, 100),
    "BIS": (0, 100),
    "BIS_SQI": (0, 100),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--history-sec", type=int, default=20)
    parser.add_argument("--horizon-sec", type=int, default=900)
    parser.add_argument("--stride-sec", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_screen"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/vitaldb_feature_screen"))
    return parser.parse_args()


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        frame.loc[~frame[column].between(low, high), column] = np.nan
    return frame


def load_case(case_id: int, interval: int, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    track_version = hashlib.sha256("\n".join(TRACKS.values()).encode()).hexdigest()[:8]
    path = cache_dir / f"case_{case_id}_i{interval}_{track_version}.pkl.gz"
    if path.exists():
        return pd.read_pickle(path)
    values = vitaldb.load_case(case_id, list(TRACKS.values()), interval=interval)
    frame = clean_frame(pd.DataFrame(values, columns=list(TRACKS)))
    frame.to_pickle(path, compression="gzip")
    return frame


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def first_true_run_start(values: np.ndarray, required: int) -> int | None:
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        if current >= required:
            return index - required + 1
    return None


def slope(values: np.ndarray, interval: int) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return np.nan
    x = np.arange(values.size, dtype=float)[valid] * interval / 60.0
    return float(np.polyfit(x, values[valid], 1)[0])


def summarize(history: pd.DataFrame, interval: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for column in history:
        values = history[column].to_numpy(dtype=float)
        valid = np.isfinite(values)
        observed = values[valid]
        row[f"{column}__coverage"] = float(valid.mean())
        if observed.size:
            row[f"{column}__last"] = float(observed[-1])
            row[f"{column}__mean"] = float(observed.mean())
            row[f"{column}__std"] = float(observed.std())
            row[f"{column}__slope_per_min"] = slope(values, interval)
        else:
            for suffix in ("last", "mean", "std", "slope_per_min"):
                row[f"{column}__{suffix}"] = np.nan
    return row


def make_windows(
    frame: pd.DataFrame,
    case_id: int,
    subject_id: int,
    static_context: dict[str, float],
    interval: int,
    history_sec: int,
    horizon_sec: int,
    stride_sec: int,
) -> list[dict[str, float]]:
    history_n, horizon_n, stride_n = (
        history_sec // interval,
        horizon_sec // interval,
        stride_sec // interval,
    )
    required_low_n = 60 // interval
    rows = []
    for cutoff in range(max(history_n, required_low_n), len(frame) - horizon_n + 1, stride_n):
        history = frame.iloc[cutoff - history_n : cutoff]
        prior_map = frame["MAP"].iloc[cutoff - required_low_n : cutoff].to_numpy(dtype=float)
        future_map = frame["MAP"].iloc[cutoff : cutoff + horizon_n].to_numpy(dtype=float)
        if history["MAP"].notna().mean() < 0.8 or history["HR"].notna().mean() < 0.8:
            continue
        if np.isfinite(future_map).mean() < 0.8:
            continue
        # Exclude cutoffs where a sustained episode is already in progress.
        prior_low = np.isfinite(prior_map) & (prior_map < 65)
        if longest_true_run(prior_low) >= required_low_n:
            continue
        low = np.isfinite(future_map) & (future_map < 65)
        onset_index = first_true_run_start(low, required_low_n)
        onset_sec = None if onset_index is None else onset_index * interval
        if onset_sec is None:
            horizon_bucket = "none_within_15"
        elif onset_sec <= 180:
            horizon_bucket = "within_3"
        elif onset_sec <= 300:
            horizon_bucket = "within_5"
        elif onset_sec <= 600:
            horizon_bucket = "within_10"
        else:
            horizon_bucket = "within_15"
        row = summarize(history, interval)
        row.update({f"{name}__value": value for name, value in static_context.items()})
        row.update(
            case_id=case_id,
            subject_id=subject_id,
            cutoff_sec=cutoff * interval,
            hypotension_within_15min=int(onset_index is not None),
            onset_seconds=onset_sec,
            onset_bucket=horizon_bucket,
        )
        rows.append(row)
    return rows


def patient_split(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + offset)
        train_idx, valid_idx = next(
            splitter.split(data, data["hypotension_within_15min"], groups=data["subject_id"])
        )
        if data.iloc[train_idx]["hypotension_within_15min"].nunique() == 2 and data.iloc[valid_idx]["hypotension_within_15min"].nunique() == 2:
            return train_idx, valid_idx
    raise RuntimeError("Could not form patient-disjoint splits containing both target classes")


def grouped_parameter_importance(
    estimator: Pipeline,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    baseline_auprc: float,
    seed: int,
    repeats: int = 8,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for parameter in [*TRACKS, *STATIC_COLUMNS]:
        columns = [column for column in valid_x if column.startswith(f"{parameter}__")]
        if not columns:
            continue
        drops = []
        for _ in range(repeats):
            shuffled = valid_x.copy()
            order = rng.permutation(len(shuffled))
            shuffled.loc[:, columns] = shuffled[columns].to_numpy()[order]
            score = average_precision_score(valid_y, estimator.predict_proba(shuffled)[:, 1])
            drops.append(baseline_auprc - score)
        coverage_column = f"{parameter}__coverage"
        if coverage_column in valid_x:
            adequate_coverage = float((valid_x[coverage_column] >= 0.8).mean())
            median_coverage = float(valid_x[coverage_column].median())
        else:
            adequate_coverage = float(valid_x[columns].notna().all(axis=1).mean())
            median_coverage = adequate_coverage
        rows.append(
            {
                "parameter": parameter,
                "grouped_permutation_auprc_drop_mean": float(np.mean(drops)),
                "grouped_permutation_auprc_drop_std": float(np.std(drops)),
                "validation_windows_at_least_80pct_observed": adequate_coverage,
                "validation_median_sample_coverage": median_coverage,
                "summary_features": len(columns),
            }
        )
    return pd.DataFrame(rows).sort_values("grouped_permutation_auprc_drop_mean", ascending=False)


def make_estimator(features: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", ColumnTransformer([("numeric", SimpleImputer(strategy="median", add_indicator=True), features)])),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(vitaldb.find_cases([TARGET_TRACK, TRACKS["HR"]]))
    clinical_fields = ["caseid", "subjectid", *STATIC_COLUMNS.values()]
    clinical = vitaldb.load_clinical_data(caseids=case_ids, params=clinical_fields)
    subject_by_case = clinical.set_index("caseid")["subjectid"].to_dict()
    context_by_case = clinical.set_index("caseid").to_dict(orient="index")
    rng = np.random.default_rng(args.seed)
    # A stable prefix means expanding --max-cases reuses the existing cache.
    selected = rng.permutation(case_ids)[: min(args.max_cases, len(case_ids))]

    rows: list[dict[str, float]] = []
    failures = []
    for case_id in tqdm(selected, desc="Loading VitalDB cases"):
        try:
            frame = load_case(int(case_id), args.interval, args.cache_dir)
            static_context = {}
            for name, source_column in STATIC_COLUMNS.items():
                try:
                    static_context[name] = float(context_by_case[int(case_id)][source_column])
                except (KeyError, TypeError, ValueError):
                    static_context[name] = np.nan
            rows.extend(
                make_windows(
                    frame,
                    int(case_id),
                    int(subject_by_case[int(case_id)]),
                    static_context,
                    args.interval,
                    args.history_sec,
                    args.horizon_sec,
                    args.stride_sec,
                )
            )
        except Exception as error:  # preserve failed case IDs in the report
            failures.append({"case_id": int(case_id), "error": repr(error)})

    data = pd.DataFrame(rows)
    if data.empty or data["hypotension_within_15min"].nunique() < 2:
        raise RuntimeError("Sample did not produce both classes; increase --max-cases")

    train_idx, valid_idx = patient_split(data, args.seed)
    train, valid = data.iloc[train_idx], data.iloc[valid_idx]
    metadata = {"case_id", "subject_id", "cutoff_sec", "hypotension_within_15min", "onset_seconds", "onset_bucket"}
    candidate_features = [column for column in data.columns if column not in metadata]
    # Drop almost-empty and constant summaries using training data only.
    features = [
        column
        for column in candidate_features
        if train[column].notna().mean() >= 0.20 and train[column].nunique(dropna=True) > 1
    ]

    estimator = make_estimator(features, args.seed)
    estimator.fit(train[features], train["hypotension_within_15min"])
    probabilities = estimator.predict_proba(valid[features])[:, 1]
    baseline_auprc = average_precision_score(valid["hypotension_within_15min"], probabilities)
    importance = permutation_importance(
        estimator,
        valid[features],
        valid["hypotension_within_15min"],
        scoring="average_precision",
        n_repeats=15,
        random_state=args.seed,
        n_jobs=-1,
    )
    ranking = pd.DataFrame(
        {
            "feature": features,
            "permutation_auprc_drop_mean": importance.importances_mean,
            "permutation_auprc_drop_std": importance.importances_std,
            "train_coverage": [train[column].notna().mean() for column in features],
        }
    ).sort_values("permutation_auprc_drop_mean", ascending=False)
    parameter_ranking = grouped_parameter_importance(
        estimator,
        valid[features],
        valid["hypotension_within_15min"],
        baseline_auprc,
        args.seed,
    )
    reliable_parameter_ranking = parameter_ranking[
        parameter_ranking["validation_windows_at_least_80pct_observed"] >= 0.5
    ]
    top_parameters = reliable_parameter_ranking.head(min(20, len(reliable_parameter_ranking)))["parameter"].tolist()
    heatmap_features = []
    for parameter in top_parameters:
        for suffix in ("mean", "value", "last"):
            feature = f"{parameter}__{suffix}"
            if feature in features:
                heatmap_features.append(feature)
                break

    heatmap_data = train[heatmap_features + ["hypotension_within_15min"]].corr(method="spearman")
    plt.figure(figsize=(16, 14))
    sns.heatmap(heatmap_data, cmap="vlag", center=0, vmin=-1, vmax=1, square=True)
    plt.title("VitalDB development cohort: top feature relationships (Spearman)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "top20_feature_heatmap.png", dpi=180)
    plt.close()

    data.to_csv(args.output_dir / "window_features.csv.gz", index=False, compression="gzip")
    ranking.to_csv(args.output_dir / "feature_ranking.csv", index=False)
    parameter_ranking.to_csv(args.output_dir / "parameter_ranking.csv", index=False)
    reliable_parameter_ranking.to_csv(args.output_dir / "reliable_parameter_ranking.csv", index=False)
    ablation_groups = {
        "MAP": ["MAP"],
        "MAP+basic": ["MAP", "HR", "SpO2", "EtCO2", "RR_CO2"],
        "MAP+BIS": ["MAP", "BIS", "BIS_SQI", "BIS_EMG", "BIS_SR", "BIS_SEF"],
        "MAP+remifentanil": ["MAP", "RemiRate", "RemiEffectSite"],
        "MAP+basic+BIS+remifentanil": ["MAP", "HR", "SpO2", "EtCO2", "RR_CO2", "BIS", "BIS_SQI", "BIS_EMG", "BIS_SR", "BIS_SEF", "RemiRate", "RemiEffectSite"],
    }
    ablation_rows = []
    for group_name, parameters in ablation_groups.items():
        group_features = [column for column in features if any(column.startswith(f"{parameter}__") for parameter in parameters)]
        group_model = make_estimator(group_features, args.seed)
        group_model.fit(train[group_features], train["hypotension_within_15min"])
        group_probability = group_model.predict_proba(valid[group_features])[:, 1]
        ablation_rows.append(
            {
                "group": group_name,
                "summary_features": len(group_features),
                "validation_auprc": average_precision_score(valid["hypotension_within_15min"], group_probability),
                "validation_auroc": roc_auc_score(valid["hypotension_within_15min"], group_probability),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(args.output_dir / "group_ablation.csv", index=False)
    report = {
        "definition": "onset of MAP <65 mmHg for >=60 contiguous observed seconds within 15 minutes",
        "sampled_cases": int(len(selected)),
        "usable_cases": int(data["case_id"].nunique()),
        "usable_subjects": int(data["subject_id"].nunique()),
        "windows": int(len(data)),
        "positive_windows": int(data["hypotension_within_15min"].sum()),
        "onset_bucket_counts": data["onset_bucket"].value_counts().to_dict(),
        "train_subjects": int(train["subject_id"].nunique()),
        "validation_subjects": int(valid["subject_id"].nunique()),
        "validation_auprc": float(baseline_auprc),
        "validation_auroc": float(roc_auc_score(valid["hypotension_within_15min"], probabilities)),
        "top20_parameters": top_parameters,
        "failed_cases": failures,
        "warning": "Exploratory patient-held-out screen; validation was used for ranking and is not a final test set.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
