"""Select the configured operating threshold from calibration predictions.

The probability calibrator is fitted by 04_calibrate_probabilities.py. This
script selects one threshold per algorithm/variant/fold using calibration rows
only, freezes it, and evaluates the threshold on the later forward-validation
rows. Ranking metrics such as ROC AUC and Top-K remain threshold-free.

All controls are in config.py; no command-line arguments are required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

import config
from modeling_utils import select_operating_threshold


def _calibration_dir() -> Path:
    """Return the directory containing probability-calibration artifacts."""
    return config.OUTPUT_DIR / config.CALIBRATION_OUTPUT_SUBDIR


def _output_dir() -> Path:
    """Return and create the operating-threshold output directory."""
    path = config.OUTPUT_DIR / config.THRESHOLD_OUTPUT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evaluate_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    """Calculate classification metrics for one probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "threshold": float(threshold),
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "flagged_rows": int(predictions.sum()),
        "flagged_rate": float(predictions.mean()),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "f2": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def main() -> None:
    """Select and audit the configured F1 or F2 operating threshold."""
    calibration_dir = _calibration_dir()
    input_path = calibration_dir / "probability_calibration_predictions.csv.gz"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path}. Run 04_calibrate_probabilities.py first."
        )
    output_dir = _output_dir()
    predictions = pd.read_csv(
        input_path, parse_dates=["snapshot_date"], low_memory=False
    )

    selected_rows: list[dict] = []
    role_metric_rows: list[dict] = []
    curve_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []

    group_columns = ["algorithm", "variant", "fold"]
    for keys, group in predictions.groupby(group_columns, sort=True):
        algorithm, variant, fold = keys
        calibration = group[group["dataset_role"].eq("calibration")].copy()
        if calibration.empty:
            raise ValueError(
                f"No calibration rows for algorithm={algorithm}, "
                f"variant={variant}, fold={fold}."
            )
        y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
        p_cal = calibration["calibrated_probability"].astype(float).to_numpy()
        threshold, selected_score, curve = select_operating_threshold(
            y_cal,
            p_cal,
            metric=config.THRESHOLD_SELECTION_METRIC,
        )
        selected_curve_row = curve.loc[curve["is_selected"]].iloc[0]
        selected_rows.append(
            {
                "algorithm": algorithm,
                "variant": variant,
                "fold": fold,
                "selection_dataset_role": "calibration",
                "threshold_metric": config.THRESHOLD_SELECTION_METRIC,
                "selected_threshold": float(threshold),
                "selected_metric_value": float(selected_score),
                "calibration_precision": float(selected_curve_row["precision"]),
                "calibration_recall": float(selected_curve_row["recall"]),
                "calibration_f1": float(selected_curve_row["f1"]),
                "calibration_f2": float(selected_curve_row["f2"]),
                "calibration_flagged_rows": int(
                    selected_curve_row["flagged_rows"]
                ),
                "calibration_flagged_rate": float(
                    selected_curve_row["flagged_rate"]
                ),
                "calibration_rows": int(len(calibration)),
                "calibration_positive_rows": int(y_cal.sum()),
            }
        )
        curve.insert(0, "fold", fold)
        curve.insert(0, "variant", variant)
        curve.insert(0, "algorithm", algorithm)
        curve_parts.append(curve)

        for role, role_frame in group.groupby("dataset_role", sort=True):
            labels = role_frame[config.TARGET_COLUMN].astype(int).to_numpy()
            probabilities = role_frame["calibrated_probability"].astype(float).to_numpy()
            role_metric_rows.append(
                {
                    "algorithm": algorithm,
                    "variant": variant,
                    "fold": fold,
                    "dataset_role": role,
                    "threshold_metric": config.THRESHOLD_SELECTION_METRIC,
                    **_evaluate_threshold(labels, probabilities, threshold),
                }
            )
            scored = role_frame.copy()
            scored["selected_threshold"] = threshold
            scored["threshold_metric"] = config.THRESHOLD_SELECTION_METRIC
            scored["prediction_label"] = (
                scored["calibrated_probability"] >= threshold
            ).astype("int8")
            scored["prediction_outcome"] = np.select(
                [
                    (scored[config.TARGET_COLUMN] == 1)
                    & (scored["prediction_label"] == 1),
                    (scored[config.TARGET_COLUMN] == 0)
                    & (scored["prediction_label"] == 1),
                    (scored[config.TARGET_COLUMN] == 1)
                    & (scored["prediction_label"] == 0),
                ],
                ["TP", "FP", "FN"],
                default="TN",
            )
            prediction_parts.append(scored)

        print(
            f"algorithm={algorithm}, variant={variant}, fold={fold}: "
            f"threshold={threshold:.6f}, calibration "
            f"F1={selected_curve_row['f1']:.4f}, "
            f"P={selected_curve_row['precision']:.4f}, "
            f"R={selected_curve_row['recall']:.4f}",
            flush=True,
        )

    selected = pd.DataFrame(selected_rows)
    role_metrics = pd.DataFrame(role_metric_rows)
    curves = pd.concat(curve_parts, ignore_index=True)
    scored_predictions = pd.concat(prediction_parts, ignore_index=True)

    selected.to_csv(output_dir / "selected_operating_thresholds.csv", index=False)
    role_metrics.to_csv(
        output_dir / "threshold_metrics_by_fold_and_role.csv", index=False
    )
    curves.to_csv(
        output_dir / "operating_threshold_search_curves.csv.gz",
        index=False,
        compression="gzip",
    )
    scored_predictions.to_csv(
        output_dir / "threshold_scored_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = (
        role_metrics.groupby(
            ["algorithm", "variant", "dataset_role"], as_index=False
        )
        .agg(
            fold_count=("fold", "nunique"),
            mean_threshold=("threshold", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_f2=("f2", "mean"),
            mean_flagged_rate=("flagged_rate", "mean"),
        )
    )
    summary.to_csv(output_dir / "operating_threshold_summary.csv", index=False)
    print(f"Operating-threshold outputs: {output_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
