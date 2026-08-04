"""Fit probability calibration on each fold's reserved calibration set.

For every configured algorithm, feature variant, and rolling-origin fold it:

1. fits the classifier on the historical fit set;
2. uses the reserved calibration set for early stopping where supported;
3. fits the configured probability calibration transform on calibration scores;
4. applies the frozen transform to the later forward-validation set; and
5. exports raw/calibrated probabilities and calibration-quality metrics.

All controls are in config.py; no command-line arguments are required.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from modeling_utils import (
    calibration_quality_metrics,
    ensure_directories,
    feature_list_for_variant,
    fit_algorithm,
    fit_probability_calibration,
    load_snapshot_dataframe,
    make_algorithm,
    probability_calibration_bins,
    rolling_origin_split,
)


def _output_dir() -> Path:
    """Return and create the probability-calibration output directory."""
    path = config.OUTPUT_DIR / config.CALIBRATION_OUTPUT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metrics_row(
    *,
    algorithm: str,
    variant: str,
    fold: str,
    role: str,
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> dict:
    """Calculate one audit row of raw and calibrated probability metrics."""
    raw_metrics = calibration_quality_metrics(labels, raw)
    calibrated_metrics = calibration_quality_metrics(labels, calibrated)
    row = {
        "algorithm": algorithm,
        "variant": variant,
        "fold": fold,
        "dataset_role": role,
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
    }
    row.update({f"raw_{key}": value for key, value in raw_metrics.items()})
    row.update(
        {f"calibrated_{key}": value for key, value in calibrated_metrics.items()}
    )
    row["brier_improvement"] = raw_metrics["brier"] - calibrated_metrics["brier"]
    row["log_loss_improvement"] = (
        raw_metrics["log_loss"] - calibrated_metrics["log_loss"]
    )
    row["ece_improvement"] = raw_metrics["ece"] - calibrated_metrics["ece"]
    return row


def main() -> None:
    """Evaluate configured probability calibration methods on rolling-origin folds."""
    ensure_directories()
    output_dir = _output_dir()
    dataframe = load_snapshot_dataframe()

    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    parameter_rows: list[dict] = []
    bin_parts: list[pd.DataFrame] = []
    error_rows: list[dict] = []

    for variant in config.CALIBRATION_MODEL_VARIANTS:
        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            features, top_code_detail = feature_list_for_variant(
                dataframe, fit, variant
            )
            x_fit = fit[features].astype(float)
            y_fit = fit[config.TARGET_COLUMN].astype(int).to_numpy()
            x_cal = calibration[features].astype(float)
            y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            x_val = validation[features].astype(float)
            y_val = validation[config.TARGET_COLUMN].astype(int).to_numpy()

            for algorithm in config.CALIBRATION_ALGORITHMS:
                started = time.time()
                print(
                    f"Calibrating algorithm={algorithm}, variant={variant}, "
                    f"fold={fold_name}...",
                    flush=True,
                )
                try:
                    model = make_algorithm(algorithm)
                    model, best_iteration = fit_algorithm(
                        algorithm,
                        model,
                        x_fit,
                        y_fit,
                        x_cal,
                        y_cal,
                    )
                    raw_cal = model.predict_proba(x_cal)[:, 1]
                    raw_val = model.predict_proba(x_val)[:, 1]
                    calibrator = fit_probability_calibration(y_cal, raw_cal)
                    calibrated_cal = calibrator.apply(raw_cal)
                    calibrated_val = calibrator.apply(raw_val)

                    parameter_rows.append(
                        {
                            "algorithm": algorithm,
                            "variant": variant,
                            "fold": fold_name,
                            "calibration_method": (
                                config.PROBABILITY_CALIBRATION_METHOD
                                if config.PROBABILITY_CALIBRATION_ENABLED
                                else "none"
                            ),
                            "platt_coefficient": float(calibrator.coefficient),
                            "platt_intercept": float(calibrator.intercept),
                            "fit_rows": int(len(fit)),
                            "calibration_rows": int(len(calibration)),
                            "validation_rows": int(len(validation)),
                            "feature_count": int(len(features)),
                            "features": "|".join(features),
                            "top_failure_code_features": "|".join(
                                item["feature"] for item in top_code_detail
                            ),
                            "best_iteration": int(best_iteration),
                            "elapsed_seconds": float(time.time() - started),
                        }
                    )

                    for role, frame, labels, raw, calibrated in (
                        (
                            "calibration",
                            calibration,
                            y_cal,
                            raw_cal,
                            calibrated_cal,
                        ),
                        (
                            "forward_validation",
                            validation,
                            y_val,
                            raw_val,
                            calibrated_val,
                        ),
                    ):
                        metric_rows.append(
                            _metrics_row(
                                algorithm=algorithm,
                                variant=variant,
                                fold=fold_name,
                                role=role,
                                labels=labels,
                                raw=raw,
                                calibrated=calibrated,
                            )
                        )
                        predictions = frame[
                            ["machine_key", "snapshot_date", config.TARGET_COLUMN]
                        ].copy()
                        predictions.insert(0, "dataset_role", role)
                        predictions.insert(0, "fold", fold_name)
                        predictions.insert(0, "variant", variant)
                        predictions.insert(0, "algorithm", algorithm)
                        predictions["raw_probability"] = raw
                        predictions["calibrated_probability"] = calibrated
                        prediction_parts.append(predictions)

                        for probability_type, values in (
                            ("raw", raw),
                            ("calibrated", calibrated),
                        ):
                            bins = probability_calibration_bins(labels, values)
                            bins.insert(0, "probability_type", probability_type)
                            bins.insert(0, "dataset_role", role)
                            bins.insert(0, "fold", fold_name)
                            bins.insert(0, "variant", variant)
                            bins.insert(0, "algorithm", algorithm)
                            bin_parts.append(bins)

                    print(
                        f"  coefficient={calibrator.coefficient:.4f}, "
                        f"intercept={calibrator.intercept:.4f}, "
                        f"best_iteration={best_iteration}",
                        flush=True,
                    )
                except Exception as exc:
                    error_rows.append(
                        {
                            "algorithm": algorithm,
                            "variant": variant,
                            "fold": fold_name,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    print(
                        f"  ERROR: {type(exc).__name__}: {exc}", flush=True
                    )

    if not prediction_parts:
        raise RuntimeError(
            "No probability-calibration run completed. See errors in the console."
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    parameters = pd.DataFrame(parameter_rows)
    bins = pd.concat(bin_parts, ignore_index=True)

    predictions.to_csv(
        output_dir / "probability_calibration_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metrics.to_csv(
        output_dir / "probability_calibration_metrics.csv", index=False
    )
    parameters.to_csv(
        output_dir / "probability_calibration_parameters.csv", index=False
    )
    bins.to_csv(output_dir / "probability_calibration_bins.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(
            output_dir / "probability_calibration_errors.csv", index=False
        )

    summary = (
        metrics.groupby(
            ["algorithm", "variant", "dataset_role"], as_index=False
        )
        .agg(
            fold_count=("fold", "nunique"),
            mean_positive_rate=("positive_rate", "mean"),
            mean_raw_brier=("raw_brier", "mean"),
            mean_calibrated_brier=("calibrated_brier", "mean"),
            mean_brier_improvement=("brier_improvement", "mean"),
            mean_raw_log_loss=("raw_log_loss", "mean"),
            mean_calibrated_log_loss=("calibrated_log_loss", "mean"),
            mean_log_loss_improvement=("log_loss_improvement", "mean"),
            mean_raw_ece=("raw_ece", "mean"),
            mean_calibrated_ece=("calibrated_ece", "mean"),
            mean_ece_improvement=("ece_improvement", "mean"),
            mean_roc_auc=("calibrated_roc_auc", "mean"),
            mean_average_precision=("calibrated_average_precision", "mean"),
        )
    )
    summary.to_csv(
        output_dir / "probability_calibration_summary.csv", index=False
    )
    print(f"Probability calibration outputs: {output_dir}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
