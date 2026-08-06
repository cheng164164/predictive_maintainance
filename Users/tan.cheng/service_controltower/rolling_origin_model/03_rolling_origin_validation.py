"""Run honest rolling-origin validation for algorithms and feature variants.

Each fold separates historical fitting data, a reserved calibration period, and
later forward-validation data. Probability calibration and the operating
threshold are fitted only on the calibration period before the validation rows
are evaluated.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd

import config
from modeling_utils import (
    choose_configured_threshold,
    ensure_directories,
    evaluate_predictions,
    feature_list_for_variant,
    fit_algorithm,
    fit_probability_calibration,
    load_snapshot_dataframe,
    make_algorithm,
    risk_decile_table,
    rolling_origin_split,
    summarize_oof_predictions,
    top_k_table,
)

warnings.filterwarnings("ignore")


def evaluate_algorithm_comparison(dataframe: pd.DataFrame) -> None:
    """Compare configured algorithms on the common base feature set."""
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict] = []

    for algorithm_name in config.ALGORITHMS:
        print(f"\nAlgorithm: {algorithm_name}", flush=True)
        try:
            make_algorithm(algorithm_name)
        except ImportError as exc:
            print(f"  skipped: {exc}", flush=True)
            skipped_rows.append({"algorithm": algorithm_name, "reason": str(exc)})
            continue

        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            features, top_code_detail = feature_list_for_variant(
                dataframe, fit, "base27"
            )
            model = make_algorithm(algorithm_name)
            started = time.time()
            model, best_iteration = fit_algorithm(
                algorithm_name,
                model,
                fit[features].astype(float),
                fit[config.TARGET_COLUMN].astype(int).to_numpy(),
                calibration[features].astype(float),
                calibration[config.TARGET_COLUMN].astype(int).to_numpy(),
            )

            raw_calibration = model.predict_proba(
                calibration[features].astype(float)
            )[:, 1]
            raw_validation = model.predict_proba(
                validation[features].astype(float)
            )[:, 1]
            calibration_labels = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            validation_labels = validation[config.TARGET_COLUMN].astype(int).to_numpy()
            calibrator = fit_probability_calibration(
                calibration_labels, raw_calibration
            )
            calibrated_calibration = calibrator.apply(raw_calibration)
            calibrated_validation = calibrator.apply(raw_validation)
            threshold, threshold_score, _ = choose_configured_threshold(
                calibration_labels, calibrated_calibration
            )
            metrics = evaluate_predictions(
                validation_labels,
                raw_validation,
                calibrated_validation,
                threshold,
            )

            metric_rows.append(
                {
                    "algorithm": algorithm_name,
                    "fold": fold_name,
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "validation_rows": len(validation),
                    "validation_base_rate": float(validation_labels.mean()),
                    "feature_count": len(features),
                    "threshold": threshold,
                    "threshold_metric": config.THRESHOLD_SELECTION_METRIC,
                    "calibration_threshold_score": threshold_score,
                    "best_iteration": best_iteration,
                    "elapsed_seconds": time.time() - started,
                    "selected_fault_features": "|".join(
                        item["feature"] for item in top_code_detail
                    ),
                    "features": "|".join(features),
                    **metrics,
                }
            )
            predictions = validation[
                ["machine_key", "snapshot_date", config.TARGET_COLUMN]
            ].copy()
            predictions["algorithm"] = algorithm_name
            predictions["fold"] = fold_name
            predictions["score"] = raw_validation
            predictions["calibrated_probability"] = calibrated_validation
            predictions["prediction"] = (
                calibrated_validation >= threshold
            ).astype("int8")
            prediction_frames.append(predictions)

            print(
                f"  {fold_name}: AUC={metrics['roc_auc']:.4f}, "
                f"AP={metrics['average_precision']:.4f}, "
                f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                f"F1={metrics['f1']:.3f}, Top10P={metrics['precision_top10']:.3f}",
                flush=True,
            )

    if not metric_rows:
        raise RuntimeError("No configured algorithm completed successfully.")
    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_oof_predictions(
        fold_metrics, oof_predictions, grouping_column="algorithm"
    )
    fold_metrics.to_csv(config.OUTPUT_DIR / "algorithm_fold_metrics.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "algorithm_summary.csv", index=False)
    if config.WRITE_OOF_PREDICTIONS:
        oof_predictions.to_csv(
            config.OUTPUT_DIR / "algorithm_oof_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            config.OUTPUT_DIR / "algorithm_skipped.csv", index=False
        )
    print("\nAlgorithm summary")
    print(summary.to_string(index=False))


def evaluate_xgboost_variants(dataframe: pd.DataFrame) -> None:
    """Compare configured XGBoost feature variants on identical folds."""
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for variant in config.MODEL_VARIANTS:
        print(f"\nXGBoost variant: {variant}", flush=True)
        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            features, top_code_detail = feature_list_for_variant(
                dataframe, fit, variant
            )
            model = make_algorithm("xgboost")
            started = time.time()
            model, best_iteration = fit_algorithm(
                "xgboost",
                model,
                fit[features].astype(float),
                fit[config.TARGET_COLUMN].astype(int).to_numpy(),
                calibration[features].astype(float),
                calibration[config.TARGET_COLUMN].astype(int).to_numpy(),
            )

            raw_calibration = model.predict_proba(
                calibration[features].astype(float)
            )[:, 1]
            raw_validation = model.predict_proba(
                validation[features].astype(float)
            )[:, 1]
            calibration_labels = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            validation_labels = validation[config.TARGET_COLUMN].astype(int).to_numpy()
            calibrator = fit_probability_calibration(
                calibration_labels, raw_calibration
            )
            calibrated_calibration = calibrator.apply(raw_calibration)
            calibrated_validation = calibrator.apply(raw_validation)
            threshold, threshold_score, _ = choose_configured_threshold(
                calibration_labels, calibrated_calibration
            )
            metrics = evaluate_predictions(
                validation_labels,
                raw_validation,
                calibrated_validation,
                threshold,
            )

            metric_rows.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "validation_rows": len(validation),
                    "validation_base_rate": float(validation_labels.mean()),
                    "feature_count": len(features),
                    "threshold": threshold,
                    "threshold_metric": config.THRESHOLD_SELECTION_METRIC,
                    "calibration_threshold_score": threshold_score,
                    "best_iteration": best_iteration,
                    "elapsed_seconds": time.time() - started,
                    "selected_fault_features": "|".join(
                        item["feature"] for item in top_code_detail
                    ),
                    "features": "|".join(features),
                    **metrics,
                }
            )
            predictions = validation[
                ["machine_key", "snapshot_date", config.TARGET_COLUMN]
            ].copy()
            predictions["variant"] = variant
            predictions["fold"] = fold_name
            predictions["score"] = raw_validation
            predictions["calibrated_probability"] = calibrated_validation
            predictions["prediction"] = (
                calibrated_validation >= threshold
            ).astype("int8")
            prediction_frames.append(predictions)
            print(
                f"  {fold_name}: AUC={metrics['roc_auc']:.4f}, "
                f"AP={metrics['average_precision']:.4f}, "
                f"Top10P={metrics['precision_top10']:.3f}, "
                f"features={len(features)}",
                flush=True,
            )

    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_oof_predictions(
        fold_metrics, oof_predictions, grouping_column="variant"
    )
    fold_metrics.to_csv(
        config.OUTPUT_DIR / "xgboost_variant_fold_metrics.csv", index=False
    )
    summary.to_csv(
        config.OUTPUT_DIR / "xgboost_variant_summary.csv", index=False
    )
    top_k_table(oof_predictions, grouping_column="variant").to_csv(
        config.OUTPUT_DIR / "top_k_ranking_metrics.csv", index=False
    )
    risk_decile_table(oof_predictions, grouping_column="variant").to_csv(
        config.OUTPUT_DIR / "risk_decile_lift.csv", index=False
    )
    if config.WRITE_OOF_PREDICTIONS:
        oof_predictions.to_csv(
            config.OUTPUT_DIR / "xgboost_variant_oof_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    print("\nXGBoost variant summary")
    print(summary.to_string(index=False))


def main() -> None:
    """Execute the complete rolling-origin model validation stage."""
    ensure_directories()
    dataframe = load_snapshot_dataframe()
    print(
        f"Loaded {len(dataframe):,} snapshots, "
        f"{dataframe['machine_key'].nunique():,} machines, "
        f"positive rate={dataframe[config.TARGET_COLUMN].mean():.2%}",
        flush=True,
    )
    evaluate_algorithm_comparison(dataframe)
    evaluate_xgboost_variants(dataframe)
    print("\nRolling-origin validation complete.")


if __name__ == "__main__":
    main()
