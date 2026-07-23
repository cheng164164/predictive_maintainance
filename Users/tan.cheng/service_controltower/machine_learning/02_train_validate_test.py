"""Step 02: train final model(s), choose threshold on validation, evaluate test once."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from ml_utils import (
    apply_sentinel_cleaning,
    apply_snapshot_training_filters,
    chronological_split,
    ensure_dir,
    fit_transform_prepared_features,
    machine_level_top_k_metrics,
    make_algorithm_params,
    metrics_at_threshold,
    model_feature_importance,
    predict_probability,
    prediction_frame,
    prepared_features_for_available_sources,
    read_json,
    read_snapshot,
    save_model_artifacts,
    select_best_threshold,
    source_features_for_model_variants,
    source_features_for_prepared_features,
    target_series,
    threshold_free_metrics,
    threshold_grid,
    threshold_search,
    top_k_metrics,
    train_classifier,
    validate_source_features,
    write_json,
)



def _ordered_unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _snapshot_filter_feature_columns():
    """Return source columns used to detect empty or inactive snapshot rows."""

    override = list(getattr(config, "SNAPSHOT_SPARSITY_FEATURE_COLUMNS", []) or [])
    if override:
        return _ordered_unique(override)

    variants = list(getattr(config, "MODEL_VARIANTS_TO_RUN", []) or [])
    variants.append(getattr(config, "FINAL_MODEL_VARIANT", None))

    return source_features_for_model_variants(
        feature_sets=config.FEATURE_SETS,
        prepared_to_source=config.PREPARED_TO_SOURCE_FEATURE,
        model_variants=_ordered_unique(variants),
    )

def _load_and_prepare_snapshot():
    df = read_snapshot(config.INPUT_DATA_PATH, config.DATE_COL)
    source_cols_to_drop = [
        c for c in getattr(config, "SOURCE_COLUMNS_TO_DROP_BEFORE_MODELING", []) if c in df.columns
    ]
    if source_cols_to_drop:
        df = df.drop(columns=source_cols_to_drop)

    df, sentinel_report = apply_sentinel_cleaning(
        df=df,
        enabled=getattr(config, "SENTINEL_CLEANING_ENABLED", False),
        sentinel_value=getattr(config, "SENTINEL_VALUE", 9999),
        columns_to_clean=getattr(config, "SENTINEL_COLUMNS_TO_CLEAN", {}),
        replace_with=getattr(config, "SENTINEL_REPLACE_WITH", None),
    )

    df, snapshot_filter_report = apply_snapshot_training_filters(
        df=df,
        date_col=config.DATE_COL,
        target_col=config.TARGET_COL,
        id_cols=config.ID_COLS,
        feature_columns=_snapshot_filter_feature_columns(),
        enabled=getattr(config, "SNAPSHOT_TRAINING_FILTERS_ENABLED", False),
        drop_after_cutoff=getattr(config, "DROP_SNAPSHOTS_AFTER_FULL_TARGET_WINDOW", False),
        cutoff_reference_end_date=getattr(config, "SNAPSHOT_CUTOFF_REFERENCE_END_DATE", None),
        prediction_horizon_days=getattr(config, "SNAPSHOT_TARGET_HORIZON_DAYS", 90),
        drop_all_zero_rows=getattr(config, "DROP_ALL_ZERO_SNAPSHOT_ROWS", True),
        drop_extreme_sparse_rows=getattr(config, "DROP_EXTREME_SPARSE_SNAPSHOT_ROWS", True),
        min_nonzero_feature_count=getattr(config, "SPARSE_ROW_MIN_NONZERO_FEATURE_COUNT", 3),
        nonzero_epsilon=getattr(config, "SPARSE_ROW_NONZERO_EPSILON", 1e-12),
        numeric_only=getattr(config, "SPARSE_ROW_NUMERIC_ONLY", True),
        add_diagnostic_columns=getattr(config, "SNAPSHOT_FILTER_ADD_DIAGNOSTIC_COLUMNS", False),
        activity_feature_columns=getattr(config, "SNAPSHOT_ACTIVITY_FEATURE_COLUMNS", []),
        activity_exclude_columns=getattr(config, "SNAPSHOT_ACTIVITY_EXCLUDE_COLUMNS", []),
        sentinel_columns=list(getattr(config, "SENTINEL_COLUMNS_TO_CLEAN", {}).keys()),
    )
    return df, sentinel_report, snapshot_filter_report


def _prepare_split():
    df, sentinel_report, snapshot_filter_report = _load_and_prepare_snapshot()
    split, effective_ratios = chronological_split(
        df=df,
        date_col=config.DATE_COL,
        train_ratio=config.TRAIN_RATIO,
        validation_ratio=config.VALIDATION_RATIO,
        test_ratio=config.TEST_RATIO,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )
    return df, split, effective_ratios, sentinel_report, snapshot_filter_report


def _load_selected_cv_params() -> dict:
    if not bool(getattr(config, "HYPERPARAMETER_TUNING_ENABLED", False)):
        return {}

    path = config.OUTPUT_DIR / "01_cross_validation" / "04_selected_hyperparameters_by_variant.json"
    if path.exists():
        return read_json(path)
    return {}


def _default_params_for_algorithm(algorithm: str) -> dict:
    algorithm = str(algorithm).lower()
    if algorithm == "xgboost":
        return dict(config.XGB_DEFAULT_PARAMS)
    if algorithm == "lightgbm":
        return dict(getattr(config, "LIGHTGBM_DEFAULT_PARAMS", {}))
    if algorithm == "random_forest":
        return dict(getattr(config, "RANDOM_FOREST_DEFAULT_PARAMS", {}))
    raise ValueError(f"Unsupported algorithm configured: {algorithm}")


def _effective_algorithms() -> list[str]:
    if bool(getattr(config, "HYPERPARAMETER_TUNING_ENABLED", False)):
        return [str(getattr(config, "HYPERPARAMETER_TUNING_ALGORITHM", "xgboost")).lower()]
    return [str(a).lower() for a in getattr(config, "MODEL_ALGORITHMS_TO_RUN", ["xgboost"])]


def _params_for_algorithm_variant(algorithm: str, variant: str, selected_cv: dict) -> dict:
    if not selected_cv:
        return {}
    return selected_cv.get(algorithm, {}).get(variant, {}).get("param_override") or {}


def run() -> None:
    step_dir = config.OUTPUT_DIR / "02_final_model_validation_test"
    ensure_dir(step_dir)
    models_dir = step_dir / "model_artifacts"
    ensure_dir(models_dir)

    _, split, effective_ratios, sentinel_report, snapshot_filter_report = _prepare_split()
    if not sentinel_report.empty:
        sentinel_report.to_csv(step_dir / "00b_sentinel_cleaning_report.csv", index=False)
    if not snapshot_filter_report.empty:
        snapshot_filter_report.to_csv(step_dir / "00c_snapshot_training_filter_report.csv", index=False)

    selected_cv = _load_selected_cv_params()
    algorithms = _effective_algorithms()

    print("Final train/validation/test run plan")
    print(f"  algorithms: {algorithms}")
    print(f"  model variants: {config.MODEL_VARIANTS_TO_RUN}")
    print(f"  hyperparameter tuning enabled: {bool(getattr(config, 'HYPERPARAMETER_TUNING_ENABLED', False))}")

    y_train = target_series(split.train, config.TARGET_COL)
    y_validation = target_series(split.validation, config.TARGET_COL)
    y_test = target_series(split.test, config.TARGET_COL)

    validation_metric_rows = []
    threshold_free_rows = []
    trained_objects = {}

    thresholds = threshold_grid(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_GRID_SIZE)
    total_runs = len(algorithms) * len(config.MODEL_VARIANTS_TO_RUN)
    current_run = 0

    for algorithm in algorithms:
        default_params = _default_params_for_algorithm(algorithm)
        for variant in config.MODEL_VARIANTS_TO_RUN:
            current_run += 1
            print(f"[Final training {current_run}/{total_runs}] algorithm={algorithm} variant={variant}")

            configured_selected_prepared = list(config.FEATURE_SETS[variant])
            source_features = source_features_for_prepared_features(
                configured_selected_prepared, config.PREPARED_TO_SOURCE_FEATURE
            )
            present_sources, missing_sources = validate_source_features(
                split.train,
                source_features,
                error_on_missing=config.ERROR_ON_MISSING_SOURCE_FEATURES,
            )
            selected_prepared, skipped_prepared, skipped_sources = (
                prepared_features_for_available_sources(
                    prepared_features=configured_selected_prepared,
                    prepared_to_source=config.PREPARED_TO_SOURCE_FEATURE,
                    available_source_features=present_sources,
                )
            )
            if missing_sources:
                print(
                    f"[WARN] algorithm={algorithm} variant={variant}: "
                    f"skipping missing source columns={missing_sources}; "
                    f"prepared features removed={skipped_prepared}"
                )

            prepared = fit_transform_prepared_features(
                train_df=split.train,
                validation_df=split.validation,
                test_df=split.test,
                source_features=present_sources,
                selected_prepared_features=selected_prepared,
                numeric_impute_strategy=config.NUMERIC_IMPUTE_STRATEGY,
                categorical_impute_strategy=config.CATEGORICAL_IMPUTE_STRATEGY,
                one_hot_encode_categorical=config.ONE_HOT_ENCODE_CATEGORICAL,
                add_missing_prepared_features_as_zero=config.ADD_MISSING_PREPARED_FEATURES_AS_ZERO,
            )

            override = _params_for_algorithm_variant(algorithm, variant, selected_cv)
            params = make_algorithm_params(
                algorithm=algorithm,
                default_params=default_params,
                override_params=override,
                y=y_train,
                use_scale_pos_weight=config.USE_SCALE_POS_WEIGHT if algorithm == "xgboost" else False,
            )
            model = train_classifier(algorithm, prepared.X_train, y_train, params)
            val_prob = predict_probability(model, prepared.X_validation)
            test_prob = predict_probability(model, prepared.X_test)

            val_free = threshold_free_metrics(y_validation, val_prob)
            threshold_free_rows.append({"algorithm": algorithm, "model_variant": variant, "split": "validation", **val_free})

            val_search = threshold_search(
                y_true=y_validation,
                probability=val_prob,
                thresholds=thresholds,
                beta=config.THRESHOLD_BETA,
                max_flagged_rate=config.MAX_FLAGGED_RATE,
            )
            val_search.insert(0, "model_variant", variant)
            val_search.insert(0, "algorithm", algorithm)
            val_search.to_csv(step_dir / f"01_threshold_search_validation_{algorithm}_model_{variant}.csv", index=False)
            selected_threshold = select_best_threshold(val_search, beta=config.THRESHOLD_BETA)

            val_metrics = {"algorithm": algorithm, "model_variant": variant, "split": "validation", **selected_threshold}
            val_metrics["configured_prepared_feature_count"] = len(configured_selected_prepared)
            val_metrics["effective_configured_prepared_feature_count"] = len(selected_prepared)
            val_metrics["selected_prepared_feature_count"] = len(prepared.selected_prepared_features)
            val_metrics["required_source_feature_count"] = len(source_features)
            val_metrics["available_source_feature_count"] = len(present_sources)
            val_metrics["missing_source_feature_count"] = len(missing_sources)
            val_metrics["missing_source_features"] = ";".join(missing_sources)
            val_metrics["skipped_prepared_feature_count"] = len(skipped_prepared)
            val_metrics["skipped_prepared_features"] = ";".join(skipped_prepared)
            val_metrics["missing_selected_prepared_feature_count"] = len(prepared.missing_selected_prepared_features)
            val_metrics["hyperparameter_tuning_enabled"] = bool(getattr(config, "HYPERPARAMETER_TUNING_ENABLED", False))
            val_metrics["cv_selected_param_set_id"] = selected_cv.get(algorithm, {}).get(variant, {}).get("param_set_id", "default_or_cv_not_run")
            validation_metric_rows.append(val_metrics)

            val_topk = top_k_metrics(y_validation, val_prob, config.TOP_K_RATES)
            val_topk.insert(0, "model_variant", variant)
            val_topk.insert(0, "algorithm", algorithm)
            val_topk.to_csv(step_dir / f"02_top_k_validation_{algorithm}_model_{variant}.csv", index=False)

            importance = model_feature_importance(model, prepared.selected_prepared_features, algorithm)
            importance.to_csv(step_dir / f"03_final_feature_importance_{algorithm}_model_{variant}.csv", index=False)

            val_pred = None
            if config.SAVE_VALIDATION_AND_TEST_PREDICTIONS or config.ENABLE_MACHINE_LEVEL_TOP_K:
                val_pred = prediction_frame(
                    split.validation,
                    config.DATE_COL,
                    config.ID_COLS,
                    y_validation,
                    val_prob,
                    threshold=selected_threshold["threshold"],
                )
                val_pred.insert(0, "model_variant", variant)
                val_pred.insert(0, "algorithm", algorithm)

            if config.ENABLE_MACHINE_LEVEL_TOP_K and val_pred is not None:
                val_machine_topk = machine_level_top_k_metrics(
                    prediction_df=val_pred,
                    probability_col="probability",
                    target_col="y_true",
                    machine_id_col=config.MACHINE_ID_COL,
                    top_k_rates=config.MACHINE_TOP_K_RATES,
                    date_col=config.DATE_COL,
                    probability_aggregation=config.MACHINE_PROBABILITY_AGGREGATION,
                    target_aggregation=config.MACHINE_TARGET_AGGREGATION,
                )
                val_machine_topk.insert(0, "model_variant", variant)
                val_machine_topk.insert(0, "algorithm", algorithm)
                val_machine_topk.to_csv(
                    step_dir / f"02b_machine_top_k_validation_{algorithm}_model_{variant}.csv", index=False
                )

            if config.SAVE_VALIDATION_AND_TEST_PREDICTIONS and val_pred is not None:
                val_pred.to_csv(step_dir / f"04_validation_predictions_{algorithm}_model_{variant}.csv", index=False)

            key = f"{algorithm}__{variant}"
            trained_objects[key] = {
                "algorithm": algorithm,
                "variant": variant,
                "model": model,
                "prepared": prepared,
                "params": params,
                "param_override": override,
                "validation_probability": val_prob,
                "test_probability": test_prob,
                "selected_threshold": selected_threshold,
                "validation_metrics": val_metrics,
                "configured_selected_prepared": configured_selected_prepared,
                "effective_selected_prepared": selected_prepared,
                "missing_source_features": missing_sources,
                "skipped_prepared_features": skipped_prepared,
            }

            print(
                f"Validation algorithm={algorithm} variant={variant}: "
                f"AP={val_free.get('average_precision', np.nan):.4f}, "
                f"selected_threshold={selected_threshold['threshold']:.4f}, "
                f"F2={selected_threshold.get('f2', np.nan):.4f}, "
                f"recall={selected_threshold.get('recall', np.nan):.4f}, "
                f"precision={selected_threshold.get('precision', np.nan):.4f}, "
                f"flagged={selected_threshold.get('flagged_rate', np.nan):.2%}"
            )

    validation_metrics_df = pd.DataFrame(validation_metric_rows).sort_values(
        ["f2", "recall", "precision", "flagged_rate"], ascending=[False, False, False, True]
    )
    validation_metrics_df.to_csv(step_dir / "05_validation_selected_threshold_metrics_by_model.csv", index=False)

    if config.AUTO_SELECT_FINAL_VARIANT_BY_VALIDATION_F2:
        selected_row = validation_metrics_df.iloc[0]
        final_algorithm = str(selected_row["algorithm"])
        final_variant = str(selected_row["model_variant"])
        final_selection_reason = "selected_by_validation_f2_under_flagged_rate_constraint"
    else:
        final_algorithm = str(getattr(config, "FINAL_MODEL_ALGORITHM", "xgboost"))
        final_variant = config.FINAL_MODEL_VARIANT
        final_selection_reason = "selected_from_config_FINAL_MODEL_ALGORITHM_and_FINAL_MODEL_VARIANT"

    final_key = f"{final_algorithm}__{final_variant}"
    if final_key not in trained_objects:
        raise ValueError(
            f"Final model '{final_key}' was not trained. Check MODEL_ALGORITHMS_TO_RUN and MODEL_VARIANTS_TO_RUN in config.py."
        )

    if config.EVALUATE_TEST_FOR_ALL_VARIANTS:
        keys_to_test = list(trained_objects.keys())
    else:
        keys_to_test = [final_key]

    test_metric_rows = []
    for key in keys_to_test:
        obj = trained_objects[key]
        algorithm = obj["algorithm"]
        variant = obj["variant"]
        threshold = float(obj["selected_threshold"]["threshold"])
        test_free = threshold_free_metrics(y_test, obj["test_probability"])
        threshold_free_rows.append({"algorithm": algorithm, "model_variant": variant, "split": "test", **test_free})

        test_metrics = metrics_at_threshold(y_test, obj["test_probability"], threshold, beta=config.THRESHOLD_BETA)
        test_metrics = {"algorithm": algorithm, "model_variant": variant, "split": "test", **test_metrics}
        test_metric_rows.append(test_metrics)

        test_topk = top_k_metrics(y_test, obj["test_probability"], config.TOP_K_RATES)
        test_topk.insert(0, "model_variant", variant)
        test_topk.insert(0, "algorithm", algorithm)
        test_topk.to_csv(step_dir / f"07_top_k_test_{algorithm}_model_{variant}.csv", index=False)

        test_pred = None
        if config.SAVE_VALIDATION_AND_TEST_PREDICTIONS or config.ENABLE_MACHINE_LEVEL_TOP_K:
            test_pred = prediction_frame(
                split.test,
                config.DATE_COL,
                config.ID_COLS,
                y_test,
                obj["test_probability"],
                threshold=threshold,
            )
            test_pred.insert(0, "model_variant", variant)
            test_pred.insert(0, "algorithm", algorithm)

        if config.ENABLE_MACHINE_LEVEL_TOP_K and test_pred is not None:
            test_machine_topk = machine_level_top_k_metrics(
                prediction_df=test_pred,
                probability_col="probability",
                target_col="y_true",
                machine_id_col=config.MACHINE_ID_COL,
                top_k_rates=config.MACHINE_TOP_K_RATES,
                date_col=config.DATE_COL,
                probability_aggregation=config.MACHINE_PROBABILITY_AGGREGATION,
                target_aggregation=config.MACHINE_TARGET_AGGREGATION,
            )
            test_machine_topk.insert(0, "model_variant", variant)
            test_machine_topk.insert(0, "algorithm", algorithm)
            test_machine_topk.to_csv(step_dir / f"07b_machine_top_k_test_{algorithm}_model_{variant}.csv", index=False)

        if config.SAVE_VALIDATION_AND_TEST_PREDICTIONS and test_pred is not None:
            test_pred.to_csv(step_dir / f"08_test_predictions_{algorithm}_model_{variant}.csv", index=False)

    pd.DataFrame(threshold_free_rows).to_csv(step_dir / "06_threshold_free_metrics_by_model.csv", index=False)
    pd.DataFrame(test_metric_rows).to_csv(step_dir / "09_test_metrics_at_validation_selected_threshold.csv", index=False)

    final_obj = trained_objects[final_key]
    metadata = {
        "algorithm": final_algorithm,
        "final_model_variant": final_variant,
        "final_selection_reason": final_selection_reason,
        "selected_threshold_from_validation": final_obj["selected_threshold"],
        "model_params_used": final_obj["params"],
        "param_override_from_cv": final_obj["param_override"],
        "configured_prepared_feature_count": len(final_obj["configured_selected_prepared"]),
        "effective_configured_prepared_feature_count": len(final_obj["effective_selected_prepared"]),
        "selected_prepared_feature_count": len(final_obj["prepared"].selected_prepared_features),
        "selected_prepared_features": final_obj["prepared"].selected_prepared_features,
        "missing_source_features_skipped": final_obj["missing_source_features"],
        "prepared_features_skipped_for_missing_sources": final_obj["skipped_prepared_features"],
        "numeric_input_cols": final_obj["prepared"].numeric_input_cols,
        "categorical_input_cols": final_obj["prepared"].categorical_input_cols,
        "missing_selected_prepared_features_added_as_zero": final_obj["prepared"].missing_selected_prepared_features,
        "sentinel_cleaning_enabled": bool(getattr(config, "SENTINEL_CLEANING_ENABLED", False)),
        "snapshot_training_filters_enabled": bool(getattr(config, "SNAPSHOT_TRAINING_FILTERS_ENABLED", False)),
        "snapshot_cutoff_reference_end_date": getattr(config, "SNAPSHOT_CUTOFF_REFERENCE_END_DATE", None),
        "snapshot_target_horizon_days": getattr(config, "SNAPSHOT_TARGET_HORIZON_DAYS", 90),
        "sparse_row_min_nonzero_feature_count": getattr(config, "SPARSE_ROW_MIN_NONZERO_FEATURE_COUNT", 3),
    }
    write_json(metadata, step_dir / "10_final_model_selection.json")

    if config.SAVE_FINAL_MODEL_ARTIFACTS:
        save_model_artifacts(
            final_obj["model"],
            final_obj["prepared"],
            models_dir / f"model_{final_algorithm}_{final_variant}",
            metadata=metadata,
        )

    write_json(
        {
            "step": "02_train_validate_test",
            "output_dir": str(step_dir),
            "effective_outer_split_ratios": effective_ratios,
            "algorithms_run": algorithms,
            "model_variants_run": config.MODEL_VARIANTS_TO_RUN,
            "hyperparameter_tuning_enabled": bool(getattr(config, "HYPERPARAMETER_TUNING_ENABLED", False)),
            "auto_select_final_variant_by_validation_f2": config.AUTO_SELECT_FINAL_VARIANT_BY_VALIDATION_F2,
            "final_algorithm": final_algorithm,
            "final_model_variant": final_variant,
            "final_selection_reason": final_selection_reason,
            "test_evaluated_model_keys": keys_to_test,
            "max_flagged_rate": config.MAX_FLAGGED_RATE,
            "threshold_beta": config.THRESHOLD_BETA,
            "sentinel_cleaning_enabled": bool(getattr(config, "SENTINEL_CLEANING_ENABLED", False)),
            "snapshot_training_filters_enabled": bool(getattr(config, "SNAPSHOT_TRAINING_FILTERS_ENABLED", False)),
            "snapshot_training_filter_report_file": "00c_snapshot_training_filter_report.csv" if not snapshot_filter_report.empty else None,
            "snapshot_rows_after_training_filters": int(len(split.train) + len(split.validation) + len(split.test)),
            "snapshot_cutoff_reference_end_date": getattr(config, "SNAPSHOT_CUTOFF_REFERENCE_END_DATE", None),
            "snapshot_target_horizon_days": getattr(config, "SNAPSHOT_TARGET_HORIZON_DAYS", 90),
            "sparse_row_min_nonzero_feature_count": getattr(config, "SPARSE_ROW_MIN_NONZERO_FEATURE_COUNT", 3),
            "machine_level_top_k_enabled": config.ENABLE_MACHINE_LEVEL_TOP_K,
            "machine_id_col": config.MACHINE_ID_COL if config.ENABLE_MACHINE_LEVEL_TOP_K else None,
            "notes": [
                "Final models are trained on full training_main only.",
                "When hyperparameter tuning is disabled, XGBoost/LightGBM/Random Forest use configured defaults.",
                "When hyperparameter tuning is enabled, selected CV XGBoost parameters are used when available.",
                "Probability threshold is selected on validation_holdout using F2 with max flagged-rate constraint.",
                "Test metrics are evaluated after threshold selection.",
                "9999 sentinel values are cleaned before splitting when SENTINEL_CLEANING_ENABLED=True.",
                "Prepared features mapped to completely missing source columns are skipped and reported instead of added as permanent zero columns.",
                "Snapshot rows after the configured full target-window cutoff are removed before splitting.",
                "All-zero and extreme-sparse numeric feature rows are removed before splitting when snapshot filters are enabled.",
            ],
        },
        step_dir / "00_run_summary.json",
    )
    print(f"02_train_validate_test completed. Outputs: {step_dir}")
    print(f"Final test-evaluated model: algorithm={final_algorithm}, variant={final_variant}")


if __name__ == "__main__":
    run()
