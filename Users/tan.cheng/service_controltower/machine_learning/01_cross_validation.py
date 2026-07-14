"""Step 01: expanding-window date-based CV inside training_main only."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from ml_utils import (
    build_expanding_window_folds,
    chronological_split,
    ensure_dir,
    fit_transform_prepared_features,
    make_param_grid,
    make_xgb_params,
    machine_level_top_k_metrics,
    metrics_at_threshold,
    param_set_id,
    prediction_frame,
    read_snapshot,
    select_best_threshold,
    source_features_for_prepared_features,
    target_series,
    threshold_free_metrics,
    threshold_grid,
    threshold_search,
    top_k_metrics,
    train_xgboost_classifier,
    validate_source_features,
    write_json,
)


def _prepare_split():
    df = read_snapshot(config.INPUT_DATA_PATH, config.DATE_COL)
    source_cols_to_drop = [
        c for c in getattr(config, "SOURCE_COLUMNS_TO_DROP_BEFORE_MODELING", []) if c in df.columns
    ]
    if source_cols_to_drop:
        df = df.drop(columns=source_cols_to_drop)
    split, effective_ratios = chronological_split(
        df=df,
        date_col=config.DATE_COL,
        train_ratio=config.TRAIN_RATIO,
        validation_ratio=config.VALIDATION_RATIO,
        test_ratio=config.TEST_RATIO,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )
    return df, split, effective_ratios


def run() -> None:
    step_dir = config.OUTPUT_DIR / "01_cross_validation"
    ensure_dir(step_dir)

    _, split, effective_ratios = _prepare_split()
    training_df = split.train.sort_values(config.DATE_COL, kind="mergesort").reset_index(drop=True)

    folds, fold_audit = build_expanding_window_folds(
        training_df=training_df,
        date_col=config.DATE_COL,
        target_col=config.TARGET_COL,
        n_splits=config.CV_N_SPLITS,
        validation_window_days=config.CV_VALIDATION_WINDOW_DAYS,
        gap_days=config.CV_GAP_DAYS,
        min_train_rows=config.CV_MIN_TRAIN_ROWS,
        min_validation_rows=config.CV_MIN_VALIDATION_ROWS,
        min_positives_in_train=config.CV_MIN_POSITIVES_IN_TRAIN,
        min_positives_in_validation=config.CV_MIN_POSITIVES_IN_VALIDATION,
    )
    fold_audit.to_csv(step_dir / "01_cv_fold_definitions.csv", index=False)
    if not folds:
        raise ValueError(
            "No usable CV folds were generated. Review CV_N_SPLITS, CV_VALIDATION_WINDOW_DAYS, "
            "CV_GAP_DAYS, and minimum row/positive thresholds in config.py."
        )

    param_overrides = make_param_grid(config.XGB_DEFAULT_PARAMS, config.HYPERPARAMETER_GRID)
    all_metric_rows = []
    all_topk_rows = []
    all_machine_topk_rows = []
    all_prediction_frames = []

    for variant in config.MODEL_VARIANTS_TO_RUN:
        selected_prepared = config.FEATURE_SETS[variant]
        source_features = source_features_for_prepared_features(
            selected_prepared, config.PREPARED_TO_SOURCE_FEATURE
        )
        present_sources, missing_sources = validate_source_features(
            training_df, source_features, error_on_missing=config.ERROR_ON_MISSING_SOURCE_FEATURES
        )
        if missing_sources:
            print(f"[WARN] Variant {variant}: missing source features ignored: {missing_sources}")

        for param_idx, override in enumerate(param_overrides, start=1):
            pid = param_set_id(override, param_idx)
            for fold in folds:
                fold_train = training_df.iloc[fold.train_index].copy()
                fold_val = training_df.iloc[fold.validation_index].copy()
                y_train = target_series(fold_train, config.TARGET_COL)
                y_val = target_series(fold_val, config.TARGET_COL)

                prepared = fit_transform_prepared_features(
                    train_df=fold_train,
                    validation_df=fold_val,
                    test_df=None,
                    source_features=present_sources,
                    selected_prepared_features=selected_prepared,
                    numeric_impute_strategy=config.NUMERIC_IMPUTE_STRATEGY,
                    categorical_impute_strategy=config.CATEGORICAL_IMPUTE_STRATEGY,
                    one_hot_encode_categorical=config.ONE_HOT_ENCODE_CATEGORICAL,
                    add_missing_prepared_features_as_zero=config.ADD_MISSING_PREPARED_FEATURES_AS_ZERO,
                )

                params = make_xgb_params(
                    config.XGB_DEFAULT_PARAMS,
                    override,
                    y_train,
                    use_scale_pos_weight=config.USE_SCALE_POS_WEIGHT,
                )
                model = train_xgboost_classifier(prepared.X_train, y_train, params)
                prob = model.predict_proba(prepared.X_validation)[:, 1]

                free = threshold_free_metrics(y_val, prob)
                default_metrics = metrics_at_threshold(
                    y_val, prob, threshold=config.CV_DEFAULT_THRESHOLD, beta=config.THRESHOLD_BETA
                )
                cv_search = threshold_search(
                    y_true=y_val,
                    probability=prob,
                    thresholds=threshold_grid(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_GRID_SIZE),
                    beta=config.THRESHOLD_BETA,
                    max_flagged_rate=config.CV_MAX_FLAGGED_RATE_FOR_BEST_F2,
                )
                best = select_best_threshold(cv_search, beta=config.THRESHOLD_BETA)

                row = {
                    "model_variant": variant,
                    "param_set_id": pid,
                    "param_override": override,
                    "fold_id": fold.fold_id,
                    "train_start_date": fold.train_start_date,
                    "train_end_date": fold.train_end_date,
                    "gap_start_date": fold.gap_start_date,
                    "gap_end_date": fold.gap_end_date,
                    "validation_start_date": fold.validation_start_date,
                    "validation_end_date": fold.validation_end_date,
                    "selected_prepared_feature_count": len(selected_prepared),
                    "required_source_feature_count": len(present_sources),
                    "missing_selected_prepared_feature_count": len(prepared.missing_selected_prepared_features),
                }
                row.update({f"threshold_free_{k}": v for k, v in free.items()})
                row.update({f"default_threshold_{k}": v for k, v in default_metrics.items()})
                row.update({f"best_cv_threshold_{k}": v for k, v in best.items()})
                all_metric_rows.append(row)

                topk_df = top_k_metrics(y_val, prob, config.TOP_K_RATES)
                topk_df.insert(0, "validation_end_date", fold.validation_end_date)
                topk_df.insert(0, "validation_start_date", fold.validation_start_date)
                topk_df.insert(0, "fold_id", fold.fold_id)
                topk_df.insert(0, "param_set_id", pid)
                topk_df.insert(0, "model_variant", variant)
                all_topk_rows.append(topk_df)

                pred_df = None
                if config.SAVE_CV_PREDICTIONS or config.ENABLE_MACHINE_LEVEL_TOP_K:
                    pred_df = prediction_frame(
                        fold_val,
                        config.DATE_COL,
                        config.ID_COLS,
                        y_val,
                        prob,
                        threshold=best["threshold"],
                    )
                    pred_df.insert(0, "fold_id", fold.fold_id)
                    pred_df.insert(0, "param_set_id", pid)
                    pred_df.insert(0, "model_variant", variant)

                if config.ENABLE_MACHINE_LEVEL_TOP_K and pred_df is not None:
                    machine_topk_df = machine_level_top_k_metrics(
                        prediction_df=pred_df,
                        probability_col="probability",
                        target_col="y_true",
                        machine_id_col=config.MACHINE_ID_COL,
                        top_k_rates=config.MACHINE_TOP_K_RATES,
                        date_col=config.DATE_COL,
                        probability_aggregation=config.MACHINE_PROBABILITY_AGGREGATION,
                        target_aggregation=config.MACHINE_TARGET_AGGREGATION,
                    )
                    machine_topk_df.insert(0, "validation_end_date", fold.validation_end_date)
                    machine_topk_df.insert(0, "validation_start_date", fold.validation_start_date)
                    machine_topk_df.insert(0, "fold_id", fold.fold_id)
                    machine_topk_df.insert(0, "param_set_id", pid)
                    machine_topk_df.insert(0, "model_variant", variant)
                    all_machine_topk_rows.append(machine_topk_df)

                if config.SAVE_CV_PREDICTIONS and pred_df is not None:
                    all_prediction_frames.append(pred_df)

                print(
                    f"CV variant={variant} param={pid} fold={fold.fold_id}: "
                    f"AP={free.get('average_precision', np.nan):.4f}, "
                    f"best_F2={best.get('f2', np.nan):.4f}, "
                    f"flagged={best.get('flagged_rate', np.nan):.2%}"
                )

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = step_dir / "02_cv_metrics_by_fold.csv"
    metrics_df.to_csv(metrics_path, index=False)

    group_cols = ["model_variant", "param_set_id"]
    summary = (
        metrics_df.groupby(group_cols, dropna=False)
        .agg(
            fold_count=("fold_id", "count"),
            mean_average_precision=("threshold_free_average_precision", "mean"),
            std_average_precision=("threshold_free_average_precision", "std"),
            mean_roc_auc=("threshold_free_roc_auc", "mean"),
            mean_default_f2=("default_threshold_f2", "mean"),
            mean_default_recall=("default_threshold_recall", "mean"),
            mean_default_precision=("default_threshold_precision", "mean"),
            mean_default_flagged_rate=("default_threshold_flagged_rate", "mean"),
            mean_best_cv_f2=("best_cv_threshold_f2", "mean"),
            mean_best_cv_recall=("best_cv_threshold_recall", "mean"),
            mean_best_cv_precision=("best_cv_threshold_precision", "mean"),
            mean_best_cv_flagged_rate=("best_cv_threshold_flagged_rate", "mean"),
        )
        .reset_index()
        .sort_values(["model_variant", "mean_average_precision"], ascending=[True, False])
    )
    summary["selected_by_cv_mean_average_precision"] = False
    for variant in summary["model_variant"].unique():
        idx = summary[summary["model_variant"] == variant]["mean_average_precision"].idxmax()
        summary.loc[idx, "selected_by_cv_mean_average_precision"] = True
    summary.to_csv(step_dir / "03_cv_param_summary.csv", index=False)

    if all_topk_rows:
        topk_metrics_df = pd.concat(all_topk_rows, ignore_index=True)
        topk_metrics_df.to_csv(step_dir / "06_cv_top_k_metrics_by_fold.csv", index=False)

        topk_summary = (
            topk_metrics_df.groupby(["model_variant", "param_set_id", "top_k_rate"], dropna=False)
            .agg(
                fold_count=("fold_id", "count"),
                mean_precision_at_k=("precision_at_k", "mean"),
                std_precision_at_k=("precision_at_k", "std"),
                mean_recall_at_k=("recall_at_k", "mean"),
                std_recall_at_k=("recall_at_k", "std"),
                mean_lift_vs_random=("lift_vs_random", "mean"),
                std_lift_vs_random=("lift_vs_random", "std"),
                mean_flagged_rate_actual=("flagged_rate_actual", "mean"),
                mean_flagged_count=("flagged_count", "mean"),
                mean_min_probability_in_top_k=("min_probability_in_top_k", "mean"),
            )
            .reset_index()
            .sort_values(["model_variant", "param_set_id", "top_k_rate"])
        )
        topk_summary.to_csv(step_dir / "07_cv_top_k_summary.csv", index=False)

    if config.ENABLE_MACHINE_LEVEL_TOP_K and all_machine_topk_rows:
        machine_topk_metrics_df = pd.concat(all_machine_topk_rows, ignore_index=True)
        machine_topk_metrics_df.to_csv(step_dir / "08_cv_machine_top_k_metrics_by_fold.csv", index=False)

        machine_topk_summary = (
            machine_topk_metrics_df.groupby(["model_variant", "param_set_id", "top_k_rate"], dropna=False)
            .agg(
                fold_count=("fold_id", "count"),
                mean_machine_count=("machine_count", "mean"),
                mean_positive_machine_count=("positive_machine_count", "mean"),
                mean_machine_positive_rate=("machine_positive_rate", "mean"),
                mean_flagged_machine_count=("flagged_machine_count", "mean"),
                mean_flagged_machine_rate_actual=("flagged_machine_rate_actual", "mean"),
                mean_true_positive_machines=("true_positive_machines", "mean"),
                mean_false_positive_machines=("false_positive_machines", "mean"),
                mean_machine_precision_at_k=("machine_precision_at_k", "mean"),
                std_machine_precision_at_k=("machine_precision_at_k", "std"),
                mean_machine_recall_at_k=("machine_recall_at_k", "mean"),
                std_machine_recall_at_k=("machine_recall_at_k", "std"),
                mean_machine_lift_vs_random=("machine_lift_vs_random", "mean"),
                std_machine_lift_vs_random=("machine_lift_vs_random", "std"),
                mean_min_probability_in_top_k=("min_probability_in_top_k", "mean"),
            )
            .reset_index()
            .sort_values(["model_variant", "param_set_id", "top_k_rate"])
        )
        machine_topk_summary.to_csv(step_dir / "09_cv_machine_top_k_summary.csv", index=False)

    selected = {}
    for _, row in summary[summary["selected_by_cv_mean_average_precision"]].iterrows():
        variant = row["model_variant"]
        pid = row["param_set_id"]
        match = metrics_df[(metrics_df["model_variant"] == variant) & (metrics_df["param_set_id"] == pid)].iloc[0]
        override = match["param_override"]
        selected[variant] = {
            "param_set_id": pid,
            "param_override": override,
            "mean_average_precision": float(row["mean_average_precision"]),
            "fold_count": int(row["fold_count"]),
        }
    write_json(selected, step_dir / "04_selected_hyperparameters_by_variant.json")

    if config.SAVE_CV_PREDICTIONS and all_prediction_frames:
        pd.concat(all_prediction_frames, ignore_index=True).to_csv(
            step_dir / "05_cv_predictions.csv", index=False
        )

    write_json(
        {
            "step": "01_cross_validation",
            "output_dir": str(step_dir),
            "effective_outer_split_ratios": effective_ratios,
            "model_variants_to_run": config.MODEL_VARIANTS_TO_RUN,
            "cv_n_splits_requested": config.CV_N_SPLITS,
            "cv_n_splits_used": len(folds),
            "cv_gap_days": config.CV_GAP_DAYS,
            "cv_validation_window_days": config.CV_VALIDATION_WINDOW_DAYS,
            "main_hyperparameter_metric": "mean_average_precision",
            "machine_level_top_k_enabled": config.ENABLE_MACHINE_LEVEL_TOP_K,
            "machine_id_col": config.MACHINE_ID_COL if config.ENABLE_MACHINE_LEVEL_TOP_K else None,
            "notes": [
                "Expanding-window CV is performed inside training_main only.",
                "Each fold uses a 45-day configurable gap by default between training end and CV validation start.",
                "F2, recall, precision, flagged rate, and snapshot top-K precision/recall are reported per fold but mean average precision selects hyperparameters.",
                "If ENABLE_MACHINE_LEVEL_TOP_K=True, machine-level top-K precision/recall reports are also generated by aggregating repeated snapshots per machine.",
            ],
        },
        step_dir / "00_run_summary.json",
    )
    print(f"01_cross_validation completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
