"""Step 03: grouped cross validation for window-based case-control datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import config
from cc_utils import (
    ensure_dir,
    make_model_pipeline,
    fit_model_pipeline,
    metrics_at_threshold,
    predict_score,
    prediction_frame,
    get_evaluation_target,
    future_claim_lead_time_summary,
    threshold_free_metrics,
    top_k_metrics,
    validate_dataset_features,
    write_json,
)


def _load_dataset_index() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}. Run 02_build_case_control_dataset.py first.")
    return pd.read_csv(path)


def _cv_one_dataset(dataset_row: pd.Series, output_dir) -> dict:
    dataset_id = dataset_row["dataset_id"]
    dataset_path = dataset_row["training_dataset_path"]
    df = pd.read_csv(dataset_path, parse_dates=["window_start", "window_end", "future_claim_date"])
    validate_dataset_features(df, config)

    X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y = df["target"].astype(int).reset_index(drop=True)
    y_eval_all, eval_target_col, eval_target_mode, eval_horizon_days = get_evaluation_target(df, config)
    y_eval_all = y_eval_all.reset_index(drop=True)
    groups = df["case_control_group_id"].astype(str).reset_index(drop=True)
    n_groups = groups.nunique()
    n_splits = min(int(config.CV_N_SPLITS), int(n_groups))
    if n_splits < 2:
        raise ValueError(f"Need at least 2 groups for CV. Dataset {dataset_id} has {n_groups} groups.")

    cv = GroupKFold(n_splits=n_splits)
    metric_rows = []
    topk_rows = []
    prediction_rows = []
    fold_rows = []

    for algorithm in config.MODELS_TO_RUN:
        for fold_id, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
            print(f"  CV dataset={dataset_id} algorithm={algorithm} fold={fold_id}/{n_splits}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            y_val_eval = y_eval_all.iloc[val_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].copy()
            fold_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "fold_id": fold_id,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_positive_rows": int(y_train.sum()),
                "validation_positive_rows": int(y_val.sum()),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "validation_groups": int(groups.iloc[val_idx].nunique()),
            })

            model = make_model_pipeline(algorithm, config)
            if model is None:
                metric_rows.append({
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "fold_id": fold_id,
                    "status": "skipped_missing_dependency",
                })
                continue
            try:
                fit_metadata = fit_model_pipeline(model, algorithm, X_train, y_train, config)
                score = predict_score(model, X_val, algorithm)
                free = threshold_free_metrics(y_val_eval, score)
                thresh = metrics_at_threshold(y_val_eval, score, threshold=0.5)
                lead = future_claim_lead_time_summary(val_df, y_val_eval)
                row = {
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "fold_id": fold_id,
                    "status": "used",
                    "training_target_col": "target",
                    "evaluation_target_col": eval_target_col,
                    "evaluation_target_mode": eval_target_mode,
                    "evaluation_horizon_days": eval_horizon_days,
                }
                row.update({f"fit_{k}": v for k, v in fit_metadata.items() if k != "algorithm"})
                row.update({f"threshold_free_{k}": v for k, v in free.items()})
                row.update({f"threshold_0p5_{k}": v for k, v in thresh.items()})
                row.update({f"lead_time_{k}": v for k, v in lead.items()})
                metric_rows.append(row)

                tk = top_k_metrics(y_val_eval, score, config.CV_TOP_K_RATES)
                tk.insert(0, "evaluation_target_col", eval_target_col)
                tk.insert(0, "evaluation_target_mode", eval_target_mode)
                tk.insert(0, "evaluation_horizon_days", eval_horizon_days)
                tk.insert(0, "fold_id", fold_id)
                tk.insert(0, "algorithm", algorithm)
                tk.insert(0, "dataset_id", dataset_id)
                topk_rows.append(tk)

                if config.SAVE_CV_PREDICTIONS:
                    pred = prediction_frame(val_df, score)
                    pred.insert(0, "evaluation_target", y_val_eval.to_numpy())
                    pred.insert(0, "evaluation_target_col", eval_target_col)
                    pred.insert(0, "fold_id", fold_id)
                    pred.insert(0, "algorithm", algorithm)
                    pred.insert(0, "dataset_id", dataset_id)
                    prediction_rows.append(pred)
            except Exception as exc:
                print(f"    failed {algorithm} fold={fold_id}: {exc}")
                metric_rows.append({
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "fold_id": fold_id,
                    "status": "failed",
                    "error": str(exc),
                })

    pd.DataFrame(fold_rows).to_csv(output_dir / f"{dataset_id}__cv_fold_summary.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / f"{dataset_id}__cv_metrics_by_fold.csv", index=False)
    if topk_rows:
        pd.concat(topk_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__cv_top_k_by_fold.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__cv_predictions.csv", index=False)

    used = metrics_df[metrics_df["status"].eq("used")].copy()
    summary = pd.DataFrame()
    if not used.empty:
        for _optional_col in ["fit_xgboost_scale_pos_weight", "fit_xgboost_class_importance_mode"]:
            if _optional_col not in used.columns:
                used[_optional_col] = np.nan
        agg_cols = {
            "fold_id": "count",
            "threshold_free_average_precision": "mean",
            "threshold_free_roc_auc": "mean",
            "threshold_0p5_precision": "mean",
            "threshold_0p5_recall": "mean",
            "threshold_0p5_f1": "mean",
            "threshold_0p5_flagged_rate": "mean",
        }
        summary = (
            used.groupby(["dataset_id", "algorithm"], dropna=False)
            .agg(**{
                "fold_count": ("fold_id", "count"),
                "mean_average_precision": ("threshold_free_average_precision", "mean"),
                "std_average_precision": ("threshold_free_average_precision", "std"),
                "mean_roc_auc": ("threshold_free_roc_auc", "mean"),
                "mean_threshold_0p5_precision": ("threshold_0p5_precision", "mean"),
                "mean_threshold_0p5_recall": ("threshold_0p5_recall", "mean"),
                "mean_threshold_0p5_f1": ("threshold_0p5_f1", "mean"),
                "mean_threshold_0p5_flagged_rate": ("threshold_0p5_flagged_rate", "mean"),
                "mean_fit_xgboost_scale_pos_weight": ("fit_xgboost_scale_pos_weight", "mean"),
                "fit_xgboost_class_importance_mode": ("fit_xgboost_class_importance_mode", "first"),
            })
            .reset_index()
            .sort_values(["dataset_id", "mean_average_precision"], ascending=[True, False])
        )
        summary.to_csv(output_dir / f"{dataset_id}__cv_summary_by_model.csv", index=False)

    return {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "groups": int(n_groups),
        "n_splits_used": int(n_splits),
    }


def run() -> None:
    step_dir = config.OUTPUT_DIR / "03_cross_validation"
    ensure_dir(step_dir)
    dataset_index = _load_dataset_index()

    summaries = []
    for _, dataset_row in dataset_index.iterrows():
        summaries.append(_cv_one_dataset(dataset_row, step_dir))

    all_metrics = []
    all_summary = []
    all_topk = []
    for dataset_id in dataset_index["dataset_id"]:
        p = step_dir / f"{dataset_id}__cv_metrics_by_fold.csv"
        if p.exists():
            all_metrics.append(pd.read_csv(p))
        s = step_dir / f"{dataset_id}__cv_summary_by_model.csv"
        if s.exists():
            all_summary.append(pd.read_csv(s))
        t = step_dir / f"{dataset_id}__cv_top_k_by_fold.csv"
        if t.exists():
            all_topk.append(pd.read_csv(t))
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(step_dir / "cv_metrics_by_fold_all_datasets.csv", index=False)
    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(step_dir / "cv_summary_by_model_all_datasets.csv", index=False)
    if all_topk:
        topk = pd.concat(all_topk, ignore_index=True)
        topk.to_csv(step_dir / "cv_top_k_by_fold_all_datasets.csv", index=False)
        topk_summary = (
            topk.groupby(["dataset_id", "algorithm", "top_k_rate"], dropna=False)
            .agg(
                fold_count=("fold_id", "count"),
                mean_precision_at_k=("precision_at_k", "mean"),
                std_precision_at_k=("precision_at_k", "std"),
                mean_recall_at_k=("recall_at_k", "mean"),
                std_recall_at_k=("recall_at_k", "std"),
                mean_lift_vs_random=("lift_vs_random", "mean"),
                std_lift_vs_random=("lift_vs_random", "std"),
            )
            .reset_index()
            .sort_values(["dataset_id", "algorithm", "top_k_rate"])
        )
        topk_summary.to_csv(step_dir / "cv_top_k_summary_all_datasets.csv", index=False)

    write_json(
        {
            "step": "03_cross_validation",
            "output_dir": str(step_dir),
            "summaries": summaries,
            "models_to_run": config.MODELS_TO_RUN,
            "cv_n_splits_requested": config.CV_N_SPLITS,
            "group_column": "case_control_group_id",
            "training_split_used": "train",
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "notes": [
                "GroupKFold is used only within the chronological train split.",
                "Each positive case and its matched controls stay in the same fold.",
                "Validation and test splits remain untouched by CV for holdout evaluation and final reporting.",
            ],
        },
        step_dir / "run_summary.json",
    )
    print(f"03_cross_validation completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
