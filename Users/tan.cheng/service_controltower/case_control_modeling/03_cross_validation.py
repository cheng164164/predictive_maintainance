"""Step 03: machine-grouped cross validation on the fixed training split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import config
from cc_utils import (
    ensure_dir,
    fit_model_pipeline,
    future_claim_lead_time_summary,
    get_evaluation_target,
    make_model_pipeline,
    metrics_at_threshold,
    predict_score,
    prediction_frame,
    threshold_free_metrics,
    top_k_metrics,
    validate_dataset_features,
    write_json,
)


def _load_dataset_index() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {path}. Run 02_build_case_control_dataset.py first."
        )
    return pd.read_csv(path)


def _build_machine_grouped_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Prefer shuffled stratified group folds and fall back to GroupKFold."""
    try:
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(config.RANDOM_STATE),
        )
        folds = list(cv.split(X, y, groups))
        return folds, "StratifiedGroupKFold(machine_key, shuffled, fixed seed)"
    except (ValueError, TypeError) as exc:
        print(f"  [WARN] StratifiedGroupKFold unavailable for this dataset: {exc}")
        cv = GroupKFold(n_splits=n_splits)
        folds = list(cv.split(X, y, groups))
        return folds, "GroupKFold(machine_key) fallback"


def _cv_one_dataset(dataset_row: pd.Series, output_dir) -> dict:
    dataset_id = str(dataset_row["dataset_id"])
    dataset_path = dataset_row["training_dataset_path"]
    df = pd.read_csv(
        dataset_path,
        parse_dates=["window_start", "window_end", "future_claim_date"],
        low_memory=False,
    )
    validate_dataset_features(df, config)

    feature_cols = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
    X = df[feature_cols].reset_index(drop=True)
    y = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
    y_eval_all, eval_target_col, eval_target_mode, eval_horizon_days = get_evaluation_target(
        df, config
    )
    y_eval_all = y_eval_all.reset_index(drop=True)
    groups = df["machine_key"].astype(str).reset_index(drop=True)
    n_groups = int(groups.nunique())
    n_splits = min(int(config.CV_N_SPLITS), n_groups)
    if n_splits < 2:
        raise ValueError(
            f"Need at least two distinct machines for CV. Dataset {dataset_id} has {n_groups}."
        )

    folds, cv_strategy = _build_machine_grouped_folds(X, y, groups, n_splits)
    fold_id_by_row = np.full(len(df), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(folds, start=1):
        fold_id_by_row[val_idx] = fold_id
    if (fold_id_by_row < 1).any():
        raise ValueError("Cross-validation fold assignment did not cover every training row.")

    machine_fold = df[["machine_key", "full_model"]].copy()
    machine_fold["target"] = y.to_numpy()
    machine_fold["cv_fold_id"] = fold_id_by_row
    machine_fold = (
        machine_fold.groupby(["machine_key", "full_model", "cv_fold_id"], dropna=False)
        .agg(
            rows=("target", "size"),
            has_positive_row=("target", "max"),
            negative_rows=("target", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    if machine_fold.groupby("machine_key")["cv_fold_id"].nunique().max() > 1:
        raise ValueError("A machine was assigned to more than one cross-validation fold.")
    machine_fold.to_csv(output_dir / f"{dataset_id}__cv_machine_fold_assignments.csv", index=False)

    metric_rows: list[dict] = []
    topk_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        fold_rows.append(
            {
                "dataset_id": dataset_id,
                "fold_id": fold_id,
                "cv_strategy": cv_strategy,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_positive_rows": int(y_train.sum()),
                "validation_positive_rows": int(y_val.sum()),
                "train_machines": int(groups.iloc[train_idx].nunique()),
                "validation_machines": int(groups.iloc[val_idx].nunique()),
                "train_case_control_groups": int(
                    df.iloc[train_idx]["case_control_group_id"].nunique(dropna=True)
                ),
                "validation_case_control_groups": int(
                    df.iloc[val_idx]["case_control_group_id"].nunique(dropna=True)
                ),
            }
        )

    for algorithm in config.MODELS_TO_RUN:
        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            print(
                f"  CV dataset={dataset_id} algorithm={algorithm} fold={fold_id}/{n_splits}"
            )
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            y_val_eval = y_eval_all.iloc[val_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].copy()

            model = make_model_pipeline(algorithm, config)
            if model is None:
                metric_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "algorithm": algorithm,
                        "fold_id": fold_id,
                        "status": "skipped_missing_dependency",
                    }
                )
                continue
            try:
                fit_metadata = fit_model_pipeline(
                    model, algorithm, X_train, y_train, config
                )
                score = predict_score(model, X_val, algorithm)
                free = threshold_free_metrics(y_val_eval, score)
                threshold = float(getattr(config, "VALIDATION_SCORE_THRESHOLD", 0.5))
                thresh = metrics_at_threshold(y_val_eval, score, threshold=threshold)
                lead = future_claim_lead_time_summary(val_df, y_val_eval)
                row = {
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "fold_id": fold_id,
                    "status": "used",
                    "cv_strategy": cv_strategy,
                    "training_target_col": "target",
                    "evaluation_target_col": eval_target_col,
                    "evaluation_target_mode": eval_target_mode,
                    "evaluation_horizon_days": eval_horizon_days,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "train_machines": int(groups.iloc[train_idx].nunique()),
                    "validation_machines": int(groups.iloc[val_idx].nunique()),
                }
                row.update(
                    {f"fit_{k}": v for k, v in fit_metadata.items() if k != "algorithm"}
                )
                row.update({f"threshold_free_{k}": v for k, v in free.items()})
                row.update(
                    {
                        f"threshold_{str(threshold).replace('.', 'p')}_{k}": v
                        for k, v in thresh.items()
                    }
                )
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
                    pred.insert(0, "cv_fold_id", fold_id)
                    pred.insert(0, "algorithm", algorithm)
                    pred.insert(0, "dataset_id", dataset_id)
                    prediction_rows.append(pred)
            except Exception as exc:  # keep the experiment matrix running
                print(f"    failed {algorithm} fold={fold_id}: {exc}")
                metric_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "algorithm": algorithm,
                        "fold_id": fold_id,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    pd.DataFrame(fold_rows).to_csv(
        output_dir / f"{dataset_id}__cv_fold_summary.csv", index=False
    )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / f"{dataset_id}__cv_metrics_by_fold.csv", index=False)
    if topk_rows:
        pd.concat(topk_rows, ignore_index=True).to_csv(
            output_dir / f"{dataset_id}__cv_top_k_by_fold.csv", index=False
        )
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / f"{dataset_id}__cv_predictions.csv", index=False
        )

    used = metrics_df[metrics_df.get("status", "").eq("used")].copy()
    if not used.empty:
        for optional_col in [
            "fit_xgboost_scale_pos_weight",
            "fit_xgboost_class_importance_mode",
        ]:
            if optional_col not in used.columns:
                used[optional_col] = np.nan
        threshold_prefix = str(
            float(getattr(config, "VALIDATION_SCORE_THRESHOLD", 0.5))
        ).replace(".", "p")
        summary = (
            used.groupby(["dataset_id", "algorithm"], dropna=False)
            .agg(
                fold_count=("fold_id", "count"),
                mean_average_precision=("threshold_free_average_precision", "mean"),
                std_average_precision=("threshold_free_average_precision", "std"),
                mean_roc_auc=("threshold_free_roc_auc", "mean"),
                mean_precision=(f"threshold_{threshold_prefix}_precision", "mean"),
                mean_recall=(f"threshold_{threshold_prefix}_recall", "mean"),
                mean_f1=(f"threshold_{threshold_prefix}_f1", "mean"),
                mean_flagged_rate=(f"threshold_{threshold_prefix}_flagged_rate", "mean"),
                mean_fit_xgboost_scale_pos_weight=(
                    "fit_xgboost_scale_pos_weight",
                    "mean",
                ),
                fit_xgboost_class_importance_mode=(
                    "fit_xgboost_class_importance_mode",
                    "first",
                ),
            )
            .reset_index()
            .sort_values(
                ["dataset_id", "mean_average_precision"],
                ascending=[True, False],
            )
        )
        summary.to_csv(
            output_dir / f"{dataset_id}__cv_summary_by_model.csv", index=False
        )

    return {
        "dataset_id": dataset_id,
        "rows": int(len(df)),
        "machines": n_groups,
        "n_splits_used": n_splits,
        "cv_strategy": cv_strategy,
    }


def run() -> None:
    config.refresh_derived_config()
    step_dir = config.OUTPUT_DIR / "03_cross_validation"
    ensure_dir(step_dir)
    dataset_index = _load_dataset_index()

    summaries = [
        _cv_one_dataset(dataset_row, step_dir)
        for _, dataset_row in dataset_index.iterrows()
    ]

    all_metrics = []
    all_summary = []
    all_topk = []
    for dataset_id in dataset_index["dataset_id"]:
        p = step_dir / f"{dataset_id}__cv_metrics_by_fold.csv"
        if p.exists():
            all_metrics.append(pd.read_csv(p, low_memory=False))
        s = step_dir / f"{dataset_id}__cv_summary_by_model.csv"
        if s.exists():
            all_summary.append(pd.read_csv(s, low_memory=False))
        t = step_dir / f"{dataset_id}__cv_top_k_by_fold.csv"
        if t.exists():
            all_topk.append(pd.read_csv(t, low_memory=False))
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(
            step_dir / "cv_metrics_by_fold_all_datasets.csv", index=False
        )
    if all_summary:
        pd.concat(all_summary, ignore_index=True).to_csv(
            step_dir / "cv_summary_by_model_all_datasets.csv", index=False
        )
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
        )
        topk_summary.to_csv(
            step_dir / "cv_top_k_summary_all_datasets.csv", index=False
        )

    write_json(
        {
            "step": "03_cross_validation",
            "output_dir": str(step_dir),
            "models_to_run": list(config.MODELS_TO_RUN),
            "grouping_column": "machine_key",
            "random_state": int(config.RANDOM_STATE),
            "summaries": summaries,
        },
        step_dir / "run_summary.json",
    )
    print(f"03_cross_validation completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
