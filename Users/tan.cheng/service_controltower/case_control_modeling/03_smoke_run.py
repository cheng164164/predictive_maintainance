"""Step 03: quick train/validation smoke run using configured holdout split."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from cc_utils import (
    ensure_dir,
    make_model_pipeline,
    fit_model_pipeline,
    metrics_at_threshold,
    predict_score,
    prediction_frame,
    model_feature_importance_frame,
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


def _read_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["window_start", "window_end", "future_claim_date"])


def _evaluate_one_dataset(dataset_row: pd.Series, output_dir) -> dict:
    dataset_id = dataset_row["dataset_id"]
    train_path = dataset_row["training_dataset_path"]
    validation_path = dataset_row["validation_dataset_path"]

    train_df = _read_dataset(train_path)
    valid_df = _read_dataset(validation_path)
    validate_dataset_features(train_df, config)
    validate_dataset_features(valid_df, config)

    split_summary = pd.DataFrame([
        {
            "split": "train",
            "rows": len(train_df),
            "positive_rows": int(train_df["target"].sum()),
            "positive_rate": float(train_df["target"].mean()) if len(train_df) else np.nan,
            "groups": train_df["case_control_group_id"].nunique(),
            "path": train_path,
        },
        {
            "split": "validation",
            "rows": len(valid_df),
            "positive_rows": int(valid_df["target"].sum()),
            "positive_rate": float(valid_df["target"].mean()) if len(valid_df) else np.nan,
            "groups": valid_df["case_control_group_id"].nunique(),
            "path": validation_path,
        },
    ])
    split_summary.to_csv(output_dir / f"{dataset_id}__split_summary.csv", index=False)

    X_train = train_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y_train = train_df["target"].astype(int)
    X_valid = valid_df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]
    y_valid = valid_df["target"].astype(int)

    metric_rows = []
    topk_rows = []
    prediction_rows = []
    importance_rows = []
    for algorithm in config.MODELS_TO_RUN:
        print(f"  Smoke run dataset={dataset_id} algorithm={algorithm}")
        model = make_model_pipeline(algorithm, config)
        if model is None:
            print(f"    skipped {algorithm}: optional dependency not installed")
            metric_rows.append({"dataset_id": dataset_id, "algorithm": algorithm, "status": "skipped_missing_dependency"})
            continue
        try:
            fit_metadata = fit_model_pipeline(model, algorithm, X_train, y_train, config)
            score = predict_score(model, X_valid, algorithm)
            free = threshold_free_metrics(y_valid, score)
            thresh = metrics_at_threshold(y_valid, score, threshold=0.5)
            row = {"dataset_id": dataset_id, "algorithm": algorithm, "status": "used"}
            row.update({f"fit_{k}": v for k, v in fit_metadata.items() if k != "algorithm"})
            row.update({f"threshold_free_{k}": v for k, v in free.items()})
            row.update({f"threshold_0p5_{k}": v for k, v in thresh.items()})
            metric_rows.append(row)

            tk = top_k_metrics(y_valid, score, config.SMOKE_TOP_K_RATES)
            tk.insert(0, "algorithm", algorithm)
            tk.insert(0, "dataset_id", dataset_id)
            topk_rows.append(tk)

            imp = model_feature_importance_frame(model, algorithm)
            if not imp.empty:
                imp.insert(0, "dataset_id", dataset_id)
                importance_rows.append(imp.head(50))

            if config.SAVE_SMOKE_PREDICTIONS:
                pred = prediction_frame(valid_df, score)
                pred.insert(0, "algorithm", algorithm)
                pred.insert(0, "dataset_id", dataset_id)
                prediction_rows.append(pred)
        except Exception as exc:
            print(f"    failed {algorithm}: {exc}")
            metric_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "status": "failed",
                "error": str(exc),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / f"{dataset_id}__smoke_metrics_by_model.csv", index=False)
    if topk_rows:
        pd.concat(topk_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__smoke_top_k_by_model.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__smoke_predictions.csv", index=False)
    if importance_rows:
        pd.concat(importance_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__smoke_feature_importance_top50.csv", index=False)
    return {
        "dataset_id": dataset_id,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "train_groups": int(train_df["case_control_group_id"].nunique()),
        "validation_groups": int(valid_df["case_control_group_id"].nunique()),
        "models_attempted": list(config.MODELS_TO_RUN),
    }


def run() -> None:
    step_dir = config.OUTPUT_DIR / "03_smoke_run"
    ensure_dir(step_dir)
    dataset_index = _load_dataset_index()

    summaries = []
    for _, dataset_row in dataset_index.iterrows():
        summaries.append(_evaluate_one_dataset(dataset_row, step_dir))

    # Combined summary table across datasets.
    all_metrics = []
    all_topk = []
    all_importance = []
    for dataset_id in dataset_index["dataset_id"]:
        p = step_dir / f"{dataset_id}__smoke_metrics_by_model.csv"
        if p.exists():
            all_metrics.append(pd.read_csv(p))
        t = step_dir / f"{dataset_id}__smoke_top_k_by_model.csv"
        if t.exists():
            all_topk.append(pd.read_csv(t))
        imp = step_dir / f"{dataset_id}__smoke_feature_importance_top50.csv"
        if imp.exists():
            all_importance.append(pd.read_csv(imp))
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(step_dir / "smoke_metrics_all_datasets.csv", index=False)
    if all_topk:
        pd.concat(all_topk, ignore_index=True).to_csv(step_dir / "smoke_top_k_all_datasets.csv", index=False)
    if all_importance:
        pd.concat(all_importance, ignore_index=True).to_csv(step_dir / "smoke_feature_importance_top50_all_datasets.csv", index=False)

    write_json(
        {
            "step": "03_smoke_run",
            "output_dir": str(step_dir),
            "summaries": summaries,
            "models_to_run": config.MODELS_TO_RUN,
            "evaluation_split": "validation",
            "training_split": "train",
            "notes": [
                "Smoke run trains on the chronological train split and evaluates on the chronological validation split.",
                "The test split is not used here; keep it for final holdout evaluation after model choices are locked.",
            ],
        },
        step_dir / "run_summary.json",
    )
    print(f"03_smoke_run completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
