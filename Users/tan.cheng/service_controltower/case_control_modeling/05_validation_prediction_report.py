"""Step 05: train on the training split and inspect validation-set predictions.

This script is intentionally separate from 03_smoke_run.py.  The smoke run gives
quick aggregate validation metrics.  This step creates inspection-friendly files
that show which validation machines and time windows were scored as high risk.

Expected prerequisite:
    python 02_build_case_control_dataset.py

Typical usage:
    python 05_validation_prediction_report.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config
from cc_utils import (
    ensure_dir,
    fit_model_pipeline,
    make_model_pipeline,
    metrics_at_threshold,
    model_feature_importance_frame,
    predict_score,
    threshold_free_metrics,
    top_k_metrics,
    validate_dataset_features,
    write_json,
)


DATE_COLUMNS = [
    "window_start",
    "window_end",
    "future_claim_date",
    "control_no_claim_start",
    "control_no_claim_end",
]


METADATA_COLUMNS = [
    "dataset_id",
    "algorithm",
    "split",
    "window_name",
    "case_control_group_id",
    "row_role",
    "target",
    "machine_key",
    "full_model",
    "serial",
    "window_start",
    "window_end",
    "lead_max_days",
    "lead_min_days",
    "future_claim_date",
    "days_from_window_end_to_claim",
    "claim_episode_id",
    "claim_numbers",
    "claim_type_descriptions",
    "critical_fail_part_numbers",
    "control_number_within_group",
    "control_no_claim_start",
    "control_no_claim_end",
    "control_sampling_reason",
]


PREDICTION_COLUMNS = [
    "score",
    "score_rank_overall",
    "score_percentile",
    "prediction_threshold",
    "predicted_label",
    "top_1pct_flag",
    "top_5pct_flag",
    "top_10pct_flag",
    "top_20pct_flag",
]


def _load_dataset_index() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {path}. Run 02_build_case_control_dataset.py first."
        )
    return pd.read_csv(path)


def _read_dataset(path_value: str) -> pd.DataFrame:
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    parse_dates = [c for c in DATE_COLUMNS if c in pd.read_csv(path, nrows=0).columns]
    return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)


def _configured_top_k_rates() -> list[float]:
    rates = getattr(config, "VALIDATION_TOP_K_RATES", getattr(config, "SMOKE_TOP_K_RATES", [0.01, 0.05, 0.10, 0.20]))
    return [float(x) for x in rates]


def _configured_threshold() -> float:
    return float(getattr(config, "VALIDATION_SCORE_THRESHOLD", 0.50))


def _top_k_flag_column(rate: float) -> str:
    pct = int(round(rate * 100))
    return f"top_{pct}pct_flag"


def _add_prediction_columns(df: pd.DataFrame, score: np.ndarray, threshold: float, top_k_rates: Sequence[float]) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["score"] = np.asarray(score, dtype=float)
    out["prediction_threshold"] = float(threshold)
    out["predicted_label"] = (out["score"] >= float(threshold)).astype(int)
    out["score_rank_overall"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["score_percentile"] = out["score"].rank(pct=True, ascending=True, method="average")

    order = out.sort_values("score", ascending=False).index.to_numpy()
    n = len(out)
    for rate in top_k_rates:
        col = _top_k_flag_column(rate)
        out[col] = 0
        k = int(np.ceil(n * float(rate))) if n else 0
        k = max(1, min(k, n)) if n else 0
        if k:
            out.loc[order[:k], col] = 1
    return out


def _select_output_columns(pred: pd.DataFrame, include_features: bool) -> pd.DataFrame:
    first_cols = [c for c in METADATA_COLUMNS + PREDICTION_COLUMNS if c in pred.columns]
    if include_features:
        feature_cols = [
            c for c in list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
            if c in pred.columns and c not in first_cols
        ]
        other_cols = [c for c in pred.columns if c not in first_cols and c not in feature_cols]
        return pred[first_cols + feature_cols + other_cols]
    return pred[first_cols]


def _summarize_machine_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["dataset_id", "algorithm", "machine_key"]
    for keys, g in pred.groupby(group_cols, dropna=False):
        g_sorted_score = g.sort_values("score", ascending=False, kind="mergesort")
        g_sorted_time = g.sort_values("window_end", kind="mergesort") if "window_end" in g.columns else g
        top = g_sorted_score.iloc[0]
        latest = g_sorted_time.iloc[-1]
        rows.append({
            "dataset_id": keys[0],
            "algorithm": keys[1],
            "machine_key": keys[2],
            "full_model": top.get("full_model", ""),
            "serial": top.get("serial", ""),
            "validation_window_rows": int(len(g)),
            "positive_window_rows": int(pd.to_numeric(g.get("target", 0), errors="coerce").fillna(0).sum()),
            "case_rows": int(g.get("row_role", pd.Series(dtype=str)).astype(str).eq("case").sum()),
            "control_rows": int(g.get("row_role", pd.Series(dtype=str)).astype(str).eq("control").sum()),
            "max_score": float(g["score"].max()),
            "mean_score": float(g["score"].mean()),
            "min_score": float(g["score"].min()),
            "latest_window_end": latest.get("window_end", pd.NaT),
            "latest_window_score": float(latest.get("score", np.nan)),
            "top_score_window_start": top.get("window_start", pd.NaT),
            "top_score_window_end": top.get("window_end", pd.NaT),
            "top_score": float(top.get("score", np.nan)),
            "top_score_target": int(top.get("target", 0)) if pd.notna(top.get("target", np.nan)) else np.nan,
            "top_score_row_role": top.get("row_role", ""),
            "top_score_group_id": top.get("case_control_group_id", ""),
            "top_score_claim_episode_id": top.get("claim_episode_id", ""),
            "top_score_future_claim_date": top.get("future_claim_date", pd.NaT),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "algorithm", "max_score", "mean_score"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )


def _summarize_case_control_group_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "case_control_group_id" not in pred.columns:
        return pd.DataFrame()

    rows = []
    group_cols = ["dataset_id", "algorithm", "case_control_group_id"]
    for keys, g in pred.groupby(group_cols, dropna=False):
        ranked = g.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        ranked["rank_within_group"] = np.arange(1, len(ranked) + 1)
        cases = ranked[ranked["target"].astype(int).eq(1)] if "target" in ranked.columns else ranked.iloc[0:0]
        controls = ranked[ranked["target"].astype(int).eq(0)] if "target" in ranked.columns else ranked.iloc[0:0]
        case = cases.iloc[0] if len(cases) else None
        case_rank = int(case["rank_within_group"]) if case is not None else np.nan
        case_score = float(case["score"]) if case is not None else np.nan
        max_control_score = float(controls["score"].max()) if len(controls) else np.nan
        rows.append({
            "dataset_id": keys[0],
            "algorithm": keys[1],
            "case_control_group_id": keys[2],
            "window_name": ranked.iloc[0].get("window_name", ""),
            "window_start": ranked.iloc[0].get("window_start", pd.NaT),
            "window_end": ranked.iloc[0].get("window_end", pd.NaT),
            "group_size": int(len(ranked)),
            "case_count": int(len(cases)),
            "control_count": int(len(controls)),
            "case_machine_key": case.get("machine_key", "") if case is not None else "",
            "case_full_model": case.get("full_model", "") if case is not None else "",
            "case_serial": case.get("serial", "") if case is not None else "",
            "case_score": case_score,
            "case_rank_within_group": case_rank,
            "case_is_top_score": bool(case_rank == 1) if pd.notna(case_rank) else False,
            "case_is_top_2": bool(case_rank <= 2) if pd.notna(case_rank) else False,
            "max_control_score": max_control_score,
            "case_score_minus_max_control_score": float(case_score - max_control_score) if pd.notna(case_score) and pd.notna(max_control_score) else np.nan,
            "top_rank_machine_key": ranked.iloc[0].get("machine_key", ""),
            "top_rank_row_role": ranked.iloc[0].get("row_role", ""),
            "top_rank_target": int(ranked.iloc[0].get("target", 0)) if pd.notna(ranked.iloc[0].get("target", np.nan)) else np.nan,
            "top_rank_score": float(ranked.iloc[0].get("score", np.nan)),
            "future_claim_date": case.get("future_claim_date", pd.NaT) if case is not None else pd.NaT,
            "claim_episode_id": case.get("claim_episode_id", "") if case is not None else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "algorithm", "window_end", "case_control_group_id"],
        kind="mergesort",
    )


def _group_ranking_metrics(group_summary: pd.DataFrame) -> dict:
    if group_summary.empty:
        return {}
    usable = group_summary[group_summary["case_count"] > 0].copy()
    if usable.empty:
        return {}
    ranks = pd.to_numeric(usable["case_rank_within_group"], errors="coerce")
    return {
        "case_control_groups": int(len(usable)),
        "mean_case_rank_within_group": float(ranks.mean()),
        "median_case_rank_within_group": float(ranks.median()),
        "case_top_score_rate": float(usable["case_is_top_score"].mean()),
        "case_top_2_rate": float(usable["case_is_top_2"].mean()),
        "mean_reciprocal_case_rank": float((1.0 / ranks.replace(0, np.nan)).mean()),
        "mean_case_score_minus_max_control_score": float(pd.to_numeric(usable["case_score_minus_max_control_score"], errors="coerce").mean()),
    }


def _train_score_one_dataset(dataset_row: pd.Series, output_dir: Path) -> dict:
    dataset_id = dataset_row["dataset_id"]
    train_path = dataset_row["training_dataset_path"]
    validation_path = dataset_row["validation_dataset_path"]

    train_df = _read_dataset(train_path)
    valid_df = _read_dataset(validation_path)
    validate_dataset_features(train_df, config)
    validate_dataset_features(valid_df, config)

    X_train = train_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
    y_train = train_df["target"].astype(int)
    X_valid = valid_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
    y_valid = valid_df["target"].astype(int)

    threshold = _configured_threshold()
    top_k_rates = _configured_top_k_rates()
    include_features = bool(getattr(config, "VALIDATION_INCLUDE_FEATURE_COLUMNS", True))

    metric_rows = []
    topk_rows = []
    group_metric_rows = []
    run_summaries = []

    for algorithm in config.MODELS_TO_RUN:
        print(f"  Validation prediction report dataset={dataset_id} algorithm={algorithm}")
        model = make_model_pipeline(algorithm, config)
        if model is None:
            metric_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "status": "skipped_missing_dependency",
            })
            continue

        try:
            fit_metadata = fit_model_pipeline(model, algorithm, X_train, y_train, config)
            score = predict_score(model, X_valid, algorithm)
            free = threshold_free_metrics(y_valid, score)
            thresh = metrics_at_threshold(y_valid, score, threshold=threshold)

            pred = valid_df.copy()
            pred.insert(0, "algorithm", algorithm)
            pred.insert(0, "dataset_id", dataset_id)
            pred = _add_prediction_columns(pred, score, threshold=threshold, top_k_rates=top_k_rates)
            pred = _select_output_columns(pred, include_features=include_features)
            pred = pred.sort_values(
                ["dataset_id", "algorithm", "score", "machine_key", "window_end"],
                ascending=[True, True, False, True, True],
                kind="mergesort",
            )

            compact = _select_output_columns(pred, include_features=False)
            machine_summary = _summarize_machine_predictions(pred)
            group_summary = _summarize_case_control_group_predictions(pred)
            group_metrics = _group_ranking_metrics(group_summary)

            pred.to_csv(output_dir / f"{dataset_id}__{algorithm}__validation_window_predictions.csv", index=False)
            compact.to_csv(output_dir / f"{dataset_id}__{algorithm}__validation_window_predictions_compact.csv", index=False)
            if not machine_summary.empty:
                machine_summary.to_csv(output_dir / f"{dataset_id}__{algorithm}__validation_machine_summary.csv", index=False)
            if not group_summary.empty:
                group_summary.to_csv(output_dir / f"{dataset_id}__{algorithm}__validation_case_control_group_summary.csv", index=False)

            imp = model_feature_importance_frame(model, algorithm)
            if not imp.empty:
                imp.insert(0, "dataset_id", dataset_id)
                imp.to_csv(output_dir / f"{dataset_id}__{algorithm}__validation_feature_importance.csv", index=False)

            if bool(getattr(config, "VALIDATION_SAVE_MODEL_ARTIFACTS", False)):
                try:
                    import joblib
                    model_dir = output_dir / "models"
                    ensure_dir(model_dir)
                    joblib.dump(model, model_dir / f"{dataset_id}__{algorithm}__fitted_on_train.joblib")
                except Exception as exc:
                    print(f"    warning: model artifact was not saved: {exc}")

            metric_row = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "status": "used",
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(valid_df)),
                "train_positive_rows": int(y_train.sum()),
                "validation_positive_rows": int(y_valid.sum()),
                "validation_positive_rate": float(y_valid.mean()) if len(y_valid) else np.nan,
            }
            metric_row.update({f"fit_{k}": v for k, v in fit_metadata.items() if k != "algorithm"})
            metric_row.update({f"threshold_free_{k}": v for k, v in free.items()})
            metric_row.update({f"threshold_{str(threshold).replace('.', 'p')}_{k}": v for k, v in thresh.items()})
            metric_row.update({f"group_rank_{k}": v for k, v in group_metrics.items()})
            metric_rows.append(metric_row)

            topk = top_k_metrics(y_valid, score, top_k_rates)
            topk.insert(0, "algorithm", algorithm)
            topk.insert(0, "dataset_id", dataset_id)
            topk_rows.append(topk)

            if group_metrics:
                gm = {"dataset_id": dataset_id, "algorithm": algorithm}
                gm.update(group_metrics)
                group_metric_rows.append(gm)

            run_summaries.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(valid_df)),
                "validation_machines": int(valid_df["machine_key"].nunique(dropna=True)),
                "validation_groups": int(valid_df["case_control_group_id"].nunique(dropna=True)),
                "prediction_threshold": float(threshold),
                "window_prediction_file": str(output_dir / f"{dataset_id}__{algorithm}__validation_window_predictions.csv"),
                "compact_prediction_file": str(output_dir / f"{dataset_id}__{algorithm}__validation_window_predictions_compact.csv"),
                "machine_summary_file": str(output_dir / f"{dataset_id}__{algorithm}__validation_machine_summary.csv"),
                "case_control_group_summary_file": str(output_dir / f"{dataset_id}__{algorithm}__validation_case_control_group_summary.csv"),
            })

        except Exception as exc:
            print(f"    failed {algorithm}: {exc}")
            metric_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "status": "failed",
                "error": str(exc),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / f"{dataset_id}__validation_metrics_by_model.csv", index=False)
    if topk_rows:
        pd.concat(topk_rows, ignore_index=True).to_csv(output_dir / f"{dataset_id}__validation_top_k_by_model.csv", index=False)
    if group_metric_rows:
        pd.DataFrame(group_metric_rows).to_csv(output_dir / f"{dataset_id}__validation_group_ranking_metrics.csv", index=False)

    return {
        "dataset_id": dataset_id,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "validation_machines": int(valid_df["machine_key"].nunique(dropna=True)),
        "validation_groups": int(valid_df["case_control_group_id"].nunique(dropna=True)),
        "models_attempted": list(config.MODELS_TO_RUN),
        "model_outputs": run_summaries,
    }


def run() -> None:
    step_dir = config.OUTPUT_DIR / "05_validation_prediction_report"
    ensure_dir(step_dir)
    dataset_index = _load_dataset_index()

    summaries = []
    for _, dataset_row in dataset_index.iterrows():
        summaries.append(_train_score_one_dataset(dataset_row, step_dir))

    all_metrics = []
    all_topk = []
    all_group_metrics = []
    for dataset_id in dataset_index["dataset_id"]:
        p = step_dir / f"{dataset_id}__validation_metrics_by_model.csv"
        if p.exists():
            all_metrics.append(pd.read_csv(p))
        t = step_dir / f"{dataset_id}__validation_top_k_by_model.csv"
        if t.exists():
            all_topk.append(pd.read_csv(t))
        g = step_dir / f"{dataset_id}__validation_group_ranking_metrics.csv"
        if g.exists():
            all_group_metrics.append(pd.read_csv(g))

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(step_dir / "validation_metrics_all_datasets.csv", index=False)
    if all_topk:
        pd.concat(all_topk, ignore_index=True).to_csv(step_dir / "validation_top_k_all_datasets.csv", index=False)
    if all_group_metrics:
        pd.concat(all_group_metrics, ignore_index=True).to_csv(step_dir / "validation_group_ranking_metrics_all_datasets.csv", index=False)

    write_json(
        {
            "step": "05_validation_prediction_report",
            "output_dir": str(step_dir),
            "summaries": summaries,
            "models_to_run": config.MODELS_TO_RUN,
            "training_split": "train",
            "evaluation_split": "validation",
            "prediction_threshold": _configured_threshold(),
            "top_k_rates": _configured_top_k_rates(),
            "notes": [
                "Each configured model is fitted on the chronological training split and scored on the chronological validation split.",
                "The main output is validation_window_predictions, sorted by risk score and containing machine, time-window, target, score, rank, and optional feature columns.",
                "The machine summary collapses validation window scores by machine.",
                "The case-control group summary checks whether the true claim case outranks its matched controls in each validation group.",
                "The test split is not used by this step.",
            ],
        },
        step_dir / "run_summary.json",
    )
    print(f"05_validation_prediction_report completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
