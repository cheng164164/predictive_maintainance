"""Step 08: evaluate validation/test sensitivity to negative:positive ratio.

The model is fitted once on the fixed training dataset. It is then scored on the
nested fixed holdout cohorts created by Step 02, for example 1:1 through 5:1.
Because every holdout row represents a different physical machine, row-level
Top-K metrics are also machine-level Top-K metrics. When claim-within-horizon
evaluation is active, the script supports either backward-compatible relabeling
of the same rows or exact horizon-specific random prevalence cohorts. Prediction
files retain the actual next claim date and days from window_end to that claim.

This script intentionally scores the test ratio cohorts. Use it for prevalence
sensitivity analysis after the model/data design is reasonably stable; do not use
repeated test results to tune hyperparameters.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from cc_utils import (
    bootstrap_ranking_metric_intervals,
    configured_evaluation_horizons,
    ensure_dir,
    fit_model_pipeline,
    get_evaluation_target,
    make_model_pipeline,
    metrics_at_threshold,
    predict_score,
    threshold_free_metrics,
    top_k_metrics,
    top_n_metrics,
    validate_dataset_features,
    write_json,
)


DATE_COLUMNS = [
    "window_start",
    "window_end",
    "future_claim_date",
    "control_no_claim_start",
    "control_no_claim_end",
    "next_claim_date_on_or_after_window_end",
    "as_of_anchor_date",
    "as_of_actual_next_claim_date",
]


def _read_dataset(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _load_dataset_index(dataset_index_path: str | Path | None = None) -> pd.DataFrame:
    path = (
        Path(dataset_index_path)
        if dataset_index_path is not None
        else config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {path}. Run 02_build_case_control_dataset.py first."
        )
    return pd.read_csv(path, low_memory=False)


def _horizon_ratio_mode() -> str:
    mode = str(
        getattr(config, "HOLDOUT_HORIZON_RATIO_MODE", "reuse_same_rows")
    ).strip().lower()
    aliases = {
        "reuse": "reuse_same_rows",
        "same_rows": "reuse_same_rows",
        "reuse_same_rows": "reuse_same_rows",
        "horizon_specific": "horizon_specific_random",
        "per_horizon_random": "horizon_specific_random",
        "horizon_specific_random": "horizon_specific_random",
    }
    if mode not in aliases:
        raise ValueError(
            "HOLDOUT_HORIZON_RATIO_MODE must be 'reuse_same_rows' or "
            "'horizon_specific_random'."
        )
    return aliases[mode]


def _claim_horizon_mode_active() -> bool:
    mode = str(getattr(config, "EVALUATION_TARGET_MODE", "training_target")).strip().lower()
    return mode in {
        "claim_within_horizon", "future_claim_horizon", "relaxed",
        "relaxed_future_claim",
    }


def _load_ratio_index(dataset_row: pd.Series) -> pd.DataFrame:
    value = dataset_row.get("holdout_ratio_index_path")
    if value is None or pd.isna(value) or not str(value).strip():
        value = dataset_row.get("random_holdout_ratio_index_path")
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(
            "Dataset index does not contain holdout_ratio_index_path. "
            "Re-run Step 02 with a fixed holdout ratio design."
        )
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Holdout ratio index not found: {path}")
    ratio_index = pd.read_csv(path, low_memory=False)
    required = {
        "split",
        "negative_to_positive_ratio_requested",
        "positive_rows",
        "negative_rows",
        "dataset_path",
    }
    missing = sorted(required - set(ratio_index.columns))
    if missing:
        raise ValueError(f"Holdout ratio index is missing columns: {missing}")
    ratio_index["negative_to_positive_ratio_requested"] = pd.to_numeric(
        ratio_index["negative_to_positive_ratio_requested"], errors="raise"
    ).astype(int)
    return ratio_index.sort_values(
        ["split", "negative_to_positive_ratio_requested"], kind="mergesort"
    ).reset_index(drop=True)

def _load_horizon_ratio_index(dataset_row: pd.Series) -> pd.DataFrame:
    value = dataset_row.get("horizon_specific_ratio_index_path")
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(
            "Dataset index does not contain horizon_specific_ratio_index_path. "
            "Re-run Step 02 with HOLDOUT_HORIZON_RATIO_MODE="
            "'horizon_specific_random' and claim-within-horizon evaluation enabled."
        )
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Horizon-specific ratio index not found: {path}")
    index = pd.read_csv(path, low_memory=False)
    required = {
        "split", "evaluation_horizon_days",
        "negative_to_positive_ratio_requested", "positive_rows",
        "negative_rows", "dataset_path",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Horizon-specific ratio index is missing columns: {missing}")
    for col in ["evaluation_horizon_days", "negative_to_positive_ratio_requested"]:
        index[col] = pd.to_numeric(index[col], errors="raise").astype(int)
    return index.sort_values(
        ["split", "evaluation_horizon_days", "negative_to_positive_ratio_requested"],
        kind="mergesort",
    ).reset_index(drop=True)


def _top_k_rates() -> list[float]:
    values = getattr(config, "VALIDATION_TOP_K_RATES", [0.01, 0.05, 0.10, 0.20])
    return [float(x) for x in values]


def _top_n_counts() -> list[int]:
    values = getattr(config, "HOLDOUT_TOP_N_COUNTS", [10, 20, 50])
    counts = sorted({int(x) for x in values})
    if not counts or any(x < 1 for x in counts):
        raise ValueError("HOLDOUT_TOP_N_COUNTS must contain integers >= 1.")
    return counts


def _bootstrap_enabled() -> bool:
    return bool(getattr(config, "HOLDOUT_BOOTSTRAP_ENABLED", True))


def _bootstrap_n_resamples() -> int:
    value = int(getattr(config, "HOLDOUT_BOOTSTRAP_N_RESAMPLES", 1000))
    if value < 1:
        raise ValueError("HOLDOUT_BOOTSTRAP_N_RESAMPLES must be at least 1.")
    return value


def _bootstrap_confidence_level() -> float:
    value = float(getattr(config, "HOLDOUT_BOOTSTRAP_CONFIDENCE_LEVEL", 0.95))
    if not 0 < value < 1:
        raise ValueError("HOLDOUT_BOOTSTRAP_CONFIDENCE_LEVEL must be between 0 and 1.")
    return value


def _bootstrap_seed(*parts: object) -> int:
    base = int(getattr(config, "HOLDOUT_BOOTSTRAP_RANDOM_STATE", 20260727))
    payload = "|".join([str(base), *[str(x) for x in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _threshold() -> float:
    return float(getattr(config, "VALIDATION_SCORE_THRESHOLD", 0.50))


def _evaluation_horizon_sweep() -> list[Optional[int]]:
    mode = str(getattr(config, "EVALUATION_TARGET_MODE", "training_target")).strip().lower()
    if mode in {"training", "train", "target", "training_target", "original", "original_target"}:
        return [None]
    horizons = configured_evaluation_horizons(config)
    if not horizons:
        raise ValueError(
            "EVALUATION_CLAIM_HORIZON_DAYS must contain at least one horizon when "
            "EVALUATION_TARGET_MODE='claim_within_horizon'."
        )
    return [int(x) for x in horizons]


def _horizon_slug(horizon_days: Optional[int]) -> str:
    return "training_target" if horizon_days is None else f"horizon_{int(horizon_days)}d"


def _safe_name(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text)).strip("_") or "item"


def _prediction_frame(
    df: pd.DataFrame,
    score: np.ndarray,
    y_eval: pd.Series,
    dataset_id: str,
    algorithm: str,
    split_name: str,
    ratio: int,
    target_col: str,
    target_mode: str,
    horizon_days: Optional[int],
) -> pd.DataFrame:
    columns = [
        c for c in [
            "machine_key",
            "full_model",
            "serial",
            "window_name",
            "window_start",
            "window_end",
            "row_role",
            "target",
            "future_claim_date",
            "next_claim_date_on_or_after_window_end",
            "days_to_next_claim_on_or_after_window_end",
            "has_future_claim_on_or_after_window_end",
            "future_claim_lead_time_bucket",
            "as_of_anchor_date",
            "as_of_prediction_horizon_days",
            "as_of_actual_next_claim_date",
            "as_of_days_to_next_claim",
            "claim_episode_id",
            "negative_sampling_type",
            "holdout_sampling_design",
            "holdout_match_id",
            "matched_positive_machine_key",
            "holdout_positive_rank",
            "matched_holdout_positive_rank",
            "holdout_negative_rank",
            "holdout_control_rank_within_positive",
        ] if c in df.columns
    ]
    if bool(getattr(config, "VALIDATION_INCLUDE_FEATURE_COLUMNS", True)):
        for feature in list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES):
            if feature in df.columns and feature not in columns:
                columns.append(feature)
    pred = df[columns].copy().reset_index(drop=True)
    pred.insert(0, "evaluation_target", y_eval.to_numpy())
    pred.insert(0, "evaluation_target_col", target_col)
    pred.insert(0, "evaluation_target_mode", target_mode)
    pred.insert(0, "evaluation_horizon_days", horizon_days)
    pred.insert(0, "negative_to_positive_ratio_requested", int(ratio))
    pred.insert(0, "split", split_name)
    pred.insert(0, "algorithm", algorithm)
    pred.insert(0, "dataset_id", dataset_id)
    pred["score"] = np.asarray(score, dtype=float)
    pred["score_rank"] = pred["score"].rank(
        method="first", ascending=False
    ).astype(int)
    pred["predicted_label"] = (pred["score"] >= _threshold()).astype(int)
    return pred.sort_values(
        ["score", "machine_key"], ascending=[False, True], kind="mergesort"
    )


def _review_summary(
    metrics: pd.DataFrame,
    topk: pd.DataFrame,
    topn: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metrics = metrics.copy()
    topk = topk.copy()
    topn = topn.copy()
    for frame in [metrics, topk, topn]:
        if not frame.empty and "evaluation_horizon_days" in frame.columns:
            frame["evaluation_horizon_days"] = pd.to_numeric(
                frame["evaluation_horizon_days"], errors="coerce"
            ).fillna(-1).astype(int)
    key_cols = [
        "dataset_id",
        "algorithm",
        "split",
        "negative_to_positive_ratio_requested",
        "evaluation_target_mode",
        "evaluation_horizon_days",
    ]
    requested_base_cols = key_cols + [
        "holdout_negative_sampling_mode",
        "evaluation_rows",
        "evaluation_positive_rows",
        "evaluation_negative_rows",
        "evaluation_positive_rate",
        "actual_negative_to_positive_ratio",
        "threshold_free_average_precision",
        "threshold_free_average_precision_ci_lower",
        "threshold_free_average_precision_ci_upper",
        "threshold_free_roc_auc",
        "threshold_free_roc_auc_ci_lower",
        "threshold_free_roc_auc_ci_upper",
    ]
    base_cols = [c for c in requested_base_cols if c in metrics.columns]
    base = metrics[base_cols].drop_duplicates(key_cols).copy()

    if not topk.empty:
        tk = topk.copy()
        tk["top_k_label"] = tk["top_k_rate"].map(
            lambda x: f"top_{int(round(float(x) * 100))}pct"
        )
        for value_col in [
            "flagged_count",
            "precision_at_k",
            "precision_at_k_ci_lower",
            "precision_at_k_ci_upper",
            "recall_at_k",
            "recall_at_k_ci_lower",
            "recall_at_k_ci_upper",
            "lift_vs_random",
            "lift_vs_random_ci_lower",
            "lift_vs_random_ci_upper",
        ]:
            if value_col not in tk.columns:
                continue
            wide = tk.pivot_table(
                index=key_cols,
                columns="top_k_label",
                values=value_col,
                aggfunc="first",
            )
            wide.columns = [f"{label}_{value_col}" for label in wide.columns]
            base = base.merge(wide.reset_index(), on=key_cols, how="left")

    if not topn.empty:
        tn = topn.copy()
        tn["top_n_label"] = tn["top_n_requested"].map(
            lambda x: f"top_{int(x)}_machines"
        )
        for value_col in [
            "flagged_count",
            "true_positive_at_n",
            "precision_at_n",
            "precision_at_n_ci_lower",
            "precision_at_n_ci_upper",
            "recall_at_n",
            "recall_at_n_ci_lower",
            "recall_at_n_ci_upper",
            "lift_vs_random",
            "lift_vs_random_ci_lower",
            "lift_vs_random_ci_upper",
        ]:
            if value_col not in tn.columns:
                continue
            wide = tn.pivot_table(
                index=key_cols,
                columns="top_n_label",
                values=value_col,
                aggfunc="first",
            )
            wide.columns = [f"{label}_{value_col}" for label in wide.columns]
            base = base.merge(wide.reset_index(), on=key_cols, how="left")

    sort_cols = [
        c for c in [
            "dataset_id", "algorithm", "split", "evaluation_horizon_days",
            "negative_to_positive_ratio_requested",
        ] if c in base.columns
    ]
    base = base.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if "evaluation_horizon_days" in base.columns:
        base["evaluation_horizon_days"] = base["evaluation_horizon_days"].replace(-1, np.nan)
    return base

def _errorbar_yerr(group: pd.DataFrame, point_col: str, lower_col: str, upper_col: str):
    if lower_col not in group.columns or upper_col not in group.columns:
        return None
    point = pd.to_numeric(group[point_col], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(group[lower_col], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(group[upper_col], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(lower).any() and np.isfinite(upper).any()):
        return None
    return np.vstack([np.maximum(point - lower, 0.0), np.maximum(upper - point, 0.0)])


def _save_metric_plot(review: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    sub = review[review["split"].eq(split_name)].copy()
    if sub.empty:
        return
    group_cols = ["dataset_id", "algorithm"]
    if "evaluation_horizon_days" in sub.columns:
        group_cols.append("evaluation_horizon_days")
    for group_key, group in sub.groupby(group_cols, dropna=False):
        if len(group_cols) == 3:
            dataset_id, algorithm, horizon_days = group_key
        else:
            dataset_id, algorithm = group_key
            horizon_days = None
        horizon_label = _horizon_slug(
            None if pd.isna(horizon_days) else int(horizon_days)
        )
        group = group.sort_values("negative_to_positive_ratio_requested")
        x = group["negative_to_positive_ratio_requested"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            x,
            group["threshold_free_average_precision"],
            yerr=_errorbar_yerr(
                group,
                "threshold_free_average_precision",
                "threshold_free_average_precision_ci_lower",
                "threshold_free_average_precision_ci_upper",
            ),
            marker="o",
            capsize=3,
            label="Average precision",
        )
        ax.errorbar(
            x,
            group["threshold_free_roc_auc"],
            yerr=_errorbar_yerr(
                group,
                "threshold_free_roc_auc",
                "threshold_free_roc_auc_ci_lower",
                "threshold_free_roc_auc_ci_upper",
            ),
            marker="o",
            capsize=3,
            label="ROC AUC",
        )
        ax.set_xlabel("Negative-to-positive ratio")
        ax.set_ylabel("Metric")
        ax.set_title(
            f"{split_name.title()} metric sensitivity: {dataset_id} / {algorithm} / {horizon_label}"
        )
        ax.set_xticks(group["negative_to_positive_ratio_requested"].tolist())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir
            / (
                f"{_safe_name(dataset_id)}__{_safe_name(algorithm)}__{split_name}__"
                f"{horizon_label}__ap_roc_ratio.png"
            ),
            dpi=160,
        )
        plt.close(fig)


def _save_fixed_top_n_plots(topn: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    sub = topn[topn["split"].eq(split_name)].copy()
    if sub.empty:
        return
    group_cols = ["dataset_id", "algorithm"]
    if "evaluation_horizon_days" in sub.columns:
        group_cols.append("evaluation_horizon_days")
    for group_key, model_group in sub.groupby(group_cols, dropna=False):
        if len(group_cols) == 3:
            dataset_id, algorithm, horizon_days = group_key
        else:
            dataset_id, algorithm = group_key
            horizon_days = None
        horizon_label = _horizon_slug(
            None if pd.isna(horizon_days) else int(horizon_days)
        )
        for metric, ylabel in [
            ("precision_at_n", "Precision"),
            ("recall_at_n", "Recall"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for top_n, group in model_group.groupby("top_n_requested"):
                group = group.sort_values("negative_to_positive_ratio_requested")
                ax.errorbar(
                    group["negative_to_positive_ratio_requested"],
                    group[metric],
                    yerr=_errorbar_yerr(
                        group,
                        metric,
                        f"{metric}_ci_lower",
                        f"{metric}_ci_upper",
                    ),
                    marker="o",
                    capsize=3,
                    label=f"Top {int(top_n)} machines",
                )
            ax.set_xlabel("Negative-to-positive ratio")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{split_name.title()} fixed-workload {ylabel.lower()}: "
                f"{dataset_id} / {algorithm} / {horizon_label}"
            )
            ax.set_xticks(
                sorted(model_group["negative_to_positive_ratio_requested"].unique().tolist())
            )
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_dir
                / (
                    f"{_safe_name(dataset_id)}__{_safe_name(algorithm)}__{split_name}__"
                    f"{horizon_label}__fixed_top_n_{metric}.png"
                ),
                dpi=160,
            )
            plt.close(fig)

def _run_one_dataset(
    dataset_row: pd.Series,
    output_dir: Path,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    dataset_id = str(dataset_row["dataset_id"])
    train_df = _read_dataset(dataset_row["training_dataset_path"])
    validate_dataset_features(train_df, config)
    base_ratio_index = _load_ratio_index(dataset_row)
    if _claim_horizon_mode_active() and _horizon_ratio_mode() == "horizon_specific_random":
        evaluation_index = _load_horizon_ratio_index(dataset_row)
    else:
        evaluation_index = base_ratio_index

    validation_rows = base_ratio_index[base_ratio_index["split"].eq("validation")]
    if validation_rows.empty:
        raise ValueError("Holdout ratio index has no validation datasets.")
    reference = validation_rows.sort_values(
        "negative_to_positive_ratio_requested", ascending=False
    ).iloc[0]
    fit_eval_df = _read_dataset(reference["dataset_path"])
    validate_dataset_features(fit_eval_df, config)

    feature_cols = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    X_fit_eval = fit_eval_df[feature_cols]
    y_fit_eval = fit_eval_df["target"].astype(int)

    metric_rows: list[dict] = []
    topk_frames: list[pd.DataFrame] = []
    topn_frames: list[pd.DataFrame] = []
    model_summaries: list[dict] = []
    for algorithm in config.MODELS_TO_RUN:
        print(
            f"  ratio sensitivity dataset={dataset_id} algorithm={algorithm}",
            flush=True,
        )
        model = make_model_pipeline(algorithm, config)
        if model is None:
            metric_rows.append(
                {
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "status": "skipped_missing_dependency",
                }
            )
            continue
        fit_metadata = fit_model_pipeline(
            model,
            algorithm,
            X_train,
            y_train,
            config,
            X_eval=X_fit_eval,
            y_eval=y_fit_eval,
            eval_name=(
                f"validation_ratio_{int(reference['negative_to_positive_ratio_requested'])}_to_1"
            ),
        )

        for _, ratio_row in evaluation_index.iterrows():
            split_name = str(ratio_row["split"])
            ratio = int(ratio_row["negative_to_positive_ratio_requested"])
            holdout_mode = str(
                ratio_row.get(
                    "holdout_negative_sampling_mode",
                    dataset_row.get("holdout_negative_sampling_mode", "random"),
                )
            )
            eval_df = _read_dataset(ratio_row["dataset_path"])
            validate_dataset_features(eval_df, config)
            X_eval = eval_df[feature_cols]
            # Scores depend only on machine features. Horizon-specific ratio files
            # already contain one fixed cohort for one horizon, while the legacy
            # mode reuses the same scored rows across the full horizon sweep.
            score = predict_score(model, X_eval, algorithm)
            row_horizon = pd.to_numeric(
                pd.Series([ratio_row.get("evaluation_horizon_days")]),
                errors="coerce",
            ).iloc[0]
            requested_horizons = (
                [int(row_horizon)]
                if pd.notna(row_horizon)
                else _evaluation_horizon_sweep()
            )

            for requested_horizon in requested_horizons:
                y_eval, target_col, target_mode, horizon_days = get_evaluation_target(
                    eval_df, config, horizon_days=requested_horizon
                )
                free = threshold_free_metrics(y_eval, score)
                thresh = metrics_at_threshold(y_eval, score, threshold=_threshold())
                positive_rows = int(y_eval.sum())
                negative_rows = int(len(y_eval) - positive_rows)

                topk = top_k_metrics(y_eval, score, _top_k_rates())
                topn = top_n_metrics(y_eval, score, _top_n_counts())
                bootstrap_free: dict = {}
                if _bootstrap_enabled():
                    seed = _bootstrap_seed(
                        dataset_id,
                        algorithm,
                        split_name,
                        ratio,
                        target_col,
                        horizon_days,
                    )
                    bootstrap_free, bootstrap_topk, bootstrap_topn = (
                        bootstrap_ranking_metric_intervals(
                            y_true=y_eval,
                            score=score,
                            top_k_rates=_top_k_rates(),
                            top_n_counts=_top_n_counts(),
                            n_resamples=_bootstrap_n_resamples(),
                            confidence_level=_bootstrap_confidence_level(),
                            random_state=seed,
                        )
                    )
                    topk = topk.merge(
                        bootstrap_topk, on="top_k_rate", how="left", validate="one_to_one"
                    )
                    topn = topn.merge(
                        bootstrap_topn, on="top_n_requested", how="left", validate="one_to_one"
                    )

                metric_row = {
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "status": "used",
                    "split": split_name,
                    "holdout_negative_sampling_mode": holdout_mode,
                    "holdout_horizon_ratio_mode": _horizon_ratio_mode(),
                    "negative_to_positive_ratio_requested": ratio,
                    "design_positive_rows": int(
                        pd.to_numeric(eval_df["target"], errors="coerce").fillna(0).eq(1).sum()
                    ),
                    "design_negative_rows": int(
                        pd.to_numeric(eval_df["target"], errors="coerce").fillna(0).eq(0).sum()
                    ),
                    "actual_negative_to_positive_ratio": (
                        float(negative_rows / positive_rows) if positive_rows else np.nan
                    ),
                    "training_rows": int(len(train_df)),
                    "training_positive_rows": int(y_train.sum()),
                    "evaluation_rows": int(len(eval_df)),
                    "evaluation_machines": int(eval_df["machine_key"].nunique(dropna=True)),
                    "evaluation_positive_rows": positive_rows,
                    "evaluation_negative_rows": negative_rows,
                    "evaluation_positive_rate": float(y_eval.mean()) if len(y_eval) else np.nan,
                    "evaluation_target_col": target_col,
                    "evaluation_target_mode": target_mode,
                    "evaluation_horizon_days": horizon_days,
                    "evaluation_path": str(ratio_row["dataset_path"]),
                    "ranking_unit": "machine",
                }
                if "as_of_anchor_date" in eval_df.columns:
                    anchors = pd.to_datetime(eval_df["as_of_anchor_date"], errors="coerce").dropna().unique()
                    metric_row["as_of_anchor_date"] = (
                        pd.Timestamp(anchors[0]).strftime("%Y-%m-%d") if len(anchors) == 1 else None
                    )
                for key, value in fit_metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metric_row[f"fit_{key}"] = value
                metric_row.update({f"threshold_free_{k}": v for k, v in free.items()})
                metric_row.update(
                    {f"threshold_free_{k}": v for k, v in bootstrap_free.items()}
                )
                metric_row.update(
                    {
                        f"threshold_{str(_threshold()).replace('.', 'p')}_{k}": v
                        for k, v in thresh.items()
                    }
                )
                metric_rows.append(metric_row)

                for frame in [topk, topn]:
                    frame.insert(0, "ranking_unit", "machine")
                    frame.insert(0, "evaluation_target_col", target_col)
                    frame.insert(0, "evaluation_target_mode", target_mode)
                    frame.insert(0, "evaluation_horizon_days", horizon_days)
                    frame.insert(0, "negative_to_positive_ratio_requested", ratio)
                    frame.insert(0, "holdout_negative_sampling_mode", holdout_mode)
                    frame.insert(0, "split", split_name)
                    frame.insert(0, "algorithm", algorithm)
                    frame.insert(0, "dataset_id", dataset_id)
                topk_frames.append(topk)
                topn_frames.append(topn)

                pred = _prediction_frame(
                    eval_df,
                    score,
                    y_eval,
                    dataset_id,
                    algorithm,
                    split_name,
                    ratio,
                    target_col,
                    target_mode,
                    horizon_days,
                )
                pred.insert(4, "holdout_negative_sampling_mode", holdout_mode)
                horizon_slug = _horizon_slug(horizon_days)
                pred.to_csv(
                    output_dir
                    / (
                        f"{_safe_name(dataset_id)}__{_safe_name(algorithm)}__{split_name}__"
                        f"ratio_{ratio}_to_1__{horizon_slug}__machine_predictions.csv"
                    ),
                    index=False,
                )

        model_summaries.append(
            {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "training_rows": int(len(train_df)),
                "training_positive_rows": int(y_train.sum()),
                "reference_validation_ratio": int(
                    reference["negative_to_positive_ratio_requested"]
                ),
                "ratios_scored": sorted(
                    evaluation_index["negative_to_positive_ratio_requested"].unique().tolist()
                ),
                "splits_scored": sorted(evaluation_index["split"].unique().tolist()),
                "horizon_ratio_mode": _horizon_ratio_mode(),
                "bootstrap_enabled": _bootstrap_enabled(),
                "bootstrap_n_resamples": (
                    _bootstrap_n_resamples() if _bootstrap_enabled() else 0
                ),
                "bootstrap_confidence_level": (
                    _bootstrap_confidence_level() if _bootstrap_enabled() else None
                ),
                "top_k_rates": _top_k_rates(),
                "top_n_counts": _top_n_counts(),
            }
        )
    return metric_rows, topk_frames, topn_frames, model_summaries

def run(
    dataset_index_path: str | Path | None = None,
    step_dir: str | Path | None = None,
) -> None:
    config.refresh_derived_config()
    output_dir = (
        Path(step_dir)
        if step_dir is not None
        else config.OUTPUT_DIR / "08_holdout_ratio_sensitivity"
    )
    ensure_dir(output_dir)
    dataset_index = _load_dataset_index(dataset_index_path)

    all_metrics: list[dict] = []
    all_topk: list[pd.DataFrame] = []
    all_topn: list[pd.DataFrame] = []
    all_model_summaries: list[dict] = []
    for _, dataset_row in dataset_index.iterrows():
        metrics, topk, topn, summaries = _run_one_dataset(dataset_row, output_dir)
        all_metrics.extend(metrics)
        all_topk.extend(topk)
        all_topn.extend(topn)
        all_model_summaries.extend(summaries)

    metrics_df = pd.DataFrame(all_metrics)
    topk_df = pd.concat(all_topk, ignore_index=True) if all_topk else pd.DataFrame()
    topn_df = pd.concat(all_topn, ignore_index=True) if all_topn else pd.DataFrame()
    metrics_path = output_dir / "holdout_ratio_metrics_all_datasets.csv"
    topk_path = output_dir / "holdout_ratio_top_k_all_datasets.csv"
    topn_path = output_dir / "holdout_ratio_fixed_top_n_all_datasets.csv"
    metrics_df.to_csv(metrics_path, index=False)
    if not topk_df.empty:
        topk_df.to_csv(topk_path, index=False)
    if not topn_df.empty:
        topn_df.to_csv(topn_path, index=False)

    if "status" in metrics_df.columns:
        used_metrics = metrics_df[metrics_df["status"].astype(str).eq("used")].copy()
    else:
        used_metrics = metrics_df.copy()
    review = _review_summary(used_metrics, topk_df, topn_df)
    review_path = output_dir / "holdout_ratio_sensitivity_for_review.csv"
    review.to_csv(review_path, index=False)

    for split_name in ["validation", "test"]:
        split_review = review[review["split"].eq(split_name)].copy()
        split_review.to_csv(
            output_dir / f"{split_name}_negative_ratio_sensitivity_for_review.csv",
            index=False,
        )
        if not topk_df.empty:
            topk_df[topk_df["split"].eq(split_name)].to_csv(
                output_dir / f"{split_name}_percentage_top_k_with_confidence_intervals.csv",
                index=False,
            )
        if not topn_df.empty:
            topn_df[topn_df["split"].eq(split_name)].to_csv(
                output_dir / f"{split_name}_fixed_top_n_with_confidence_intervals.csv",
                index=False,
            )
        _save_metric_plot(review, split_name, output_dir)
        _save_fixed_top_n_plots(topn_df, split_name, output_dir)

    write_json(
        {
            "step": "08_holdout_ratio_sensitivity",
            "output_dir": str(output_dir),
            "dataset_index_path": (
                str(dataset_index_path)
                if dataset_index_path is not None
                else str(config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv")
            ),
            "models_to_run": list(config.MODELS_TO_RUN),
            "top_k_rates": _top_k_rates(),
            "fixed_top_n_counts": _top_n_counts(),
            "bootstrap_enabled": _bootstrap_enabled(),
            "bootstrap_method": (
                "stratified_machine_level_percentile" if _bootstrap_enabled() else None
            ),
            "bootstrap_n_resamples": (
                _bootstrap_n_resamples() if _bootstrap_enabled() else 0
            ),
            "bootstrap_confidence_level": (
                _bootstrap_confidence_level() if _bootstrap_enabled() else None
            ),
            "bootstrap_random_state": int(
                getattr(config, "HOLDOUT_BOOTSTRAP_RANDOM_STATE", 20260727)
            ),
            "prediction_threshold": _threshold(),
            "holdout_horizon_ratio_mode": _horizon_ratio_mode(),
            "metrics_path": str(metrics_path),
            "top_k_path": str(topk_path),
            "fixed_top_n_path": str(topn_path),
            "review_summary_path": str(review_path),
            "model_summaries": all_model_summaries,
            "notes": [
                "The model is fitted once per dataset and algorithm.",
                "All ratio cohorts use the same positive machines within a window design.",
                "Higher ratios add nested negative machines; they do not resample positives.",
                "Each holdout row represents one unique physical machine.",
                "Bootstrap resampling is stratified by class and therefore preserves each evaluated prevalence ratio.",
                "Percentage Top-K and fixed-count Top-N confidence intervals are machine-level percentile intervals.",
                "Fixed Top 10/20/50 comparisons keep maintenance workload constant across prevalence ratios.",
                "With horizon_specific_random mode, each horizon has an exact nested ratio cohort with fixed random machine identities.",
                "With reuse_same_rows mode, the same rows are relabeled and the realized ratio may change by horizon.",
                "Machine prediction files include the actual next claim date and days from window_end when a later claim exists.",
                "Test-ratio results are sensitivity analysis and should not be repeatedly used for tuning.",
            ],
        },
        output_dir / "run_summary.json",
    )
    print(f"08_holdout_ratio_sensitivity completed. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    run()
