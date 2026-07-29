"""Step 09: compare feature-window designs on common holdout machines.

Step 08 evaluates each configured feature window on its own fixed holdout pool.
When several ``WINDOW_CONFIGS`` are run together, Step 02 shares one deterministic
machine-ranking scope so those pools are already highly aligned. This script goes
one step further: for every split and negative:positive ratio it keeps only the
machine/label pairs present in *all* window designs, then recomputes ranking
metrics and bootstrap confidence intervals. The resulting tables isolate feature-
window effects from residual differences in eligible holdout machines.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from cc_utils import (
    bootstrap_ranking_metric_intervals,
    ensure_dir,
    threshold_free_metrics,
    top_k_metrics,
    top_n_metrics,
    write_json,
)


PREDICTION_SUFFIX = "__machine_predictions.csv"


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
    payload = "|".join([str(base), "common_window_comparison", *map(str, parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _prediction_files(step08_dir: Path) -> list[Path]:
    files = sorted(step08_dir.glob(f"*{PREDICTION_SUFFIX}"))
    if not files:
        raise FileNotFoundError(
            f"No Step 08 machine prediction files found in {step08_dir}. "
            "Run 08_holdout_ratio_sensitivity.py first."
        )
    return files


def _load_prediction_file(path: Path) -> pd.DataFrame:
    required = {
        "dataset_id",
        "algorithm",
        "split",
        "negative_to_positive_ratio_requested",
        "evaluation_target_mode",
        "evaluation_target",
        "machine_key",
        "score",
    }
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"Prediction file is empty: {path}")
    for col in [
        "dataset_id",
        "algorithm",
        "split",
        "negative_to_positive_ratio_requested",
        "evaluation_target_mode",
    ]:
        if df[col].nunique(dropna=False) != 1:
            raise ValueError(f"Prediction file must contain one {col}: {path}")
    df = df.copy()
    df["machine_key"] = df["machine_key"].astype(str)
    if df["machine_key"].duplicated().any():
        raise ValueError(f"Prediction file contains duplicate machines: {path}")
    df["evaluation_target"] = pd.to_numeric(
        df["evaluation_target"], errors="raise"
    ).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
    df["negative_to_positive_ratio_requested"] = pd.to_numeric(
        df["negative_to_positive_ratio_requested"], errors="raise"
    ).astype(int)
    if "evaluation_horizon_days" not in df.columns:
        df["evaluation_horizon_days"] = np.nan
    df["prediction_file"] = str(path)
    return df


def _group_key_columns() -> list[str]:
    return [
        "algorithm",
        "split",
        "negative_to_positive_ratio_requested",
        "evaluation_target_mode",
        "evaluation_horizon_days",
    ]


def _common_labeled_machines(frames: dict[str, pd.DataFrame]) -> tuple[list[str], int, int]:
    machine_sets = [set(df["machine_key"].astype(str)) for df in frames.values()]
    common = set.intersection(*machine_sets) if machine_sets else set()
    consistent: list[str] = []
    conflicts = 0
    for machine_key in sorted(common):
        labels = {
            int(df.loc[df["machine_key"].eq(machine_key), "evaluation_target"].iloc[0])
            for df in frames.values()
        }
        if len(labels) == 1:
            consistent.append(machine_key)
        else:
            conflicts += 1
    return consistent, len(common), conflicts


def _add_group_columns(frame: pd.DataFrame, values: dict) -> pd.DataFrame:
    out = frame.copy()
    for key, value in reversed(list(values.items())):
        out.insert(0, key, value)
    return out


def _review_summary(
    metrics: pd.DataFrame,
    topk: pd.DataFrame,
    topn: pd.DataFrame,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = [
        "dataset_id",
        "window_name",
        "algorithm",
        "split",
        "negative_to_positive_ratio_requested",
    ]
    base = metrics.copy()
    for source, label_col, metrics_to_widen in [
        (
            topk,
            "top_k_rate",
            [
                "precision_at_k",
                "precision_at_k_ci_lower",
                "precision_at_k_ci_upper",
                "recall_at_k",
                "recall_at_k_ci_lower",
                "recall_at_k_ci_upper",
            ],
        ),
        (
            topn,
            "top_n_requested",
            [
                "precision_at_n",
                "precision_at_n_ci_lower",
                "precision_at_n_ci_upper",
                "recall_at_n",
                "recall_at_n_ci_lower",
                "recall_at_n_ci_upper",
            ],
        ),
    ]:
        if source.empty:
            continue
        work = source.copy()
        if label_col == "top_k_rate":
            work["metric_cutoff"] = work[label_col].map(
                lambda x: f"top_{int(round(float(x) * 100))}pct"
            )
        else:
            work["metric_cutoff"] = work[label_col].map(
                lambda x: f"top_{int(x)}_machines"
            )
        for value_col in metrics_to_widen:
            if value_col not in work.columns:
                continue
            wide = work.pivot_table(
                index=keys,
                columns="metric_cutoff",
                values=value_col,
                aggfunc="first",
            )
            wide.columns = [f"{cutoff}_{value_col}" for cutoff in wide.columns]
            base = base.merge(wide.reset_index(), on=keys, how="left")
    return base.sort_values(
        ["split", "negative_to_positive_ratio_requested", "dataset_id"],
        kind="mergesort",
    )


def run(
    step08_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    source_dir = (
        Path(step08_dir)
        if step08_dir is not None
        else config.OUTPUT_DIR / "08_holdout_ratio_sensitivity"
    )
    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.OUTPUT_DIR / "09_window_design_comparison"
    )
    ensure_dir(target_dir)

    loaded = [_load_prediction_file(path) for path in _prediction_files(source_dir)]
    all_predictions = pd.concat(loaded, ignore_index=True, sort=False)
    group_cols = _group_key_columns()
    all_predictions["evaluation_horizon_days"] = pd.to_numeric(
        all_predictions["evaluation_horizon_days"], errors="coerce"
    )

    overlap_rows: list[dict] = []
    metric_rows: list[dict] = []
    topk_frames: list[pd.DataFrame] = []
    topn_frames: list[pd.DataFrame] = []

    # dropna=False keeps training_target groups whose horizon is blank.
    for group_key, group in all_predictions.groupby(group_cols, dropna=False, sort=True):
        dataset_ids = sorted(group["dataset_id"].astype(str).unique())
        if len(dataset_ids) < 2:
            continue
        frames = {
            dataset_id: group[group["dataset_id"].astype(str).eq(dataset_id)].copy()
            for dataset_id in dataset_ids
        }
        common_keys, common_before_label_check, label_conflicts = _common_labeled_machines(frames)
        if not common_keys:
            continue
        first = frames[dataset_ids[0]].set_index("machine_key").loc[common_keys]
        common_positive_rows = int(first["evaluation_target"].eq(1).sum())
        common_negative_rows = int(first["evaluation_target"].eq(0).sum())
        if common_positive_rows < 1 or common_negative_rows < 1:
            continue

        group_values = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        overlap_rows.append(
            {
                **group_values,
                "dataset_count": len(dataset_ids),
                "dataset_ids": "|".join(dataset_ids),
                "union_machine_count": len(
                    set.union(*[set(df["machine_key"].astype(str)) for df in frames.values()])
                ),
                "common_machine_count_before_label_check": common_before_label_check,
                "label_conflict_machine_count": label_conflicts,
                "common_labeled_machine_count": len(common_keys),
                "common_positive_rows": common_positive_rows,
                "common_negative_rows": common_negative_rows,
                "common_positive_rate": common_positive_rows / len(common_keys),
                "minimum_dataset_common_machine_coverage": min(
                    len(common_keys) / len(df) for df in frames.values()
                ),
            }
        )

        for dataset_id, df in frames.items():
            common_df = (
                df[df["machine_key"].isin(common_keys)]
                .sort_values("machine_key", kind="mergesort")
                .reset_index(drop=True)
            )
            y = common_df["evaluation_target"].astype(int)
            score = common_df["score"].to_numpy(dtype=float)
            free = threshold_free_metrics(y, score)
            topk = top_k_metrics(y, score, _top_k_rates())
            topn = top_n_metrics(y, score, _top_n_counts())
            bootstrap_free: dict = {}
            if _bootstrap_enabled():
                seed = _bootstrap_seed(*group_key, dataset_id)
                bootstrap_free, bootstrap_topk, bootstrap_topn = (
                    bootstrap_ranking_metric_intervals(
                        y,
                        score,
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
                    bootstrap_topn,
                    on="top_n_requested",
                    how="left",
                    validate="one_to_one",
                )

            window_name = (
                str(common_df["window_name"].iloc[0])
                if "window_name" in common_df.columns
                else dataset_id
            )
            common = {
                "dataset_id": dataset_id,
                "window_name": window_name,
                **group_values,
                "common_machine_count": int(len(common_df)),
                "common_positive_rows": int(y.sum()),
                "common_negative_rows": int((1 - y).sum()),
                "common_positive_rate": float(y.mean()),
                "ranking_unit": "common_machine",
            }
            metric_rows.append(
                {
                    **common,
                    **{f"threshold_free_{k}": v for k, v in free.items()},
                    **{f"threshold_free_{k}": v for k, v in bootstrap_free.items()},
                }
            )
            topk_frames.append(_add_group_columns(topk, common))
            topn_frames.append(_add_group_columns(topn, common))

    overlap = pd.DataFrame(overlap_rows)
    metrics = pd.DataFrame(metric_rows)
    topk_all = pd.concat(topk_frames, ignore_index=True) if topk_frames else pd.DataFrame()
    topn_all = pd.concat(topn_frames, ignore_index=True) if topn_frames else pd.DataFrame()
    review = _review_summary(metrics, topk_all, topn_all)

    overlap_path = target_dir / "common_machine_overlap_audit.csv"
    metrics_path = target_dir / "common_machine_window_metrics.csv"
    topk_path = target_dir / "common_machine_percentage_top_k.csv"
    topn_path = target_dir / "common_machine_fixed_top_n.csv"
    review_path = target_dir / "common_machine_window_comparison_for_review.csv"
    overlap.to_csv(overlap_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    topk_all.to_csv(topk_path, index=False)
    topn_all.to_csv(topn_path, index=False)
    review.to_csv(review_path, index=False)

    write_json(
        {
            "step": "09_compare_window_designs",
            "source_step08_dir": str(source_dir),
            "output_dir": str(target_dir),
            "prediction_file_count": len(loaded),
            "comparison_group_count": int(len(overlap)),
            "dataset_result_rows": int(len(metrics)),
            "top_k_rates": _top_k_rates(),
            "fixed_top_n_counts": _top_n_counts(),
            "bootstrap_enabled": _bootstrap_enabled(),
            "bootstrap_n_resamples": (
                _bootstrap_n_resamples() if _bootstrap_enabled() else 0
            ),
            "bootstrap_confidence_level": (
                _bootstrap_confidence_level() if _bootstrap_enabled() else None
            ),
            "method": (
                "Within each algorithm/split/ratio/target group, intersect machine IDs "
                "across all window designs, exclude machines whose label differs across "
                "windows, and recompute every ranking metric on the same labeled machines."
            ),
            "overlap_audit_path": str(overlap_path),
            "metrics_path": str(metrics_path),
            "percentage_top_k_path": str(topk_path),
            "fixed_top_n_path": str(topn_path),
            "review_path": str(review_path),
        },
        target_dir / "run_summary.json",
    )
    print(f"09_compare_window_designs completed. Outputs: {target_dir}", flush=True)


if __name__ == "__main__":
    run()
