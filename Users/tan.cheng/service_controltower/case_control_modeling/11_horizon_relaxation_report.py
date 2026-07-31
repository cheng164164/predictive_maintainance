"""Step 11: summarize fixed-cohort horizon-relaxation performance.

Use this after Step 08 with:

    EVALUATION_TARGET_MODE = "claim_within_horizon"
    HOLDOUT_HORIZON_RATIO_MODE = "reuse_same_rows"

The same validation/test machines, feature windows, and model scores are reused
for every horizon. Only the future-claim label changes. This script verifies that
invariant and writes stakeholder-friendly trend tables showing how many apparent
short-horizon false positives become true positives at longer horizons.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from cc_utils import ensure_dir, write_json


PREDICTION_PATTERN = re.compile(
    r"^(?P<prefix>.+)__(?P<split>validation|test)__ratio_(?P<ratio>\d+)_to_1__"
    r"horizon_(?P<horizon>\d+)d__machine_predictions\.csv$"
)


def _step08_dir(step08_dir: str | Path | None = None) -> Path:
    return (
        Path(step08_dir)
        if step08_dir is not None
        else config.OUTPUT_DIR / "08_holdout_ratio_sensitivity"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required Step 08 output not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _add_baseline_deltas(
    frame: pd.DataFrame,
    metric_columns: list[str],
    group_columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["evaluation_horizon_days"] = pd.to_numeric(
        out["evaluation_horizon_days"], errors="raise"
    ).astype(int)
    out = out.sort_values(group_columns + ["evaluation_horizon_days"], kind="mergesort")
    for metric in metric_columns:
        if metric not in out.columns:
            continue
        values = pd.to_numeric(out[metric], errors="coerce")
        out[metric] = values
        baseline = out.groupby(group_columns, dropna=False)[metric].transform("first")
        out[f"{metric}_change_vs_shortest_horizon"] = values - baseline
    return out.reset_index(drop=True)


def _prediction_files(step08_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    found: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(step08_dir.glob("*__machine_predictions.csv")):
        match = PREDICTION_PATTERN.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        found.append(
            (
                path,
                {
                    "prediction_prefix": meta["prefix"],
                    "split": meta["split"],
                    "negative_to_positive_ratio_requested": int(meta["ratio"]),
                    "evaluation_horizon_days": int(meta["horizon"]),
                },
            )
        )
    if not found:
        raise FileNotFoundError(
            f"No horizon-specific machine prediction files were found in {step08_dir}. "
            "Run Step 08 with claim-within-horizon evaluation first."
        )
    return found


def _cohort_verification_and_claim_timing(
    step08_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = _prediction_files(step08_dir)
    groups: dict[tuple[str, str, int], list[tuple[int, Path]]] = {}
    for path, meta in records:
        key = (
            str(meta["prediction_prefix"]),
            str(meta["split"]),
            int(meta["negative_to_positive_ratio_requested"]),
        )
        groups.setdefault(key, []).append((int(meta["evaluation_horizon_days"]), path))

    verification_rows: list[dict[str, object]] = []
    timing_frames: list[pd.DataFrame] = []
    configured_horizons = sorted({int(x) for x in config.EVALUATION_CLAIM_HORIZON_DAYS})

    for (prefix, split_name, ratio), horizon_files in sorted(groups.items()):
        horizon_files = sorted(horizon_files)
        base_horizon, base_path = horizon_files[0]
        base = _read_csv(base_path)
        required = {"machine_key", "score", "evaluation_target"}
        missing = sorted(required - set(base.columns))
        if missing:
            raise ValueError(f"Prediction file {base_path} is missing columns: {missing}")
        base = base.sort_values("machine_key", kind="mergesort").reset_index(drop=True)
        base_keys = base["machine_key"].astype(str)
        max_score_difference = 0.0
        all_same_machines = True
        all_same_windows = True
        horizon_positive_counts: dict[int, int] = {}

        for horizon, path in horizon_files:
            current = _read_csv(path).sort_values("machine_key", kind="mergesort").reset_index(drop=True)
            current_keys = current["machine_key"].astype(str)
            same_keys = len(current) == len(base) and current_keys.equals(base_keys)
            all_same_machines = all_same_machines and same_keys
            if same_keys:
                score_diff = np.nanmax(
                    np.abs(
                        pd.to_numeric(current["score"], errors="coerce").to_numpy(dtype=float)
                        - pd.to_numeric(base["score"], errors="coerce").to_numpy(dtype=float)
                    )
                )
                max_score_difference = max(max_score_difference, float(score_diff))
                for col in ["window_start", "window_end"]:
                    if col in base.columns and col in current.columns:
                        all_same_windows = all_same_windows and base[col].astype(str).equals(
                            current[col].astype(str)
                        )
            horizon_positive_counts[horizon] = int(
                pd.to_numeric(current["evaluation_target"], errors="coerce").fillna(0).sum()
            )

        positive_sequence = [horizon_positive_counts[h] for h, _ in horizon_files]
        nondecreasing = all(
            later >= earlier
            for earlier, later in zip(positive_sequence, positive_sequence[1:])
        )
        verification_rows.append(
            {
                "prediction_prefix": prefix,
                "split": split_name,
                "negative_to_positive_ratio_requested": ratio,
                "shortest_horizon_days": base_horizon,
                "largest_horizon_days": horizon_files[-1][0],
                "horizons_found": ",".join(str(h) for h, _ in horizon_files),
                "configured_horizons": ",".join(str(h) for h in configured_horizons),
                "machine_rows": int(len(base)),
                "same_machine_ids_across_horizons": bool(all_same_machines),
                "same_feature_windows_across_horizons": bool(all_same_windows),
                "maximum_absolute_score_difference": max_score_difference,
                "same_scores_across_horizons": bool(max_score_difference <= 1e-12),
                "positive_counts_nondecreasing": bool(nondecreasing),
                **{f"positive_rows_{h}d": count for h, count in horizon_positive_counts.items()},
            }
        )

        timing_cols = [
            c
            for c in [
                "dataset_id", "algorithm", "split",
                "negative_to_positive_ratio_requested", "machine_key", "full_model",
                "window_start", "window_end", "score", "score_rank",
                "next_claim_date_on_or_after_window_end",
                "days_to_next_claim_on_or_after_window_end",
            ]
            if c in base.columns
        ]
        timing = base[timing_cols].copy()
        lead = pd.to_numeric(
            timing.get("days_to_next_claim_on_or_after_window_end"), errors="coerce"
        )
        timing["has_observed_future_claim"] = lead.notna()
        timing["first_configured_horizon_counting_claim"] = np.nan
        for horizon in configured_horizons:
            mask = (
                timing["first_configured_horizon_counting_claim"].isna()
                & lead.notna()
                & lead.le(horizon)
                & lead.ge(0)
            )
            timing.loc[mask, "first_configured_horizon_counting_claim"] = horizon
            timing[f"claim_within_{horizon}d"] = (
                lead.notna() & lead.ge(0) & lead.le(horizon)
            ).astype(int)
        timing["claim_after_shortest_but_within_largest_horizon"] = (
            lead.gt(configured_horizons[0]) & lead.le(configured_horizons[-1])
        ).astype(int)
        timing["prediction_prefix"] = prefix
        timing_frames.append(timing)

    return pd.DataFrame(verification_rows), pd.concat(timing_frames, ignore_index=True)


def run(
    step08_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    mode = str(getattr(config, "HOLDOUT_HORIZON_RATIO_MODE", "")).strip().lower()
    if mode not in {"reuse_same_rows", "reuse", "same_rows"}:
        raise ValueError(
            "Step 11 requires HOLDOUT_HORIZON_RATIO_MODE='reuse_same_rows' so the "
            "same machines and scores are compared across horizons."
        )
    source_dir = _step08_dir(step08_dir)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else config.OUTPUT_DIR / "11_horizon_relaxation_report"
    )
    ensure_dir(destination)

    metrics = _read_csv(source_dir / "holdout_ratio_metrics_all_datasets.csv")
    topk = _read_csv(source_dir / "holdout_ratio_top_k_all_datasets.csv")
    topn = _read_csv(source_dir / "holdout_ratio_fixed_top_n_all_datasets.csv")
    if {"precision_at_k", "flagged_count"}.issubset(topk.columns):
        topk["true_positive_at_k"] = (
            pd.to_numeric(topk["precision_at_k"], errors="coerce")
            * pd.to_numeric(topk["flagged_count"], errors="coerce")
        ).round().astype("Int64")

    common_groups = [
        "dataset_id", "algorithm", "split", "negative_to_positive_ratio_requested"
    ]
    metrics_trend = _add_baseline_deltas(
        metrics,
        [
            "evaluation_positive_rows", "evaluation_positive_rate",
            "threshold_free_average_precision", "threshold_free_roc_auc",
        ],
        common_groups,
    )
    topk_groups = common_groups + ["top_k_rate"]
    topk_trend = _add_baseline_deltas(
        topk,
        ["true_positive_at_k", "precision_at_k", "recall_at_k", "lift_vs_random"],
        topk_groups,
    )
    topn_groups = common_groups + ["top_n_requested"]
    topn_trend = _add_baseline_deltas(
        topn,
        ["true_positive_at_n", "precision_at_n", "recall_at_n", "lift_vs_random"],
        topn_groups,
    )

    verification, timing = _cohort_verification_and_claim_timing(source_dir)

    metrics_path = destination / "horizon_relaxation_metrics_trend.csv"
    topk_path = destination / "horizon_relaxation_top_k_trend.csv"
    topn_path = destination / "horizon_relaxation_top_n_trend.csv"
    verification_path = destination / "horizon_relaxation_cohort_verification.csv"
    timing_path = destination / "horizon_relaxation_machine_claim_timing.csv"
    metrics_trend.to_csv(metrics_path, index=False)
    topk_trend.to_csv(topk_path, index=False)
    topn_trend.to_csv(topn_path, index=False)
    verification.to_csv(verification_path, index=False)
    timing.to_csv(timing_path, index=False)

    max_ratio = int(max(config.HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS))
    review_metrics = metrics_trend[
        metrics_trend["negative_to_positive_ratio_requested"].eq(max_ratio)
    ].copy()
    review_topn = topn_trend[
        topn_trend["negative_to_positive_ratio_requested"].eq(max_ratio)
    ].copy()
    review_topk = topk_trend[
        topk_trend["negative_to_positive_ratio_requested"].eq(max_ratio)
    ].copy()
    review_metrics.to_csv(destination / f"ratio_{max_ratio}_to_1_metrics_trend.csv", index=False)
    review_topk.to_csv(destination / f"ratio_{max_ratio}_to_1_top_k_trend.csv", index=False)
    review_topn.to_csv(destination / f"ratio_{max_ratio}_to_1_top_n_trend.csv", index=False)

    failed = verification[
        ~(
            verification["same_machine_ids_across_horizons"]
            & verification["same_feature_windows_across_horizons"]
            & verification["same_scores_across_horizons"]
            & verification["positive_counts_nondecreasing"]
        )
    ]
    if not failed.empty:
        raise AssertionError(
            "Fixed-cohort horizon-relaxation verification failed. Review "
            f"{verification_path}."
        )

    write_json(
        {
            "step": "11_horizon_relaxation_report",
            "source_step08_dir": str(source_dir),
            "output_dir": str(destination),
            "evaluation_horizons_days": sorted(
                {int(x) for x in config.EVALUATION_CLAIM_HORIZON_DAYS}
            ),
            "holdout_horizon_ratio_mode": mode,
            "maximum_review_ratio": max_ratio,
            "verification_groups": int(len(verification)),
            "all_fixed_cohort_checks_passed": True,
            "metrics_trend_path": str(metrics_path),
            "top_k_trend_path": str(topk_path),
            "top_n_trend_path": str(topn_path),
            "cohort_verification_path": str(verification_path),
            "machine_claim_timing_path": str(timing_path),
        },
        destination / "run_summary.json",
    )
    print(f"11_horizon_relaxation_report completed. Outputs: {destination}", flush=True)


if __name__ == "__main__":
    run()
