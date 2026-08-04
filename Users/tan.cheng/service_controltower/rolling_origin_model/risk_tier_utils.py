"""Risk-tier policy construction and deployment utilities.

The tier policy is intentionally split into two layers:

1. Fixed Top-N capacity boundaries selected from honest rolling-origin anchor
   validation under explicit precision requirements.
2. Score-strength confirmation gates derived from the historical distribution
   of the accepted Top-N boundary scores. A weak candidate is demoted and
   tested against the next lower tier.

Precision uncertainty is reported with both pooled Wilson intervals and a
cluster bootstrap that resamples complete anchor dates. Score thresholds are
reported with empirical distributions and anchor-cluster bootstrap intervals.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import config

TIER_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
ACTIVE_TIERS = TIER_ORDER[:-1]


def calibrated_probability_to_risk_index(
    calibrated_probability: object,
) -> float | np.ndarray | pd.Series:
    """Convert calibrated event probability to a customer-facing 0-100 score.

    ``risk_index`` is not a percentile rank. It is the calibrated probability
    of the configured target event within ``config.HORIZON_DAYS``, multiplied
    by 100. Values are clipped by a tiny configurable epsilon so the returned
    score is strictly inside the open interval (0, 100).
    """
    is_series = isinstance(calibrated_probability, pd.Series)
    values = np.asarray(calibrated_probability, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Calibrated probabilities must all be finite.")
    tolerance = 1e-12
    if ((values < -tolerance) | (values > 1.0 + tolerance)).any():
        raise ValueError("Calibrated probabilities must be in [0, 1].")
    epsilon = float(config.RISK_INDEX_PROBABILITY_EPSILON)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("RISK_INDEX_PROBABILITY_EPSILON must be in (0, 0.5).")
    result = 100.0 * np.clip(values, epsilon, 1.0 - epsilon)
    if values.ndim == 0:
        return float(result)
    if is_series:
        return pd.Series(
            result,
            index=calibrated_probability.index,
            name="risk_index",
            dtype=float,
        )
    return result


def risk_index_to_calibrated_probability(risk_index: object) -> float | np.ndarray:
    """Convert a 0-100 risk index back to calibrated probability."""
    values = np.asarray(risk_index, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Risk index values must all be finite.")
    if ((values <= 0.0) | (values >= 100.0)).any():
        raise ValueError("Risk index values must be strictly between 0 and 100.")
    result = values / 100.0
    return float(result) if values.ndim == 0 else result


def _stable_seed(*parts: object) -> int:
    """Derive a deterministic bootstrap seed from configuration and input parts."""
    payload = "|".join([str(config.RANDOM_SEED), *[str(p) for p in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**32 - 1)


def _safe_float(value: object) -> float:
    """Convert a value to a finite float or return NaN when invalid."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _quantile(values: Sequence[float], q: float) -> float:
    """Calculate a finite-value quantile with an empty-input safeguard."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.quantile(array, q))


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float | None = None,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return float("nan"), float("nan")
    level = float(confidence_level or config.TIER_CONFIDENCE_LEVEL)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_ci(
    values: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    seed_parts: Sequence[object],
    replicates: int | None = None,
    confidence_level: float | None = None,
) -> tuple[float, float]:
    """Cluster bootstrap over rows where each row represents one anchor."""
    if values.empty:
        return float("nan"), float("nan")
    reps = int(replicates or config.TIER_BOOTSTRAP_REPLICATES)
    level = float(confidence_level or config.TIER_CONFIDENCE_LEVEL)
    rng = np.random.default_rng(_stable_seed(*seed_parts))
    n = len(values)
    estimates = np.empty(reps, dtype=float)
    for index in range(reps):
        positions = rng.integers(0, n, size=n)
        estimates[index] = statistic(values.iloc[positions])
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def _bootstrap_precision_ci(group: pd.DataFrame, seed_parts: Sequence[object]) -> tuple[float, float]:
    """Vectorized anchor-cluster bootstrap for pooled precision."""
    if group.empty:
        return float("nan"), float("nan")
    reps = int(config.TIER_BOOTSTRAP_REPLICATES)
    level = float(config.TIER_CONFIDENCE_LEVEL)
    selected = pd.to_numeric(group["selected_machines"], errors="coerce").fillna(0).to_numpy(dtype=float)
    true_positive = pd.to_numeric(group["true_positive_machines"], errors="coerce").fillna(0).to_numpy(dtype=float)
    n = len(group)
    rng = np.random.default_rng(_stable_seed(*seed_parts, "precision"))
    positions = rng.integers(0, n, size=(reps, n))
    selected_sum = selected[positions].sum(axis=1)
    tp_sum = true_positive[positions].sum(axis=1)
    estimates = np.divide(
        tp_sum,
        selected_sum,
        out=np.full(reps, np.nan, dtype=float),
        where=selected_sum > 0,
    )
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def _bootstrap_value_ci(
    group: pd.DataFrame,
    column: str,
    function_name: str,
    seed_parts: Sequence[object],
    q: float | None = None,
) -> tuple[float, float]:
    """Vectorized anchor-cluster bootstrap for score-distribution statistics."""
    values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    reps = int(config.TIER_BOOTSTRAP_REPLICATES)
    level = float(config.TIER_CONFIDENCE_LEVEL)
    n = len(values)
    seed_column = (
        "boundary_calibrated_probability"
        if column == "boundary_risk_index"
        else column
    )
    rng = np.random.default_rng(
        _stable_seed(*seed_parts, seed_column, function_name, q)
    )
    positions = rng.integers(0, n, size=(reps, n))
    samples = values[positions]
    if function_name == "mean":
        estimates = samples.mean(axis=1)
    elif function_name == "median":
        estimates = np.median(samples, axis=1)
    elif function_name == "quantile":
        if q is None:
            raise ValueError("q is required for quantile bootstrap.")
        estimates = np.quantile(samples, q, axis=1)
    else:
        raise ValueError(function_name)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1.0 - alpha))


def _distribution_statistics(
    values: Sequence[float],
    prefix: str,
) -> dict[str, float | int]:
    """Calculate count and distribution statistics for one score collection."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q025": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q10": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_q975": np.nan,
        }
    return {
        f"{prefix}_count": int(array.size),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_q025": float(np.quantile(array, 0.025)),
        f"{prefix}_q05": float(np.quantile(array, 0.05)),
        f"{prefix}_q10": float(np.quantile(array, 0.10)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
        f"{prefix}_q90": float(np.quantile(array, 0.90)),
        f"{prefix}_q95": float(np.quantile(array, 0.95)),
        f"{prefix}_q975": float(np.quantile(array, 0.975)),
    }


def configured_top_n_grid(maximum_fleet_size: int | None = None) -> tuple[int, ...]:
    """Return the routine coarse Top-N grid (5, 10, 15, ... by default)."""
    start = int(config.TIER_TOP_N_GRID_START)
    stop = int(config.TIER_TOP_N_GRID_MAX)
    step = int(config.TIER_TOP_N_GRID_STEP)
    values = tuple(range(start, stop + 1, step))
    if maximum_fleet_size is not None:
        values = tuple(value for value in values if value <= maximum_fleet_size)
    if not values:
        raise ValueError("The configured coarse tier Top-N grid is empty for the available fleet.")
    return values


def configured_critical_fine_top_n_grid(
    maximum_fleet_size: int | None = None,
) -> tuple[int, ...]:
    """Return the N=1..4 strict-confidence refinement used only for Critical."""
    if not bool(config.TIER_CRITICAL_FINE_SCAN_ENABLED):
        return tuple()
    start = int(config.TIER_CRITICAL_FINE_SCAN_START)
    stop = min(
        int(config.TIER_CRITICAL_FINE_SCAN_MAX),
        int(config.TIER_TOP_N_GRID_START) - 1,
    )
    step = int(config.TIER_CRITICAL_FINE_SCAN_STEP)
    values = tuple(range(start, stop + 1, step)) if stop >= start else tuple()
    if maximum_fleet_size is not None:
        values = tuple(value for value in values if value <= maximum_fleet_size)
    return values


def configured_tier_evaluation_grid(
    maximum_fleet_size: int | None = None,
) -> tuple[int, ...]:
    """Return coarse boundaries plus the Critical-only fine scan boundaries."""
    values = sorted(
        set(configured_top_n_grid(maximum_fleet_size))
        | set(configured_critical_fine_top_n_grid(maximum_fleet_size))
    )
    if not values:
        raise ValueError("The configured tier evaluation grid is empty.")
    return tuple(values)


def build_top_n_grid_by_anchor(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate coarse Top-N and Critical fine-scan boundaries at every anchor."""
    required = {
        "algorithm",
        "variant",
        "fold",
        "anchor_id",
        "snapshot_date",
        "risk_rank_within_anchor",
        "raw_score",
        "calibrated_probability",
        "true_label_90d",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise KeyError(f"Anchor predictions are missing required tier columns: {missing}")

    normalized = predictions.copy()
    normalized["risk_index"] = calibrated_probability_to_risk_index(
        normalized["calibrated_probability"]
    )
    maximum = int(normalized.groupby(["algorithm", "variant", "anchor_id"]).size().max())
    evaluation_grid = configured_tier_evaluation_grid(maximum)
    fine_grid = set(configured_critical_fine_top_n_grid(maximum))
    rows: list[dict] = []
    group_columns = ["algorithm", "variant", "fold", "anchor_id", "snapshot_date"]
    for keys, anchor in normalized.groupby(group_columns, sort=True, dropna=False):
        anchor = anchor.sort_values("risk_rank_within_anchor", kind="mergesort").reset_index(drop=True)
        labels = anchor["true_label_90d"].astype(int)
        fleet_size = len(anchor)
        total_positives = int(labels.sum())
        for requested_n in evaluation_grid:
            selected_n = min(requested_n, fleet_size)
            selected = anchor.iloc[:selected_n]
            true_positives = int(selected["true_label_90d"].sum())
            precision = true_positives / selected_n if selected_n else np.nan
            recall = true_positives / total_positives if total_positives else np.nan
            precision_low, precision_high = wilson_interval(true_positives, selected_n)
            recall_low, recall_high = wilson_interval(true_positives, total_positives)
            boundary = selected.iloc[-1]
            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "requested_top_n": int(requested_n),
                    "grid_source": (
                        "critical_fine_scan" if requested_n in fine_grid else "coarse_grid"
                    ),
                    "selected_machines": int(selected_n),
                    "eligible_machines": int(fleet_size),
                    "positive_machines": int(total_positives),
                    "true_positive_machines": int(true_positives),
                    "false_positive_machines": int(selected_n - true_positives),
                    "precision": float(precision),
                    "precision_ci95_low": float(precision_low),
                    "precision_ci95_high": float(precision_high),
                    "recall": float(recall),
                    "recall_ci95_low": float(recall_low),
                    "recall_ci95_high": float(recall_high),
                    "fleet_positive_rate": float(total_positives / fleet_size) if fleet_size else np.nan,
                    "lift_vs_fleet": float(precision / (total_positives / fleet_size))
                    if total_positives > 0
                    else np.nan,
                    "boundary_machine_key": boundary.get("machine_key", np.nan),
                    "boundary_raw_score": float(boundary["raw_score"]),
                    "boundary_calibrated_probability": float(boundary["calibrated_probability"]),
                    "boundary_risk_index": float(boundary["risk_index"]),
                }
            )
            for score_column, prefix in (
                ("raw_score", "selected_raw_score"),
                ("calibrated_probability", "selected_calibrated_probability"),
                ("risk_index", "selected_risk_index"),
            ):
                row.update(_distribution_statistics(selected[score_column], prefix))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "variant", "fold", "snapshot_date", "requested_top_n"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_top_n_grid(per_anchor: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Aggregate Top-N anchor results with precision confidence intervals."""
    rows: list[dict] = []
    boundary_columns = (
        "boundary_raw_score",
        "boundary_calibrated_probability",
        "boundary_risk_index",
    )
    for keys, group in per_anchor.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        selected = int(group["selected_machines"].sum())
        true_positives = int(group["true_positive_machines"].sum())
        positives = int(group["positive_machines"].sum())
        eligible = int(group["eligible_machines"].sum())
        precision = true_positives / selected if selected else np.nan
        recall = true_positives / positives if positives else np.nan
        wilson_low, wilson_high = wilson_interval(true_positives, selected)
        bootstrap_low, bootstrap_high = _bootstrap_precision_ci(group, [*keys, "top_n_summary"])
        row.update(
            {
                "anchor_count": int(group["anchor_id"].nunique()),
                "total_selected_machine_snapshots": selected,
                "total_true_positive_machine_snapshots": true_positives,
                "total_positive_machine_snapshots": positives,
                "total_eligible_machine_snapshots": eligible,
                "micro_precision": float(precision),
                "micro_precision_wilson_ci95_low": float(wilson_low),
                "micro_precision_wilson_ci95_high": float(wilson_high),
                "micro_precision_anchor_bootstrap_ci95_low": float(bootstrap_low),
                "micro_precision_anchor_bootstrap_ci95_high": float(bootstrap_high),
                "micro_recall": float(recall),
                "natural_positive_rate": float(positives / eligible) if eligible else np.nan,
                "micro_lift_vs_fleet": float(precision / (positives / eligible))
                if eligible and positives > 0
                else np.nan,
                "mean_anchor_precision": float(group["precision"].mean()),
                "median_anchor_precision": float(group["precision"].median()),
                "minimum_anchor_precision": float(group["precision"].min()),
                "maximum_anchor_precision": float(group["precision"].max()),
                "std_anchor_precision": float(group["precision"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_anchor_recall": float(group["recall"].mean()),
            }
        )
        for column in boundary_columns:
            row.update(_distribution_statistics(group[column], column))
            mean_low, mean_high = _bootstrap_value_ci(group, column, "mean", [*keys, "boundary"])
            median_low, median_high = _bootstrap_value_ci(group, column, "median", [*keys, "boundary"])
            q10_low, q10_high = _bootstrap_value_ci(
                group, column, "quantile", [*keys, "boundary"], q=float(config.TIER_SCORE_GATE_QUANTILE)
            )
            row[f"{column}_mean_anchor_bootstrap_ci95_low"] = mean_low
            row[f"{column}_mean_anchor_bootstrap_ci95_high"] = mean_high
            row[f"{column}_median_anchor_bootstrap_ci95_low"] = median_low
            row[f"{column}_median_anchor_bootstrap_ci95_high"] = median_high
            row[f"{column}_gate_quantile_anchor_bootstrap_ci95_low"] = q10_low
            row[f"{column}_gate_quantile_anchor_bootstrap_ci95_high"] = q10_high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns, kind="mergesort").reset_index(drop=True)


def _tier_targets() -> dict[str, float]:
    """Return validated Critical, High, and Medium precision targets."""
    targets = {str(key).upper(): float(value) for key, value in config.TIER_PRECISION_TARGETS.items()}
    for tier in ACTIVE_TIERS:
        if tier not in targets:
            raise KeyError(f"Missing precision target for tier {tier}.")
    return targets


def build_tier_boundary_candidates(
    per_anchor: pd.DataFrame,
    overall_grid: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all valid nested tier-boundary combinations for policy selection."""
    targets = _tier_targets()
    rows: list[dict] = []
    anchor_pass_rate_min = float(config.TIER_MIN_ANCHOR_PASS_RATE)
    for _, summary in overall_grid.iterrows():
        mask = (
            per_anchor["algorithm"].eq(summary["algorithm"])
            & per_anchor["variant"].eq(summary["variant"])
            & per_anchor["requested_top_n"].eq(summary["requested_top_n"])
        )
        anchor_rows = per_anchor.loc[mask]
        for tier in ACTIVE_TIERS:
            target = targets[tier]
            pass_rate = float(anchor_rows["precision"].ge(target).mean()) if len(anchor_rows) else np.nan
            point_qualified = bool(
                summary["micro_precision"] >= target
                and summary["anchor_count"] >= int(config.TIER_MIN_ANCHOR_COUNT)
                and pass_rate >= anchor_pass_rate_min
            )
            confidence_qualified = bool(
                point_qualified
                and summary["micro_precision_anchor_bootstrap_ci95_low"] >= target
            )
            row = summary.to_dict()
            row.update(
                {
                    "tier": tier,
                    "required_precision": target,
                    "anchor_precision_pass_rate": pass_rate,
                    "minimum_anchor_pass_rate_required": anchor_pass_rate_min,
                    "point_qualified": point_qualified,
                    "confidence_qualified": confidence_qualified,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "variant", "tier", "requested_top_n"], kind="mergesort"
    ).reset_index(drop=True)


def _band_precision_statistics(
    per_anchor_group: pd.DataFrame,
    lower_top_n: int,
    upper_top_n: int,
    tier: str,
) -> dict[str, float | int | bool]:
    """Summarize a non-overlapping rank band across all anchor dates."""
    key_columns = ["fold", "anchor_id", "snapshot_date"]
    value_columns = ["selected_machines", "true_positive_machines"]
    upper = per_anchor_group[
        per_anchor_group["requested_top_n"].eq(int(upper_top_n))
    ][key_columns + value_columns].copy()
    if upper.empty:
        raise ValueError(f"Missing Top-N={upper_top_n} rows for tier-band evaluation.")

    if int(lower_top_n) > 0:
        lower = per_anchor_group[
            per_anchor_group["requested_top_n"].eq(int(lower_top_n))
        ][key_columns + value_columns].copy()
        if lower.empty:
            raise ValueError(f"Missing Top-N={lower_top_n} rows for tier-band evaluation.")
        anchor_rows = upper.merge(
            lower,
            on=key_columns,
            how="inner",
            suffixes=("_upper", "_lower"),
            validate="one_to_one",
        )
        anchor_rows["selected_machines"] = (
            anchor_rows["selected_machines_upper"]
            - anchor_rows["selected_machines_lower"]
        )
        anchor_rows["true_positive_machines"] = (
            anchor_rows["true_positive_machines_upper"]
            - anchor_rows["true_positive_machines_lower"]
        )
    else:
        anchor_rows = upper

    anchor_rows["precision"] = np.divide(
        anchor_rows["true_positive_machines"],
        anchor_rows["selected_machines"],
        out=np.full(len(anchor_rows), np.nan, dtype=float),
        where=anchor_rows["selected_machines"].to_numpy(dtype=float) > 0,
    )
    selected_total = int(anchor_rows["selected_machines"].sum())
    true_positive_total = int(anchor_rows["true_positive_machines"].sum())
    precision = true_positive_total / selected_total if selected_total else np.nan
    wilson_low, wilson_high = wilson_interval(true_positive_total, selected_total)
    bootstrap_low, bootstrap_high = _bootstrap_precision_ci(
        anchor_rows,
        [
            per_anchor_group["algorithm"].iloc[0],
            per_anchor_group["variant"].iloc[0],
            tier,
            lower_top_n,
            upper_top_n,
            "non_overlapping_band",
        ],
    )
    target = _tier_targets()[tier]
    pass_rate = float(anchor_rows["precision"].ge(target).mean())
    fold_rows = (
        anchor_rows.groupby("fold", as_index=False)
        .agg(
            selected_machines=("selected_machines", "sum"),
            true_positive_machines=("true_positive_machines", "sum"),
        )
    )
    fold_rows["precision"] = np.divide(
        fold_rows["true_positive_machines"],
        fold_rows["selected_machines"],
        out=np.full(len(fold_rows), np.nan, dtype=float),
        where=fold_rows["selected_machines"].to_numpy(dtype=float) > 0,
    )
    fold_pass_rate = float(fold_rows["precision"].ge(target).mean())
    point_qualified = bool(
        precision >= target
        and len(anchor_rows) >= int(config.TIER_MIN_ANCHOR_COUNT)
        and pass_rate >= float(config.TIER_MIN_ANCHOR_PASS_RATE)
    )
    confidence_qualified = bool(
        point_qualified and bootstrap_low >= target
    )
    return {
        "band_start_rank": int(lower_top_n) + 1,
        "band_end_rank": int(upper_top_n),
        "band_width": int(upper_top_n) - int(lower_top_n),
        "band_anchor_count": int(anchor_rows["anchor_id"].nunique()),
        "band_selected_machine_snapshots": selected_total,
        "band_true_positive_machine_snapshots": true_positive_total,
        "band_precision": float(precision),
        "band_precision_wilson_ci95_low": float(wilson_low),
        "band_precision_wilson_ci95_high": float(wilson_high),
        "band_precision_anchor_bootstrap_ci95_low": float(bootstrap_low),
        "band_precision_anchor_bootstrap_ci95_high": float(bootstrap_high),
        "band_anchor_precision_pass_rate": pass_rate,
        "band_fold_count": int(fold_rows["fold"].nunique()),
        "band_fold_precision_pass_rate": fold_pass_rate,
        "band_mean_fold_precision": float(fold_rows["precision"].mean()),
        "band_median_fold_precision": float(fold_rows["precision"].median()),
        "band_minimum_fold_precision": float(fold_rows["precision"].min()),
        "band_maximum_fold_precision": float(fold_rows["precision"].max()),
        "band_mean_anchor_precision": float(anchor_rows["precision"].mean()),
        "band_median_anchor_precision": float(anchor_rows["precision"].median()),
        "band_minimum_anchor_precision": float(anchor_rows["precision"].min()),
        "band_maximum_anchor_precision": float(anchor_rows["precision"].max()),
        "band_point_qualified": point_qualified,
        "band_confidence_qualified": confidence_qualified,
    }


def build_tier_boundary_combinations(
    per_anchor: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Audit nested boundary combinations, including an empty Critical option.

    Selection itself is severity-sequential and strictly confidence-qualified,
    but this table preserves the full combination grid for stakeholder review.
    """
    rows: list[dict] = []
    targets = _tier_targets()
    for (algorithm, variant), anchor_group in per_anchor.groupby(
        ["algorithm", "variant"], sort=True
    ):
        candidate_group = candidates[
            candidates["algorithm"].eq(algorithm)
            & candidates["variant"].eq(variant)
        ].copy()
        fleet_max = int(anchor_group["eligible_machines"].max())
        coarse_grid = list(configured_top_n_grid(fleet_max))
        fine_grid = list(configured_critical_fine_top_n_grid(fleet_max))
        critical_grid = [0, *fine_grid, *coarse_grid]
        cumulative_lookup = {
            (str(row.tier), int(row.requested_top_n)): row._asdict()
            for row in candidate_group.itertuples(index=False)
        }

        band_cache: dict[tuple[str, int, int], dict] = {}
        for upper in fine_grid + coarse_grid:
            band_cache[("CRITICAL", 0, upper)] = _band_precision_statistics(
                anchor_group, 0, upper, "CRITICAL"
            )
        for lower in critical_grid:
            for upper in coarse_grid:
                if upper > lower:
                    band_cache[("HIGH", lower, upper)] = _band_precision_statistics(
                        anchor_group, lower, upper, "HIGH"
                    )
        for lower in coarse_grid:
            for upper in coarse_grid:
                if upper > lower:
                    band_cache[("MEDIUM", lower, upper)] = _band_precision_statistics(
                        anchor_group, lower, upper, "MEDIUM"
                    )

        for critical_n in critical_grid:
            for high_n in coarse_grid:
                if high_n <= critical_n:
                    continue
                for medium_n in coarse_grid:
                    if medium_n <= high_n:
                        continue
                    row: dict[str, object] = {
                        "algorithm": algorithm,
                        "variant": variant,
                        "critical_top_n": int(critical_n),
                        "high_top_n": int(high_n),
                        "medium_top_n": int(medium_n),
                        "critical_enabled": bool(critical_n > 0),
                        "critical_grid_source": (
                            "disabled"
                            if critical_n == 0
                            else "critical_fine_scan"
                            if critical_n in fine_grid
                            else "coarse_grid"
                        ),
                        "require_non_overlapping_band_precision": bool(
                            config.TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION
                        ),
                    }
                    all_enabled_point = True
                    all_enabled_confidence = True
                    confidence_count = 0
                    enabled_count = 0
                    lower_n = 0
                    for tier, upper_n in (
                        ("CRITICAL", critical_n),
                        ("HIGH", high_n),
                        ("MEDIUM", medium_n),
                    ):
                        prefix = tier.lower()
                        enabled = not (tier == "CRITICAL" and upper_n == 0)
                        row[f"{prefix}_enabled"] = enabled
                        if not enabled:
                            row[f"{prefix}_required_precision"] = targets[tier]
                            for key in (
                                "cumulative_precision",
                                "cumulative_precision_wilson_ci95_low",
                                "cumulative_precision_wilson_ci95_high",
                                "cumulative_precision_anchor_bootstrap_ci95_low",
                                "cumulative_precision_anchor_bootstrap_ci95_high",
                                "cumulative_anchor_precision_pass_rate",
                                "band_precision",
                                "band_precision_wilson_ci95_low",
                                "band_precision_wilson_ci95_high",
                                "band_precision_anchor_bootstrap_ci95_low",
                                "band_precision_anchor_bootstrap_ci95_high",
                                "band_anchor_precision_pass_rate",
                            ):
                                row[f"{prefix}_{key}"] = np.nan
                            row[f"{prefix}_band_start_rank"] = 0
                            row[f"{prefix}_band_end_rank"] = 0
                            row[f"{prefix}_band_width"] = 0
                            row[f"{prefix}_band_anchor_count"] = 0
                            row[f"{prefix}_point_qualified"] = False
                            row[f"{prefix}_confidence_qualified"] = False
                            lower_n = 0
                            continue

                        cumulative = cumulative_lookup[(tier, int(upper_n))]
                        band = band_cache[(tier, int(lower_n), int(upper_n))]
                        for key, value in (
                            ("required_precision", cumulative["required_precision"]),
                            ("cumulative_precision", cumulative["micro_precision"]),
                            (
                                "cumulative_precision_wilson_ci95_low",
                                cumulative["micro_precision_wilson_ci95_low"],
                            ),
                            (
                                "cumulative_precision_wilson_ci95_high",
                                cumulative["micro_precision_wilson_ci95_high"],
                            ),
                            (
                                "cumulative_precision_anchor_bootstrap_ci95_low",
                                cumulative["micro_precision_anchor_bootstrap_ci95_low"],
                            ),
                            (
                                "cumulative_precision_anchor_bootstrap_ci95_high",
                                cumulative["micro_precision_anchor_bootstrap_ci95_high"],
                            ),
                            (
                                "cumulative_anchor_precision_pass_rate",
                                cumulative["anchor_precision_pass_rate"],
                            ),
                        ):
                            row[f"{prefix}_{key}"] = value
                        for key, value in band.items():
                            row[f"{prefix}_{key}"] = value

                        cumulative_point = bool(cumulative["point_qualified"])
                        cumulative_confidence = bool(cumulative["confidence_qualified"])
                        if config.TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION:
                            tier_point = cumulative_point and bool(band["band_point_qualified"])
                            tier_confidence = cumulative_confidence and bool(
                                band["band_confidence_qualified"]
                            )
                        else:
                            tier_point = cumulative_point
                            tier_confidence = cumulative_confidence
                        row[f"{prefix}_point_qualified"] = tier_point
                        row[f"{prefix}_confidence_qualified"] = tier_confidence
                        all_enabled_point = all_enabled_point and tier_point
                        all_enabled_confidence = all_enabled_confidence and tier_confidence
                        confidence_count += int(tier_confidence)
                        enabled_count += 1
                        lower_n = int(upper_n)

                    row["all_enabled_tiers_point_qualified"] = all_enabled_point
                    row["all_enabled_tiers_confidence_qualified"] = all_enabled_confidence
                    # Backward-compatible names now mean all enabled tiers.
                    row["all_tiers_point_qualified"] = all_enabled_point
                    row["all_tiers_confidence_qualified"] = all_enabled_confidence
                    row["confidence_qualified_tier_count"] = confidence_count
                    row["enabled_tier_count"] = enabled_count
                    rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "variant", "critical_top_n", "high_top_n", "medium_top_n"],
        kind="mergesort",
    ).reset_index(drop=True)


def select_tier_boundaries(
    per_anchor: pd.DataFrame,
    candidates: pd.DataFrame,
    combinations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select strict, confidence-qualified boundaries in severity order.

    Critical is first searched on the coarse grid. If no coarse Critical N has
    a lower 95% anchor-bootstrap precision bound at or above the target, N=1..4
    is searched under the same confidence rule. If no Critical N qualifies, the
    policy selects the N in 1..5 with the highest pooled validation precision.
    That fallback is explicitly marked and does not claim a confidence guarantee.
    High and Medium remain strictly confidence-qualified and are selected from
    the coarse grid using the previous enabled boundary as the start of their
    non-overlapping rank band.
    """
    del combinations  # retained in the signature for audit-pipeline compatibility
    policy = str(config.TIER_SELECTION_POLICY).lower()
    if policy != "strict_confidence_with_critical_precision_fallback":
        raise ValueError(
            "Only strict_confidence_with_critical_precision_fallback is supported."
        )

    targets = _tier_targets()
    rows: list[dict] = []

    def disabled_row(
        algorithm: str,
        variant: str,
        tier: str,
        lower_n: int,
        status: str,
        selection_grid: str,
        fine_scan_triggered: bool,
    ) -> dict:
        """Build a standardized selected-policy row for a disabled tier."""
        return {
            "algorithm": algorithm,
            "variant": variant,
            "tier": tier,
            "tier_order": ACTIVE_TIERS.index(tier) + 1,
            "required_precision": targets[tier],
            "selected_top_n": 0,
            "band_start_rank": int(lower_n) + 1,
            "band_end_rank": 0,
            "band_width": 0,
            "selection_status": status,
            "selection_policy": policy,
            "selection_grid": selection_grid,
            "fine_scan_triggered": bool(fine_scan_triggered),
            "confidence_guarantee_met": False,
            "point_estimate_fallback_used": False,
            "fallback_selection_metric": None,
            "combination_selection_objective": str(
                config.TIER_COMBINATION_SELECTION_OBJECTIVE
            ),
            "cumulative_micro_precision": np.nan,
            "cumulative_precision_wilson_ci95_low": np.nan,
            "cumulative_precision_wilson_ci95_high": np.nan,
            "cumulative_precision_anchor_bootstrap_ci95_low": np.nan,
            "cumulative_precision_anchor_bootstrap_ci95_high": np.nan,
            "anchor_precision_pass_rate": np.nan,
            "band_precision": np.nan,
            "band_precision_wilson_ci95_low": np.nan,
            "band_precision_wilson_ci95_high": np.nan,
            "band_precision_anchor_bootstrap_ci95_low": np.nan,
            "band_precision_anchor_bootstrap_ci95_high": np.nan,
            "band_anchor_precision_pass_rate": np.nan,
            "band_fold_count": 0,
            "band_fold_precision_pass_rate": np.nan,
            "band_mean_fold_precision": np.nan,
            "band_median_fold_precision": np.nan,
            "band_minimum_fold_precision": np.nan,
            "band_maximum_fold_precision": np.nan,
            "anchor_count": 0,
        }

    for (algorithm, variant), anchor_group in per_anchor.groupby(
        ["algorithm", "variant"], sort=True
    ):
        candidate_group = candidates[
            candidates["algorithm"].eq(algorithm)
            & candidates["variant"].eq(variant)
        ].copy()
        fleet_max = int(anchor_group["eligible_machines"].max())
        coarse_grid = list(configured_top_n_grid(fleet_max))
        fine_grid = list(configured_critical_fine_top_n_grid(fleet_max))

        def evaluated_options(tier: str, lower_n: int, grid: Sequence[int]) -> list[dict]:
            """Return candidate boundary rows eligible for sequential tier selection."""
            options: list[dict] = []
            for n in grid:
                if int(n) <= int(lower_n):
                    continue
                cumulative_rows = candidate_group[
                    candidate_group["tier"].eq(tier)
                    & candidate_group["requested_top_n"].eq(int(n))
                ]
                if cumulative_rows.empty:
                    continue
                cumulative = cumulative_rows.iloc[0]
                band = _band_precision_statistics(
                    anchor_group, int(lower_n), int(n), tier
                )
                cumulative_confidence = bool(cumulative["confidence_qualified"])
                band_confidence = bool(band["band_confidence_qualified"])
                qualified = cumulative_confidence and (
                    band_confidence
                    if config.TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION
                    else True
                )
                options.append(
                    {
                        "n": int(n),
                        "cumulative": cumulative,
                        "band": band,
                        "qualified": bool(qualified),
                    }
                )
            return options

        def qualifying_options(tier: str, lower_n: int, grid: Sequence[int]) -> list[dict]:
            """Filter evaluated boundary options to those meeting confidence requirements."""
            return [
                option
                for option in evaluated_options(tier, lower_n, grid)
                if option["qualified"]
            ]

        def enabled_row(
            tier: str,
            option: dict,
            status: str,
            selection_grid: str,
            fine_scan_triggered: bool,
            *,
            confidence_guarantee_met: bool = True,
            point_estimate_fallback_used: bool = False,
            fallback_selection_metric: str | None = None,
        ) -> dict:
            """Build a standardized selected-policy row for an enabled tier."""
            cumulative = option["cumulative"]
            band = option["band"]
            n = int(option["n"])
            return {
                "algorithm": algorithm,
                "variant": variant,
                "tier": tier,
                "tier_order": ACTIVE_TIERS.index(tier) + 1,
                "required_precision": targets[tier],
                "selected_top_n": n,
                "band_start_rank": int(band["band_start_rank"]),
                "band_end_rank": int(band["band_end_rank"]),
                "band_width": int(band["band_width"]),
                "selection_status": status,
                "selection_policy": policy,
                "selection_grid": selection_grid,
                "fine_scan_triggered": bool(fine_scan_triggered),
                "confidence_guarantee_met": bool(confidence_guarantee_met),
                "point_estimate_fallback_used": bool(point_estimate_fallback_used),
                "fallback_selection_metric": fallback_selection_metric,
                "combination_selection_objective": str(
                    config.TIER_COMBINATION_SELECTION_OBJECTIVE
                ),
                "cumulative_micro_precision": float(cumulative["micro_precision"]),
                "cumulative_precision_wilson_ci95_low": float(
                    cumulative["micro_precision_wilson_ci95_low"]
                ),
                "cumulative_precision_wilson_ci95_high": float(
                    cumulative["micro_precision_wilson_ci95_high"]
                ),
                "cumulative_precision_anchor_bootstrap_ci95_low": float(
                    cumulative["micro_precision_anchor_bootstrap_ci95_low"]
                ),
                "cumulative_precision_anchor_bootstrap_ci95_high": float(
                    cumulative["micro_precision_anchor_bootstrap_ci95_high"]
                ),
                "anchor_precision_pass_rate": float(
                    cumulative["anchor_precision_pass_rate"]
                ),
                "band_precision": float(band["band_precision"]),
                "band_precision_wilson_ci95_low": float(
                    band["band_precision_wilson_ci95_low"]
                ),
                "band_precision_wilson_ci95_high": float(
                    band["band_precision_wilson_ci95_high"]
                ),
                "band_precision_anchor_bootstrap_ci95_low": float(
                    band["band_precision_anchor_bootstrap_ci95_low"]
                ),
                "band_precision_anchor_bootstrap_ci95_high": float(
                    band["band_precision_anchor_bootstrap_ci95_high"]
                ),
                "band_anchor_precision_pass_rate": float(
                    band["band_anchor_precision_pass_rate"]
                ),
                "band_fold_count": int(band["band_fold_count"]),
                "band_fold_precision_pass_rate": float(
                    band["band_fold_precision_pass_rate"]
                ),
                "band_mean_fold_precision": float(band["band_mean_fold_precision"]),
                "band_median_fold_precision": float(
                    band["band_median_fold_precision"]
                ),
                "band_minimum_fold_precision": float(
                    band["band_minimum_fold_precision"]
                ),
                "band_maximum_fold_precision": float(
                    band["band_maximum_fold_precision"]
                ),
                "anchor_count": int(band["band_anchor_count"]),
            }

        # Critical: use strict confidence first. If no confidence-qualified N
        # exists, choose the maximum-precision boundary from N=1..5 and mark it
        # as an explicit fallback without a confidence guarantee.
        critical_coarse = qualifying_options("CRITICAL", 0, coarse_grid)
        fine_scan_triggered = not bool(critical_coarse)
        if critical_coarse:
            critical_option = max(critical_coarse, key=lambda item: item["n"])
            critical_row = enabled_row(
                "CRITICAL",
                critical_option,
                "confidence_qualified_coarse_grid",
                "coarse_grid",
                False,
            )
            previous_n = int(critical_option["n"])
        else:
            critical_fine = qualifying_options("CRITICAL", 0, fine_grid)
            if critical_fine:
                critical_option = max(critical_fine, key=lambda item: item["n"])
                critical_row = enabled_row(
                    "CRITICAL",
                    critical_option,
                    "confidence_qualified_fine_scan",
                    "critical_fine_scan",
                    True,
                )
                previous_n = int(critical_option["n"])
            elif bool(config.TIER_CRITICAL_MAX_PRECISION_FALLBACK_ENABLED):
                fallback_grid = list(
                    range(
                        int(config.TIER_CRITICAL_FALLBACK_MIN_N),
                        min(
                            int(config.TIER_CRITICAL_FALLBACK_MAX_N),
                            fleet_max,
                        )
                        + 1,
                    )
                )
                fallback_options = evaluated_options(
                    "CRITICAL", 0, fallback_grid
                )
                if not fallback_options:
                    critical_row = disabled_row(
                        algorithm,
                        variant,
                        "CRITICAL",
                        0,
                        "disabled_no_fallback_boundary_available",
                        "critical_fallback_1_to_5",
                        True,
                    )
                    previous_n = 0
                else:
                    metric = str(
                        config.TIER_CRITICAL_FALLBACK_SELECTION_METRIC
                    )
                    if metric != "micro_precision":
                        raise ValueError(
                            "TIER_CRITICAL_FALLBACK_SELECTION_METRIC must be "
                            "'micro_precision'."
                        )

                    def fallback_key(option: dict) -> tuple[float, float, int]:
                        """Return the deterministic sort key for Critical precision fallback selection."""
                        cumulative = option["cumulative"]
                        return (
                            float(cumulative[metric]),
                            float(
                                cumulative[
                                    "micro_precision_anchor_bootstrap_ci95_low"
                                ]
                            ),
                            -int(option["n"]),
                        )

                    critical_option = max(fallback_options, key=fallback_key)
                    critical_row = enabled_row(
                        "CRITICAL",
                        critical_option,
                        "maximum_precision_fallback_n_1_to_5",
                        "critical_fallback_1_to_5",
                        True,
                        confidence_guarantee_met=False,
                        point_estimate_fallback_used=True,
                        fallback_selection_metric=metric,
                    )
                    previous_n = int(critical_option["n"])
            else:
                critical_row = disabled_row(
                    algorithm,
                    variant,
                    "CRITICAL",
                    0,
                    "disabled_no_confidence_qualified_boundary_after_fine_scan",
                    "critical_fine_scan",
                    True,
                )
                previous_n = 0
        rows.append(critical_row)

        for tier in ("HIGH", "MEDIUM"):
            options = qualifying_options(tier, previous_n, coarse_grid)
            if options:
                option = max(options, key=lambda item: item["n"])
                rows.append(
                    enabled_row(
                        tier,
                        option,
                        "confidence_qualified",
                        "coarse_grid",
                        fine_scan_triggered if tier == "HIGH" else False,
                    )
                )
                previous_n = int(option["n"])
            else:
                rows.append(
                    disabled_row(
                        algorithm,
                        variant,
                        tier,
                        previous_n,
                        "disabled_no_confidence_qualified_boundary",
                        "coarse_grid",
                        False,
                    )
                )

    return pd.DataFrame(rows).sort_values(
        ["algorithm", "variant", "tier_order"], kind="mergesort"
    ).reset_index(drop=True)


def assign_validation_candidate_tiers(
    predictions: pd.DataFrame,
    selected_boundaries: pd.DataFrame,
) -> pd.DataFrame:
    """Assign provisional tiers to validation predictions using selected ranks."""
    parts: list[pd.DataFrame] = []
    for (algorithm, variant), scores in predictions.groupby(["algorithm", "variant"], sort=True):
        policy = selected_boundaries[
            selected_boundaries["algorithm"].eq(algorithm)
            & selected_boundaries["variant"].eq(variant)
        ]
        n_by_tier = policy.set_index("tier")["selected_top_n"].to_dict()
        out = scores.copy()
        rank = out["risk_rank_within_anchor"].astype(int)
        conditions: list[pd.Series] = []
        labels: list[str] = []
        for tier in ACTIVE_TIERS:
            n = int(n_by_tier.get(tier, 0))
            if n > 0:
                conditions.append(rank.le(n))
                labels.append(tier)
        out["validation_candidate_tier"] = (
            np.select(conditions, labels, default="LOW")
            if conditions
            else "LOW"
        )
        out["critical_top_n_boundary"] = int(n_by_tier.get("CRITICAL", 0))
        out["high_top_n_boundary"] = int(n_by_tier.get("HIGH", 0))
        out["medium_top_n_boundary"] = int(n_by_tier.get("MEDIUM", 0))
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def summarize_tier_score_bands(
    assigned: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize raw, calibrated, and risk-index score distributions by tier."""
    rows: list[dict] = []
    score_columns = (
        ("raw_score", "raw_score"),
        ("calibrated_probability", "calibrated_probability"),
        ("risk_index", "risk_index"),
    )
    full_group_columns = [*group_columns, "validation_candidate_tier"]
    for keys, group in assigned.groupby(full_group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(full_group_columns, keys))
        selected = int(len(group))
        positives = int(group["true_label_90d"].sum())
        precision = positives / selected if selected else np.nan
        low, high = wilson_interval(positives, selected)
        row.update(
            {
                "tier_machine_snapshots": selected,
                "tier_positive_machine_snapshots": positives,
                "tier_precision": float(precision),
                "tier_precision_wilson_ci95_low": float(low),
                "tier_precision_wilson_ci95_high": float(high),
                "anchor_count": int(group["anchor_id"].nunique()),
                "unique_machines": int(group["machine_key"].nunique()) if "machine_key" in group else np.nan,
            }
        )
        for column, prefix in score_columns:
            row.update(_distribution_statistics(group[column], prefix))

        if "fold" not in group_columns:
            anchor_stats = (
                group.groupby("anchor_id", as_index=False)
                .agg(
                    selected_machines=("true_label_90d", "size"),
                    true_positive_machines=("true_label_90d", "sum"),
                    raw_score=("raw_score", "median"),
                    calibrated_probability=("calibrated_probability", "median"),
                    risk_index=("risk_index", "median"),
                )
            )
        else:
            anchor_stats = (
                group.groupby(["fold", "anchor_id"], as_index=False)
                .agg(
                    selected_machines=("true_label_90d", "size"),
                    true_positive_machines=("true_label_90d", "sum"),
                    raw_score=("raw_score", "median"),
                    calibrated_probability=("calibrated_probability", "median"),
                    risk_index=("risk_index", "median"),
                )
            )
        precision_boot_low, precision_boot_high = _bootstrap_precision_ci(
            anchor_stats, [*keys, "tier_band_precision"]
        )
        row["tier_precision_anchor_bootstrap_ci95_low"] = precision_boot_low
        row["tier_precision_anchor_bootstrap_ci95_high"] = precision_boot_high
        for column, _ in score_columns:
            median_low, median_high = _bootstrap_value_ci(
                anchor_stats, column, "median", [*keys, "tier_band_score"]
            )
            row[f"{column}_median_anchor_bootstrap_ci95_low"] = median_low
            row[f"{column}_median_anchor_bootstrap_ci95_high"] = median_high
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["tier_order"] = result["validation_candidate_tier"].map(
        {tier: index for index, tier in enumerate(TIER_ORDER)}
    )
    return result.sort_values([*group_columns, "tier_order"], kind="mergesort").reset_index(drop=True)


def derive_validation_score_gates(
    per_anchor_grid: pd.DataFrame,
    selected_boundaries: pd.DataFrame,
    group_by_fold: bool,
) -> pd.DataFrame:
    """Derive confidence-adjusted score gates from validation boundary histories."""
    rows: list[dict] = []
    gate_q = float(config.TIER_SCORE_GATE_QUANTILE)
    for _, selected in selected_boundaries.iterrows():
        n = int(selected["selected_top_n"])
        if n <= 0:
            continue
        subset = per_anchor_grid[
            per_anchor_grid["algorithm"].eq(selected["algorithm"])
            & per_anchor_grid["variant"].eq(selected["variant"])
            & per_anchor_grid["requested_top_n"].eq(n)
        ].copy()
        grouping: Iterable[tuple[object, pd.DataFrame]]
        if group_by_fold:
            grouping = subset.groupby("fold", sort=True)
        else:
            grouping = [("ALL", subset)]
        for fold, group in grouping:
            row = {
                "algorithm": selected["algorithm"],
                "variant": selected["variant"],
                "fold": fold,
                "tier": selected["tier"],
                "tier_order": selected["tier_order"],
                "selected_top_n": n,
                "required_precision": selected["required_precision"],
                "selection_status": selected["selection_status"],
                "anchor_count": int(group["anchor_id"].nunique()),
                "gate_quantile": gate_q,
            }
            for column in (
                "boundary_raw_score",
                "boundary_calibrated_probability",
                "boundary_risk_index",
            ):
                row.update(_distribution_statistics(group[column], column))
                gate_value = _quantile(group[column], gate_q)
                gate_low, gate_high = _bootstrap_value_ci(
                    group, column, "quantile", [selected["algorithm"], selected["variant"], selected["tier"], fold, "gate"], q=gate_q
                )
                row[f"{column}_gate"] = gate_value
                row[f"{column}_gate_anchor_bootstrap_ci95_low"] = gate_low
                row[f"{column}_gate_anchor_bootstrap_ci95_high"] = gate_high
            selected_total = int(group["selected_machines"].sum())
            tp_total = int(group["true_positive_machines"].sum())
            precision = tp_total / selected_total if selected_total else np.nan
            p_low, p_high = wilson_interval(tp_total, selected_total)
            boot_low, boot_high = _bootstrap_precision_ci(group, [selected["algorithm"], selected["variant"], selected["tier"], fold, "gate_precision"])
            row.update(
                {
                    "cumulative_selected_machine_snapshots": selected_total,
                    "cumulative_true_positive_machine_snapshots": tp_total,
                    "cumulative_precision": precision,
                    "cumulative_precision_wilson_ci95_low": p_low,
                    "cumulative_precision_wilson_ci95_high": p_high,
                    "cumulative_precision_anchor_bootstrap_ci95_low": boot_low,
                    "cumulative_precision_anchor_bootstrap_ci95_high": boot_high,
                }
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "variant",
                "fold",
                "tier",
                "tier_order",
                "selected_top_n",
                "required_precision",
                "selection_status",
                "anchor_count",
                "gate_quantile",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["algorithm", "variant", "tier_order", "fold"], kind="mergesort"
    ).reset_index(drop=True)


def build_validation_tier_policy(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """Build and persist the complete validation-derived tier policy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = predictions.copy()
    predictions["risk_index"] = calibrated_probability_to_risk_index(
        predictions["calibrated_probability"]
    )
    per_anchor_grid = build_top_n_grid_by_anchor(predictions)
    fold_grid = summarize_top_n_grid(
        per_anchor_grid,
        ["algorithm", "variant", "fold", "requested_top_n"],
    )
    overall_grid = summarize_top_n_grid(
        per_anchor_grid,
        ["algorithm", "variant", "requested_top_n"],
    )
    candidates = build_tier_boundary_candidates(per_anchor_grid, overall_grid)
    combinations = build_tier_boundary_combinations(per_anchor_grid, candidates)
    selected = select_tier_boundaries(per_anchor_grid, candidates, combinations)
    assigned = assign_validation_candidate_tiers(predictions, selected)

    tier_by_anchor = summarize_tier_score_bands(
        assigned,
        ["algorithm", "variant", "fold", "anchor_id", "snapshot_date"],
    )
    tier_by_fold = summarize_tier_score_bands(
        assigned,
        ["algorithm", "variant", "fold"],
    )
    tier_overall = summarize_tier_score_bands(
        assigned,
        ["algorithm", "variant"],
    )
    gates_by_fold = derive_validation_score_gates(per_anchor_grid, selected, group_by_fold=True)
    gates_overall = derive_validation_score_gates(per_anchor_grid, selected, group_by_fold=False)

    per_anchor_grid.to_csv(output_dir / "tier_top_n_grid_by_anchor.csv", index=False)
    fold_grid.to_csv(output_dir / "tier_top_n_grid_by_fold.csv", index=False)
    overall_grid.to_csv(output_dir / "tier_top_n_grid_overall.csv", index=False)
    candidates.to_csv(output_dir / "tier_boundary_candidates.csv", index=False)
    combinations.to_csv(
        output_dir / "tier_boundary_combination_candidates.csv", index=False
    )
    selected.to_csv(output_dir / "tier_selected_boundaries.csv", index=False)
    tier_by_anchor.to_csv(output_dir / "tier_score_statistics_by_anchor.csv", index=False)
    tier_by_fold.to_csv(output_dir / "tier_score_statistics_by_fold.csv", index=False)
    tier_overall.to_csv(output_dir / "tier_score_statistics_overall.csv", index=False)
    gates_by_fold.to_csv(output_dir / "tier_validation_score_gates_by_fold.csv", index=False)
    gates_overall.to_csv(output_dir / "tier_validation_score_gates_overall.csv", index=False)
    assigned.to_csv(
        output_dir / "tier_validation_machine_details.csv.gz",
        index=False,
        compression="gzip",
    )

    summary = {
        "precision_targets": _tier_targets(),
        "selection_policy": config.TIER_SELECTION_POLICY,
        "confidence_level": float(config.TIER_CONFIDENCE_LEVEL),
        "risk_index_definition": config.RISK_INDEX_DEFINITION,
        "risk_horizon_days": int(config.HORIZON_DAYS),
        "bootstrap_replicates": int(config.TIER_BOOTSTRAP_REPLICATES),
        "minimum_anchor_count": int(config.TIER_MIN_ANCHOR_COUNT),
        "minimum_anchor_pass_rate": float(config.TIER_MIN_ANCHOR_PASS_RATE),
        "require_non_overlapping_band_precision": bool(
            config.TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION
        ),
        "combination_selection_objective": str(
            config.TIER_COMBINATION_SELECTION_OBJECTIVE
        ),
        "score_gate_quantile": float(config.TIER_SCORE_GATE_QUANTILE),
        "score_gate_confidence_mode": str(
            config.TIER_SCORE_GATE_CONFIDENCE_MODE
        ),
        "coarse_top_n_grid": list(configured_top_n_grid()),
        "critical_fine_top_n_grid": list(configured_critical_fine_top_n_grid()),
        "tier_evaluation_grid": list(configured_tier_evaluation_grid()),
        "boundary_combination_count": int(len(combinations)),
        "point_qualified_boundary_combination_count": int(
            combinations["all_tiers_point_qualified"].sum()
        ),
        "all_tiers_confidence_qualified_combination_count": int(
            combinations["all_tiers_confidence_qualified"].sum()
        ),
        "maximum_confidence_qualified_tier_count": int(
            combinations["confidence_qualified_tier_count"].max()
        ) if not combinations.empty else 0,
        "selected_boundaries": selected.to_dict(orient="records"),
        "notes": [
            "Cumulative Top-N precision and non-overlapping tier-band precision are both evaluated.",
            "Boundaries are selected sequentially by severity. High and Medium must meet both cumulative and non-overlapping-band confidence requirements.",
            "Critical uses the coarse grid first, then N=1..4 under the same confidence rule.",
            "When no Critical boundary is confidence-qualified, Critical falls back to the N in 1..5 with the highest pooled validation precision; this fallback is explicitly marked and does not claim a 95% confidence guarantee.",
            "Anchor-cluster bootstrap intervals resample complete anchor dates.",
            "Score confirmation gates use the configured lower quantile of the accepted historical Top-N boundary score distribution.",
            "A tier may be disabled when no boundary satisfies the configured qualification policy.",
        ],
    }
    (output_dir / "tier_policy_validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def _load_validation_policy_tables(
    validation_policy_dir: Path,
    algorithm: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected tier boundaries and validation score gates from disk."""
    selected = pd.read_csv(validation_policy_dir / "tier_selected_boundaries.csv")
    gates = pd.read_csv(validation_policy_dir / "tier_validation_score_gates_overall.csv")
    selected = selected[selected["algorithm"].eq(algorithm) & selected["variant"].eq(variant)].copy()
    gates = gates[gates["algorithm"].eq(algorithm) & gates["variant"].eq(variant)].copy()
    if selected.empty:
        raise ValueError(f"No validation tier policy found for algorithm={algorithm}, variant={variant}.")
    return selected, gates


def _rank_score_frame(
    dataframe: pd.DataFrame,
    raw_column: str,
    calibrated_column: str,
    group_column: str = "snapshot_date",
) -> pd.DataFrame:
    """Rank each snapshot while defining risk_index from calibrated probability."""
    parts: list[pd.DataFrame] = []
    for _, group in dataframe.groupby(group_column, sort=True, dropna=False):
        sort_columns = [calibrated_column, raw_column]
        ascending = [False, False]
        if "machine_key" in group.columns:
            sort_columns.append("machine_key")
            ascending.append(True)
        out = group.sort_values(
            sort_columns, ascending=ascending, kind="mergesort"
        ).reset_index(drop=True)
        out["risk_rank_within_snapshot"] = np.arange(1, len(out) + 1, dtype=int)
        out["risk_index"] = calibrated_probability_to_risk_index(
            out[calibrated_column]
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def finalize_tier_policy_for_model(
    calibration_scores: pd.DataFrame,
    validation_policy_dir: Path,
    algorithm: str,
    variant: str,
    model_dir: Path,
    output_dir: Path,
    raw_column: str = "raw_model_score",
    calibrated_column: str = "failure_probability",
    label_column: str = "true_label_90d",
) -> dict:
    """Map validation-selected Top-N boundaries to the final fitted model scale."""
    selected, validation_gates = _load_validation_policy_tables(
        Path(validation_policy_dir), algorithm, variant
    )
    ranked = _rank_score_frame(
        calibration_scores,
        raw_column=raw_column,
        calibrated_column=calibrated_column,
    )
    boundary_rows: list[dict] = []
    performance_rows: list[dict] = []
    for _, policy_row in selected.iterrows():
        tier = str(policy_row["tier"])
        n = int(policy_row["selected_top_n"])
        if n <= 0:
            continue
        for snapshot_date, snapshot in ranked.groupby("snapshot_date", sort=True):
            selected_rows = snapshot.head(min(n, len(snapshot)))
            boundary = selected_rows.iloc[-1]
            true_positives = int(selected_rows[label_column].sum()) if label_column in selected_rows else 0
            selected_count = len(selected_rows)
            boundary_rows.append(
                {
                    "variant": variant,
                    "tier": tier,
                    "snapshot_date": snapshot_date,
                    "selected_top_n": n,
                    "fleet_size": len(snapshot),
                    "boundary_raw_score": float(boundary[raw_column]),
                    "boundary_calibrated_probability": float(boundary[calibrated_column]),
                    "boundary_risk_index": float(boundary["risk_index"]),
                }
            )
            performance_rows.append(
                {
                    "variant": variant,
                    "tier": tier,
                    "snapshot_date": snapshot_date,
                    "selected_top_n": n,
                    "selected_machines": selected_count,
                    "true_positive_machines": true_positives,
                    "precision": true_positives / selected_count if selected_count else np.nan,
                }
            )

    boundaries = pd.DataFrame(boundary_rows)
    performance = pd.DataFrame(performance_rows)
    mapping_rows: list[dict] = []
    gate_q = float(config.TIER_SCORE_GATE_QUANTILE)
    for _, policy_row in selected.iterrows():
        tier = str(policy_row["tier"])
        n = int(policy_row["selected_top_n"])
        validation_gate_row = validation_gates[validation_gates["tier"].eq(tier)]
        if n <= 0:
            mapping_rows.append(
                {
                    "variant": variant,
                    "tier": tier,
                    "tier_order": int(policy_row["tier_order"]),
                    "selected_top_n": 0,
                    "enabled": False,
                    "required_precision": float(policy_row["required_precision"]),
                    "selection_status": str(policy_row["selection_status"]),
                    "confidence_guarantee_met": bool(
                        policy_row.get("confidence_guarantee_met", False)
                    ),
                    "point_estimate_fallback_used": bool(
                        policy_row.get("point_estimate_fallback_used", False)
                    ),
                    "fallback_selection_metric": policy_row.get(
                        "fallback_selection_metric", None
                    ),
                    "final_raw_score_gate": np.nan,
                    "final_calibrated_probability_gate": np.nan,
                    "final_risk_index_gate": np.nan,
                }
            )
            continue
        subset = boundaries[boundaries["tier"].eq(tier)]
        model_raw_gate = _quantile(subset["boundary_raw_score"], gate_q)
        model_cal_gate = _quantile(subset["boundary_calibrated_probability"], gate_q)
        model_risk_gate = calibrated_probability_to_risk_index(model_cal_gate)
        model_raw_gate_low, model_raw_gate_high = _bootstrap_value_ci(
            subset,
            "boundary_raw_score",
            "quantile",
            [variant, tier, "final_model_boundary_gate"],
            q=gate_q,
        )
        model_cal_gate_low, model_cal_gate_high = _bootstrap_value_ci(
            subset,
            "boundary_calibrated_probability",
            "quantile",
            [variant, tier, "final_model_boundary_gate"],
            q=gate_q,
        )
        model_risk_gate_low = calibrated_probability_to_risk_index(model_cal_gate_low)
        model_risk_gate_high = calibrated_probability_to_risk_index(model_cal_gate_high)

        def validation_gate_value(column: str) -> float:
            """Select the empirical or conservative validation gate component."""
            if validation_gate_row.empty or column not in validation_gate_row:
                return float("nan")
            return _safe_float(validation_gate_row[column].iloc[0])

        validation_raw_gate = validation_gate_value("boundary_raw_score_gate")
        validation_raw_gate_low = validation_gate_value(
            "boundary_raw_score_gate_anchor_bootstrap_ci95_low"
        )
        validation_raw_gate_high = validation_gate_value(
            "boundary_raw_score_gate_anchor_bootstrap_ci95_high"
        )
        validation_cal_gate = validation_gate_value(
            "boundary_calibrated_probability_gate"
        )
        validation_cal_gate_low = validation_gate_value(
            "boundary_calibrated_probability_gate_anchor_bootstrap_ci95_low"
        )
        validation_cal_gate_high = validation_gate_value(
            "boundary_calibrated_probability_gate_anchor_bootstrap_ci95_high"
        )
        validation_risk_gate = (
            calibrated_probability_to_risk_index(validation_cal_gate)
            if np.isfinite(validation_cal_gate)
            else float("nan")
        )
        validation_risk_gate_low = (
            calibrated_probability_to_risk_index(validation_cal_gate_low)
            if np.isfinite(validation_cal_gate_low)
            else float("nan")
        )
        validation_risk_gate_high = (
            calibrated_probability_to_risk_index(validation_cal_gate_high)
            if np.isfinite(validation_cal_gate_high)
            else float("nan")
        )

        gate_confidence_mode = str(config.TIER_SCORE_GATE_CONFIDENCE_MODE)

        def gate_component(point: float, upper: float) -> float:
            """Select the empirical or conservative final-model gate component."""
            if gate_confidence_mode == "conservative_upper_ci" and np.isfinite(upper):
                return float(upper)
            return float(point)

        selected_model_raw_gate = gate_component(model_raw_gate, model_raw_gate_high)
        selected_model_cal_gate = gate_component(model_cal_gate, model_cal_gate_high)
        selected_model_risk_gate = calibrated_probability_to_risk_index(
            selected_model_cal_gate
        )
        selected_validation_cal_gate = gate_component(
            validation_cal_gate, validation_cal_gate_high
        )
        selected_validation_risk_gate = (
            calibrated_probability_to_risk_index(selected_validation_cal_gate)
            if np.isfinite(selected_validation_cal_gate)
            else float("nan")
        )
        final_cal_gate = float(
            np.nanmax([selected_model_cal_gate, selected_validation_cal_gate])
        )
        final_risk_gate = calibrated_probability_to_risk_index(final_cal_gate)
        final_raw_gate = selected_model_raw_gate
        perf = performance[performance["tier"].eq(tier)]
        selected_total = int(perf["selected_machines"].sum())
        tp_total = int(perf["true_positive_machines"].sum())
        precision = tp_total / selected_total if selected_total else np.nan
        p_low, p_high = wilson_interval(tp_total, selected_total)
        mapping_rows.append(
            {
                "variant": variant,
                "tier": tier,
                "tier_order": int(policy_row["tier_order"]),
                "selected_top_n": n,
                "enabled": True,
                "required_precision": float(policy_row["required_precision"]),
                "selection_status": str(policy_row["selection_status"]),
                "confidence_guarantee_met": bool(
                    policy_row.get("confidence_guarantee_met", False)
                ),
                "point_estimate_fallback_used": bool(
                    policy_row.get("point_estimate_fallback_used", False)
                ),
                "fallback_selection_metric": policy_row.get(
                    "fallback_selection_metric", None
                ),
                "validation_band_start_rank": int(policy_row.get("band_start_rank", 0)),
                "validation_band_end_rank": int(policy_row.get("band_end_rank", n)),
                "validation_band_width": int(policy_row.get("band_width", n)),
                "validation_band_precision": _safe_float(
                    policy_row.get("band_precision", np.nan)
                ),
                "validation_band_precision_wilson_ci95_low": _safe_float(
                    policy_row.get("band_precision_wilson_ci95_low", np.nan)
                ),
                "validation_band_precision_wilson_ci95_high": _safe_float(
                    policy_row.get("band_precision_wilson_ci95_high", np.nan)
                ),
                "validation_band_precision_anchor_bootstrap_ci95_low": _safe_float(
                    policy_row.get(
                        "band_precision_anchor_bootstrap_ci95_low", np.nan
                    )
                ),
                "validation_band_precision_anchor_bootstrap_ci95_high": _safe_float(
                    policy_row.get(
                        "band_precision_anchor_bootstrap_ci95_high", np.nan
                    )
                ),
                "validation_band_fold_precision_pass_rate": _safe_float(
                    policy_row.get("band_fold_precision_pass_rate", np.nan)
                ),
                "validation_band_minimum_fold_precision": _safe_float(
                    policy_row.get("band_minimum_fold_precision", np.nan)
                ),
                "validation_band_maximum_fold_precision": _safe_float(
                    policy_row.get("band_maximum_fold_precision", np.nan)
                ),
                "validation_cumulative_precision": _safe_float(policy_row["cumulative_micro_precision"]),
                "validation_precision_anchor_bootstrap_ci95_low": _safe_float(policy_row["cumulative_precision_anchor_bootstrap_ci95_low"]),
                "validation_precision_anchor_bootstrap_ci95_high": _safe_float(policy_row["cumulative_precision_anchor_bootstrap_ci95_high"]),
                "final_calibration_reference_precision": precision,
                "final_calibration_reference_precision_wilson_ci95_low": p_low,
                "final_calibration_reference_precision_wilson_ci95_high": p_high,
                "score_gate_confidence_mode": gate_confidence_mode,
                "final_model_boundary_raw_score_gate": model_raw_gate,
                "final_model_boundary_raw_score_gate_ci95_low": model_raw_gate_low,
                "final_model_boundary_raw_score_gate_ci95_high": model_raw_gate_high,
                "final_model_boundary_calibrated_probability_gate": model_cal_gate,
                "final_model_boundary_calibrated_probability_gate_ci95_low": model_cal_gate_low,
                "final_model_boundary_calibrated_probability_gate_ci95_high": model_cal_gate_high,
                "final_model_boundary_risk_index_gate": model_risk_gate,
                "final_model_boundary_risk_index_gate_ci95_low": model_risk_gate_low,
                "final_model_boundary_risk_index_gate_ci95_high": model_risk_gate_high,
                "validation_boundary_raw_score_gate": validation_raw_gate,
                "validation_boundary_raw_score_gate_ci95_low": validation_raw_gate_low,
                "validation_boundary_raw_score_gate_ci95_high": validation_raw_gate_high,
                "validation_boundary_calibrated_probability_gate": validation_cal_gate,
                "validation_boundary_calibrated_probability_gate_ci95_low": validation_cal_gate_low,
                "validation_boundary_calibrated_probability_gate_ci95_high": validation_cal_gate_high,
                "validation_boundary_risk_index_gate": validation_risk_gate,
                "validation_boundary_risk_index_gate_ci95_low": validation_risk_gate_low,
                "validation_boundary_risk_index_gate_ci95_high": validation_risk_gate_high,
                "selected_model_raw_score_gate_component": selected_model_raw_gate,
                "selected_model_calibrated_probability_gate_component": selected_model_cal_gate,
                "selected_validation_calibrated_probability_gate_component": selected_validation_cal_gate,
                "selected_model_risk_index_gate_component": selected_model_risk_gate,
                "selected_validation_risk_index_gate_component": selected_validation_risk_gate,
                "final_raw_score_gate": final_raw_gate,
                "final_calibrated_probability_gate": final_cal_gate,
                "final_risk_index_gate": final_risk_gate,
                "gate_quantile": gate_q,
            }
        )

    mapping = pd.DataFrame(mapping_rows).sort_values("tier_order", kind="mergesort")
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries.to_csv(output_dir / f"final_model_tier_boundary_scores_{variant}.csv", index=False)
    performance.to_csv(output_dir / f"final_model_tier_reference_performance_{variant}.csv", index=False)
    mapping.to_csv(output_dir / f"final_model_tier_thresholds_{variant}.csv", index=False)
    ranked.to_csv(
        output_dir / f"final_model_calibration_reference_scores_{variant}.csv.gz",
        index=False,
        compression="gzip",
    )

    policy = {
        "policy_version": 3,
        "algorithm": algorithm,
        "variant": variant,
        "candidate_selection": (
            "nested cumulative fixed Top-N boundaries with an explicitly marked "
            "Critical maximum-precision N=1..5 fallback"
        ),
        "confirmation_rule": str(config.TIER_SCORE_CONFIRMATION_RULE),
        "score_gate_quantile": gate_q,
        "score_gate_confidence_mode": str(config.TIER_SCORE_GATE_CONFIDENCE_MODE),
        "confidence_level": float(config.TIER_CONFIDENCE_LEVEL),
        "risk_index_definition": config.RISK_INDEX_DEFINITION,
        "risk_horizon_days": int(config.HORIZON_DAYS),
        "tiers": {
            row["tier"]: {
                key: (None if pd.isna(value) else value)
                for key, value in row.items()
                if key not in {"variant", "tier", "tier_order"}
            }
            for row in mapping.to_dict(orient="records")
        },
        "notes": [
            "Top-N boundaries come from joint cumulative and non-overlapping tier-band rolling-origin validation.",
            "High and Medium remain strictly confidence-qualified; Critical may use the explicitly marked maximum-precision N=1..5 fallback when no confidence-qualified Critical boundary exists.",
            "Raw-score gates are remapped on the final model calibration reference snapshots.",
            "risk_index equals 100 times calibrated target-event probability and is strictly inside (0, 100).",
            "The operational risk-index gate is exactly 100 times the selected calibrated-probability gate.",
            "The configured gate confidence mode determines whether each component uses the empirical quantile or its conservative upper 95% bootstrap bound.",
            "Candidates failing a tier gate are demoted and re-tested against the next lower enabled tier.",
        ],
    }
    policy_path = model_dir / f"tier_policy_{variant}.json"
    policy_path.write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")
    return policy


def _passes_gate(
    calibrated_probability: float,
    risk_index: float,
    raw_score: float,
    tier_policy: Mapping[str, object],
    rule: str,
) -> tuple[bool, bool, bool, bool]:
    """Evaluate one machine against a tier score-confirmation rule."""
    cal_gate = tier_policy.get("final_calibrated_probability_gate")
    risk_gate = tier_policy.get("final_risk_index_gate")
    raw_gate = tier_policy.get("final_raw_score_gate")
    cal_pass = True if cal_gate is None else calibrated_probability >= float(cal_gate)
    risk_pass = True if risk_gate is None else risk_index >= float(risk_gate)
    raw_pass = True if raw_gate is None else raw_score >= float(raw_gate)
    if rule == "both_calibrated_and_risk_index":
        passed = cal_pass and risk_pass
    elif rule == "either_calibrated_or_risk_index":
        passed = cal_pass or risk_pass
    elif rule == "calibrated_probability_only":
        passed = cal_pass
    elif rule == "risk_index_only":
        passed = risk_pass
    elif rule == "all_three_scores":
        passed = cal_pass and risk_pass and raw_pass
    else:
        raise ValueError(f"Unsupported tier confirmation rule: {rule}")
    return passed, cal_pass, risk_pass, raw_pass


def apply_final_tier_policy(
    scores: pd.DataFrame,
    policy: Mapping[str, object],
    raw_column: str = "raw_model_score",
    calibrated_column: str = "failure_probability",
) -> pd.DataFrame:
    """Assign Top-N candidates, then confirm/demote using probability-based gates."""
    out = scores.sort_values(
        [calibrated_column, raw_column, "machine_key"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["risk_rank_within_score_date"] = np.arange(1, len(out) + 1, dtype=int)
    out["risk_index"] = calibrated_probability_to_risk_index(out[calibrated_column])
    out["risk_score_0_5"] = out["risk_index"] / 20.0
    out["risk_index_horizon_days"] = int(policy.get("risk_horizon_days", config.HORIZON_DAYS))
    out["risk_index_definition"] = str(
        policy.get("risk_index_definition", config.RISK_INDEX_DEFINITION)
    )

    tiers = policy["tiers"]
    enabled_n = {
        tier: int((tiers.get(tier) or {}).get("selected_top_n") or 0)
        for tier in ACTIVE_TIERS
    }
    conditions: list[pd.Series] = []
    labels: list[str] = []
    rank = out["risk_rank_within_score_date"]
    for tier in ACTIVE_TIERS:
        n = enabled_n[tier]
        if n > 0:
            conditions.append(rank.le(n))
            labels.append(tier)
    out["candidate_risk_tier"] = (
        np.select(conditions, labels, default="LOW")
        if conditions
        else "LOW"
    )

    rule = str(policy.get("confirmation_rule", "risk_index_only"))
    final_tiers: list[str] = []
    confirmed: list[int] = []
    demotion_steps: list[int] = []
    reasons: list[str] = []
    pass_audit: dict[str, list[int]] = {f"{tier.lower()}_gate_pass": [] for tier in ACTIVE_TIERS}
    cal_audit: dict[str, list[int]] = {f"{tier.lower()}_calibrated_gate_pass": [] for tier in ACTIVE_TIERS}
    risk_audit: dict[str, list[int]] = {f"{tier.lower()}_risk_index_gate_pass": [] for tier in ACTIVE_TIERS}
    raw_audit: dict[str, list[int]] = {f"{tier.lower()}_raw_gate_pass": [] for tier in ACTIVE_TIERS}

    for _, row in out.iterrows():
        candidate = str(row["candidate_risk_tier"])
        row_results: dict[str, tuple[bool, bool, bool, bool]] = {}
        for tier in ACTIVE_TIERS:
            tier_info = tiers.get(tier) or {}
            if not bool(tier_info.get("enabled", False)):
                row_results[tier] = (False, False, False, False)
            else:
                row_results[tier] = _passes_gate(
                    float(row[calibrated_column]),
                    float(row["risk_index"]),
                    float(row[raw_column]),
                    tier_info,
                    rule,
                )
        for tier in ACTIVE_TIERS:
            passed, cal_pass, risk_pass, raw_pass = row_results[tier]
            pass_audit[f"{tier.lower()}_gate_pass"].append(int(passed))
            cal_audit[f"{tier.lower()}_calibrated_gate_pass"].append(int(cal_pass))
            risk_audit[f"{tier.lower()}_risk_index_gate_pass"].append(int(risk_pass))
            raw_audit[f"{tier.lower()}_raw_gate_pass"].append(int(raw_pass))

        if candidate == "LOW":
            final_tiers.append("LOW")
            confirmed.append(1)
            demotion_steps.append(0)
            reasons.append("outside_all_selected_top_n_boundaries")
            continue
        start = ACTIVE_TIERS.index(candidate)
        chosen = "LOW"
        for tier in ACTIVE_TIERS[start:]:
            if enabled_n[tier] <= 0:
                continue
            if row_results[tier][0]:
                chosen = tier
                break
        final_tiers.append(chosen)
        confirmed.append(int(chosen == candidate))
        steps = TIER_ORDER.index(chosen) - TIER_ORDER.index(candidate)
        demotion_steps.append(int(max(0, steps)))
        if chosen == candidate:
            reasons.append(f"{candidate.lower()}_candidate_passed_score_confirmation")
        elif chosen == "LOW":
            reasons.append(f"{candidate.lower()}_candidate_failed_all_lower_score_gates")
        else:
            reasons.append(f"{candidate.lower()}_candidate_demoted_to_{chosen.lower()}_after_score_gate")

    out["risk_tier"] = final_tiers
    out["candidate_tier_confirmed"] = confirmed
    out["tier_demotion_steps"] = demotion_steps
    out["tier_decision_reason"] = reasons
    out["tier_confirmation_rule"] = rule
    for audit in (pass_audit, cal_audit, risk_audit, raw_audit):
        for column, values in audit.items():
            out[column] = np.asarray(values, dtype="int8")
    for tier in ACTIVE_TIERS:
        info = tiers.get(tier) or {}
        prefix = tier.lower()
        out[f"{prefix}_top_n_boundary"] = int(info.get("selected_top_n") or 0)
        out[f"{prefix}_calibrated_probability_gate"] = info.get("final_calibrated_probability_gate")
        out[f"{prefix}_risk_index_gate"] = info.get("final_risk_index_gate")
        out[f"{prefix}_raw_score_gate"] = info.get("final_raw_score_gate")
    return out


