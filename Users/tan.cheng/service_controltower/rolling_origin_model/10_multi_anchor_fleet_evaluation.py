"""Step 10: deployment-style multi-anchor natural-prevalence fleet evaluation.

For each rolling-origin fold, this script:

1. Fits the model on the fold's historical monthly snapshots.
2. Uses the reserved three-month calibration set for early stopping, Platt
   probability calibration, and the operating threshold.
3. Samples reproducible random calendar anchor days inside the later forward
   validation period.
4. Rebuilds the previous 90 days of features for every machine at each anchor.
5. Ranks the full fleet independently within each anchor and evaluates the
   configured target outcome in [anchor, anchor + 90 days).
6. Exports every machine, sorted by score, together with its true outcome,
   threshold prediction, Top-K/Top-N selection flags, and next-event timing.

This complements row-pooled AUC/AP evaluation with the actual deployment unit:
one complete fleet ranking at one as-of date.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

from snapshot_builder import build_snapshot_dataframe, load_sources


OUTPUT_DIR = config.OUTPUT_DIR / config.MULTI_ANCHOR_FLEET_OUTPUT_SUBDIR
PREDICTION_DIR = OUTPUT_DIR / "predictions_by_anchor"


def _stable_seed(*parts: object) -> int:
    payload = "|".join(
        [str(config.MULTI_ANCHOR_FLEET_RANDOM_STATE), *[str(part) for part in parts]]
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**32 - 1)


def _safe_token(value: object) -> str:
    return (
        str(value)
        .replace("%", "pct")
        .replace(".", "p")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _random_spread_dates(
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    count: int,
    minimum_spacing_days: int,
    seed: int,
) -> list[pd.Timestamp]:
    """Select reproducible random dates spread across a date interval.

    Dates are sampled from equal-width temporal strata and retried until the
    configured minimum spacing is met. If the interval is too short, an evenly
    spaced fallback is used.
    """
    candidates = pd.date_range(start.normalize(), end_exclusive.normalize() - pd.Timedelta(days=1), freq="D")
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} eligible anchor days are available between "
            f"{start.date()} and {(end_exclusive - pd.Timedelta(days=1)).date()}, "
            f"but {count} were requested."
        )
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(candidates), count + 1, dtype=int)
    for _ in range(500):
        chosen: list[pd.Timestamp] = []
        for index in range(count):
            low = int(edges[index])
            high = int(edges[index + 1])
            if high <= low:
                high = min(len(candidates), low + 1)
            position = int(rng.integers(low, high))
            chosen.append(pd.Timestamp(candidates[position]))
        chosen = sorted(chosen)
        if all(
            (right - left).days >= minimum_spacing_days
            for left, right in zip(chosen, chosen[1:])
        ):
            return chosen

    fallback_positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    fallback = [pd.Timestamp(candidates[position]) for position in fallback_positions]
    if any(
        (right - left).days < minimum_spacing_days
        for left, right in zip(fallback, fallback[1:])
    ):
        raise ValueError(
            "The fold is too short for the requested anchor count and minimum spacing. "
            "Reduce MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD or "
            "MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS."
        )
    return fallback


def build_anchor_manifest(sources) -> pd.DataFrame:
    """Create a common anchor manifest for either target source.

    When ``MULTI_ANCHOR_FLEET_FIXED_DATES`` is configured, both target runs use
    exactly the same dates. Otherwise dates are reproducibly sampled within
    each forward-validation fold.
    """
    rows: list[dict] = []
    label_data_through = pd.Timestamp(
        sources.target_events["event_date"].max()
    ).normalize()
    latest_mature_anchor_exclusive = (
        label_data_through
        + pd.Timedelta(days=1)
        - pd.Timedelta(days=config.HORIZON_DAYS)
        + pd.Timedelta(days=1)
    )
    fixed_dates = getattr(config, "MULTI_ANCHOR_FLEET_FIXED_DATES", None)

    for fold_name, validation_start, validation_end in config.VALIDATION_FOLDS:
        fold_start = pd.Timestamp(validation_start)
        requested_end = pd.Timestamp(validation_end)
        mature_end = min(requested_end, latest_mature_anchor_exclusive)

        if fixed_dates:
            configured = fixed_dates.get(fold_name)
            if not configured:
                raise ValueError(
                    f"No fixed anchor dates configured for fold {fold_name}."
                )
            anchors = sorted(pd.Timestamp(value).normalize() for value in configured)
            for anchor_date in anchors:
                if not (fold_start <= anchor_date < requested_end):
                    raise ValueError(
                        f"Fixed anchor {anchor_date.date()} is outside fold {fold_name} "
                        f"[{fold_start.date()}, {requested_end.date()})."
                    )
                complete = anchor_date < mature_end
                if (
                    not complete
                    and not bool(
                        getattr(
                            config,
                            "MULTI_ANCHOR_ALLOW_INCOMPLETE_FIXED_DATES",
                            False,
                        )
                    )
                ):
                    raise ValueError(
                        f"Fixed anchor {anchor_date.date()} does not have a complete "
                        f"{config.HORIZON_DAYS}-day {config.TARGET_SOURCE} outcome "
                        f"window. Target data ends {label_data_through.date()}. "
                        "Update the target data, choose an earlier common anchor, or "
                        "set MULTI_ANCHOR_ALLOW_INCOMPLETE_FIXED_DATES=True."
                    )
                if not complete:
                    print(
                        f"WARNING: {config.TARGET_SOURCE} anchor {anchor_date.date()} "
                        f"is right-censored; data ends {label_data_through.date()}.",
                        flush=True,
                    )
            anchor_mode = "fixed_common_dates"
        else:
            anchors = _random_spread_dates(
                start=fold_start,
                end_exclusive=mature_end,
                count=int(config.MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD),
                minimum_spacing_days=int(
                    config.MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS
                ),
                seed=_stable_seed(fold_name, "forward_validation"),
            )
            anchor_mode = "reproducible_random_dates"

        for index, anchor_date in enumerate(anchors, start=1):
            rows.append(
                {
                    "target_source": config.TARGET_SOURCE,
                    "target_display_name": config.TARGET_DISPLAY_NAME,
                    "anchor_mode": anchor_mode,
                    "fold": fold_name,
                    "evaluation_role": "forward_validation",
                    "anchor_id": f"{fold_name}_A{index:02d}",
                    "anchor_date": anchor_date,
                    "feature_window_start": anchor_date
                    - pd.Timedelta(days=config.LOOKBACK_DAYS),
                    "feature_window_end_exclusive": anchor_date,
                    "outcome_window_start": anchor_date,
                    "outcome_window_end_exclusive": anchor_date
                    + pd.Timedelta(days=config.HORIZON_DAYS),
                    "outcome_window_complete": bool(anchor_date < mature_end),
                    "fold_validation_start": fold_start,
                    "fold_validation_end_exclusive": requested_end,
                    "label_data_through": label_data_through,
                    "random_seed": _stable_seed(fold_name, "forward_validation"),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["fold", "anchor_date"], kind="mergesort"
    )
    if manifest["anchor_date"].duplicated().any():
        duplicates = manifest.loc[
            manifest["anchor_date"].duplicated(keep=False), "anchor_date"
        ].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate anchor dates were generated: {duplicates}")
    return manifest.reset_index(drop=True)


def annotate_next_target_event(scored_base: pd.DataFrame, sources) -> pd.DataFrame:
    """Add the first configured target event on/after each anchor."""
    target_events = sources.target_events.sort_values(
        ["event_date", "machine_key", "event_source"], kind="mergesort"
    )
    parts: list[pd.DataFrame] = []
    for anchor_date, group in scored_base.groupby("snapshot_date", sort=True):
        future = target_events[target_events["event_date"].ge(anchor_date)]
        first = (
            future.drop_duplicates("machine_key", keep="first")
            [["machine_key", "event_date", "event_source"]]
            .rename(
                columns={
                    "event_date": "next_target_event_date",
                    "event_source": "next_target_event_source",
                }
            )
        )
        out = group.merge(first, on="machine_key", how="left", validate="one_to_one")
        out["days_to_next_target_event"] = (
            out["next_target_event_date"] - pd.Timestamp(anchor_date)
        ).dt.days
        out["next_event_within_horizon_check"] = (
            out["days_to_next_target_event"].ge(0)
            & out["days_to_next_target_event"].lt(config.HORIZON_DAYS)
        ).astype("int8")
        parts.append(out)
    annotated = pd.concat(parts, ignore_index=True)
    mismatch = annotated[
        annotated["next_event_within_horizon_check"].ne(
            annotated[config.TARGET_COLUMN].astype(int)
        )
    ]
    if not mismatch.empty:
        raise AssertionError(
            f"Next-event annotation disagrees with the target for {len(mismatch):,} rows."
        )
    return annotated.drop(columns="next_event_within_horizon_check")


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _rank_anchor(anchor: pd.DataFrame) -> pd.DataFrame:
    out = anchor.sort_values(
        ["raw_score", "machine_key"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["risk_rank_within_anchor"] = np.arange(1, len(out) + 1, dtype=int)
    out["eligible_machines_at_anchor"] = int(len(out))
    if len(out) > 1:
        out["risk_index"] = 100.0 * (
            len(out) - out["risk_rank_within_anchor"]
        ) / (len(out) - 1)
    else:
        out["risk_index"] = 100.0
    out["score_top_fraction_within_anchor"] = (
        out["risk_rank_within_anchor"] / len(out)
    )

    for rate in sorted({float(value) for value in config.MULTI_ANCHOR_FLEET_TOP_K_RATES}):
        selected_count = max(1, int(math.ceil(len(out) * rate)))
        column = f"selected_top_{_safe_token(int(round(rate * 100)))}pct"
        out[column] = out["risk_rank_within_anchor"].le(selected_count).astype("int8")
    for count in sorted({int(value) for value in config.MULTI_ANCHOR_FLEET_TOP_N_COUNTS}):
        column = f"selected_top_n_{count}"
        out[column] = out["risk_rank_within_anchor"].le(min(count, len(out))).astype("int8")
    return out


def _selection_row(
    anchor: pd.DataFrame,
    selection_type: str,
    selection_value: float | int,
) -> dict:
    n_rows = len(anchor)
    if selection_type == "top_k_fraction":
        requested_count = max(1, int(math.ceil(n_rows * float(selection_value))))
    elif selection_type == "top_n":
        requested_count = min(int(selection_value), n_rows)
    else:
        raise ValueError(selection_type)
    selected = anchor.head(requested_count)
    total_positives = int(anchor["true_label_90d"].sum())
    true_positives = int(selected["true_label_90d"].sum())
    selected_count = int(len(selected))
    precision = true_positives / selected_count if selected_count else np.nan
    recall = true_positives / total_positives if total_positives else np.nan
    base_rate = total_positives / n_rows if n_rows else np.nan
    precision_low, precision_high = _wilson_interval(true_positives, selected_count)
    recall_low, recall_high = _wilson_interval(true_positives, total_positives)
    positive_days = pd.to_numeric(
        selected.loc[selected["true_label_90d"].eq(1), "days_to_next_target_event"],
        errors="coerce",
    )
    return {
        "selection_type": selection_type,
        "selection_value": selection_value,
        "eligible_machines": n_rows,
        "positive_machines": total_positives,
        "selected_machines": selected_count,
        "true_positive_machines": true_positives,
        "false_positive_machines": selected_count - true_positives,
        "precision": precision,
        "precision_ci95_low": precision_low,
        "precision_ci95_high": precision_high,
        "recall": recall,
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "fleet_positive_rate": base_rate,
        "lift_vs_fleet": precision / base_rate if base_rate and base_rate > 0 else np.nan,
        "mean_calibrated_probability_selected": float(
            selected["calibrated_probability"].mean()
        ),
        "median_calibrated_probability_selected": float(
            selected["calibrated_probability"].median()
        ),
        "minimum_raw_score_selected": float(selected["raw_score"].min()),
        "median_days_to_failure_selected_positives": float(positive_days.median())
        if positive_days.notna().any()
        else np.nan,
    }


def evaluate_anchor(anchor: pd.DataFrame) -> tuple[dict, list[dict]]:
    labels = anchor["true_label_90d"].astype(int).to_numpy()
    raw_scores = anchor["raw_score"].to_numpy(dtype=float)
    probabilities = anchor["calibrated_probability"].to_numpy(dtype=float)
    predictions = anchor["prediction_label_at_calibration_threshold"].astype(int).to_numpy()

    if np.unique(labels).size == 2:
        auc = float(roc_auc_score(labels, raw_scores))
        ap = float(average_precision_score(labels, raw_scores))
        brier = float(brier_score_loss(labels, probabilities))
        loss = float(log_loss(labels, probabilities))
    else:
        auc = ap = brier = loss = np.nan

    metric = {
        "eligible_machines": int(len(anchor)),
        "positive_machines": int(labels.sum()),
        "negative_machines": int(len(labels) - labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": auc,
        "average_precision": ap,
        "brier_score": brier,
        "log_loss": loss,
        "threshold": float(anchor["calibration_threshold"].iloc[0]),
        "threshold_flagged_machines": int(predictions.sum()),
        "threshold_flagged_rate": float(predictions.mean()),
        "threshold_precision": float(precision_score(labels, predictions, zero_division=0)),
        "threshold_recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold_f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold_f2": float(
            fbeta_score(labels, predictions, beta=2, zero_division=0)
        ),
        "mean_calibrated_probability": float(probabilities.mean()),
    }

    selections: list[dict] = []
    for rate in sorted({float(value) for value in config.MULTI_ANCHOR_FLEET_TOP_K_RATES}):
        selections.append(_selection_row(anchor, "top_k_fraction", rate))
    for count in sorted({int(value) for value in config.MULTI_ANCHOR_FLEET_TOP_N_COUNTS}):
        selections.append(_selection_row(anchor, "top_n", count))
    return metric, selections


def aggregate_selection_metrics(per_anchor: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in per_anchor.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        selected = int(group["selected_machines"].sum())
        true_positive = int(group["true_positive_machines"].sum())
        positives = int(group["positive_machines"].sum())
        eligible = int(group["eligible_machines"].sum())
        micro_precision = true_positive / selected if selected else np.nan
        micro_recall = true_positive / positives if positives else np.nan
        base_rate = positives / eligible if eligible else np.nan
        precision_low, precision_high = _wilson_interval(true_positive, selected)
        recall_low, recall_high = _wilson_interval(true_positive, positives)
        row.update(
            {
                "anchor_count": int(group["anchor_id"].nunique()),
                "total_eligible_machine_snapshots": eligible,
                "total_positive_machine_snapshots": positives,
                "total_selected_machine_snapshots": selected,
                "total_true_positive_machine_snapshots": true_positive,
                "micro_precision": micro_precision,
                "micro_precision_ci95_low": precision_low,
                "micro_precision_ci95_high": precision_high,
                "micro_recall": micro_recall,
                "micro_recall_ci95_low": recall_low,
                "micro_recall_ci95_high": recall_high,
                "natural_positive_rate": base_rate,
                "micro_lift_vs_fleet": micro_precision / base_rate
                if base_rate and base_rate > 0
                else np.nan,
                "mean_anchor_precision": float(group["precision"].mean()),
                "median_anchor_precision": float(group["precision"].median()),
                "minimum_anchor_precision": float(group["precision"].min()),
                "maximum_anchor_precision": float(group["precision"].max()),
                "std_anchor_precision": float(group["precision"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "mean_anchor_recall": float(group["recall"].mean()),
                "mean_anchor_lift": float(group["lift_vs_fleet"].mean()),
                "mean_selected_calibrated_probability": float(
                    np.average(
                        group["mean_calibrated_probability_selected"],
                        weights=group["selected_machines"],
                    )
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _report_markdown(
    manifest: pd.DataFrame,
    anchor_metrics: pd.DataFrame,
    fold_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Multi-Anchor Fleet Evaluation",
        "",
        "## Design",
        "",
        f"- Feature lookback: {config.LOOKBACK_DAYS} days before each anchor.",
        f"- Outcome horizon: {config.HORIZON_DAYS} days after each anchor.",
        f"- Target source: {config.TARGET_SOURCE} ({config.TARGET_DISPLAY_NAME}).",
        f"- Anchors per fold: {config.MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD} unless fixed dates are configured.",
        "- Every reconstructed fleet machine is scored at every anchor; no negative sampling is used.",
        "- Models are fitted on older monthly snapshots. The reserved calibration months are used for early stopping, Platt calibration, and threshold selection only.",
        "- Headline ranking results use later forward-validation anchors, not calibration anchors.",
        "",
        "## Anchor dates",
        "",
        manifest[["fold", "anchor_id", "anchor_date", "outcome_window_end_exclusive", "outcome_window_complete"]]
        .to_markdown(index=False),
        "",
        "## Threshold-free metrics by variant and fold",
        "",
    ]
    metric_summary = (
        anchor_metrics.groupby(["variant", "fold"], as_index=False)
        .agg(
            anchor_count=("anchor_id", "nunique"),
            mean_positive_rate=("positive_rate", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            minimum_roc_auc=("roc_auc", "min"),
            mean_average_precision=("average_precision", "mean"),
        )
    )
    lines.append(metric_summary.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Top-K and Top-N results by fold", ""])
    selected_columns = [
        "variant",
        "fold",
        "selection_type",
        "selection_value",
        "anchor_count",
        "micro_precision",
        "micro_recall",
        "micro_lift_vs_fleet",
        "minimum_anchor_precision",
        "maximum_anchor_precision",
    ]
    lines.append(fold_summary[selected_columns].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Overall results across all forward anchors", ""])
    overall_columns = [
        "variant",
        "selection_type",
        "selection_value",
        "anchor_count",
        "micro_precision",
        "micro_precision_ci95_low",
        "micro_precision_ci95_high",
        "micro_recall",
        "micro_lift_vs_fleet",
        "minimum_anchor_precision",
        "maximum_anchor_precision",
    ]
    lines.append(overall_summary[overall_columns].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Consecutive anchors can share machines and overlapping 90-day outcome windows. The aggregate rows therefore summarize deployment opportunities, not independent observations.",
            "- Wilson intervals describe binomial uncertainty for the pooled selected machine-snapshots. They do not fully adjust for repeated machines across anchors.",
            "- The F2 threshold is recall-oriented and is not the recommended inspection-capacity rule. Top-K/Top-N ranking is the operational decision layer evaluated here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not bool(config.MULTI_ANCHOR_FLEET_ENABLED):
        print("Multi-anchor evaluation skipped: MULTI_ANCHOR_FLEET_ENABLED=False")
        return

    for directory in [config.OUTPUT_DIR, config.MODEL_DIR, config.CHART_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Load and process raw sources before the monthly training table to keep
    # peak memory lower on constrained smoke-run environments.
    print("Loading raw sources and selecting random anchors...", flush=True)
    sources = load_sources(include_operation_features=False)
    fleet_size = int(len(sources.machine_index))
    manifest = build_anchor_manifest(sources)
    manifest.to_csv(OUTPUT_DIR / "multi_anchor_anchor_manifest.csv", index=False)
    print(manifest[["fold", "anchor_id", "anchor_date"]].to_string(index=False))

    print(
        f"Building features for {len(manifest)} random anchor days x "
        f"{len(sources.machine_index):,} machines...",
        flush=True,
    )
    anchor_base = build_snapshot_dataframe(
        sources=sources,
        snapshot_dates=manifest["anchor_date"].tolist(),
        include_targets=True,
        verbose=True,
    )
    anchor_base = anchor_base.merge(
        manifest.rename(columns={"anchor_date": "snapshot_date"}),
        on="snapshot_date",
        how="left",
        validate="many_to_one",
    )
    anchor_base = annotate_next_target_event(anchor_base, sources)
    anchor_base.to_csv(
        OUTPUT_DIR / "multi_anchor_feature_dataset.csv.gz",
        index=False,
        compression="gzip",
    )

    # Raw event tables are no longer needed after all random-anchor features and
    # future outcomes have been materialized. Releasing them avoids holding the
    # large fault/operation tables alongside the monthly training table.
    del sources
    gc.collect()
    # Import XGBoost/modeling utilities only after releasing the large raw event
    # tables. This keeps peak memory lower while parsing source CSVs.
    from modeling_utils import (
        choose_f2_threshold,
        feature_list_for_variant,
        fit_algorithm,
        fit_platt_calibration,
        load_snapshot_dataframe,
        make_algorithm,
        rolling_origin_split,
    )

    print("Loading cached monthly snapshot dataframe...", flush=True)
    monthly = load_snapshot_dataframe()

    all_prediction_parts: list[pd.DataFrame] = []
    anchor_metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    training_rows: list[dict] = []

    for fold in config.VALIDATION_FOLDS:
        fold_name, fit, calibration, _ = rolling_origin_split(monthly, fold)
        fold_anchors = anchor_base[anchor_base["fold"].eq(fold_name)].copy()
        for variant in config.MULTI_ANCHOR_FLEET_MODEL_VARIANTS:
            print(f"\nTraining {variant} for {fold_name}...", flush=True)
            features, top_code_detail = feature_list_for_variant(monthly, fit, variant)
            x_fit = fit[features].astype(float)
            y_fit = fit[config.TARGET_COLUMN].astype(int).to_numpy()
            x_cal = calibration[features].astype(float)
            y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()

            model = make_algorithm("xgboost")
            model, best_iteration = fit_algorithm(
                "xgboost", model, x_fit, y_fit, x_cal, y_cal
            )
            raw_cal = model.predict_proba(x_cal)[:, 1]
            platt = fit_platt_calibration(y_cal, raw_cal)
            calibrated_cal = platt.apply(raw_cal)
            threshold, calibration_f2 = choose_f2_threshold(y_cal, calibrated_cal)

            raw_anchor = model.predict_proba(fold_anchors[features].astype(float))[:, 1]
            calibrated_anchor = platt.apply(raw_anchor)
            scored = fold_anchors.copy()
            scored["variant"] = variant
            scored["raw_score"] = raw_anchor
            scored["calibrated_probability"] = calibrated_anchor
            scored["calibration_threshold"] = threshold
            scored["prediction_label_at_calibration_threshold"] = (
                calibrated_anchor >= threshold
            ).astype("int8")
            scored["true_label_90d"] = scored[config.TARGET_COLUMN].astype("int8")
            scored["prediction_correct_at_threshold"] = (
                scored["prediction_label_at_calibration_threshold"]
                == scored["true_label_90d"]
            ).astype("int8")

            ranked_parts: list[pd.DataFrame] = []
            for anchor_id, anchor in scored.groupby("anchor_id", sort=True):
                ranked = _rank_anchor(anchor)
                metric, selections = evaluate_anchor(ranked)
                common = {
                    "variant": variant,
                    "fold": fold_name,
                    "evaluation_role": "forward_validation",
                    "anchor_id": anchor_id,
                    "anchor_date": ranked["snapshot_date"].iloc[0],
                    "outcome_window_complete": bool(
                        ranked["outcome_window_complete"].iloc[0]
                    ),
                }
                anchor_metric_rows.append({**common, **metric})
                selection_rows.extend({**common, **row} for row in selections)
                ranked_parts.append(ranked)

                if config.MULTI_ANCHOR_FLEET_WRITE_PER_ANCHOR_FILES:
                    filename = (
                        f"{variant}__{fold_name}__"
                        f"{pd.Timestamp(ranked['snapshot_date'].iloc[0]).strftime('%Y-%m-%d')}"
                        "__fleet_scores_sorted.csv.gz"
                    )
                    ranked.to_csv(
                        PREDICTION_DIR / filename,
                        index=False,
                        compression="gzip",
                    )

            ranked_fold = pd.concat(ranked_parts, ignore_index=True)
            all_prediction_parts.append(ranked_fold)
            training_rows.append(
                {
                    "variant": variant,
                    "fold": fold_name,
                    "fit_rows": int(len(fit)),
                    "fit_snapshot_dates": int(fit["snapshot_date"].nunique()),
                    "calibration_rows": int(len(calibration)),
                    "calibration_snapshot_dates": int(
                        calibration["snapshot_date"].nunique()
                    ),
                    "random_anchor_count": int(ranked_fold["anchor_id"].nunique()),
                    "random_anchor_rows": int(len(ranked_fold)),
                    "feature_count": int(len(features)),
                    "features": "|".join(features),
                    "top_failure_code_features": "|".join(
                        item["feature"] for item in top_code_detail
                    ),
                    "best_iteration": int(best_iteration),
                    "platt_coefficient": float(platt.coefficient),
                    "platt_intercept": float(platt.intercept),
                    "calibration_threshold": float(threshold),
                    "calibration_f2": float(calibration_f2),
                }
            )
            print(
                f"  anchors={ranked_fold['anchor_id'].nunique()}, "
                f"rows={len(ranked_fold):,}, threshold={threshold:.4f}, "
                f"best_iteration={best_iteration}",
                flush=True,
            )

    predictions = pd.concat(all_prediction_parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["variant", "fold", "snapshot_date", "risk_rank_within_anchor"],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions.to_csv(
        OUTPUT_DIR / "multi_anchor_all_fleet_scores_sorted.csv.gz",
        index=False,
        compression="gzip",
    )

    anchor_metrics = pd.DataFrame(anchor_metric_rows).sort_values(
        ["variant", "fold", "anchor_date"], kind="mergesort"
    )
    selections = pd.DataFrame(selection_rows).sort_values(
        ["variant", "fold", "anchor_date", "selection_type", "selection_value"],
        kind="mergesort",
    )
    training = pd.DataFrame(training_rows).sort_values(
        ["variant", "fold"], kind="mergesort"
    )

    fold_summary = aggregate_selection_metrics(
        selections,
        ["variant", "fold", "selection_type", "selection_value"],
    )
    overall_summary = aggregate_selection_metrics(
        selections,
        ["variant", "selection_type", "selection_value"],
    )

    top_machine_details = predictions[
        predictions["risk_rank_within_anchor"].le(
            max(int(value) for value in config.MULTI_ANCHOR_FLEET_TOP_N_COUNTS)
        )
    ].copy()

    anchor_metrics.to_csv(OUTPUT_DIR / "multi_anchor_metrics_by_anchor.csv", index=False)
    selections.to_csv(
        OUTPUT_DIR / "multi_anchor_top_k_top_n_by_anchor.csv", index=False
    )
    fold_summary.to_csv(
        OUTPUT_DIR / "multi_anchor_top_k_top_n_by_fold.csv", index=False
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "multi_anchor_top_k_top_n_overall.csv", index=False
    )
    training.to_csv(OUTPUT_DIR / "multi_anchor_training_audit.csv", index=False)
    top_machine_details.to_csv(
        OUTPUT_DIR / "multi_anchor_top_20_machine_details.csv.gz",
        index=False,
        compression="gzip",
    )

    report = _report_markdown(manifest, anchor_metrics, fold_summary, overall_summary)
    (OUTPUT_DIR / "MULTI_ANCHOR_VALIDATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    run_summary = {
        "step": "10_multi_anchor_fleet_evaluation",
        "strategy": "fixed/common or reproducible forward-validation anchor days; all fleet machines; natural prevalence; rank within anchor",
        "model_variants": list(config.MULTI_ANCHOR_FLEET_MODEL_VARIANTS),
        "fold_count": len(config.VALIDATION_FOLDS),
        "anchors_per_fold": int(config.MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD),
        "total_anchor_dates": int(manifest["anchor_id"].nunique()),
        "fleet_machines_per_anchor": fleet_size,
        "lookback_days": int(config.LOOKBACK_DAYS),
        "outcome_horizon_days": int(config.HORIZON_DAYS),
        "top_k_rates": [float(value) for value in config.MULTI_ANCHOR_FLEET_TOP_K_RATES],
        "top_n_counts": [int(value) for value in config.MULTI_ANCHOR_FLEET_TOP_N_COUNTS],
        "random_state": int(config.MULTI_ANCHOR_FLEET_RANDOM_STATE),
        "target_source": config.TARGET_SOURCE,
        "target_display_name": config.TARGET_DISPLAY_NAME,
        "fixed_anchor_dates_used": bool(getattr(config, "MULTI_ANCHOR_FLEET_FIXED_DATES", None)),
        "elapsed_seconds": float(time.time() - started),
        "notes": [
            "The calibration set is used for early stopping, Platt calibration, and threshold selection, not headline anchor metrics.",
            "Each anchor ranking includes all reconstructed machines at natural prevalence.",
            "Each per-anchor prediction file is sorted from highest to lowest raw model score.",
            "Consecutive anchors may share machines and overlapping 90-day outcome windows.",
        ],
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )

    print("\nOverall Top-K/Top-N summary", flush=True)
    print(
        overall_summary[
            [
                "variant",
                "selection_type",
                "selection_value",
                "anchor_count",
                "micro_precision",
                "micro_recall",
                "micro_lift_vs_fleet",
                "minimum_anchor_precision",
                "maximum_anchor_precision",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nOutputs written to {OUTPUT_DIR}", flush=True)
    print(f"Elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
