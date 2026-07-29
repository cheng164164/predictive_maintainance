"""Step 10: deployment-style multi-anchor natural-prevalence fleet evaluation.

This optional evaluation complements the ratio-sampled holdouts. It selects
several fixed earlier validation anchors and later test anchors. At each anchor,
all eligible machines in the corresponding fixed machine split are scored using
one shared calendar feature window. Rankings and Top-K/Top-N metrics are computed
within each anchor, then summarized across anchors.

The same scored snapshots are relabeled for every configured future-claim
horizon. Prediction outputs include the actual next claim date and days from the
anchor to that claim so engineers can inspect whether the highest-ranked machines
experienced near-future claims.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from cc_utils import (
    annotate_future_claim_outcomes,
    bootstrap_ranking_metric_intervals,
    build_machine_master,
    build_multi_anchor_fleet_base_rows,
    build_window_features,
    configured_evaluation_horizons,
    ensure_dir,
    fit_model_pipeline,
    future_claim_target_col,
    load_sources,
    make_model_pipeline,
    predict_score,
    threshold_free_metrics,
    top_k_metrics,
    top_n_metrics,
    validate_dataset_features,
    window_config_name,
    write_json,
)

DATE_COLUMNS = [
    "window_start",
    "window_end",
    "as_of_anchor_date",
    "as_of_actual_next_claim_date",
    "next_claim_date_on_or_after_window_end",
    "future_claim_date",
]


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _stable_frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    work = df.reindex(columns=columns).copy()
    for col in columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = pd.to_datetime(work[col], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("<NA>")
        else:
            work[col] = work[col].astype("string").fillna("<NA>").replace("", "<NA>")
    if columns:
        work = work.sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(work.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def _load_dataset_index() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}. Run Steps 01 and 02 first.")
    return pd.read_csv(path, low_memory=False)


def _load_episodes() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "01_claim_episodes" / "claim_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Claim episodes not found: {path}. Run Step 01 first.")
    return pd.read_csv(path, parse_dates=["claim_date", "episode_end_date"], low_memory=False)


def _top_k_rates() -> list[float]:
    values = getattr(config, "MULTI_ANCHOR_FLEET_TOP_K_RATES", [0.01, 0.05, 0.10, 0.20])
    rates = sorted({float(x) for x in values})
    if not rates or any(x <= 0 or x > 1 for x in rates):
        raise ValueError("MULTI_ANCHOR_FLEET_TOP_K_RATES must be in (0, 1].")
    return rates


def _top_n_counts() -> list[int]:
    values = getattr(config, "MULTI_ANCHOR_FLEET_TOP_N_COUNTS", [10, 20, 50])
    counts = sorted({int(x) for x in values})
    if not counts or any(x < 1 for x in counts):
        raise ValueError("MULTI_ANCHOR_FLEET_TOP_N_COUNTS must contain integers >= 1.")
    return counts


def _bootstrap_enabled() -> bool:
    return bool(getattr(config, "MULTI_ANCHOR_FLEET_BOOTSTRAP_ENABLED", True))


def _bootstrap_seed(*parts: object) -> int:
    base = int(getattr(config, "MULTI_ANCHOR_FLEET_RANDOM_STATE", 20260728))
    payload = "|".join([str(base), *[str(x) for x in parts]])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**32 - 1)


def _configured_horizons() -> list[int]:
    horizons = configured_evaluation_horizons(config)
    if not horizons:
        raise ValueError("EVALUATION_CLAIM_HORIZON_DAYS must contain at least one positive horizon.")
    return [int(x) for x in horizons]


def _lock_expected_metadata(
    dataset_row: pd.Series,
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict:
    return {
        "lock_version": 1,
        "strategy": "fixed_multi_anchor_natural_prevalence_fleet_snapshots",
        "dataset_id": str(dataset_row["dataset_id"]),
        "lead_max_days": int(dataset_row["lead_max_days"]),
        "lead_min_days": int(dataset_row["lead_min_days"]),
        "validation_dates": [str(x) for x in getattr(config, "MULTI_ANCHOR_FLEET_VALIDATION_DATES", [])],
        "test_dates": [str(x) for x in getattr(config, "MULTI_ANCHOR_FLEET_TEST_DATES", [])],
        "validation_anchor_count": int(getattr(config, "MULTI_ANCHOR_FLEET_VALIDATION_ANCHOR_COUNT", 3)),
        "test_anchor_count": int(getattr(config, "MULTI_ANCHOR_FLEET_TEST_ANCHOR_COUNT", 2)),
        "minimum_positive_machines": int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_POSITIVE_MACHINES", 5)),
        "minimum_eligible_machines": int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_ELIGIBLE_MACHINES", 50)),
        "minimum_days_between_anchors": int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS", 60)),
        "validation_period_fraction": float(getattr(config, "MULTI_ANCHOR_FLEET_VALIDATION_PERIOD_FRACTION", 0.70)),
        "test_start_gap_days": int(getattr(config, "MULTI_ANCHOR_FLEET_TEST_START_GAP_DAYS", 30)),
        "random_state": int(getattr(config, "MULTI_ANCHOR_FLEET_RANDOM_STATE", 20260728)),
        "max_machines_per_anchor": getattr(config, "MULTI_ANCHOR_FLEET_MAX_MACHINES_PER_ANCHOR", None),
        "evaluation_horizons": _configured_horizons(),
        "include_claim_on_window_end": bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True)),
        "claim_history_fingerprint": _stable_frame_fingerprint(
            episodes, ["claim_episode_id", "machine_key", "claim_date"]
        ),
        "machine_coverage_fingerprint": _stable_frame_fingerprint(
            machine_master, ["machine_key", "full_model", "first_source_date", "last_source_date"]
        ),
        "machine_assignment_fingerprint": _stable_frame_fingerprint(
            assignments, ["machine_key", "split", "full_model"]
        ),
    }


def _load_or_build_base_rows(
    dataset_row: pd.Series,
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    assignments: pd.DataFrame,
    asset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    ensure_dir(asset_dir)
    base_path = asset_dir / "fixed_multi_anchor_fleet_base_rows.csv"
    audit_path = asset_dir / "fixed_multi_anchor_fleet_sampling_audit.csv"
    summary_path = asset_dir / "fixed_multi_anchor_fleet_anchor_summary.csv"
    metadata_path = asset_dir / "fixed_multi_anchor_fleet_metadata.json"
    expected = _lock_expected_metadata(dataset_row, episodes, machine_master, assignments)

    if base_path.exists() and metadata_path.exists():
        observed = json.loads(metadata_path.read_text())
        if observed != expected:
            raise ValueError(
                "The locked multi-anchor fleet cohort no longer matches the current source/configuration. "
                f"Delete {asset_dir} only when you intentionally want to create a new locked cohort."
            )
        base = _read_csv(base_path)
        audit = _read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
        summary = _read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        return base, audit, summary, True

    window_config = {
        "lead_max_days": int(dataset_row["lead_max_days"]),
        "lead_min_days": int(dataset_row["lead_min_days"]),
    }
    base, audit, summary, _ = build_multi_anchor_fleet_base_rows(
        episodes=episodes,
        machine_master=machine_master,
        window_config=window_config,
        config=config,
        claim_history_episodes=episodes,
        split_assignments=assignments,
    )
    base.to_csv(base_path, index=False)
    audit.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(expected, indent=2, default=str))
    return base, audit, summary, False


def _rank_within_anchor(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["score"] = np.asarray(score, dtype=float)
    out["score_rank_within_anchor"] = (
        out.groupby("fleet_anchor_id", sort=False)["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["eligible_machines_at_anchor"] = out.groupby("fleet_anchor_id")["machine_key"].transform("size").astype(int)
    out["score_top_fraction_within_anchor"] = (
        out["score_rank_within_anchor"] / out["eligible_machines_at_anchor"]
    )
    return out.sort_values(
        ["split", "as_of_anchor_date", "score", "machine_key"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _claim_timing_summary(selected: pd.DataFrame, all_anchor: pd.DataFrame, horizon: int) -> dict:
    days = pd.to_numeric(selected["days_to_next_claim_on_or_after_window_end"], errors="coerce")
    y = pd.to_numeric(selected[future_claim_target_col(horizon)], errors="coerce").fillna(0).astype(int)
    all_y = pd.to_numeric(all_anchor[future_claim_target_col(horizon)], errors="coerce").fillna(0).astype(int)
    row = {
        "eligible_machines_at_anchor": int(len(all_anchor)),
        "positive_machines_at_anchor": int(all_y.sum()),
        "flagged_count": int(len(selected)),
        "true_positive_count": int(y.sum()),
        "precision": float(y.mean()) if len(y) else np.nan,
        "recall": float(y.sum() / all_y.sum()) if int(all_y.sum()) else np.nan,
        "overall_positive_rate": float(all_y.mean()) if len(all_y) else np.nan,
        "lift_vs_fleet": float(y.mean() / all_y.mean()) if len(y) and float(all_y.mean()) > 0 else np.nan,
        "future_claim_observed_count": int(days.notna().sum()),
        "median_days_to_next_claim_all_observed": float(days.dropna().median()) if days.notna().any() else np.nan,
        "median_days_to_next_claim_within_horizon": float(days[(days >= 0) & (days <= horizon)].median()) if ((days >= 0) & (days <= horizon)).any() else np.nan,
    }
    for check_horizon in _configured_horizons():
        row[f"claim_count_within_{check_horizon}d"] = int((days.notna() & days.ge(0) & days.le(check_horizon)).sum())
        row[f"claim_rate_within_{check_horizon}d"] = float((days.notna() & days.ge(0) & days.le(check_horizon)).mean()) if len(days) else np.nan
    return row


def _score_deciles(anchor: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = anchor.copy()
    n = len(out)
    if n == 0:
        return pd.DataFrame()
    out["score_decile_from_top"] = np.minimum(
        10,
        np.floor((out["score_rank_within_anchor"] - 1) * 10 / n).astype(int) + 1,
    )
    target_col = future_claim_target_col(horizon)
    overall = float(pd.to_numeric(out[target_col], errors="coerce").fillna(0).mean())
    rows = []
    for decile, group in out.groupby("score_decile_from_top", sort=True):
        rate = float(pd.to_numeric(group[target_col], errors="coerce").fillna(0).mean())
        rows.append({
            "score_decile_from_top": int(decile),
            "rows": int(len(group)),
            "positive_rows": int(pd.to_numeric(group[target_col], errors="coerce").fillna(0).sum()),
            "claim_rate": rate,
            "overall_anchor_claim_rate": overall,
            "lift_vs_anchor": float(rate / overall) if overall > 0 else np.nan,
            "min_score": float(group["score"].min()),
            "max_score": float(group["score"].max()),
        })
    return pd.DataFrame(rows)


def _evaluate_scored_snapshots(
    scored: pd.DataFrame,
    dataset_id: str,
    algorithm: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    topk_frames: list[pd.DataFrame] = []
    topn_frames: list[pd.DataFrame] = []
    timing_rows: list[dict] = []
    detail_parts: list[pd.DataFrame] = []
    decile_parts: list[pd.DataFrame] = []
    horizons = _configured_horizons()

    for horizon in horizons:
        target_col = future_claim_target_col(horizon)
        for (split_name, anchor_id), anchor in scored.groupby(["split", "fleet_anchor_id"], sort=True):
            anchor = anchor.sort_values(["score", "machine_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
            y = pd.to_numeric(anchor[target_col], errors="coerce").fillna(0).astype(int)
            score = anchor["score"].to_numpy(dtype=float)
            if y.nunique() < 2:
                free = {"average_precision": np.nan, "roc_auc": np.nan}
            else:
                free = threshold_free_metrics(y, score)
            metric = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "fleet_anchor_id": anchor_id,
                "anchor_date": pd.to_datetime(anchor["as_of_anchor_date"], errors="coerce").iloc[0],
                "evaluation_horizon_days": horizon,
                "evaluation_target_col": target_col,
                "evaluation_rows": int(len(anchor)),
                "evaluation_positive_rows": int(y.sum()),
                "evaluation_negative_rows": int(len(y) - y.sum()),
                "evaluation_positive_rate": float(y.mean()),
                "ranking_unit": "machine_within_anchor",
                **free,
            }
            topk = top_k_metrics(y, score, _top_k_rates())
            topn = top_n_metrics(y, score, _top_n_counts())
            if _bootstrap_enabled() and y.nunique() == 2:
                boot_free, boot_topk, boot_topn = bootstrap_ranking_metric_intervals(
                    y_true=y,
                    score=score,
                    top_k_rates=_top_k_rates(),
                    top_n_counts=_top_n_counts(),
                    n_resamples=int(getattr(config, "MULTI_ANCHOR_FLEET_BOOTSTRAP_N_RESAMPLES", 1000)),
                    confidence_level=float(getattr(config, "MULTI_ANCHOR_FLEET_BOOTSTRAP_CONFIDENCE_LEVEL", 0.95)),
                    random_state=_bootstrap_seed(dataset_id, algorithm, split_name, anchor_id, horizon),
                )
                metric.update(boot_free)
                topk = topk.merge(boot_topk, on="top_k_rate", how="left", validate="one_to_one")
                topn = topn.merge(boot_topn, on="top_n_requested", how="left", validate="one_to_one")
            metric_rows.append(metric)

            common = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "fleet_anchor_id": anchor_id,
                "anchor_date": metric["anchor_date"],
                "evaluation_horizon_days": horizon,
                "evaluation_target_col": target_col,
                "ranking_unit": "machine_within_anchor",
            }
            for key, value in reversed(list(common.items())):
                topk.insert(0, key, value)
                topn.insert(0, key, value)
            topk_frames.append(topk)
            topn_frames.append(topn)

            for requested in _top_n_counts():
                selected = anchor.head(min(int(requested), len(anchor))).copy()
                timing_rows.append({
                    **common,
                    "selection_type": "fixed_top_n",
                    "selection_value": int(requested),
                    **_claim_timing_summary(selected, anchor, horizon),
                })
                detail = selected.copy()
                detail.insert(0, "selection_value", int(requested))
                detail.insert(0, "selection_type", "fixed_top_n")
                detail.insert(0, "evaluation_horizon_days", horizon)
                detail.insert(0, "algorithm", algorithm)
                detail.insert(0, "dataset_id", dataset_id)
                detail_parts.append(detail)
            for rate in _top_k_rates():
                count = max(1, int(np.ceil(len(anchor) * float(rate))))
                selected = anchor.head(count).copy()
                timing_rows.append({
                    **common,
                    "selection_type": "percentage_top_k",
                    "selection_value": float(rate),
                    **_claim_timing_summary(selected, anchor, horizon),
                })

            deciles = _score_deciles(anchor, horizon)
            if not deciles.empty:
                for key, value in reversed(list(common.items())):
                    deciles.insert(0, key, value)
                decile_parts.append(deciles)

        # Save one detailed scored file per split/horizon. Scores and ranks are
        # unchanged across horizons; target columns differ by future horizon.
        for split_name in ["validation", "test"]:
            pred = scored[scored["split"].eq(split_name)].copy()
            if pred.empty:
                continue
            pred.insert(0, "evaluation_target", pd.to_numeric(pred[target_col], errors="coerce").fillna(0).astype(int))
            pred.insert(0, "evaluation_target_col", target_col)
            pred.insert(0, "evaluation_horizon_days", horizon)
            pred.insert(0, "algorithm", algorithm)
            pred.insert(0, "dataset_id", dataset_id)
            pred.to_csv(
                output_dir / f"{dataset_id}__{algorithm}__{split_name}__horizon_{horizon}d__fleet_predictions.csv",
                index=False,
            )

    metrics = pd.DataFrame(metric_rows)
    topk_all = pd.concat(topk_frames, ignore_index=True) if topk_frames else pd.DataFrame()
    topn_all = pd.concat(topn_frames, ignore_index=True) if topn_frames else pd.DataFrame()
    timing = pd.DataFrame(timing_rows)
    details = pd.concat(detail_parts, ignore_index=True, sort=False) if detail_parts else pd.DataFrame()
    deciles = pd.concat(decile_parts, ignore_index=True, sort=False) if decile_parts else pd.DataFrame()

    aggregate_rows = []
    if not metrics.empty:
        for (split_name, horizon), group in metrics.groupby(["split", "evaluation_horizon_days"], sort=True):
            aggregate_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "evaluation_horizon_days": int(horizon),
                "anchor_count": int(group["fleet_anchor_id"].nunique()),
                "mean_positive_rate": float(group["evaluation_positive_rate"].mean()),
                "mean_average_precision": float(group["average_precision"].mean()),
                "median_average_precision": float(group["average_precision"].median()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "median_roc_auc": float(group["roc_auc"].median()),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    return metrics, topk_all, topn_all, timing, details, deciles, aggregate


def run(
    dataset_index_path: str | Path | None = None,
    step_dir: str | Path | None = None,
) -> None:
    config.refresh_derived_config()
    if not bool(getattr(config, "MULTI_ANCHOR_FLEET_ENABLED", True)):
        print("10_multi_anchor_fleet_evaluation skipped: MULTI_ANCHOR_FLEET_ENABLED=False", flush=True)
        return

    output_dir = Path(step_dir) if step_dir is not None else config.OUTPUT_DIR / "10_multi_anchor_fleet_evaluation"
    ensure_dir(output_dir)
    dataset_index = (
        pd.read_csv(Path(dataset_index_path), low_memory=False)
        if dataset_index_path is not None
        else _load_dataset_index()
    )
    episodes = _load_episodes()
    print("Loading sources for multi-anchor fleet evaluation...", flush=True)
    sources = load_sources(config)
    machine_master = build_machine_master(sources)

    all_metrics = []
    all_topk = []
    all_topn = []
    all_timing = []
    all_details = []
    all_deciles = []
    all_aggregate = []
    run_summaries = []

    for _, dataset_row in dataset_index.iterrows():
        dataset_id = str(dataset_row["dataset_id"])
        print(f"Building/scoring multi-anchor fleet cohort: {dataset_id}", flush=True)
        assignment_path = Path(str(dataset_row["fixed_machine_split_assignment_path"]))
        assignments = pd.read_csv(assignment_path, low_memory=False)
        asset_dir = Path(config.FIXED_SPLIT_ASSET_DIR) / dataset_id / "multi_anchor_fleet"
        base_rows, audit, anchor_summary, reused = _load_or_build_base_rows(
            dataset_row, episodes, machine_master, assignments, asset_dir
        )
        base_rows.to_csv(output_dir / f"{dataset_id}__multi_anchor_base_rows.csv", index=False)
        audit.to_csv(output_dir / f"{dataset_id}__multi_anchor_sampling_audit.csv", index=False)
        anchor_summary.to_csv(output_dir / f"{dataset_id}__multi_anchor_anchor_summary.csv", index=False)

        print(f"  Engineering {config.FEATURE_SET} features for {len(base_rows):,} fleet snapshots...", flush=True)
        fleet_df = build_window_features(base_rows, sources=sources, episodes=episodes, config=config)
        fleet_df = annotate_future_claim_outcomes(
            fleet_df,
            claim_history_episodes=episodes,
            config=config,
            horizons=_configured_horizons(),
        )
        validate_dataset_features(fleet_df, config)
        fleet_path = output_dir / f"{dataset_id}__multi_anchor_fleet_dataset.csv"
        fleet_df.to_csv(fleet_path, index=False)

        train_df = _read_csv(Path(str(dataset_row["training_dataset_path"])))
        validate_dataset_features(train_df, config)
        feature_cols = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
        X_train = train_df[feature_cols]
        y_train = pd.to_numeric(train_df["target"], errors="raise").astype(int)
        validation_df = fleet_df[fleet_df["split"].eq("validation")].copy()
        X_fit_eval = validation_df[feature_cols]
        y_fit_eval = pd.to_numeric(validation_df["target"], errors="coerce").fillna(0).astype(int)

        for algorithm in config.MODELS_TO_RUN:
            model = make_model_pipeline(algorithm, config)
            if model is None:
                continue
            fit_metadata = fit_model_pipeline(
                model,
                algorithm,
                X_train,
                y_train,
                config,
                X_eval=X_fit_eval,
                y_eval=y_fit_eval,
                eval_name="multi_anchor_validation_natural_prevalence",
            )
            score = predict_score(model, fleet_df[feature_cols], algorithm)
            scored = _rank_within_anchor(fleet_df, score)
            outputs = _evaluate_scored_snapshots(scored, dataset_id, algorithm, output_dir)
            metrics, topk, topn, timing, details, deciles, aggregate = outputs
            all_metrics.append(metrics)
            all_topk.append(topk)
            all_topn.append(topn)
            all_timing.append(timing)
            all_details.append(details)
            all_deciles.append(deciles)
            all_aggregate.append(aggregate)
            run_summaries.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "training_rows": int(len(train_df)),
                "fleet_snapshot_rows": int(len(fleet_df)),
                "unique_machines": int(fleet_df["machine_key"].nunique()),
                "validation_anchors": int(fleet_df.loc[fleet_df["split"].eq("validation"), "fleet_anchor_id"].nunique()),
                "test_anchors": int(fleet_df.loc[fleet_df["split"].eq("test"), "fleet_anchor_id"].nunique()),
                "locked_base_rows_reused": bool(reused),
                "fit_metadata": fit_metadata,
            })

    def concat(frames):
        kept = [x for x in frames if x is not None and not x.empty]
        return pd.concat(kept, ignore_index=True, sort=False) if kept else pd.DataFrame()

    timing_all = concat(all_timing)
    aggregate_selection_rows = []
    if not timing_all.empty:
        group_cols = [
            "dataset_id", "algorithm", "split", "evaluation_horizon_days",
            "selection_type", "selection_value",
        ]
        for keys, group in timing_all.groupby(group_cols, dropna=False, sort=True):
            row = dict(zip(group_cols, keys))
            total_flagged = int(group["flagged_count"].sum())
            total_tp = int(group["true_positive_count"].sum())
            total_positive = int(group["positive_machines_at_anchor"].sum())
            total_eligible = int(group["eligible_machines_at_anchor"].sum())
            row.update({
                "anchor_count": int(group["fleet_anchor_id"].nunique()),
                "total_eligible_machine_snapshots": total_eligible,
                "total_positive_machine_snapshots": total_positive,
                "total_flagged_machine_snapshots": total_flagged,
                "total_true_positive_machine_snapshots": total_tp,
                "micro_precision": float(total_tp / total_flagged) if total_flagged else np.nan,
                "micro_recall": float(total_tp / total_positive) if total_positive else np.nan,
                "natural_positive_rate": float(total_positive / total_eligible) if total_eligible else np.nan,
                "micro_lift_vs_fleet": (
                    float((total_tp / total_flagged) / (total_positive / total_eligible))
                    if total_flagged and total_positive and total_eligible else np.nan
                ),
                "mean_anchor_precision": float(group["precision"].mean()),
                "minimum_anchor_precision": float(group["precision"].min()),
                "maximum_anchor_precision": float(group["precision"].max()),
                "mean_anchor_recall": float(group["recall"].mean()),
                "mean_anchor_lift": float(group["lift_vs_fleet"].mean()),
            })
            aggregate_selection_rows.append(row)
    selection_across_anchors = pd.DataFrame(aggregate_selection_rows)

    outputs = {
        "multi_anchor_metrics_by_anchor.csv": concat(all_metrics),
        "multi_anchor_percentage_top_k_by_anchor.csv": concat(all_topk),
        "multi_anchor_fixed_top_n_by_anchor.csv": concat(all_topn),
        "multi_anchor_top_selection_claim_timing.csv": timing_all,
        "multi_anchor_top_selection_across_anchors.csv": selection_across_anchors,
        "multi_anchor_top_n_machine_claim_details.csv": concat(all_details),
        "multi_anchor_score_decile_claim_rates.csv": concat(all_deciles),
        "multi_anchor_metrics_across_anchors.csv": concat(all_aggregate),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    write_json(
        {
            "step": "10_multi_anchor_fleet_evaluation",
            "strategy": "fixed earlier validation anchors and later test anchors; all eligible machines; natural prevalence",
            "output_dir": str(output_dir),
            "evaluation_horizons": _configured_horizons(),
            "top_k_rates": _top_k_rates(),
            "top_n_counts": _top_n_counts(),
            "bootstrap_enabled": _bootstrap_enabled(),
            "run_summaries": run_summaries,
            "notes": [
                "Each ranking is calculated independently within one anchor date.",
                "Validation anchors are earlier; test anchors are later when dates are selected automatically.",
                "All eligible machines are included at natural prevalence unless a development cap is configured.",
                "The same scored snapshots are relabeled for every configured future-claim horizon.",
                "Top-N detail files include actual next claim date and days to next claim.",
            ],
        },
        output_dir / "run_summary.json",
    )
    print(f"10_multi_anchor_fleet_evaluation completed. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    run()
