"""Step 12: negative-only score-distribution diagnostics.

This optional evaluation builds a fixed random cohort containing no positive rows
under the training/design horizon (``lead_min_days``). It intentionally mixes:

1. strict negatives with no claim through
   ``NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END``; and
2. delayed-claim negatives with no claim within ``lead_min_days`` but a first
   later claim before the strict exclusion horizon.

The delayed group is sampled as evenly as feasible across claim lead-time bins.
Every output row includes the score, thresholded prediction, next claim date,
days to claim, and relaxed-horizon labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from cc_utils import (
    _max_claim_observation_date,
    _random_window_end_candidates_by_machine,
    _stable_hash_int,
    annotate_future_claim_outcomes,
    build_machine_master,
    build_window_features,
    claim_dates_by_machine,
    configured_evaluation_horizons,
    ensure_dir,
    fit_model_pipeline,
    load_sources,
    make_model_pipeline,
    predict_score,
    validate_dataset_features,
    window_config_name,
    write_json,
)

DATE_COLUMNS = [
    "window_start",
    "window_end",
    "future_claim_date",
    "next_claim_date_on_or_after_window_end",
    "control_no_claim_start",
    "control_no_claim_end",
]


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _stable_frame_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    work = frame.reindex(columns=list(columns)).copy()
    for column in columns:
        if "date" in column.lower() or pd.api.types.is_datetime64_any_dtype(work[column]):
            work[column] = (
                pd.to_datetime(work[column], errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%S")
                .fillna("<NA>")
            )
        else:
            work[column] = (
                work[column].astype("string").str.strip().fillna("<NA>").replace("", "<NA>")
            )
    if columns:
        work = work.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(
        work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _configured_splits() -> list[str]:
    raw = getattr(config, "NEGATIVE_ONLY_EVALUATION_SPLITS", ["validation"])
    if isinstance(raw, str):
        raw = [raw]
    splits = []
    for value in raw:
        name = str(value).strip().lower()
        if name not in {"validation", "test"}:
            raise ValueError(
                "NEGATIVE_ONLY_EVALUATION_SPLITS may contain only 'validation' and 'test'."
            )
        if name not in splits:
            splits.append(name)
    if not splits:
        raise ValueError("NEGATIVE_ONLY_EVALUATION_SPLITS cannot be empty.")
    return splits


def _configured_horizons(lead_min_days: int) -> list[int]:
    raw = getattr(config, "NEGATIVE_ONLY_EVALUATION_HORIZONS", None)
    if raw in (None, [], (), ""):
        horizons = configured_evaluation_horizons(config)
    elif isinstance(raw, (int, float, np.integer, np.floating, str)):
        horizons = [int(raw)]
    else:
        horizons = [int(value) for value in raw]
    horizons = sorted({int(value) for value in horizons if int(value) >= 0})
    if int(lead_min_days) not in horizons:
        horizons.append(int(lead_min_days))
        horizons = sorted(set(horizons))
    return horizons


def _sample_size() -> Optional[int]:
    raw = getattr(config, "NEGATIVE_ONLY_SAMPLE_SIZE_PER_SPLIT", 500)
    if raw in (None, "", 0, "0"):
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("NEGATIVE_ONLY_SAMPLE_SIZE_PER_SPLIT must be positive or None.")
    return value


def _strict_fraction() -> float:
    value = float(getattr(config, "NEGATIVE_ONLY_STRICT_FRACTION", 0.50))
    if not 0 <= value <= 1:
        raise ValueError("NEGATIVE_ONLY_STRICT_FRACTION must be between 0 and 1.")
    return value


def _review_thresholds() -> list[float]:
    raw = getattr(
        config,
        "NEGATIVE_ONLY_SCORE_REVIEW_THRESHOLDS",
        [0.10, 0.25, 0.50, 0.75, 0.90],
    )
    values = sorted({float(value) for value in raw})
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("NEGATIVE_ONLY_SCORE_REVIEW_THRESHOLDS must be in [0, 1].")
    return values


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


def _prepare_master(
    machine_master: pd.DataFrame,
    assignments: pd.DataFrame,
    requested_splits: Sequence[str],
) -> pd.DataFrame:
    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    columns = [
        column
        for column in ["machine_key", "split", "full_model", "serial"]
        if column in assignments.columns
    ]
    assigned = assignments[columns].drop_duplicates("machine_key").copy()
    assigned["machine_key"] = assigned["machine_key"].astype(str)
    master = master.merge(assigned, on="machine_key", how="left", suffixes=("", "_assigned"))
    for column in ["full_model", "serial"]:
        other = f"{column}_assigned"
        if other in master.columns:
            current = master.get(column, pd.Series("", index=master.index)).astype("string")
            master[column] = master.get(column, pd.Series("", index=master.index)).where(
                current.fillna("").str.strip().ne(""), master[other]
            )
            master = master.drop(columns=[other])
    master["full_model"] = master.get("full_model", pd.Series("", index=master.index)).fillna("").astype(str)
    master["serial"] = master.get("serial", pd.Series("", index=master.index)).fillna("").astype(str)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    return master[master["split"].astype(str).isin(list(requested_splits))].copy()


def _bin_edges(lead_min: int, strict_days: int, width: int) -> list[int]:
    if strict_days <= lead_min:
        return [lead_min, strict_days]
    values = [int(lead_min)]
    current = int(lead_min)
    while current < strict_days:
        current = min(strict_days, current + int(width))
        values.append(current)
    return values


def _lead_bin_label(days: float, edges: Sequence[int]) -> str:
    for lower, upper in zip(edges[:-1], edges[1:]):
        if float(days) > float(lower) and float(days) <= float(upper):
            return f"{int(lower) + 1}-{int(upper)}d"
    return "outside_configured_delayed_range"


def _one_candidate_per_machine_and_bin(
    rows: list[dict],
    seed: int,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "machine_key",
                "full_model",
                "serial",
                "split",
                "window_start",
                "window_end",
                "future_claim_date",
                "days_from_window_end_to_claim",
                "negative_case_type",
                "delayed_claim_lead_bin",
                "required_no_claim_days_after_window_end",
                "control_no_claim_start",
                "control_no_claim_end",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["_candidate_hash"] = [
        _stable_hash_int(
            f"{seed}|{machine}|{category}|{lead_bin}|{pd.Timestamp(end).date()}",
            seed,
        )
        for machine, category, lead_bin, end in zip(
            frame["machine_key"],
            frame["negative_case_type"],
            frame["delayed_claim_lead_bin"],
            frame["window_end"],
        )
    ]
    frame = frame.sort_values(
        ["negative_case_type", "delayed_claim_lead_bin", "machine_key", "_candidate_hash"],
        kind="mergesort",
    )
    return frame.drop_duplicates(
        ["negative_case_type", "delayed_claim_lead_bin", "machine_key"], keep="first"
    ).reset_index(drop=True)


def _candidate_rows_for_split(
    split_name: str,
    prepared_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    dates_by_machine: Mapping[str, np.ndarray],
    window_config: Mapping,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    lookback_days = lead_max - lead_min
    strict_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    prior_days = int(
        getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
    )
    width = max(1, int(getattr(config, "NEGATIVE_ONLY_DELAYED_CLAIM_BIN_WIDTH_DAYS", 30)))
    if _strict_fraction() < 1 and strict_days <= lead_min:
        raise ValueError(
            "Delayed-claim negative sampling requires "
            "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END > lead_min_days."
        )
    include_cutoff = bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True))
    max_observation = _max_claim_observation_date(config).normalize()
    require_coverage = bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True))
    raw_dates = _random_window_end_candidates_by_machine(sources)
    edges = _bin_edges(lead_min, strict_days, width)

    master = prepared_master[prepared_master["split"].astype(str).eq(str(split_name))].copy()
    master = master.drop_duplicates("machine_key").set_index("machine_key", drop=False)
    rows: list[dict] = []
    audits: list[dict] = []

    for machine_key, machine in master.iterrows():
        machine_key = str(machine_key)
        ends = pd.DatetimeIndex(
            pd.to_datetime(raw_dates.get(machine_key, []), errors="coerce")
        ).dropna()
        if ends.empty:
            audits.append({
                "split": split_name,
                "machine_key": machine_key,
                "candidate_status": "no_source_activity_dates",
                "strict_candidate_windows": 0,
                "delayed_candidate_windows": 0,
            })
            continue
        ends = ends.normalize().unique().sort_values()
        starts = ends - pd.Timedelta(days=lookback_days)
        base_mask = np.ones(len(ends), dtype=bool)
        if require_coverage:
            first_source = pd.to_datetime(machine.get("first_source_date"), errors="coerce")
            last_source = pd.to_datetime(machine.get("last_source_date"), errors="coerce")
            if pd.isna(first_source) or pd.isna(last_source):
                base_mask[:] = False
            else:
                base_mask &= np.asarray(
                    (first_source <= ends) & (last_source >= starts), dtype=bool
                )

        claim_dates = np.asarray(
            dates_by_machine.get(machine_key, np.array([], dtype="datetime64[ns]")),
            dtype="datetime64[ns]",
        )
        claim_dates = np.sort(claim_dates)
        if len(claim_dates):
            no_claim_starts = (
                starts - pd.Timedelta(days=prior_days)
            ).values.astype("datetime64[ns]")
            design_no_claim_ends = (
                ends + pd.Timedelta(days=lead_min)
            ).values.astype("datetime64[ns]")
            left = np.searchsorted(claim_dates, no_claim_starts, side="left")
            right = np.searchsorted(claim_dates, design_no_claim_ends, side="right")
            base_mask &= left == right
            end_values = ends.values.astype("datetime64[ns]")
            side = "left" if include_cutoff else "right"
            next_indices = np.searchsorted(claim_dates, end_values, side=side)
            valid_next = next_indices < len(claim_dates)
            next_dates = np.full(len(ends), np.datetime64("NaT"), dtype="datetime64[ns]")
            next_dates[valid_next] = claim_dates[next_indices[valid_next]]
            days = np.full(len(ends), np.nan, dtype=float)
            days[valid_next] = (
                next_dates[valid_next] - end_values[valid_next]
            ).astype("timedelta64[D]").astype(float)
        else:
            next_dates = np.full(len(ends), np.datetime64("NaT"), dtype="datetime64[ns]")
            days = np.full(len(ends), np.nan, dtype=float)

        strict_mask = (
            base_mask
            & np.asarray(
                (ends + pd.Timedelta(days=strict_days)) <= max_observation,
                dtype=bool,
            )
            & (np.isnan(days) | (days > float(strict_days)))
        )
        delayed_mask = (
            base_mask
            & np.isfinite(days)
            & (days > float(lead_min))
            & (days <= float(strict_days))
        )

        for idx in np.flatnonzero(strict_mask):
            next_date = pd.Timestamp(next_dates[idx]) if not np.isnat(next_dates[idx]) else pd.NaT
            rows.append({
                "machine_key": machine_key,
                "full_model": machine.get("full_model", ""),
                "serial": machine.get("serial", ""),
                "split": split_name,
                "window_start": pd.Timestamp(starts[idx]),
                "window_end": pd.Timestamp(ends[idx]),
                "future_claim_date": next_date,
                "days_from_window_end_to_claim": float(days[idx]) if np.isfinite(days[idx]) else np.nan,
                "negative_case_type": "strict_negative",
                "delayed_claim_lead_bin": "no_claim_within_strict_window",
                "required_no_claim_days_after_window_end": strict_days,
                "control_no_claim_start": pd.Timestamp(starts[idx]) - pd.Timedelta(days=prior_days),
                "control_no_claim_end": pd.Timestamp(ends[idx]) + pd.Timedelta(days=strict_days),
            })
        for idx in np.flatnonzero(delayed_mask):
            rows.append({
                "machine_key": machine_key,
                "full_model": machine.get("full_model", ""),
                "serial": machine.get("serial", ""),
                "split": split_name,
                "window_start": pd.Timestamp(starts[idx]),
                "window_end": pd.Timestamp(ends[idx]),
                "future_claim_date": pd.Timestamp(next_dates[idx]),
                "days_from_window_end_to_claim": float(days[idx]),
                "negative_case_type": "delayed_claim_negative",
                "delayed_claim_lead_bin": _lead_bin_label(float(days[idx]), edges),
                "required_no_claim_days_after_window_end": lead_min,
                "control_no_claim_start": pd.Timestamp(starts[idx]) - pd.Timedelta(days=prior_days),
                "control_no_claim_end": pd.Timestamp(ends[idx]) + pd.Timedelta(days=lead_min),
            })
        audits.append({
            "split": split_name,
            "machine_key": machine_key,
            "candidate_status": "screened",
            "strict_candidate_windows": int(strict_mask.sum()),
            "delayed_candidate_windows": int(delayed_mask.sum()),
        })

    return _one_candidate_per_machine_and_bin(rows, seed), pd.DataFrame(audits)


def _balanced_delayed_selection(
    candidates: pd.DataFrame,
    requested: int,
    seed: int,
    used_machines: set[str],
) -> pd.DataFrame:
    delayed = candidates[candidates["negative_case_type"].eq("delayed_claim_negative")].copy()
    if requested <= 0 or delayed.empty:
        return delayed.iloc[0:0].copy()
    delayed = delayed[~delayed["machine_key"].astype(str).isin(used_machines)].copy()
    bins = sorted(delayed["delayed_claim_lead_bin"].dropna().astype(str).unique())
    if not bins:
        return delayed.iloc[0:0].copy()
    quota_base, remainder = divmod(int(requested), len(bins))
    selected_parts: list[pd.DataFrame] = []
    selected_machines = set(used_machines)
    for index, lead_bin in enumerate(bins):
        quota = quota_base + (1 if index < remainder else 0)
        pool = delayed[
            delayed["delayed_claim_lead_bin"].astype(str).eq(lead_bin)
            & ~delayed["machine_key"].astype(str).isin(selected_machines)
        ].copy()
        pool["_selection_hash"] = pool.apply(
            lambda row: _stable_hash_int(
                f"{seed}|delayed|{lead_bin}|{row['machine_key']}|{pd.Timestamp(row['window_end']).date()}",
                seed,
            ),
            axis=1,
        )
        chosen = pool.sort_values(["_selection_hash", "machine_key"], kind="mergesort").head(quota)
        selected_parts.append(chosen)
        selected_machines.update(chosen["machine_key"].astype(str))

    selected = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else delayed.iloc[0:0].copy()
    )
    remaining = int(requested) - len(selected)
    if remaining > 0:
        pool = delayed[~delayed["machine_key"].astype(str).isin(selected_machines)].copy()
        pool["_selection_hash"] = pool.apply(
            lambda row: _stable_hash_int(
                f"{seed}|delayed_fill|{row['machine_key']}|{pd.Timestamp(row['window_end']).date()}",
                seed,
            ),
            axis=1,
        )
        fill = pool.sort_values(["_selection_hash", "machine_key"], kind="mergesort").head(remaining)
        selected = pd.concat([selected, fill], ignore_index=True, sort=False)
    return selected.drop(columns=["_selection_hash", "_candidate_hash"], errors="ignore")


def _strict_selection(
    candidates: pd.DataFrame,
    requested: int,
    seed: int,
    used_machines: set[str],
) -> pd.DataFrame:
    strict = candidates[
        candidates["negative_case_type"].eq("strict_negative")
        & ~candidates["machine_key"].astype(str).isin(used_machines)
    ].copy()
    if requested <= 0 or strict.empty:
        return strict.iloc[0:0].copy()
    strict["_selection_hash"] = strict.apply(
        lambda row: _stable_hash_int(
            f"{seed}|strict|{row['machine_key']}|{pd.Timestamp(row['window_end']).date()}",
            seed,
        ),
        axis=1,
    )
    return (
        strict.sort_values(["_selection_hash", "machine_key"], kind="mergesort")
        .drop_duplicates("machine_key", keep="first")
        .head(requested)
        .drop(columns=["_selection_hash", "_candidate_hash"], errors="ignore")
    )


def build_negative_only_base_rows(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    split_assignments: pd.DataFrame,
    window_config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build fixed random all-negative machine snapshots for diagnostics."""
    splits = _configured_splits()
    seed = int(getattr(config, "NEGATIVE_ONLY_RANDOM_STATE", 20260729))
    strict_fraction = _strict_fraction()
    sample_size = _sample_size()
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    strict_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    prepared_master = _prepare_master(machine_master, split_assignments, splits)
    dates_by_machine = claim_dates_by_machine(episodes)
    window_name = window_config_name(window_config)

    selected_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for split_name in splits:
        candidates, audit = _candidate_rows_for_split(
            split_name,
            prepared_master,
            sources,
            dates_by_machine,
            window_config,
            seed,
        )
        audit_parts.append(audit)
        strict_available = int(
            candidates.loc[candidates["negative_case_type"].eq("strict_negative"), "machine_key"].nunique()
        )
        delayed_available = int(
            candidates.loc[candidates["negative_case_type"].eq("delayed_claim_negative"), "machine_key"].nunique()
        )
        if sample_size is None:
            target_total = strict_available + delayed_available
        else:
            target_total = min(int(sample_size), strict_available + delayed_available)
        desired_strict = int(round(target_total * strict_fraction))
        desired_delayed = max(0, target_total - desired_strict)

        used: set[str] = set()
        delayed = _balanced_delayed_selection(
            candidates,
            desired_delayed,
            seed + _stable_hash_int(split_name, seed),
            used,
        )
        used.update(delayed["machine_key"].astype(str))
        strict = _strict_selection(
            candidates,
            desired_strict,
            seed + _stable_hash_int(split_name + "|strict", seed),
            used,
        )
        used.update(strict["machine_key"].astype(str))

        # Fill any shortage from the other category without duplicating machines.
        selected = pd.concat([strict, delayed], ignore_index=True, sort=False)
        shortage = target_total - len(selected)
        if shortage > 0:
            strict_fill = _strict_selection(
                candidates,
                shortage,
                seed + _stable_hash_int(split_name + "|strict_fill", seed),
                set(selected["machine_key"].astype(str)),
            )
            selected = pd.concat([selected, strict_fill], ignore_index=True, sort=False)
            shortage = target_total - len(selected)
        if shortage > 0:
            delayed_fill = _balanced_delayed_selection(
                candidates,
                shortage,
                seed + _stable_hash_int(split_name + "|delayed_fill", seed),
                set(selected["machine_key"].astype(str)),
            )
            selected = pd.concat([selected, delayed_fill], ignore_index=True, sort=False)

        selected = selected.drop_duplicates("machine_key", keep="first").copy()
        selected["snapshot_id"] = [
            f"negative_only__{window_name}__{split_name}__{machine}"
            for machine in selected["machine_key"].astype(str)
        ]
        selected["case_control_group_id"] = selected["snapshot_id"]
        selected["row_role"] = "control"
        selected["target"] = 0
        selected["true_label_design_horizon"] = 0
        selected["window_name"] = window_name
        selected["lead_max_days"] = lead_max
        selected["lead_min_days"] = lead_min
        selected["negative_sampling_type"] = selected["negative_case_type"]
        selected["control_sampling_reason"] = np.where(
            selected["negative_case_type"].eq("strict_negative"),
            f"no_claim_within_{strict_days}_days_after_window_end",
            f"no_claim_within_{lead_min}_days_but_claim_within_{strict_days}_days",
        )
        selected["holdout_sampling_design"] = "negative_only_mixed_strict_and_delayed_claim"
        selected["evaluation_population"] = "all_negative_at_design_horizon"
        selected["holdout_random_state"] = seed
        selected["case_machine_key"] = selected["machine_key"]
        selected["claim_episode_id"] = ""
        selected["control_number_within_group"] = 1
        selected_parts.append(selected)

        strict_selected = int(selected["negative_case_type"].eq("strict_negative").sum())
        delayed_selected = int(selected["negative_case_type"].eq("delayed_claim_negative").sum())
        summary_rows.append({
            "split": split_name,
            "window_name": window_name,
            "lead_min_days": lead_min,
            "strict_no_claim_days": strict_days,
            "requested_sample_size": sample_size,
            "requested_strict_fraction": strict_fraction,
            "strict_candidate_machines": strict_available,
            "delayed_candidate_machines": delayed_available,
            "selected_rows": int(len(selected)),
            "strict_selected_rows": strict_selected,
            "delayed_selected_rows": delayed_selected,
            "actual_strict_fraction": float(strict_selected / len(selected)) if len(selected) else np.nan,
            "unique_machines": int(selected["machine_key"].nunique()),
        })

    base = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    if base.empty:
        raise ValueError("Negative-only sampling produced no rows.")
    if not base["target"].eq(0).all():
        raise ValueError("Negative-only cohort unexpectedly contains positive design labels.")
    if base.groupby("split")["machine_key"].apply(lambda x: x.astype(str).duplicated().any()).any():
        raise ValueError("Negative-only cohort contains duplicate machines within a split.")
    return (
        base.sort_values(["split", "negative_case_type", "delayed_claim_lead_bin", "machine_key"], kind="mergesort").reset_index(drop=True),
        pd.concat(audit_parts, ignore_index=True, sort=False) if audit_parts else pd.DataFrame(),
        pd.DataFrame(summary_rows),
    )


def _expected_metadata(
    dataset_row: pd.Series,
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict:
    lead_min = int(dataset_row["lead_min_days"])
    return {
        "lock_version": 1,
        "strategy": "negative_only_mixed_strict_and_delayed_claim",
        "dataset_id": str(dataset_row["dataset_id"]),
        "lead_max_days": int(dataset_row["lead_max_days"]),
        "lead_min_days": lead_min,
        "strict_no_claim_days": int(
            getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180)
        ),
        "prior_claim_exclusion_days": int(
            getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
        ),
        "splits": _configured_splits(),
        "sample_size_per_split": _sample_size(),
        "strict_fraction": _strict_fraction(),
        "delayed_bin_width_days": int(
            getattr(config, "NEGATIVE_ONLY_DELAYED_CLAIM_BIN_WIDTH_DAYS", 30)
        ),
        "random_state": int(getattr(config, "NEGATIVE_ONLY_RANDOM_STATE", 20260729)),
        "evaluation_horizons": _configured_horizons(lead_min),
        "claim_history_fingerprint": _stable_frame_fingerprint(
            episodes, ["claim_episode_id", "machine_key", "claim_date"]
        ),
        "machine_coverage_fingerprint": _stable_frame_fingerprint(
            machine_master,
            ["machine_key", "full_model", "first_source_date", "last_source_date"],
        ),
        "machine_assignment_fingerprint": _stable_frame_fingerprint(
            assignments, ["machine_key", "split", "full_model"]
        ),
    }


def _load_or_build_base_rows(
    dataset_row: pd.Series,
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    assignments: pd.DataFrame,
    asset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    ensure_dir(asset_dir)
    base_path = asset_dir / "fixed_negative_only_base_rows.csv"
    audit_path = asset_dir / "fixed_negative_only_sampling_audit.csv"
    summary_path = asset_dir / "fixed_negative_only_sampling_summary.csv"
    metadata_path = asset_dir / "fixed_negative_only_metadata.json"
    expected = _expected_metadata(dataset_row, episodes, machine_master, assignments)

    if base_path.exists() and metadata_path.exists():
        observed = json.loads(metadata_path.read_text())
        if observed != expected:
            raise ValueError(
                "The locked negative-only cohort no longer matches the current configuration. "
                f"Delete {asset_dir} only when a new cohort is intentionally required."
            )
        return (
            _read_csv(base_path),
            _read_csv(audit_path) if audit_path.exists() else pd.DataFrame(),
            _read_csv(summary_path) if summary_path.exists() else pd.DataFrame(),
            True,
        )

    window_config = {
        "lead_max_days": int(dataset_row["lead_max_days"]),
        "lead_min_days": int(dataset_row["lead_min_days"]),
    }
    base, audit, summary = build_negative_only_base_rows(
        episodes,
        machine_master,
        sources,
        assignments,
        window_config,
    )
    base.to_csv(base_path, index=False)
    audit.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(expected, indent=2, default=str))
    return base, audit, summary, False


def _score_distribution_summary(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    group_columns = ["dataset_id", "algorithm", "split", "negative_case_type", "delayed_claim_lead_bin"]
    rows: list[dict] = []
    groupers = [column for column in group_columns if column in scored.columns]
    for keys, group in scored.groupby(groupers, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupers, keys))
        scores = pd.to_numeric(group["risk_score"], errors="coerce")
        row.update({
            "rows": int(len(group)),
            "unique_machines": int(group["machine_key"].nunique()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "minimum_score": float(scores.min()),
            "p01_score": float(scores.quantile(0.01)),
            "p05_score": float(scores.quantile(0.05)),
            "p10_score": float(scores.quantile(0.10)),
            "p25_score": float(scores.quantile(0.25)),
            "median_score": float(scores.median()),
            "p75_score": float(scores.quantile(0.75)),
            "p90_score": float(scores.quantile(0.90)),
            "p95_score": float(scores.quantile(0.95)),
            "p99_score": float(scores.quantile(0.99)),
            "maximum_score": float(scores.max()),
            "predicted_positive_rows": int((scores >= threshold).sum()),
            "predicted_positive_rate": float((scores >= threshold).mean()),
            "mean_squared_score_against_zero": float(np.mean(np.square(scores))),
        })
        for review_threshold in _review_thresholds():
            label = str(review_threshold).replace(".", "p")
            row[f"rows_score_ge_{label}"] = int((scores >= review_threshold).sum())
            row[f"rate_score_ge_{label}"] = float((scores >= review_threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _histogram_table(scored: pd.DataFrame) -> pd.DataFrame:
    bins = max(2, int(getattr(config, "NEGATIVE_ONLY_SCORE_HISTOGRAM_BINS", 20)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for (dataset_id, algorithm, split_name, case_type), group in scored.groupby(
        ["dataset_id", "algorithm", "split", "negative_case_type"],
        dropna=False,
        sort=True,
    ):
        scores = pd.to_numeric(group["risk_score"], errors="coerce").dropna().to_numpy()
        counts, _ = np.histogram(scores, bins=edges)
        for index, count in enumerate(counts):
            rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "negative_case_type": case_type,
                "score_bin_lower": float(edges[index]),
                "score_bin_upper": float(edges[index + 1]),
                "rows": int(count),
                "fraction_within_group": float(count / len(scores)) if len(scores) else np.nan,
            })
    return pd.DataFrame(rows)


def _horizon_outcome_summary(scored: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset_id, algorithm, split_name, case_type), group in scored.groupby(
        ["dataset_id", "algorithm", "split", "negative_case_type"],
        dropna=False,
        sort=True,
    ):
        for horizon in horizons:
            column = f"eval_target_claim_within_next_{int(horizon)}d"
            y = pd.to_numeric(group[column], errors="coerce").fillna(0).astype(int)
            scores = pd.to_numeric(group["risk_score"], errors="coerce")
            positive_scores = scores[y.eq(1)]
            negative_scores = scores[y.eq(0)]
            rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "negative_case_type": case_type,
                "evaluation_horizon_days": int(horizon),
                "rows": int(len(group)),
                "future_claim_positive_rows": int(y.sum()),
                "future_claim_positive_rate": float(y.mean()),
                "mean_score_future_claim_positive": float(positive_scores.mean()) if len(positive_scores) else np.nan,
                "median_score_future_claim_positive": float(positive_scores.median()) if len(positive_scores) else np.nan,
                "mean_score_still_negative": float(negative_scores.mean()) if len(negative_scores) else np.nan,
                "median_score_still_negative": float(negative_scores.median()) if len(negative_scores) else np.nan,
            })
    return pd.DataFrame(rows)


def _save_histogram_plot(scored: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    bins = max(2, int(getattr(config, "NEGATIVE_ONLY_SCORE_HISTOGRAM_BINS", 20)))
    for case_type, group in scored.groupby("negative_case_type", sort=True):
        plt.hist(
            pd.to_numeric(group["risk_score"], errors="coerce").dropna(),
            bins=bins,
            range=(0, 1),
            alpha=0.45,
            label=str(case_type),
        )
    plt.xlabel("Model risk score")
    plt.ylabel("Negative machine snapshots")
    plt.title("Negative-only risk-score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run(
    dataset_index_path: str | Path | None = None,
    step_dir: str | Path | None = None,
) -> None:
    config.refresh_derived_config()
    if not bool(getattr(config, "NEGATIVE_ONLY_EVALUATION_ENABLED", True)):
        print("12_negative_only_score_diagnostics skipped: disabled in config.", flush=True)
        return

    output_dir = (
        Path(step_dir)
        if step_dir is not None
        else config.OUTPUT_DIR / "12_negative_only_score_diagnostics"
    )
    ensure_dir(output_dir)
    dataset_index = (
        pd.read_csv(Path(dataset_index_path), low_memory=False)
        if dataset_index_path is not None
        else _load_dataset_index()
    )
    episodes = _load_episodes()
    print("Loading sources for negative-only score diagnostics...", flush=True)
    sources = load_sources(config)
    machine_master = build_machine_master(sources)
    threshold = float(getattr(config, "NEGATIVE_ONLY_SCORE_THRESHOLD", 0.50))

    all_predictions: list[pd.DataFrame] = []
    all_score_summaries: list[pd.DataFrame] = []
    all_histograms: list[pd.DataFrame] = []
    all_horizon_summaries: list[pd.DataFrame] = []
    all_sampling_summaries: list[pd.DataFrame] = []
    run_summaries: list[dict] = []

    for _, dataset_row in dataset_index.iterrows():
        dataset_id = str(dataset_row["dataset_id"])
        lead_min = int(dataset_row["lead_min_days"])
        horizons = _configured_horizons(lead_min)
        print(f"Building/scoring negative-only cohort: {dataset_id}", flush=True)
        assignments = pd.read_csv(
            Path(str(dataset_row["fixed_machine_split_assignment_path"])),
            low_memory=False,
        )
        asset_dir = Path(config.FIXED_SPLIT_ASSET_DIR) / dataset_id / "negative_only"
        base_rows, audit, sampling_summary, reused = _load_or_build_base_rows(
            dataset_row,
            episodes,
            machine_master,
            sources,
            assignments,
            asset_dir,
        )
        base_rows.to_csv(output_dir / f"{dataset_id}__negative_only_base_rows.csv", index=False)
        audit.to_csv(output_dir / f"{dataset_id}__negative_only_sampling_audit.csv", index=False)
        sampling_summary["dataset_id"] = dataset_id
        all_sampling_summaries.append(sampling_summary)

        print(
            f"  Engineering {config.FEATURE_SET} features for {len(base_rows):,} negative snapshots...",
            flush=True,
        )
        negative_df = build_window_features(
            base_rows,
            sources=sources,
            episodes=episodes,
            config=config,
        )
        negative_df = annotate_future_claim_outcomes(
            negative_df,
            claim_history_episodes=episodes,
            config=config,
            horizons=horizons,
        )
        validate_dataset_features(negative_df, config)
        negative_df.to_csv(
            output_dir / f"{dataset_id}__negative_only_feature_dataset.csv", index=False
        )

        train_df = _read_csv(Path(str(dataset_row["training_dataset_path"])))
        validate_path = Path(str(dataset_row["validation_dataset_path"]))
        validation_df = _read_csv(validate_path) if validate_path.exists() else train_df
        validate_dataset_features(train_df, config)
        validate_dataset_features(validation_df, config)
        feature_columns = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
        X_train = train_df[feature_columns]
        y_train = pd.to_numeric(train_df["target"], errors="raise").astype(int)
        X_eval = validation_df[feature_columns]
        y_eval = pd.to_numeric(validation_df["target"], errors="raise").astype(int)

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
                X_eval=X_eval,
                y_eval=y_eval,
                eval_name="standard_validation_for_negative_only_fit",
            )
            score = predict_score(model, negative_df[feature_columns], algorithm)
            scored = negative_df.copy().reset_index(drop=True)
            scored["dataset_id"] = dataset_id
            scored["algorithm"] = algorithm
            scored["risk_score"] = np.asarray(score, dtype=float)
            scored["predicted_label"] = (scored["risk_score"] >= threshold).astype(int)
            scored["true_label_design_horizon"] = 0
            scored["prediction_correct_design_horizon"] = scored["predicted_label"].eq(0).astype(int)
            scored["score_rank_within_split"] = (
                scored.groupby("split", sort=False)["risk_score"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            for horizon in horizons:
                column = f"eval_target_claim_within_next_{int(horizon)}d"
                scored[f"true_label_{int(horizon)}d"] = pd.to_numeric(
                    scored[column], errors="coerce"
                ).fillna(0).astype(int)
                scored[f"prediction_correct_{int(horizon)}d"] = (
                    scored["predicted_label"].eq(scored[f"true_label_{int(horizon)}d"])
                ).astype(int)

            prediction_path = output_dir / (
                f"{dataset_id}__{algorithm}__negative_only_machine_predictions.csv"
            )
            scored.to_csv(prediction_path, index=False)
            all_predictions.append(scored)
            all_score_summaries.append(_score_distribution_summary(scored, threshold))
            all_histograms.append(_histogram_table(scored))
            all_horizon_summaries.append(_horizon_outcome_summary(scored, horizons))
            _save_histogram_plot(
                scored,
                output_dir / f"{dataset_id}__{algorithm}__negative_only_score_histogram.png",
            )
            run_summaries.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "training_rows": int(len(train_df)),
                "negative_only_rows": int(len(scored)),
                "strict_negative_rows": int(scored["negative_case_type"].eq("strict_negative").sum()),
                "delayed_claim_negative_rows": int(scored["negative_case_type"].eq("delayed_claim_negative").sum()),
                "predicted_positive_rows": int(scored["predicted_label"].sum()),
                "predicted_positive_rate": float(scored["predicted_label"].mean()),
                "mean_risk_score": float(scored["risk_score"].mean()),
                "median_risk_score": float(scored["risk_score"].median()),
                "locked_base_rows_reused": bool(reused),
                "fit_metadata": fit_metadata,
            })

    def concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        kept = [frame for frame in frames if frame is not None and not frame.empty]
        return pd.concat(kept, ignore_index=True, sort=False) if kept else pd.DataFrame()

    outputs = {
        "negative_only_machine_predictions_all.csv": concat(all_predictions),
        "negative_only_score_distribution_summary.csv": concat(all_score_summaries),
        "negative_only_score_histogram.csv": concat(all_histograms),
        "negative_only_horizon_outcome_summary.csv": concat(all_horizon_summaries),
        "negative_only_sampling_summary.csv": concat(all_sampling_summaries),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    write_json(
        {
            "step": "12_negative_only_score_diagnostics",
            "output_dir": str(output_dir),
            "evaluation_splits": _configured_splits(),
            "sample_size_per_split": _sample_size(),
            "strict_fraction": _strict_fraction(),
            "strict_no_claim_days": int(
                getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180)
            ),
            "score_threshold": threshold,
            "run_summaries": run_summaries,
            "notes": [
                "Every row is negative at lead_min_days.",
                "Strict rows have no claim through the full configured exclusion period.",
                "Delayed rows claim after lead_min_days and are distributed across lead-time bins.",
                "Prediction outputs include actual next claim date, days to claim, score, and horizon labels.",
            ],
        },
        output_dir / "run_summary.json",
    )
    print(f"12_negative_only_score_diagnostics completed. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    run()
