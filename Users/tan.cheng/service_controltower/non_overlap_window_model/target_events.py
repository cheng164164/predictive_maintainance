"""Target-event loaders for warranty and physical-failure experiments.

Both loaders return one row per machine and physical event date with the common
columns ``machine_key``, ``event_date``, and ``target_source``.  The comparison
runner can therefore switch targets without changing feature, split, model, or
evaluation logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from data_utils import clean_text, make_machine_key

TargetSource = Literal["warranty", "physical_failure"]
WarrantyFilterMode = Literal["all_rows_machine_day", "original_script_cleaned"]


def _finalize_events(frame: pd.DataFrame, target_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["machine_key", "event_date"]).copy()
    raw_valid_rows = len(work)
    # A machine-day is the event unit used by the rolling-origin reference run.
    work = (
        work.sort_values(["machine_key", "event_date"], kind="mergesort")
        .drop_duplicates(["machine_key", "event_date"], keep="first")
        .reset_index(drop=True)
    )
    work["target_source"] = target_source
    summary = pd.DataFrame(
        [
            ("target_source", target_source),
            ("valid_event_rows_before_machine_day_dedup", raw_valid_rows),
            ("unique_machine_day_events", len(work)),
            ("machines_with_events", int(work["machine_key"].nunique())),
            ("first_event_date", work["event_date"].min()),
            ("last_event_date", work["event_date"].max()),
        ],
        columns=["metric", "value"],
    )
    return work[["machine_key", "event_date", "target_source"]], summary


def load_warranty_events(
    path: Path,
    *,
    filter_mode: WarrantyFilterMode = "all_rows_machine_day",
    allowed_claim_types: tuple[str, ...] = (),
    minimum_failure_smr: float = 25.0,
    invalid_part_codes: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the original warranty CSV as a generic target event table.

    ``all_rows_machine_day`` matches the prior rolling-origin comparison: every
    valid dated warranty row is eligible, with duplicate machine dates collapsed.

    ``original_script_cleaned`` reproduces the uploaded future-claim script's
    claim-type, failed-part, and minimum-SMR filters before machine-day collapse.
    """

    df = pd.read_csv(path, dtype=str, low_memory=False)
    required = {
        "local_date",
        "full_model",
        "serial",
        "claim_type_description",
        "failure_smr",
        "critical_fail_part_number",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Warranty file is missing required columns: {missing}")

    df["event_date"] = pd.to_datetime(df["local_date"], errors="coerce")
    df["machine_key"] = make_machine_key(df["full_model"], df["serial"])
    raw_rows = len(df)

    if filter_mode == "original_script_cleaned":
        part = clean_text(df["critical_fail_part_number"]).fillna("")
        failure_smr = pd.to_numeric(df["failure_smr"], errors="coerce")
        type_ok = df["claim_type_description"].isin(allowed_claim_types)
        part_ok = ~part.isin(invalid_part_codes) & ~part.str.fullmatch(r"0+", na=False)
        smr_ok = failure_smr.ge(float(minimum_failure_smr))
        keep = type_ok & part_ok & smr_ok & df["event_date"].notna() & df["machine_key"].notna()
        df = df[keep].copy()
    elif filter_mode == "all_rows_machine_day":
        df = df[df["event_date"].notna() & df["machine_key"].notna()].copy()
    else:
        raise ValueError(
            "filter_mode must be 'all_rows_machine_day' or 'original_script_cleaned'; "
            f"got {filter_mode!r}."
        )

    events, summary = _finalize_events(df, "warranty")
    extra = pd.DataFrame(
        [
            ("raw_warranty_rows", raw_rows),
            ("warranty_filter_mode", filter_mode),
            ("rows_after_configured_filter", len(df)),
        ],
        columns=["metric", "value"],
    )
    return events, pd.concat([extra, summary], ignore_index=True)


def load_physical_failure_events(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the unified physical-failure table as generic target events."""

    df = pd.read_csv(path, low_memory=False)
    required = {"machine", "event_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Physical-failure file is missing required columns: {missing}")

    machine = df["machine"].astype("string").str.strip().str.upper()
    parsed = machine.str.extract(r"^(?P<full_model>.+)-(?P<serial>[^-]+)$")
    df["machine_key"] = make_machine_key(parsed["full_model"], parsed["serial"])
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    raw_rows = len(df)
    events, summary = _finalize_events(df, "physical_failure")
    extra = pd.DataFrame(
        [("raw_physical_failure_rows", raw_rows)], columns=["metric", "value"]
    )
    return events, pd.concat([extra, summary], ignore_index=True)


def load_target_events(
    target_source: TargetSource,
    *,
    warranty_path: Path,
    physical_failure_path: Path,
    warranty_filter_mode: WarrantyFilterMode = "all_rows_machine_day",
    allowed_claim_types: tuple[str, ...] = (),
    minimum_failure_smr: float = 25.0,
    invalid_part_codes: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one configured target source through the common event schema."""

    if target_source == "warranty":
        return load_warranty_events(
            warranty_path,
            filter_mode=warranty_filter_mode,
            allowed_claim_types=allowed_claim_types,
            minimum_failure_smr=minimum_failure_smr,
            invalid_part_codes=invalid_part_codes,
        )
    if target_source == "physical_failure":
        return load_physical_failure_events(physical_failure_path)
    raise ValueError(
        "target_source must be 'warranty' or 'physical_failure'; "
        f"got {target_source!r}."
    )
