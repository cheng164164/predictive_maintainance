"""Load either target source into one standardized event schema."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config


@dataclass(frozen=True)
class TargetEventTables:
    """Standardized raw events, machine-day events, and source profile metadata."""

    raw: pd.DataFrame
    by_machine_day: pd.DataFrame
    profile: dict


def normalize_machine_key(series: pd.Series) -> pd.Series:
    """Normalize machine identifiers to the canonical ``MODEL-SERIAL`` form."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+(\d+)$", r"-\1", regex=True)
    )


def _load_physical_failure(path: Path) -> pd.DataFrame:
    """Load physical-failure events and map them to the standard event schema."""
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"machine", "event_date"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    out = raw.copy()
    out["machine_key"] = normalize_machine_key(out["machine"])
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    if "event_source" not in out.columns:
        out["event_source"] = "PHYSICAL_FAILURE"
    out["event_source"] = out["event_source"].astype("string").fillna("UNKNOWN")
    if "title" in out.columns:
        out["event_description"] = out["title"].astype("string")
    else:
        out["event_description"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out


def _load_warranty(path: Path) -> pd.DataFrame:
    """Load warranty claims and map them to the standard event schema."""
    raw = pd.read_csv(path, low_memory=False)
    required = {"machine_id", "local_date"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    out = raw.copy()
    out["machine_key"] = normalize_machine_key(out["machine_id"])
    out["event_date"] = pd.to_datetime(out["local_date"], errors="coerce")
    out["event_source"] = "WARRANTY"
    if "claim_number" in out.columns:
        out["event_id"] = out["claim_number"].astype("string")
    if "claim_type_description" in out.columns:
        out["event_description"] = out["claim_type_description"].astype("string")
    else:
        out["event_description"] = pd.Series(pd.NA, index=out.index, dtype="string")
    return out


def load_target_events(path: Path | None = None) -> TargetEventTables:
    """Load the configured target source from ``path`` or ``config.TARGET_FILE``.

    Parameters
    ----------
    path:
        Optional target-source override used by the production incoming-data
        pipeline after it prepares a merged historical source bundle.

    Returns
    -------
    TargetEventTables
        Raw standardized events, one-row-per-machine-day events, and a source
        profile suitable for audit logging.
    """
    path = Path(path or config.TARGET_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Target source file not found for TARGET_SOURCE={config.TARGET_SOURCE!r}: {path}"
        )
    if config.TARGET_SOURCE == "physical_failure":
        raw = _load_physical_failure(path)
    elif config.TARGET_SOURCE == "warranty":
        raw = _load_warranty(path)
    else:  # guarded by config, kept as defensive programming
        raise ValueError(f"Unsupported target source: {config.TARGET_SOURCE}")

    source_rows = int(len(raw))
    raw = raw.dropna(subset=["machine_key", "event_date"]).copy()
    raw = raw.sort_values(
        ["machine_key", "event_date", "event_source"], kind="mergesort"
    ).reset_index(drop=True)
    by_day = raw.drop_duplicates(["machine_key", "event_date"]).copy()
    profile = {
        "target_source": config.TARGET_SOURCE,
        "target_display_name": config.TARGET_DISPLAY_NAME,
        "target_file": str(path),
        "target_raw_rows": source_rows,
        "target_usable_rows": int(len(raw)),
        "target_unique_machine_days": int(len(by_day)),
        "target_unique_machines": int(by_day["machine_key"].nunique()),
        "target_date_min": str(by_day["event_date"].min().date()) if len(by_day) else None,
        "target_date_max": str(by_day["event_date"].max().date()) if len(by_day) else None,
        "target_source_counts": {
            str(key): int(value)
            for key, value in raw["event_source"].value_counts(dropna=False).items()
        },
    }
    return TargetEventTables(raw=raw, by_machine_day=by_day, profile=profile)
