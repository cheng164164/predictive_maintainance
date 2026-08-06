"""Reusable pandas-to-snapshot bridge for Palantir feature preparation.

The local snapshot builder reads CSV paths. This bridge keeps that tested logic
unchanged by materializing Foundry pandas inputs in a temporary directory and
then calling the same ``load_sources`` and ``build_snapshot_dataframe``
functions used by local development and deployment.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd

import config
from snapshot_builder import SourceFileSet, build_snapshot_dataframe, load_sources


def _normalize_roster(roster: pd.DataFrame) -> tuple[str, ...]:
    """Return normalized machine keys from a Foundry roster dataframe."""
    if "machine_key" in roster.columns:
        values = roster["machine_key"].dropna().astype(str).str.strip()
        return tuple(values[values.ne("")].drop_duplicates().tolist())

    candidates = (
        ("full_model", "SERIAL"),
        ("FULL_MODEL", "SERIAL"),
        ("full_model", "serial_number"),
    )
    for model_column, serial_column in candidates:
        if {model_column, serial_column}.issubset(roster.columns):
            model = roster[model_column].astype("string").str.strip().str.upper()
            serial = roster[serial_column].astype("string").str.extract(
                r"(\d+)", expand=False
            )
            keys = (model + "-" + serial).dropna().drop_duplicates()
            return tuple(keys.tolist())
    raise ValueError(
        "Machine roster must contain machine_key or a supported model/serial pair."
    )


def infer_score_date(operation: pd.DataFrame, override: str | None = None) -> pd.Timestamp:
    """Return an explicit score date or the latest valid operation date."""
    if override:
        score_date = pd.Timestamp(override).normalize()
    else:
        if "LOCAL_DATE" not in operation.columns:
            raise ValueError("Operation input is missing LOCAL_DATE.")
        parsed = pd.to_datetime(operation["LOCAL_DATE"], errors="coerce", format="mixed")
        if not parsed.notna().any():
            raise ValueError("Operation input contains no valid LOCAL_DATE values.")
        score_date = pd.Timestamp(parsed.max()).normalize()
    if pd.isna(score_date):
        raise ValueError("A valid Palantir scoring date could not be determined.")
    return score_date


def build_feature_snapshot_from_frames(
    source_frames: Mapping[str, pd.DataFrame],
    machine_roster: pd.DataFrame,
    score_date: pd.Timestamp | str,
) -> pd.DataFrame:
    """Build one production feature snapshot from Foundry pandas dataframes.

    Required ``source_frames`` keys are ``fault``, ``fluid``, ``maintenance``,
    ``operation``, and ``target``. Inputs should already respect the retention
    requirements in ``config.SCORING_HISTORY_RETENTION_DAYS``. The target input
    is historical-only and is used solely for prior-event features.
    """
    required = {"fault", "fluid", "maintenance", "operation", "target"}
    missing = sorted(required.difference(source_frames))
    if missing:
        raise ValueError(f"Missing Palantir source frames: {missing}")

    normalized_date = pd.Timestamp(score_date).normalize()
    roster_keys = _normalize_roster(machine_roster)
    if not roster_keys:
        raise ValueError("The Palantir machine roster is empty.")

    with tempfile.TemporaryDirectory(prefix="machine_risk_palantir_") as temp_dir:
        root = Path(temp_dir)
        paths: dict[str, Path] = {}
        for key in sorted(required):
            frame = source_frames[key]
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"source_frames[{key!r}] must be a pandas DataFrame.")
            path = root / f"{key}.csv"
            frame.to_csv(path, index=False)
            paths[key] = path

        source_files = SourceFileSet(
            fault=paths["fault"],
            fluid=paths["fluid"],
            maintenance=paths["maintenance"],
            operation=paths["operation"],
            target=paths["target"],
        )
        sources = load_sources(
            include_operation_features=True,
            source_files=source_files,
            candidate_serious_codes=tuple(config.PRODUCTION_SELECTED_FAULT_CODES),
            machine_roster=roster_keys,
        )
        snapshot = build_snapshot_dataframe(
            sources,
            [normalized_date],
            include_targets=False,
            verbose=False,
        )
    return snapshot
