"""Shared file I/O and selected-target helpers for the numbered pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from target_events import load_target_events


def load_selected_target_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_target_events(
        config.TARGET_SOURCE,
        warranty_path=config.WARRANTY_PATH,
        physical_failure_path=config.PHYSICAL_FAILURE_PATH,
        warranty_filter_mode=config.WARRANTY_FILTER_MODE,
        allowed_claim_types=config.WARRANTY_ALLOWED_CLAIM_TYPES,
        minimum_failure_smr=config.WARRANTY_MIN_FAILURE_SMR,
        invalid_part_codes=config.WARRANTY_INVALID_PART_CODES,
    )


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_split_definition() -> dict[str, list[pd.Timestamp]]:
    payload = json.loads(config.SPLIT_DEFINITION_PATH.read_text(encoding="utf-8"))
    return {
        name: [pd.Timestamp(value) for value in values]
        for name, values in payload["splits"].items()
    }
