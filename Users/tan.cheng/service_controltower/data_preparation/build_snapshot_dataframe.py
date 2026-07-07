"""
Build leakage-safe source-level and unified snapshot dataframes for predictive maintenance.

Current supported source files:
    1. machine.csv          canonical model_id + snapshot_date backbone
    2. fault_codes.csv      fault/event history
    3. maintenance.csv      maintenance-monitor / PM history

Designed for future extension:
    - oil / fluid sample data
    - warranty data / target labels
    - service/work-order data

Expected project layout:

    service_controltower/
    ├── data_preparation/
    │   ├── build_snapshot_dataframe.py
    │   ├── config.py
    │   └── output/
    ├── enriched_data/
    │   ├── machine.csv
    │   ├── fault_codes.csv
    │   └── maintenance.csv
    └── requirements.txt

Output grain:
    One row per model_id / snapshot_date.

Important design choice:
    machine.csv is the canonical snapshot backbone. All source-specific snapshot
    tables must follow the same model_id + snapshot_date rows from machine.csv.
    Source tables do not create their own snapshot calendars.

Core leakage-control rule:
    Features only use source records with event_date < snapshot_date.
    Target labels, when warranty data is added later, should only use dates after
    snapshot_date and on/before snapshot_date + prediction_horizon.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Optional run configuration
# -----------------------------------------------------------------------------
try:
    import config as run_config
except ImportError:  # pragma: no cover
    run_config = None


def cfg(name: str, default):
    """Read a value from config.py, or return a built-in default."""
    return getattr(run_config, name, default) if run_config is not None else default


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(cfg("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
INPUT_DIR = Path(cfg("INPUT_DIR", PROJECT_ROOT / "enriched_data")).resolve()
OUTPUT_DIR = Path(cfg("OUTPUT_DIR", PROJECT_ROOT / "data_preparation" / "output")).resolve()
SOURCE_SNAPSHOT_DIR = Path(cfg("SOURCE_SNAPSHOT_DIR", OUTPUT_DIR / "source_snapshots")).resolve()

TARGET_MODEL_FAMILIES = tuple(
    str(x).upper().strip() for x in cfg("TARGET_MODEL_FAMILIES", ("D51", "D61", "D71"))
)

MODEL_ID_CANDIDATE_COLUMNS = tuple(
    cfg("MODEL_ID_CANDIDATE_COLUMNS", ("model_id", "machine_id", "MACHINE_ID", "Machine_ID"))
)

MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS = tuple(
    cfg(
        "MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS",
        ("snapshot_date", "SNAPSHOT_DATE", "as_of_date", "AS_OF_DATE", "snapshot_dt"),
    )
)

ALLOW_MODEL_ID_FALLBACK = bool(cfg("ALLOW_MODEL_ID_FALLBACK", True))
PROGRESS_EVERY_MACHINES = int(cfg("PROGRESS_EVERY_MACHINES", 100))


# -----------------------------------------------------------------------------
# Frozen model feature list
# -----------------------------------------------------------------------------
FROZEN_FEATURES = [
    "fault_count_7d",
    "fault_count_30d",
    "fault_count_90d",
    "fault_count_previous_30d",
    "fault_growth_rate",
    "days_since_last_fault",
    "days_since_last_severe_fault",
    "faults_per_100_hours",
    "unique_fault_code_count_90d",
    "repeat_fault_ratio_90d",
    "unique_component_count_90d",
    "mechanical_fault_count_90d",
    "mechanical_fault_count_30d",
    "electrical_fault_count_90d",
    "electrical_fault_count_30d",
    "action_L01_count_90d",
    "action_L02_count_90d",
    "action_L03_count_90d",
    "action_L04_count_90d",
    "max_action_level_90d",
    "sum_log_occurrence_90d",
    "max_log_occurrence_90d",
    "occurrence_severity_score_90d",
    "strong_fault_count_90d",
    "moderate_fault_count_90d",
    "max_event_evidence_score_90d",
    "avg_event_evidence_score_90d",
    "max_context_evidence_score_90d",
    "engine_fault_count_90d",
    "hydraulic_fault_count_90d",
    "powertrain_fault_count_90d",
    "scr_fault_count_90d",
    "workequipment_fault_count_90d",
    "cooling_fault_count_90d",
    "top_component_fault_ratio_90d",
    "maintenance_events_180d",
    "monitor_reset_count_180d",
    "maintenance_reset_ratio_180d",
    "maintenance_events_90d",
    "monitor_reset_count_90d",
    "active_maintenance_items",
    "overdue_item_count",
    "due_now_item_count",
    "maintenance_due_or_overdue_ratio",
    "avg_remaining_hours",
    "min_remaining_hours",
    "engine_reset_count_180d",
    "transmission_reset_count_180d",
    "final_drive_reset_count_180d",
    "cooling_system_reset_count_180d",
    "urea_scr_system_reset_count_180d",
    "engine_overdue_item_count",
    "transmission_overdue_item_count",
    "final_drive_overdue_item_count",
    "cooling_system_overdue_item_count",
    "urea_scr_system_overdue_item_count",
    "oil_reset_count_180d",
    "filter_reset_count_180d",
    "breather_reset_count_180d",
    "coolant_reset_count_180d",
    "unique_maintenance_type_count_180d",
    "days_since_last_reset",
    "days_since_last_oil_reset",
    "days_since_last_filter_reset",
    "smr_since_last_reset",
]

COUNT_FEATURES = [
    c
    for c in FROZEN_FEATURES
    if c.endswith("_count_90d")
    or c.endswith("_count_30d")
    or c.endswith("_count_180d")
    or c.endswith("_7d")
    or c
    in {
        "fault_count_previous_30d",
        "maintenance_events_180d",
        "maintenance_events_90d",
        "active_maintenance_items",
        "overdue_item_count",
        "due_now_item_count",
        "unique_fault_code_count_90d",
        "unique_component_count_90d",
        "unique_maintenance_type_count_180d",
        "strong_fault_count_90d",
        "moderate_fault_count_90d",
    }
]

RECENCY_FEATURES = [
    "days_since_last_fault",
    "days_since_last_severe_fault",
    "days_since_last_reset",
    "days_since_last_oil_reset",
    "days_since_last_filter_reset",
]

NULL_LIKE_STRINGS = {
    "",
    " ",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "#n/a",
    "#na",
    "not available",
    "unknown",
    "undefined",
    "<na>",
}

COMPONENT_PATTERNS = {
    "engine": ["engine"],
    "hydraulic": ["hydraulic"],
    "powertrain": ["power train", "powertrain", "transmission", "hst"],
    "scr": ["scr", "urea", "adblue", "def"],
    "workequipment": ["work equipment", "workequipment"],
    "cooling": ["cooling", "coolant", "radiator"],
    "final_drive": ["final drive", "final_drive"],
}

MAINTENANCE_TYPE_PATTERNS = {
    "oil": ["oil"],
    "filter": ["filter"],
    "breather": ["breather"],
    "coolant": ["coolant"],
}


# -----------------------------------------------------------------------------
# Console progress helper
# -----------------------------------------------------------------------------
def progress(message: str) -> None:
    """Print a short progress message immediately."""
    print(f"[snapshot-build] {message}", flush=True)


# -----------------------------------------------------------------------------
# File/path helpers
# -----------------------------------------------------------------------------
def resolve_existing_path(path_value: str | Path, label: str) -> Path:
    """Resolve and validate a required input path."""
    path = Path(path_value).expanduser()
    candidates = [path]

    if not path.is_absolute():
        candidates.append((PROJECT_ROOT / path).resolve())

    candidates.append((INPUT_DIR / path.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n  - {checked}")


def optional_existing_path(path_value: Optional[str | Path]) -> Optional[Path]:
    """Return an optional path only when it exists."""
    if path_value in (None, "", "None"):
        return None
    try:
        return resolve_existing_path(path_value, "optional input")
    except FileNotFoundError:
        return None


# -----------------------------------------------------------------------------
# Data cleaning and missing-value reporting
# -----------------------------------------------------------------------------
def dedupe_column_names(columns: Iterable[object]) -> list[str]:
    """Strip whitespace from column names and make duplicate names unique."""
    seen: dict[str, int] = {}
    cleaned_columns: list[str] = []
    for col in columns:
        base = str(col).strip() or "unnamed_column"
        count = seen.get(base, 0) + 1
        seen[base] = count
        cleaned_columns.append(base if count == 1 else f"{base}_duplicate_{count}")
    return cleaned_columns


def blank_or_null_string_count(series: pd.Series) -> int:
    """Count visually blank or null-like string values."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return 0
    text = series.astype("string").str.strip().str.lower()
    return int(text.isin(NULL_LIKE_STRINGS).sum())


def sample_non_missing_values(series: pd.Series, max_values: int = 3) -> str:
    examples = series.dropna().astype(str).head(max_values).tolist()
    return " | ".join(examples)


def build_missing_profile(df: pd.DataFrame, dataset_name: str, stage: str) -> pd.DataFrame:
    """Build a column-level missingness report."""
    row_count = len(df)
    rows: list[dict] = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "dataset": dataset_name,
                "stage": stage,
                "column": col,
                "dtype": str(df[col].dtype),
                "row_count": row_count,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / row_count) * 100, 4) if row_count else 0.0,
                "blank_or_null_string_count": blank_or_null_string_count(df[col]),
                "non_missing_count": int(row_count - missing_count),
                "unique_non_missing_count": int(df[col].dropna().nunique()) if row_count else 0,
                "example_non_missing_values": sample_non_missing_values(df[col]),
            }
        )
    return pd.DataFrame(rows)


def clean_raw_dataframe(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Lightly clean one raw CSV dataframe and return audit artifacts."""
    rows_before = len(df)
    cols_before = len(df.columns)
    raw_profile = build_missing_profile(df, dataset_name=dataset_name, stage="raw")

    cleaned = df.copy()
    cleaned.columns = dedupe_column_names(cleaned.columns)

    object_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype("string").str.strip()
        cleaned[col] = cleaned[col].mask(cleaned[col].str.lower().isin(NULL_LIKE_STRINGS), pd.NA)

    fully_empty_cols = cleaned.columns[cleaned.isna().all()].tolist()
    if fully_empty_cols:
        cleaned = cleaned.drop(columns=fully_empty_cols)

    cleaned = cleaned.dropna(how="all")
    cleaned_profile = build_missing_profile(cleaned, dataset_name=dataset_name, stage="cleaned")

    summary = {
        "dataset": dataset_name,
        "rows_before": rows_before,
        "rows_after_light_cleaning": len(cleaned),
        "rows_dropped_fully_empty": rows_before - len(cleaned),
        "columns_before": cols_before,
        "columns_after_light_cleaning": len(cleaned.columns),
        "columns_dropped_fully_empty": len(fully_empty_cols),
        "fully_empty_columns_dropped": ", ".join(fully_empty_cols),
        "missing_cells_raw": int(df.isna().sum().sum()),
        "missing_cells_cleaned": int(cleaned.isna().sum().sum()),
        "blank_or_null_strings_raw": int(raw_profile["blank_or_null_string_count"].sum()) if not raw_profile.empty else 0,
    }
    return cleaned, raw_profile, cleaned_profile, summary


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def load_and_clean_csv(path: str | Path, dataset_name: str, output_dir: str | Path) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    """Read a CSV, run missing-value detection, lightly clean it, and save a profile."""
    raw = read_csv_safely(path)
    cleaned, raw_profile, cleaned_profile, summary = clean_raw_dataframe(raw, dataset_name)
    profiles = [raw_profile, cleaned_profile]

    if bool(cfg("WRITE_CLEANING_REPORTS", True)):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(profiles, ignore_index=True).to_csv(output_dir / f"missing_profile_{dataset_name}.csv", index=False)
    return cleaned, profiles, summary


def write_combined_cleaning_reports(output_dir: str | Path, profiles: list[pd.DataFrame], summaries: list[dict]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(output_dir / "missing_profile_all_files.csv", index=False)
    if summaries:
        pd.DataFrame(summaries).to_csv(output_dir / "cleaning_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Generic dataframe helpers
# -----------------------------------------------------------------------------
def safe_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Return df[col] if present, otherwise a default Series."""
    if col in df.columns:
        return df[col]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    """Return the first available source column from a list."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def first_existing_col_name(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first available column name from a list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_key(s: pd.Series) -> pd.Series:
    """Normalize identifiers such as model_id or serial_number."""
    out = s.astype("string").str.strip()
    return out.mask(out.str.lower().isin(NULL_LIKE_STRINGS), pd.NA)


def parse_dt(s: pd.Series) -> pd.Series:
    """Parse dates safely and normalize timezone handling."""
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def has_target_family(text_value: object) -> bool:
    """Return True when text contains one of the configured target model families."""
    if not TARGET_MODEL_FAMILIES:
        return True
    if pd.isna(text_value):
        return False
    text = str(text_value).upper()
    for family in TARGET_MODEL_FAMILIES:
        if re.search(rf"\b{re.escape(family)}", text):
            return True
    return False


def normalize_model_id(df: pd.DataFrame, source_name: str) -> tuple[pd.Series, str]:
    """Create the canonical model_id used for all joins."""
    for col in MODEL_ID_CANDIDATE_COLUMNS:
        if col in df.columns:
            return normalize_key(df[col]), col

    if not ALLOW_MODEL_ID_FALLBACK:
        return pd.Series(pd.NA, index=df.index, dtype="string"), "missing"

    full_model = first_existing_col(df, ["full_model", "FULL_MODEL", "MODEL", "model", "ZZMATNR"], pd.NA)
    serial = first_existing_col(df, ["serial_number", "SERIAL", "Serial", "ZZSERNR"], pd.NA)
    fallback = full_model.astype("string").str.strip() + " " + serial.astype("string").str.strip()
    fallback = fallback.mask(fallback.str.lower().isin(NULL_LIKE_STRINGS), pd.NA)
    progress(
        f"WARNING: {source_name} has no true model_id/machine_id column. "
        "Using fallback full_model + serial. Add model_id to the source extract when possible."
    )
    return normalize_key(fallback), "fallback_full_model_plus_serial"


def contains_any(series: pd.Series, patterns: Iterable[str]) -> pd.Series:
    """Boolean mask where text contains any keyword pattern."""
    txt = series.fillna("").astype(str).str.lower()
    out = pd.Series(False, index=series.index)
    for pat in patterns:
        out |= txt.str.contains(re.escape(pat.lower()), regex=True, na=False)
    return out


def action_level_to_num(s: pd.Series) -> pd.Series:
    """Convert action levels such as L01, L02, L03 into numeric values."""
    extracted = s.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def ratio(num: float, den: float) -> float:
    """Safe division used for feature ratios."""
    if den is None or pd.isna(den) or den == 0:
        return 0.0
    return float(num) / float(den)


def days_between(snapshot_date: pd.Timestamp, event_date: Optional[pd.Timestamp]) -> float:
    """Days from event_date to snapshot_date."""
    if event_date is None or pd.isna(event_date):
        return np.nan
    return float((snapshot_date - event_date).days)


def boolean_from_mixed_values(series: pd.Series, default: bool = False) -> pd.Series:
    """Convert messy true/false values into booleans."""
    text = series.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    result = text.isin(true_values)
    result = result.mask(text.isin(false_values), False)
    return result.fillna(default).astype(bool)


# -----------------------------------------------------------------------------
# Machine backbone standardization
# -----------------------------------------------------------------------------
def standardize_machine_backbone(machine: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize machine.csv as the canonical model_id + snapshot_date backbone.

    Unlike earlier versions, this function does not create snapshot dates from
    event-source min/max dates. It trusts machine.csv to define the official
    snapshot calendar. Every source snapshot table must later match this exact
    backbone.
    """
    m = machine.copy()
    m["model_id"], model_id_source = normalize_model_id(m, "machine")
    snapshot_date_col = first_existing_col_name(m, MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS)
    if snapshot_date_col is None:
        raise ValueError(
            "machine.csv must contain a snapshot date column. Checked: "
            + ", ".join(MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS)
        )

    m["snapshot_date"] = parse_dt(m[snapshot_date_col]).dt.normalize()
    m["full_model"] = first_existing_col(m, ["full_model", "FULL_MODEL", "MODEL", "model", "ZZMATNR"], pd.NA).astype("string").str.strip()

    total_rows = len(m)
    missing_model_id_rows = int(m["model_id"].isna().sum())
    missing_snapshot_date_rows = int(m["snapshot_date"].isna().sum())

    target_mask = m["full_model"].map(has_target_family) | m["model_id"].map(has_target_family)
    out = m[m["model_id"].notna() & m["snapshot_date"].notna() & target_mask].copy()

    duplicate_key_rows = int(out.duplicated(["model_id", "snapshot_date"]).sum())
    out = out.sort_values(["model_id", "snapshot_date"]).drop_duplicates(["model_id", "snapshot_date"], keep="last")

    # Carry optional machine/master columns into the backbone as machine_* fields.
    # This preserves future static or per-snapshot machine features without
    # colliding with source-level feature names.
    identity_cols = set(MODEL_ID_CANDIDATE_COLUMNS) | set(MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS) | {
        "model_id",
        "snapshot_date",
        "full_model",
        snapshot_date_col,
    }
    base_cols = ["model_id", "snapshot_date", "full_model"]
    extra_cols = [c for c in out.columns if c not in identity_cols]
    rename_map = {c: c if c.startswith("machine_") else f"machine_{c}" for c in extra_cols}
    out = out[base_cols + extra_cols].rename(columns=rename_map)
    out = out.reset_index(drop=True)

    summary = {
        "source": "machine",
        "source_role": "canonical_snapshot_backbone",
        "input_rows": total_rows,
        "model_id_source": model_id_source,
        "snapshot_date_source": snapshot_date_col,
        "missing_model_id_rows": missing_model_id_rows,
        "missing_usable_event_date_rows": missing_snapshot_date_rows,
        "dropped_missing_event_date_rows": missing_snapshot_date_rows,
        "duplicate_model_id_snapshot_rows_removed": duplicate_key_rows,
        "rows_after_standardization": len(out),
        "unique_model_ids_after_standardization": out["model_id"].nunique(),
        "first_snapshot_date": str(out["snapshot_date"].min()) if len(out) else None,
        "last_snapshot_date": str(out["snapshot_date"].max()) if len(out) else None,
    }
    return out, summary


def apply_backbone_filters(backbone: pd.DataFrame) -> pd.DataFrame:
    """Apply date and mini-run filters to the machine-defined backbone."""
    out = backbone.copy()

    min_snapshot_date = cfg("MIN_SNAPSHOT_DATE", None)
    max_snapshot_date = cfg("MAX_SNAPSHOT_DATE", None)
    if min_snapshot_date:
        out = out[out["snapshot_date"] >= pd.to_datetime(min_snapshot_date)]
    if max_snapshot_date:
        out = out[out["snapshot_date"] <= pd.to_datetime(max_snapshot_date)]

    mini_enabled = bool(cfg("MINI_RUN_ENABLED", False))
    if mini_enabled:
        model_ids = [str(x).strip() for x in cfg("MINI_RUN_MODEL_IDS", []) if str(x).strip()]
        if model_ids:
            out = out[out["model_id"].astype(str).isin(model_ids)].copy()
            progress(f"Mini-run enabled using explicit MINI_RUN_MODEL_IDS. Selected {out['model_id'].nunique():,} model_ids.")
        else:
            n = int(cfg("MINI_RUN_MACHINE_COUNT", 3))
            selected = out["model_id"].drop_duplicates().head(n)
            out = out[out["model_id"].isin(selected)].copy()
            progress(f"Mini-run enabled. Selected first {out['model_id'].nunique():,} model_ids from machine backbone.")
    else:
        max_machines = cfg("MAX_MACHINES", None)
        if max_machines is not None:
            selected = out["model_id"].drop_duplicates().head(int(max_machines))
            out = out[out["model_id"].isin(selected)].copy()
            progress(f"MAX_MACHINES limiter active. Selected {out['model_id'].nunique():,} model_ids.")

    out = out.sort_values(["model_id", "snapshot_date"]).reset_index(drop=True)
    validate_backbone(out)
    return out


def validate_backbone(backbone: pd.DataFrame) -> None:
    """Validate that the canonical backbone has one row per model_id/snapshot_date."""
    required = {"model_id", "snapshot_date"}
    missing = required - set(backbone.columns)
    if missing:
        raise ValueError(f"Backbone is missing required columns: {sorted(missing)}")
    if backbone[["model_id", "snapshot_date"]].isna().any().any():
        raise ValueError("Backbone contains missing model_id or snapshot_date values after cleaning.")
    duplicate_count = int(backbone.duplicated(["model_id", "snapshot_date"]).sum())
    if duplicate_count:
        raise ValueError(f"Backbone has duplicate model_id + snapshot_date rows: {duplicate_count:,}")


# -----------------------------------------------------------------------------
# Source standardization
# -----------------------------------------------------------------------------
def standardize_faults(fault: pd.DataFrame, allowed_model_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Convert the raw fault/event table into a standardized event table."""
    f = fault.copy()
    f["model_id"], model_id_source = normalize_model_id(f, "fault_codes")
    f["full_model"] = first_existing_col(f, ["full_model", "FULL_MODEL", "MODEL", "model", "ZZMATNR"], pd.NA).astype("string").str.strip()

    event_time = parse_dt(first_existing_col(f, ["event_time", "UPDATE_DATETIME", "update_datetime"], pd.NaT))
    event_date = parse_dt(first_existing_col(f, ["event_date", "LOCAL_DATE", "local_date"], pd.NaT))
    f["fault_event_date"] = event_time.fillna(event_date)

    f["fault_code_clean"] = first_existing_col(f, ["fault_code", "EVENT_CODE", "event_code", "ERROR_CODE", "error_code"], "").astype("string").str.strip()
    f["event_action_level_clean"] = first_existing_col(f, ["event_action_level", "Action_level", "ACTION_LEVEL", "action_level"], "").astype("string").str.upper().str.strip()
    f["action_level_num_clean"] = pd.to_numeric(first_existing_col(f, ["action_level_num", "ACTION_LEVEL_NUM"], np.nan), errors="coerce")
    f["action_level_num_clean"] = f["action_level_num_clean"].fillna(action_level_to_num(f["event_action_level_clean"]))
    f["occurrence_count_clean"] = pd.to_numeric(first_existing_col(f, ["occurrence_count", "OCCURRENCE_COUNT"], 1), errors="coerce").fillna(1)
    f["log_occurrence_clean"] = pd.to_numeric(first_existing_col(f, ["log_occurrence_count"], np.nan), errors="coerce")
    f["log_occurrence_clean"] = f["log_occurrence_clean"].fillna(np.log1p(f["occurrence_count_clean"]))
    f["occurrence_class_clean"] = pd.to_numeric(first_existing_col(f, ["occurrence_class"], 0), errors="coerce").fillna(0)
    f["smr_hours_clean"] = pd.to_numeric(first_existing_col(f, ["smr_hours", "SMR", "TELEMETRY_SMR", "telemetry_smr"], np.nan), errors="coerce")
    f["failure_code_evidence_score_clean"] = pd.to_numeric(first_existing_col(f, ["failure_code_evidence_score"], np.nan), errors="coerce")
    f["evidence_strength_clean"] = first_existing_col(f, ["failure_code_evidence_strength_class"], "").astype("string").str.upper().str.strip()
    f["evidence_group_clean"] = first_existing_col(f, ["failure_code_evidence_group"], "").astype("string").str.upper().str.strip()
    f["history_category_clean"] = first_existing_col(f, ["history_category"], "").astype("string").str.lower()
    f["applicable_component_clean"] = first_existing_col(f, ["applicable_component", "applicableComponent"], "").astype("string")
    f["related_component_clean"] = (
        first_existing_col(f, ["related_component"], "").astype("string")
        + " "
        + first_existing_col(f, ["related_component_1"], "").astype("string")
        + " "
        + first_existing_col(f, ["applicable_component", "applicableComponent"], "").astype("string")
    )
    f["is_mechanical_failure_code_clean"] = pd.to_numeric(first_existing_col(f, ["is_mechanical_failure_code"], 0), errors="coerce").fillna(0)
    f["is_electrical_failure_code_clean"] = pd.to_numeric(first_existing_col(f, ["is_electrical_failure_code"], 0), errors="coerce").fillna(0)

    for comp, patterns in COMPONENT_PATTERNS.items():
        f[f"is_component_{comp}"] = contains_any(f["related_component_clean"], patterns)
    f["is_event_evidence"] = f["evidence_group_clean"].eq("EVENT") | f["history_category_clean"].str.contains("event", na=False)
    f["is_context_evidence"] = f["evidence_group_clean"].eq("CONTEXT") | f["history_category_clean"].str.contains("context", na=False)

    total_rows = len(f)
    missing_model_id_rows = int(f["model_id"].isna().sum())
    missing_date_rows = int(f["fault_event_date"].isna().sum())
    not_in_backbone_rows = int((~f["model_id"].astype("string").isin(allowed_model_ids) & f["model_id"].notna()).sum())

    out = f[
        f["model_id"].notna()
        & f["fault_event_date"].notna()
        & f["model_id"].astype("string").isin(allowed_model_ids)
    ].copy()
    out = out.sort_values(["model_id", "fault_event_date"]).reset_index(drop=True)

    summary = {
        "source": "fault_codes",
        "source_role": "event_features",
        "input_rows": total_rows,
        "model_id_source": model_id_source,
        "snapshot_date_source": "event_time/event_date",
        "missing_model_id_rows": missing_model_id_rows,
        "missing_usable_event_date_rows": missing_date_rows,
        "dropped_missing_event_date_rows": missing_date_rows,
        "rows_not_in_machine_backbone": not_in_backbone_rows,
        "rows_after_standardization": len(out),
        "unique_model_ids_after_standardization": out["model_id"].nunique(),
    }
    return out, summary


def standardize_maintenance(pm: pd.DataFrame, allowed_model_ids: set[str]) -> tuple[pd.DataFrame, dict]:
    """Convert the raw maintenance-monitor table into a standardized event table."""
    m = pm.copy()
    m["model_id"], model_id_source = normalize_model_id(m, "maintenance")
    m["full_model"] = first_existing_col(m, ["full_model", "FULL_MODEL", "MODEL", "model"], pd.NA).astype("string").str.strip()

    event_time = parse_dt(first_existing_col(m, ["event_time", "UPDATE_DATETIME", "update_datetime"], pd.NaT))
    event_date = parse_dt(first_existing_col(m, ["event_date", "date", "LOCAL_DATE", "local_date"], pd.NaT))
    m["maintenance_event_date"] = event_time.fillna(event_date)

    m["smr_hours_clean"] = pd.to_numeric(first_existing_col(m, ["smr_hours", "SMR", "TELEMETRY_SMR", "telemetry_smr"], np.nan), errors="coerce")
    m["remaining_hours_clean"] = pd.to_numeric(first_existing_col(m, ["remaining_hours", "REMAINING_HOURS"], np.nan), errors="coerce")
    m["is_monitor_reset_clean"] = boolean_from_mixed_values(first_existing_col(m, ["is_monitor_reset"], False), default=False)
    m["is_overdue_clean"] = boolean_from_mixed_values(first_existing_col(m, ["is_overdue"], False), default=False)
    m["is_due_now_clean"] = boolean_from_mixed_values(first_existing_col(m, ["is_due_now"], False), default=False)
    m["available_clean"] = boolean_from_mixed_values(first_existing_col(m, ["AVAILABLE", "available"], True), default=True)
    m["maintenance_type_clean"] = first_existing_col(m, ["maintenance_type", "service_types", "SERVICE_TYPES"], "").astype("string")
    m["related_component_clean"] = (
        first_existing_col(m, ["related_component"], "").astype("string")
        + " "
        + first_existing_col(m, ["related_component_1"], "").astype("string")
        + " "
        + first_existing_col(m, ["related_component_2"], "").astype("string")
    )

    for comp, patterns in COMPONENT_PATTERNS.items():
        m[f"is_component_{comp}"] = contains_any(m["related_component_clean"], patterns)
    for mtype, patterns in MAINTENANCE_TYPE_PATTERNS.items():
        m[f"is_maintenance_type_{mtype}"] = contains_any(m["maintenance_type_clean"], patterns)

    total_rows = len(m)
    missing_model_id_rows = int(m["model_id"].isna().sum())
    missing_date_rows = int(m["maintenance_event_date"].isna().sum())
    not_in_backbone_rows = int((~m["model_id"].astype("string").isin(allowed_model_ids) & m["model_id"].notna()).sum())

    out = m[
        m["model_id"].notna()
        & m["maintenance_event_date"].notna()
        & m["model_id"].astype("string").isin(allowed_model_ids)
    ].copy()
    out = out.sort_values(["model_id", "maintenance_event_date"]).reset_index(drop=True)

    summary = {
        "source": "maintenance",
        "source_role": "event_features",
        "input_rows": total_rows,
        "model_id_source": model_id_source,
        "snapshot_date_source": "event_time/event_date/date",
        "missing_model_id_rows": missing_model_id_rows,
        "missing_usable_event_date_rows": missing_date_rows,
        "dropped_missing_event_date_rows": missing_date_rows,
        "rows_not_in_machine_backbone": not_in_backbone_rows,
        "rows_after_standardization": len(out),
        "unique_model_ids_after_standardization": out["model_id"].nunique(),
    }
    return out, summary


# -----------------------------------------------------------------------------
# Fault source feature engineering
# -----------------------------------------------------------------------------
def fault_features_for_model(snap_m: pd.DataFrame, f_m: pd.DataFrame) -> list[dict]:
    """Create fault-derived features for one model_id across all snapshot dates."""
    out: list[dict] = []
    dates = f_m["fault_event_date"] if "fault_event_date" in f_m.columns else pd.Series(dtype="datetime64[ns]")

    for snap in snap_m["snapshot_date"]:
        before = f_m[dates < snap]
        w90 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=90)]
        w30 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=30)]
        w7 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=7)]
        prev30 = before[
            (before["fault_event_date"] >= snap - pd.Timedelta(days=60))
            & (before["fault_event_date"] < snap - pd.Timedelta(days=30))
        ]
        severe_before = before[before["event_action_level_clean"].isin(["L03", "L04", "L05"])]

        row: dict = {"model_id": snap_m["model_id"].iloc[0], "snapshot_date": snap}
        row["fault_count_7d"] = len(w7)
        row["fault_count_30d"] = len(w30)
        row["fault_count_90d"] = len(w90)
        row["fault_count_previous_30d"] = len(prev30)
        row["fault_growth_rate"] = row["fault_count_30d"] - row["fault_count_previous_30d"]
        row["days_since_last_fault"] = days_between(snap, before["fault_event_date"].max())
        row["days_since_last_severe_fault"] = days_between(snap, severe_before["fault_event_date"].max())

        smr_latest = before["smr_hours_clean"].dropna().max()
        smr_90_ago_candidates = before[
            before["fault_event_date"] <= snap - pd.Timedelta(days=90)
        ]["smr_hours_clean"].dropna()
        smr_90_ago = smr_90_ago_candidates.max() if len(smr_90_ago_candidates) else np.nan
        smr_delta_90d = (
            max(float(smr_latest - smr_90_ago), 0.0)
            if pd.notna(smr_latest) and pd.notna(smr_90_ago)
            else np.nan
        )
        row["faults_per_100_hours"] = ratio(row["fault_count_90d"], max((smr_delta_90d or 0) / 100.0, 1.0))

        row["unique_fault_code_count_90d"] = w90["fault_code_clean"].replace("", np.nan).nunique()
        row["repeat_fault_ratio_90d"] = ratio(row["fault_count_90d"], max(row["unique_fault_code_count_90d"], 1))
        row["unique_component_count_90d"] = w90["applicable_component_clean"].replace("", np.nan).nunique()

        row["mechanical_fault_count_90d"] = int((w90["is_mechanical_failure_code_clean"] == 1).sum())
        row["mechanical_fault_count_30d"] = int((w30["is_mechanical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_90d"] = int((w90["is_electrical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_30d"] = int((w30["is_electrical_failure_code_clean"] == 1).sum())

        for lvl in ["L01", "L02", "L03", "L04"]:
            row[f"action_{lvl}_count_90d"] = int((w90["event_action_level_clean"] == lvl).sum())
        row["max_action_level_90d"] = w90["action_level_num_clean"].max()
        row["sum_log_occurrence_90d"] = w90["log_occurrence_clean"].sum()
        row["max_log_occurrence_90d"] = w90["log_occurrence_clean"].max()
        row["occurrence_severity_score_90d"] = w90["occurrence_class_clean"].sum()

        row["strong_fault_count_90d"] = int((w90["evidence_strength_clean"] == "STRONG").sum())
        row["moderate_fault_count_90d"] = int(w90["evidence_strength_clean"].isin(["MEDIUM", "MODERATE"]).sum())
        event_w90 = w90[w90["is_event_evidence"]]
        context_w90 = w90[w90["is_context_evidence"]]
        row["max_event_evidence_score_90d"] = event_w90["failure_code_evidence_score_clean"].max()
        row["avg_event_evidence_score_90d"] = event_w90["failure_code_evidence_score_clean"].mean()
        row["max_context_evidence_score_90d"] = context_w90["failure_code_evidence_score_clean"].max()

        component_counts: dict[str, int] = {}
        component_feature_map = {
            "engine": "engine_fault_count_90d",
            "hydraulic": "hydraulic_fault_count_90d",
            "powertrain": "powertrain_fault_count_90d",
            "scr": "scr_fault_count_90d",
            "workequipment": "workequipment_fault_count_90d",
            "cooling": "cooling_fault_count_90d",
        }
        for comp, feature in component_feature_map.items():
            cnt = int(w90[f"is_component_{comp}"].sum()) if f"is_component_{comp}" in w90.columns else 0
            row[feature] = cnt
            component_counts[feature] = cnt
        row["top_component_fault_ratio_90d"] = ratio(max(component_counts.values()) if component_counts else 0, row["fault_count_90d"])

        row["has_fault_90d"] = int(row["fault_count_90d"] > 0)
        row["smr_latest_before_snapshot"] = smr_latest
        row["smr_delta_90d"] = smr_delta_90d
        out.append(row)

    return out


def build_fault_snapshot(backbone: pd.DataFrame, fault: pd.DataFrame) -> pd.DataFrame:
    """Build the source-specific fault snapshot dataframe on the machine backbone."""
    start = time.perf_counter()
    rows: list[pd.DataFrame] = []
    fault_groups = {k: v for k, v in fault.groupby("model_id", sort=False)}

    total_models = backbone["model_id"].nunique() if not backbone.empty else 0
    total_snapshot_rows = len(backbone)
    processed_snapshot_rows = 0
    progress(f"Building fault snapshot for {total_models:,} model_ids and {total_snapshot_rows:,} machine-backbone rows...")

    for idx, (model_id, snap_m) in enumerate(backbone.groupby("model_id", sort=False), start=1):
        f_m = fault_groups.get(model_id, fault.iloc[0:0])
        rows.append(pd.DataFrame(fault_features_for_model(snap_m, f_m)))
        processed_snapshot_rows += len(snap_m)

        if idx == 1 or idx % PROGRESS_EVERY_MACHINES == 0 or idx == total_models:
            progress(
                f"Fault snapshot progress: {idx:,}/{total_models:,} model_ids; "
                f"{processed_snapshot_rows:,}/{total_snapshot_rows:,} snapshot rows"
            )

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model_id", "snapshot_date"])
    progress(f"Fault snapshot complete in {(time.perf_counter() - start) / 60:.2f} minutes. Rows: {len(result):,}")
    return result


# -----------------------------------------------------------------------------
# Maintenance source feature engineering
# -----------------------------------------------------------------------------
def maintenance_features_for_model(snap_m: pd.DataFrame, m_m: pd.DataFrame) -> list[dict]:
    """Create maintenance-derived features for one model_id across snapshots."""
    out: list[dict] = []
    dates = m_m["maintenance_event_date"] if "maintenance_event_date" in m_m.columns else pd.Series(dtype="datetime64[ns]")

    for snap in snap_m["snapshot_date"]:
        before = m_m[dates < snap]
        w180 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=180)]
        w90 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=90)]
        reset180 = w180[w180["is_monitor_reset_clean"]]
        reset90 = w90[w90["is_monitor_reset_clean"]]

        latest_cols = ["EVENT_NAME_EN"] if "EVENT_NAME_EN" in before.columns else []
        if latest_cols and len(before):
            current = before.sort_values("maintenance_event_date").groupby(latest_cols, dropna=False).tail(1)
            current = current[current["available_clean"]]
        else:
            current = before.tail(0)

        row: dict = {"model_id": snap_m["model_id"].iloc[0], "snapshot_date": snap}
        row["maintenance_events_180d"] = len(w180)
        row["monitor_reset_count_180d"] = len(reset180)
        row["maintenance_reset_ratio_180d"] = ratio(row["monitor_reset_count_180d"], row["maintenance_events_180d"])
        row["maintenance_events_90d"] = len(w90)
        row["monitor_reset_count_90d"] = len(reset90)
        row["active_maintenance_items"] = len(current)
        row["overdue_item_count"] = int(current["is_overdue_clean"].sum()) if len(current) else 0
        row["due_now_item_count"] = int(current["is_due_now_clean"].sum()) if len(current) else 0

        if len(current) and "remaining_hours_clean" in current.columns:
            row["overdue_item_count"] = max(row["overdue_item_count"], int((current["remaining_hours_clean"] < 0).sum()))
            row["due_now_item_count"] = max(row["due_now_item_count"], int((current["remaining_hours_clean"] == 0).sum()))
        row["maintenance_due_or_overdue_ratio"] = ratio(
            row["due_now_item_count"] + row["overdue_item_count"],
            row["active_maintenance_items"],
        )
        row["avg_remaining_hours"] = current["remaining_hours_clean"].mean() if len(current) else np.nan
        row["min_remaining_hours"] = current["remaining_hours_clean"].min() if len(current) else np.nan

        component_reset_feature_map = {
            "engine": "engine_reset_count_180d",
            "powertrain": "transmission_reset_count_180d",
            "final_drive": "final_drive_reset_count_180d",
            "cooling": "cooling_system_reset_count_180d",
            "scr": "urea_scr_system_reset_count_180d",
        }
        component_overdue_feature_map = {
            "engine": "engine_overdue_item_count",
            "powertrain": "transmission_overdue_item_count",
            "final_drive": "final_drive_overdue_item_count",
            "cooling": "cooling_system_overdue_item_count",
            "scr": "urea_scr_system_overdue_item_count",
        }
        for comp, feature in component_reset_feature_map.items():
            row[feature] = int(reset180[f"is_component_{comp}"].sum()) if f"is_component_{comp}" in reset180.columns else 0
        for comp, feature in component_overdue_feature_map.items():
            if len(current) and f"is_component_{comp}" in current.columns:
                row[feature] = int(current[current["is_overdue_clean"]][f"is_component_{comp}"].sum())
            else:
                row[feature] = 0

        for mtype in MAINTENANCE_TYPE_PATTERNS:
            col = f"is_maintenance_type_{mtype}"
            row[f"{mtype}_reset_count_180d"] = int(reset180[col].sum()) if col in reset180.columns else 0

        row["unique_maintenance_type_count_180d"] = reset180["maintenance_type_clean"].replace("", np.nan).nunique()
        row["days_since_last_reset"] = days_between(snap, reset180["maintenance_event_date"].max())

        oil_reset = reset180[reset180.get("is_maintenance_type_oil", pd.Series(False, index=reset180.index))]
        filter_reset = reset180[reset180.get("is_maintenance_type_filter", pd.Series(False, index=reset180.index))]
        row["days_since_last_oil_reset"] = days_between(snap, oil_reset["maintenance_event_date"].max())
        row["days_since_last_filter_reset"] = days_between(snap, filter_reset["maintenance_event_date"].max())

        latest_smr = before["smr_hours_clean"].dropna().max()
        last_reset = reset180.sort_values("maintenance_event_date").tail(1)
        last_reset_smr = last_reset["smr_hours_clean"].iloc[0] if len(last_reset) else np.nan
        row["smr_since_last_reset"] = (
            float(latest_smr - last_reset_smr)
            if pd.notna(latest_smr) and pd.notna(last_reset_smr)
            else np.nan
        )
        row["has_maintenance_180d"] = int(row["maintenance_events_180d"] > 0)
        out.append(row)

    return out


def build_maintenance_snapshot(backbone: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Build the source-specific maintenance snapshot dataframe on the machine backbone."""
    start = time.perf_counter()
    rows: list[pd.DataFrame] = []
    pm_groups = {k: v for k, v in pm.groupby("model_id", sort=False)}

    total_models = backbone["model_id"].nunique() if not backbone.empty else 0
    total_snapshot_rows = len(backbone)
    processed_snapshot_rows = 0
    progress(f"Building maintenance snapshot for {total_models:,} model_ids and {total_snapshot_rows:,} machine-backbone rows...")

    for idx, (model_id, snap_m) in enumerate(backbone.groupby("model_id", sort=False), start=1):
        m_m = pm_groups.get(model_id, pm.iloc[0:0])
        rows.append(pd.DataFrame(maintenance_features_for_model(snap_m, m_m)))
        processed_snapshot_rows += len(snap_m)

        if idx == 1 or idx % PROGRESS_EVERY_MACHINES == 0 or idx == total_models:
            progress(
                f"Maintenance snapshot progress: {idx:,}/{total_models:,} model_ids; "
                f"{processed_snapshot_rows:,}/{total_snapshot_rows:,} snapshot rows"
            )

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model_id", "snapshot_date"])
    progress(f"Maintenance snapshot complete in {(time.perf_counter() - start) / 60:.2f} minutes. Rows: {len(result):,}")
    return result


# -----------------------------------------------------------------------------
# Joining, validation, and saving
# -----------------------------------------------------------------------------
def key_count(df: pd.DataFrame) -> int:
    return int(df[["model_id", "snapshot_date"]].drop_duplicates().shape[0]) if not df.empty else 0


def validate_source_snapshot_alignment(source_name: str, backbone: pd.DataFrame, source_snapshot: pd.DataFrame) -> None:
    """Make sure a source snapshot exactly follows the machine.csv backbone."""
    required = {"model_id", "snapshot_date"}
    missing_cols = required - set(source_snapshot.columns)
    if missing_cols:
        raise ValueError(f"{source_name} snapshot is missing key columns: {sorted(missing_cols)}")

    duplicate_count = int(source_snapshot.duplicated(["model_id", "snapshot_date"]).sum())
    if duplicate_count:
        raise ValueError(f"{source_name} snapshot has duplicate model_id + snapshot_date keys: {duplicate_count:,}")

    if len(source_snapshot) != len(backbone):
        raise ValueError(
            f"{source_name} snapshot row count mismatch. Expected {len(backbone):,} from machine backbone, "
            f"got {len(source_snapshot):,}."
        )

    b_keys = backbone[["model_id", "snapshot_date"]].drop_duplicates()
    s_keys = source_snapshot[["model_id", "snapshot_date"]].drop_duplicates()
    merged = b_keys.merge(s_keys, on=["model_id", "snapshot_date"], how="outer", indicator=True)
    missing_from_source = int((merged["_merge"] == "left_only").sum())
    extra_in_source = int((merged["_merge"] == "right_only").sum())
    if missing_from_source or extra_in_source:
        raise ValueError(
            f"{source_name} snapshot key mismatch against machine backbone. "
            f"missing_from_source={missing_from_source:,}, extra_in_source={extra_in_source:,}."
        )

    progress(f"{source_name} alignment validated: {len(source_snapshot):,} rows match machine backbone.")


def finalize_snapshot_df(df: pd.DataFrame, feature_freeze_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Fill feature missing values and order columns for model training."""
    out = df.copy()

    for col in FROZEN_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    for col in COUNT_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    for col in RECENCY_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(9999).astype(float)

    ratio_cols = [c for c in FROZEN_FEATURES if c.endswith("ratio_90d") or c.endswith("ratio_180d") or c.endswith("rate")]
    for col in ratio_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    amount_or_score_cols = [
        "faults_per_100_hours",
        "max_action_level_90d",
        "sum_log_occurrence_90d",
        "max_log_occurrence_90d",
        "occurrence_severity_score_90d",
        "max_event_evidence_score_90d",
        "avg_event_evidence_score_90d",
        "max_context_evidence_score_90d",
        "avg_remaining_hours",
        "min_remaining_hours",
        "smr_since_last_reset",
        "smr_latest_before_snapshot",
        "smr_delta_90d",
    ]
    for col in amount_or_score_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    if feature_freeze_path is not None and Path(feature_freeze_path).exists():
        freeze = pd.read_excel(feature_freeze_path, sheet_name="all")
        if "Feature" in freeze.columns:
            frozen_from_file = (
                freeze["Feature"].astype(str)
                .str.replace(r"\s*\([a-z]\)$", "", regex=True)
                .str.strip()
                .tolist()
            )
            missing = sorted(set(frozen_from_file) - set(out.columns))
            if missing:
                raise ValueError(f"Missing frozen features from output: {missing}")

    ordered_cols = ["model_id", "snapshot_date", "full_model"]
    if "claim_next_45d" in out.columns:
        ordered_cols.append("claim_next_45d")
    ordered_cols += FROZEN_FEATURES
    extra_cols = [c for c in out.columns if c not in ordered_cols]

    for col in ordered_cols:
        if col not in out.columns:
            out[col] = pd.Series(dtype="float64")

    return out[ordered_cols + extra_cols].sort_values(["model_id", "snapshot_date"]).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save dataframe as CSV.

    For the current QA/review phase, every dataframe output is forced to CSV so
    it can be opened directly in Excel, VS Code, or similar tools. If config.py
    still points to a .parquet path, this function automatically changes the
    suffix to .csv instead of writing parquet.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        csv_path = output_path.with_suffix(".csv")
        progress(f"CSV output mode is enabled. Rewriting output path from {output_path} to {csv_path}")
        output_path = csv_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_source_snapshot(df: pd.DataFrame, filename: str) -> Path:
    """Save an intermediate source-level snapshot dataframe."""
    if not bool(cfg("SAVE_SOURCE_SNAPSHOTS", True)):
        return Path("")
    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return save_dataframe(df, SOURCE_SNAPSHOT_DIR / filename)


def write_source_standardization_summary(output_dir: str | Path, summaries: list[dict]) -> None:
    """Write standardization/date-drop summary for all sources."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_dir / "source_standardization_summary.csv", index=False)


def write_source_alignment_summary(output_dir: str | Path, backbone: pd.DataFrame, source_snapshots: dict[str, pd.DataFrame]) -> None:
    """Write a compact source alignment report."""
    rows = [
        {
            "source_snapshot": "machine_backbone",
            "rows": len(backbone),
            "unique_model_ids": backbone["model_id"].nunique() if not backbone.empty else 0,
            "unique_keys": key_count(backbone),
            "matches_backbone_rows": True,
        }
    ]
    for name, df in source_snapshots.items():
        rows.append(
            {
                "source_snapshot": name,
                "rows": len(df),
                "unique_model_ids": df["model_id"].nunique() if not df.empty else 0,
                "unique_keys": key_count(df),
                "matches_backbone_rows": len(df) == len(backbone) and key_count(df) == key_count(backbone),
            }
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "source_snapshot_alignment_summary.csv", index=False)


def write_mini_validation_outputs(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Write small QA outputs for mini validation runs."""
    if not bool(cfg("MINI_RUN_ENABLED", False)):
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.head(200).to_csv(output_dir / "mini_snapshot_validation_sample_rows.csv", index=False)

    summary = (
        df.groupby("model_id")
        .agg(
            snapshot_rows=("snapshot_date", "count"),
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
            total_faults_90d=("fault_count_90d", "sum"),
            total_maintenance_events_180d=("maintenance_events_180d", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "mini_snapshot_validation_by_model_id.csv", index=False)


# -----------------------------------------------------------------------------
# End-to-end orchestration
# -----------------------------------------------------------------------------
def build_snapshot_dataframe(
    fault_codes_path: str | Path,
    machine_path: str | Path,
    maintenance_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    feature_freeze_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build separate source snapshots following machine.csv, then join them."""
    overall_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    progress("Starting machine-backbone-driven snapshot dataframe build...")
    progress(f"Project root: {PROJECT_ROOT}")
    progress(f"Input folder: {INPUT_DIR}")
    progress(f"Output folder: {output_dir}")
    progress("Canonical backbone: machine.csv model_id + snapshot_date")
    progress("Join key: model_id + snapshot_date")

    all_profiles: list[pd.DataFrame] = []
    cleaning_summaries: list[dict] = []
    standardization_summaries: list[dict] = []

    progress("Step 1/8: Loading and lightly cleaning source CSV files...")
    raw_machine, profiles, summary = load_and_clean_csv(machine_path, "machine", output_dir)
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"machine loaded: {len(raw_machine):,} rows, {len(raw_machine.columns):,} columns")

    raw_fault, profiles, summary = load_and_clean_csv(fault_codes_path, "fault_codes", output_dir)
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"fault_codes loaded: {len(raw_fault):,} rows, {len(raw_fault.columns):,} columns")

    raw_pm, profiles, summary = load_and_clean_csv(maintenance_path, "maintenance", output_dir)
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"maintenance loaded: {len(raw_pm):,} rows, {len(raw_pm.columns):,} columns")

    if bool(cfg("WRITE_CLEANING_REPORTS", True)):
        write_combined_cleaning_reports(output_dir, all_profiles, cleaning_summaries)
        progress("Missing-value and light-cleaning reports written.")

    progress("Step 2/8: Standardizing machine.csv as canonical snapshot backbone...")
    backbone, summary = standardize_machine_backbone(raw_machine)
    standardization_summaries.append(summary)
    progress(
        "Machine backbone date quality: "
        f"missing snapshot rows={summary['missing_usable_event_date_rows']:,}; "
        f"duplicate keys removed={summary['duplicate_model_id_snapshot_rows_removed']:,}; "
        f"rows after standardization={summary['rows_after_standardization']:,}"
    )

    backbone = apply_backbone_filters(backbone)
    allowed_model_ids = set(backbone["model_id"].astype("string"))
    progress(
        f"Canonical backbone after filters: {len(backbone):,} rows across "
        f"{backbone['model_id'].nunique() if not backbone.empty else 0:,} model_ids"
    )
    save_source_snapshot(backbone, "machine_backbone.csv")

    progress("Step 3/8: Standardizing event sources and forcing them to machine backbone model_ids...")
    fault, summary = standardize_faults(raw_fault, allowed_model_ids=allowed_model_ids)
    standardization_summaries.append(summary)
    progress(
        "Fault date/model-id quality: "
        f"missing usable date rows={summary['missing_usable_event_date_rows']:,}; "
        f"rows not in machine backbone={summary['rows_not_in_machine_backbone']:,}; "
        f"rows after standardization={summary['rows_after_standardization']:,}"
    )

    pm, summary = standardize_maintenance(raw_pm, allowed_model_ids=allowed_model_ids)
    standardization_summaries.append(summary)
    progress(
        "Maintenance date/model-id quality: "
        f"missing usable date rows={summary['missing_usable_event_date_rows']:,}; "
        f"rows not in machine backbone={summary['rows_not_in_machine_backbone']:,}; "
        f"rows after standardization={summary['rows_after_standardization']:,}"
    )

    write_source_standardization_summary(output_dir, standardization_summaries)
    progress("Source standardization summary written.")

    progress("Step 4/8: Building fault source snapshot dataframe on machine backbone...")
    fault_snapshot = build_fault_snapshot(backbone, fault)
    validate_source_snapshot_alignment("fault_snapshot", backbone, fault_snapshot)
    save_source_snapshot(fault_snapshot, "fault_snapshot.csv")

    progress("Step 5/8: Building maintenance source snapshot dataframe on machine backbone...")
    maintenance_snapshot = build_maintenance_snapshot(backbone, pm)
    validate_source_snapshot_alignment("maintenance_snapshot", backbone, maintenance_snapshot)
    save_source_snapshot(maintenance_snapshot, "maintenance_snapshot.csv")

    progress("Step 6/8: Writing source snapshot alignment summary...")
    write_source_alignment_summary(
        output_dir,
        backbone,
        {
            "fault_snapshot": fault_snapshot,
            "maintenance_snapshot": maintenance_snapshot,
        },
    )

    progress("Step 7/8: Joining source snapshots into unified snapshot dataframe...")
    unified = backbone.copy()
    unified = unified.merge(fault_snapshot, on=["model_id", "snapshot_date"], how="left")
    unified = unified.merge(maintenance_snapshot, on=["model_id", "snapshot_date"], how="left")
    progress(f"Unified snapshot shape before finalization: {unified.shape[0]:,} rows x {unified.shape[1]:,} columns")

    progress("Step 8/8: Finalizing missing values and validating frozen features...")
    unified = finalize_snapshot_df(unified, feature_freeze_path=feature_freeze_path)
    write_mini_validation_outputs(unified, output_dir)

    elapsed_min = (time.perf_counter() - overall_start) / 60
    progress(f"Snapshot build complete in {elapsed_min:.2f} minutes. Final shape: {unified.shape[0]:,} rows x {unified.shape[1]:,} columns")
    return unified


# -----------------------------------------------------------------------------
# Future source-extension placeholders
# -----------------------------------------------------------------------------
# When oil, warranty, or service data becomes available, add each source in the
# same pattern used above:
#
#   1. standardize_oil(raw_oil, allowed_model_ids) -> oil_event_table, summary
#   2. build_oil_snapshot(backbone, oil_event_table) -> model_id/snapshot_date features
#   3. validate_source_snapshot_alignment("oil_snapshot", backbone, oil_snapshot)
#   4. save_source_snapshot(oil_snapshot, "oil_snapshot.csv")
#   5. unified = unified.merge(oil_snapshot, on=["model_id", "snapshot_date"], how="left")
#
# The key rule stays the same: oil/service/warranty source snapshots must use the
# machine.csv backbone dates. They should never generate independent calendars.
# Warranty target should look after snapshot_date, not before it.


def main() -> None:
    """Config-driven entry point. No command-line arguments are required."""
    fault_codes_path = resolve_existing_path(cfg("FAULT_CODES_PATH", INPUT_DIR / "fault_codes.csv"), "FAULT_CODES_PATH")
    machine_path = resolve_existing_path(cfg("MACHINE_PATH", INPUT_DIR / "machine.csv"), "MACHINE_PATH")
    maintenance_path = resolve_existing_path(cfg("MAINTENANCE_PATH", INPUT_DIR / "maintenance.csv"), "MAINTENANCE_PATH")
    feature_freeze_path = optional_existing_path(cfg("FEATURE_FREEZE_PATH", INPUT_DIR / "xgb_feature_freeze.xlsx"))

    output_path = Path(cfg("OUTPUT_PATH", OUTPUT_DIR / "snapshot_dataframe.csv"))
    if bool(cfg("MINI_RUN_ENABLED", False)):
        output_path = Path(cfg("MINI_OUTPUT_PATH", OUTPUT_DIR / "snapshot_dataframe_mini.csv"))

    print("Using config.py" if run_config is not None else "config.py not found; using built-in defaults")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output path: {output_path}")
    print(f"Target model families used for filtering: {', '.join(TARGET_MODEL_FAMILIES)}")
    print(f"Mini run enabled: {bool(cfg('MINI_RUN_ENABLED', False))}")

    df = build_snapshot_dataframe(
        fault_codes_path=fault_codes_path,
        machine_path=machine_path,
        maintenance_path=maintenance_path,
        output_dir=OUTPUT_DIR,
        feature_freeze_path=feature_freeze_path,
    )

    progress("Saving unified snapshot dataframe...")
    saved_path = save_dataframe(df, output_path)

    run_summary = {
        "saved_path": str(saved_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_model_ids": int(df["model_id"].nunique()) if "model_id" in df.columns else 0,
        "min_snapshot_date": str(df["snapshot_date"].min()) if "snapshot_date" in df.columns and len(df) else None,
        "max_snapshot_date": str(df["snapshot_date"].max()) if "snapshot_date" in df.columns and len(df) else None,
        "source_snapshot_dir": str(SOURCE_SNAPSHOT_DIR),
        "canonical_backbone": "machine.csv model_id + snapshot_date",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([run_summary]).to_csv(OUTPUT_DIR / "snapshot_build_run_summary.csv", index=False)

    print(f"Saved unified snapshot dataframe: {saved_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Unique model_id count: {df['model_id'].nunique() if 'model_id' in df.columns else 0:,}")
    print(f"Source snapshot folder: {SOURCE_SNAPSHOT_DIR}")
    print(f"Reports folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
