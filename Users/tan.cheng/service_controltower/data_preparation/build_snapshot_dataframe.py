"""
Build a leakage-safe snapshot dataframe for D51 / D61 / D71 predictive maintenance.

Default project layout expected by this script:

    enriched_data/
        fault_codes.csv
        machine.csv
        maintenance.csv
        warranty.csv                 # optional, only needed for claim_next_45d labels
        xgb_feature_freeze.xlsx       # optional, used only for validation

    data_preparation/output/
        snapshot_dataframe.parquet
        missing_profile_all_files.csv
        missing_profile_fault_codes.csv
        missing_profile_machine.csv
        missing_profile_maintenance.csv
        missing_profile_warranty.csv  # only if warranty is provided
        cleaning_summary.csv

Output grain:
    One row per machine / snapshot_date.

Core leakage-control rule:
    Feature windows use only records with event_date < snapshot_date.
    The optional target uses warranty failure dates after the snapshot date and on/before
    snapshot_date + horizon_days.

Notes about data cleaning in this script:
    The cleaning step is intentionally light and auditable. It standardizes obvious null-like
    strings such as "", "nan", "null", "N/A" to real missing values, strips whitespace from
    object columns, and drops only fully empty rows or columns. It does not drop duplicate rows
    by default because duplicate-looking event records may still represent real repeated events.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Optional run configuration
# -----------------------------------------------------------------------------
# The script is designed to run from config.py so you do not have to type a long
# command every time. Keep config.py in the same folder as this script, edit the
# parameter values there, and run:
#
#     python build_snapshot_dataframe.py
#
# If config.py is not present, the script falls back to the built-in defaults
# defined below. The build_snapshot_dataframe(...) function still accepts direct
# arguments, which is useful when calling it from notebooks or tests.
try:
    import config as run_config
except ImportError:  # pragma: no cover - allows module reuse without config.py
    run_config = None


def cfg(name: str, default):
    """Read a value from config.py, or return a safe built-in default."""
    return getattr(run_config, name, default) if run_config is not None else default


TARGET_MODEL_FAMILIES = tuple(
    str(x).upper().strip() for x in cfg("TARGET_MODEL_FAMILIES", ("D51", "D61", "D71"))
)


# -----------------------------------------------------------------------------
# Console progress helper
# -----------------------------------------------------------------------------
# Keep this inside the script so config.py does not need to change.
# Increase this number if you want fewer progress messages.
PROGRESS_EVERY_MACHINES = 100


def progress(message: str) -> None:
    """Print a short progress message immediately.

    flush=True matters because long-running scripts sometimes buffer output,
    especially in VS Code terminals, remote compute, notebooks, or Azure jobs.
    """
    print(f"[snapshot-build] {message}", flush=True)


# -----------------------------------------------------------------------------
# Frozen model feature list
# -----------------------------------------------------------------------------
# These are the feature columns that came from the provided xgb_feature_freeze.xlsx
# workbook. The script guarantees that each of these columns exists in the final
# dataframe, even when a source column is unavailable and the feature has to be
# filled with a safe default.
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

# Features that represent counts should become 0 when no record exists in the
# lookback window. This is different from recency or measurement features, where
# missing can have a different meaning.
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

# Recency features use a large sentinel value when no event was observed before
# the snapshot. This tells the model "very long time / no prior event" without
# dropping the row.
RECENCY_FEATURES = [
    "days_since_last_fault",
    "days_since_last_severe_fault",
    "days_since_last_reset",
    "days_since_last_oil_reset",
    "days_since_last_filter_reset",
]

# Null-like values that often appear in CSV exports. The cleaning step converts
# these to real missing values before feature generation.
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

# Keyword patterns used to aggregate source records into component-level model
# features. These are intentionally simple and transparent so the business can
# review and revise them later.
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
# Input/output path helpers
# -----------------------------------------------------------------------------
def resolve_project_input_path(
    input_dir: str | Path,
    path_arg: Optional[str | Path],
    default_filename: str,
    required: bool,
) -> Optional[Path]:
    """Resolve a file path from either an explicit CLI path or the input folder.

    Examples:
        --input-dir enriched_data and no --fault-codes argument becomes
        enriched_data/fault_codes.csv.

        If an optional file such as warranty.csv does not exist, return None.
    """
    input_dir = Path(input_dir)
    candidate = Path(path_arg) if path_arg else input_dir / default_filename

    if required and not candidate.exists():
        raise FileNotFoundError(f"Required input file not found: {candidate}")

    if not required and not candidate.exists():
        return None

    return candidate


def resolve_feature_freeze_path(
    input_dir: str | Path,
    path_arg: Optional[str | Path],
) -> Optional[Path]:
    """Find the optional Excel feature-freeze file.

    The file is optional because FROZEN_FEATURES is already embedded above. When
    the Excel file is present, the script uses it as a validation check to make
    sure the output still contains all expected features.
    """
    if path_arg:
        p = Path(path_arg)
        return p if p.exists() else None

    candidates = [
        Path(input_dir) / "xgb_feature_freeze.xlsx",
        Path("xgb_feature_freeze.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_output_path(output_dir: str | Path, output_arg: str | Path) -> Path:
    """Resolve the snapshot output path.

    If --output is just a filename, save it inside --output-dir. If --output has
    a directory component or is absolute, respect it as provided.
    """
    output_dir = Path(output_dir)
    output = Path(output_arg)
    if output.is_absolute() or output.parent != Path("."):
        return output
    return output_dir / output


# -----------------------------------------------------------------------------
# Data cleaning and missing-value reporting
# -----------------------------------------------------------------------------
def dedupe_column_names(columns: Iterable[object]) -> list[str]:
    """Strip whitespace from column names and make duplicate names unique.

    CSV exports sometimes contain columns like "SERIAL " or duplicate headers
    after trimming. Duplicate column names can break pandas selection, so this
    function appends a suffix to repeated names.
    """
    seen: dict[str, int] = {}
    cleaned_columns: list[str] = []

    for col in columns:
        base = str(col).strip()
        base = base if base else "unnamed_column"
        count = seen.get(base, 0) + 1
        seen[base] = count
        cleaned_columns.append(base if count == 1 else f"{base}_duplicate_{count}")

    return cleaned_columns


def blank_or_null_string_count(series: pd.Series) -> int:
    """Count string values that are visually blank or null-like.

    This catches values such as "", "nan", "NULL", and "N/A" before they are
    converted to actual missing values.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return 0
    text = series.astype("string").str.strip().str.lower()
    return int(text.isin(NULL_LIKE_STRINGS).sum())


def sample_non_missing_values(series: pd.Series, max_values: int = 3) -> str:
    """Return a compact example-value string for the missing-value report."""
    examples = series.dropna().astype(str).head(max_values).tolist()
    return " | ".join(examples)


def build_missing_profile(df: pd.DataFrame, dataset_name: str, stage: str) -> pd.DataFrame:
    """Build a column-level missingness report for one dataframe.

    The report is generated both before and after light cleaning. This helps you
    audit whether missing values are coming from true pandas NaN values or from
    CSV strings like "nan" and "N/A".
    """
    row_count = len(df)
    profile_rows: list[dict] = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        blank_like_count = blank_or_null_string_count(df[col])
        non_missing = int(row_count - missing_count)
        unique_non_missing = int(df[col].dropna().nunique()) if row_count else 0

        profile_rows.append(
            {
                "dataset": dataset_name,
                "stage": stage,
                "column": col,
                "dtype": str(df[col].dtype),
                "row_count": row_count,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / row_count) * 100, 4) if row_count else 0.0,
                "blank_or_null_string_count": blank_like_count,
                "non_missing_count": non_missing,
                "unique_non_missing_count": unique_non_missing,
                "example_non_missing_values": sample_non_missing_values(df[col]),
            }
        )

    return pd.DataFrame(profile_rows)


def clean_raw_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Lightly clean one raw CSV dataframe and return audit artifacts.

    Cleaning choices are deliberately conservative:
        1. Strip column-name whitespace and make duplicate names unique.
        2. Strip whitespace in text columns.
        3. Convert null-like strings to missing values.
        4. Drop only rows or columns that are completely empty.

    The function returns:
        cleaned_df, raw_missing_profile, cleaned_missing_profile, cleaning_summary
    """
    rows_before = len(df)
    cols_before = len(df.columns)

    raw_profile = build_missing_profile(df, dataset_name=dataset_name, stage="raw")

    cleaned = df.copy()
    cleaned.columns = dedupe_column_names(cleaned.columns)

    # Normalize object/string columns. Numeric columns are left untouched here;
    # individual feature-standardization functions convert numeric fields later.
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
        "rows_after": len(cleaned),
        "rows_dropped_fully_empty": rows_before - len(cleaned),
        "columns_before": cols_before,
        "columns_after": len(cleaned.columns),
        "columns_dropped_fully_empty": len(fully_empty_cols),
        "fully_empty_columns_dropped": ", ".join(fully_empty_cols),
        "missing_cells_raw": int(df.isna().sum().sum()),
        "missing_cells_cleaned": int(cleaned.isna().sum().sum()),
        "blank_or_null_strings_raw": int(raw_profile["blank_or_null_string_count"].sum())
        if not raw_profile.empty
        else 0,
    }

    return cleaned, raw_profile, cleaned_profile, summary


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """Read a CSV without dtype guessing warnings on mixed-type columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def load_and_clean_csv(
    path: str | Path,
    dataset_name: str,
    output_dir: str | Path,
    write_cleaning_reports: bool = True,
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    """Read a CSV, run missing-value detection, clean it, and optionally save reports."""
    raw = read_csv_safely(path)
    cleaned, raw_profile, cleaned_profile, summary = clean_raw_dataframe(raw, dataset_name)
    profiles = [raw_profile, cleaned_profile]

    if write_cleaning_reports:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(profiles, ignore_index=True).to_csv(
            output_dir / f"missing_profile_{dataset_name}.csv",
            index=False,
        )

    return cleaned, profiles, summary


def write_combined_cleaning_reports(
    output_dir: str | Path,
    profiles: list[pd.DataFrame],
    summaries: list[dict],
) -> None:
    """Save all per-file missing profiles and cleaning summaries in one place."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(
            output_dir / "missing_profile_all_files.csv",
            index=False,
        )

    if summaries:
        pd.DataFrame(summaries).to_csv(output_dir / "cleaning_summary.csv", index=False)


# -----------------------------------------------------------------------------
# Generic dataframe helpers
# -----------------------------------------------------------------------------
def safe_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """Return df[col] if it exists; otherwise return a default Series.

    This allows the script to keep running when a source file is missing an
    optional column. The feature is then filled with a safe default later.
    """
    if col in df.columns:
        return df[col]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series(default, index=df.index)


def first_existing_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default=np.nan,
) -> pd.Series:
    """Return the first available source column from a list of possible names."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def normalize_key(s: pd.Series) -> pd.Series:
    """Normalize machine identifiers such as SERIAL or serial_number."""
    out = s.astype("string").str.strip()
    return out.mask(out.str.lower().isin(NULL_LIKE_STRINGS), pd.NA)


def parse_dt(s: pd.Series) -> pd.Series:
    """Parse a date column safely and normalize timezone handling.

    errors="coerce" turns invalid dates into NaT instead of crashing.
    utc=True makes mixed timezone formats comparable.
    tz_convert(None) removes timezone metadata after converting everything to UTC.
    """
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def model_family(full_model: object) -> Optional[str]:
    """Extract a configured target model family such as D51, D61, or D71."""
    if pd.isna(full_model):
        return None

    text = str(full_model).upper()
    for family in TARGET_MODEL_FAMILIES:
        # Match model-family prefixes at word boundaries. This catches values
        # such as D51, D51PX, D61EX, and D71PX while avoiding unrelated strings.
        if re.search(rf"\b{re.escape(family)}", text):
            return family
    return None


def contains_any(series: pd.Series, patterns: Iterable[str]) -> pd.Series:
    """Boolean mask for rows where a text field contains any keyword pattern."""
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
    """Days from event_date to snapshot_date. Missing event dates stay missing."""
    if event_date is None or pd.isna(event_date):
        return np.nan
    return float((snapshot_date - event_date).days)


def boolean_from_mixed_values(series: pd.Series, default: bool = False) -> pd.Series:
    """Convert messy true/false values into booleans."""
    if series is None:
        return pd.Series(default)
    text = series.astype("string").str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f"}
    result = text.isin(true_values)
    result = result.mask(text.isin(false_values), False)
    return result.fillna(default).astype(bool)


# -----------------------------------------------------------------------------
# Source standardization
# -----------------------------------------------------------------------------
def standardize_faults(fault: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw fault/event file into a standardized event table.

    Required downstream columns created here include:
        SERIAL, full_model, model_family, fault_event_date, fault_code_clean,
        event_action_level_clean, occurrence features, SMR, evidence fields,
        and component text fields.
    """
    f = fault.copy()

    f["SERIAL"] = normalize_key(first_existing_col(f, ["serial_number", "SERIAL", "Serial", "ZZSERNR"]))
    f["full_model"] = first_existing_col(
        f,
        ["full_model", "FULL_MODEL", "MODEL", "model", "ZZMATNR"],
        "",
    ).astype("string").str.strip()
    f["model_family"] = f["full_model"].map(model_family)

    # Prefer event_time if available because it has timestamp precision. If it is
    # missing, fall back to event_date. Both are cleaned to timezone-naive UTC.
    event_time = parse_dt(first_existing_col(f, ["event_time", "UPDATE_DATETIME", "update_datetime"], pd.NaT))
    event_date = parse_dt(first_existing_col(f, ["event_date", "LOCAL_DATE", "local_date"], pd.NaT))
    f["fault_event_date"] = event_time.fillna(event_date)

    # Standardize field variants used by the feature functions below.
    f["fault_code_clean"] = first_existing_col(
        f,
        ["fault_code", "EVENT_CODE", "event_code", "ERROR_CODE", "error_code"],
        "",
    ).astype("string").str.strip()
    f["event_action_level_clean"] = first_existing_col(
        f,
        ["event_action_level", "Action_level", "ACTION_LEVEL", "action_level"],
        "",
    ).astype("string").str.upper().str.strip()
    f["action_level_num_clean"] = pd.to_numeric(
        first_existing_col(f, ["action_level_num", "ACTION_LEVEL_NUM"], np.nan),
        errors="coerce",
    )
    f["action_level_num_clean"] = f["action_level_num_clean"].fillna(
        action_level_to_num(f["event_action_level_clean"])
    )
    f["occurrence_count_clean"] = pd.to_numeric(
        first_existing_col(f, ["occurrence_count", "OCCURRENCE_COUNT"], 1),
        errors="coerce",
    ).fillna(1)
    f["log_occurrence_clean"] = pd.to_numeric(
        first_existing_col(f, ["log_occurrence_count"], np.nan),
        errors="coerce",
    )
    f["log_occurrence_clean"] = f["log_occurrence_clean"].fillna(np.log1p(f["occurrence_count_clean"]))
    f["occurrence_class_clean"] = pd.to_numeric(
        first_existing_col(f, ["occurrence_class"], 0),
        errors="coerce",
    ).fillna(0)
    f["smr_hours_clean"] = pd.to_numeric(
        first_existing_col(f, ["smr_hours", "SMR", "TELEMETRY_SMR", "telemetry_smr"], np.nan),
        errors="coerce",
    )
    f["failure_code_evidence_score_clean"] = pd.to_numeric(
        first_existing_col(f, ["failure_code_evidence_score"], np.nan),
        errors="coerce",
    )
    f["evidence_strength_clean"] = first_existing_col(
        f,
        ["failure_code_evidence_strength_class"],
        "",
    ).astype("string").str.upper().str.strip()
    f["evidence_group_clean"] = first_existing_col(
        f,
        ["failure_code_evidence_group"],
        "",
    ).astype("string").str.upper().str.strip()
    f["history_category_clean"] = first_existing_col(
        f,
        ["history_category"],
        "",
    ).astype("string").str.lower()
    f["applicable_component_clean"] = first_existing_col(
        f,
        ["applicable_component", "applicableComponent"],
        "",
    ).astype("string")
    f["related_component_clean"] = (
        first_existing_col(f, ["related_component"], "").astype("string")
        + " "
        + first_existing_col(f, ["related_component_1"], "").astype("string")
        + " "
        + first_existing_col(f, ["applicable_component", "applicableComponent"], "").astype("string")
    )
    f["is_mechanical_failure_code_clean"] = pd.to_numeric(
        first_existing_col(f, ["is_mechanical_failure_code"], 0),
        errors="coerce",
    ).fillna(0)
    f["is_electrical_failure_code_clean"] = pd.to_numeric(
        first_existing_col(f, ["is_electrical_failure_code"], 0),
        errors="coerce",
    ).fillna(0)

    # Keep only usable D51/D61/D71 fault records with a machine id and date.
    f = f[
        f["fault_event_date"].notna()
        & f["SERIAL"].notna()
        & f["model_family"].isin(TARGET_MODEL_FAMILIES)
    ]
    return f.sort_values(["SERIAL", "fault_event_date"]).reset_index(drop=True)


def standardize_maintenance(pm: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw maintenance-monitor file into a standardized event table."""
    m = pm.copy()

    m["SERIAL"] = normalize_key(first_existing_col(m, ["SERIAL", "serial_number", "Serial"]))
    m["full_model"] = first_existing_col(
        m,
        ["full_model", "FULL_MODEL", "MODEL", "model"],
        "",
    ).astype("string").str.strip()
    m["model_family"] = m["full_model"].map(model_family)

    event_time = parse_dt(first_existing_col(m, ["event_time", "UPDATE_DATETIME", "update_datetime"], pd.NaT))
    event_date = parse_dt(first_existing_col(m, ["event_date", "date", "LOCAL_DATE", "local_date"], pd.NaT))
    m["maintenance_event_date"] = event_time.fillna(event_date)

    m["smr_hours_clean"] = pd.to_numeric(
        first_existing_col(m, ["smr_hours", "SMR", "TELEMETRY_SMR", "telemetry_smr"], np.nan),
        errors="coerce",
    )
    m["remaining_hours_clean"] = pd.to_numeric(
        first_existing_col(m, ["remaining_hours", "REMAINING_HOURS"], np.nan),
        errors="coerce",
    )
    m["is_monitor_reset_clean"] = boolean_from_mixed_values(
        first_existing_col(m, ["is_monitor_reset"], False),
        default=False,
    )
    m["is_overdue_clean"] = boolean_from_mixed_values(
        first_existing_col(m, ["is_overdue"], False),
        default=False,
    )
    m["is_due_now_clean"] = boolean_from_mixed_values(
        first_existing_col(m, ["is_due_now"], False),
        default=False,
    )
    m["available_clean"] = boolean_from_mixed_values(
        first_existing_col(m, ["AVAILABLE", "available"], True),
        default=True,
    )
    m["maintenance_type_clean"] = first_existing_col(
        m,
        ["maintenance_type", "service_types", "SERVICE_TYPES"],
        "",
    ).astype("string")
    m["related_component_clean"] = (
        first_existing_col(m, ["related_component"], "").astype("string")
        + " "
        + first_existing_col(m, ["related_component_1"], "").astype("string")
        + " "
        + first_existing_col(m, ["related_component_2"], "").astype("string")
    )

    m = m[
        m["maintenance_event_date"].notna()
        & m["SERIAL"].notna()
        & m["model_family"].isin(TARGET_MODEL_FAMILIES)
    ]
    return m.sort_values(["SERIAL", "maintenance_event_date"]).reset_index(drop=True)


def standardize_warranty(warranty: pd.DataFrame) -> pd.DataFrame:
    """Standardize the optional warranty table used to create claim_next_45d."""
    w = warranty.copy()
    w["SERIAL"] = normalize_key(first_existing_col(w, ["ZZSERNR", "SERIAL", "serial_number", "Serial"]))
    w["warranty_failure_date"] = parse_dt(
        first_existing_col(w, ["ZZFAILDAT", "warranty_failure_date", "failure_date"], pd.NaT)
    )
    w = w[w["SERIAL"].notna() & w["warranty_failure_date"].notna()]
    return w.sort_values(["SERIAL", "warranty_failure_date"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Snapshot backbone
# -----------------------------------------------------------------------------
def build_machine_universe(
    fault: pd.DataFrame,
    machine: pd.DataFrame,
    pm: pd.DataFrame,
    max_machines: Optional[int] = None,
) -> pd.DataFrame:
    """Create the D51/D61/D71 machine universe.

    The machine file is preferred for static model information. However, because
    the current machine.csv may be a proxy or duplicate of the fault-code file,
    this function also looks at the standardized fault and maintenance tables so
    that machines are not accidentally lost.
    """
    chunks: list[pd.DataFrame] = []

    for df in [machine, fault, pm]:
        d = df.copy()
        d["SERIAL"] = normalize_key(first_existing_col(d, ["SERIAL", "serial_number", "Serial", "ZZSERNR"]))
        d["full_model"] = first_existing_col(
            d,
            ["full_model", "FULL_MODEL", "MODEL", "model", "ZZMATNR"],
            "",
        ).astype("string").str.strip()
        d["model_family"] = d["full_model"].map(model_family)
        chunks.append(d[["SERIAL", "full_model", "model_family"]].dropna(subset=["SERIAL"]))

    universe = pd.concat(chunks, ignore_index=True)
    universe = universe[universe["model_family"].isin(TARGET_MODEL_FAMILIES)]
    universe = (
        universe.sort_values(["SERIAL", "full_model"])
        .drop_duplicates("SERIAL", keep="last")
        .reset_index(drop=True)
    )

    if max_machines is not None:
        universe = universe.head(max_machines).copy()

    return universe


def build_snapshot_backbone(
    universe: pd.DataFrame,
    fault: pd.DataFrame,
    pm: pd.DataFrame,
    warranty: Optional[pd.DataFrame] = None,
    snapshot_freq_days: int = 14,
    min_snapshot_date: Optional[str] = None,
    max_snapshot_date: Optional[str] = None,
) -> pd.DataFrame:
    """Generate one row per machine per snapshot date.

    For each machine, the snapshot range runs from the first observed event date
    to the last observed event date. Optional min/max date arguments are useful
    for development runs or time-based train/validation/test construction.
    """
    event_frames = [
        fault[["SERIAL", "fault_event_date"]].rename(columns={"fault_event_date": "event_date"}),
        pm[["SERIAL", "maintenance_event_date"]].rename(columns={"maintenance_event_date": "event_date"}),
    ]
    if warranty is not None and not warranty.empty:
        event_frames.append(
            warranty[["SERIAL", "warranty_failure_date"]].rename(columns={"warranty_failure_date": "event_date"})
        )

    events = pd.concat(event_frames, ignore_index=True).dropna(subset=["event_date"])
    bounds = events.groupby("SERIAL")["event_date"].agg(
        first_observed_date="min",
        last_observed_date="max",
    ).reset_index()
    backbone_base = universe.merge(bounds, on="SERIAL", how="inner")

    min_dt = pd.to_datetime(min_snapshot_date) if min_snapshot_date else None
    max_dt = pd.to_datetime(max_snapshot_date) if max_snapshot_date else None

    rows: list[tuple] = []
    total_machines = len(backbone_base)
    progress(f"Creating snapshot dates for {total_machines:,} machines...")

    for machine_idx, r in enumerate(backbone_base.itertuples(index=False), start=1):
        start = pd.Timestamp(r.first_observed_date).normalize()
        end = pd.Timestamp(r.last_observed_date).normalize()

        if min_dt is not None:
            start = max(start, min_dt)
        if max_dt is not None:
            end = min(end, max_dt)
        if start > end:
            continue

        for snap in pd.date_range(start=start, end=end, freq=f"{snapshot_freq_days}D"):
            rows.append((r.SERIAL, r.full_model, r.model_family, snap))

        if machine_idx == 1 or machine_idx % PROGRESS_EVERY_MACHINES == 0 or machine_idx == total_machines:
            progress(f"Backbone progress: {machine_idx:,}/{total_machines:,} machines; {len(rows):,} snapshot rows")

    return pd.DataFrame(rows, columns=["SERIAL", "full_model", "model_family", "snapshot_date"])


# -----------------------------------------------------------------------------
# Fault feature engineering
# -----------------------------------------------------------------------------
def fault_features_for_machine(snap_m: pd.DataFrame, f_m: pd.DataFrame) -> list[dict]:
    """Create fault-derived features for one machine across all snapshot dates."""
    out: list[dict] = []
    dates = f_m["fault_event_date"] if "fault_event_date" in f_m.columns else pd.Series(dtype="datetime64[ns]")

    for snap in snap_m["snapshot_date"]:
        # All feature calculations below use dates before the snapshot only.
        before = f_m[dates < snap]
        w90 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=90)]
        w30 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=30)]
        w7 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=7)]
        prev30 = before[
            (before["fault_event_date"] >= snap - pd.Timedelta(days=60))
            & (before["fault_event_date"] < snap - pd.Timedelta(days=30))
        ]
        severe_before = before[before["event_action_level_clean"].isin(["L03", "L04", "L05"])]

        row: dict = {"snapshot_date": snap}

        # Recent fault volume and trend features.
        row["fault_count_7d"] = len(w7)
        row["fault_count_30d"] = len(w30)
        row["fault_count_90d"] = len(w90)
        row["fault_count_previous_30d"] = len(prev30)
        row["fault_growth_rate"] = row["fault_count_30d"] - row["fault_count_previous_30d"]
        row["days_since_last_fault"] = days_between(snap, before["fault_event_date"].max())
        row["days_since_last_severe_fault"] = days_between(snap, severe_before["fault_event_date"].max())

        # Normalize fault volume by recent SMR growth when SMR is available.
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

        # Fault diversity and repetition features.
        row["unique_fault_code_count_90d"] = w90["fault_code_clean"].replace("", np.nan).nunique()
        row["repeat_fault_ratio_90d"] = ratio(row["fault_count_90d"], max(row["unique_fault_code_count_90d"], 1))
        row["unique_component_count_90d"] = w90["applicable_component_clean"].replace("", np.nan).nunique()

        # Mechanical/electrical flags from the enriched fault-code file.
        row["mechanical_fault_count_90d"] = int((w90["is_mechanical_failure_code_clean"] == 1).sum())
        row["mechanical_fault_count_30d"] = int((w30["is_mechanical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_90d"] = int((w90["is_electrical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_30d"] = int((w30["is_electrical_failure_code_clean"] == 1).sum())

        # Severity/action-level features.
        for lvl in ["L01", "L02", "L03", "L04"]:
            row[f"action_{lvl}_count_90d"] = int((w90["event_action_level_clean"] == lvl).sum())
        row["max_action_level_90d"] = w90["action_level_num_clean"].max()
        row["sum_log_occurrence_90d"] = w90["log_occurrence_clean"].sum()
        row["max_log_occurrence_90d"] = w90["log_occurrence_clean"].max()
        row["occurrence_severity_score_90d"] = w90["occurrence_class_clean"].sum()

        # Evidence-strength features from enriched failure-code logic.
        row["strong_fault_count_90d"] = int((w90["evidence_strength_clean"] == "STRONG").sum())
        row["moderate_fault_count_90d"] = int(w90["evidence_strength_clean"].isin(["MEDIUM", "MODERATE"]).sum())
        event_w90 = w90[
            w90["evidence_group_clean"].eq("EVENT")
            | w90["history_category_clean"].str.contains("event", na=False)
        ]
        context_w90 = w90[
            w90["evidence_group_clean"].eq("CONTEXT")
            | w90["history_category_clean"].str.contains("context", na=False)
        ]
        row["max_event_evidence_score_90d"] = event_w90["failure_code_evidence_score_clean"].max()
        row["avg_event_evidence_score_90d"] = event_w90["failure_code_evidence_score_clean"].mean()
        row["max_context_evidence_score_90d"] = context_w90["failure_code_evidence_score_clean"].max()

        # Component bucket features based on related/applicable component text.
        component_counts = {}
        for comp, pats in COMPONENT_PATTERNS.items():
            feature = {
                "engine": "engine_fault_count_90d",
                "hydraulic": "hydraulic_fault_count_90d",
                "powertrain": "powertrain_fault_count_90d",
                "scr": "scr_fault_count_90d",
                "workequipment": "workequipment_fault_count_90d",
                "cooling": "cooling_fault_count_90d",
            }.get(comp)
            if feature:
                cnt = int(contains_any(w90["related_component_clean"], pats).sum())
                row[feature] = cnt
                component_counts[feature] = cnt
        row["top_component_fault_ratio_90d"] = ratio(
            max(component_counts.values()) if component_counts else 0,
            row["fault_count_90d"],
        )

        # Extra non-frozen helper columns are useful for QA and model debugging.
        row["has_fault_90d"] = int(row["fault_count_90d"] > 0)
        row["smr_latest_before_snapshot"] = smr_latest
        row["smr_delta_90d"] = smr_delta_90d
        out.append(row)

    return out


# -----------------------------------------------------------------------------
# Maintenance feature engineering
# -----------------------------------------------------------------------------
def maintenance_features_for_machine(snap_m: pd.DataFrame, m_m: pd.DataFrame) -> list[dict]:
    """Create maintenance-monitor features for one machine across snapshots."""
    out: list[dict] = []
    dates = m_m["maintenance_event_date"] if "maintenance_event_date" in m_m.columns else pd.Series(dtype="datetime64[ns]")

    for snap in snap_m["snapshot_date"]:
        before = m_m[dates < snap]
        w180 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=180)]
        w90 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=90)]
        reset180 = w180[w180["is_monitor_reset_clean"]]
        reset90 = w90[w90["is_monitor_reset_clean"]]

        # Current maintenance state is the latest available record for each item
        # before the snapshot. EVENT_NAME_EN is the best item name in the current
        # enriched file. If it is missing, current-state features safely become 0.
        latest_cols = ["EVENT_NAME_EN"] if "EVENT_NAME_EN" in before.columns else []
        if latest_cols and len(before):
            current = before.sort_values("maintenance_event_date").groupby(latest_cols, dropna=False).tail(1)
            current = current[current["available_clean"]]
        else:
            current = before.tail(0)

        row: dict = {"snapshot_date": snap}

        # Maintenance frequency and reset behavior.
        row["maintenance_events_180d"] = len(w180)
        row["monitor_reset_count_180d"] = len(reset180)
        row["maintenance_reset_ratio_180d"] = ratio(row["monitor_reset_count_180d"], row["maintenance_events_180d"])
        row["maintenance_events_90d"] = len(w90)
        row["monitor_reset_count_90d"] = len(reset90)

        # Current maintenance due/overdue status.
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

        # Component-level reset and overdue features.
        for comp, pats in COMPONENT_PATTERNS.items():
            reset_feature = {
                "engine": "engine_reset_count_180d",
                "powertrain": "transmission_reset_count_180d",
                "final_drive": "final_drive_reset_count_180d",
                "cooling": "cooling_system_reset_count_180d",
                "scr": "urea_scr_system_reset_count_180d",
            }.get(comp)
            overdue_feature = {
                "engine": "engine_overdue_item_count",
                "powertrain": "transmission_overdue_item_count",
                "final_drive": "final_drive_overdue_item_count",
                "cooling": "cooling_system_overdue_item_count",
                "scr": "urea_scr_system_overdue_item_count",
            }.get(comp)

            if reset_feature:
                row[reset_feature] = int(contains_any(reset180["related_component_clean"], pats).sum())
            if overdue_feature:
                row[overdue_feature] = (
                    int(contains_any(current[current["is_overdue_clean"]]["related_component_clean"], pats).sum())
                    if len(current)
                    else 0
                )

        # Maintenance type features.
        for mtype, pats in MAINTENANCE_TYPE_PATTERNS.items():
            row[f"{mtype}_reset_count_180d"] = int(contains_any(reset180["maintenance_type_clean"], pats).sum())

        row["unique_maintenance_type_count_180d"] = reset180["maintenance_type_clean"].replace("", np.nan).nunique()
        row["days_since_last_reset"] = days_between(snap, reset180["maintenance_event_date"].max())

        oil_reset = reset180[contains_any(reset180["maintenance_type_clean"], MAINTENANCE_TYPE_PATTERNS["oil"])]
        filter_reset = reset180[contains_any(reset180["maintenance_type_clean"], MAINTENANCE_TYPE_PATTERNS["filter"])]
        row["days_since_last_oil_reset"] = days_between(snap, oil_reset["maintenance_event_date"].max())
        row["days_since_last_filter_reset"] = days_between(snap, filter_reset["maintenance_event_date"].max())

        # SMR since latest reset. If SMR is unavailable, this is filled to 0 in
        # finalize_snapshot_df.
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


def build_features(snapshots: pd.DataFrame, fault: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Build all source-derived features and merge them onto the backbone."""
    all_rows: list[pd.DataFrame] = []
    fault_groups = {k: v for k, v in fault.groupby("SERIAL", sort=False)}
    pm_groups = {k: v for k, v in pm.groupby("SERIAL", sort=False)}

    total_machines = snapshots["SERIAL"].nunique() if not snapshots.empty else 0
    total_snapshot_rows = len(snapshots)
    processed_snapshot_rows = 0

    progress(
        f"Building features for {total_machines:,} machines "
        f"and {total_snapshot_rows:,} snapshot rows..."
    )

    for machine_idx, (serial, snap_m) in enumerate(snapshots.groupby("SERIAL", sort=False), start=1):
        base = snap_m[["SERIAL", "full_model", "model_family", "snapshot_date"]].copy()
        f_m = fault_groups.get(serial, fault.iloc[0:0])
        m_m = pm_groups.get(serial, pm.iloc[0:0])

        f_feat = pd.DataFrame(fault_features_for_machine(snap_m, f_m))
        m_feat = pd.DataFrame(maintenance_features_for_machine(snap_m, m_m))
        merged = base.merge(f_feat, on="snapshot_date", how="left").merge(m_feat, on="snapshot_date", how="left")
        all_rows.append(merged)

        processed_snapshot_rows += len(snap_m)

        if (
            machine_idx == 1
            or machine_idx % PROGRESS_EVERY_MACHINES == 0
            or machine_idx == total_machines
        ):
            progress(
                f"Feature progress: {machine_idx:,}/{total_machines:,} machines; "
                f"{processed_snapshot_rows:,}/{total_snapshot_rows:,} snapshot rows processed"
            )

    features = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # Guarantee every frozen feature exists. This makes downstream training code
    # stable even if a source column is missing in one extract.
    for col in FROZEN_FEATURES:
        if col not in features.columns:
            features[col] = np.nan

    progress(f"Feature build complete. Feature dataframe rows: {len(features):,}")

    return features


# -----------------------------------------------------------------------------
# Optional warranty target
# -----------------------------------------------------------------------------
def add_warranty_target(
    snapshots: pd.DataFrame,
    warranty: Optional[pd.DataFrame],
    horizon_days: int = 45,
) -> pd.DataFrame:
    """Add claim_next_45d from warranty failure dates.

    A positive label means the same machine has at least one warranty failure
    date strictly after the snapshot and on/before snapshot + horizon_days.
    """
    df = snapshots.copy()
    if warranty is None or warranty.empty:
        df["claim_next_45d"] = np.nan
        return df

    target: list[int] = []
    war_groups = {
        k: v["warranty_failure_date"].sort_values().to_numpy(dtype="datetime64[ns]")
        for k, v in warranty.groupby("SERIAL")
    }

    total_rows = len(df)
    progress(f"Creating warranty target for {total_rows:,} snapshot rows...")

    for row_idx, r in enumerate(df[["SERIAL", "snapshot_date"]].itertuples(index=False), start=1):
        dates = war_groups.get(r.SERIAL)
        if dates is None or len(dates) == 0:
            target.append(0)
        else:
            snap = np.datetime64(r.snapshot_date)
            end = np.datetime64(r.snapshot_date + pd.Timedelta(days=horizon_days))
            has_claim = bool(((dates > snap) & (dates <= end)).any())
            target.append(int(has_claim))

        if total_rows and (row_idx == 1 or row_idx % 100000 == 0 or row_idx == total_rows):
            progress(f"Target progress: {row_idx:,}/{total_rows:,} snapshot rows")

    df["claim_next_45d"] = target
    return df


# -----------------------------------------------------------------------------
# Final missing-value handling and feature validation
# -----------------------------------------------------------------------------
def finalize_snapshot_df(
    df: pd.DataFrame,
    feature_freeze_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Fill feature missing values and order columns for model training."""
    out = df.copy()

    # Count features: no source event in the lookback window means zero events.
    for col in COUNT_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    # Recency features: no prior event gets a large sentinel.
    for col in RECENCY_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna(9999).astype(float)

    # Ratios and trend rates: default to zero when denominator/history is missing.
    ratio_cols = [c for c in FROZEN_FEATURES if c.endswith("ratio_90d") or c.endswith("ratio_180d") or c.endswith("rate")]
    for col in ratio_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(float)

    # Continuous score/amount/SMR-like fields: fill to zero for the first version.
    # If later model experiments show missingness itself is predictive, add explicit
    # missingness indicator columns before this fill.
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

    # Optional validation against the Excel feature-freeze workbook.
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

    ordered_cols = ["SERIAL", "full_model", "model_family", "snapshot_date"]
    if "claim_next_45d" in out.columns:
        ordered_cols.append("claim_next_45d")
    ordered_cols += FROZEN_FEATURES
    extra_cols = [c for c in out.columns if c not in ordered_cols]

    # If no snapshots were produced, return a correctly shaped empty dataframe.
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = pd.Series(dtype="float64")

    return out[ordered_cols + extra_cols].sort_values(["SERIAL", "snapshot_date"]).reset_index(drop=True)


def save_snapshot_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the snapshot dataframe as CSV or Parquet.

    Parquet is preferred for large data. If pyarrow/fastparquet is not installed,
    the script falls back to CSV instead of failing silently.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return output_path

    try:
        df.to_parquet(output_path, index=False)
        return output_path
    except ImportError:
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        print(f"Parquet engine not installed. Saved CSV fallback instead: {fallback}")
        return fallback


# -----------------------------------------------------------------------------
# End-to-end orchestration
# -----------------------------------------------------------------------------
def build_snapshot_dataframe(
    fault_codes_path: str | Path,
    machine_path: str | Path,
    maintenance_path: str | Path,
    output_dir: str | Path = "data_preparation/output",
    warranty_path: Optional[str | Path] = None,
    feature_freeze_path: Optional[str | Path] = None,
    snapshot_freq_days: int = 14,
    horizon_days: int = 45,
    min_snapshot_date: Optional[str] = None,
    max_snapshot_date: Optional[str] = None,
    write_cleaning_reports: bool = True,
    max_machines: Optional[int] = None,
) -> pd.DataFrame:
    """Build the final model-ready snapshot dataframe.

    This is the main function to call from notebooks or pipelines. It handles:
        1. CSV loading.
        2. Per-file missing-value detection and light cleaning.
        3. Source standardization.
        4. Snapshot backbone creation.
        5. Feature engineering.
        6. Optional warranty target creation.
        7. Final missing-value filling and feature validation.
    """
    progress("Starting snapshot dataframe build...")

    all_profiles: list[pd.DataFrame] = []
    cleaning_summaries: list[dict] = []

    progress("Step 1/7: Loading and cleaning fault_codes file...")
    raw_fault, profiles, summary = load_and_clean_csv(
        fault_codes_path,
        dataset_name="fault_codes",
        output_dir=output_dir,
        write_cleaning_reports=write_cleaning_reports,
    )
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"fault_codes loaded: {len(raw_fault):,} rows, {len(raw_fault.columns):,} columns")

    progress("Step 2/7: Loading and cleaning machine file...")
    raw_machine, profiles, summary = load_and_clean_csv(
        machine_path,
        dataset_name="machine",
        output_dir=output_dir,
        write_cleaning_reports=write_cleaning_reports,
    )
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"machine loaded: {len(raw_machine):,} rows, {len(raw_machine.columns):,} columns")

    progress("Step 3/7: Loading and cleaning maintenance file...")
    raw_pm, profiles, summary = load_and_clean_csv(
        maintenance_path,
        dataset_name="maintenance",
        output_dir=output_dir,
        write_cleaning_reports=write_cleaning_reports,
    )
    all_profiles.extend(profiles)
    cleaning_summaries.append(summary)
    progress(f"maintenance loaded: {len(raw_pm):,} rows, {len(raw_pm.columns):,} columns")

    raw_warranty = None
    if warranty_path:
        progress("Optional warranty file found. Loading and cleaning warranty file...")
        raw_warranty, profiles, summary = load_and_clean_csv(
            warranty_path,
            dataset_name="warranty",
            output_dir=output_dir,
            write_cleaning_reports=write_cleaning_reports,
        )
        all_profiles.extend(profiles)
        cleaning_summaries.append(summary)
        progress(f"warranty loaded: {len(raw_warranty):,} rows, {len(raw_warranty.columns):,} columns")
    else:
        progress("No warranty file found. claim_next_45d will be left blank.")

    if write_cleaning_reports:
        progress("Writing missing-value and cleaning summary reports...")
        write_combined_cleaning_reports(output_dir, all_profiles, cleaning_summaries)
        progress(f"Cleaning reports saved to: {output_dir}")

    progress("Step 4/7: Standardizing source tables...")
    fault = standardize_faults(raw_fault)
    pm = standardize_maintenance(raw_pm)

    progress(f"Standardized fault rows after filtering: {len(fault):,}")
    progress(f"Standardized maintenance rows after filtering: {len(pm):,}")

    progress("Step 5/7: Building machine universe...")
    universe = build_machine_universe(fault, raw_machine, pm, max_machines=max_machines)
    progress(f"Machine universe size: {len(universe):,}")

    warranty = None
    if raw_warranty is not None:
        warranty = standardize_warranty(raw_warranty)
        warranty = warranty[warranty["SERIAL"].isin(universe["SERIAL"])]
        progress(f"Standardized warranty rows after filtering to machine universe: {len(warranty):,}")

    progress("Step 6/7: Building snapshot backbone...")
    snapshots = build_snapshot_backbone(
        universe=universe,
        fault=fault,
        pm=pm,
        warranty=warranty,
        snapshot_freq_days=snapshot_freq_days,
        min_snapshot_date=min_snapshot_date,
        max_snapshot_date=max_snapshot_date,
    )
    progress(
        f"Snapshot backbone created: {len(snapshots):,} rows "
        f"across {snapshots['SERIAL'].nunique() if not snapshots.empty else 0:,} machines"
    )

    progress("Step 7/7: Engineering features. This is usually the slowest step...")
    features = build_features(snapshots, fault, pm)

    progress("Adding warranty target if available...")
    features = add_warranty_target(features, warranty, horizon_days=horizon_days)

    progress("Finalizing missing values and validating frozen features...")
    features = finalize_snapshot_df(features, feature_freeze_path=feature_freeze_path)

    progress(f"Snapshot dataframe build complete. Final shape: {features.shape[0]:,} rows x {features.shape[1]:,} columns")

    return features


def require_existing_path(path_value: str | Path, label: str) -> Path:
    """Validate required input paths from config.py before running the build."""
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def optional_existing_path(path_value: Optional[str | Path]) -> Optional[Path]:
    """Return an optional configured path only when it exists."""
    if path_value in (None, "", "None"):
        return None
    path = Path(path_value)
    return path if path.exists() else None


def main() -> None:
    """Config-driven entry point.

    Edit config.py, then run:

        python build_snapshot_dataframe.py

    No command-line arguments are required. This avoids long commands and keeps
    your project parameters version-controlled in one place.
    """
    input_dir = Path(cfg("INPUT_DIR", "enriched_data"))
    output_dir = Path(cfg("OUTPUT_DIR", "data_preparation/output"))

    # Required source files. These default to files inside enriched_data/.
    fault_codes_path = require_existing_path(
        cfg("FAULT_CODES_PATH", input_dir / "fault_codes.csv"),
        "FAULT_CODES_PATH",
    )
    machine_path = require_existing_path(
        cfg("MACHINE_PATH", input_dir / "machine.csv"),
        "MACHINE_PATH",
    )
    maintenance_path = require_existing_path(
        cfg("MAINTENANCE_PATH", input_dir / "maintenance.csv"),
        "MAINTENANCE_PATH",
    )

    # Optional source files. Missing warranty.csv is allowed; in that case the
    # script still creates the feature dataframe but leaves claim_next_45d blank.
    warranty_path = optional_existing_path(cfg("WARRANTY_PATH", input_dir / "warranty.csv"))
    feature_freeze_path = optional_existing_path(
        cfg("FEATURE_FREEZE_PATH", input_dir / "xgb_feature_freeze.xlsx")
    )

    output_path = Path(cfg("OUTPUT_PATH", output_dir / "snapshot_dataframe.parquet"))

    print("Using config.py" if run_config is not None else "config.py not found; using built-in defaults")
    print(f"Input folder: {input_dir}")
    print(f"Output path: {output_path}")
    print(f"Target model families: {', '.join(TARGET_MODEL_FAMILIES)}")

    df = build_snapshot_dataframe(
        fault_codes_path=fault_codes_path,
        machine_path=machine_path,
        maintenance_path=maintenance_path,
        output_dir=output_dir,
        warranty_path=warranty_path,
        feature_freeze_path=feature_freeze_path,
        snapshot_freq_days=int(cfg("SNAPSHOT_FREQ_DAYS", 14)),
        horizon_days=int(cfg("HORIZON_DAYS", 45)),
        min_snapshot_date=cfg("MIN_SNAPSHOT_DATE", None),
        max_snapshot_date=cfg("MAX_SNAPSHOT_DATE", None),
        write_cleaning_reports=bool(cfg("WRITE_CLEANING_REPORTS", True)),
        max_machines=cfg("MAX_MACHINES", None),
    )

    progress("Saving snapshot dataframe...")
    saved_path = save_snapshot_dataframe(df, output_path)

    print(f"Saved snapshot dataframe: {saved_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Cleaning reports folder: {output_dir}")
    if "claim_next_45d" in df.columns:
        print("Target distribution:")
        print(df["claim_next_45d"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
