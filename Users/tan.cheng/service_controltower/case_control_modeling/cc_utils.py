"""Shared utilities for window-based case-control modeling."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# General IO helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(obj: Mapping, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_source_file(source_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        path = Path(source_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of these source files were found in {source_dir}: {list(candidates)}"
    )


def read_csv_selected(path: Path, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if columns is None:
        return pd.read_csv(path, low_memory=False)
    wanted = set(columns)
    return pd.read_csv(path, low_memory=False, usecols=lambda c: c in wanted)


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def clean_serial(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return re.sub(r"\s+", "", text)


def clean_model(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "-", text)
    return text


def normalize_machine_key_from_model_serial(model, serial) -> str:
    model_clean = clean_model(model)
    serial_clean = clean_serial(serial)
    if model_clean and serial_clean:
        return f"{model_clean}-{serial_clean}"
    return ""


def extract_model_serial_from_machine_id(machine_id) -> Tuple[str, str]:
    if pd.isna(machine_id):
        return "", ""
    text = str(machine_id).strip().upper().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    # Common forms: D71EX-24 70155, D71EX-24-70155
    m = re.match(r"^(.+?)[\s-]+(\d{4,})$", text)
    if m:
        return clean_model(m.group(1)), clean_serial(m.group(2))
    return "", ""


def add_machine_key(
    df: pd.DataFrame,
    model_col: Optional[str],
    serial_col: Optional[str],
    machine_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()
    if model_col in out.columns and serial_col in out.columns:
        out["full_model_norm"] = out[model_col].map(clean_model)
        out["serial_norm"] = out[serial_col].map(clean_serial)
    else:
        out["full_model_norm"] = ""
        out["serial_norm"] = ""

    if machine_col in out.columns:
        missing = (out["full_model_norm"].eq("")) | (out["serial_norm"].eq(""))
        parsed = out.loc[missing, machine_col].map(extract_model_serial_from_machine_id)
        if len(parsed):
            out.loc[missing, "full_model_norm"] = parsed.map(lambda x: x[0])
            out.loc[missing, "serial_norm"] = parsed.map(lambda x: x[1])

    out["machine_key"] = [
        normalize_machine_key_from_model_serial(m, s)
        for m, s in zip(out["full_model_norm"], out["serial_norm"])
    ]
    out.loc[out["machine_key"].eq("-"), "machine_key"] = ""
    return out


def _filter_event_dates(
    df: pd.DataFrame,
    date_col: str,
    min_date: Optional[str],
    max_date: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = parse_date(out[date_col])
    out = out.dropna(subset=["machine_key", date_col])
    out = out[out["machine_key"].astype(str).str.len() > 0]
    if min_date is not None:
        out = out[out[date_col] >= pd.Timestamp(min_date)]
    if max_date is not None:
        out = out[out[date_col] <= pd.Timestamp(max_date)]
    return out


# -----------------------------------------------------------------------------
# Source loading
# -----------------------------------------------------------------------------
def load_warranty(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.WARRANTY_FILE_CANDIDATES)
    cols = [
        "machine_id", "claim_number", "local_date", "claim_type_description",
        "warranty_claim_data_source", "full_model", "serial", "failure_smr",
        "critical_fail_part_number", "claim_amount", "CLAIM_AMOUNT",
        "total_claim_amount", "total_amount", "net_claim_amount", "paid_amount",
        "claim_type", "claim_category",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "full_model", "serial", "machine_id")
    df = _filter_event_dates(df, "local_date", config.MIN_CLAIM_DATE, config.MAX_CLAIM_DATE)
    df["claim_date"] = df["local_date"]
    df["failure_smr"] = pd.to_numeric(df.get("failure_smr"), errors="coerce")
    for col in ["claim_amount", "CLAIM_AMOUNT", "total_claim_amount", "total_amount", "net_claim_amount", "paid_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["critical_fail_part_number_clean"] = (
        df.get("critical_fail_part_number", "").astype(str).str.strip().str.lower()
    )
    invalid = set(str(x).strip().lower() for x in getattr(config, "INVALID_CRITICAL_PART_VALUES", set()))
    df["has_valid_critical_part"] = ~df["critical_fail_part_number_clean"].isin(invalid)
    if bool(getattr(config, "KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS", False)):
        df = df[df["has_valid_critical_part"]].copy()
    return df.reset_index(drop=True)


def load_fault_codes(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.FAULT_CODES_FILE_CANDIDATES)
    cols = [
        "serial_number", "full_model", "machine_id", "event_date", "event_time",
        "UPDATE_DATETIME", "fault_code", "event_error_name_en", "event_action_level",
        "occurrence_count", "log_occurrence_count", "occurrence_class", "smr_hours",
        "applicable_component", "related_component", "related_component_1",
        "is_mechanical_failure_code", "is_electrical_failure_code", "action_level_num",
        "failure_code_evidence_score", "failure_code_evidence_strength_class",
        "failure_code_evidence_group", "history_category",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "full_model", "serial_number", "machine_id")
    df = _filter_event_dates(df, "event_date", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    numeric_cols = [
        "occurrence_count", "log_occurrence_count", "occurrence_class", "smr_hours",
        "is_mechanical_failure_code", "is_electrical_failure_code", "action_level_num",
        "failure_code_evidence_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_fluid_samples(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.FLUID_SAMPLES_FILE_CANDIDATES)
    cols = [
        "FULL_MODEL", "SERIAL", "machine_id", "TELEMETRY_SMR_NUMERIC", "LAB_NAME",
        "LABS_SAMPLE_NUMBER", "sample_drawn_date", "sample_result_severity_order",
        "Ag_Silver_PPM", "Al_Aluminum_PPM", "Cr_Chromium_PPM", "Cu_Copper_PPM",
        "Fe_Iron_PPM", "Ni_Nickel_PPM", "Pb_Lead_PPM", "Sn_Tin_PPM",
        "Ti_Titanium_PPM", "V_Vanadium_PPM",
        "EthyleneGlycol_Ethylene_Glycol_PERCENT", "Fuel_Fuel_PERCENT",
        "Gly_Glycol_PERCENT", "K_Potassium_PPM", "Li_Lithium_PPM", "Na_Sodium_PPM",
        "PolypropyleneGlycol_Polypropylene_Glycol_PERCENT", "Sediment_Sediment_MG_PER_L",
        "Si_Silicon_PPM", "Solids_Solids_PERCENT", "Soot_Soot_Abs",
        "Soot_Soot_Abs_cm", "Soot_Soot_METHOD_DEPENDENT", "Soot_Soot_PERCENT",
        "Water_Water_PERCENT",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "FULL_MODEL", "SERIAL", "machine_id")
    df = _filter_event_dates(df, "sample_drawn_date", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    non_numeric = {"FULL_MODEL", "SERIAL", "machine_id", "LAB_NAME", "LABS_SAMPLE_NUMBER", "sample_drawn_date"}
    for col in [c for c in cols if c not in non_numeric]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_maintenance(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.MAINTENANCE_FILE_CANDIDATES)
    cols = [
        "full_model", "machine_id", "SERIAL", "EVENT_NAME_EN", "event_date",
        "event_time", "UPDATE_DATETIME", "smr_hours", "remaining_hours", "INTERVAL_HOURS",
        "is_monitor_reset", "is_overdue", "is_due_now", "is_notice_or_status", "AVAILABLE",
        "related_component", "related_component_1", "related_component_2",
        "maintenance_type", "service_types", "SERVICE_TYPES",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "full_model", "SERIAL", "machine_id")
    df = _filter_event_dates(df, "event_date", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    for col in ["smr_hours", "remaining_hours", "INTERVAL_HOURS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["is_monitor_reset", "is_overdue", "is_due_now", "is_notice_or_status", "AVAILABLE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)
    return df.reset_index(drop=True)


def load_operation(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.OPERATION_FILE_CANDIDATES)
    cols = [
        "machine_id", "LOCAL_DATE", "full_model", "SERIAL", "smr_hours",
        "smr_delta_clean_since_prev_obs_hours", "smr_valid_for_utilization_flag",
        "smr_present_flag", "actual_working_hours_clean", "working_hours_clean",
        "actual_work_streak_through_current_day", "actual_work_day_flag",
        "actual_work_valid_flag", "actual_work_seconds_invalid_flag",
        "fuel_actual_work_conflict_flag", "last_actual_work_date_through_current_day",
        "engine_running_hours_clean", "engine_idling_hours_clean", "engine_idle_share_daily",
        "engine_running_day_flag", "engine_seconds_valid_flag", "engine_seconds_observed_flag",
        "throttle_full_hours_clean", "throttle_full_share_clean",
        "throttle_average_dial_position_clean", "throttle_observed_flag",
        "work_idle_sum_exceeds_engine_flag", "high_throttle_day_flag", "long_engine_day_flag",
        "traveling_hours_clean", "moving_back_forth_hours_clean", "steering_hours_clean",
        "travel_day_flag", "travel_usable_flag", "movement_observed_count",
        "auto_quick_shift_hours_clean", "manual_variable_shift_hours_clean",
    ]
    df = read_csv_selected(path, cols)
    df = df.dropna(subset=["machine_id", "LOCAL_DATE"])
    df = add_machine_key(df, "full_model", "SERIAL", "machine_id")
    df = _filter_event_dates(df, "LOCAL_DATE", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    non_numeric = {"machine_id", "LOCAL_DATE", "full_model", "SERIAL", "last_actual_work_date_through_current_day"}
    for col in [c for c in cols if c not in non_numeric]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "last_actual_work_date_through_current_day" in df.columns:
        df["last_actual_work_date_through_current_day"] = pd.to_datetime(
            df["last_actual_work_date_through_current_day"], errors="coerce"
        )
    return df.reset_index(drop=True)


def load_sources(config, include_operation: bool = True) -> Dict[str, pd.DataFrame]:
    """Load date-filtered source tables without additional SMR eligibility censoring."""
    sources: Dict[str, pd.DataFrame] = {
        "warranty": load_warranty(config),
        "fault": load_fault_codes(config),
        "fluid": load_fluid_samples(config),
        "maintenance": load_maintenance(config),
    }
    if include_operation:
        sources["operation"] = load_operation(config)
    return sources


# -----------------------------------------------------------------------------
# Claim episode and machine master helpers
# -----------------------------------------------------------------------------
def build_claim_episodes(warranty: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    rows = []
    episode_id = 0
    w = warranty.sort_values(["machine_key", "claim_date", "claim_number"], kind="mergesort")
    for machine_key, g in w.groupby("machine_key", dropna=False):
        g = g.sort_values("claim_date", kind="mergesort").reset_index(drop=True)
        current = []
        prev_date = None
        for _, row in g.iterrows():
            claim_date = row["claim_date"]
            if prev_date is None or (claim_date - prev_date).days > gap_days:
                if current:
                    episode_id += 1
                    rows.append(_summarize_episode(current, episode_id))
                current = [row]
            else:
                current.append(row)
            prev_date = claim_date
        if current:
            episode_id += 1
            rows.append(_summarize_episode(current, episode_id))
    return pd.DataFrame(rows).sort_values(["claim_date", "machine_key"], kind="mergesort").reset_index(drop=True)


def _summarize_episode(rows: List[pd.Series], episode_id: int) -> dict:
    df = pd.DataFrame(rows)
    first = df.sort_values("claim_date", kind="mergesort").iloc[0]
    claim_numbers = ";".join(sorted(df.get("claim_number", pd.Series([], dtype=str)).astype(str).unique()))
    claim_types = ";".join(sorted(df.get("claim_type_description", pd.Series([], dtype=str)).dropna().astype(str).unique()))
    critical_parts = ";".join(sorted(df.get("critical_fail_part_number", pd.Series([], dtype=str)).dropna().astype(str).unique()))
    return {
        "claim_episode_id": f"E{episode_id:07d}",
        "machine_key": first["machine_key"],
        "full_model": first.get("full_model_norm", first.get("full_model", "")),
        "serial": first.get("serial_norm", first.get("serial", "")),
        "claim_date": first["claim_date"],
        "episode_end_date": df["claim_date"].max(),
        "claim_count_in_episode": int(len(df)),
        "claim_numbers": claim_numbers,
        "claim_type_descriptions": claim_types,
        "critical_fail_part_numbers": critical_parts,
        "has_valid_critical_part_episode": bool(df.get("has_valid_critical_part", pd.Series([False])).any()),
        "min_failure_smr": pd.to_numeric(df.get("failure_smr", pd.Series(dtype=float)), errors="coerce").min(),
        "max_failure_smr": pd.to_numeric(df.get("failure_smr", pd.Series(dtype=float)), errors="coerce").max(),
    }


def build_source_coverage(sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    date_frames = []
    for name, date_col in [
        ("fault", "event_date"),
        ("fluid", "sample_drawn_date"),
        ("maintenance", "event_date"),
        ("operation", "LOCAL_DATE"),
    ]:
        if name in sources and not sources[name].empty:
            tmp = sources[name][["machine_key", date_col]].copy()
            tmp = tmp.rename(columns={date_col: "source_date"})
            tmp["source_name"] = name
            date_frames.append(tmp)
    if not date_frames:
        return pd.DataFrame(columns=["machine_key", "first_source_date", "last_source_date", "source_record_count_total"])
    all_dates = pd.concat(date_frames, ignore_index=True).dropna(subset=["machine_key", "source_date"])
    summary = (
        all_dates.groupby("machine_key", dropna=False)
        .agg(
            first_source_date=("source_date", "min"),
            last_source_date=("source_date", "max"),
            source_record_count_total=("source_date", "count"),
        )
        .reset_index()
    )
    pivot = (
        all_dates.pivot_table(index="machine_key", columns="source_name", values="source_date", aggfunc="count")
        .fillna(0)
        .astype(int)
        .add_prefix("source_count_total_")
        .reset_index()
    )
    return summary.merge(pivot, on="machine_key", how="left")


def build_machine_master(sources: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in sources.items():
        if str(name).startswith("_") or df is None or df.empty or "machine_key" not in df.columns:
            continue
        model_col = next(
            (c for c in ["full_model_norm", "full_model", "FULL_MODEL"] if c in df.columns),
            None,
        )
        serial_col = next(
            (c for c in ["serial_norm", "serial", "SERIAL", "serial_number"] if c in df.columns),
            None,
        )
        tmp = df[["machine_key"]].copy()
        tmp["full_model"] = df[model_col].map(clean_model) if model_col else ""
        tmp["serial"] = df[serial_col].map(clean_serial) if serial_col else ""
        tmp = tmp.drop_duplicates(["machine_key", "full_model", "serial"])
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["machine_key", "full_model", "serial"])

    def first_nonblank(values: pd.Series) -> str:
        cleaned = values.astype("string").fillna("").str.strip()
        cleaned = cleaned[cleaned.ne("")]
        return str(cleaned.iloc[0]) if len(cleaned) else ""

    # Metadata can be absent from one source but present in another. Consolidate
    # all sources per machine instead of keeping whichever source happened to be
    # encountered first.
    stacked = pd.concat(frames, ignore_index=True)
    master = (
        stacked.groupby("machine_key", dropna=False, sort=False)
        .agg(full_model=("full_model", first_nonblank), serial=("serial", first_nonblank))
        .reset_index()
    )
    coverage = build_source_coverage(sources)
    master = master.merge(coverage, on="machine_key", how="left")
    return master


def claim_dates_by_machine(episodes: pd.DataFrame) -> Dict[str, np.ndarray]:
    out = {}
    for m, g in episodes.groupby("machine_key"):
        out[m] = np.array(sorted(pd.to_datetime(g["claim_date"]).to_numpy()))
    return out


def select_positive_claims_for_window_config(
    episodes: pd.DataFrame,
    window_config: Mapping,
    config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select positive claim events for one configured observation window.

    Two simple modes are supported through config.POSITIVE_CLAIM_SELECTION_MODE:

    - "first": keep only the first claim event for each machine.
    - "multiple": keep the first claim event, plus later claim events only when
      the later event is at least lead_max_days after the immediately previous
      chronological claim event for the same machine.

    The threshold intentionally uses the window_config's lead_max_days. This
    ensures that every selected repeated claim has a full pre-claim monitoring
    window that does not include the immediately previous claim date. The
    selection does not compare failure causes, components, or critical parts.
    """

    if episodes.empty:
        empty = episodes.copy()
        return empty, empty

    mode = str(getattr(config, "POSITIVE_CLAIM_SELECTION_MODE", "first")).strip().lower()
    mode_aliases = {
        "first": "first",
        "first_claim": "first",
        "first_only": "first",
        "first_claim_only": "first",
        "multiple": "multiple",
        "multi": "multiple",
        "multiple_claims": "multiple",
        "recurrent": "multiple",
        "recurrent_claims": "multiple",
    }
    if mode not in mode_aliases:
        raise ValueError(
            "Unsupported POSITIVE_CLAIM_SELECTION_MODE="
            f"{getattr(config, 'POSITIVE_CLAIM_SELECTION_MODE', None)!r}. "
            "Use 'first' or 'multiple'."
        )
    mode = mode_aliases[mode]

    lead_max = int(window_config["lead_max_days"])
    e = episodes.copy()
    e["claim_date"] = pd.to_datetime(e["claim_date"], errors="coerce")
    e = e.dropna(subset=["machine_key", "claim_date"]).copy()
    sort_cols = ["machine_key", "claim_date"]
    if "claim_episode_id" in e.columns:
        sort_cols.append("claim_episode_id")
    e = e.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    e["claim_sequence_number"] = e.groupby("machine_key", dropna=False).cumcount() + 1
    e["machine_claim_event_count"] = e.groupby("machine_key", dropna=False)["claim_date"].transform("size")
    e["previous_claim_date_same_machine"] = e.groupby("machine_key", dropna=False)["claim_date"].shift(1)
    e["days_since_previous_claim_same_machine"] = (
        e["claim_date"] - e["previous_claim_date_same_machine"]
    ).dt.days.astype(float)
    e["is_first_claim_for_machine"] = e["claim_sequence_number"].eq(1)
    e["positive_claim_selection_mode"] = mode
    e["lead_max_days_threshold_for_repeat_claim"] = lead_max

    if mode == "first":
        e["selected_as_positive_claim"] = e["is_first_claim_for_machine"]
        e["claim_selection_reason"] = np.where(
            e["selected_as_positive_claim"],
            "first_claim_for_machine",
            "excluded_not_first_claim_for_machine",
        )
    else:
        gap_ok = e["days_since_previous_claim_same_machine"].ge(float(lead_max))
        e["selected_as_positive_claim"] = e["is_first_claim_for_machine"] | gap_ok.fillna(False)
        e["claim_selection_reason"] = np.where(
            e["is_first_claim_for_machine"],
            "first_claim_for_machine",
            np.where(
                gap_ok.fillna(False),
                "included_gap_from_previous_claim_ge_lead_max_days",
                "excluded_gap_from_previous_claim_lt_lead_max_days",
            ),
        )

    audit_cols = [
        "positive_claim_selection_mode",
        "lead_max_days_threshold_for_repeat_claim",
        "selected_as_positive_claim",
        "claim_selection_reason",
        "machine_key",
        "full_model",
        "serial",
        "claim_episode_id",
        "claim_date",
        "claim_sequence_number",
        "machine_claim_event_count",
        "is_first_claim_for_machine",
        "previous_claim_date_same_machine",
        "days_since_previous_claim_same_machine",
        "claim_count_in_episode",
        "claim_numbers",
        "critical_fail_part_numbers",
    ]
    audit_cols = [c for c in audit_cols if c in e.columns]
    audit = e[audit_cols].copy()
    selected = e[e["selected_as_positive_claim"]].copy().reset_index(drop=True)
    return selected, audit.reset_index(drop=True)


def has_claim_between(dates_by_machine: Mapping[str, np.ndarray], machine_key: str, start, end) -> bool:
    dates = dates_by_machine.get(machine_key)
    if dates is None or len(dates) == 0:
        return False
    start64 = np.datetime64(pd.Timestamp(start).to_datetime64())
    end64 = np.datetime64(pd.Timestamp(end).to_datetime64())
    idx = np.searchsorted(dates, start64, side="left")
    return idx < len(dates) and dates[idx] <= end64


def count_claims_before(dates_by_machine: Mapping[str, np.ndarray], machine_key: str, cutoff) -> Tuple[int, float]:
    dates = dates_by_machine.get(machine_key)
    if dates is None or len(dates) == 0:
        return 0, np.nan
    cutoff64 = np.datetime64(pd.Timestamp(cutoff).to_datetime64())
    idx = np.searchsorted(dates, cutoff64, side="left")
    if idx <= 0:
        return 0, np.nan
    latest = pd.Timestamp(dates[idx - 1])
    return int(idx), float((pd.Timestamp(cutoff) - latest).days)


# -----------------------------------------------------------------------------
# Evaluation-only future-claim horizon helpers
# -----------------------------------------------------------------------------
def _clean_horizon_days(values) -> List[int]:
    """Normalize one horizon or a nested/list-like horizon config into ints."""
    out: List[int] = []
    if values is None:
        return out
    if isinstance(values, (str, int, float, np.integer, np.floating)):
        values = [values]
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            for nested in _clean_horizon_days(value):
                if nested not in out:
                    out.append(nested)
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        h = int(value)
        if h >= 0 and h not in out:
            out.append(h)
    return out


def configured_evaluation_horizons(config) -> List[int]:
    """Return future-claim horizons materialized as eval target columns.

    EVALUATION_CLAIM_HORIZON_DAYS can be a scalar or a list.  This is the single
    source of truth for horizon sweep columns; the older
    EVALUATION_ADDITIONAL_CLAIM_HORIZON_DAYS parameter was removed.
    """
    horizons = []
    primary = getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None)
    horizons.extend(_clean_horizon_days(primary))
    final_primary = getattr(config, "FINAL_EVALUATION_CLAIM_HORIZON_DAYS", None)
    horizons.extend(_clean_horizon_days(final_primary))
    return sorted(set(horizons))


def future_claim_target_col(horizon_days: int) -> str:
    return f"eval_target_claim_within_next_{int(horizon_days)}d"


def next_claim_on_or_after(
    dates_by_machine: Mapping[str, np.ndarray],
    machine_key: str,
    cutoff,
    include_cutoff: bool = True,
) -> Tuple[pd.Timestamp, float]:
    """Return the first claim date on/after cutoff and days from cutoff.

    If include_cutoff is True, a claim on window_end counts as 0 days later.
    This is useful for lead_min_days=0 windows where the claim date equals the
    end of the observation window. If there is no later claim, returns NaT/NaN.
    """
    dates = dates_by_machine.get(machine_key)
    if dates is None or len(dates) == 0 or pd.isna(cutoff):
        return pd.NaT, np.nan
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff64 = np.datetime64(cutoff_ts.to_datetime64())
    side = "left" if include_cutoff else "right"
    idx = int(np.searchsorted(dates, cutoff64, side=side))
    if idx >= len(dates):
        return pd.NaT, np.nan
    claim_date = pd.Timestamp(dates[idx])
    return claim_date, float((claim_date - cutoff_ts).days)


def annotate_future_claim_outcomes(
    df: pd.DataFrame,
    claim_history_episodes: pd.DataFrame,
    config=None,
    horizons: Optional[Sequence[int]] = None,
    include_window_end: Optional[bool] = None,
) -> pd.DataFrame:
    """Add next-claim lead-time columns and evaluation-only target columns.

    This does not modify the training `target`.  The added columns are used by
    CV/validation/test metric code when EVALUATION_TARGET_MODE is set to
    claim_within_horizon, and are also useful for reviewing prediction lead time.
    """
    out = df.copy()
    if out.empty:
        return out
    if "window_end" not in out.columns or "machine_key" not in out.columns:
        return out
    if horizons is None:
        if config is not None:
            horizons = configured_evaluation_horizons(config)
        else:
            horizons = [90, 120, 180, 365]
    horizons = _clean_horizon_days(horizons)
    if include_window_end is None:
        include_window_end = True if config is None else bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True))

    dates_by_machine = claim_dates_by_machine(claim_history_episodes)
    out["window_end"] = pd.to_datetime(out["window_end"], errors="coerce")

    next_dates = []
    days_to_next = []
    for m, end in zip(out["machine_key"], out["window_end"]):
        claim_date, days = next_claim_on_or_after(dates_by_machine, m, end, include_cutoff=include_window_end)
        next_dates.append(claim_date)
        days_to_next.append(days)
    out["next_claim_date_on_or_after_window_end"] = pd.to_datetime(next_dates, errors="coerce")
    out["days_to_next_claim_on_or_after_window_end"] = pd.to_numeric(pd.Series(days_to_next, index=out.index), errors="coerce")
    out["has_future_claim_on_or_after_window_end"] = out["next_claim_date_on_or_after_window_end"].notna().astype(int)
    out["future_claim_lead_time_bucket"] = pd.cut(
        out["days_to_next_claim_on_or_after_window_end"],
        bins=[-0.1, 0, 30, 60, 90, 120, 180, 365, np.inf],
        labels=["0d", "1-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", "365d+"],
    ).astype(object)
    out.loc[out["days_to_next_claim_on_or_after_window_end"].isna(), "future_claim_lead_time_bucket"] = "no_future_claim_observed"

    for horizon in horizons:
        col = future_claim_target_col(horizon)
        days = out["days_to_next_claim_on_or_after_window_end"]
        out[col] = days.notna().astype(int)
        out.loc[days.isna() | (days > float(horizon)), col] = 0
        out[col] = out[col].astype(int)
    return out


def evaluation_target_settings(
    config,
    prefix: str = "",
    horizon_days: Optional[int] = None,
) -> Tuple[str, Optional[int], str]:
    """Return (mode, horizon_days, target_column_name) for evaluation metrics.

    When EVALUATION_CLAIM_HORIZON_DAYS is a list and no explicit horizon is
    supplied, the largest configured horizon is used as a safe default for
    scripts that do not perform a horizon sweep. Step 04 passes explicit horizons
    and evaluates every configured value.
    """
    if prefix:
        mode = getattr(config, f"{prefix}_EVALUATION_TARGET_MODE", None)
        horizon = getattr(config, f"{prefix}_EVALUATION_CLAIM_HORIZON_DAYS", None)
    else:
        mode = None
        horizon = None
    if mode is None:
        mode = getattr(config, "EVALUATION_TARGET_MODE", "training_target")
    mode = str(mode).strip().lower()
    if horizon_days is not None:
        horizon = horizon_days
    elif horizon is None:
        horizon = getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None)
    if mode in {"training", "train", "target", "training_target", "original", "original_target"}:
        return "training_target", None, "target"
    if mode in {"claim_within_horizon", "future_claim_horizon", "relaxed", "relaxed_future_claim"}:
        horizons = _clean_horizon_days(horizon)
        if not horizons:
            raise ValueError("EVALUATION_CLAIM_HORIZON_DAYS must be set when using claim_within_horizon evaluation.")
        h = int(max(horizons))
        return "claim_within_horizon", h, future_claim_target_col(h)
    raise ValueError(f"Unsupported evaluation target mode: {mode!r}")


def get_evaluation_target(
    df: pd.DataFrame,
    config,
    prefix: str = "",
    horizon_days: Optional[int] = None,
) -> Tuple[pd.Series, str, str, Optional[int]]:
    """Return the evaluation y vector without changing model training target.

    Future-claim labels are derived from the stored next-claim lead time on every
    call.  Materialized ``eval_target_claim_within_next_*`` columns are retained
    for inspection, but they are not trusted as the source of truth.  This keeps
    validation/test sample identities fixed while allowing the configured horizon
    list to change without accidentally reusing stale labels.
    """
    mode, horizon, col = evaluation_target_settings(config, prefix=prefix, horizon_days=horizon_days)
    if mode == "training_target":
        if col not in df.columns:
            raise ValueError("Dataset is missing required target column.")
        y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return y, col, mode, horizon

    lead_col = "days_to_next_claim_on_or_after_window_end"
    if lead_col in df.columns:
        days = pd.to_numeric(df[lead_col], errors="coerce")
        include_window_end = bool(
            getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True)
        )
        lower_bound = days.ge(0) if include_window_end else days.gt(0)
        y = (days.notna() & lower_bound & days.le(float(horizon))).astype(int)
        return y, col, mode, horizon

    if col not in df.columns:
        raise ValueError(
            f"Dataset is missing both {lead_col!r} and {col!r}. Re-run "
            "02_build_case_control_dataset.py so future-claim lead times are added."
        )
    y = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return y, col, mode, horizon


def future_claim_lead_time_summary(df: pd.DataFrame, y_eval: Optional[pd.Series] = None) -> dict:
    """Small summary of how far in the future claims occur in an evaluation set."""
    out = {}
    if "days_to_next_claim_on_or_after_window_end" not in df.columns:
        return out
    days = pd.to_numeric(df["days_to_next_claim_on_or_after_window_end"], errors="coerce")
    out["future_claim_observed_rows"] = int(days.notna().sum())
    out["future_claim_never_observed_rows"] = int(days.isna().sum())
    if days.notna().any():
        out["future_claim_days_min"] = float(days.min())
        out["future_claim_days_median"] = float(days.median())
        out["future_claim_days_mean"] = float(days.mean())
        out["future_claim_days_p90"] = float(days.quantile(0.90))
    if y_eval is not None:
        y = pd.Series(y_eval).astype(int).reset_index(drop=True)
        d = days.reset_index(drop=True)
        pos_days = d[y.eq(1)]
        out["evaluation_target_positive_rows"] = int(y.sum())
        out["evaluation_target_positive_rate"] = float(y.mean()) if len(y) else np.nan
        if pos_days.notna().any():
            out["evaluation_positive_days_median"] = float(pos_days.median())
            out["evaluation_positive_days_max"] = float(pos_days.max())
    return out


# -----------------------------------------------------------------------------
# Case-control row building
# -----------------------------------------------------------------------------
def window_config_name(window_config: Mapping) -> str:
    """Return the canonical window name derived from lead-day settings.

    WINDOW_CONFIGS no longer needs a redundant name field.  If older configs
    still provide name, it is ignored for file naming and group IDs so outputs
    remain compact and consistent.
    """

    return f"lead_{int(window_config['lead_max_days'])}_to_{int(window_config['lead_min_days'])}"


def normalize_negative_sampling_mode(value, setting_name: str = "NEGATIVE_SAMPLING_MODE") -> str:
    mode = str(value).strip().lower()
    aliases = {
        "controlled": "controlled",
        "matched": "controlled",
        "case_control": "controlled",
        "random": "random",
        "random_same_model": "random",
        "mixed": "mixed",
        "hybrid": "mixed",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported {setting_name}={mode!r}. "
            "Use 'controlled', 'random', or 'mixed'."
        )
    return aliases[mode]


def validate_negative_count(value, setting_name: str = "NEGATIVES_PER_POSITIVE_CASE") -> int:
    count = int(value)
    if count < 1:
        raise ValueError(f"{setting_name} must be at least 1; received {value!r}.")
    return count


def negatives_per_positive(config) -> int:
    return validate_negative_count(getattr(config, "NEGATIVES_PER_POSITIVE_CASE", 3))


def negative_sampling_mode(config) -> str:
    return normalize_negative_sampling_mode(
        getattr(config, "NEGATIVE_SAMPLING_MODE", "controlled")
    )


def controls_per_positive(config) -> int:
    """Backward-compatible alias used by older reporting code."""
    return negatives_per_positive(config)


def window_dataset_id(window_config: Mapping, config) -> str:
    lead_label = window_config_name(window_config)
    base = (
        f"{lead_label}__neg_{negative_sampling_mode(config)}_"
        f"{negatives_per_positive(config)}__features_{str(config.FEATURE_SET).lower()}"
    )
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", base)


def _normalized_split_ratios(config) -> dict:
    ratios = {
        "train": max(float(getattr(config, "TRAIN_RATIO", 0.70)), 0.0),
        "validation": max(float(getattr(config, "VALIDATION_RATIO", 0.15)), 0.0),
        "test": max(float(getattr(config, "TEST_RATIO", 0.15)), 0.0),
    }
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must be positive.")
    return {k: v / total for k, v in ratios.items()}


def _stable_hash_int(value: str, random_state: int) -> int:
    payload = f"{int(random_state)}|{value}".encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:16], 16)


def _machine_set_fingerprint(machine_keys: Sequence[str]) -> str:
    payload = "\n".join(sorted(str(x) for x in machine_keys))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _machine_split_design_fingerprint(master: pd.DataFrame) -> str:
    """Fingerprint fields that determine the fixed stratified assignment."""
    cols = ["machine_key", "full_model", "has_eligible_claim_history"]
    work = master.reindex(columns=cols).copy()
    for col in cols:
        work[col] = work[col].astype("string").fillna("<NA>")
    work = work.sort_values(cols, kind="mergesort").reset_index(drop=True)
    return hashlib.md5(
        work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _allocate_stratum_splits(n: int, ratios: Mapping[str, float], stratum: str, random_state: int) -> List[str]:
    active = [name for name in ["train", "validation", "test"] if ratios[name] > 0]
    if n <= 0:
        return []
    if n == 1:
        u = (_stable_hash_int(stratum, random_state) % 10_000_000) / 10_000_000.0
        cumulative = 0.0
        for name in ["train", "validation", "test"]:
            cumulative += ratios[name]
            if u < cumulative:
                return [name]
        return [active[-1]]
    if n == 2 and len(active) >= 2:
        holdouts = [name for name in ["validation", "test"] if ratios[name] > 0]
        second = holdouts[_stable_hash_int(stratum + "|holdout", random_state) % len(holdouts)] if holdouts else active[-1]
        return ["train" if ratios["train"] > 0 else active[0], second]

    raw = {name: n * ratios[name] for name in active}
    counts = {name: int(np.floor(raw[name])) for name in active}
    remainder = n - sum(counts.values())
    order = sorted(
        active,
        key=lambda name: (raw[name] - counts[name], ratios[name], -_stable_hash_int(stratum + name, random_state)),
        reverse=True,
    )
    for i in range(remainder):
        counts[order[i % len(order)]] += 1

    if n >= len(active):
        for name in active:
            if counts[name] == 0:
                donor = max(active, key=lambda x: counts[x])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[name] += 1

    labels: List[str] = []
    for name in ["train", "validation", "test"]:
        labels.extend([name] * counts.get(name, 0))
    return labels[:n]


def build_or_load_fixed_machine_split_assignments(
    machine_master: pd.DataFrame,
    claim_history_episodes: pd.DataFrame,
    config,
    assignment_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create or reuse deterministic machine-level train/validation/test assignments.

    The split is stratified by full model and whether the machine has eligible
    claim history. Every machine is assigned exactly once, and the saved file is
    treated as locked for repeatable experiments.
    """
    assignment_path = Path(assignment_path)
    metadata_path = assignment_path.with_name(assignment_path.stem + "_metadata.json")
    ensure_dir(assignment_path.parent)

    master = machine_master[[c for c in ["machine_key", "full_model", "serial"] if c in machine_master.columns]].copy()
    master = master.dropna(subset=["machine_key"]).drop_duplicates("machine_key")
    master["machine_key"] = master["machine_key"].astype(str)
    master["full_model"] = master.get("full_model", "UNKNOWN").map(clean_model).replace("", "UNKNOWN")
    claim_machines = set(claim_history_episodes.get("machine_key", pd.Series(dtype=str)).dropna().astype(str))
    master["has_eligible_claim_history"] = master["machine_key"].isin(claim_machines).astype(int)

    ratios = _normalized_split_ratios(config)
    split_random_state = int(getattr(config, "FIXED_SPLIT_RANDOM_STATE", config.RANDOM_STATE))
    fingerprint = _machine_set_fingerprint(master["machine_key"].tolist())
    design_fingerprint = _machine_split_design_fingerprint(master)
    expected_meta = {
        "machine_fingerprint": fingerprint,
        "split_design_fingerprint": design_fingerprint,
        "machine_count": int(len(master)),
        "random_state": split_random_state,
        "ratios": ratios,
        "strategy": "fixed_random_machine_level_stratified_by_full_model_and_claim_history",
    }

    if assignment_path.exists():
        if not metadata_path.exists():
            raise ValueError(
                "The fixed split assignment file exists without its metadata lock. "
                f"Delete {assignment_path} only if you intentionally want to create a new split."
            )
        existing = pd.read_csv(assignment_path)
        required = {"machine_key", "split"}
        if not required.issubset(existing.columns):
            raise ValueError(f"Existing split assignment file is invalid: {assignment_path}")
        if existing["machine_key"].astype(str).duplicated().any():
            raise ValueError(f"Existing split assignment contains duplicate machines: {assignment_path}")
        invalid_splits = sorted(
            set(existing["split"].dropna().astype(str)) - {"train", "validation", "test"}
        )
        if invalid_splits or existing["split"].isna().any():
            raise ValueError(
                f"Existing split assignment contains invalid split labels: {invalid_splits}"
            )
        existing_keys = set(existing["machine_key"].astype(str))
        current_keys = set(master["machine_key"].astype(str))
        if existing_keys != current_keys:
            raise ValueError(
                "The eligible machine population changed after fixed split assignments were created. "
                f"Delete {assignment_path} only if you intentionally want to create a new holdout split."
            )
        meta = read_json(metadata_path)
        if (
            meta.get("machine_fingerprint") != fingerprint
            or meta.get("split_design_fingerprint") != design_fingerprint
            or int(meta.get("random_state", -1)) != split_random_state
            or meta.get("ratios") != ratios
        ):
            raise ValueError(
                "Existing fixed split metadata does not match the current eligible machine "
                "population, full-model/claim strata, or split settings. "
                f"Delete {assignment_path} and {metadata_path} only for an intentional redesign."
            )
        existing["machine_key"] = existing["machine_key"].astype(str)
        existing = master.merge(existing.drop(columns=[c for c in ["full_model", "serial", "has_eligible_claim_history"] if c in existing.columns]), on="machine_key", how="left")
        summary = summarize_machine_split_assignments(existing, ratios)
        return existing, summary

    assigned_parts = []
    for (full_model, claim_flag), group in master.groupby(["full_model", "has_eligible_claim_history"], dropna=False, sort=True):
        stratum = f"{full_model}|claim={int(claim_flag)}"
        g = group.copy()
        g["_hash"] = g["machine_key"].map(lambda x: _stable_hash_int(str(x), split_random_state))
        g = g.sort_values(["_hash", "machine_key"], kind="mergesort").reset_index(drop=True)
        labels = _allocate_stratum_splits(len(g), ratios, stratum, split_random_state)
        g["split"] = labels
        g["split_stratum"] = stratum
        g["split_assignment_hash"] = g["_hash"].map(lambda x: f"{int(x):016x}")
        assigned_parts.append(g.drop(columns=["_hash"]))
    assignments = pd.concat(assigned_parts, ignore_index=True) if assigned_parts else master.assign(split=pd.Series(dtype=str))

    # Global safety check: when enough positive machines exist, every active
    # split must contain at least one. Move one deterministic positive machine
    # from a donor split if a rare-stratum allocation left a holdout empty.
    active_splits = [s for s in ["train", "validation", "test"] if ratios[s] > 0]
    positive_total = int(assignments["has_eligible_claim_history"].sum())
    if positive_total >= len(active_splits):
        for missing_split in active_splits:
            if int(assignments.loc[assignments["split"].eq(missing_split), "has_eligible_claim_history"].sum()) > 0:
                continue
            donor_counts = (
                assignments[assignments["has_eligible_claim_history"].eq(1)]
                .groupby("split")["machine_key"].count()
                .sort_values(ascending=False)
            )
            donor = next((s for s, count in donor_counts.items() if count > 1 and s != missing_split), None)
            if donor is None:
                continue
            candidate = (
                assignments[
                    assignments["split"].eq(donor)
                    & assignments["has_eligible_claim_history"].eq(1)
                ]
                .sort_values(["split_assignment_hash", "machine_key"], kind="mergesort")
                .tail(1)
            )
            assignments.loc[candidate.index, "split"] = missing_split

    if assignments["split"].isna().any() or assignments["machine_key"].duplicated().any():
        raise ValueError("Failed to create one valid fixed split assignment per machine.")
    assignments.to_csv(assignment_path, index=False)
    write_json(expected_meta, metadata_path)
    summary = summarize_machine_split_assignments(assignments, ratios)
    return assignments, summary


def summarize_machine_split_assignments(assignments: pd.DataFrame, ratios: Mapping[str, float]) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "validation", "test"]:
        sub = assignments[assignments["split"].eq(split_name)]
        rows.append({
            "split": split_name,
            "machines": int(len(sub)),
            "machines_with_eligible_claim_history": int(sub.get("has_eligible_claim_history", pd.Series(dtype=int)).sum()),
            "machines_without_eligible_claim_history": int((sub.get("has_eligible_claim_history", pd.Series(dtype=int)) == 0).sum()),
            "full_models": int(sub.get("full_model", pd.Series(dtype=str)).nunique(dropna=True)),
            "configured_ratio": float(ratios[split_name]),
            "actual_machine_ratio": float(len(sub) / len(assignments)) if len(assignments) else np.nan,
        })
    return pd.DataFrame(rows)


def _negative_counts(mode: str, total: int) -> Tuple[int, int]:
    if mode == "controlled":
        return total, 0
    if mode == "random":
        return 0, total
    controlled = int(np.ceil(total / 2.0))
    return controlled, total - controlled


_RANDOM_WINDOW_END_CACHE: Dict[int, Dict[str, np.ndarray]] = {}
_RANDOM_ELIGIBLE_WINDOW_CACHE: Dict[tuple, Dict[str, np.ndarray]] = {}


def _random_window_end_candidates_by_machine(
    sources: Mapping[str, pd.DataFrame],
) -> Dict[str, np.ndarray]:
    """Return candidate activity dates once per Step 02 process.

    The previous implementation rebuilt this machine/date dictionary separately
    for training and holdout construction. Operation data is usually the largest
    table, so caching it removes a substantial duplicate groupby cost.
    """
    cache_key = id(sources)
    cached = _RANDOM_WINDOW_END_CACHE.get(cache_key)
    if cached is not None:
        return cached

    operation = sources.get("operation", pd.DataFrame())
    out: Dict[str, np.ndarray] = {}
    if operation is not None and not operation.empty and "LOCAL_DATE" in operation.columns:
        work = operation[["machine_key", "LOCAL_DATE"]].copy()
        work["LOCAL_DATE"] = pd.to_datetime(work["LOCAL_DATE"], errors="coerce").dt.normalize()
        work = work.dropna(subset=["machine_key", "LOCAL_DATE"])
        for machine_key, group in work.groupby("machine_key", sort=False):
            dates = np.sort(group["LOCAL_DATE"].unique().astype("datetime64[ns]"))
            if len(dates):
                out[str(machine_key)] = dates

    # Fallback to another source date only for machines without operation dates.
    fallback_frames = []
    for name, date_col in [
        ("fault", "event_date"),
        ("fluid", "sample_drawn_date"),
        ("maintenance", "event_date"),
    ]:
        df = sources.get(name, pd.DataFrame())
        if df is not None and not df.empty and date_col in df.columns:
            fallback_frames.append(
                df[["machine_key", date_col]].rename(columns={date_col: "candidate_date"})
            )
    if fallback_frames:
        fallback = pd.concat(fallback_frames, ignore_index=True)
        fallback["candidate_date"] = pd.to_datetime(
            fallback["candidate_date"], errors="coerce"
        ).dt.normalize()
        fallback = fallback.dropna(subset=["machine_key", "candidate_date"])
        for machine_key, group in fallback.groupby("machine_key", sort=False):
            key = str(machine_key)
            if key not in out:
                dates = np.sort(group["candidate_date"].unique().astype("datetime64[ns]"))
                if len(dates):
                    out[key] = dates

    _RANDOM_WINDOW_END_CACHE[cache_key] = out
    return out


def build_fixed_horizon_evaluation_base_rows(
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    split_assignments: pd.DataFrame,
    window_config: Mapping,
    config,
    included_splits: Sequence[str] = ("validation", "test"),
    minimum_followup_days: int = 365,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build one fixed, outcome-independent evaluation window per holdout machine.

    The matched case-control holdout is appropriate for evaluating the original
    training target, but it is not appropriate for a future-claim horizon sweep:
    case windows are anchored a fixed number of days before a known claim and
    matched controls are deliberately selected to have no claim during a future
    exclusion interval.  That construction can make several horizon labels
    identical by design.

    This function instead selects a deterministic random window for every
    validation/test machine with sufficient source history and fully observable
    claim follow-up.  Selection does not inspect future claims.  The row identity
    therefore stays fixed while ``claim_within_horizon`` labels can be recomputed
    independently for 30, 60, 90, 120, 180, 365 days, or another supported
    horizon.
    """

    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = int(lead_max - lead_min)
    if observation_days <= 0:
        raise ValueError("lead_max_days must be greater than lead_min_days.")

    followup_days = int(minimum_followup_days)
    if followup_days <= 0:
        raise ValueError("minimum_followup_days must be positive.")

    requested_splits = {str(x) for x in included_splits}
    assignments = split_assignments.copy()
    assignments["machine_key"] = assignments["machine_key"].astype(str)
    assignments = assignments[assignments["split"].astype(str).isin(requested_splits)].copy()

    coverage = build_source_coverage(sources)
    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    missing_coverage = [
        c for c in ["first_source_date", "last_source_date"] if c not in master.columns
    ]
    if missing_coverage:
        master = master.merge(
            coverage[["machine_key", *missing_coverage]],
            on="machine_key",
            how="left",
        )

    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in assignments.columns
    ]
    assignment_meta = assignments[assignment_cols].drop_duplicates("machine_key")
    master = master.merge(
        assignment_meta,
        on="machine_key",
        how="inner",
        suffixes=("", "_assigned"),
        validate="one_to_one",
    )
    for field in ["full_model", "serial"]:
        assigned = f"{field}_assigned"
        if assigned in master.columns:
            current = master.get(field, pd.Series("", index=master.index)).astype("string").fillna("").str.strip()
            master[field] = master.get(field, pd.Series("", index=master.index)).where(
                current.ne(""), master[assigned]
            )
            master = master.drop(columns=[assigned])
    if "full_model" not in master.columns:
        master["full_model"] = ""
    if "serial" not in master.columns:
        master["serial"] = ""
    master["full_model"] = master["full_model"].map(clean_model)
    master["first_source_date"] = pd.to_datetime(master["first_source_date"], errors="coerce")
    master["last_source_date"] = pd.to_datetime(master["last_source_date"], errors="coerce")

    candidate_dates = _random_window_end_candidates_by_machine(sources)
    max_observation = _max_claim_observation_date(config).normalize()
    latest_globally_observable_end = max_observation - pd.Timedelta(days=followup_days)
    seed_value = int(getattr(config, "FIXED_SPLIT_RANDOM_STATE", 42))
    window_name = window_config_name(window_config)

    rows: List[dict] = []
    audit_rows: List[dict] = []
    for _, machine in master.sort_values(["split", "machine_key"], kind="mergesort").iterrows():
        machine_key = str(machine["machine_key"])
        split_name = str(machine["split"])
        first_source = pd.to_datetime(machine.get("first_source_date"), errors="coerce")
        last_source = pd.to_datetime(machine.get("last_source_date"), errors="coerce")
        raw_dates = np.asarray(
            candidate_dates.get(machine_key, np.array([], dtype="datetime64[ns]")),
            dtype="datetime64[ns]",
        )

        reason = "eligible"
        selected_end = pd.NaT
        eligible_count = 0
        if pd.isna(first_source) or pd.isna(last_source):
            reason = "missing_source_coverage"
        elif len(raw_dates) == 0:
            reason = "no_activity_dates"
        else:
            ends = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="coerce")).dropna()
            if not ends.empty:
                ends = ends.normalize().unique().sort_values()
            earliest_end = pd.Timestamp(first_source).normalize() + pd.Timedelta(days=observation_days)
            latest_end = min(
                pd.Timestamp(last_source).normalize(),
                pd.Timestamp(latest_globally_observable_end).normalize(),
            )
            if latest_end < earliest_end:
                reason = "insufficient_history_or_future_followup"
            else:
                ends = ends[(ends >= earliest_end) & (ends <= latest_end)]
                eligible_count = int(len(ends))
                if not eligible_count:
                    reason = "no_activity_date_in_eligible_range"
                else:
                    selector = _stable_hash_int(
                        f"{window_name}|{split_name}|{machine_key}|fixed_horizon_window",
                        seed_value,
                    )
                    selected_end = pd.Timestamp(ends[selector % eligible_count])

        audit_rows.append(
            {
                "machine_key": machine_key,
                "split": split_name,
                "window_name": window_name,
                "status": "selected" if pd.notna(selected_end) else "excluded",
                "reason": reason,
                "candidate_activity_dates": int(len(raw_dates)),
                "eligible_activity_dates": eligible_count,
                "selected_window_end": selected_end,
                "minimum_followup_days": followup_days,
                "max_claim_observation_date": max_observation,
            }
        )
        if pd.isna(selected_end):
            continue

        window_end = pd.Timestamp(selected_end)
        window_start = window_end - pd.Timedelta(days=observation_days)
        sample_id = f"{window_name}__fixed_horizon__{split_name}__{machine_key}"
        rows.append(
            {
                "row_role": "fixed_horizon_evaluation_window",
                # This placeholder is replaced with the largest configured
                # horizon label after future-claim outcomes are annotated. It is
                # never used to fit the model.
                "target": 0,
                "case_control_group_id": sample_id,
                "evaluation_sample_id": sample_id,
                "claim_episode_id": "",
                "case_machine_key": machine_key,
                "control_number_within_group": np.nan,
                "machine_key": machine_key,
                "full_model": machine.get("full_model", ""),
                "serial": machine.get("serial", ""),
                "split": split_name,
                "window_name": window_name,
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "window_start": window_start,
                "window_end": window_end,
                "linked_case_window_start": pd.NaT,
                "linked_case_window_end": pd.NaT,
                "future_claim_date": pd.NaT,
                "days_from_window_end_to_claim": np.nan,
                "negative_sampling_type": "not_applicable",
                "control_sampling_reason": "fixed_outcome_independent_horizon_evaluation_window",
                "control_no_claim_start": pd.NaT,
                "control_no_claim_end": pd.NaT,
                "minimum_observable_followup_days": followup_days,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def _eligible_random_windows_by_machine(
    sources: Mapping[str, pd.DataFrame],
    machine_master: pd.DataFrame,
    dates_by_machine: Mapping[str, np.ndarray],
    lookback_days: int,
    config,
) -> Dict[str, np.ndarray]:
    """Precompute all eligible random window ends once for a window design.

    Eligibility for a random negative depends on the machine, the fixed window
    length, the no-claim interval, source coverage, and the observation end date;
    it does not depend on the linked positive case. Precomputing therefore
    replaces the old nested loop that tested as many as 80 dates for every
    candidate machine for every case.
    """
    prior_days = int(
        getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
    )
    future_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    max_observation = _max_claim_observation_date(config).normalize()
    require_coverage = bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True))
    cache_key = (
        id(sources),
        int(lookback_days),
        prior_days,
        future_days,
        max_observation.value,
        require_coverage,
        sum(len(v) for v in dates_by_machine.values()),
    )
    cached = _RANDOM_ELIGIBLE_WINDOW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    master = machine_master.copy()
    if "first_source_date" not in master.columns or "last_source_date" not in master.columns:
        coverage = build_source_coverage(sources)
        missing_cols = [
            col for col in ["first_source_date", "last_source_date"]
            if col not in master.columns
        ]
        master = master.merge(
            coverage[["machine_key", *missing_cols]], on="machine_key", how="left"
        )
    master["machine_key"] = master["machine_key"].astype(str)
    master["first_source_date"] = pd.to_datetime(
        master.get("first_source_date"), errors="coerce"
    )
    master["last_source_date"] = pd.to_datetime(
        master.get("last_source_date"), errors="coerce"
    )
    coverage_map = master.drop_duplicates("machine_key").set_index("machine_key")

    raw_candidates = _random_window_end_candidates_by_machine(sources)
    eligible: Dict[str, np.ndarray] = {}
    lookback_delta = pd.Timedelta(days=int(lookback_days))
    prior_delta = pd.Timedelta(days=prior_days)
    future_delta = pd.Timedelta(days=future_days)

    for machine_key, raw_dates in raw_candidates.items():
        ends = pd.DatetimeIndex(pd.to_datetime(raw_dates, errors="coerce")).dropna()
        if ends.empty:
            continue
        ends = ends.normalize().unique().sort_values()
        starts = ends - lookback_delta
        mask = np.asarray((ends + future_delta) <= max_observation, dtype=bool)

        if require_coverage:
            if machine_key not in coverage_map.index:
                continue
            row = coverage_map.loc[machine_key]
            first_source = pd.to_datetime(row.get("first_source_date"), errors="coerce")
            last_source = pd.to_datetime(row.get("last_source_date"), errors="coerce")
            if pd.isna(first_source) or pd.isna(last_source):
                continue
            mask &= np.asarray((first_source <= ends) & (last_source >= starts), dtype=bool)

        claim_dates = np.asarray(
            dates_by_machine.get(machine_key, np.array([], dtype="datetime64[ns]")),
            dtype="datetime64[ns]",
        )
        if len(claim_dates):
            claim_dates = np.sort(claim_dates)
            no_claim_starts = (starts - prior_delta).values.astype("datetime64[ns]")
            no_claim_ends = (ends + future_delta).values.astype("datetime64[ns]")
            left = np.searchsorted(claim_dates, no_claim_starts, side="left")
            right = np.searchsorted(claim_dates, no_claim_ends, side="right")
            mask &= left == right

        kept = ends[mask]
        if len(kept):
            eligible[machine_key] = kept.values.astype("datetime64[ns]")

    _RANDOM_ELIGIBLE_WINDOW_CACHE[cache_key] = eligible
    return eligible


def _negative_window_eligibility(
    machine_key: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    machine_row: pd.Series,
    dates_by_machine: Mapping[str, np.ndarray],
    sources: Mapping[str, pd.DataFrame],
    config,
) -> Tuple[bool, str, dict]:
    """Check one controlled negative window using the original lightweight rules."""
    prior_days = int(
        getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
    )
    future_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    no_claim_start = window_start - pd.Timedelta(days=prior_days)
    no_claim_end = window_end + pd.Timedelta(days=future_days)
    details = {
        "control_no_claim_start": no_claim_start,
        "control_no_claim_end": no_claim_end,
    }

    if no_claim_end > _max_claim_observation_date(config):
        return False, "future_claim_followup_not_fully_observable", details
    if bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
        first_source = pd.to_datetime(machine_row.get("first_source_date"), errors="coerce")
        last_source = pd.to_datetime(machine_row.get("last_source_date"), errors="coerce")
        if (
            pd.isna(first_source)
            or pd.isna(last_source)
            or first_source > window_end
            or last_source < window_start
        ):
            return False, "no_source_coverage_overlap_window", details
    if has_claim_between(dates_by_machine, machine_key, no_claim_start, no_claim_end):
        return False, "claim_in_negative_exclusion_interval", details
    return True, "eligible", details


def _controlled_negative_rows(
    case: pd.Series,
    candidates: pd.DataFrame,
    count: int,
    dates_by_machine: Mapping[str, np.ndarray],
    sources: Mapping[str, pd.DataFrame],
    config,
    excluded_machines: Optional[set] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """Sample same-window controls with the original scan-until-filled approach."""
    excluded_machines = set(excluded_machines or set())
    selected: List[dict] = []
    rejection_counts: Dict[str, int] = {}
    candidate_count = int(len(candidates))
    if count <= 0 or candidates.empty:
        return selected, {
            "candidate_count": candidate_count,
            "checked": 0,
            "rejections": rejection_counts,
        }

    window_start = pd.Timestamp(case["window_start"])
    window_end = pd.Timestamp(case["window_end"])
    prior_days = int(
        getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
    )
    future_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    no_claim_start = window_start - pd.Timedelta(days=prior_days)
    no_claim_end = window_end + pd.Timedelta(days=future_days)
    if no_claim_end > _max_claim_observation_date(config):
        rejection_counts["future_claim_followup_not_fully_observable"] = candidate_count
        return selected, {
            "candidate_count": candidate_count,
            "checked": 0,
            "rejections": rejection_counts,
        }

    eligible_pool = candidates
    if bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
        first_source = pd.to_datetime(eligible_pool.get("first_source_date"), errors="coerce")
        last_source = pd.to_datetime(eligible_pool.get("last_source_date"), errors="coerce")
        coverage_mask = (
            first_source.notna()
            & last_source.notna()
            & (first_source <= window_end)
            & (last_source >= window_start)
        )
        rejected = int((~coverage_mask).sum())
        if rejected:
            rejection_counts["no_source_coverage_overlap_window"] = rejected
        eligible_pool = eligible_pool.loc[coverage_mask]

    seed_value = int(config.RANDOM_STATE if random_state is None else random_state)
    seed = _stable_hash_int(
        str(case["case_control_group_id"]) + "|controlled", seed_value
    ) % (2**32 - 1)
    ordered = eligible_pool.sample(frac=1.0, random_state=seed)
    checked = 0
    for _, ctrl in ordered.iterrows():
        machine_key = str(ctrl["machine_key"])
        if machine_key in excluded_machines:
            continue
        checked += 1
        if has_claim_between(
            dates_by_machine, machine_key, no_claim_start, no_claim_end
        ):
            rejection_counts["claim_in_negative_exclusion_interval"] = (
                rejection_counts.get("claim_in_negative_exclusion_interval", 0) + 1
            )
            continue
        selected.append(
            {
                "machine_key": machine_key,
                "full_model": ctrl["full_model"],
                "serial": ctrl.get("serial", ""),
                "window_start": window_start,
                "window_end": window_end,
                "negative_sampling_type": "controlled",
                "control_sampling_reason": (
                    "same_calendar_window_same_full_model_no_claim_in_exclusion_interval"
                ),
                "control_no_claim_start": no_claim_start,
                "control_no_claim_end": no_claim_end,
            }
        )
        excluded_machines.add(machine_key)
        if len(selected) >= count:
            break
    return selected, {
        "candidate_count": candidate_count,
        "checked": checked,
        "rejections": rejection_counts,
    }


def _random_negative_rows(
    case: pd.Series,
    candidates: pd.DataFrame,
    count: int,
    eligible_windows: Mapping[str, np.ndarray],
    config,
    excluded_machines: Optional[set] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    """Sample random same-model negatives from precomputed eligible windows."""
    excluded_machines = set(excluded_machines or set())
    selected: List[dict] = []
    rejection_counts: Dict[str, int] = {}
    if count <= 0 or candidates.empty:
        return selected, {
            "candidate_count": int(len(candidates)),
            "checked_machines": 0,
            "rejections": rejection_counts,
        }

    seed_value = int(config.RANDOM_STATE if random_state is None else random_state)
    group_seed = _stable_hash_int(
        str(case["case_control_group_id"]) + "|random", seed_value
    )
    ordered = candidates.sample(
        frac=1.0, random_state=group_seed % (2**32 - 1)
    )
    lookback_days = int(case["lead_max_days"] - case["lead_min_days"])
    prior_days = int(
        getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30)
    )
    future_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))
    checked_machines = 0

    for _, ctrl in ordered.iterrows():
        machine_key = str(ctrl["machine_key"])
        if machine_key in excluded_machines:
            continue
        checked_machines += 1
        dates = np.asarray(
            eligible_windows.get(machine_key, np.array([], dtype="datetime64[ns]")),
            dtype="datetime64[ns]",
        )
        if len(dates) == 0:
            rejection_counts["no_eligible_random_window"] = (
                rejection_counts.get("no_eligible_random_window", 0) + 1
            )
            continue

        date_seed = _stable_hash_int(
            f"{group_seed}|{machine_key}|window", seed_value
        )
        window_end = pd.Timestamp(dates[date_seed % len(dates)])
        window_start = window_end - pd.Timedelta(days=lookback_days)
        selected.append(
            {
                "machine_key": machine_key,
                "full_model": ctrl["full_model"],
                "serial": ctrl.get("serial", ""),
                "window_start": window_start,
                "window_end": window_end,
                "negative_sampling_type": "random",
                "control_sampling_reason": (
                    "random_precomputed_eligible_window_same_full_model_no_claim"
                ),
                "control_no_claim_start": window_start - pd.Timedelta(days=prior_days),
                "control_no_claim_end": window_end + pd.Timedelta(days=future_days),
            }
        )
        excluded_machines.add(machine_key)
        if len(selected) >= count:
            break

    return selected, {
        "candidate_count": int(len(candidates)),
        "checked_machines": checked_machines,
        "rejections": rejection_counts,
    }


def build_case_control_base_rows(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
    included_splits: Optional[Sequence[str]] = None,
    sampling_mode_override: Optional[str] = None,
    negatives_per_positive_override: Optional[int] = None,
    random_state_override: Optional[int] = None,
    apply_positive_case_cap: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build positive and negative base rows for selected fixed machine splits.

    `included_splits` lets Step 02 construct configurable training rows separately
    from locked validation/test rows. Sampling overrides affect only this call and
    therefore do not require mutating the global configuration module. A separate
    random-state override keeps holdout sampling fixed during training experiments.
    """
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    if lead_max <= lead_min:
        raise ValueError("lead_max_days must be greater than lead_min_days.")

    history_for_claim_checks = claim_history_episodes if claim_history_episodes is not None else episodes
    dates_by_machine = claim_dates_by_machine(history_for_claim_checks)
    coverage = build_source_coverage(sources)
    episodes_work = episodes.merge(
        coverage[["machine_key", "first_source_date", "last_source_date", "source_record_count_total"]],
        on="machine_key",
        how="left",
    )
    if split_assignments is not None:
        episodes_work = episodes_work.merge(
            split_assignments[["machine_key", "split"]], on="machine_key", how="left", validate="many_to_one"
        )
    else:
        episodes_work["split"] = "train"

    selected_splits = None
    if included_splits is not None:
        selected_splits = {str(x).strip().lower() for x in included_splits}
        invalid_splits = sorted(selected_splits - {"train", "validation", "test"})
        if invalid_splits:
            raise ValueError(f"included_splits contains invalid labels: {invalid_splits}")
        episodes_work = episodes_work[episodes_work["split"].isin(selected_splits)].copy()

    positive_rows = []
    audit_rows = []
    for _, ep in episodes_work.iterrows():
        claim_date = pd.Timestamp(ep["claim_date"])
        window_start = claim_date - pd.Timedelta(days=lead_max)
        window_end = claim_date - pd.Timedelta(days=lead_min)
        if pd.isna(ep.get("split")):
            audit_rows.append({
                "claim_episode_id": ep.get("claim_episode_id"),
                "machine_key": ep.get("machine_key"),
                "status": "positive_excluded",
                "reason": "machine_missing_fixed_split_assignment",
            })
            continue
        if bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
            first_src = pd.to_datetime(ep.get("first_source_date"), errors="coerce")
            last_src = pd.to_datetime(ep.get("last_source_date"), errors="coerce")
            if pd.isna(first_src) or pd.isna(last_src) or first_src > window_end or last_src < window_start:
                audit_rows.append({
                    "claim_episode_id": ep.get("claim_episode_id"),
                    "machine_key": ep.get("machine_key"),
                    "split": ep.get("split"),
                    "status": "positive_excluded",
                    "reason": "positive_no_source_coverage_overlap_window",
                    "window_start": window_start,
                    "window_end": window_end,
                })
                continue
        row = {
            "row_role": "case",
            "target": 1,
            "case_control_group_id": f"{window_config_name(window_config)}__{ep['claim_episode_id']}",
            "case_machine_key": ep["machine_key"],
            "claim_episode_id": ep["claim_episode_id"],
            "machine_key": ep["machine_key"],
            "full_model": clean_model(ep["full_model"]),
            "serial": ep.get("serial", ""),
            "split": ep["split"],
            "window_name": window_config_name(window_config),
            "lead_max_days": lead_max,
            "lead_min_days": lead_min,
            "window_start": window_start,
            "window_end": window_end,
            "future_claim_date": claim_date,
            "days_from_window_end_to_claim": float((claim_date - window_end).days),
            "claim_count_in_episode": ep.get("claim_count_in_episode", np.nan),
            "claim_numbers": ep.get("claim_numbers", ""),
            "claim_type_descriptions": ep.get("claim_type_descriptions", ""),
            "critical_fail_part_numbers": ep.get("critical_fail_part_numbers", ""),
            "negative_sampling_type": "case",
        }
        for extra_col in [
            "positive_claim_selection_mode", "lead_max_days_threshold_for_repeat_claim",
            "claim_sequence_number", "machine_claim_event_count", "is_first_claim_for_machine",
            "previous_claim_date_same_machine", "days_since_previous_claim_same_machine",
            "claim_selection_reason",
        ]:
            if extra_col in ep.index:
                row[extra_col] = ep.get(extra_col)
        positive_rows.append(row)

    positives = pd.DataFrame(positive_rows)
    sampling_random_state = int(
        config.RANDOM_STATE if random_state_override is None else random_state_override
    )
    max_cases = (
        getattr(config, "MAX_POSITIVE_CASES_PER_WINDOW", None)
        if apply_positive_case_cap
        else None
    )
    if max_cases is not None and len(positives) > int(max_cases):
        total = int(max_cases)
        if total < 1:
            positives = positives.iloc[0:0].copy()
        else:
            # Preserve the currently included split mix. A single-split training
            # call therefore receives the full development cap rather than only
            # TRAIN_RATIO times the cap.
            split_sizes = positives["split"].value_counts().to_dict()
            active = [name for name in ["train", "validation", "test"] if split_sizes.get(name, 0) > 0]
            raw = {name: total * split_sizes[name] / len(positives) for name in active}
            counts = {name: int(np.floor(raw[name])) for name in active}
            for name in active:
                if total >= len(active) and counts[name] == 0:
                    counts[name] = 1
            while sum(counts.values()) > total:
                donor = max(active, key=lambda name: (counts[name], raw[name]))
                if counts[donor] <= 1 and total >= len(active):
                    break
                counts[donor] -= 1
            remainder = total - sum(counts.values())
            order = sorted(active, key=lambda name: (raw[name] - counts[name], split_sizes[name]), reverse=True)
            for i in range(remainder):
                counts[order[i % len(order)]] += 1
            sampled = []
            for split_name in active:
                sub = positives[positives["split"].eq(split_name)]
                sampled.append(
                    sub.sample(
                        n=min(counts[split_name], len(sub)),
                        random_state=sampling_random_state,
                    )
                )
            positives = pd.concat(sampled, ignore_index=True).drop_duplicates("case_control_group_id")

    master = machine_master.copy()
    if "first_source_date" not in master.columns:
        master = master.merge(coverage, on="machine_key", how="left")
    if split_assignments is not None:
        assignment_cols = [
            c for c in ["machine_key", "split", "full_model", "serial"]
            if c in split_assignments.columns
        ]
        assignment_meta = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
        assignment_meta = assignment_meta.rename(
            columns={c: f"assigned_{c}" for c in assignment_cols if c != "machine_key"}
        )
        master = master.merge(
            assignment_meta,
            on="machine_key",
            how="left",
            validate="one_to_one",
        )
        for field in ["split", "full_model", "serial"]:
            assigned = f"assigned_{field}"
            if assigned not in master.columns:
                continue
            if field not in master.columns:
                master[field] = master[assigned]
            else:
                current = master[field].astype("string").fillna("").str.strip()
                master[field] = master[field].where(current.ne(""), master[assigned])
            master = master.drop(columns=[assigned])
    if "full_model" not in master.columns:
        master["full_model"] = ""
    if "serial" not in master.columns:
        master["serial"] = ""
    if "split" not in master.columns:
        master["split"] = "train"
    master["full_model"] = master["full_model"].map(clean_model)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master_by_model_split = {
        (model, split): group.reset_index(drop=True)
        for (model, split), group in master.groupby(["full_model", "split"], dropna=False)
    }

    mode = (
        normalize_negative_sampling_mode(sampling_mode_override, "sampling_mode_override")
        if sampling_mode_override is not None
        else negative_sampling_mode(config)
    )
    total_requested = (
        validate_negative_count(negatives_per_positive_override, "negatives_per_positive_override")
        if negatives_per_positive_override is not None
        else negatives_per_positive(config)
    )
    requested_controlled, requested_random = _negative_counts(mode, total_requested)
    random_windows: Dict[str, np.ndarray] = {}
    if mode in {"random", "mixed"}:
        random_windows = _eligible_random_windows_by_machine(
            sources=sources,
            machine_master=master,
            dates_by_machine=dates_by_machine,
            lookback_days=lead_max - lead_min,
            config=config,
        )
    accepted_cases = []
    controls = []

    for _, case in positives.iterrows():
        pool = master_by_model_split.get((case["full_model"], case["split"]), pd.DataFrame()).copy()
        if not pool.empty:
            pool = pool[pool["machine_key"].astype(str) != str(case["machine_key"])]
        selected_controlled, ctrl_audit = _controlled_negative_rows(
            case, pool, requested_controlled, dates_by_machine, sources, config,
            random_state=sampling_random_state,
        )
        used = {str(r["machine_key"]) for r in selected_controlled}
        selected_random, random_audit = _random_negative_rows(
            case,
            pool,
            requested_random,
            random_windows,
            config,
            excluded_machines=used,
            random_state=sampling_random_state,
        )
        used.update(str(r["machine_key"]) for r in selected_random)

        # In mixed mode, let the other method fill a shortfall without adding a
        # separate mix-ratio knob.
        selected = selected_controlled + selected_random
        if mode == "mixed" and len(selected) < total_requested:
            shortfall = total_requested - len(selected)
            extra_random, extra_random_audit = _random_negative_rows(
                case,
                pool,
                shortfall,
                random_windows,
                config,
                excluded_machines=used,
                random_state=sampling_random_state,
            )
            selected.extend(extra_random)
            used.update(str(r["machine_key"]) for r in extra_random)
            if len(selected) < total_requested:
                extra_controlled, extra_ctrl_audit = _controlled_negative_rows(
                    case, pool, total_requested - len(selected), dates_by_machine, sources, config,
                    excluded_machines=used, random_state=sampling_random_state,
                )
                selected.extend(extra_controlled)
                ctrl_audit["mixed_fallback"] = extra_ctrl_audit
            random_audit["mixed_fallback"] = extra_random_audit

        if not selected:
            audit_rows.append({
                "case_control_group_id": case["case_control_group_id"],
                "claim_episode_id": case["claim_episode_id"],
                "case_machine_key": case["machine_key"],
                "split": case["split"],
                "status": "case_excluded_no_eligible_negatives",
                "negative_sampling_mode": mode,
                "requested_negative_count": total_requested,
                "selected_negative_count": 0,
                "controlled_audit": json.dumps(ctrl_audit, default=str),
                "random_audit": json.dumps(random_audit, default=str),
            })
            continue

        accepted_cases.append(case.to_dict())
        for j, negative in enumerate(selected, start=1):
            controls.append({
                "row_role": "control",
                "target": 0,
                "case_control_group_id": case["case_control_group_id"],
                "case_machine_key": case["machine_key"],
                "claim_episode_id": case["claim_episode_id"],
                "control_number_within_group": j,
                "machine_key": negative["machine_key"],
                "full_model": negative["full_model"],
                "serial": negative.get("serial", ""),
                "split": case["split"],
                "window_name": case["window_name"],
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "window_start": negative["window_start"],
                "window_end": negative["window_end"],
                "linked_case_window_start": case["window_start"],
                "linked_case_window_end": case["window_end"],
                "future_claim_date": pd.NaT,
                "days_from_window_end_to_claim": np.nan,
                **negative,
            })
        audit_rows.append({
            "case_control_group_id": case["case_control_group_id"],
            "claim_episode_id": case["claim_episode_id"],
            "case_machine_key": case["machine_key"],
            "split": case["split"],
            "status": "selected",
            "negative_sampling_mode": mode,
            "requested_negative_count": total_requested,
            "requested_controlled_count": requested_controlled,
            "requested_random_count": requested_random,
            "selected_negative_count": len(selected),
            "selected_controlled_count": sum(r["negative_sampling_type"] == "controlled" for r in selected),
            "selected_random_count": sum(r["negative_sampling_type"] == "random" for r in selected),
            "controlled_audit": json.dumps(ctrl_audit, default=str),
            "random_audit": json.dumps(random_audit, default=str),
        })

    base = pd.concat([pd.DataFrame(accepted_cases), pd.DataFrame(controls)], ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    return base, audit


def _max_claim_observation_date(config) -> pd.Timestamp:
    for attr in ["MAX_CLAIM_DATE", "MAX_VALID_EVENT_DATE"]:
        value = getattr(config, attr, None)
        if value is not None:
            return pd.Timestamp(value)
    return pd.Timestamp.today().normalize()


def make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str], scale_numeric: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    transformers = []
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers.append(("num", Pipeline(numeric_steps), list(numeric_features)))
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ]), list(categorical_features)))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def make_calibrated_linear_svm(params: dict, random_state: int):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    base = LinearSVC(random_state=random_state, **params)
    try:
        return CalibratedClassifierCV(estimator=base, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, cv=3)


def make_model_pipeline(algorithm: str, config):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    algorithm = str(algorithm).lower()
    scale = algorithm in {"logistic_regression", "linear_svm", "linear_regression"}
    pre = build_preprocessor(config.NUMERIC_FEATURES, config.CATEGORICAL_FEATURES, scale_numeric=scale)

    if algorithm == "logistic_regression":
        model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    elif algorithm == "linear_regression":
        model = LinearRegression()
    elif algorithm == "linear_svm":
        model = make_calibrated_linear_svm(config.LINEAR_SVM_PARAMS, config.RANDOM_STATE)
    elif algorithm == "random_forest":
        model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    elif algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            if bool(getattr(config, "SKIP_MISSING_OPTIONAL_ALGORITHMS", True)):
                return None
            raise exc
        model = XGBClassifier(**config.XGBOOST_PARAMS)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return Pipeline([("preprocessor", pre), ("model", model)])


def resolve_xgboost_scale_pos_weight(y_train, config) -> Tuple[Optional[float], str]:
    """Resolve XGBoost positive-class importance for one training split.

    XGBoost's scale_pos_weight is the standard binary-class weighting parameter.
    A value greater than 1 increases the importance of positive claim cases.
    """

    mode = str(getattr(config, "XGBOOST_CLASS_IMPORTANCE_MODE", "none")).strip().lower()
    if mode in {"none", "off", "false"}:
        return None, "none"
    if mode == "fixed":
        return float(getattr(config, "XGBOOST_FIXED_SCALE_POS_WEIGHT", 1.0)), "fixed"
    if mode == "auto":
        y = pd.Series(y_train).astype(int)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos <= 0:
            return 1.0, "auto_no_positive_rows"
        return float(max(neg / pos, 1.0)), "auto_neg_div_pos"
    raise ValueError(
        "Unsupported XGBOOST_CLASS_IMPORTANCE_MODE="
        f"{getattr(config, 'XGBOOST_CLASS_IMPORTANCE_MODE', None)!r}. "
        "Use 'auto', 'fixed', or 'none'."
    )


def _to_numpy_dense(matrix) -> np.ndarray:
    """Return a dense numpy array from a numpy/scipy/pandas matrix."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def prepared_feature_names(pipeline) -> List[str]:
    """Return feature names emitted by a fitted sklearn ColumnTransformer."""
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        return []
    try:
        return [str(x) for x in pipeline.named_steps["preprocessor"].get_feature_names_out()]
    except Exception:
        return []


def transform_with_fitted_preprocessor(pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transform X with a fitted pipeline preprocessor and return matrix plus names."""
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a fitted preprocessor step.")
    pre = pipeline.named_steps["preprocessor"]
    matrix = _to_numpy_dense(pre.transform(X))
    names = prepared_feature_names(pipeline)
    if not names:
        names = [f"f{i}" for i in range(matrix.shape[1])]
    return matrix, names


def _fit_xgboost_pipeline_with_eval(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    config,
    eval_name: str = "validation",
) -> dict:
    """Fit an XGBoost pipeline while recording train/eval learning curves.

    The normal sklearn Pipeline cannot pass a transformed eval_set into XGBoost.
    This helper fits the preprocessor first, transforms train/eval data, and then
    fits the XGBClassifier directly with eval_set=[train, eval].  Early stopping
    is optional and disabled by default in Phase 1 design sweeps.
    """

    metadata = {"algorithm": "xgboost", "xgboost_fit_mode": "manual_preprocess_with_eval_set"}
    spw, mode = resolve_xgboost_scale_pos_weight(y_train, config)
    metadata["xgboost_class_importance_mode"] = mode
    metadata["xgboost_scale_pos_weight"] = spw

    pre = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]
    if spw is not None:
        xgb_model.set_params(scale_pos_weight=float(spw))

    use_early_stopping = bool(getattr(config, "XGBOOST_USE_EARLY_STOPPING", False))
    early_rounds = int(getattr(config, "XGBOOST_EARLY_STOPPING_ROUNDS", 0) or 0)
    if use_early_stopping and early_rounds > 0:
        xgb_model.set_params(early_stopping_rounds=early_rounds)
        metadata["xgboost_early_stopping_enabled"] = True
        metadata["xgboost_early_stopping_rounds"] = early_rounds
    else:
        metadata["xgboost_early_stopping_enabled"] = False
        metadata["xgboost_early_stopping_rounds"] = 0

    X_train_prepared = _to_numpy_dense(pre.fit_transform(X_train))
    X_eval_prepared = _to_numpy_dense(pre.transform(X_eval))
    y_train_arr = pd.Series(y_train).astype(int).to_numpy()
    y_eval_arr = pd.Series(y_eval).astype(int).to_numpy()

    fit_kwargs = {
        "eval_set": [(X_train_prepared, y_train_arr), (X_eval_prepared, y_eval_arr)],
        "verbose": bool(getattr(config, "XGBOOST_FIT_VERBOSE", False)),
    }
    try:
        xgb_model.fit(X_train_prepared, y_train_arr, **fit_kwargs)
    except TypeError:
        # Compatibility fallback for older xgboost versions that expect
        # early_stopping_rounds in fit rather than constructor parameters.
        if use_early_stopping and early_rounds > 0:
            try:
                xgb_model.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            fit_kwargs["early_stopping_rounds"] = early_rounds
            xgb_model.fit(X_train_prepared, y_train_arr, **fit_kwargs)
        else:
            raise

    metadata["xgboost_eval_name"] = eval_name
    try:
        evals_result = xgb_model.evals_result()
    except Exception:
        evals_result = {}
    metadata["_xgboost_evals_result"] = evals_result
    metadata["xgboost_learning_curve_available"] = bool(evals_result)
    if hasattr(xgb_model, "best_iteration"):
        try:
            metadata["xgboost_best_iteration"] = int(xgb_model.best_iteration)
        except Exception:
            metadata["xgboost_best_iteration"] = np.nan
    if hasattr(xgb_model, "best_score"):
        try:
            metadata["xgboost_best_score"] = float(xgb_model.best_score)
        except Exception:
            metadata["xgboost_best_score"] = np.nan
    return metadata


def fit_model_pipeline(
    model,
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[pd.Series] = None,
    eval_name: str = "validation",
) -> dict:
    """Fit a model pipeline and return fit metadata.

    This helper centralizes per-split configuration such as XGBoost
    scale_pos_weight.  For XGBoost, it can also record train/evaluation
    learning curves through eval_set without enabling early stopping.
    """

    metadata = {"algorithm": str(algorithm).lower()}
    algorithm = str(algorithm).lower()
    if algorithm == "xgboost":
        use_eval_set = (
            X_eval is not None
            and y_eval is not None
            and (
                bool(getattr(config, "XGBOOST_ENABLE_LEARNING_CURVE", True))
                or bool(getattr(config, "XGBOOST_USE_EARLY_STOPPING", False))
            )
        )
        if use_eval_set and hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            return _fit_xgboost_pipeline_with_eval(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                config=config,
                eval_name=eval_name,
            )
        spw, mode = resolve_xgboost_scale_pos_weight(y_train, config)
        metadata["xgboost_class_importance_mode"] = mode
        metadata["xgboost_scale_pos_weight"] = spw
        metadata["xgboost_fit_mode"] = "sklearn_pipeline"
        metadata["xgboost_learning_curve_available"] = False
        if spw is not None and hasattr(model, "set_params"):
            model.set_params(model__scale_pos_weight=float(spw))
    model.fit(X_train, y_train)
    return metadata


def predict_score(model, X: pd.DataFrame, algorithm: str) -> np.ndarray:
    algorithm = str(algorithm).lower()
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            return np.asarray(prob[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=float)
        if np.nanmax(score) > np.nanmin(score):
            return (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score))
        return np.full(len(score), 0.5)
    pred = np.asarray(model.predict(X), dtype=float)
    if algorithm == "linear_regression":
        return np.clip(pred, 0.0, 1.0)
    return pred


def threshold_free_metrics(y_true, score) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = pd.Series(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    out = {
        "rows": int(len(y)),
        "positive_count": int((y == 1).sum()),
        "positive_rate": float((y == 1).mean()) if len(y) else np.nan,
    }
    if y.nunique() < 2:
        out["warning"] = "only_one_class"
        return out
    out["average_precision"] = float(average_precision_score(y, score))
    out["roc_auc"] = float(roc_auc_score(y, score))
    return out


def metrics_at_threshold(y_true, score, threshold: float = 0.5) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
    y = pd.Series(y_true).astype(int).to_numpy()
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "flagged_count": int(pred.sum()),
        "flagged_rate": float(pred.mean()) if len(pred) else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def top_k_metrics(y_true, score, top_k_rates: Sequence[float]) -> pd.DataFrame:
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    s = pd.Series(score).reset_index(drop=True)
    order = s.sort_values(ascending=False).index.to_numpy()
    total_pos = int((y == 1).sum())
    n = len(y)
    rows = []
    for rate in top_k_rates:
        k = int(np.ceil(n * float(rate)))
        k = max(1, min(k, n)) if n else 0
        top_idx = order[:k]
        tp = int(y.iloc[top_idx].sum()) if k else 0
        precision = tp / k if k else np.nan
        recall = tp / total_pos if total_pos else np.nan
        base_rate = total_pos / n if n else np.nan
        rows.append({
            "top_k_rate": float(rate),
            "rows": int(n),
            "flagged_count": int(k),
            "positive_count": total_pos,
            "precision_at_k": float(precision),
            "recall_at_k": float(recall),
            "lift_vs_random": float(precision / base_rate) if base_rate and base_rate > 0 else np.nan,
            "min_score_in_top_k": float(s.iloc[top_idx].min()) if k else np.nan,
        })
    return pd.DataFrame(rows)


def dataset_feature_columns(config) -> List[str]:
    return list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)


def validate_dataset_features(df: pd.DataFrame, config) -> Tuple[List[str], List[str]]:
    features = dataset_feature_columns(config)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset is missing configured features: {missing}")
    return features, missing


def summarize_fixed_dataset_splits(df: pd.DataFrame, config) -> pd.DataFrame:
    """Summarize the already-assigned machine-level random splits."""
    if "split" not in df.columns:
        raise ValueError("Dataset is missing the fixed machine-level 'split' column.")
    ratios = _normalized_split_ratios(config)
    rows = []
    for split_name in ["train", "validation", "test"]:
        sub = df[df["split"].eq(split_name)].copy()
        target = pd.to_numeric(sub.get("target", pd.Series(dtype=float)), errors="coerce")
        rows.append({
            "split": split_name,
            "rows": int(len(sub)),
            "positive_rows": int(target.fillna(0).sum()) if len(sub) else 0,
            "negative_rows": int(target.eq(0).sum()) if len(sub) else 0,
            "positive_rate": float(target.mean()) if len(sub) else np.nan,
            "machines": int(sub["machine_key"].nunique(dropna=True)) if "machine_key" in sub.columns else 0,
            "case_machines": int(sub["case_machine_key"].nunique(dropna=True)) if "case_machine_key" in sub.columns else 0,
            "case_control_groups": int(sub["case_control_group_id"].nunique(dropna=True)) if "case_control_group_id" in sub.columns else 0,
            "full_models": int(sub["full_model"].nunique(dropna=True)) if "full_model" in sub.columns else 0,
            "configured_machine_ratio": float(ratios[split_name]),
        })
    return pd.DataFrame(rows)


def validate_fixed_dataset_splits(df: pd.DataFrame, config) -> pd.DataFrame:
    """Validate fixed random holdouts and return their summary.

    Every physical machine must occur in one split only, every case-control group
    must stay within one split, and each configured split must contain both target
    classes. This protects validation/test from machine-history leakage.
    """
    required = {"split", "machine_key", "case_control_group_id", "target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required fixed-split columns: {missing}")

    allowed = {"train", "validation", "test"}
    observed = set(df["split"].dropna().astype(str).unique())
    invalid = sorted(observed - allowed)
    if invalid:
        raise ValueError(f"Unexpected split labels: {invalid}")

    machine_split_count = df.groupby("machine_key", dropna=False)["split"].nunique(dropna=False)
    leaking_machines = machine_split_count[machine_split_count > 1]
    if not leaking_machines.empty:
        sample = leaking_machines.head(10).index.astype(str).tolist()
        raise ValueError(
            "Machine leakage detected across train/validation/test. "
            f"Example machine keys: {sample}"
        )

    group_split_count = df.groupby("case_control_group_id", dropna=False)["split"].nunique(dropna=False)
    leaking_groups = group_split_count[group_split_count > 1]
    if not leaking_groups.empty:
        sample = leaking_groups.head(10).index.astype(str).tolist()
        raise ValueError(
            "Case-control groups span multiple splits. "
            f"Example group IDs: {sample}"
        )

    summary = summarize_fixed_dataset_splits(df, config)
    ratios = _normalized_split_ratios(config)
    for split_name in ["train", "validation", "test"]:
        if ratios[split_name] <= 0:
            continue
        sub = df[df["split"].eq(split_name)]
        if sub.empty:
            raise ValueError(
                f"Fixed split '{split_name}' is empty. The eligible machine/claim population "
                "is too small for the configured holdout ratios."
            )
        classes = set(pd.to_numeric(sub["target"], errors="coerce").dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(
                f"Fixed split '{split_name}' does not contain both positive and negative rows. "
                "Review the machine split summary and negative-sampling audit."
            )
    return summary


def prediction_frame(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    cols = [
        "split",
        "window_name",
        "case_control_group_id",
        "row_role",
        "target",
        "machine_key",
        "full_model",
        "serial",
        "window_start",
        "window_end",
        "future_claim_date",
        "next_claim_date_on_or_after_window_end",
        "days_to_next_claim_on_or_after_window_end",
        "has_future_claim_on_or_after_window_end",
        "future_claim_lead_time_bucket",
        "claim_episode_id",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy().reset_index(drop=True)
    out["score"] = np.asarray(score)
    return out

# -----------------------------------------------------------------------------
# Vectorized base window feature extraction.
# -----------------------------------------------------------------------------
def _base_with_row_id(base_rows: pd.DataFrame) -> pd.DataFrame:
    base = base_rows.reset_index(drop=True).copy()
    base["row_id"] = np.arange(len(base), dtype=int)
    base["window_start"] = pd.to_datetime(base["window_start"], errors="coerce")
    base["window_end"] = pd.to_datetime(base["window_end"], errors="coerce")
    return base


def _source_window_join(base: pd.DataFrame, source: pd.DataFrame, date_col: str, keep_cols: Sequence[str]) -> pd.DataFrame:
    if source is None or source.empty:
        return pd.DataFrame()
    cols = ["machine_key", date_col] + [c for c in keep_cols if c in source.columns and c not in {"machine_key", date_col}]
    src = source[cols].copy()
    src[date_col] = pd.to_datetime(src[date_col], errors="coerce")
    src = src.dropna(subset=["machine_key", date_col])
    b = base[["row_id", "machine_key", "window_start", "window_end"]]
    merged = src.merge(b, on="machine_key", how="inner")
    if merged.empty:
        return merged
    mask = (merged[date_col] >= merged["window_start"]) & (merged[date_col] <= merged["window_end"])
    return merged.loc[mask].copy()


def _default_feature_frame(base: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"row_id": base["row_id"].to_numpy()})


def _merge_features(base_feat: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    if add is None or add.empty:
        return base_feat
    return base_feat.merge(add, on="row_id", how="left")


def _fill_numeric_categorical(features: pd.DataFrame, config) -> pd.DataFrame:
    for col in config.NUMERIC_FEATURES:
        if col not in features.columns:
            features[col] = np.nan
    zero_default = [c for c in config.NUMERIC_FEATURES if not c.startswith("days_since") and not c.endswith("latest_smr_window") and not c.endswith("smr_delta_window") and not c.endswith("idle_share_window") and not c.endswith("min_remaining_hours_window")]
    for col in zero_default:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
    for col in config.CATEGORICAL_FEATURES:
        if col not in features.columns:
            features[col] = "NONE"
        features[col] = features[col].fillna("NONE").astype(str)
    return features


def _aggregate_faults_vectorized(base: pd.DataFrame, fault: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "fault_code",
        "action_level_num",
        "failure_code_evidence_score",
        "log_occurrence_count",
        "is_mechanical_failure_code",
        "is_electrical_failure_code",
    ]
    m = _source_window_join(base, fault, "event_date", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    if "fault_code" not in m.columns:
        m["fault_code"] = ""
    for col in [
        "action_level_num",
        "failure_code_evidence_score",
        "log_occurrence_count",
        "is_mechanical_failure_code",
        "is_electrical_failure_code",
    ]:
        if col not in m.columns:
            m[col] = 0.0
        m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0)
    m["_l03plus"] = (m["action_level_num"] >= 3).astype(int)
    m["_l04plus"] = (m["action_level_num"] >= 4).astype(int)
    ag = m.groupby("row_id", dropna=False).agg(
        has_fault_window=("event_date", lambda x: 1),
        fault_count_window=("event_date", "size"),
        fault_unique_code_count_window=("fault_code", "nunique"),
        fault_l03plus_count_window=("_l03plus", "sum"),
        fault_l04plus_count_window=("_l04plus", "sum"),
        fault_max_action_level_window=("action_level_num", "max"),
        fault_max_evidence_score_window=("failure_code_evidence_score", "max"),
        fault_mean_evidence_score_window=("failure_code_evidence_score", "mean"),
        fault_max_log_occurrence_window=("log_occurrence_count", "max"),
        latest_fault_date=("event_date", "max"),
        fault_mechanical_count_window=("is_mechanical_failure_code", "sum"),
        fault_electrical_count_window=("is_electrical_failure_code", "sum"),
    ).reset_index()
    ag = ag.merge(base[["row_id", "window_end"]], on="row_id", how="left")
    ag["fault_days_since_latest_in_window"] = (ag["window_end"] - ag["latest_fault_date"]).dt.days.astype(float)
    return ag.drop(columns=["latest_fault_date", "window_end"], errors="ignore")


def _aggregate_fluids_vectorized(base: pd.DataFrame, fluid: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "sample_result_severity_order",
        "Cu_Copper_PPM",
        "Fe_Iron_PPM",
        "Pb_Lead_PPM",
        "Soot_Soot_PERCENT",
        "Water_Water_PERCENT",
    ]
    m = _source_window_join(base, fluid, "sample_drawn_date", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    for col in keep:
        if col not in m.columns:
            m[col] = np.nan
        m[col] = pd.to_numeric(m[col], errors="coerce")
    ag = m.groupby("row_id", dropna=False).agg(
        has_fluid_window=("sample_drawn_date", lambda x: 1),
        fluid_sample_count_window=("sample_drawn_date", "size"),
        fluid_max_severity_window=("sample_result_severity_order", "max"),
        latest_fluid_sample_date=("sample_drawn_date", "max"),
        fluid_max_cu_ppm_window=("Cu_Copper_PPM", "max"),
        fluid_max_fe_ppm_window=("Fe_Iron_PPM", "max"),
        fluid_max_pb_ppm_window=("Pb_Lead_PPM", "max"),
        fluid_max_soot_percent_window=("Soot_Soot_PERCENT", "max"),
        fluid_max_water_percent_window=("Water_Water_PERCENT", "max"),
    ).reset_index()
    latest = m.sort_values(["row_id", "sample_drawn_date"], kind="mergesort").groupby("row_id").tail(1)[["row_id", "sample_result_severity_order"]]
    latest = latest.rename(columns={"sample_result_severity_order": "fluid_latest_severity_window"})
    ag = ag.merge(latest, on="row_id", how="left")
    ag = ag.merge(base[["row_id", "window_end"]], on="row_id", how="left")
    ag["fluid_days_since_latest_sample_window"] = (ag["window_end"] - ag["latest_fluid_sample_date"]).dt.days.astype(float)
    return ag.drop(columns=["latest_fluid_sample_date", "window_end"], errors="ignore")


def _aggregate_maintenance_vectorized(base: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    keep = ["is_monitor_reset", "is_overdue", "is_due_now", "remaining_hours"]
    m = _source_window_join(base, maintenance, "event_date", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    for col in keep:
        if col not in m.columns:
            m[col] = np.nan if col == "remaining_hours" else 0.0
        m[col] = pd.to_numeric(m[col], errors="coerce")
    ag = m.groupby("row_id", dropna=False).agg(
        has_maintenance_window=("event_date", lambda x: 1),
        maintenance_event_count_window=("event_date", "size"),
        maintenance_monitor_reset_count_window=("is_monitor_reset", "sum"),
        maintenance_overdue_count_window=("is_overdue", "sum"),
        maintenance_due_now_count_window=("is_due_now", "sum"),
        maintenance_min_remaining_hours_window=("remaining_hours", "min"),
        latest_maintenance_date=("event_date", "max"),
    ).reset_index()
    ag = ag.merge(base[["row_id", "window_end"]], on="row_id", how="left")
    ag["maintenance_days_since_latest_event_window"] = (
        ag["window_end"] - ag["latest_maintenance_date"]
    ).dt.days.astype(float)
    return ag.drop(columns=["latest_maintenance_date", "window_end"], errors="ignore")


def _aggregate_operation_vectorized(base: pd.DataFrame, operation: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "smr_hours",
        "working_hours_clean",
        "actual_working_hours_clean",
        "engine_running_hours_clean",
        "engine_idling_hours_clean",
        "high_throttle_day_flag",
    ]
    m = _source_window_join(base, operation, "LOCAL_DATE", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    if "working_hours_clean" not in m.columns and "actual_working_hours_clean" in m.columns:
        m["working_hours_clean"] = m["actual_working_hours_clean"]
    defaults = {
        "smr_hours": np.nan,
        "working_hours_clean": 0.0,
        "engine_running_hours_clean": 0.0,
        "engine_idling_hours_clean": 0.0,
        "high_throttle_day_flag": 0.0,
    }
    for col, default in defaults.items():
        if col not in m.columns:
            m[col] = default
        m[col] = pd.to_numeric(m[col], errors="coerce")
    ag = m.groupby("row_id", dropna=False).agg(
        has_operation_window=("LOCAL_DATE", lambda x: 1),
        operation_day_count_window=("LOCAL_DATE", "size"),
        operation_working_hours_sum_window=("working_hours_clean", "sum"),
        operation_working_hours_mean_window=("working_hours_clean", "mean"),
        operation_working_hours_max_window=("working_hours_clean", "max"),
        operation_engine_running_hours_sum_window=("engine_running_hours_clean", "sum"),
        operation_idle_hours_sum_window=("engine_idling_hours_clean", "sum"),
        operation_high_throttle_day_count_window=("high_throttle_day_flag", "sum"),
    ).reset_index()
    ag["operation_idle_share_window"] = np.where(
        ag["operation_engine_running_hours_sum_window"] > 0,
        ag["operation_idle_hours_sum_window"] / ag["operation_engine_running_hours_sum_window"],
        np.nan,
    )
    sorted_m = m.sort_values(["row_id", "LOCAL_DATE"], kind="mergesort")
    first = sorted_m.groupby("row_id").head(1)[["row_id", "smr_hours"]].rename(columns={"smr_hours": "_first_smr"})
    latest = sorted_m.groupby("row_id").tail(1)[["row_id", "smr_hours"]].rename(columns={"smr_hours": "operation_latest_smr_window"})
    ag = ag.merge(first, on="row_id", how="left").merge(latest, on="row_id", how="left")
    ag["operation_smr_delta_window"] = ag["operation_latest_smr_window"] - ag["_first_smr"]
    return ag.drop(columns=["_first_smr"], errors="ignore")


def _build_base_window_features(
    base_rows: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    episodes: pd.DataFrame,
    config,
) -> pd.DataFrame:
    base = _base_with_row_id(base_rows)
    features = _default_feature_frame(base)
    features = _merge_features(features, _aggregate_faults_vectorized(base, sources.get("fault", pd.DataFrame())))
    features = _merge_features(features, _aggregate_fluids_vectorized(base, sources.get("fluid", pd.DataFrame())))
    features = _merge_features(features, _aggregate_maintenance_vectorized(base, sources.get("maintenance", pd.DataFrame())))
    features = _merge_features(features, _aggregate_operation_vectorized(base, sources.get("operation", pd.DataFrame())))

    dates_by_machine = claim_dates_by_machine(episodes)
    prior_counts = []
    days_since = []
    for _, row in base.iterrows():
        prior_count, days = count_claims_before(
            dates_by_machine, row["machine_key"], row["window_start"]
        )
        prior_counts.append(prior_count)
        days_since.append(days)
    features["prior_claim_count_before_window"] = prior_counts
    features["days_since_prior_claim_before_window"] = days_since

    count_cols = [
        "fault_count_window",
        "fluid_sample_count_window",
        "maintenance_event_count_window",
        "operation_day_count_window",
    ]
    for col in count_cols:
        if col not in features.columns:
            features[col] = 0
    features["source_record_count_window"] = features[count_cols].fillna(0).sum(axis=1)
    features["has_any_source_window"] = (features["source_record_count_window"] > 0).astype(int)
    features = _fill_numeric_categorical(features, config)

    base_no_id = base.drop(columns=["row_id"]).reset_index(drop=True)
    feature_part = features.drop(columns=["row_id"]).reset_index(drop=True)
    overlap = [col for col in feature_part.columns if col in base_no_id.columns]
    if overlap:
        feature_part = feature_part.drop(columns=overlap)
    return base_no_id.join(feature_part)


def build_window_features(
    base_rows: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    episodes: pd.DataFrame,
    config=None,
) -> pd.DataFrame:
    """Build either the compact base features or the frozen snapshot features."""
    cfg = config if config is not None else __import__("config")
    if hasattr(cfg, "refresh_derived_config"):
        cfg.refresh_derived_config()
    mode = str(getattr(cfg, "FEATURE_SET", "base")).strip().lower()
    if mode == "base":
        return _build_base_window_features(base_rows, sources, episodes, cfg)
    if mode == "frozen":
        from feature_engineering import build_frozen_window_features

        return build_frozen_window_features(base_rows, sources, cfg)
    raise ValueError("FEATURE_SET must be 'base' or 'frozen'.")


def xgboost_learning_curve_frame(pipeline, fit_metadata: Optional[Mapping] = None) -> pd.DataFrame:
    """Return XGBoost eval_set learning-curve history as a long dataframe."""
    result = {}
    if fit_metadata and isinstance(fit_metadata.get("_xgboost_evals_result"), Mapping):
        result = fit_metadata.get("_xgboost_evals_result") or {}
    if not result and hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
        if hasattr(model, "evals_result"):
            try:
                result = model.evals_result()
            except Exception:
                result = {}
    rows = []
    dataset_alias = {"validation_0": "train", "validation_1": str((fit_metadata or {}).get("xgboost_eval_name", "validation"))}
    for dataset_name, metric_map in (result or {}).items():
        dataset_label = dataset_alias.get(str(dataset_name), str(dataset_name))
        if not isinstance(metric_map, Mapping):
            continue
        for metric_name, values in metric_map.items():
            for i, value in enumerate(values):
                rows.append({
                    "iteration": int(i),
                    "dataset_name": str(dataset_name),
                    "dataset_label": dataset_label,
                    "metric": str(metric_name),
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def summarize_xgboost_learning_curve(curve: pd.DataFrame, eval_label: str = "validation") -> pd.DataFrame:
    """Summarize train/eval gap and best validation iteration for each metric."""
    if curve.empty:
        return pd.DataFrame()
    rows = []
    maximize_tokens = ("auc", "aucpr", "map", "ndcg")
    for metric, g_metric in curve.groupby("metric", dropna=False):
        eval_rows = g_metric[g_metric["dataset_label"].astype(str).eq(str(eval_label))]
        if eval_rows.empty:
            # If the alias does not match, use the last non-train dataset.
            non_train = g_metric[~g_metric["dataset_label"].astype(str).eq("train")]
            eval_rows = non_train if not non_train.empty else g_metric
        train_rows = g_metric[g_metric["dataset_label"].astype(str).eq("train")]
        maximize = any(tok in str(metric).lower() for tok in maximize_tokens)
        best_idx = eval_rows["value"].idxmax() if maximize else eval_rows["value"].idxmin()
        best = eval_rows.loc[best_idx]
        final_eval = eval_rows.sort_values("iteration").iloc[-1]
        final_train_value = np.nan
        if not train_rows.empty:
            final_train_value = float(train_rows.sort_values("iteration").iloc[-1]["value"])
        rows.append({
            "metric": metric,
            "higher_is_better": bool(maximize),
            "iteration_count": int(g_metric["iteration"].max() + 1),
            "best_eval_iteration": int(best["iteration"]),
            "best_eval_value": float(best["value"]),
            "final_eval_iteration": int(final_eval["iteration"]),
            "final_eval_value": float(final_eval["value"]),
            "final_train_value": final_train_value,
            "final_train_minus_eval": float(final_train_value - float(final_eval["value"])) if pd.notna(final_train_value) else np.nan,
        })
    return pd.DataFrame(rows)


def xgboost_booster_importance_frame(pipeline, algorithm: str) -> pd.DataFrame:
    """Return XGBoost booster importance by weight/gain/cover when available."""
    if str(algorithm).lower() != "xgboost":
        return pd.DataFrame()
    if not hasattr(pipeline, "named_steps") or "model" not in pipeline.named_steps:
        return pd.DataFrame()
    model = pipeline.named_steps["model"]
    if not hasattr(model, "get_booster"):
        return pd.DataFrame()
    try:
        booster = model.get_booster()
    except Exception:
        return pd.DataFrame()
    names = prepared_feature_names(pipeline)
    if not names:
        try:
            n = int(booster.num_features())
            names = [f"f{i}" for i in range(n)]
        except Exception:
            names = []
    def map_feature_name(key: str) -> str:
        text = str(key)
        if text.startswith("f") and text[1:].isdigit():
            idx = int(text[1:])
            if 0 <= idx < len(names):
                return names[idx]
        return text
    frames = []
    for importance_type in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        try:
            scores = booster.get_score(importance_type=importance_type)
        except Exception:
            scores = {}
        if not scores:
            continue
        frame = pd.DataFrame([
            {
                "algorithm": "xgboost",
                "prepared_feature": map_feature_name(k),
                "booster_feature_key": str(k),
                "importance_type": importance_type,
                "importance_value": float(v),
            }
            for k, v in scores.items()
        ])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["importance_type", "importance_value"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def model_feature_importance_frame(pipeline, algorithm: str) -> pd.DataFrame:
    """Return feature importance / coefficients from a fitted sklearn pipeline when available."""
    algorithm = str(algorithm).lower()
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        return pd.DataFrame()
    try:
        feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        return pd.DataFrame()
    model = pipeline.named_steps.get("model")
    if model is None:
        return pd.DataFrame()
    values = None
    value_col = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        value_col = "feature_importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = coef.ravel() if coef.ndim <= 2 else coef.reshape(-1)
        value_col = "coefficient"
    if values is None or len(values) != len(feature_names):
        return pd.DataFrame()
    out = pd.DataFrame({
        "algorithm": algorithm,
        "prepared_feature": feature_names,
        value_col: values,
        "absolute_value": np.abs(values),
    })
    return out.sort_values("absolute_value", ascending=False).reset_index(drop=True)
