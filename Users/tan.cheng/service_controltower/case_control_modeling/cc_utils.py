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
def normalize_warranty_claim_filter_mode(value) -> str:
    """Normalize the warranty-label cleaning mode used by all modeling steps."""
    mode = str(value).strip().lower()
    aliases = {
        "none": "none",
        "off": "none",
        "raw": "none",
        "all": "none",
        "failure_focused": "failure_focused",
        "failure-focused": "failure_focused",
        "failure": "failure_focused",
        "clean": "failure_focused",
        "cleaned": "failure_focused",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported WARRANTY_CLAIM_FILTER_MODE={value!r}. "
            "Use 'none' or 'failure_focused'."
        )
    return aliases[mode]


def apply_warranty_claim_filter(
    warranty: pd.DataFrame,
    config,
    mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply transparent claim-label cleaning and return filtered rows plus audit.

    The failure-focused mode is intentionally conservative. It removes claim
    categories that are explicitly inspection/adjustment/routine-wear oriented,
    while retaining Standard Warranty, Advantage, Parts & Components, Product,
    and Remanufactured Components claims. Missing critical-part numbers are not
    removed unless KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS is separately enabled.
    """
    out = warranty.copy()
    selected_mode = normalize_warranty_claim_filter_mode(
        getattr(config, "WARRANTY_CLAIM_FILTER_MODE", "none") if mode is None else mode
    )
    claim_type = out.get(
        "claim_type_description", pd.Series("", index=out.index, dtype=object)
    ).fillna("").astype(str).str.strip()

    keep = pd.Series(True, index=out.index, dtype=bool)
    reason = pd.Series("kept", index=out.index, dtype=object)

    if selected_mode == "failure_focused":
        excluded = getattr(
            config,
            "WARRANTY_FAILURE_FOCUSED_EXCLUDED_CLAIM_TYPES",
            {
                "Inspection Claim KF": "inspection_or_check",
                "Policy Adjustment Claim": "policy_adjustment",
                "Commercial Adjustment Policy Claim": "commercial_adjustment",
                "Replacement Undercarriage": "routine_wear_component_replacement",
            },
        )
        if isinstance(excluded, Mapping):
            excluded_map = {str(k).strip().lower(): str(v) for k, v in excluded.items()}
        else:
            excluded_map = {str(x).strip().lower(): "excluded_minor_claim_type" for x in excluded}
        normalized_type = claim_type.str.lower()
        excluded_mask = normalized_type.isin(excluded_map)
        keep.loc[excluded_mask] = False
        reason.loc[excluded_mask] = normalized_type.loc[excluded_mask].map(excluded_map)

    if bool(getattr(config, "KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS", False)):
        valid_part = out.get(
            "has_valid_critical_part", pd.Series(False, index=out.index, dtype=bool)
        ).fillna(False).astype(bool)
        newly_removed = keep & ~valid_part
        keep.loc[newly_removed] = False
        reason.loc[newly_removed] = "missing_or_invalid_critical_part"

    out["warranty_claim_filter_mode"] = selected_mode
    out["warranty_claim_filter_keep"] = keep.astype(int)
    out["warranty_claim_filter_reason"] = reason

    audit_cols = [
        c
        for c in [
            "machine_key",
            "machine_id",
            "claim_number",
            "claim_date",
            "local_date",
            "claim_type_description",
            "critical_fail_part_number",
            "failure_smr",
            "has_valid_critical_part",
            "warranty_claim_filter_mode",
            "warranty_claim_filter_keep",
            "warranty_claim_filter_reason",
        ]
        if c in out.columns
    ]
    audit = out[audit_cols].copy()
    filtered = out.loc[keep].copy().reset_index(drop=True)
    return filtered, audit.reset_index(drop=True)


def load_warranty(
    config,
    apply_filter: bool = True,
    return_filter_audit: bool = False,
):
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

    if apply_filter:
        filtered, audit = apply_warranty_claim_filter(df, config)
    else:
        # True raw view used only for stable split stratification and diagnostics.
        # It intentionally bypasses both the claim-type filter and the optional
        # valid-critical-part requirement.
        raw = df.copy()
        raw["warranty_claim_filter_mode"] = "raw_unfiltered"
        raw["warranty_claim_filter_keep"] = 1
        raw["warranty_claim_filter_reason"] = "kept_raw_for_split_stratification"
        audit_cols = [
            c
            for c in [
                "machine_key",
                "machine_id",
                "claim_number",
                "claim_date",
                "local_date",
                "claim_type_description",
                "critical_fail_part_number",
                "failure_smr",
                "has_valid_critical_part",
                "warranty_claim_filter_mode",
                "warranty_claim_filter_keep",
                "warranty_claim_filter_reason",
            ]
            if c in raw.columns
        ]
        filtered = raw.reset_index(drop=True)
        audit = raw[audit_cols].copy().reset_index(drop=True)

    if return_filter_audit:
        return filtered, audit
    return filtered


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


def configured_multi_anchor_evaluation_horizons(config) -> List[int]:
    """Return horizons used specifically by Step 10 anchor-fleet evaluation.

    An explicit MULTI_ANCHOR_FLEET_EVALUATION_HORIZONS list decouples the
    deployment-style fleet report from the ratio-sampled holdout experiment.
    Empty or missing values fall back to EVALUATION_CLAIM_HORIZON_DAYS.
    """
    explicit = _clean_horizon_days(
        getattr(config, "MULTI_ANCHOR_FLEET_EVALUATION_HORIZONS", None)
    )
    return sorted(set(explicit or configured_evaluation_horizons(config)))


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
    claim_filter_mode = normalize_warranty_claim_filter_mode(
        getattr(config, "WARRANTY_CLAIM_FILTER_MODE", "none")
    )
    base = (
        f"{lead_label}__neg_{negative_sampling_mode(config)}_"
        f"{negatives_per_positive(config)}__features_{str(config.FEATURE_SET).lower()}"
        f"__claims_{claim_filter_mode}"
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
    sampling_seed_key: Optional[str] = None,
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
    seed_key = (
        str(sampling_seed_key)
        if sampling_seed_key is not None
        else str(case["case_control_group_id"])
    )
    seed = _stable_hash_int(seed_key + "|controlled", seed_value) % (2**32 - 1)
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


def configured_holdout_negative_ratios(config) -> List[int]:
    """Return sorted unique holdout negative:positive ratios.

    The ratios control validation/test composition only. Training continues to
    use NEGATIVE_SAMPLING_MODE and NEGATIVES_PER_POSITIVE_CASE.
    """
    raw = getattr(config, "HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS", [1, 2, 3, 4, 5])
    if isinstance(raw, (str, int, float, np.integer, np.floating)):
        raw = [raw]
    ratios: List[int] = []
    for value in raw:
        ratio = int(value)
        if ratio < 1:
            raise ValueError(
                "HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS must contain integers >= 1. "
                f"Received {value!r}."
            )
        ratios.append(ratio)
    ratios = sorted(set(ratios))
    if not ratios:
        raise ValueError("HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS cannot be empty.")
    return ratios


def holdout_negative_sampling_mode(config) -> str:
    """Return the validation/test cohort design.

    Training uses :func:`negative_sampling_mode` independently. Supported holdout
    designs are ``random``, ``controlled``, and ``as_of_anchor``. The third mode
    changes both positive and negative construction so every machine is scored at
    one shared deployment-like as-of date.
    """
    raw = str(getattr(config, "HOLDOUT_NEGATIVE_SAMPLING_MODE", "random")).strip().lower()
    aliases = {
        "random": "random",
        "independent_random": "random",
        "controlled": "controlled",
        "matched": "controlled",
        "as_of_anchor": "as_of_anchor",
        "asof_anchor": "as_of_anchor",
        "anchored_date": "as_of_anchor",
        "as_of_date": "as_of_anchor",
        "deployment_snapshot": "as_of_anchor",
    }
    if raw not in aliases:
        raise ValueError(
            "HOLDOUT_NEGATIVE_SAMPLING_MODE must be 'random', 'controlled', "
            "or 'as_of_anchor'."
        )
    return aliases[raw]


def holdout_machine_selection_scope(config, window_config: Mapping) -> str:
    """Return the deterministic machine-ranking scope for holdout construction.

    A normal single-window run retains the historical window-specific hash token,
    which preserves existing fixed random holdouts and prior point estimates. When
    two or more distinct ``WINDOW_CONFIGS`` are evaluated together, every window
    uses one shared token. This aligns the positive-machine ranking, claim-event
    selection, and negative-machine ranking as closely as eligibility permits, so
    differences between window experiments are not mainly caused by resampling a
    different holdout population.
    """
    raw_configs = getattr(config, "WINDOW_CONFIGS", [window_config])
    normalized = set()
    try:
        for item in raw_configs:
            normalized.add(
                (int(item["lead_max_days"]), int(item["lead_min_days"]))
            )
    except Exception:
        normalized = {
            (int(window_config["lead_max_days"]), int(window_config["lead_min_days"]))
        }
    if len(normalized) <= 1:
        return window_config_name(window_config)
    return "shared_across_configured_window_designs_v1"


def _random_holdout_positive_candidates(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    split_assignments: pd.DataFrame,
    window_config: Mapping,
    config,
    included_splits: Sequence[str],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create at most one deterministic random eligible claim window per machine."""
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    window_name = window_config_name(window_config)
    selection_scope = holdout_machine_selection_scope(config, window_config)
    requested_splits = {str(x) for x in included_splits}

    coverage_cols = [
        c for c in ["machine_key", "first_source_date", "last_source_date"]
        if c in machine_master.columns
    ]
    coverage = machine_master[coverage_cols].drop_duplicates("machine_key").copy()
    if "first_source_date" not in coverage.columns or "last_source_date" not in coverage.columns:
        raise ValueError(
            "machine_master must contain first_source_date and last_source_date "
            "before random holdout construction."
        )

    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in split_assignments.columns
    ]
    assignments = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
    work = episodes.copy()
    work["machine_key"] = work["machine_key"].astype(str)
    work["claim_date"] = pd.to_datetime(work["claim_date"], errors="coerce")
    work = work.dropna(subset=["machine_key", "claim_date"])
    work = work.merge(assignments, on="machine_key", how="left", suffixes=("", "_assigned"))
    work = work.merge(coverage, on="machine_key", how="left", validate="many_to_one")
    work = work[work["split"].astype(str).isin(requested_splits)].copy()

    if "full_model_assigned" in work.columns:
        current = work.get("full_model", pd.Series("", index=work.index)).astype("string").fillna("").str.strip()
        work["full_model"] = work.get("full_model", pd.Series("", index=work.index)).where(
            current.ne(""), work["full_model_assigned"]
        )
    if "serial_assigned" in work.columns:
        current = work.get("serial", pd.Series("", index=work.index)).astype("string").fillna("").str.strip()
        work["serial"] = work.get("serial", pd.Series("", index=work.index)).where(
            current.ne(""), work["serial_assigned"]
        )
    work["full_model"] = work.get("full_model", pd.Series("", index=work.index)).map(clean_model)
    work["serial"] = work.get("serial", pd.Series("", index=work.index)).map(clean_serial)
    work["first_source_date"] = pd.to_datetime(work["first_source_date"], errors="coerce")
    work["last_source_date"] = pd.to_datetime(work["last_source_date"], errors="coerce")
    work["window_start"] = work["claim_date"] - pd.to_timedelta(lead_max, unit="D")
    work["window_end"] = work["claim_date"] - pd.to_timedelta(lead_min, unit="D")

    if bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
        eligible = (
            work["first_source_date"].notna()
            & work["last_source_date"].notna()
            & (work["first_source_date"] <= work["window_end"])
            & (work["last_source_date"] >= work["window_start"])
        )
    else:
        eligible = pd.Series(True, index=work.index)
    work["holdout_candidate_status"] = np.where(
        eligible, "eligible_positive_claim_window", "excluded_no_source_coverage_overlap_window"
    )

    audit_cols = [
        c for c in [
            "machine_key", "split", "full_model", "serial", "claim_episode_id",
            "claim_date", "window_start", "window_end", "holdout_candidate_status",
        ] if c in work.columns
    ]
    audit = work[audit_cols].copy()
    audit.insert(0, "candidate_type", "positive")

    eligible_work = work.loc[eligible].copy()
    if eligible_work.empty:
        return pd.DataFrame(), audit
    eligible_work["_holdout_hash"] = eligible_work.apply(
        lambda row: _stable_hash_int(
            f"{selection_scope}|{row['split']}|{row['machine_key']}|"
            f"{row.get('claim_episode_id', '')}|positive_claim_window",
            random_state,
        ),
        axis=1,
    )
    eligible_work = eligible_work.sort_values(
        ["split", "machine_key", "_holdout_hash", "claim_date", "claim_episode_id"],
        kind="mergesort",
    )
    eligible_work = eligible_work.drop_duplicates(["split", "machine_key"], keep="first")

    rows = pd.DataFrame(
        {
            "row_role": "case",
            "target": 1,
            "case_control_group_id": eligible_work.apply(
                lambda row: (
                    f"{window_name}__random_holdout__{row['split']}__case__"
                    f"{row.get('claim_episode_id', row['machine_key'])}"
                ),
                axis=1,
            ),
            "case_machine_key": eligible_work["machine_key"].astype(str),
            "claim_episode_id": eligible_work.get("claim_episode_id", ""),
            "control_number_within_group": np.nan,
            "machine_key": eligible_work["machine_key"].astype(str),
            "full_model": eligible_work["full_model"],
            "serial": eligible_work["serial"],
            "split": eligible_work["split"].astype(str),
            "window_name": window_name,
            "lead_max_days": lead_max,
            "lead_min_days": lead_min,
            "window_start": eligible_work["window_start"],
            "window_end": eligible_work["window_end"],
            "linked_case_window_start": pd.NaT,
            "linked_case_window_end": pd.NaT,
            "future_claim_date": eligible_work["claim_date"],
            "days_from_window_end_to_claim": float(lead_min),
            "negative_sampling_type": "case",
            "control_sampling_reason": "claim_anchored_positive_random_machine_holdout",
            "control_no_claim_start": pd.NaT,
            "control_no_claim_end": pd.NaT,
            "holdout_sampling_design": "random_machine_level_ratio_sweep",
            "_holdout_hash": eligible_work["_holdout_hash"].to_numpy(),
        }
    )
    for extra_col in [
        "claim_count_in_episode", "claim_numbers", "claim_type_descriptions",
        "critical_fail_part_numbers", "positive_claim_selection_mode",
        "claim_sequence_number", "machine_claim_event_count",
        "days_since_previous_claim_same_machine", "claim_selection_reason",
    ]:
        if extra_col in eligible_work.columns:
            rows[extra_col] = eligible_work[extra_col].to_numpy()
    return rows.reset_index(drop=True), audit.reset_index(drop=True)


def _random_holdout_negative_candidates(
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    claim_history_episodes: pd.DataFrame,
    split_assignments: pd.DataFrame,
    positive_candidate_machines: set[str],
    window_config: Mapping,
    config,
    included_splits: Sequence[str],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one deterministic eligible no-claim window per candidate machine."""
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = int(lead_max - lead_min)
    window_name = window_config_name(window_config)
    selection_scope = holdout_machine_selection_scope(config, window_config)
    requested_splits = {str(x) for x in included_splits}

    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in split_assignments.columns
    ]
    assignments = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
    master = master.merge(assignments, on="machine_key", how="left", suffixes=("", "_assigned"))
    master = master[master["split"].astype(str).isin(requested_splits)].copy()
    for field in ["full_model", "serial"]:
        assigned = f"{field}_assigned"
        if assigned in master.columns:
            current = master.get(field, pd.Series("", index=master.index)).astype("string").fillna("").str.strip()
            master[field] = master.get(field, pd.Series("", index=master.index)).where(
                current.ne(""), master[assigned]
            )
            master = master.drop(columns=[assigned])
    master["full_model"] = master.get("full_model", pd.Series("", index=master.index)).map(clean_model)
    master["serial"] = master.get("serial", pd.Series("", index=master.index)).map(clean_serial)

    dates_by_machine = claim_dates_by_machine(claim_history_episodes)
    eligible_windows = _eligible_random_windows_by_machine(
        sources=sources,
        machine_master=master,
        dates_by_machine=dates_by_machine,
        lookback_days=observation_days,
        config=config,
    )
    prior_days = int(getattr(config, "NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START", 30))
    future_days = int(getattr(config, "NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))

    rows: List[dict] = []
    audit_rows: List[dict] = []
    for _, machine in master.sort_values(["split", "machine_key"], kind="mergesort").iterrows():
        machine_key = str(machine["machine_key"])
        split_name = str(machine["split"])
        status = "eligible_negative_window"
        selected_end = pd.NaT
        raw_dates = np.asarray(
            eligible_windows.get(machine_key, np.array([], dtype="datetime64[ns]")),
            dtype="datetime64[ns]",
        )
        # A machine may be eligible as both a claim-anchored positive and a
        # no-claim-window negative. The final joint sampler assigns it to only
        # one class, which preserves machine independence while avoiding the
        # severe holdout-size reduction caused by excluding every machine that
        # has any eligible claim history.
        if len(raw_dates) == 0:
            status = "excluded_no_eligible_random_negative_window"
        else:
            selector = _stable_hash_int(
                f"{selection_scope}|{split_name}|{machine_key}|random_holdout_negative_window",
                random_state,
            )
            selected_end = pd.Timestamp(raw_dates[selector % len(raw_dates)])

        audit_rows.append(
            {
                "candidate_type": "negative",
                "machine_key": machine_key,
                "split": split_name,
                "full_model": machine.get("full_model", ""),
                "serial": machine.get("serial", ""),
                "holdout_candidate_status": status,
                "machine_has_eligible_positive_claim_window": (
                    machine_key in positive_candidate_machines
                ),
                "eligible_window_count": int(len(raw_dates)),
                "selected_window_end": selected_end,
            }
        )
        if pd.isna(selected_end):
            continue

        window_end = pd.Timestamp(selected_end)
        window_start = window_end - pd.Timedelta(days=observation_days)
        row_id = f"{window_name}__random_holdout__{split_name}__control__{machine_key}"
        sort_hash = _stable_hash_int(
            f"{selection_scope}|{split_name}|{machine_key}|random_holdout_negative_order",
            random_state,
        )
        rows.append(
            {
                "row_role": "control",
                "target": 0,
                "case_control_group_id": row_id,
                "case_machine_key": machine_key,
                "claim_episode_id": "",
                "control_number_within_group": 1,
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
                "negative_sampling_type": "random_holdout",
                "control_sampling_reason": (
                    "random_machine_window_no_claim_in_configured_exclusion_interval"
                ),
                "control_no_claim_start": window_start - pd.Timedelta(days=prior_days),
                "control_no_claim_end": window_end + pd.Timedelta(days=future_days),
                "holdout_sampling_design": "random_machine_level_ratio_sweep",
                "_holdout_hash": sort_hash,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def build_random_holdout_ratio_base_pool(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
    included_splits: Sequence[str] = ("validation", "test"),
    random_state_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a fixed random validation/test master pool for ratio experiments.

    Design properties:
    - machine-level split assignments are respected;
    - each holdout machine contributes at most one row;
    - positive rows use the configured claim-relative feature window;
    - negative rows use independently sampled eligible no-claim windows;
    - positives and negatives are not matched by case, calendar date, or full model;
    - the master pool contains enough nested negatives for the largest configured
      negative:positive ratio.
    """
    if split_assignments is None:
        raise ValueError("split_assignments are required for random holdout construction.")
    ratios = configured_holdout_negative_ratios(config)
    max_ratio = int(max(ratios))
    seed = int(
        getattr(config, "HOLDOUT_RANDOM_STATE", getattr(config, "FIXED_SPLIT_RANDOM_STATE", 42))
        if random_state_override is None
        else random_state_override
    )
    history = claim_history_episodes if claim_history_episodes is not None else episodes

    positives, positive_audit = _random_holdout_positive_candidates(
        episodes=episodes,
        machine_master=machine_master,
        split_assignments=split_assignments,
        window_config=window_config,
        config=config,
        included_splits=included_splits,
        random_state=seed,
    )
    positive_candidate_machines = set(positives.get("machine_key", pd.Series(dtype=str)).astype(str))
    negatives, negative_audit = _random_holdout_negative_candidates(
        machine_master=machine_master,
        sources=sources,
        claim_history_episodes=history,
        split_assignments=split_assignments,
        positive_candidate_machines=positive_candidate_machines,
        window_config=window_config,
        config=config,
        included_splits=included_splits,
        random_state=seed,
    )

    selected_parts: List[pd.DataFrame] = []
    summary_rows: List[dict] = []
    selection_audit_rows: List[dict] = []
    for split_name in [str(x) for x in included_splits]:
        pos = positives[positives["split"].astype(str).eq(split_name)].copy()
        neg = negatives[negatives["split"].astype(str).eq(split_name)].copy()
        pos = pos.sort_values(["_holdout_hash", "machine_key"], kind="mergesort")
        neg = neg.sort_values(["_holdout_hash", "machine_key"], kind="mergesort")

        # Positives and negatives are selected jointly because a machine can
        # have both an eligible claim-anchored window and an eligible no-claim
        # window. Start from the largest feasible positive count and decrease
        # only until enough *different* machines remain for the requested
        # maximum negative ratio. This keeps the holdout random and much larger
        # than the old rule that excluded every claim-history machine from the
        # negative pool.
        union_machine_count = int(
            len(set(pos["machine_key"].astype(str)) | set(neg["machine_key"].astype(str)))
        )
        selected_positive_count = min(
            int(len(pos)), int(union_machine_count // (max_ratio + 1))
        )
        selected_pos = pd.DataFrame()
        available_neg = pd.DataFrame()
        while selected_positive_count >= 1:
            selected_pos = pos.head(selected_positive_count).copy()
            selected_positive_keys = set(selected_pos["machine_key"].astype(str))
            available_neg = neg[
                ~neg["machine_key"].astype(str).isin(selected_positive_keys)
            ].copy()
            if len(available_neg) >= selected_positive_count * max_ratio:
                break
            selected_positive_count -= 1

        if selected_positive_count < 1:
            summary_rows.append(
                {
                    "split": split_name,
                    "positive_candidates": int(len(pos)),
                    "negative_candidates": int(len(neg)),
                    "candidate_machine_overlap": int(
                        len(set(pos["machine_key"].astype(str)) & set(neg["machine_key"].astype(str)))
                    ),
                    "unique_candidate_machines": union_machine_count,
                    "selected_positive_rows": 0,
                    "selected_negative_rows_at_max_ratio": 0,
                    "max_negative_to_positive_ratio": max_ratio,
                    "status": "insufficient_candidates",
                }
            )
            continue

        selected_neg = available_neg.head(selected_positive_count * max_ratio).copy()
        selected_pos["holdout_positive_rank"] = np.arange(1, len(selected_pos) + 1)
        selected_pos["holdout_negative_rank"] = np.nan
        selected_neg["holdout_positive_rank"] = np.nan
        selected_neg["holdout_negative_rank"] = np.arange(1, len(selected_neg) + 1)
        for frame in [selected_pos, selected_neg]:
            frame["holdout_max_negative_to_positive_ratio"] = max_ratio
            frame["holdout_random_state"] = seed
        selected_parts.extend([selected_pos, selected_neg])

        selected_pos_keys = set(selected_pos["machine_key"].astype(str))
        selected_neg_keys = set(selected_neg["machine_key"].astype(str))
        for _, row in pos.iterrows():
            selection_audit_rows.append(
                {
                    "candidate_type": "positive",
                    "machine_key": row["machine_key"],
                    "split": split_name,
                    "selected_in_master_pool": str(row["machine_key"]) in selected_pos_keys,
                    "selection_reason": (
                        "selected_random_positive_machine"
                        if str(row["machine_key"]) in selected_pos_keys
                        else "excluded_to_preserve_all_requested_negative_ratios"
                    ),
                }
            )
        for _, row in neg.iterrows():
            selection_audit_rows.append(
                {
                    "candidate_type": "negative",
                    "machine_key": row["machine_key"],
                    "split": split_name,
                    "selected_in_master_pool": str(row["machine_key"]) in selected_neg_keys,
                    "selection_reason": (
                        "selected_random_negative_machine"
                        if str(row["machine_key"]) in selected_neg_keys
                        else (
                            "excluded_machine_selected_as_positive"
                            if str(row["machine_key"]) in selected_pos_keys
                            else "not_needed_for_configured_max_ratio"
                        )
                    ),
                }
            )
        summary_rows.append(
            {
                "split": split_name,
                "positive_candidates": int(len(pos)),
                "negative_candidates": int(len(neg)),
                "candidate_machine_overlap": int(
                    len(set(pos["machine_key"].astype(str)) & set(neg["machine_key"].astype(str)))
                ),
                "unique_candidate_machines": union_machine_count,
                "selected_positive_rows": int(len(selected_pos)),
                "selected_negative_rows_at_max_ratio": int(len(selected_neg)),
                "max_negative_to_positive_ratio": max_ratio,
                "status": "selected",
            }
        )

    master_pool = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    if not master_pool.empty:
        master_pool = master_pool.sort_values(
            ["split", "target", "_holdout_hash", "machine_key"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if master_pool["machine_key"].duplicated().any():
            duplicate_keys = master_pool.loc[
                master_pool["machine_key"].duplicated(keep=False), "machine_key"
            ].astype(str).unique().tolist()[:10]
            raise ValueError(
                "Random holdout master pool contains duplicate machines. "
                f"Examples: {duplicate_keys}"
            )
        master_pool = master_pool.drop(columns=["_holdout_hash"], errors="ignore")

    detailed_audit = pd.concat(
        [positive_audit, negative_audit, pd.DataFrame(selection_audit_rows)],
        ignore_index=True,
        sort=False,
    )
    return master_pool, detailed_audit, pd.DataFrame(summary_rows)


def build_controlled_holdout_ratio_base_pool(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
    included_splits: Sequence[str] = ("validation", "test"),
    random_state_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build fixed matched validation/test cohorts for ratio experiments.

    Each retained positive machine is matched to ``max(ratios)`` different
    control machines from the same split and full model. Controls use the exact
    positive calendar window and satisfy the normal no-claim and source-coverage
    rules. A physical machine can appear only once in the master pool.
    """
    if split_assignments is None:
        raise ValueError("split_assignments are required for controlled holdout construction.")
    ratios = configured_holdout_negative_ratios(config)
    max_ratio = int(max(ratios))
    seed = int(
        getattr(config, "HOLDOUT_RANDOM_STATE", getattr(config, "FIXED_SPLIT_RANDOM_STATE", 42))
        if random_state_override is None
        else random_state_override
    )
    history = claim_history_episodes if claim_history_episodes is not None else episodes
    requested_splits = [str(x) for x in included_splits]
    window_name = window_config_name(window_config)
    selection_scope = holdout_machine_selection_scope(config, window_config)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])

    positives, positive_audit = _random_holdout_positive_candidates(
        episodes=episodes,
        machine_master=machine_master,
        split_assignments=split_assignments,
        window_config=window_config,
        config=config,
        included_splits=included_splits,
        random_state=seed,
    )

    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in split_assignments.columns
    ]
    assignments = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
    master = master.merge(assignments, on="machine_key", how="left", suffixes=("", "_assigned"))
    for field in ["full_model", "serial", "split"]:
        assigned = f"{field}_assigned"
        if assigned not in master.columns:
            continue
        if field not in master.columns:
            master[field] = master[assigned]
        else:
            current = master[field].astype("string").fillna("").str.strip()
            master[field] = master[field].where(current.ne(""), master[assigned])
        master = master.drop(columns=[assigned])
    master["full_model"] = master.get("full_model", pd.Series("", index=master.index)).map(clean_model)
    master["serial"] = master.get("serial", pd.Series("", index=master.index)).map(clean_serial)
    master["split"] = master.get("split", pd.Series("", index=master.index)).astype(str)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master = master[master["split"].isin(requested_splits)].copy()
    master_by_model_split = {
        (str(model), str(split)): group.reset_index(drop=True)
        for (model, split), group in master.groupby(["full_model", "split"], dropna=False)
    }
    dates_by_machine = claim_dates_by_machine(history)

    selected_parts: List[pd.DataFrame] = []
    selection_audit_rows: List[dict] = []
    summary_rows: List[dict] = []
    for split_name in requested_splits:
        split_pos = positives[positives["split"].astype(str).eq(split_name)].copy()
        split_pos = split_pos.sort_values(["_holdout_hash", "machine_key"], kind="mergesort")
        used_machines: set[str] = set()
        accepted_cases: List[dict] = []
        accepted_controls: List[dict] = []
        rejected_short_controls = 0
        rejected_used_as_control = 0

        for _, raw_case in split_pos.iterrows():
            case = raw_case.copy()
            case_machine = str(case["machine_key"])
            if case_machine in used_machines:
                rejected_used_as_control += 1
                selection_audit_rows.append({
                    "candidate_type": "positive",
                    "machine_key": case_machine,
                    "split": split_name,
                    "selected_in_master_pool": False,
                    "selection_reason": "excluded_machine_already_selected_as_control",
                })
                continue

            case_id = str(case["case_control_group_id"]).replace(
                "__random_holdout__", "__controlled_holdout__"
            )
            case["case_control_group_id"] = case_id
            case["holdout_match_id"] = case_id
            case["holdout_sampling_design"] = "controlled_same_window_same_full_model_ratio_sweep"
            case["control_sampling_reason"] = "claim_anchored_positive_controlled_machine_holdout"
            case["negative_sampling_type"] = "case"

            pool = master_by_model_split.get(
                (str(case.get("full_model", "")), split_name), pd.DataFrame()
            ).copy()
            if not pool.empty:
                pool = pool[
                    ~pool["machine_key"].astype(str).isin(used_machines | {case_machine})
                ].copy()
            selected, control_audit = _controlled_negative_rows(
                case=case,
                candidates=pool,
                count=max_ratio,
                dates_by_machine=dates_by_machine,
                sources=sources,
                config=config,
                excluded_machines=used_machines | {case_machine},
                random_state=seed,
                sampling_seed_key=(
                    f"{selection_scope}|{split_name}|{case_machine}|"
                    f"{case.get('claim_episode_id', '')}|controlled_holdout"
                ),
            )
            if len(selected) < max_ratio:
                rejected_short_controls += 1
                selection_audit_rows.append({
                    "candidate_type": "positive",
                    "machine_key": case_machine,
                    "split": split_name,
                    "full_model": case.get("full_model", ""),
                    "claim_episode_id": case.get("claim_episode_id", ""),
                    "selected_in_master_pool": False,
                    "selection_reason": "excluded_insufficient_unique_controlled_negatives",
                    "requested_control_count": max_ratio,
                    "eligible_control_count_selected": len(selected),
                    "control_audit": json.dumps(control_audit, default=str),
                })
                continue

            accepted_cases.append(case.to_dict())
            used_machines.add(case_machine)
            for j, negative in enumerate(selected, start=1):
                control_machine = str(negative["machine_key"])
                used_machines.add(control_machine)
                accepted_controls.append({
                    "row_role": "control",
                    "target": 0,
                    "case_control_group_id": (
                        f"{case_id}__control__{j}__{control_machine}"
                    ),
                    "holdout_match_id": case_id,
                    "case_machine_key": case_machine,
                    "matched_positive_machine_key": case_machine,
                    "claim_episode_id": case.get("claim_episode_id", ""),
                    "control_number_within_group": j,
                    "holdout_control_rank_within_positive": j,
                    "machine_key": control_machine,
                    "full_model": negative.get("full_model", case.get("full_model", "")),
                    "serial": negative.get("serial", ""),
                    "split": split_name,
                    "window_name": window_name,
                    "lead_max_days": lead_max,
                    "lead_min_days": lead_min,
                    "window_start": negative["window_start"],
                    "window_end": negative["window_end"],
                    "linked_case_window_start": case["window_start"],
                    "linked_case_window_end": case["window_end"],
                    "future_claim_date": pd.NaT,
                    "days_from_window_end_to_claim": np.nan,
                    "negative_sampling_type": "controlled_holdout",
                    "control_sampling_reason": negative["control_sampling_reason"],
                    "control_no_claim_start": negative["control_no_claim_start"],
                    "control_no_claim_end": negative["control_no_claim_end"],
                    "holdout_sampling_design": "controlled_same_window_same_full_model_ratio_sweep",
                    "_holdout_hash": _stable_hash_int(
                        f"{selection_scope}|{split_name}|{case_machine}|{j}|"
                        f"{control_machine}|controlled_holdout_order", seed
                    ),
                })
                selection_audit_rows.append({
                    "candidate_type": "negative",
                    "machine_key": control_machine,
                    "split": split_name,
                    "full_model": negative.get("full_model", ""),
                    "selected_in_master_pool": True,
                    "selection_reason": "selected_controlled_same_window_same_full_model_negative",
                    "matched_positive_machine_key": case_machine,
                    "holdout_control_rank_within_positive": j,
                })
            selection_audit_rows.append({
                "candidate_type": "positive",
                "machine_key": case_machine,
                "split": split_name,
                "full_model": case.get("full_model", ""),
                "claim_episode_id": case.get("claim_episode_id", ""),
                "selected_in_master_pool": True,
                "selection_reason": "selected_controlled_positive_machine",
                "requested_control_count": max_ratio,
                "eligible_control_count_selected": max_ratio,
                "control_audit": json.dumps(control_audit, default=str),
            })

        cases_df = pd.DataFrame(accepted_cases)
        controls_df = pd.DataFrame(accepted_controls)
        if not cases_df.empty:
            cases_df = cases_df.sort_values(["_holdout_hash", "machine_key"], kind="mergesort").reset_index(drop=True)
            cases_df["holdout_positive_rank"] = np.arange(1, len(cases_df) + 1)
            cases_df["holdout_negative_rank"] = np.nan
            cases_df["holdout_control_rank_within_positive"] = np.nan
            rank_map = dict(zip(cases_df["case_control_group_id"].astype(str), cases_df["holdout_positive_rank"]))
            cases_df["matched_holdout_positive_rank"] = cases_df["holdout_positive_rank"]
            if not controls_df.empty:
                controls_df["matched_holdout_positive_rank"] = controls_df["holdout_match_id"].astype(str).map(rank_map)
                controls_df = controls_df.sort_values(
                    ["matched_holdout_positive_rank", "holdout_control_rank_within_positive", "machine_key"],
                    kind="mergesort",
                ).reset_index(drop=True)
                controls_df["holdout_positive_rank"] = np.nan
                controls_df["holdout_negative_rank"] = np.arange(1, len(controls_df) + 1)
            for frame in [cases_df, controls_df]:
                if frame.empty:
                    continue
                frame["holdout_max_negative_to_positive_ratio"] = max_ratio
                frame["holdout_random_state"] = seed
            selected_parts.extend([cases_df, controls_df])

        summary_rows.append({
            "split": split_name,
            "holdout_negative_sampling_mode": "controlled",
            "positive_candidates": int(len(split_pos)),
            "selected_positive_rows": int(len(cases_df)),
            "selected_negative_rows_at_max_ratio": int(len(controls_df)),
            "max_negative_to_positive_ratio": max_ratio,
            "positive_candidates_excluded_insufficient_controls": int(rejected_short_controls),
            "positive_candidates_excluded_used_as_control": int(rejected_used_as_control),
            "status": "selected" if len(cases_df) else "insufficient_candidates",
        })

    master_pool = (
        pd.concat(selected_parts, ignore_index=True, sort=False)
        if selected_parts
        else pd.DataFrame()
    )
    if not master_pool.empty:
        master_pool = master_pool.sort_values(
            ["split", "target", "holdout_positive_rank", "holdout_negative_rank", "machine_key"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if master_pool["machine_key"].astype(str).duplicated().any():
            duplicate_keys = master_pool.loc[
                master_pool["machine_key"].astype(str).duplicated(keep=False), "machine_key"
            ].astype(str).unique().tolist()[:10]
            raise ValueError(
                "Controlled holdout master pool contains duplicate machines. "
                f"Examples: {duplicate_keys}"
            )
        master_pool = master_pool.drop(columns=["_holdout_hash"], errors="ignore")

    detailed_audit = pd.concat(
        [positive_audit, pd.DataFrame(selection_audit_rows)],
        ignore_index=True,
        sort=False,
    )
    return master_pool, detailed_audit, pd.DataFrame(summary_rows)



def _as_of_anchor_followup_days(window_config: Mapping, config) -> int:
    """Return required observable future days for an anchored holdout."""
    lead_min = int(window_config["lead_min_days"])
    mode = str(getattr(config, "EVALUATION_TARGET_MODE", "training_target")).strip().lower()
    horizons = configured_evaluation_horizons(config) if mode == "claim_within_horizon" else []
    return max([lead_min, *[int(x) for x in horizons]])


def _as_of_anchor_candidate_dates(
    episodes: pd.DataFrame,
    window_config: Mapping,
    config,
    followup_days_override: Optional[int] = None,
) -> list[pd.Timestamp]:
    """Return deterministic daily anchor candidates near observed claim dates."""
    lead_min = int(window_config["lead_min_days"])
    observation_days = int(window_config["lead_max_days"]) - lead_min
    followup_days = (
        int(followup_days_override)
        if followup_days_override is not None
        else _as_of_anchor_followup_days(window_config, config)
    )
    include_cutoff = bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True))

    claim_dates = pd.to_datetime(episodes.get("claim_date"), errors="coerce").dropna().drop_duplicates()
    if claim_dates.empty:
        return []
    lower = pd.Timestamp(getattr(config, "MIN_VALID_EVENT_DATE", claim_dates.min())) + pd.Timedelta(days=observation_days)
    upper = _max_claim_observation_date(config) - pd.Timedelta(days=followup_days)
    first_delta = 0 if include_cutoff else 1
    candidates: set[pd.Timestamp] = set()
    for claim_date in claim_dates:
        for delta in range(first_delta, lead_min + 1):
            anchor = pd.Timestamp(claim_date).normalize() - pd.Timedelta(days=delta)
            if lower <= anchor <= upper:
                candidates.add(anchor)
    return sorted(candidates)


def _as_of_anchor_machine_candidates(
    anchor_date: pd.Timestamp,
    prepared_master: pd.DataFrame,
    dates_by_machine: Mapping[str, np.ndarray],
    window_config: Mapping,
    config,
    split_name: str,
) -> pd.DataFrame:
    """Label all source-covered machines at one shared as-of date."""
    anchor = pd.Timestamp(anchor_date).normalize()
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = lead_max - lead_min
    window_start = anchor - pd.Timedelta(days=observation_days)
    include_cutoff = bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True))

    sub = prepared_master[prepared_master["split"].astype(str).eq(str(split_name))].copy()
    if bool(getattr(config, "REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
        sub = sub[
            sub["first_source_date"].notna()
            & sub["last_source_date"].notna()
            & (sub["first_source_date"] <= anchor)
            & (sub["last_source_date"] >= window_start)
        ].copy()
    if sub.empty:
        return sub

    next_dates: list[pd.Timestamp] = []
    days_values: list[float] = []
    for machine_key in sub["machine_key"].astype(str):
        next_date, days = next_claim_on_or_after(
            dates_by_machine,
            machine_key,
            anchor,
            include_cutoff=include_cutoff,
        )
        next_dates.append(next_date)
        days_values.append(days)
    sub["as_of_anchor_date"] = anchor
    sub["window_start"] = window_start
    sub["window_end"] = anchor
    sub["as_of_actual_next_claim_date"] = pd.to_datetime(next_dates, errors="coerce")
    sub["as_of_days_to_next_claim"] = pd.to_numeric(days_values, errors="coerce")
    days = sub["as_of_days_to_next_claim"]
    lower_ok = days.ge(0) if include_cutoff else days.gt(0)
    sub["target"] = (days.notna() & lower_ok & days.le(float(lead_min))).astype(int)
    return sub


def build_as_of_anchor_holdout_ratio_base_pool(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
    included_splits: Sequence[str] = ("validation", "test"),
    random_state_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a deployment-like holdout at one shared random as-of date.

    Every selected validation/test row uses the same calendar ``window_end``.
    The design target is whether a claim occurs within ``lead_min_days`` after
    the anchor. Larger evaluation horizons can relabel the same fixed machines
    without changing their feature windows or scores.
    """
    if split_assignments is None:
        raise ValueError("split_assignments are required for as_of_anchor holdout construction.")
    requested_splits = [str(x) for x in included_splits]
    ratios = configured_holdout_negative_ratios(config)
    max_ratio = int(max(ratios))
    seed = int(
        getattr(config, "HOLDOUT_RANDOM_STATE", getattr(config, "FIXED_SPLIT_RANDOM_STATE", 42))
        if random_state_override is None
        else random_state_override
    )
    history = claim_history_episodes if claim_history_episodes is not None else episodes
    dates_by_machine = claim_dates_by_machine(history)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = lead_max - lead_min
    window_name = window_config_name(window_config)
    selection_scope = holdout_machine_selection_scope(config, window_config)

    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in split_assignments.columns
    ]
    assignments = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
    master = master.merge(assignments, on="machine_key", how="left", suffixes=("", "_assigned"))
    for field in ["full_model", "serial"]:
        assigned = f"{field}_assigned"
        if assigned in master.columns:
            current = master.get(field, pd.Series("", index=master.index)).astype("string").fillna("").str.strip()
            master[field] = master.get(field, pd.Series("", index=master.index)).where(current.ne(""), master[assigned])
            master = master.drop(columns=[assigned])
    master["full_model"] = master.get("full_model", pd.Series("", index=master.index)).map(clean_model)
    master["serial"] = master.get("serial", pd.Series("", index=master.index)).map(clean_serial)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master = master[master["split"].astype(str).isin(requested_splits)].copy()

    candidates = _as_of_anchor_candidate_dates(history, window_config, config)
    if not candidates:
        raise ValueError("No eligible as-of anchor dates are available after applying follow-up limits.")
    min_positive = max(1, int(getattr(config, "HOLDOUT_AS_OF_MIN_POSITIVE_MACHINES", 10)))
    candidate_order = sorted(
        candidates,
        key=lambda date: _stable_hash_int(
            f"{selection_scope}|{pd.Timestamp(date).date()}|as_of_anchor_date", seed
        ),
    )

    search_rows: list[dict] = []
    chosen_anchor: Optional[pd.Timestamp] = None
    best_anchor: Optional[pd.Timestamp] = None
    best_capacity = -1
    for anchor in candidate_order:
        split_capacities: dict[str, int] = {}
        feasible_minimum = True
        search_row: dict = {"candidate_type": "anchor_date", "as_of_anchor_date": anchor}
        for split_name in requested_splits:
            candidates_df = _as_of_anchor_machine_candidates(
                anchor, master, dates_by_machine, window_config, config, split_name
            )
            positive_count = int(candidates_df["target"].eq(1).sum()) if not candidates_df.empty else 0
            negative_count = int(candidates_df["target"].eq(0).sum()) if not candidates_df.empty else 0
            capacity = min(positive_count, negative_count // max_ratio)
            split_capacities[split_name] = capacity
            search_row[f"{split_name}_eligible_machines"] = int(len(candidates_df))
            search_row[f"{split_name}_positive_candidates"] = positive_count
            search_row[f"{split_name}_negative_candidates"] = negative_count
            search_row[f"{split_name}_max_ratio_positive_capacity"] = capacity
            feasible_minimum = feasible_minimum and capacity >= min_positive
        common_capacity = min(split_capacities.values()) if split_capacities else 0
        search_row["common_min_positive_capacity"] = common_capacity
        search_row["meets_requested_minimum"] = bool(feasible_minimum)
        search_rows.append(search_row)
        if common_capacity > best_capacity:
            best_capacity = common_capacity
            best_anchor = pd.Timestamp(anchor)
        if feasible_minimum:
            chosen_anchor = pd.Timestamp(anchor)
            break
    if chosen_anchor is None:
        chosen_anchor = best_anchor
    if chosen_anchor is None or best_capacity < 1:
        raise ValueError(
            "No as-of anchor date has both positive and negative machines for every requested split."
        )

    episode_lookup = history.copy()
    episode_lookup["machine_key"] = episode_lookup["machine_key"].astype(str)
    episode_lookup["claim_date"] = pd.to_datetime(episode_lookup["claim_date"], errors="coerce")
    episode_lookup = episode_lookup.sort_values(
        ["machine_key", "claim_date", "claim_episode_id"], kind="mergesort"
    ).drop_duplicates(["machine_key", "claim_date"], keep="first")
    episode_id_map = {
        (str(row.machine_key), pd.Timestamp(row.claim_date)): str(getattr(row, "claim_episode_id", ""))
        for row in episode_lookup.itertuples(index=False)
        if pd.notna(row.claim_date)
    }

    selected_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = search_rows
    summary_rows: list[dict] = []
    for split_name in requested_splits:
        candidates_df = _as_of_anchor_machine_candidates(
            chosen_anchor, master, dates_by_machine, window_config, config, split_name
        )
        positives = candidates_df[candidates_df["target"].eq(1)].copy()
        negatives = candidates_df[candidates_df["target"].eq(0)].copy()
        positives["_holdout_hash"] = positives["machine_key"].astype(str).map(
            lambda m: _stable_hash_int(
                f"{selection_scope}|{chosen_anchor.date()}|{split_name}|{m}|as_of_positive", seed
            )
        )
        negatives["_holdout_hash"] = negatives["machine_key"].astype(str).map(
            lambda m: _stable_hash_int(
                f"{selection_scope}|{chosen_anchor.date()}|{split_name}|{m}|as_of_negative", seed
            )
        )
        positives = positives.sort_values(["_holdout_hash", "machine_key"], kind="mergesort")
        negatives = negatives.sort_values(["_holdout_hash", "machine_key"], kind="mergesort")
        selected_positive_count = min(len(positives), len(negatives) // max_ratio)
        selected_pos = positives.head(selected_positive_count).copy()
        selected_neg = negatives.head(selected_positive_count * max_ratio).copy()
        selected_pos["holdout_positive_rank"] = np.arange(1, len(selected_pos) + 1)
        selected_pos["holdout_negative_rank"] = np.nan
        selected_neg["holdout_positive_rank"] = np.nan
        selected_neg["holdout_negative_rank"] = np.arange(1, len(selected_neg) + 1)

        frames: list[pd.DataFrame] = []
        for target_value, selected in [(1, selected_pos), (0, selected_neg)]:
            if selected.empty:
                continue
            out = pd.DataFrame({
                "row_role": "case" if target_value == 1 else "control",
                "target": target_value,
                "machine_key": selected["machine_key"].astype(str),
                "full_model": selected["full_model"],
                "serial": selected["serial"],
                "split": split_name,
                "window_name": window_name,
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "window_start": chosen_anchor - pd.Timedelta(days=observation_days),
                "window_end": chosen_anchor,
                "linked_case_window_start": pd.NaT,
                "linked_case_window_end": pd.NaT,
                "as_of_anchor_date": chosen_anchor,
                "as_of_prediction_horizon_days": lead_min,
                "as_of_actual_next_claim_date": selected["as_of_actual_next_claim_date"].to_numpy(),
                "as_of_days_to_next_claim": selected["as_of_days_to_next_claim"].to_numpy(),
                "future_claim_date": (
                    selected["as_of_actual_next_claim_date"].to_numpy()
                    if target_value == 1 else pd.NaT
                ),
                "days_from_window_end_to_claim": (
                    selected["as_of_days_to_next_claim"].to_numpy()
                    if target_value == 1 else np.nan
                ),
                "negative_sampling_type": "case" if target_value == 1 else "as_of_anchor",
                "control_sampling_reason": (
                    "as_of_anchor_claim_within_lead_min_days"
                    if target_value == 1
                    else "as_of_anchor_no_claim_within_lead_min_days"
                ),
                "control_no_claim_start": pd.NaT if target_value == 1 else chosen_anchor,
                "control_no_claim_end": (
                    pd.NaT if target_value == 1 else chosen_anchor + pd.Timedelta(days=lead_min)
                ),
                "holdout_sampling_design": "as_of_anchor_same_calendar_snapshot_ratio_sweep",
                "holdout_max_negative_to_positive_ratio": max_ratio,
                "holdout_random_state": seed,
                "holdout_positive_rank": selected["holdout_positive_rank"].to_numpy(),
                "holdout_negative_rank": selected["holdout_negative_rank"].to_numpy(),
            })
            out["case_machine_key"] = out["machine_key"]
            out["control_number_within_group"] = np.nan if target_value == 1 else 1
            out["case_control_group_id"] = out["machine_key"].map(
                lambda m: f"{window_name}__as_of_anchor__{split_name}__{'case' if target_value == 1 else 'control'}__{m}"
            )
            out["claim_episode_id"] = [
                episode_id_map.get((str(m), pd.Timestamp(d)), "")
                if target_value == 1 and pd.notna(d) else ""
                for m, d in zip(out["machine_key"], out["as_of_actual_next_claim_date"])
            ]
            frames.append(out)
        if frames:
            selected_parts.extend(frames)

        selected_pos_keys = set(selected_pos["machine_key"].astype(str))
        selected_neg_keys = set(selected_neg["machine_key"].astype(str))
        for _, row in candidates_df.iterrows():
            machine_key = str(row["machine_key"])
            audit_rows.append({
                "candidate_type": "machine_at_selected_anchor",
                "as_of_anchor_date": chosen_anchor,
                "split": split_name,
                "machine_key": machine_key,
                "full_model": row.get("full_model", ""),
                "design_target_within_lead_min": int(row["target"]),
                "actual_next_claim_date": row.get("as_of_actual_next_claim_date"),
                "days_to_next_claim": row.get("as_of_days_to_next_claim"),
                "selected_in_master_pool": machine_key in selected_pos_keys or machine_key in selected_neg_keys,
                "selection_reason": (
                    "selected_as_of_positive" if machine_key in selected_pos_keys
                    else "selected_as_of_negative" if machine_key in selected_neg_keys
                    else "not_needed_for_configured_max_ratio"
                ),
            })
        summary_rows.append({
            "split": split_name,
            "holdout_negative_sampling_mode": "as_of_anchor",
            "as_of_anchor_date": chosen_anchor,
            "as_of_prediction_horizon_days": lead_min,
            "as_of_observation_window_days": observation_days,
            "eligible_machines_at_anchor": int(len(candidates_df)),
            "positive_candidates": int(len(positives)),
            "negative_candidates": int(len(negatives)),
            "selected_positive_rows": int(len(selected_pos)),
            "selected_negative_rows_at_max_ratio": int(len(selected_neg)),
            "max_negative_to_positive_ratio": max_ratio,
            "requested_min_positive_machines": min_positive,
            "requested_minimum_met": bool(len(selected_pos) >= min_positive),
            "status": "selected",
        })

    master_pool = pd.concat(selected_parts, ignore_index=True, sort=False) if selected_parts else pd.DataFrame()
    if not master_pool.empty:
        master_pool = master_pool.sort_values(
            ["split", "target", "holdout_positive_rank", "holdout_negative_rank", "machine_key"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if master_pool["machine_key"].astype(str).duplicated().any():
            raise ValueError("As-of anchor holdout master pool contains duplicate machines.")
    return master_pool, pd.DataFrame(audit_rows), pd.DataFrame(summary_rows)


def _configured_multi_anchor_dates(config, split_name: str) -> list[pd.Timestamp]:
    """Return explicit configured multi-anchor dates for one split."""
    attr = (
        "MULTI_ANCHOR_FLEET_VALIDATION_DATES"
        if str(split_name) == "validation"
        else "MULTI_ANCHOR_FLEET_TEST_DATES"
    )
    raw = getattr(config, attr, [])
    if raw is None:
        return []
    if isinstance(raw, (str, pd.Timestamp, np.datetime64)):
        raw = [raw]
    dates = []
    for value in raw:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid date in {attr}: {value!r}")
        dates.append(pd.Timestamp(ts).normalize())
    return sorted(set(dates))


def _select_spaced_anchor_dates(
    candidates: Sequence[pd.Timestamp],
    count: int,
    min_gap_days: int,
) -> list[pd.Timestamp]:
    """Choose deterministic, approximately evenly spaced chronological dates."""
    dates = sorted({pd.Timestamp(x).normalize() for x in candidates})
    count = max(0, int(count))
    if count == 0 or not dates:
        return []
    if count == 1:
        return [dates[len(dates) // 2]]

    targets = np.linspace(0, len(dates) - 1, num=min(count, len(dates)))
    selected: list[pd.Timestamp] = []
    used: set[pd.Timestamp] = set()
    for target in targets:
        order = sorted(
            range(len(dates)),
            key=lambda i: (abs(float(i) - float(target)), dates[i]),
        )
        chosen = None
        for i in order:
            candidate = dates[i]
            if candidate in used:
                continue
            if all(abs((candidate - prior).days) >= int(min_gap_days) for prior in selected):
                chosen = candidate
                break
        if chosen is not None:
            selected.append(chosen)
            used.add(chosen)

    if len(selected) < count:
        for candidate in dates:
            if candidate in used:
                continue
            if all(abs((candidate - prior).days) >= int(min_gap_days) for prior in selected):
                selected.append(candidate)
                used.add(candidate)
                if len(selected) >= count:
                    break
    return sorted(selected)


def select_multi_anchor_fleet_dates(
    episodes: pd.DataFrame,
    prepared_master: pd.DataFrame,
    dates_by_machine: Mapping[str, np.ndarray],
    window_config: Mapping,
    config,
) -> tuple[dict[str, list[pd.Timestamp]], pd.DataFrame]:
    """Choose earlier validation anchors and later test anchors.

    Explicit configured dates take precedence. Otherwise, daily candidate dates
    are screened for complete future follow-up, source coverage, minimum fleet
    size, and minimum positive count. Validation dates are selected from the
    earlier feasible period and test dates from the later feasible period.
    """
    # Multi-anchor evaluation has its own horizon list. Anchor eligibility must
    # provide complete future follow-up through the largest configured value.
    configured_horizons = configured_multi_anchor_evaluation_horizons(config)
    required_followup_days = max(
        [int(window_config["lead_min_days"]), *[int(x) for x in configured_horizons]]
    )
    candidate_dates = _as_of_anchor_candidate_dates(
        episodes,
        window_config,
        config,
        followup_days_override=required_followup_days,
    )
    latest_complete_anchor = _max_claim_observation_date(config) - pd.Timedelta(
        days=required_followup_days
    )
    candidate_dates = [
        pd.Timestamp(x).normalize()
        for x in candidate_dates
        if pd.Timestamp(x).normalize() <= latest_complete_anchor
    ]
    if not candidate_dates:
        raise ValueError(
            "No eligible multi-anchor fleet dates are available with complete "
            f"follow-up for {required_followup_days} days."
        )

    min_positive = max(1, int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_POSITIVE_MACHINES", 5)))
    min_eligible = max(2, int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_ELIGIBLE_MACHINES", 50)))
    min_gap = max(0, int(getattr(config, "MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS", 60)))
    validation_count = max(1, int(getattr(config, "MULTI_ANCHOR_FLEET_VALIDATION_ANCHOR_COUNT", 3)))
    test_count = max(1, int(getattr(config, "MULTI_ANCHOR_FLEET_TEST_ANCHOR_COUNT", 2)))
    validation_fraction = float(getattr(config, "MULTI_ANCHOR_FLEET_VALIDATION_PERIOD_FRACTION", 0.70))
    if not 0 < validation_fraction < 1:
        raise ValueError("MULTI_ANCHOR_FLEET_VALIDATION_PERIOD_FRACTION must be between 0 and 1.")
    test_gap = max(0, int(getattr(config, "MULTI_ANCHOR_FLEET_TEST_START_GAP_DAYS", 30)))

    audit_rows: list[dict] = []
    feasible: dict[str, list[pd.Timestamp]] = {"validation": [], "test": []}
    count_lookup: dict[tuple[str, pd.Timestamp], tuple[int, int, int]] = {}
    for anchor in candidate_dates:
        anchor = pd.Timestamp(anchor).normalize()
        for split_name in ["validation", "test"]:
            frame = _as_of_anchor_machine_candidates(
                anchor, prepared_master, dates_by_machine, window_config, config, split_name
            )
            eligible_count = int(len(frame))
            positive_count = int(frame["target"].eq(1).sum()) if not frame.empty else 0
            negative_count = int(frame["target"].eq(0).sum()) if not frame.empty else 0
            is_feasible = (
                eligible_count >= min_eligible
                and positive_count >= min_positive
                and negative_count >= 1
            )
            count_lookup[(split_name, anchor)] = (
                eligible_count, positive_count, negative_count
            )
            if is_feasible:
                feasible[split_name].append(anchor)
            audit_rows.append({
                "candidate_type": "multi_anchor_date_screen",
                "split": split_name,
                "anchor_date": anchor,
                "eligible_machines": eligible_count,
                "positive_machines_within_design_horizon": positive_count,
                "negative_machines_within_design_horizon": negative_count,
                "minimum_eligible_required": min_eligible,
                "minimum_positive_required": min_positive,
                "is_feasible": bool(is_feasible),
            })

    explicit_validation = _configured_multi_anchor_dates(config, "validation")
    explicit_test = _configured_multi_anchor_dates(config, "test")
    all_dates = sorted(set(feasible["validation"]) | set(feasible["test"]))
    if not all_dates:
        raise ValueError(
            "No date satisfies the multi-anchor minimum eligible-machine and positive-machine requirements."
        )
    boundary_index = min(
        len(all_dates) - 1,
        max(0, int(np.floor((len(all_dates) - 1) * validation_fraction))),
    )
    boundary_date = all_dates[boundary_index]

    if explicit_validation:
        selected_validation = explicit_validation
    else:
        validation_pool = [d for d in feasible["validation"] if d <= boundary_date]
        if not validation_pool:
            validation_pool = feasible["validation"]
        selected_validation = _select_spaced_anchor_dates(
            validation_pool, validation_count, min_gap
        )

    latest_validation = max(selected_validation) if selected_validation else boundary_date
    earliest_test = latest_validation + pd.Timedelta(days=test_gap)
    if explicit_test:
        selected_test = explicit_test
    else:
        test_pool = [d for d in feasible["test"] if d >= earliest_test and d > boundary_date]
        if not test_pool:
            test_pool = [d for d in feasible["test"] if d > latest_validation]
        if not test_pool:
            test_pool = feasible["test"]
        selected_test = _select_spaced_anchor_dates(test_pool, test_count, min_gap)

    selected = {"validation": selected_validation, "test": selected_test}
    for split_name, dates in selected.items():
        if not dates:
            raise ValueError(f"No {split_name} multi-anchor dates could be selected.")
        feasible_set = set(feasible[split_name])
        for order, anchor in enumerate(dates, start=1):
            if pd.Timestamp(anchor).normalize() > latest_complete_anchor:
                raise ValueError(
                    f"Configured {split_name} anchor {pd.Timestamp(anchor).date()} lacks "
                    f"complete {required_followup_days}-day future follow-up; latest "
                    f"eligible anchor is {latest_complete_anchor.date()}."
                )
            if anchor not in feasible_set:
                counts = count_lookup.get((split_name, anchor))
                if counts is None:
                    frame = _as_of_anchor_machine_candidates(
                        anchor, prepared_master, dates_by_machine, window_config, config, split_name
                    )
                    counts = (
                        int(len(frame)),
                        int(frame["target"].eq(1).sum()) if not frame.empty else 0,
                        int(frame["target"].eq(0).sum()) if not frame.empty else 0,
                    )
                if counts[0] < min_eligible or counts[1] < min_positive or counts[2] < 1:
                    raise ValueError(
                        f"Configured {split_name} anchor {anchor.date()} is not feasible: "
                        f"eligible={counts[0]}, positives={counts[1]}, negatives={counts[2]}."
                    )
            audit_rows.append({
                "candidate_type": "multi_anchor_selected_date",
                "split": split_name,
                "anchor_date": anchor,
                "anchor_order": order,
                "selection_source": "explicit" if (
                    (split_name == "validation" and explicit_validation)
                    or (split_name == "test" and explicit_test)
                ) else "automatic",
                "chronological_boundary_date": boundary_date,
                "is_feasible": True,
            })
    return selected, pd.DataFrame(audit_rows)


def build_multi_anchor_fleet_base_rows(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build natural-prevalence fleet snapshots at several locked anchor dates.

    Machines remain disjoint across train/validation/test through the fixed
    assignment table, but a validation or test machine may appear once at each
    anchor in its own split. Rankings must therefore be calculated independently
    within each anchor date rather than across the pooled rows.
    """
    if split_assignments is None:
        raise ValueError("split_assignments are required for multi-anchor fleet construction.")
    history = claim_history_episodes if claim_history_episodes is not None else episodes
    dates_by_machine = claim_dates_by_machine(history)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = lead_max - lead_min
    window_name = window_config_name(window_config)
    seed = int(getattr(config, "MULTI_ANCHOR_FLEET_RANDOM_STATE", 20260728))

    master = machine_master.copy()
    master["machine_key"] = master["machine_key"].astype(str)
    assignment_cols = [
        c for c in ["machine_key", "split", "full_model", "serial"]
        if c in split_assignments.columns
    ]
    assignments = split_assignments[assignment_cols].drop_duplicates("machine_key").copy()
    master = master.merge(assignments, on="machine_key", how="left", suffixes=("", "_assigned"))
    for field in ["full_model", "serial"]:
        assigned = f"{field}_assigned"
        if assigned in master.columns:
            current = master.get(field, pd.Series("", index=master.index)).astype("string").fillna("").str.strip()
            master[field] = master.get(field, pd.Series("", index=master.index)).where(current.ne(""), master[assigned])
            master = master.drop(columns=[assigned])
    master["full_model"] = master.get("full_model", pd.Series("", index=master.index)).map(clean_model)
    master["serial"] = master.get("serial", pd.Series("", index=master.index)).map(clean_serial)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master = master[master["split"].astype(str).isin(["validation", "test"])].copy()

    selected_dates, date_audit = select_multi_anchor_fleet_dates(
        history, master, dates_by_machine, window_config, config
    )

    episode_lookup = history.copy()
    episode_lookup["machine_key"] = episode_lookup["machine_key"].astype(str)
    episode_lookup["claim_date"] = pd.to_datetime(episode_lookup["claim_date"], errors="coerce")
    if "claim_episode_id" not in episode_lookup.columns:
        episode_lookup["claim_episode_id"] = ""
    episode_lookup = episode_lookup.sort_values(
        ["machine_key", "claim_date", "claim_episode_id"], kind="mergesort"
    ).drop_duplicates(["machine_key", "claim_date"], keep="first")
    episode_id_map = {
        (str(row.machine_key), pd.Timestamp(row.claim_date)): str(row.claim_episode_id)
        for row in episode_lookup.itertuples(index=False)
        if pd.notna(row.claim_date)
    }

    max_rows_raw = getattr(config, "MULTI_ANCHOR_FLEET_MAX_MACHINES_PER_ANCHOR", None)
    max_rows = None if max_rows_raw in (None, "", 0, "0") else max(1, int(max_rows_raw))
    base_parts: list[pd.DataFrame] = []
    audit_rows: list[dict] = date_audit.to_dict("records")
    summary_rows: list[dict] = []
    anchor_index_rows: list[dict] = []

    for split_name in ["validation", "test"]:
        for anchor_order, anchor in enumerate(selected_dates[split_name], start=1):
            candidates = _as_of_anchor_machine_candidates(
                anchor, master, dates_by_machine, window_config, config, split_name
            ).copy()
            candidates["machine_key"] = candidates["machine_key"].astype(str)
            candidates["_fleet_hash"] = candidates["machine_key"].map(
                lambda m: _stable_hash_int(
                    f"{window_name}|{split_name}|{pd.Timestamp(anchor).date()}|{m}|multi_anchor_fleet",
                    seed,
                )
            )
            candidates = candidates.sort_values(["target", "_fleet_hash", "machine_key"], ascending=[False, True, True], kind="mergesort")
            uncapped_rows = int(len(candidates))
            uncapped_positive = int(candidates["target"].eq(1).sum())
            if max_rows is not None and len(candidates) > max_rows:
                positives = candidates[candidates["target"].eq(1)].copy()
                negatives = candidates[candidates["target"].eq(0)].copy()
                negative_slots = max(0, max_rows - len(positives))
                candidates = pd.concat(
                    [positives, negatives.head(negative_slots)], ignore_index=True, sort=False
                )
            candidates = candidates.sort_values(["machine_key"], kind="mergesort").reset_index(drop=True)
            anchor_id = f"{split_name}__{pd.Timestamp(anchor).strftime('%Y%m%d')}"
            out = pd.DataFrame({
                "snapshot_id": [f"{anchor_id}__{m}" for m in candidates["machine_key"]],
                "fleet_anchor_id": anchor_id,
                "fleet_anchor_order": anchor_order,
                "row_role": np.where(candidates["target"].eq(1), "case", "control"),
                "target": candidates["target"].astype(int).to_numpy(),
                "machine_key": candidates["machine_key"].to_numpy(),
                "full_model": candidates["full_model"].to_numpy(),
                "serial": candidates["serial"].to_numpy(),
                "split": split_name,
                "window_name": window_name,
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "window_start": pd.Timestamp(anchor) - pd.Timedelta(days=observation_days),
                "window_end": pd.Timestamp(anchor),
                "as_of_anchor_date": pd.Timestamp(anchor),
                "as_of_prediction_horizon_days": lead_min,
                "as_of_actual_next_claim_date": candidates["as_of_actual_next_claim_date"].to_numpy(),
                "as_of_days_to_next_claim": candidates["as_of_days_to_next_claim"].to_numpy(),
                "future_claim_date": np.where(
                    candidates["target"].eq(1),
                    candidates["as_of_actual_next_claim_date"].to_numpy(),
                    np.datetime64("NaT"),
                ),
                "days_from_window_end_to_claim": np.where(
                    candidates["target"].eq(1),
                    candidates["as_of_days_to_next_claim"].to_numpy(),
                    np.nan,
                ),
                "negative_sampling_type": np.where(
                    candidates["target"].eq(1), "case", "natural_fleet_negative"
                ),
                "control_sampling_reason": np.where(
                    candidates["target"].eq(1),
                    "claim_within_design_horizon_at_shared_anchor",
                    "no_claim_within_design_horizon_at_shared_anchor",
                ),
                "holdout_sampling_design": "multi_anchor_natural_prevalence_fleet_snapshot",
                "evaluation_population": "all_eligible_machines_at_anchor_natural_prevalence",
                "holdout_random_state": seed,
            })
            out["case_machine_key"] = out["machine_key"]
            out["control_number_within_group"] = np.nan
            out["case_control_group_id"] = out["snapshot_id"]
            out["claim_episode_id"] = [
                episode_id_map.get((str(m), pd.Timestamp(d)), "") if pd.notna(d) else ""
                for m, d in zip(out["machine_key"], out["as_of_actual_next_claim_date"])
            ]
            base_parts.append(out)

            selected_keys = set(out["machine_key"].astype(str))
            for row in candidates.itertuples(index=False):
                audit_rows.append({
                    "candidate_type": "multi_anchor_machine_snapshot",
                    "split": split_name,
                    "anchor_date": pd.Timestamp(anchor),
                    "fleet_anchor_id": anchor_id,
                    "machine_key": str(row.machine_key),
                    "full_model": getattr(row, "full_model", ""),
                    "design_target_within_lead_min": int(row.target),
                    "actual_next_claim_date": getattr(row, "as_of_actual_next_claim_date", pd.NaT),
                    "days_to_next_claim": getattr(row, "as_of_days_to_next_claim", np.nan),
                    "selected_in_snapshot": str(row.machine_key) in selected_keys,
                })
            positive_rows = int(out["target"].eq(1).sum())
            negative_rows = int(out["target"].eq(0).sum())
            summary = {
                "split": split_name,
                "fleet_anchor_id": anchor_id,
                "anchor_order": anchor_order,
                "anchor_date": pd.Timestamp(anchor),
                "window_start": pd.Timestamp(anchor) - pd.Timedelta(days=observation_days),
                "window_end": pd.Timestamp(anchor),
                "design_horizon_days": lead_min,
                "uncapped_eligible_machines": uncapped_rows,
                "uncapped_positive_machines": uncapped_positive,
                "selected_rows": int(len(out)),
                "positive_rows": positive_rows,
                "negative_rows": negative_rows,
                "natural_positive_rate": float(positive_rows / len(out)) if len(out) else np.nan,
                "development_cap": max_rows,
            }
            summary_rows.append(summary)
            anchor_index_rows.append(summary.copy())

    base = pd.concat(base_parts, ignore_index=True, sort=False) if base_parts else pd.DataFrame()
    if base.empty:
        raise ValueError("Multi-anchor fleet construction produced no rows.")
    if base["snapshot_id"].astype(str).duplicated().any():
        raise ValueError("Multi-anchor fleet snapshots contain duplicate snapshot IDs.")
    split_counts = base.groupby("machine_key", dropna=False)["split"].nunique(dropna=False)
    if (split_counts > 1).any():
        raise ValueError("Machine leakage detected across multi-anchor validation and test splits.")
    base = base.sort_values(["split", "as_of_anchor_date", "machine_key"], kind="mergesort").reset_index(drop=True)
    return (
        base,
        pd.DataFrame(audit_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(anchor_index_rows),
    )

def build_holdout_ratio_base_pool(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
    split_assignments: Optional[pd.DataFrame] = None,
    included_splits: Sequence[str] = ("validation", "test"),
    random_state_override: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch to the configured validation/test negative-case design."""
    mode = holdout_negative_sampling_mode(config)
    builders = {
        "random": build_random_holdout_ratio_base_pool,
        "controlled": build_controlled_holdout_ratio_base_pool,
        "as_of_anchor": build_as_of_anchor_holdout_ratio_base_pool,
    }
    builder = builders[mode]
    return builder(
        episodes=episodes,
        machine_master=machine_master,
        sources=sources,
        window_config=window_config,
        config=config,
        claim_history_episodes=claim_history_episodes,
        split_assignments=split_assignments,
        included_splits=included_splits,
        random_state_override=random_state_override,
    )


def holdout_ratio_subset(
    master_pool: pd.DataFrame,
    split_name: str,
    negative_to_positive_ratio: int,
) -> pd.DataFrame:
    """Return a nested holdout subset for either random or controlled design."""
    ratio = int(negative_to_positive_ratio)
    if ratio < 1:
        raise ValueError("negative_to_positive_ratio must be >= 1.")
    sub = master_pool[master_pool["split"].astype(str).eq(str(split_name))].copy()
    positives = sub[pd.to_numeric(sub["target"], errors="coerce").eq(1)].copy()
    negatives = sub[pd.to_numeric(sub["target"], errors="coerce").eq(0)].copy()
    positives = positives.sort_values(
        ["holdout_positive_rank", "machine_key"], kind="mergesort"
    )
    if len(positives) == 0:
        raise ValueError(f"Holdout split {split_name!r} has no positive rows.")

    designs = set(sub.get("holdout_sampling_design", pd.Series(dtype=str)).dropna().astype(str))
    controlled = any(value.startswith("controlled_") for value in designs)
    if controlled:
        if "holdout_control_rank_within_positive" not in negatives.columns:
            raise ValueError(
                "Controlled holdout pool is missing holdout_control_rank_within_positive."
            )
        negatives = negatives[
            pd.to_numeric(
                negatives["holdout_control_rank_within_positive"], errors="coerce"
            ).le(ratio)
        ].copy()
        negatives = negatives.sort_values(
            ["matched_holdout_positive_rank", "holdout_control_rank_within_positive", "machine_key"],
            kind="mergesort",
        )
        nested_rule = "same positives; first N same-window same-model controls per positive"
    else:
        negatives = negatives.sort_values(
            ["holdout_negative_rank", "machine_key"], kind="mergesort"
        )
        required_negatives = int(len(positives) * ratio)
        if len(negatives) < required_negatives:
            raise ValueError(
                f"Holdout split {split_name!r} has {len(negatives)} negatives, "
                f"but ratio {ratio}:1 requires {required_negatives}."
            )
        negatives = negatives.head(required_negatives).copy()
        nested_rule = "same positives; negatives with global rank <= ratio * positive_count"

    required_negatives = int(len(positives) * ratio)
    if len(negatives) != required_negatives:
        raise ValueError(
            f"Holdout split {split_name!r} produced {len(negatives)} negatives for "
            f"{len(positives)} positives at ratio {ratio}:1."
        )
    out = pd.concat([positives, negatives], ignore_index=True, sort=False)
    out["holdout_negative_to_positive_ratio_requested"] = ratio
    out["holdout_positive_rows_in_ratio_dataset"] = int(len(positives))
    out["holdout_negative_rows_in_ratio_dataset"] = required_negatives
    out["holdout_nested_negative_rule"] = nested_rule
    out = out.sort_values(
        ["target", "machine_key"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if out["machine_key"].astype(str).duplicated().any():
        raise ValueError("A machine appears more than once in a holdout ratio dataset.")
    return out


def random_holdout_ratio_subset(
    master_pool: pd.DataFrame,
    split_name: str,
    negative_to_positive_ratio: int,
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`holdout_ratio_subset`."""
    return holdout_ratio_subset(
        master_pool=master_pool,
        split_name=split_name,
        negative_to_positive_ratio=negative_to_positive_ratio,
    )


def build_horizon_specific_random_ratio_subsets(
    evaluation_df: pd.DataFrame,
    split_name: str,
    horizon_days: int,
    ratios: Sequence[int],
    config,
    random_state: Optional[int] = None,
) -> Tuple[Dict[int, pd.DataFrame], dict]:
    """Build exact, nested random ratio cohorts for one future-claim horizon.

    The candidate rows are fixed machine snapshots with outcome-independent
    feature windows. Truth is derived for ``horizon_days`` first. Positives and
    negatives are then independently ranked by a stable hash. The same positive
    machines are retained for every requested ratio, while larger ratios only
    append lower-ranked negatives. This makes the requested prevalence exact at
    each horizon without changing rows within that horizon across experiment runs.
    """
    cleaned_ratios = sorted({int(x) for x in ratios})
    if not cleaned_ratios or any(x < 1 for x in cleaned_ratios):
        raise ValueError("ratios must contain one or more integers >= 1.")
    horizon = int(horizon_days)
    if horizon < 1:
        raise ValueError("horizon_days must be >= 1.")
    split = str(split_name)
    sub = evaluation_df[evaluation_df["split"].astype(str).eq(split)].copy()
    if sub.empty:
        raise ValueError(f"No candidate rows are available for split {split!r}.")
    if "machine_key" not in sub.columns:
        raise ValueError("Horizon ratio candidates must contain machine_key.")
    if sub["machine_key"].astype(str).duplicated().any():
        raise ValueError(
            "Horizon-specific ratio sampling requires at most one row per machine "
            f"within split {split!r}."
        )

    y_eval, target_col, target_mode, resolved_horizon = get_evaluation_target(
        sub, config, horizon_days=horizon
    )
    if target_mode != "claim_within_horizon":
        raise ValueError(
            "Horizon-specific ratio sampling requires "
            "EVALUATION_TARGET_MODE='claim_within_horizon'."
        )
    sub["target"] = y_eval.to_numpy(dtype=int)
    sub["horizon_ratio_target_col"] = target_col
    sub["horizon_ratio_evaluation_horizon_days"] = int(resolved_horizon)

    seed = int(
        getattr(config, "HOLDOUT_RANDOM_STATE", 42)
        if random_state is None
        else random_state
    )
    scope = f"horizon_specific_random|{split}|{horizon}"
    sub["_horizon_ratio_hash"] = sub["machine_key"].astype(str).map(
        lambda machine: _stable_hash_int(f"{scope}|{machine}", seed)
    )
    positives = sub[sub["target"].eq(1)].sort_values(
        ["_horizon_ratio_hash", "machine_key"], kind="mergesort"
    ).copy()
    negatives = sub[sub["target"].eq(0)].sort_values(
        ["_horizon_ratio_hash", "machine_key"], kind="mergesort"
    ).copy()
    max_ratio = int(max(cleaned_ratios))
    selected_positive_count = min(len(positives), len(negatives) // max_ratio)
    if selected_positive_count < 1:
        raise ValueError(
            f"Split {split!r}, horizon {horizon}d has {len(positives)} positives "
            f"and {len(negatives)} negatives; ratio {max_ratio}:1 is not feasible."
        )

    selected_pos = positives.head(selected_positive_count).copy()
    selected_neg = negatives.head(selected_positive_count * max_ratio).copy()
    selected_pos["horizon_ratio_positive_rank"] = np.arange(1, len(selected_pos) + 1)
    selected_pos["horizon_ratio_negative_rank"] = np.nan
    selected_neg["horizon_ratio_positive_rank"] = np.nan
    selected_neg["horizon_ratio_negative_rank"] = np.arange(1, len(selected_neg) + 1)
    for frame in [selected_pos, selected_neg]:
        frame["horizon_ratio_sampling_design"] = (
            "fixed_random_machine_level_exact_ratio_per_horizon"
        )
        frame["horizon_ratio_random_state"] = seed
        frame["horizon_ratio_max_negative_to_positive_ratio"] = max_ratio

    subsets: Dict[int, pd.DataFrame] = {}
    selected_positive_keys = set(selected_pos["machine_key"].astype(str))
    for ratio in cleaned_ratios:
        ratio_neg = selected_neg[
            pd.to_numeric(
                selected_neg["horizon_ratio_negative_rank"], errors="coerce"
            ).le(selected_positive_count * int(ratio))
        ].copy()
        out = pd.concat([selected_pos, ratio_neg], ignore_index=True, sort=False)
        out["holdout_negative_to_positive_ratio_requested"] = int(ratio)
        out["holdout_positive_rows_in_ratio_dataset"] = selected_positive_count
        out["holdout_negative_rows_in_ratio_dataset"] = selected_positive_count * int(ratio)
        out["holdout_nested_negative_rule"] = (
            "same horizon-specific positives; negatives with rank <= ratio * positive_count"
        )
        out = out.drop(columns=["_horizon_ratio_hash"], errors="ignore")
        out = out.sort_values(
            ["target", "machine_key"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        if out["machine_key"].astype(str).duplicated().any():
            raise ValueError("A machine appears more than once in a horizon ratio cohort.")
        actual_pos = int(out["target"].eq(1).sum())
        actual_neg = int(out["target"].eq(0).sum())
        if actual_neg != actual_pos * int(ratio):
            raise ValueError(
                f"Horizon {horizon}d ratio {ratio}:1 produced "
                f"{actual_pos} positives and {actual_neg} negatives."
            )
        if set(out.loc[out["target"].eq(1), "machine_key"].astype(str)) != selected_positive_keys:
            raise ValueError("Positive machines changed across nested horizon ratio cohorts.")
        subsets[int(ratio)] = out

    summary = {
        "split": split,
        "evaluation_horizon_days": horizon,
        "evaluation_target_col": target_col,
        "candidate_rows": int(len(sub)),
        "candidate_positive_rows": int(len(positives)),
        "candidate_negative_rows": int(len(negatives)),
        "selected_positive_rows": int(selected_positive_count),
        "selected_negative_rows_at_max_ratio": int(selected_positive_count * max_ratio),
        "max_negative_to_positive_ratio": max_ratio,
        "random_state": seed,
        "sampling_design": "fixed_random_machine_level_exact_ratio_per_horizon",
    }
    return subsets, summary


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


def top_n_metrics(y_true, score, top_n_counts: Sequence[int]) -> pd.DataFrame:
    """Calculate ranking metrics at fixed absolute machine counts.

    Unlike percentage-based Top-K, the selected workload is identical across
    prevalence-ratio cohorts. Counts larger than the cohort are capped at the
    cohort size and reported through ``flagged_count``.
    """
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    s = pd.Series(score).astype(float).reset_index(drop=True)
    order = s.sort_values(ascending=False).index.to_numpy()
    total_pos = int((y == 1).sum())
    n = len(y)
    base_rate = total_pos / n if n else np.nan
    rows = []
    for requested in top_n_counts:
        top_n = int(requested)
        if top_n < 1:
            raise ValueError(f"Top-N counts must be >= 1; received {requested!r}.")
        k = min(top_n, n) if n else 0
        top_idx = order[:k]
        tp = int(y.iloc[top_idx].sum()) if k else 0
        precision = tp / k if k else np.nan
        recall = tp / total_pos if total_pos else np.nan
        rows.append({
            "top_n_requested": top_n,
            "rows": int(n),
            "flagged_count": int(k),
            "positive_count": total_pos,
            "true_positive_at_n": tp,
            "precision_at_n": float(precision),
            "recall_at_n": float(recall),
            "lift_vs_random": (
                float(precision / base_rate)
                if base_rate and base_rate > 0
                else np.nan
            ),
            "min_score_in_top_n": float(s.iloc[top_idx].min()) if k else np.nan,
        })
    return pd.DataFrame(rows)


def _percentile_interval(values: Sequence[float], confidence_level: float) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if len(arr) == 0:
        return {
            "bootstrap_valid_resamples": 0,
            "bootstrap_mean": np.nan,
            "bootstrap_standard_error": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    alpha = 1.0 - float(confidence_level)
    return {
        "bootstrap_valid_resamples": int(len(arr)),
        "bootstrap_mean": float(np.mean(arr)),
        "bootstrap_standard_error": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci_lower": float(np.quantile(arr, alpha / 2.0)),
        "ci_upper": float(np.quantile(arr, 1.0 - alpha / 2.0)),
    }


def bootstrap_ranking_metric_intervals(
    y_true,
    score,
    top_k_rates: Sequence[float],
    top_n_counts: Sequence[int],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Estimate machine-level percentile bootstrap confidence intervals.

    Positive and negative machines are resampled separately with replacement.
    This stratified nonparametric bootstrap preserves the evaluated prevalence
    and negative:positive ratio in every replicate. It is appropriate here
    because Step 02 guarantees one row per holdout machine.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = pd.Series(y_true).astype(int).reset_index(drop=True).to_numpy()
    s = np.asarray(score, dtype=float)
    if len(y) != len(s):
        raise ValueError("y_true and score must have the same length.")
    if len(y) == 0:
        raise ValueError("Cannot bootstrap an empty evaluation cohort.")
    requested = int(n_resamples)
    if requested < 1:
        raise ValueError("n_resamples must be at least 1.")
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            "Bootstrap confidence intervals require both positive and negative machines."
        )

    rates = [float(x) for x in top_k_rates]
    counts = [int(x) for x in top_n_counts]
    rng = np.random.default_rng(int(random_state))
    ap_values: List[float] = []
    auc_values: List[float] = []
    topk_values = {
        rate: {"precision_at_k": [], "recall_at_k": [], "lift_vs_random": []}
        for rate in rates
    }
    topn_values = {
        count: {"precision_at_n": [], "recall_at_n": [], "lift_vs_random": []}
        for count in counts
    }

    for _ in range(requested):
        sampled_idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        rng.shuffle(sampled_idx)
        y_boot = y[sampled_idx]
        s_boot = s[sampled_idx]
        ap_values.append(float(average_precision_score(y_boot, s_boot)))
        auc_values.append(float(roc_auc_score(y_boot, s_boot)))

        # NumPy ranking calculations avoid constructing two pandas DataFrames
        # for every bootstrap replicate. This is materially faster for the
        # 1,000-resample ratio and multi-window experiments while preserving the
        # same definitions used by top_k_metrics and top_n_metrics.
        order = np.argsort(-s_boot, kind="mergesort")
        y_ranked = y_boot[order]
        cumulative_positive = np.cumsum(y_ranked)
        total_positive = int(cumulative_positive[-1]) if len(cumulative_positive) else 0
        n_rows = int(len(y_boot))
        base_rate = total_positive / n_rows if n_rows else np.nan

        for rate in rates:
            k = max(1, int(np.ceil(float(rate) * n_rows))) if n_rows else 0
            tp = int(cumulative_positive[k - 1]) if k else 0
            precision = tp / k if k else np.nan
            recall = tp / total_positive if total_positive else np.nan
            lift = (
                precision / base_rate
                if np.isfinite(base_rate) and base_rate > 0
                else np.nan
            )
            bucket = topk_values[rate]
            bucket["precision_at_k"].append(float(precision))
            bucket["recall_at_k"].append(float(recall))
            bucket["lift_vs_random"].append(float(lift))

        for count in counts:
            k = min(int(count), n_rows) if n_rows else 0
            tp = int(cumulative_positive[k - 1]) if k else 0
            precision = tp / k if k else np.nan
            recall = tp / total_positive if total_positive else np.nan
            lift = (
                precision / base_rate
                if np.isfinite(base_rate) and base_rate > 0
                else np.nan
            )
            bucket = topn_values[count]
            bucket["precision_at_n"].append(float(precision))
            bucket["recall_at_n"].append(float(recall))
            bucket["lift_vs_random"].append(float(lift))

    ap_interval = _percentile_interval(ap_values, confidence_level)
    auc_interval = _percentile_interval(auc_values, confidence_level)
    threshold_free = {
        "bootstrap_method": "stratified_machine_level_percentile",
        "bootstrap_n_resamples_requested": requested,
        "bootstrap_confidence_level": float(confidence_level),
        "bootstrap_random_state": int(random_state),
        "average_precision_bootstrap_valid_resamples": ap_interval["bootstrap_valid_resamples"],
        "average_precision_bootstrap_mean": ap_interval["bootstrap_mean"],
        "average_precision_bootstrap_standard_error": ap_interval["bootstrap_standard_error"],
        "average_precision_ci_lower": ap_interval["ci_lower"],
        "average_precision_ci_upper": ap_interval["ci_upper"],
        "roc_auc_bootstrap_valid_resamples": auc_interval["bootstrap_valid_resamples"],
        "roc_auc_bootstrap_mean": auc_interval["bootstrap_mean"],
        "roc_auc_bootstrap_standard_error": auc_interval["bootstrap_standard_error"],
        "roc_auc_ci_lower": auc_interval["ci_lower"],
        "roc_auc_ci_upper": auc_interval["ci_upper"],
    }

    topk_rows: List[dict] = []
    for rate in rates:
        row = {
            "top_k_rate": rate,
            "bootstrap_method": "stratified_machine_level_percentile",
            "bootstrap_n_resamples_requested": requested,
            "bootstrap_confidence_level": float(confidence_level),
            "bootstrap_random_state": int(random_state),
        }
        for metric, values in topk_values[rate].items():
            interval = _percentile_interval(values, confidence_level)
            row[f"{metric}_bootstrap_valid_resamples"] = interval["bootstrap_valid_resamples"]
            row[f"{metric}_bootstrap_mean"] = interval["bootstrap_mean"]
            row[f"{metric}_bootstrap_standard_error"] = interval["bootstrap_standard_error"]
            row[f"{metric}_ci_lower"] = interval["ci_lower"]
            row[f"{metric}_ci_upper"] = interval["ci_upper"]
        topk_rows.append(row)

    topn_rows: List[dict] = []
    for count in counts:
        row = {
            "top_n_requested": count,
            "bootstrap_method": "stratified_machine_level_percentile",
            "bootstrap_n_resamples_requested": requested,
            "bootstrap_confidence_level": float(confidence_level),
            "bootstrap_random_state": int(random_state),
        }
        for metric, values in topn_values[count].items():
            interval = _percentile_interval(values, confidence_level)
            row[f"{metric}_bootstrap_valid_resamples"] = interval["bootstrap_valid_resamples"]
            row[f"{metric}_bootstrap_mean"] = interval["bootstrap_mean"]
            row[f"{metric}_bootstrap_standard_error"] = interval["bootstrap_standard_error"]
            row[f"{metric}_ci_lower"] = interval["ci_lower"]
            row[f"{metric}_ci_upper"] = interval["ci_upper"]
        topn_rows.append(row)

    return threshold_free, pd.DataFrame(topk_rows), pd.DataFrame(topn_rows)


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
