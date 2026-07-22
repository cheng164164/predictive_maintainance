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
        "machine_id",
        "claim_number",
        "local_date",
        "claim_type_description",
        "warranty_claim_data_source",
        "full_model",
        "serial",
        "failure_smr",
        "critical_fail_part_number",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "full_model", "serial", "machine_id")
    df = _filter_event_dates(df, "local_date", config.MIN_CLAIM_DATE, config.MAX_CLAIM_DATE)
    df["claim_date"] = df["local_date"]
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
        "serial_number",
        "full_model",
        "machine_id",
        "event_date",
        "fault_code",
        "event_error_name_en",
        "event_action_level",
        "occurrence_count",
        "log_occurrence_count",
        "smr_hours",
        "applicable_component",
        "related_component",
        "is_mechanical_failure_code",
        "is_electrical_failure_code",
        "action_level_num",
        "failure_code_evidence_score",
        "failure_code_evidence_strength_class",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "full_model", "serial_number", "machine_id")
    df = _filter_event_dates(df, "event_date", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    numeric_cols = [
        "occurrence_count",
        "log_occurrence_count",
        "smr_hours",
        "is_mechanical_failure_code",
        "is_electrical_failure_code",
        "action_level_num",
        "failure_code_evidence_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_fluid_samples(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.FLUID_SAMPLES_FILE_CANDIDATES)
    cols = [
        "FULL_MODEL",
        "SERIAL",
        "machine_id",
        "TELEMETRY_SMR_NUMERIC",
        "LAB_NAME",
        "LABS_SAMPLE_NUMBER",
        "sample_drawn_date",
        "sample_result_severity_order",
        "Ag_Silver_PPM",
        "Cu_Copper_PPM",
        "Fe_Iron_PPM",
        "K_Potassium_PPM",
        "Ni_Nickel_PPM",
        "Pb_Lead_PPM",
        "Sn_Tin_PPM",
        "Soot_Soot_Abs_cm",
        "Soot_Soot_PERCENT",
        "Water_Water_PERCENT",
    ]
    df = read_csv_selected(path, cols)
    df = add_machine_key(df, "FULL_MODEL", "SERIAL", "machine_id")
    df = _filter_event_dates(df, "sample_drawn_date", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    for col in [c for c in cols if c not in {"FULL_MODEL", "SERIAL", "machine_id", "LAB_NAME", "LABS_SAMPLE_NUMBER", "sample_drawn_date"}]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_maintenance(config) -> pd.DataFrame:
    path = resolve_source_file(config.SOURCE_DIR, config.MAINTENANCE_FILE_CANDIDATES)
    cols = [
        "full_model",
        "machine_id",
        "SERIAL",
        "EVENT_NAME_EN",
        "event_date",
        "smr_hours",
        "remaining_hours",
        "INTERVAL_HOURS",
        "is_monitor_reset",
        "is_overdue",
        "is_due_now",
        "is_notice_or_status",
        "AVAILABLE",
        "related_component",
        "related_component_1",
        "related_component_2",
        "maintenance_type",
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
        "machine_id",
        "LOCAL_DATE",
        "full_model",
        "SERIAL",
        "smr_hours",
        "smr_delta_clean_since_prev_obs_hours",
        "actual_working_hours_clean",
        "working_hours_clean",
        "engine_running_hours_clean",
        "engine_idling_hours_clean",
        "engine_idle_share_daily",
        "throttle_full_share_clean",
        "high_throttle_day_flag",
        "long_engine_day_flag",
        "traveling_hours_clean",
        "moving_back_forth_hours_clean",
        "auto_quick_shift_hours_clean",
        "manual_variable_shift_hours_clean",
        "actual_work_day_flag",
        "engine_running_day_flag",
        "travel_day_flag",
    ]
    df = read_csv_selected(path, cols)
    # Drop blank exported rows.
    df = df.dropna(subset=["machine_id", "LOCAL_DATE"])
    df = add_machine_key(df, "full_model", "SERIAL", "machine_id")
    df = _filter_event_dates(df, "LOCAL_DATE", config.MIN_VALID_EVENT_DATE, config.MAX_VALID_EVENT_DATE)
    for col in [c for c in cols if c not in {"machine_id", "LOCAL_DATE", "full_model", "SERIAL"}]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_sources(config, include_operation: bool = True) -> Dict[str, pd.DataFrame]:
    sources = {
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
        if df.empty or "machine_key" not in df.columns:
            continue
        model_col = "full_model_norm" if "full_model_norm" in df.columns else None
        serial_col = "serial_norm" if "serial_norm" in df.columns else None
        cols = ["machine_key"]
        if model_col:
            cols.append(model_col)
        if serial_col:
            cols.append(serial_col)
        tmp = df[cols].drop_duplicates("machine_key").copy()
        tmp = tmp.rename(columns={model_col: "full_model", serial_col: "serial"})
        frames.append(tmp)
    master = pd.concat(frames, ignore_index=True).drop_duplicates("machine_key", keep="first")
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
    """Return the evaluation y vector without changing model training target."""
    mode, horizon, col = evaluation_target_settings(config, prefix=prefix, horizon_days=horizon_days)
    if col not in df.columns:
        if col == "target":
            raise ValueError("Dataset is missing required target column.")
        raise ValueError(
            f"Dataset is missing {col!r}. Re-run 02_build_case_control_dataset.py with the updated scripts "
            "so future-claim evaluation columns are added."
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


def controls_per_positive(config) -> int:
    """Return the configured matched-control ratio per positive case."""

    return int(getattr(config, "CONTROLS_PER_POSITIVE_CASE", 3))


def window_dataset_id(window_config: Mapping, config) -> str:
    """Return a compact, stable dataset id for one window configuration."""

    feature_suffix = "components_on" if bool(getattr(config, "ENABLE_COMPONENT_FEATURES", False)) else "components_off"
    lead_label = window_config_name(window_config)
    base = (
        f"{lead_label}__controls_{controls_per_positive(config)}__"
        f"neg_{int(config.CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END)}__{feature_suffix}"
    )
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", base)


def build_case_control_base_rows(
    episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    window_config: Mapping,
    config,
    claim_history_episodes: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    if lead_max <= lead_min:
        raise ValueError("lead_max_days must be greater than lead_min_days.")

    coverage = build_source_coverage(sources)
    # Use the full claim-history table for prior-claim features and control
    # exclusion checks, even when only a subset of claims is selected as
    # positive modeling cases for the current window configuration.
    history_for_claim_checks = claim_history_episodes if claim_history_episodes is not None else episodes
    dates_by_machine = claim_dates_by_machine(history_for_claim_checks)
    episodes = episodes.merge(
        coverage[["machine_key", "first_source_date", "last_source_date", "source_record_count_total"]],
        on="machine_key",
        how="left",
    )

    positive_rows = []
    skipped_rows = []
    for _, ep in episodes.iterrows():
        claim_date = pd.Timestamp(ep["claim_date"])
        window_start = claim_date - pd.Timedelta(days=lead_max)
        window_end = claim_date - pd.Timedelta(days=lead_min)
        if bool(getattr(config, "REQUIRE_POSITIVE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)):
            first_src = ep.get("first_source_date")
            last_src = ep.get("last_source_date")
            if pd.isna(first_src) or pd.isna(last_src) or pd.Timestamp(first_src) > window_end or pd.Timestamp(last_src) < window_start:
                skipped_rows.append({
                    "claim_episode_id": ep["claim_episode_id"],
                    "machine_key": ep["machine_key"],
                    "reason": "positive_no_source_coverage_overlap_window",
                    "window_start": window_start,
                    "window_end": window_end,
                })
                continue
        positive_row = {
            "row_role": "case",
            "target": 1,
            "case_control_group_id": f"{window_config_name(window_config)}__{ep['claim_episode_id']}",
            "claim_episode_id": ep["claim_episode_id"],
            "machine_key": ep["machine_key"],
            "full_model": ep["full_model"],
            "serial": ep["serial"],
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
        }
        for extra_col in [
            "positive_claim_selection_mode",
            "lead_max_days_threshold_for_repeat_claim",
            "claim_sequence_number",
            "machine_claim_event_count",
            "is_first_claim_for_machine",
            "previous_claim_date_same_machine",
            "days_since_previous_claim_same_machine",
            "claim_selection_reason",
        ]:
            if extra_col in ep.index:
                positive_row[extra_col] = ep.get(extra_col)
        positive_rows.append(positive_row)
    positives = pd.DataFrame(positive_rows)
    if getattr(config, "MAX_POSITIVE_CASES_PER_WINDOW", None):
        n = int(config.MAX_POSITIVE_CASES_PER_WINDOW)
        positives = positives.sample(n=min(n, len(positives)), random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Control pool.  For speed, sample controls by scanning a shuffled
    # candidate list until enough eligible machines are found. This avoids
    # checking every candidate for every positive case when only a few controls
    # are requested.
    master = machine_master.copy()
    if "first_source_date" not in master.columns:
        master = master.merge(coverage, on="machine_key", how="left")
    master["full_model"] = master["full_model"].map(clean_model)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")

    if bool(getattr(config, "CONTROL_MATCH_ON_FULL_MODEL", True)):
        master_by_model = {
            m: g.reset_index(drop=True)
            for m, g in master.groupby("full_model", dropna=False)
        }
    else:
        master_by_model = {"__ALL__": master.reset_index(drop=True)}

    controls = []
    control_audit_rows = []
    n_controls = controls_per_positive(config)
    for i, case in positives.iterrows():
        window_start = pd.Timestamp(case["window_start"])
        window_end = pd.Timestamp(case["window_end"])
        no_claim_start = window_start - pd.Timedelta(days=int(config.CONTROL_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START))
        no_claim_end = window_end + pd.Timedelta(days=int(config.CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END))

        if bool(getattr(config, "CONTROL_MATCH_ON_FULL_MODEL", True)):
            candidates = master_by_model.get(case["full_model"], pd.DataFrame()).copy()
        else:
            candidates = master_by_model["__ALL__"].copy()
        if not candidates.empty:
            candidates = candidates[candidates["machine_key"] != case["machine_key"]]
        if bool(getattr(config, "CONTROL_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True)) and not candidates.empty:
            candidates = candidates[
                candidates["first_source_date"].notna()
                & candidates["last_source_date"].notna()
                & (candidates["first_source_date"] <= window_end)
                & (candidates["last_source_date"] >= window_start)
            ]
        candidate_count = int(len(candidates))
        if candidates.empty:
            control_audit_rows.append({
                "case_control_group_id": case["case_control_group_id"],
                "claim_episode_id": case["claim_episode_id"],
                "candidate_count_before_claim_filter": 0,
                "eligible_control_count_checked_until_selected": 0,
                "selected_control_count": 0,
                "status": "no_candidates_before_claim_filter",
            })
            continue

        seed = int(hashlib.md5(str(case["case_control_group_id"]).encode("utf-8")).hexdigest()[:8], 16)
        candidates = candidates.sample(frac=1.0, random_state=(config.RANDOM_STATE + seed) % (2**32 - 1))

        selected_rows = []
        checked = 0
        for _, ctrl in candidates.iterrows():
            checked += 1
            m = ctrl["machine_key"]
            if has_claim_between(dates_by_machine, m, no_claim_start, no_claim_end):
                continue
            selected_rows.append(ctrl)
            if len(selected_rows) >= n_controls:
                break

        if not selected_rows:
            control_audit_rows.append({
                "case_control_group_id": case["case_control_group_id"],
                "claim_episode_id": case["claim_episode_id"],
                "candidate_count_before_claim_filter": candidate_count,
                "eligible_control_count_checked_until_selected": 0,
                "selected_control_count": 0,
                "status": "no_eligible_controls_after_claim_filter",
            })
            continue

        for j, ctrl in enumerate(selected_rows, start=1):
            controls.append({
                "row_role": "control",
                "target": 0,
                "case_control_group_id": case["case_control_group_id"],
                "claim_episode_id": case["claim_episode_id"],
                "control_number_within_group": j,
                "machine_key": ctrl["machine_key"],
                "full_model": ctrl["full_model"],
                "serial": ctrl.get("serial", ""),
                "window_name": case["window_name"],
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "window_start": window_start,
                "window_end": window_end,
                "future_claim_date": pd.NaT,
                "days_from_window_end_to_claim": np.nan,
                "control_no_claim_start": no_claim_start,
                "control_no_claim_end": no_claim_end,
                "control_sampling_reason": "same_window_same_full_model_no_claim_in_exclusion_horizon",
            })
        control_audit_rows.append({
            "case_control_group_id": case["case_control_group_id"],
            "claim_episode_id": case["claim_episode_id"],
            "candidate_count_before_claim_filter": candidate_count,
            "eligible_control_count_checked_until_selected": int(checked),
            "selected_control_count": int(len(selected_rows)),
            "status": "selected" if len(selected_rows) >= n_controls else "selected_fewer_than_requested",
        })

    base = pd.concat([positives, pd.DataFrame(controls)], ignore_index=True, sort=False)
    audit = pd.concat([pd.DataFrame(skipped_rows), pd.DataFrame(control_audit_rows)], ignore_index=True, sort=False)
    return base, audit


def _first_existing_timestamp(*values) -> pd.Timestamp:
    for value in values:
        if value is None or pd.isna(value):
            continue
        return pd.Timestamp(value)
    return pd.NaT


def _max_claim_observation_date(config) -> pd.Timestamp:
    for attr in ["MAX_CLAIM_DATE", "MAX_VALID_EVENT_DATE"]:
        value = getattr(config, attr, None)
        if value is not None:
            return pd.Timestamp(value)
    return pd.Timestamp.today().normalize()


def build_population_random_negative_base_rows(
    reference_split_df: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    claim_history_episodes: pd.DataFrame,
    window_config: Mapping,
    split_name: str,
    negatives_per_positive: int,
    config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build population-like random negative windows for validation/test.

    These rows are not matched to claim dates.  A row represents a realistic
    production scoring window:

        window_start = random_as_of_date - (lead_max_days - lead_min_days)
        window_end   = random_as_of_date

    It is labeled negative only when the machine has no claim in the configured
    future no-claim horizon after window_end.  By default, rows with a claim
    inside the observation window are also excluded to avoid labeling an active
    claim window as a clean negative.
    """

    split_name = str(split_name)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = int(lead_max - lead_min)
    if observation_days <= 0:
        raise ValueError("lead_max_days must be greater than lead_min_days for population negatives.")

    if reference_split_df is None or reference_split_df.empty or int(negatives_per_positive) <= 0:
        return pd.DataFrame(), pd.DataFrame()

    positives = reference_split_df[pd.to_numeric(reference_split_df.get("target", 0), errors="coerce").fillna(0).astype(int).eq(1)].copy()
    positive_count = int(len(positives))
    requested = int(positive_count * int(negatives_per_positive))
    if requested <= 0:
        return pd.DataFrame(), pd.DataFrame()

    dates_by_machine = claim_dates_by_machine(claim_history_episodes)
    coverage = build_source_coverage(sources)
    master = machine_master.copy()
    if "first_source_date" not in master.columns or "last_source_date" not in master.columns:
        master = master.merge(coverage, on="machine_key", how="left")
    master["full_model"] = master.get("full_model", "").map(clean_model)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master = master.dropna(subset=["machine_key", "first_source_date", "last_source_date"]).copy()
    master = master[master["machine_key"].astype(str).str.len() > 0].copy()

    if master.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "split": split_name,
            "status": "no_machine_master_candidates",
            "requested_population_negative_rows": requested,
            "selected_population_negative_rows": 0,
        }])

    # Use the chronological validation/test date range from the matched split so
    # random negatives are sampled from the same historical period as the holdout.
    date_col = str(getattr(config, "SPLIT_DATE_COL", "window_end"))
    date_source = positives if not positives.empty else reference_split_df
    if date_col not in date_source.columns:
        date_col = "window_end"
    split_min = pd.to_datetime(date_source[date_col], errors="coerce").min()
    split_max = pd.to_datetime(date_source[date_col], errors="coerce").max()
    if pd.isna(split_min) or pd.isna(split_max):
        split_min = pd.to_datetime(reference_split_df["window_end"], errors="coerce").min()
        split_max = pd.to_datetime(reference_split_df["window_end"], errors="coerce").max()

    future_horizon = int(getattr(
        config,
        "POPULATION_RANDOM_NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END",
        getattr(config, "CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180),
    ))
    require_future_observable = bool(getattr(config, "POPULATION_RANDOM_NEGATIVE_REQUIRE_FUTURE_OBSERVABILITY", True))
    max_claim_observed_date = _max_claim_observation_date(config)
    if require_future_observable:
        split_max = min(pd.Timestamp(split_max), max_claim_observed_date - pd.Timedelta(days=future_horizon))

    exclude_claims_in_window = bool(getattr(config, "POPULATION_RANDOM_NEGATIVE_EXCLUDE_CLAIMS_DURING_OBSERVATION_WINDOW", True))
    require_coverage = bool(getattr(config, "POPULATION_RANDOM_NEGATIVE_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True))
    max_attempts = int(getattr(config, "POPULATION_RANDOM_NEGATIVE_MAX_ATTEMPTS_MULTIPLIER", 80)) * max(requested, 1)
    random_state_offset = 100000 if split_name == "validation" else 200000
    rng = np.random.default_rng(int(getattr(config, "RANDOM_STATE", 42)) + random_state_offset + lead_max * 10 + lead_min)

    rows = []
    audit_rows = []
    seen = set()
    checked = 0
    master_records = master.reset_index(drop=True)
    n_master = len(master_records)

    while len(rows) < requested and checked < max_attempts:
        checked += 1
        ctrl = master_records.iloc[int(rng.integers(0, n_master))]
        m = ctrl["machine_key"]

        earliest_end = max(pd.Timestamp(ctrl["first_source_date"]) + pd.Timedelta(days=observation_days), pd.Timestamp(split_min))
        latest_end = min(pd.Timestamp(ctrl["last_source_date"]), pd.Timestamp(split_max))
        if pd.isna(earliest_end) or pd.isna(latest_end) or latest_end < earliest_end:
            continue
        span_days = int((latest_end - earliest_end).days)
        offset = int(rng.integers(0, span_days + 1)) if span_days > 0 else 0
        window_end = earliest_end + pd.Timedelta(days=offset)
        window_start = window_end - pd.Timedelta(days=observation_days)

        if require_coverage:
            if pd.Timestamp(ctrl["first_source_date"]) > window_end or pd.Timestamp(ctrl["last_source_date"]) < window_start:
                continue

        future_start = window_end + pd.Timedelta(days=1)
        future_end = window_end + pd.Timedelta(days=future_horizon)
        if has_claim_between(dates_by_machine, m, future_start, future_end):
            continue
        if exclude_claims_in_window and has_claim_between(dates_by_machine, m, window_start, window_end):
            continue

        key = (str(m), pd.Timestamp(window_end).date().isoformat(), split_name)
        if key in seen:
            continue
        seen.add(key)

        idx = len(rows) + 1
        rows.append({
            "row_role": "population_negative",
            "target": 0,
            "split": split_name,
            "case_control_group_id": f"{window_config_name(window_config)}__population_negative__{split_name}__{idx:07d}",
            "claim_episode_id": "",
            "control_number_within_group": np.nan,
            "machine_key": m,
            "full_model": ctrl.get("full_model", ""),
            "serial": ctrl.get("serial", ""),
            "window_name": window_config_name(window_config),
            "lead_max_days": lead_max,
            "lead_min_days": lead_min,
            "population_window_length_days": observation_days,
            "window_start": window_start,
            "window_end": window_end,
            "future_claim_date": pd.NaT,
            "days_from_window_end_to_claim": np.nan,
            "control_no_claim_start": future_start,
            "control_no_claim_end": future_end,
            "control_sampling_reason": "population_random_negative_no_claim_in_future_horizon",
            "population_negative_split": split_name,
            "population_negative_requested_ratio": int(negatives_per_positive),
            "population_negative_future_horizon_days": future_horizon,
            "population_negative_observation_days": observation_days,
            "population_negative_excluded_claims_in_window": exclude_claims_in_window,
        })

    audit_rows.append({
        "split": split_name,
        "window_name": window_config_name(window_config),
        "lead_max_days": lead_max,
        "lead_min_days": lead_min,
        "observation_days": observation_days,
        "positive_rows_in_reference_split": positive_count,
        "negatives_per_positive_requested": int(negatives_per_positive),
        "requested_population_negative_rows": requested,
        "selected_population_negative_rows": int(len(rows)),
        "candidate_machines": int(len(master_records)),
        "attempts_checked": int(checked),
        "max_attempts": int(max_attempts),
        "split_date_min_used": split_min,
        "split_date_max_used": split_max,
        "future_horizon_days": future_horizon,
        "require_future_observability": require_future_observable,
        "exclude_claims_during_observation_window": exclude_claims_in_window,
        "require_source_coverage_overlap_window": require_coverage,
        "status": "selected_requested_count" if len(rows) >= requested else "selected_fewer_than_requested",
    })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)


def _max_evaluation_horizon_days_for_asof(config) -> int:
    """Return the maximum horizon needed to label as-of population rows safely."""
    horizons = configured_evaluation_horizons(config)
    if horizons:
        return int(max(horizons))
    return int(getattr(config, "CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END", 180))


def build_asof_population_evaluation_base_rows(
    reference_split_df: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    claim_history_episodes: pd.DataFrame,
    window_config: Mapping,
    split_name: str,
    config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build realistic as-of-date population evaluation windows.

    These rows are evaluation-only and are not used for training.  They mimic a
    production scoring snapshot: for each historical as-of date in the holdout
    period, build one lookback window per eligible machine, then let future
    claim-horizon columns decide whether the row is positive for 30/60/90/etc.
    day evaluation.
    """

    split_name = str(split_name)
    lead_max = int(window_config["lead_max_days"])
    lead_min = int(window_config["lead_min_days"])
    observation_days = int(lead_max - lead_min)
    if observation_days <= 0:
        raise ValueError("lead_max_days must be greater than lead_min_days for as-of population evaluation.")

    if reference_split_df is None or reference_split_df.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "split": split_name,
            "status": "empty_reference_split",
            "selected_asof_rows": 0,
        }])

    coverage = build_source_coverage(sources)
    master = machine_master.copy()
    if "first_source_date" not in master.columns or "last_source_date" not in master.columns:
        master = master.merge(coverage, on="machine_key", how="left")
    master["full_model"] = master.get("full_model", "").map(clean_model)
    master["first_source_date"] = pd.to_datetime(master.get("first_source_date"), errors="coerce")
    master["last_source_date"] = pd.to_datetime(master.get("last_source_date"), errors="coerce")
    master = master.dropna(subset=["machine_key", "first_source_date", "last_source_date"]).copy()
    master = master[master["machine_key"].astype(str).str.len() > 0].copy()
    if master.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "split": split_name,
            "status": "no_machine_master_candidates",
            "selected_asof_rows": 0,
        }])

    date_col = str(getattr(config, "SPLIT_DATE_COL", "window_end"))
    date_source = reference_split_df.copy()
    if date_col not in date_source.columns:
        date_col = "window_end"
    split_min = pd.to_datetime(date_source[date_col], errors="coerce").min()
    split_max = pd.to_datetime(date_source[date_col], errors="coerce").max()
    if pd.isna(split_min) or pd.isna(split_max):
        return pd.DataFrame(), pd.DataFrame([{
            "split": split_name,
            "status": "missing_reference_date_range",
            "selected_asof_rows": 0,
        }])

    max_horizon = _max_evaluation_horizon_days_for_asof(config)
    require_future_observable = bool(getattr(config, "ASOF_EVALUATION_REQUIRE_FUTURE_OBSERVABILITY", True))
    max_claim_observed_date = _max_claim_observation_date(config)
    if require_future_observable:
        split_max = min(pd.Timestamp(split_max), max_claim_observed_date - pd.Timedelta(days=max_horizon))
    if split_max < split_min:
        return pd.DataFrame(), pd.DataFrame([{
            "split": split_name,
            "status": "no_dates_after_future_observability_filter",
            "selected_asof_rows": 0,
            "reference_split_min": split_min,
            "reference_split_max_after_filter": split_max,
            "max_evaluation_horizon_days": max_horizon,
            "max_claim_observed_date": max_claim_observed_date,
        }])

    frequency_days = int(getattr(config, "ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS", 30) or 30)
    frequency_days = max(1, frequency_days)
    snapshot_dates = list(pd.date_range(pd.Timestamp(split_min), pd.Timestamp(split_max), freq=f"{frequency_days}D"))
    if not snapshot_dates or snapshot_dates[-1] != pd.Timestamp(split_max):
        snapshot_dates.append(pd.Timestamp(split_max))

    max_machines_per_snapshot = getattr(config, "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT", None)
    if max_machines_per_snapshot is not None:
        max_machines_per_snapshot = int(max_machines_per_snapshot)
        if max_machines_per_snapshot <= 0:
            max_machines_per_snapshot = None

    split_max_rows_attr = f"{split_name.upper()}_ASOF_EVALUATION_MAX_ROWS"
    max_rows = getattr(config, split_max_rows_attr, None)
    if max_rows is None:
        max_rows = getattr(config, "ASOF_EVALUATION_MAX_ROWS_PER_SPLIT", None)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows <= 0:
            max_rows = None

    require_coverage = bool(getattr(config, "ASOF_EVALUATION_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW", True))
    exclude_claims_in_window = bool(getattr(config, "ASOF_EVALUATION_EXCLUDE_CLAIMS_DURING_OBSERVATION_WINDOW", False))
    dates_by_machine = claim_dates_by_machine(claim_history_episodes)
    random_state_offset = 300000 if split_name == "validation" else 400000
    rng = np.random.default_rng(int(getattr(config, "RANDOM_STATE", 42)) + random_state_offset + lead_max * 10 + lead_min)

    rows = []
    audit_rows = []
    seen = set()
    for snapshot_idx, as_of in enumerate(snapshot_dates, start=1):
        window_end = pd.Timestamp(as_of)
        window_start = window_end - pd.Timedelta(days=observation_days)
        candidates = master.copy()
        if require_coverage:
            candidates = candidates[
                (candidates["first_source_date"] <= window_end)
                & (candidates["last_source_date"] >= window_start)
            ].copy()
        if exclude_claims_in_window and not candidates.empty:
            keep_mask = [not has_claim_between(dates_by_machine, m, window_start, window_end) for m in candidates["machine_key"]]
            candidates = candidates.loc[keep_mask].copy()
        eligible_count = int(len(candidates))
        if candidates.empty:
            audit_rows.append({
                "split": split_name,
                "snapshot_index": snapshot_idx,
                "as_of_date": window_end,
                "eligible_machines": 0,
                "selected_rows": 0,
                "status": "no_eligible_machines",
            })
            continue
        if max_machines_per_snapshot is not None and len(candidates) > max_machines_per_snapshot:
            seed = int(rng.integers(0, 2**31 - 1))
            candidates = candidates.sample(n=max_machines_per_snapshot, random_state=seed)
        selected_count = 0
        for _, ctrl in candidates.iterrows():
            if max_rows is not None and len(rows) >= max_rows:
                break
            m = ctrl["machine_key"]
            key = (str(m), window_end.date().isoformat(), split_name)
            if key in seen:
                continue
            seen.add(key)
            idx = len(rows) + 1
            rows.append({
                "row_role": "asof_population_window",
                "target": 0,
                "split": split_name,
                "case_control_group_id": f"{window_config_name(window_config)}__asof_population__{split_name}__{window_end.strftime('%Y%m%d')}__{idx:07d}",
                "claim_episode_id": "",
                "control_number_within_group": np.nan,
                "machine_key": m,
                "full_model": ctrl.get("full_model", ""),
                "serial": ctrl.get("serial", ""),
                "window_name": window_config_name(window_config),
                "lead_max_days": lead_max,
                "lead_min_days": lead_min,
                "population_window_length_days": observation_days,
                "as_of_date": window_end,
                "window_start": window_start,
                "window_end": window_end,
                "future_claim_date": pd.NaT,
                "days_from_window_end_to_claim": np.nan,
                "control_no_claim_start": pd.NaT,
                "control_no_claim_end": pd.NaT,
                "control_sampling_reason": "asof_population_snapshot_window",
                "asof_population_split": split_name,
                "asof_snapshot_frequency_days": frequency_days,
                "asof_max_evaluation_horizon_days": max_horizon,
                "asof_require_future_observability": require_future_observable,
                "asof_excluded_claims_in_observation_window": exclude_claims_in_window,
            })
            selected_count += 1
        audit_rows.append({
            "split": split_name,
            "snapshot_index": snapshot_idx,
            "as_of_date": window_end,
            "window_start": window_start,
            "window_end": window_end,
            "eligible_machines": eligible_count,
            "selected_rows": int(selected_count),
            "max_machines_per_snapshot": max_machines_per_snapshot,
            "status": "selected",
        })
        if max_rows is not None and len(rows) >= max_rows:
            break

    audit_rows.append({
        "split": split_name,
        "snapshot_index": "summary",
        "as_of_date": pd.NaT,
        "window_start": pd.NaT,
        "window_end": pd.NaT,
        "reference_split_min": split_min,
        "reference_split_max_used": split_max,
        "snapshot_count": int(len(snapshot_dates)),
        "selected_rows": int(len(rows)),
        "candidate_machines": int(len(master)),
        "observation_days": observation_days,
        "max_evaluation_horizon_days": max_horizon,
        "require_future_observability": require_future_observable,
        "require_source_coverage_overlap_window": require_coverage,
        "exclude_claims_during_observation_window": exclude_claims_in_window,
        "status": "selected_rows" if rows else "no_rows_selected",
    })
    return pd.DataFrame(rows), pd.DataFrame(audit_rows)

def latest_operation_smr_before(operation_df: pd.DataFrame, machine_key: str, cutoff) -> float:
    if operation_df.empty:
        return np.nan
    sub = operation_df[(operation_df["machine_key"] == machine_key) & (operation_df["LOCAL_DATE"] <= pd.Timestamp(cutoff))]
    if sub.empty or "smr_hours" not in sub.columns:
        return np.nan
    sub = sub.sort_values("LOCAL_DATE", kind="mergesort")
    val = pd.to_numeric(sub["smr_hours"], errors="coerce").dropna()
    if val.empty:
        return np.nan
    return float(val.iloc[-1])


# -----------------------------------------------------------------------------
# Window feature extraction
# -----------------------------------------------------------------------------
def make_group_dict(df: pd.DataFrame, date_col: str) -> Dict[str, pd.DataFrame]:
    out = {}
    if df.empty:
        return out
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    for m, g in work.sort_values(["machine_key", date_col], kind="mergesort").groupby("machine_key"):
        g = g.reset_index(drop=True)
        # Cache an int64 datetime array for fast binary-search slicing.
        g.attrs["_date_col"] = date_col
        g.attrs["_date_values_ns"] = g[date_col].values.astype("datetime64[ns]").astype("int64")
        out[m] = g
    return out


def window_slice(grouped: Mapping[str, pd.DataFrame], machine_key: str, date_col: str, start, end) -> pd.DataFrame:
    g = grouped.get(machine_key)
    if g is None or g.empty:
        return pd.DataFrame()
    date_values = g.attrs.get("_date_values_ns")
    if date_values is None:
        date_values = pd.to_datetime(g[date_col]).values.astype("datetime64[ns]").astype("int64")
    start_ns = pd.Timestamp(start).to_datetime64().astype("datetime64[ns]").astype("int64")
    end_ns = pd.Timestamp(end).to_datetime64().astype("datetime64[ns]").astype("int64")
    left = int(np.searchsorted(date_values, start_ns, side="left"))
    right = int(np.searchsorted(date_values, end_ns, side="right"))
    if right <= left:
        return g.iloc[0:0]
    return g.iloc[left:right]


def build_window_features(base_rows: pd.DataFrame, sources: Mapping[str, pd.DataFrame], episodes: pd.DataFrame) -> pd.DataFrame:
    fault_groups = make_group_dict(sources.get("fault", pd.DataFrame()), "event_date")
    fluid_groups = make_group_dict(sources.get("fluid", pd.DataFrame()), "sample_drawn_date")
    maintenance_groups = make_group_dict(sources.get("maintenance", pd.DataFrame()), "event_date")
    operation_groups = make_group_dict(sources.get("operation", pd.DataFrame()), "LOCAL_DATE")
    dates_by_machine = claim_dates_by_machine(episodes)

    feature_rows = []
    for _, row in base_rows.iterrows():
        machine_key = row["machine_key"]
        start = pd.Timestamp(row["window_start"])
        end = pd.Timestamp(row["window_end"])
        f_fault = aggregate_faults(window_slice(fault_groups, machine_key, "event_date", start, end), end)
        f_fluid = aggregate_fluids(window_slice(fluid_groups, machine_key, "sample_drawn_date", start, end), end)
        f_maint = aggregate_maintenance(window_slice(maintenance_groups, machine_key, "event_date", start, end), end)
        f_oper = aggregate_operation(window_slice(operation_groups, machine_key, "LOCAL_DATE", start, end), end)
        prior_count, days_since_prior = count_claims_before(dates_by_machine, machine_key, start)
        source_record_count = (
            f_fault["fault_count_window"]
            + f_fluid["fluid_sample_count_window"]
            + f_maint["maintenance_event_count_window"]
            + f_oper["operation_day_count_window"]
        )
        features = {
            "prior_claim_count_before_window": prior_count,
            "days_since_prior_claim_before_window": days_since_prior,
            "source_record_count_window": source_record_count,
            "has_any_source_window": int(source_record_count > 0),
        }
        features.update(f_fault)
        features.update(f_fluid)
        features.update(f_maint)
        features.update(f_oper)
        feature_rows.append(features)
    return pd.concat([base_rows.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)


def _mode_or_none(series: pd.Series, default: str = "NONE") -> str:
    s = series.dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return default
    return str(s.value_counts().index[0])


def aggregate_faults(df: pd.DataFrame, window_end) -> dict:
    if df.empty:
        return {
            "has_fault_window": 0,
            "fault_count_window": 0,
            "fault_unique_code_count_window": 0,
            "fault_l03plus_count_window": 0,
            "fault_l04plus_count_window": 0,
            "fault_max_action_level_window": 0.0,
            "fault_max_evidence_score_window": 0.0,
            "fault_mean_evidence_score_window": 0.0,
            "fault_max_log_occurrence_window": 0.0,
            "fault_days_since_latest_in_window": np.nan,
            "fault_mechanical_count_window": 0,
            "fault_electrical_count_window": 0,
            "fault_dominant_component_window": "NONE",
        }
    action = pd.to_numeric(df.get("action_level_num", 0), errors="coerce").fillna(0)
    score = pd.to_numeric(df.get("failure_code_evidence_score", 0), errors="coerce").fillna(0)
    log_occ = pd.to_numeric(df.get("log_occurrence_count", 0), errors="coerce").fillna(0)
    latest_date = pd.to_datetime(df["event_date"]).max()
    comp_col = "related_component" if "related_component" in df.columns else "applicable_component"
    return {
        "has_fault_window": 1,
        "fault_count_window": int(len(df)),
        "fault_unique_code_count_window": int(df.get("fault_code", pd.Series(dtype=str)).nunique(dropna=True)),
        "fault_l03plus_count_window": int((action >= 3).sum()),
        "fault_l04plus_count_window": int((action >= 4).sum()),
        "fault_max_action_level_window": float(action.max()) if len(action) else 0.0,
        "fault_max_evidence_score_window": float(score.max()) if len(score) else 0.0,
        "fault_mean_evidence_score_window": float(score.mean()) if len(score) else 0.0,
        "fault_max_log_occurrence_window": float(log_occ.max()) if len(log_occ) else 0.0,
        "fault_days_since_latest_in_window": float((pd.Timestamp(window_end) - latest_date).days),
        "fault_mechanical_count_window": int(pd.to_numeric(df.get("is_mechanical_failure_code", 0), errors="coerce").fillna(0).sum()),
        "fault_electrical_count_window": int(pd.to_numeric(df.get("is_electrical_failure_code", 0), errors="coerce").fillna(0).sum()),
        "fault_dominant_component_window": _mode_or_none(df.get(comp_col, pd.Series(dtype=str))),
    }


def aggregate_fluids(df: pd.DataFrame, window_end) -> dict:
    base = {
        "has_fluid_window": 0,
        "fluid_sample_count_window": 0,
        "fluid_max_severity_window": 0.0,
        "fluid_latest_severity_window": 0.0,
        "fluid_days_since_latest_sample_window": np.nan,
        "fluid_max_cu_ppm_window": 0.0,
        "fluid_max_fe_ppm_window": 0.0,
        "fluid_max_pb_ppm_window": 0.0,
        "fluid_max_soot_percent_window": 0.0,
        "fluid_max_water_percent_window": 0.0,
    }
    if df.empty:
        return base
    df = df.sort_values("sample_drawn_date", kind="mergesort")
    latest = df.iloc[-1]
    severity = pd.to_numeric(df.get("sample_result_severity_order", 0), errors="coerce").fillna(0)
    latest_date = pd.Timestamp(latest["sample_drawn_date"])
    base.update({
        "has_fluid_window": 1,
        "fluid_sample_count_window": int(len(df)),
        "fluid_max_severity_window": float(severity.max()) if len(severity) else 0.0,
        "fluid_latest_severity_window": float(pd.to_numeric(pd.Series([latest.get("sample_result_severity_order", 0)]), errors="coerce").fillna(0).iloc[0]),
        "fluid_days_since_latest_sample_window": float((pd.Timestamp(window_end) - latest_date).days),
        "fluid_max_cu_ppm_window": _max_numeric(df, "Cu_Copper_PPM"),
        "fluid_max_fe_ppm_window": _max_numeric(df, "Fe_Iron_PPM"),
        "fluid_max_pb_ppm_window": _max_numeric(df, "Pb_Lead_PPM"),
        "fluid_max_soot_percent_window": _max_numeric(df, "Soot_Soot_PERCENT"),
        "fluid_max_water_percent_window": _max_numeric(df, "Water_Water_PERCENT"),
    })
    return base


def _max_numeric(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.max()) if len(s) else 0.0


def aggregate_maintenance(df: pd.DataFrame, window_end) -> dict:
    if df.empty:
        return {
            "has_maintenance_window": 0,
            "maintenance_event_count_window": 0,
            "maintenance_monitor_reset_count_window": 0,
            "maintenance_overdue_count_window": 0,
            "maintenance_due_now_count_window": 0,
            "maintenance_min_remaining_hours_window": np.nan,
            "maintenance_days_since_latest_event_window": np.nan,
            "maintenance_dominant_component_window": "NONE",
        }
    latest_date = pd.to_datetime(df["event_date"]).max()
    remaining = pd.to_numeric(df.get("remaining_hours", np.nan), errors="coerce")
    comp_col = "related_component" if "related_component" in df.columns else "maintenance_type"
    return {
        "has_maintenance_window": 1,
        "maintenance_event_count_window": int(len(df)),
        "maintenance_monitor_reset_count_window": int(pd.to_numeric(df.get("is_monitor_reset", 0), errors="coerce").fillna(0).sum()),
        "maintenance_overdue_count_window": int(pd.to_numeric(df.get("is_overdue", 0), errors="coerce").fillna(0).sum()),
        "maintenance_due_now_count_window": int(pd.to_numeric(df.get("is_due_now", 0), errors="coerce").fillna(0).sum()),
        "maintenance_min_remaining_hours_window": float(remaining.min()) if remaining.notna().any() else np.nan,
        "maintenance_days_since_latest_event_window": float((pd.Timestamp(window_end) - latest_date).days),
        "maintenance_dominant_component_window": _mode_or_none(df.get(comp_col, pd.Series(dtype=str))),
    }


def aggregate_operation(df: pd.DataFrame, window_end) -> dict:
    if df.empty:
        return {
            "has_operation_window": 0,
            "operation_day_count_window": 0,
            "operation_working_hours_sum_window": 0.0,
            "operation_working_hours_mean_window": 0.0,
            "operation_working_hours_max_window": 0.0,
            "operation_engine_running_hours_sum_window": 0.0,
            "operation_idle_hours_sum_window": 0.0,
            "operation_idle_share_window": np.nan,
            "operation_latest_smr_window": np.nan,
            "operation_smr_delta_window": np.nan,
            "operation_high_throttle_day_count_window": 0,
        }
    df = df.sort_values("LOCAL_DATE", kind="mergesort")
    working = pd.to_numeric(df.get("working_hours_clean", df.get("actual_working_hours_clean", 0)), errors="coerce").fillna(0)
    engine = pd.to_numeric(df.get("engine_running_hours_clean", 0), errors="coerce").fillna(0)
    idle = pd.to_numeric(df.get("engine_idling_hours_clean", 0), errors="coerce").fillna(0)
    smr = pd.to_numeric(df.get("smr_hours", np.nan), errors="coerce").dropna()
    high_throttle = pd.to_numeric(df.get("high_throttle_day_flag", 0), errors="coerce").fillna(0)
    return {
        "has_operation_window": 1,
        "operation_day_count_window": int(len(df)),
        "operation_working_hours_sum_window": float(working.sum()),
        "operation_working_hours_mean_window": float(working.mean()) if len(working) else 0.0,
        "operation_working_hours_max_window": float(working.max()) if len(working) else 0.0,
        "operation_engine_running_hours_sum_window": float(engine.sum()),
        "operation_idle_hours_sum_window": float(idle.sum()),
        "operation_idle_share_window": float(idle.sum() / engine.sum()) if engine.sum() > 0 else np.nan,
        "operation_latest_smr_window": float(smr.iloc[-1]) if len(smr) else np.nan,
        "operation_smr_delta_window": float(smr.iloc[-1] - smr.iloc[0]) if len(smr) >= 2 else np.nan,
        "operation_high_throttle_day_count_window": int((high_throttle > 0).sum()),
    }


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------
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


def split_groups_random(df: pd.DataFrame, group_col: str, test_size: float, random_state: int):
    rng = np.random.default_rng(random_state)
    groups = np.array(sorted(df[group_col].dropna().unique()))
    rng.shuffle(groups)
    n_test = max(1, int(np.ceil(len(groups) * test_size)))
    test_groups = set(groups[:n_test])
    test_mask = df[group_col].isin(test_groups)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def split_case_control_train_validation_test(
    df: pd.DataFrame,
    config,
    group_col: str = "case_control_group_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign chronological train / validation / test splits by case-control group.

    Each positive case and its sampled controls share one case_control_group_id.
    This function assigns the whole group to one split using the configured split
    date, usually window_end. The returned dataframe includes a `split` column.
    """

    if group_col not in df.columns:
        raise ValueError(f"Required group column is missing: {group_col}")

    split_date_col = str(getattr(config, "SPLIT_DATE_COL", "window_end"))
    if split_date_col not in df.columns:
        raise ValueError(f"Configured SPLIT_DATE_COL is missing: {split_date_col}")

    ratios = {
        "train": float(getattr(config, "TRAIN_RATIO", 0.70)),
        "validation": float(getattr(config, "VALIDATION_RATIO", 0.15)),
        "test": float(getattr(config, "TEST_RATIO", 0.15)),
    }
    ratio_sum = sum(v for v in ratios.values() if v > 0)
    if ratio_sum <= 0:
        raise ValueError("TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must be positive.")
    ratios = {k: max(v, 0.0) / ratio_sum for k, v in ratios.items()}

    work = df.copy()
    work[split_date_col] = pd.to_datetime(work[split_date_col], errors="coerce")
    if work[split_date_col].isna().any():
        bad = int(work[split_date_col].isna().sum())
        raise ValueError(f"Split date column {split_date_col} has {bad} missing/unparseable values.")

    group_summary = (
        work.groupby(group_col, dropna=False)
        .agg(
            split_date=(split_date_col, "min"),
            rows=(group_col, "size"),
            positive_rows=("target", "sum"),
        )
        .reset_index()
        .sort_values(["split_date", group_col], kind="mergesort")
        .reset_index(drop=True)
    )

    n_groups = len(group_summary)
    train_end = int(np.floor(n_groups * ratios["train"]))
    validation_end = int(np.floor(n_groups * (ratios["train"] + ratios["validation"])))

    # Keep at least one group in validation/test when their ratios are positive
    # and enough groups exist. This avoids empty holdouts on small debug samples.
    if ratios["validation"] > 0 and n_groups >= 3 and validation_end <= train_end:
        validation_end = min(train_end + 1, n_groups)
    if ratios["test"] > 0 and n_groups >= 3 and validation_end >= n_groups:
        validation_end = n_groups - 1
    if train_end <= 0 and n_groups > 0:
        train_end = 1
    if validation_end < train_end:
        validation_end = train_end

    group_summary["split"] = "test"
    if train_end > 0:
        group_summary.loc[: train_end - 1, "split"] = "train"
    if validation_end > train_end:
        group_summary.loc[train_end: validation_end - 1, "split"] = "validation"

    split_map = dict(zip(group_summary[group_col], group_summary["split"]))
    work["split"] = work[group_col].map(split_map).astype(str)

    split_summary = (
        work.groupby("split", dropna=False)
        .agg(
            rows=("target", "size"),
            positive_rows=("target", "sum"),
            groups=(group_col, "nunique"),
            split_date_min=(split_date_col, "min"),
            split_date_max=(split_date_col, "max"),
        )
        .reset_index()
    )
    split_summary["positive_rate"] = split_summary["positive_rows"] / split_summary["rows"]
    split_summary["train_ratio_configured"] = ratios["train"]
    split_summary["validation_ratio_configured"] = ratios["validation"]
    split_summary["test_ratio_configured"] = ratios["test"]
    split_summary["split_date_col"] = split_date_col

    return work, split_summary


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
# Faster vectorized window feature extraction. This definition intentionally
# overrides the earlier row-loop build_window_features function.
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


def _safe_component_group_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip().lower()).strip("_")


def _component_text_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in present:
        out = out + " " + df[col].fillna("").astype(str)
    return out.str.lower()


def _component_mask(text: pd.Series, keywords: Sequence[str]) -> pd.Series:
    kws = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not kws:
        return pd.Series(False, index=text.index)
    pattern = "|".join(re.escape(k) for k in kws)
    return text.str.contains(pattern, regex=True, na=False)


def _component_groups(config) -> Mapping[str, Sequence[str]]:
    return getattr(config, "COMPONENT_FEATURE_GROUPS", {}) or {}


def _aggregate_faults_vectorized(base: pd.DataFrame, fault: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "fault_code",
        "action_level_num",
        "failure_code_evidence_score",
        "log_occurrence_count",
        "is_mechanical_failure_code",
        "is_electrical_failure_code",
        "related_component",
        "applicable_component",
        "event_error_name_en",
    ]
    m = _source_window_join(base, fault, "event_date", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    m["action_level_num"] = pd.to_numeric(m.get("action_level_num", 0), errors="coerce").fillna(0)
    m["failure_code_evidence_score"] = pd.to_numeric(m.get("failure_code_evidence_score", 0), errors="coerce").fillna(0)
    m["log_occurrence_count"] = pd.to_numeric(m.get("log_occurrence_count", 0), errors="coerce").fillna(0)
    m["is_mechanical_failure_code"] = pd.to_numeric(m.get("is_mechanical_failure_code", 0), errors="coerce").fillna(0)
    m["is_electrical_failure_code"] = pd.to_numeric(m.get("is_electrical_failure_code", 0), errors="coerce").fillna(0)
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
    comp_col = "related_component" if "related_component" in m.columns else "applicable_component"
    dom = (
        m[["row_id", comp_col]]
        .dropna()
        .assign(_component=lambda x: x[comp_col].astype(str).str.strip())
    )
    if not dom.empty:
        dom = dom[dom["_component"].ne("")]
        dom = dom.groupby("row_id")['_component'].agg(lambda x: x.value_counts().index[0]).reset_index()
        dom = dom.rename(columns={"_component": "fault_dominant_component_window"})
        ag = ag.merge(dom, on="row_id", how="left")
    if bool(getattr(__import__("config"), "ENABLE_COMPONENT_FEATURES", False)):
        cfg = __import__("config")
        text = _component_text_frame(
            m,
            ["related_component", "applicable_component", "event_error_name_en", "fault_code"],
        )
        for raw_group, keywords in _component_groups(cfg).items():
            group = _safe_component_group_name(raw_group)
            sub = m.loc[_component_mask(text, keywords)].copy()
            if sub.empty:
                continue
            comp = sub.groupby("row_id", dropna=False).agg(
                **{
                    f"fault_component_{group}_count_window": ("event_date", "size"),
                    f"fault_component_{group}_l03plus_count_window": ("_l03plus", "sum"),
                    f"fault_component_{group}_max_action_level_window": ("action_level_num", "max"),
                    f"fault_component_{group}_max_evidence_score_window": ("failure_code_evidence_score", "max"),
                }
            ).reset_index()
            ag = ag.merge(comp, on="row_id", how="left")
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
        if col in m.columns:
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
    keep = [
        "is_monitor_reset",
        "is_overdue",
        "is_due_now",
        "remaining_hours",
        "related_component",
        "related_component_1",
        "related_component_2",
        "EVENT_NAME_EN",
        "maintenance_type",
    ]
    m = _source_window_join(base, maintenance, "event_date", keep)
    if m.empty:
        return pd.DataFrame(columns=["row_id"])
    for col in ["is_monitor_reset", "is_overdue", "is_due_now", "remaining_hours"]:
        if col in m.columns:
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
    comp_col = "related_component" if "related_component" in m.columns else "maintenance_type"
    dom = (
        m[["row_id", comp_col]]
        .dropna()
        .assign(_component=lambda x: x[comp_col].astype(str).str.strip())
    )
    if not dom.empty:
        dom = dom[dom["_component"].ne("")]
        dom = dom.groupby("row_id")['_component'].agg(lambda x: x.value_counts().index[0]).reset_index()
        dom = dom.rename(columns={"_component": "maintenance_dominant_component_window"})
        ag = ag.merge(dom, on="row_id", how="left")
    if bool(getattr(__import__("config"), "ENABLE_COMPONENT_FEATURES", False)):
        cfg = __import__("config")
        text = _component_text_frame(
            m,
            [
                "related_component",
                "related_component_1",
                "related_component_2",
                "EVENT_NAME_EN",
                "maintenance_type",
            ],
        )
        for raw_group, keywords in _component_groups(cfg).items():
            group = _safe_component_group_name(raw_group)
            sub = m.loc[_component_mask(text, keywords)].copy()
            if sub.empty:
                continue
            comp = sub.groupby("row_id", dropna=False).agg(
                **{
                    f"maintenance_component_{group}_count_window": ("event_date", "size"),
                    f"maintenance_component_{group}_overdue_count_window": ("is_overdue", "sum"),
                    f"maintenance_component_{group}_due_now_count_window": ("is_due_now", "sum"),
                    f"maintenance_component_{group}_monitor_reset_count_window": ("is_monitor_reset", "sum"),
                    f"maintenance_component_{group}_min_remaining_hours_window": ("remaining_hours", "min"),
                }
            ).reset_index()
            ag = ag.merge(comp, on="row_id", how="left")

    ag = ag.merge(base[["row_id", "window_end"]], on="row_id", how="left")
    ag["maintenance_days_since_latest_event_window"] = (ag["window_end"] - ag["latest_maintenance_date"]).dt.days.astype(float)
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
    for col in ["smr_hours", "working_hours_clean", "engine_running_hours_clean", "engine_idling_hours_clean", "high_throttle_day_flag"]:
        if col in m.columns:
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


def build_window_features(base_rows: pd.DataFrame, sources: Mapping[str, pd.DataFrame], episodes: pd.DataFrame) -> pd.DataFrame:
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
        prior_count, days = count_claims_before(dates_by_machine, row["machine_key"], row["window_start"])
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
    for c in count_cols:
        if c not in features.columns:
            features[c] = 0
    features["source_record_count_window"] = features[count_cols].fillna(0).sum(axis=1)
    features["has_any_source_window"] = (features["source_record_count_window"] > 0).astype(int)
    features = _fill_numeric_categorical(features, __import__("config"))
    base_no_id = base.drop(columns=["row_id"]).reset_index(drop=True)
    feature_part = features.drop(columns=["row_id"]).reset_index(drop=True)
    overlap = [c for c in feature_part.columns if c in base_no_id.columns]
    if overlap:
        feature_part = feature_part.drop(columns=overlap)
    out = base_no_id.join(feature_part)
    return out


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
