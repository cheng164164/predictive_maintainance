"""Feature engineering for the case-control modeling workflow.

The base feature path remains in cc_utils.py for speed. This module implements
only the frozen feature set carried over from the prior snapshot dataframe
builder. Frozen features are calculated at each row's window_end and use only
source records strictly earlier than that cutoff.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


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
RECENCY_FEATURES = {
    "days_since_last_fault",
    "days_since_last_severe_fault",
    "days_since_last_reset",
    "days_since_last_oil_reset",
    "days_since_last_filter_reset",
    "days_since_last_smr",
    "days_since_last_actual_work_day",
    "days_since_last_engine_running_day",
    "days_since_last_travel_day",
    "days_since_last_claim",
}


def _first_existing(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index)


def _numeric(df: pd.DataFrame, candidates: Iterable[str], default=np.nan) -> pd.Series:
    return pd.to_numeric(_first_existing(df, candidates, default), errors="coerce")


def _boolean(df: pd.DataFrame, candidates: Iterable[str], default: bool = False) -> pd.Series:
    raw = _first_existing(df, candidates, default)
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(default).astype(bool)
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype("string").str.strip().str.lower()
    out = numeric.eq(1) | text.isin({"true", "t", "yes", "y", "1"})
    missing = raw.isna()
    if default:
        out = out | missing
    return out.fillna(default).astype(bool)


def _contains_any(series: pd.Series, patterns: Iterable[str]) -> pd.Series:
    pattern = "|".join(str(p).lower() for p in patterns if str(p).strip())
    if not pattern:
        return pd.Series(False, index=series.index)
    return series.astype("string").str.lower().str.contains(pattern, regex=True, na=False)


def _ratio(num: float, den: float) -> float:
    if den is None or pd.isna(den) or float(den) == 0.0:
        return 0.0
    if num is None or pd.isna(num):
        return 0.0
    return float(num) / float(den)


def _days_between(snapshot_date: pd.Timestamp, event_date) -> float:
    if event_date is None or pd.isna(event_date):
        return np.nan
    return float((pd.Timestamp(snapshot_date) - pd.Timestamp(event_date)).days)


def _sum_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _max_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").max()


def _mean_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").mean()


def _std_col(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").std(ddof=1)


def _normalize_action_label(series: pd.Series, numeric: pd.Series) -> pd.Series:
    text = series.astype("string").str.upper().str.strip()
    extracted = pd.to_numeric(text.str.extract(r"(\d+)", expand=False), errors="coerce")
    level = numeric.fillna(extracted)
    labels = level.map(lambda x: f"L{int(x):02d}" if pd.notna(x) else "")
    return text.where(text.str.match(r"^L\d{2}$", na=False), labels).fillna("")


def _prepare_fault(df: pd.DataFrame) -> pd.DataFrame:
    # Run the same canonicalization path for empty and non-empty inputs. This
    # guarantees that downstream frozen-feature formulas can safely reference
    # every expected column even when a source has no records for the project
    # or for a particular machine.
    f = pd.DataFrame() if df is None else df.copy()
    if "machine_key" not in f.columns:
        f["machine_key"] = pd.Series(index=f.index, dtype="string")
    f["fault_event_date"] = pd.to_datetime(_first_existing(f, ["event_date", "fault_event_date"]), errors="coerce")
    f["fault_code_clean"] = _first_existing(f, ["fault_code", "EVENT_CODE", "event_code"], "").astype("string").str.strip()
    action_num = _numeric(f, ["action_level_num", "ACTION_LEVEL_NUM"], np.nan)
    action_text = _first_existing(f, ["event_action_level", "Action_level", "ACTION_LEVEL"], "")
    f["event_action_level_clean"] = _normalize_action_label(action_text, action_num)
    extracted = pd.to_numeric(f["event_action_level_clean"].str.extract(r"(\d+)", expand=False), errors="coerce")
    f["action_level_num_clean"] = action_num.fillna(extracted)
    f["occurrence_count_clean"] = _numeric(f, ["occurrence_count", "OCCURRENCE_COUNT"], 1).fillna(1)
    f["log_occurrence_clean"] = _numeric(f, ["log_occurrence_count"], np.nan)
    f["log_occurrence_clean"] = f["log_occurrence_clean"].fillna(np.log1p(f["occurrence_count_clean"]))
    f["occurrence_class_clean"] = _numeric(f, ["occurrence_class"], 0).fillna(0)
    f["smr_hours_clean"] = _numeric(f, ["smr_hours", "SMR", "TELEMETRY_SMR"], np.nan)
    f["failure_code_evidence_score_clean"] = _numeric(f, ["failure_code_evidence_score"], np.nan)
    f["evidence_strength_clean"] = _first_existing(f, ["failure_code_evidence_strength_class"], "").astype("string").str.upper().str.strip()
    f["evidence_group_clean"] = _first_existing(f, ["failure_code_evidence_group"], "").astype("string").str.upper().str.strip()
    f["history_category_clean"] = _first_existing(f, ["history_category"], "").astype("string").str.lower()
    f["applicable_component_clean"] = _first_existing(f, ["applicable_component", "applicableComponent"], "").astype("string")
    f["related_component_clean"] = (
        _first_existing(f, ["related_component"], "").astype("string")
        + " "
        + _first_existing(f, ["related_component_1"], "").astype("string")
        + " "
        + f["applicable_component_clean"]
    )
    f["is_mechanical_failure_code_clean"] = _numeric(f, ["is_mechanical_failure_code"], 0).fillna(0)
    f["is_electrical_failure_code_clean"] = _numeric(f, ["is_electrical_failure_code"], 0).fillna(0)
    for component, patterns in COMPONENT_PATTERNS.items():
        f[f"is_component_{component}"] = _contains_any(f["related_component_clean"], patterns)
    f["is_event_evidence"] = f["evidence_group_clean"].eq("EVENT") | f["history_category_clean"].str.contains("event", na=False)
    f["is_context_evidence"] = f["evidence_group_clean"].eq("CONTEXT") | f["history_category_clean"].str.contains("context", na=False)
    return f.dropna(subset=["machine_key", "fault_event_date"]).sort_values(["machine_key", "fault_event_date"])


def _prepare_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame() if df is None else df.copy()
    if "machine_key" not in m.columns:
        m["machine_key"] = pd.Series(index=m.index, dtype="string")
    m["maintenance_event_date"] = pd.to_datetime(_first_existing(m, ["event_date", "maintenance_event_date"]), errors="coerce")
    m["smr_hours_clean"] = _numeric(m, ["smr_hours", "SMR", "TELEMETRY_SMR"], np.nan)
    m["remaining_hours_clean"] = _numeric(m, ["remaining_hours", "REMAINING_HOURS"], np.nan)
    m["is_monitor_reset_clean"] = _boolean(m, ["is_monitor_reset"], False)
    m["is_overdue_clean"] = _boolean(m, ["is_overdue"], False)
    m["is_due_now_clean"] = _boolean(m, ["is_due_now"], False)
    m["available_clean"] = _boolean(m, ["AVAILABLE", "available"], True)
    m["event_name_clean"] = _first_existing(m, ["EVENT_NAME_EN", "event_name"], "").astype("string")
    m["maintenance_type_clean"] = _first_existing(m, ["maintenance_type", "service_types", "SERVICE_TYPES"], "").astype("string")
    m["related_component_clean"] = (
        _first_existing(m, ["related_component"], "").astype("string")
        + " "
        + _first_existing(m, ["related_component_1"], "").astype("string")
        + " "
        + _first_existing(m, ["related_component_2"], "").astype("string")
    )
    for component, patterns in COMPONENT_PATTERNS.items():
        m[f"is_component_{component}"] = _contains_any(m["related_component_clean"], patterns)
    for maintenance_type, patterns in MAINTENANCE_TYPE_PATTERNS.items():
        m[f"is_maintenance_type_{maintenance_type}"] = _contains_any(m["maintenance_type_clean"], patterns)
    return m.dropna(subset=["machine_key", "maintenance_event_date"]).sort_values(["machine_key", "maintenance_event_date"])


def _prepare_operation(df: pd.DataFrame) -> pd.DataFrame:
    o = pd.DataFrame() if df is None else df.copy()
    if "machine_key" not in o.columns:
        o["machine_key"] = pd.Series(index=o.index, dtype="string")
    o["operation_event_date"] = pd.to_datetime(_first_existing(o, ["LOCAL_DATE", "operation_event_date", "event_date"]), errors="coerce").dt.normalize()
    numeric_map = {
        "smr_hours_clean": ["smr_hours"],
        "smr_delta_clean": ["smr_delta_clean_since_prev_obs_hours"],
        "actual_working_hours": ["actual_working_hours_clean", "working_hours_clean"],
        "actual_work_streak": ["actual_work_streak_through_current_day"],
        "engine_running_hours": ["engine_running_hours_clean"],
        "engine_idling_hours": ["engine_idling_hours_clean"],
        "throttle_full_hours": ["throttle_full_hours_clean"],
        "throttle_average_dial_position": ["throttle_average_dial_position_clean"],
        "traveling_hours": ["traveling_hours_clean"],
        "moving_back_forth_hours": ["moving_back_forth_hours_clean"],
        "steering_hours": ["steering_hours_clean"],
        "auto_quick_shift_hours": ["auto_quick_shift_hours_clean"],
        "manual_variable_shift_hours": ["manual_variable_shift_hours_clean"],
    }
    for canonical, candidates in numeric_map.items():
        default = np.nan if canonical in {"smr_hours_clean", "throttle_average_dial_position"} else 0.0
        o[canonical] = _numeric(o, candidates, default).fillna(default if not pd.isna(default) else np.nan)

    # Some source versions provide only throttle_full_share_clean. Derive hours
    # from engine hours when the direct hours field is unavailable.
    if "throttle_full_hours_clean" not in o.columns and "throttle_full_share_clean" in o.columns:
        share = pd.to_numeric(o["throttle_full_share_clean"], errors="coerce").fillna(0)
        o["throttle_full_hours"] = share * pd.to_numeric(o["engine_running_hours"], errors="coerce").fillna(0)

    flag_map = {
        "smr_valid": ["smr_valid_for_utilization_flag"],
        "actual_work_day": ["actual_work_day_flag"],
        "actual_work_valid": ["actual_work_valid_flag"],
        "actual_work_seconds_invalid": ["actual_work_seconds_invalid_flag"],
        "fuel_actual_work_conflict": ["fuel_actual_work_conflict_flag"],
        "engine_running_day": ["engine_running_day_flag"],
        "engine_seconds_valid": ["engine_seconds_valid_flag"],
        "engine_seconds_observed": ["engine_seconds_observed_flag"],
        "throttle_observed": ["throttle_observed_flag"],
        "work_idle_sum_exceeds_engine": ["work_idle_sum_exceeds_engine_flag"],
        "high_throttle_day": ["high_throttle_day_flag"],
        "long_engine_day": ["long_engine_day_flag"],
        "travel_day": ["travel_day_flag"],
        "travel_usable": ["travel_usable_flag"],
    }
    for canonical, candidates in flag_map.items():
        o[canonical] = _boolean(o, candidates, False)

    # Reasonable fallbacks when explicit validity flags are absent.
    if "smr_valid_for_utilization_flag" not in o.columns:
        o["smr_valid"] = o["smr_hours_clean"].notna()
    if "actual_work_valid_flag" not in o.columns:
        o["actual_work_valid"] = True
    if "engine_seconds_valid_flag" not in o.columns:
        o["engine_seconds_valid"] = o["engine_running_hours"].notna()
    if "engine_seconds_observed_flag" not in o.columns:
        o["engine_seconds_observed"] = o["engine_running_hours"].notna()
    if "throttle_observed_flag" not in o.columns:
        o["throttle_observed"] = o["throttle_average_dial_position"].notna() | o["throttle_full_hours"].gt(0)
    if "travel_usable_flag" not in o.columns:
        o["travel_usable"] = o["traveling_hours"].notna()

    o["last_actual_work_date"] = pd.to_datetime(
        _first_existing(o, ["last_actual_work_date_through_current_day"], pd.NaT), errors="coerce"
    ).dt.normalize()
    o = o.dropna(subset=["machine_key", "operation_event_date"]).sort_values(["machine_key", "operation_event_date"])
    if o["last_actual_work_date"].isna().any():
        derived = o["operation_event_date"].where(o["actual_work_day"])
        derived = derived.groupby(o["machine_key"]).ffill()
        o["last_actual_work_date"] = o["last_actual_work_date"].fillna(derived)
    return o


def _prepare_fluid(df: pd.DataFrame, fluid_features: list[str]) -> pd.DataFrame:
    f = pd.DataFrame() if df is None else df.copy()
    if "machine_key" not in f.columns:
        f["machine_key"] = pd.Series(index=f.index, dtype="string")
    f["fluid_sample_event_date"] = pd.to_datetime(_first_existing(f, ["sample_drawn_date", "fluid_sample_event_date"]), errors="coerce").dt.normalize()
    f["fluid_sample_severity_order_clean"] = _numeric(f, ["sample_result_severity_order", "severity_order"], np.nan)
    f["fluid_sample_smr_clean"] = _numeric(f, ["TELEMETRY_SMR_NUMERIC", "telemetry_smr_numeric", "smr_hours"], np.nan)
    for feature in fluid_features:
        f[feature] = _numeric(f, [feature], np.nan)
    return f.dropna(subset=["machine_key", "fluid_sample_event_date"]).sort_values(["machine_key", "fluid_sample_event_date"])


def _prepare_warranty(df: pd.DataFrame) -> pd.DataFrame:
    w = pd.DataFrame() if df is None else df.copy()
    if "machine_key" not in w.columns:
        w["machine_key"] = pd.Series(index=w.index, dtype="string")
    w["warranty_event_date"] = pd.to_datetime(_first_existing(w, ["claim_date", "local_date", "warranty_event_date"]), errors="coerce").dt.normalize()
    w["claim_type_description_clean"] = _first_existing(
        w,
        ["claim_type_description", "claim_type", "claim_category", "warranty_claim_data_source"],
        "",
    ).astype("string").str.strip()
    w["claim_amount_clean"] = _numeric(
        w,
        ["claim_amount", "CLAIM_AMOUNT", "total_claim_amount", "total_amount", "net_claim_amount", "paid_amount"],
        0,
    ).fillna(0)
    return w.dropna(subset=["machine_key", "warranty_event_date"]).sort_values(["machine_key", "warranty_event_date"])


def _fault_rows(snapshot_dates: list[pd.Timestamp], machine_key: str, f: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    dates = f["fault_event_date"] if not f.empty else pd.Series(dtype="datetime64[ns]")
    for snap in snapshot_dates:
        before = f[dates < snap]
        w90 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=90)]
        w30 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=30)]
        w7 = before[before["fault_event_date"] >= snap - pd.Timedelta(days=7)]
        prev30 = before[
            (before["fault_event_date"] >= snap - pd.Timedelta(days=60))
            & (before["fault_event_date"] < snap - pd.Timedelta(days=30))
        ]
        severe = before[before["event_action_level_clean"].isin(["L03", "L04", "L05"])]
        row: dict = {"machine_key": machine_key, "window_end": snap}
        row["fault_count_7d"] = len(w7)
        row["fault_count_30d"] = len(w30)
        row["fault_count_90d"] = len(w90)
        row["fault_count_previous_30d"] = len(prev30)
        row["fault_growth_rate"] = row["fault_count_30d"] - row["fault_count_previous_30d"]
        row["days_since_last_fault"] = _days_between(snap, before["fault_event_date"].max())
        row["days_since_last_severe_fault"] = _days_between(snap, severe["fault_event_date"].max())
        smr_latest = before["smr_hours_clean"].dropna().max()
        prior_smr = before.loc[before["fault_event_date"] <= snap - pd.Timedelta(days=90), "smr_hours_clean"].dropna()
        smr_90_ago = prior_smr.max() if len(prior_smr) else np.nan
        smr_delta = max(float(smr_latest - smr_90_ago), 0.0) if pd.notna(smr_latest) and pd.notna(smr_90_ago) else np.nan
        denominator = max((float(smr_delta) if pd.notna(smr_delta) else 0.0) / 100.0, 1.0)
        row["faults_per_100_hours"] = _ratio(row["fault_count_90d"], denominator)
        row["unique_fault_code_count_90d"] = w90["fault_code_clean"].replace("", np.nan).nunique()
        row["repeat_fault_ratio_90d"] = _ratio(row["fault_count_90d"], max(row["unique_fault_code_count_90d"], 1))
        row["unique_component_count_90d"] = w90["applicable_component_clean"].replace("", np.nan).nunique()
        row["mechanical_fault_count_90d"] = int((w90["is_mechanical_failure_code_clean"] == 1).sum())
        row["mechanical_fault_count_30d"] = int((w30["is_mechanical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_90d"] = int((w90["is_electrical_failure_code_clean"] == 1).sum())
        row["electrical_fault_count_30d"] = int((w30["is_electrical_failure_code_clean"] == 1).sum())
        for level in ["L01", "L02", "L03", "L04"]:
            row[f"action_{level}_count_90d"] = int((w90["event_action_level_clean"] == level).sum())
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
        component_map = {
            "engine": "engine_fault_count_90d",
            "hydraulic": "hydraulic_fault_count_90d",
            "powertrain": "powertrain_fault_count_90d",
            "scr": "scr_fault_count_90d",
            "workequipment": "workequipment_fault_count_90d",
            "cooling": "cooling_fault_count_90d",
        }
        component_counts = []
        for component, feature in component_map.items():
            count = int(w90[f"is_component_{component}"].sum()) if f"is_component_{component}" in w90 else 0
            row[feature] = count
            component_counts.append(count)
        row["top_component_fault_ratio_90d"] = _ratio(max(component_counts) if component_counts else 0, row["fault_count_90d"])
        rows.append(row)
    return rows


def _maintenance_rows(snapshot_dates: list[pd.Timestamp], machine_key: str, m: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    dates = m["maintenance_event_date"] if not m.empty else pd.Series(dtype="datetime64[ns]")
    for snap in snapshot_dates:
        before = m[dates < snap]
        w180 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=180)]
        w90 = before[before["maintenance_event_date"] >= snap - pd.Timedelta(days=90)]
        reset180 = w180[w180["is_monitor_reset_clean"]]
        reset90 = w90[w90["is_monitor_reset_clean"]]
        if len(before):
            current = before.sort_values("maintenance_event_date").groupby("event_name_clean", dropna=False).tail(1)
            current = current[current["available_clean"]]
        else:
            current = before
        row: dict = {"machine_key": machine_key, "window_end": snap}
        row["maintenance_events_180d"] = len(w180)
        row["monitor_reset_count_180d"] = len(reset180)
        row["maintenance_reset_ratio_180d"] = _ratio(len(reset180), len(w180))
        row["maintenance_events_90d"] = len(w90)
        row["monitor_reset_count_90d"] = len(reset90)
        row["active_maintenance_items"] = len(current)
        row["overdue_item_count"] = int(current["is_overdue_clean"].sum()) if len(current) else 0
        row["due_now_item_count"] = int(current["is_due_now_clean"].sum()) if len(current) else 0
        if len(current):
            row["overdue_item_count"] = max(row["overdue_item_count"], int((current["remaining_hours_clean"] < 0).sum()))
            row["due_now_item_count"] = max(row["due_now_item_count"], int((current["remaining_hours_clean"] == 0).sum()))
        row["maintenance_due_or_overdue_ratio"] = _ratio(
            row["due_now_item_count"] + row["overdue_item_count"], row["active_maintenance_items"]
        )
        row["avg_remaining_hours"] = current["remaining_hours_clean"].mean() if len(current) else np.nan
        row["min_remaining_hours"] = current["remaining_hours_clean"].min() if len(current) else np.nan
        reset_map = {
            "engine": "engine_reset_count_180d",
            "powertrain": "transmission_reset_count_180d",
            "final_drive": "final_drive_reset_count_180d",
            "cooling": "cooling_system_reset_count_180d",
            "scr": "urea_scr_system_reset_count_180d",
        }
        overdue_map = {
            "engine": "engine_overdue_item_count",
            "powertrain": "transmission_overdue_item_count",
            "final_drive": "final_drive_overdue_item_count",
            "cooling": "cooling_system_overdue_item_count",
            "scr": "urea_scr_system_overdue_item_count",
        }
        for component, feature in reset_map.items():
            row[feature] = int(reset180[f"is_component_{component}"].sum()) if f"is_component_{component}" in reset180 else 0
        for component, feature in overdue_map.items():
            row[feature] = int(current.loc[current["is_overdue_clean"], f"is_component_{component}"].sum()) if len(current) else 0
        for maintenance_type in MAINTENANCE_TYPE_PATTERNS:
            row[f"{maintenance_type}_reset_count_180d"] = int(reset180[f"is_maintenance_type_{maintenance_type}"].sum())
        row["unique_maintenance_type_count_180d"] = reset180["maintenance_type_clean"].replace("", np.nan).nunique()
        row["days_since_last_reset"] = _days_between(snap, reset180["maintenance_event_date"].max())
        oil_reset = reset180[reset180["is_maintenance_type_oil"]]
        filter_reset = reset180[reset180["is_maintenance_type_filter"]]
        row["days_since_last_oil_reset"] = _days_between(snap, oil_reset["maintenance_event_date"].max())
        row["days_since_last_filter_reset"] = _days_between(snap, filter_reset["maintenance_event_date"].max())
        latest_smr = before["smr_hours_clean"].dropna().max()
        last_reset = reset180.sort_values("maintenance_event_date").tail(1)
        last_reset_smr = last_reset["smr_hours_clean"].iloc[0] if len(last_reset) else np.nan
        row["smr_since_last_reset"] = float(latest_smr - last_reset_smr) if pd.notna(latest_smr) and pd.notna(last_reset_smr) else np.nan
        rows.append(row)
    return rows


def _operation_rows(snapshot_dates: list[pd.Timestamp], machine_key: str, o: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    dates = o["operation_event_date"] if not o.empty else pd.Series(dtype="datetime64[ns]")
    for snap in snapshot_dates:
        before = o[dates < snap]
        w90 = before[before["operation_event_date"] >= snap - pd.Timedelta(days=90)]
        w30 = before[before["operation_event_date"] >= snap - pd.Timedelta(days=30)]
        w7 = before[before["operation_event_date"] >= snap - pd.Timedelta(days=7)]
        row: dict = {"machine_key": machine_key, "window_end": snap}
        valid_smr = before[before["smr_valid"] & before["smr_hours_clean"].notna()].sort_values("operation_event_date")
        if len(valid_smr):
            row["smr_latest_hours"] = valid_smr["smr_hours_clean"].iloc[-1]
            row["days_since_last_smr"] = _days_between(snap, valid_smr["operation_event_date"].iloc[-1])
        else:
            row["smr_latest_hours"] = np.nan
            row["days_since_last_smr"] = np.nan
        row["smr_delta_7d"] = _sum_col(w7, "smr_delta_clean")
        row["smr_delta_30d"] = _sum_col(w30, "smr_delta_clean")
        row["smr_delta_90d"] = _sum_col(w90, "smr_delta_clean")

        work_sum_7 = _sum_col(w7, "actual_working_hours")
        work_sum_30 = _sum_col(w30, "actual_working_hours")
        work_sum_90 = _sum_col(w90, "actual_working_hours")
        work_day_7 = _sum_col(w7, "actual_work_day")
        work_day_30 = _sum_col(w30, "actual_work_day")
        work_day_90 = _sum_col(w90, "actual_work_day")
        work_valid_30 = _sum_col(w30, "actual_work_valid")
        work_valid_90 = _sum_col(w90, "actual_work_valid")
        row["working_hours_sum_7d"] = work_sum_7
        row["working_hours_sum_30d"] = work_sum_30
        row["working_hours_sum_90d"] = work_sum_90
        row["actual_work_day_count_7d"] = work_day_7
        row["actual_work_day_count_30d"] = work_day_30
        row["actual_work_day_count_90d"] = work_day_90
        row["actual_work_day_ratio_30d"] = _ratio(work_day_30, work_valid_30)
        row["actual_work_day_ratio_90d"] = _ratio(work_day_90, work_valid_90)
        row["actual_work_day_ratio_change_30d_vs_90d"] = row["actual_work_day_ratio_30d"] - row["actual_work_day_ratio_90d"]
        row["actual_work_valid_flag"] = work_valid_90
        row["working_hours_rate_change_30d_vs_90d"] = (work_sum_30 / 30.0) - (work_sum_90 / 90.0)
        row["avg_working_hours_per_actual_work_day_30d"] = _ratio(work_sum_30, work_day_30)
        row["avg_working_hours_per_actual_work_day_90d"] = _ratio(work_sum_90, work_day_90)
        row["max_working_hours_day_90d"] = _max_col(w90, "actual_working_hours")
        active_work = w90[w90["actual_work_day"]]
        row["working_hours_stddev_actual_work_day_90d"] = _std_col(active_work, "actual_working_hours")
        row["actual_work_seconds_invalid_count_90d"] = _sum_col(w90, "actual_work_seconds_invalid")
        row["fuel_actual_work_conflict_count_90d"] = _sum_col(w90, "fuel_actual_work_conflict")
        latest = before.sort_values("operation_event_date").tail(1)
        if len(latest):
            row["days_since_last_actual_work_day"] = _days_between(snap, latest["last_actual_work_date"].iloc[0])
            row["current_actual_work_streak_days"] = latest["actual_work_streak"].iloc[0] if _days_between(snap, latest["operation_event_date"].iloc[0]) == 1 else 0
        else:
            row["days_since_last_actual_work_day"] = np.nan
            row["current_actual_work_streak_days"] = 0

        engine_sum_7 = _sum_col(w7, "engine_running_hours")
        engine_sum_30 = _sum_col(w30, "engine_running_hours")
        engine_sum_90 = _sum_col(w90, "engine_running_hours")
        engine_day_30 = _sum_col(w30, "engine_running_day")
        engine_day_90 = _sum_col(w90, "engine_running_day")
        engine_valid_30 = _sum_col(w30, "engine_seconds_valid")
        engine_valid_90 = _sum_col(w90, "engine_seconds_valid")
        throttle_full_30 = _sum_col(w30, "throttle_full_hours")
        throttle_full_90 = _sum_col(w90, "throttle_full_hours")
        row["engine_running_hours_sum_7d"] = engine_sum_7
        row["engine_running_hours_sum_30d"] = engine_sum_30
        row["engine_running_hours_sum_90d"] = engine_sum_90
        row["engine_running_day_count_30d"] = engine_day_30
        row["engine_running_day_count_90d"] = engine_day_90
        row["engine_running_day_ratio_30d"] = _ratio(engine_day_30, engine_valid_30)
        row["engine_running_day_ratio_90d"] = _ratio(engine_day_90, engine_valid_90)
        row["engine_running_rate_change_30d_vs_90d"] = (engine_sum_30 / 30.0) - (engine_sum_90 / 90.0)
        row["avg_engine_running_hours_per_engine_day_90d"] = _ratio(engine_sum_90, engine_day_90)
        engine_active_90 = w90[w90["engine_running_day"]]
        engine_active_30 = w30[w30["engine_running_day"]]
        row["avg_throttle_dial_position_active_30d"] = _mean_col(engine_active_30, "throttle_average_dial_position")
        row["avg_throttle_dial_position_active_90d"] = _mean_col(engine_active_90, "throttle_average_dial_position")
        row["days_since_last_engine_running_day"] = _days_between(
            snap, before.loc[before["engine_running_day"], "operation_event_date"].max()
        )
        row["engine_idling_share_90d"] = _ratio(_sum_col(w90, "engine_idling_hours"), engine_sum_90)
        row["throttle_full_hours_sum_90d"] = throttle_full_90
        row["throttle_full_engine_share_30d"] = _ratio(throttle_full_30, engine_sum_30)
        row["throttle_full_engine_share_90d"] = _ratio(throttle_full_90, engine_sum_90)
        row["throttle_full_share_change_30d_vs_90d"] = row["throttle_full_engine_share_30d"] - row["throttle_full_engine_share_90d"]
        row["engine_observed_day_count_90d"] = _sum_col(w90, "engine_seconds_observed")
        row["throttle_observed_day_count_90d"] = _sum_col(w90, "throttle_observed")
        row["work_idle_sum_exceeds_engine_count_90d"] = _sum_col(w90, "work_idle_sum_exceeds_engine")
        row["engine_running_hours_max_day_90d"] = _max_col(w90, "engine_running_hours")
        row["engine_running_hours_stddev_engine_day_90d"] = _std_col(engine_active_90, "engine_running_hours")
        row["high_throttle_day_count_90d"] = _sum_col(w90, "high_throttle_day")
        row["long_engine_day_count_90d"] = _sum_col(w90, "long_engine_day")

        travel_sum_30 = _sum_col(w30, "traveling_hours")
        travel_sum_90 = _sum_col(w90, "traveling_hours")
        travel_day_30 = _sum_col(w30, "travel_day")
        travel_day_90 = _sum_col(w90, "travel_day")
        travel_observed_30 = _sum_col(w30, "travel_usable") or len(w30)
        travel_observed_90 = _sum_col(w90, "travel_usable") or len(w90)
        moving_90 = _sum_col(w90, "moving_back_forth_hours")
        steering_90 = _sum_col(w90, "steering_hours")
        row["travel_hours_sum_30d"] = travel_sum_30
        row["travel_hours_sum_90d"] = travel_sum_90
        row["travel_day_count_30d"] = travel_day_30
        row["travel_day_count_90d"] = travel_day_90
        row["avg_travel_hours_per_travel_day_30d"] = _ratio(travel_sum_30, travel_day_30)
        row["avg_travel_hours_per_travel_day_90d"] = _ratio(travel_sum_90, travel_day_90)
        row["days_since_last_travel_day"] = _days_between(
            snap, before.loc[before["travel_day"], "operation_event_date"].max()
        )
        row["moving_back_forth_hours_sum_90d"] = moving_90
        row["steering_hours_sum_90d"] = steering_90
        row["moving_back_forth_to_travel_ratio_90d"] = _ratio(moving_90, travel_sum_90)
        row["travel_day_ratio_observed_30d"] = _ratio(travel_day_30, travel_observed_30)
        row["travel_day_ratio_observed_90d"] = _ratio(travel_day_90, travel_observed_90)
        row["travel_rate_change_30d_vs_90d"] = (travel_sum_30 / 30.0) - (travel_sum_90 / 90.0)
        row["has_travel_data_90d"] = int(travel_observed_90 > 0)
        row["travel_share_of_working_hours_90d"] = _ratio(travel_sum_90, work_sum_90)
        row["steering_to_travel_ratio_90d"] = _ratio(steering_90, travel_sum_90)
        row["auto_quick_shift_hours_sum_90d"] = _sum_col(w90, "auto_quick_shift_hours")
        row["manual_variable_shift_hours_sum_90d"] = _sum_col(w90, "manual_variable_shift_hours")
        rows.append(row)
    return rows


def _latest_non_null_by_date(window: pd.DataFrame, feature: str) -> float:
    if window.empty or feature not in window.columns:
        return np.nan
    values = window[["fluid_sample_event_date", feature]].dropna(subset=[feature])
    if values.empty:
        return np.nan
    by_date = values.groupby("fluid_sample_event_date", dropna=False)[feature].max().sort_index()
    return by_date.iloc[-1] if len(by_date) else np.nan


def _fluid_rows(
    snapshot_dates: list[pd.Timestamp],
    machine_key: str,
    f: pd.DataFrame,
    fluid_features: list[str],
    lookback_days: int,
) -> list[dict]:
    rows: list[dict] = []
    dates = f["fluid_sample_event_date"] if not f.empty else pd.Series(dtype="datetime64[ns]")
    for snap in snapshot_dates:
        before = f[dates < snap]
        window = before[before["fluid_sample_event_date"] >= snap - pd.Timedelta(days=lookback_days)]
        row: dict = {"machine_key": machine_key, "window_end": snap}
        for feature in fluid_features:
            row[feature] = _latest_non_null_by_date(window, feature)
        rows.append(row)
    return rows


def _warranty_rows(snapshot_dates: list[pd.Timestamp], machine_key: str, w: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    dates = w["warranty_event_date"] if not w.empty else pd.Series(dtype="datetime64[ns]")
    for snap in snapshot_dates:
        before = w[dates < snap]
        w365 = before[before["warranty_event_date"] >= snap - pd.Timedelta(days=365)]
        w180 = before[before["warranty_event_date"] >= snap - pd.Timedelta(days=180)]
        w90 = before[before["warranty_event_date"] >= snap - pd.Timedelta(days=90)]
        rows.append(
            {
                "machine_key": machine_key,
                "window_end": snap,
                "prior_claim_count_365d": len(w365),
                "prior_claim_count_180d": len(w180),
                "prior_claim_count_90d": len(w90),
                "days_since_last_claim": _days_between(snap, before["warranty_event_date"].max()),
                "prior_claim_amount_sum_365d": w365["claim_amount_clean"].sum(),
                "prior_claim_amount_max_365d": w365["claim_amount_clean"].max(),
                "unique_claim_type_count_365d": w365["claim_type_description_clean"].replace("", np.nan).nunique(),
                "has_prior_claim_365d": int(len(w365) > 0),
            }
        )
    return rows


def build_frozen_window_features(
    base_rows: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    config,
) -> pd.DataFrame:
    """Return base rows plus the frozen snapshot feature set."""
    base = base_rows.copy().reset_index(drop=True)
    base["window_end"] = pd.to_datetime(base["window_end"], errors="coerce")
    if base["window_end"].isna().any():
        raise ValueError("Frozen feature engineering found missing/unparseable window_end values.")

    snapshots = (
        base[["machine_key", "window_end"]]
        .drop_duplicates()
        .sort_values(["machine_key", "window_end"], kind="mergesort")
        .reset_index(drop=True)
    )
    fluid_features = [
        c
        for c in config.FROZEN_NUMERIC_FEATURES
        if c
        in {
            "Ag_Silver_PPM", "Al_Aluminum_PPM", "Cr_Chromium_PPM", "Cu_Copper_PPM",
            "Fe_Iron_PPM", "Ni_Nickel_PPM", "Pb_Lead_PPM", "Sn_Tin_PPM",
            "Ti_Titanium_PPM", "V_Vanadium_PPM", "EthyleneGlycol_Ethylene_Glycol_PERCENT",
            "Fuel_Fuel_PERCENT", "Gly_Glycol_PERCENT", "K_Potassium_PPM", "Li_Lithium_PPM",
            "Na_Sodium_PPM", "PolypropyleneGlycol_Polypropylene_Glycol_PERCENT",
            "Sediment_Sediment_MG_PER_L", "Si_Silicon_PPM", "Solids_Solids_PERCENT",
            "Soot_Soot_Abs", "Soot_Soot_Abs_cm", "Soot_Soot_METHOD_DEPENDENT",
            "Soot_Soot_PERCENT", "Water_Water_PERCENT",
        }
    ]

    fault = _prepare_fault(sources.get("fault", pd.DataFrame()))
    maintenance = _prepare_maintenance(sources.get("maintenance", pd.DataFrame()))
    operation = _prepare_operation(sources.get("operation", pd.DataFrame()))
    fluid = _prepare_fluid(sources.get("fluid", pd.DataFrame()), fluid_features)
    warranty = _prepare_warranty(sources.get("warranty", pd.DataFrame()))

    groups = {
        "fault": {k: g for k, g in fault.groupby("machine_key", sort=False)},
        "maintenance": {k: g for k, g in maintenance.groupby("machine_key", sort=False)},
        "operation": {k: g for k, g in operation.groupby("machine_key", sort=False)},
        "fluid": {k: g for k, g in fluid.groupby("machine_key", sort=False)},
        "warranty": {k: g for k, g in warranty.groupby("machine_key", sort=False)},
    }
    empty = {
        "fault": fault.iloc[0:0],
        "maintenance": maintenance.iloc[0:0],
        "operation": operation.iloc[0:0],
        "fluid": fluid.iloc[0:0],
        "warranty": warranty.iloc[0:0],
    }

    feature_rows: list[dict] = []
    for machine_key, snap_group in snapshots.groupby("machine_key", sort=False):
        dates = list(pd.to_datetime(snap_group["window_end"]).sort_values())
        parts = [
            pd.DataFrame(_fault_rows(dates, machine_key, groups["fault"].get(machine_key, empty["fault"]))),
            pd.DataFrame(_maintenance_rows(dates, machine_key, groups["maintenance"].get(machine_key, empty["maintenance"]))),
            pd.DataFrame(_operation_rows(dates, machine_key, groups["operation"].get(machine_key, empty["operation"]))),
            pd.DataFrame(
                _fluid_rows(
                    dates,
                    machine_key,
                    groups["fluid"].get(machine_key, empty["fluid"]),
                    fluid_features,
                    int(getattr(config, "FLUID_SAMPLE_LOOKBACK_DAYS", 365)),
                )
            ),
            pd.DataFrame(_warranty_rows(dates, machine_key, groups["warranty"].get(machine_key, empty["warranty"]))),
        ]
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on=["machine_key", "window_end"], how="outer")
        feature_rows.extend(merged.to_dict(orient="records"))

    features = pd.DataFrame(feature_rows)
    if features.empty:
        features = snapshots.copy()
    for col in config.FROZEN_NUMERIC_FEATURES:
        if col not in features.columns:
            features[col] = np.nan
        values = pd.to_numeric(features[col], errors="coerce")
        features[col] = values.fillna(9999.0 if col in RECENCY_FEATURES else 0.0).astype(float)

    if "full_model" not in base.columns:
        base["full_model"] = "unknown"
    base["full_model"] = base["full_model"].astype("string").fillna("unknown").replace("", "unknown")
    overlap = [c for c in config.FROZEN_NUMERIC_FEATURES if c in base.columns]
    if overlap:
        base = base.drop(columns=overlap)
    out = base.merge(
        features[["machine_key", "window_end"] + list(config.FROZEN_NUMERIC_FEATURES)],
        on=["machine_key", "window_end"],
        how="left",
        validate="many_to_one",
    )
    for col in config.FROZEN_NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(9999.0 if col in RECENCY_FEATURES else 0.0)
    return out
