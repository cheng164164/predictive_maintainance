"""Build exact 90-day segment and arbitrary-anchor condition features.

The feature definitions intentionally mirror the uploaded future-claim model's
aggregate feature engineering.  The only timing change is that the original
fixed January/May/September four-month buckets are replaced by exact 90-day
windows.  The same aggregation code is used for fixed training segments and for
arbitrary deployment-style anchor dates.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import config as cfg
from data_utils import add_category_indicators, make_machine_key, parse_bool, safe_divide

FIXED_KEYS = ["machine_key", "period_start"]
ANCHOR_KEYS = ["machine_key", "period_start"]


@dataclass
class FeatureBuildResult:
    segment_features: pd.DataFrame
    anchor_features: pd.DataFrame
    roster: pd.DataFrame
    source_audit: pd.DataFrame


def _period_start_for_date(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    start = pd.Timestamp(cfg.ANALYSIS_START)
    day_offset = (parsed - start).dt.days
    index = np.floor_divide(day_offset, int(cfg.SEGMENT_STRIDE_DAYS))
    result = start + pd.to_timedelta(index * int(cfg.SEGMENT_STRIDE_DAYS), unit="D")
    return result.where(day_offset.ge(0))


def _period_end(period_start: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(period_start) + pd.Timedelta(days=int(cfg.LOOKBACK_DAYS) - 1)


def _season_label(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates)
    return "Q" + parsed.dt.quarter.astype(str)


def _read_identity(path: Path, model_col: str, serial_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[model_col, serial_col], dtype=str, low_memory=False)
    frame["machine_key"] = make_machine_key(frame[model_col], frame[serial_col])
    frame["full_model"] = frame[model_col].astype("string").str.strip().str.upper()
    return frame[["machine_key", "full_model"]].dropna().drop_duplicates()


def build_machine_roster() -> pd.DataFrame:
    """Reconstruct the fleet from the union of condition-source identities."""

    parts = [
        _read_identity(cfg.FAULT_CODES_PATH, "full_model", "serial_number"),
        _read_identity(cfg.FLUID_SAMPLES_PATH, "FULL_MODEL", "SERIAL"),
        _read_identity(cfg.MAINTENANCE_PATH, "full_model", "SERIAL"),
        _read_identity(cfg.OPERATION_PATH, "full_model", "SERIAL"),
    ]
    combined = pd.concat(parts, ignore_index=True).dropna(subset=["machine_key"])
    model_counts = (
        combined.groupby(["machine_key", "full_model"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["machine_key", "rows", "full_model"], ascending=[True, False, True])
        .drop_duplicates("machine_key")
    )
    roster = model_counts[["machine_key", "full_model"]].sort_values("machine_key").reset_index(drop=True)
    return roster


def build_segment_grid(roster: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(cfg.ANALYSIS_START)
    end = pd.Timestamp(cfg.ANALYSIS_END)
    latest_start = end - pd.Timedelta(days=int(cfg.LOOKBACK_DAYS) - 1)
    starts = pd.date_range(start, latest_start, freq=f"{int(cfg.SEGMENT_STRIDE_DAYS)}D")
    grid = roster.assign(_key=1).merge(
        pd.DataFrame({"period_start": starts, "_key": 1}), on="_key", how="inner"
    ).drop(columns="_key")
    grid["period_end"] = _period_end(grid["period_start"])
    grid["segment_season"] = _season_label(grid["period_start"])
    grid["segment_year"] = grid["period_start"].dt.year.astype(int)
    return grid.sort_values(FIXED_KEYS, kind="mergesort").reset_index(drop=True)


def build_anchor_plan() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold, validation_start, validation_end in cfg.VALIDATION_FOLDS:
        for sequence, anchor in enumerate(cfg.ANCHOR_DATES_BY_FOLD[fold], start=1):
            anchor_ts = pd.Timestamp(anchor)
            rows.append(
                {
                    "fold": fold,
                    "anchor_sequence": sequence,
                    "anchor_id": f"{fold}_A{sequence:02d}",
                    "anchor_date": anchor_ts,
                    "validation_start": pd.Timestamp(validation_start),
                    "validation_end_exclusive": pd.Timestamp(validation_end),
                    "period_start": anchor_ts - pd.Timedelta(days=int(cfg.LOOKBACK_DAYS)),
                    "period_end": anchor_ts - pd.Timedelta(days=1),
                }
            )
    return pd.DataFrame(rows).sort_values(["fold", "anchor_date"]).reset_index(drop=True)


def build_anchor_grid(roster: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    grid = roster.assign(_key=1).merge(plan.assign(_key=1), on="_key", how="inner").drop(columns="_key")
    grid["segment_season"] = _season_label(grid["period_start"])
    grid["segment_year"] = grid["period_start"].dt.year.astype(int)
    return grid.sort_values(["anchor_date", "machine_key"], kind="mergesort").reset_index(drop=True)


def _prepare_common(
    df: pd.DataFrame,
    *,
    date_col: str,
    model_col: str,
    serial_col: str,
) -> pd.DataFrame:
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["machine_key"] = make_machine_key(frame[model_col], frame[serial_col])
    min_date = min(
        pd.Timestamp(cfg.ANALYSIS_START),
        min(pd.Timestamp(x) for dates in cfg.ANCHOR_DATES_BY_FOLD.values() for x in dates)
        - pd.Timedelta(days=int(cfg.LOOKBACK_DAYS)),
    )
    max_date = max(
        pd.Timestamp(cfg.ANALYSIS_END),
        max(pd.Timestamp(x) for dates in cfg.ANCHOR_DATES_BY_FOLD.values() for x in dates)
        - pd.Timedelta(days=1),
    )
    frame = frame[
        frame[date_col].between(min_date, max_date)
        & frame["machine_key"].notna()
    ].copy()
    frame["period_start"] = _period_start_for_date(frame[date_col])
    fixed_end = _period_end(frame["period_start"])
    frame["eligible_fixed_segment"] = (
        frame["period_start"].notna()
        & frame[date_col].le(fixed_end)
        & fixed_end.le(pd.Timestamp(cfg.ANALYSIS_END))
    )
    return frame


def _aggregate_fixed_and_anchors(
    prepared: pd.DataFrame,
    *,
    date_col: str,
    aggregate: Callable[[pd.DataFrame, list[str]], pd.DataFrame],
    anchor_plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_input = prepared[prepared["eligible_fixed_segment"]].copy()
    fixed = aggregate(fixed_input, FIXED_KEYS)

    anchor_parts: list[pd.DataFrame] = []
    for row in anchor_plan.itertuples(index=False):
        subset = prepared[
            prepared[date_col].ge(row.period_start)
            & prepared[date_col].le(row.period_end)
        ].copy()
        if subset.empty:
            continue
        subset["period_start"] = pd.Timestamp(row.period_start)
        agg = aggregate(subset, ANCHOR_KEYS)
        agg["anchor_date"] = pd.Timestamp(row.anchor_date)
        anchor_parts.append(agg)
    anchor = pd.concat(anchor_parts, ignore_index=True, sort=False) if anchor_parts else pd.DataFrame()
    return fixed, anchor


def _prepare_fault() -> pd.DataFrame:
    usecols = [
        "full_model", "serial_number", "event_date", "fault_code",
        "occurrence_count", "log_occurrence_count", "engine_on_hours", "smr_hours",
        "is_mechanical_failure_code", "is_electrical_failure_code", "action_level_num",
        "failure_code_evidence_score", "event_action_level",
        "failure_code_evidence_strength_class", "failure_code_evidence_group",
        "logical_name", "related_component",
    ]
    df = pd.read_csv(cfg.FAULT_CODES_PATH, usecols=usecols, low_memory=False)
    df = _prepare_common(df, date_col="event_date", model_col="full_model", serial_col="serial_number")
    numeric = [
        "occurrence_count", "log_occurrence_count", "engine_on_hours", "smr_hours",
        "is_mechanical_failure_code", "is_electrical_failure_code", "action_level_num",
        "failure_code_evidence_score",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    indicator_cols: list[str] = []
    indicator_cols += add_category_indicators(df, "event_action_level", "fault_action", ["L00", "L01", "L02", "L03", "L04"])
    indicator_cols += add_category_indicators(df, "failure_code_evidence_strength_class", "fault_strength", ["VERY_WEAK", "WEAK", "MEDIUM", "STRONG"])
    indicator_cols += add_category_indicators(df, "failure_code_evidence_group", "fault_group", ["CONTEXT", "EVENT", "OTHER"])
    indicator_cols += add_category_indicators(df, "logical_name", "fault_logical", ["MON", "ENG/M", "W/E", "HST", "P/T"])
    indicator_cols += add_category_indicators(
        df, "related_component", "fault_component",
        ["Urea SCR System", "Engine", "Power Train System", "Work Equipment System", "Cooling System", "Control System"],
    )
    df.attrs["indicator_cols"] = indicator_cols
    return df


def _aggregate_fault(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    indicator_cols = [c for c in df.columns if c.startswith(("fault_action_", "fault_strength_", "fault_group_", "fault_logical_", "fault_component_")) and c.endswith("_count")]
    agg_spec: dict[str, tuple[str, str]] = {
        "fault_event_count": ("fault_code", "size"),
        "fault_active_days": ("event_date", "nunique"),
        "fault_unique_code_count": ("fault_code", "nunique"),
        "fault_occurrence_sum": ("occurrence_count", "sum"),
        "fault_occurrence_max": ("occurrence_count", "max"),
        "fault_log_occurrence_sum": ("log_occurrence_count", "sum"),
        "fault_log_occurrence_mean": ("log_occurrence_count", "mean"),
        "fault_engine_on_hours_sum": ("engine_on_hours", "sum"),
        "fault_engine_on_hours_max": ("engine_on_hours", "max"),
        "fault_smr_min": ("smr_hours", "min"),
        "fault_smr_max": ("smr_hours", "max"),
        "fault_mechanical_count": ("is_mechanical_failure_code", "sum"),
        "fault_electrical_count": ("is_electrical_failure_code", "sum"),
        "fault_action_level_mean": ("action_level_num", "mean"),
        "fault_action_level_max": ("action_level_num", "max"),
        "fault_evidence_score_sum": ("failure_code_evidence_score", "sum"),
        "fault_evidence_score_mean": ("failure_code_evidence_score", "mean"),
        "fault_evidence_score_max": ("failure_code_evidence_score", "max"),
    }
    for col in indicator_cols:
        agg_spec[col] = (col, "sum")
    agg = df.groupby(keys, as_index=False).agg(**agg_spec)
    agg["fault_smr_range"] = agg["fault_smr_max"] - agg["fault_smr_min"]
    agg["fault_events_per_active_day"] = safe_divide(agg["fault_event_count"], agg["fault_active_days"])
    agg["fault_strong_rate"] = safe_divide(agg.get("fault_strength_strong_count", 0), agg["fault_event_count"])
    agg["fault_medium_or_strong_rate"] = safe_divide(
        agg.get("fault_strength_medium_count", 0) + agg.get("fault_strength_strong_count", 0),
        agg["fault_event_count"],
    )
    agg["fault_action_ge3_rate"] = safe_divide(
        agg.get("fault_action_l03_count", 0) + agg.get("fault_action_l04_count", 0),
        agg["fault_event_count"],
    )
    return agg


def _prepare_maintenance() -> pd.DataFrame:
    usecols = [
        "full_model", "SERIAL", "event_date", "EVENT_NAME_ID", "smr_hours",
        "previous_reset_smr_hours", "remaining_hours", "INTERVAL_HOURS", "THRESHOLD_HOURS",
        "is_monitor_reset", "is_overdue", "is_due_now", "is_notice_or_status", "AVAILABLE",
        "maintenance_type", "related_component",
    ]
    df = pd.read_csv(cfg.MAINTENANCE_PATH, usecols=usecols, low_memory=False)
    df = _prepare_common(df, date_col="event_date", model_col="full_model", serial_col="SERIAL")
    for col in ["smr_hours", "previous_reset_smr_hours", "remaining_hours", "INTERVAL_HOURS", "THRESHOLD_HOURS"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["is_monitor_reset", "is_overdue", "is_due_now", "is_notice_or_status", "AVAILABLE"]:
        df[col] = parse_bool(df[col])
    df["remaining_negative"] = df["remaining_hours"].lt(0).astype("int16")
    df["remaining_zero"] = df["remaining_hours"].eq(0).astype("int16")
    df["maintenance_unavailable"] = df["AVAILABLE"].eq(0).astype("int16")
    add_category_indicators(df, "maintenance_type", "maint_type", ["Filter", "Oil", "Breather", "Cleaning", "Coolant"])
    add_category_indicators(df, "related_component", "maint_component", ["Engine", "Urea SCR System", "Transmission", "Final Drive", "Cooling System"])
    return df


def _aggregate_maintenance(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    indicator_cols = [c for c in df.columns if c.startswith(("maint_type_", "maint_component_")) and c.endswith("_count")]
    agg_spec: dict[str, tuple[str, str]] = {
        "maint_event_count": ("EVENT_NAME_ID", "size"),
        "maint_active_days": ("event_date", "nunique"),
        "maint_unique_event_count": ("EVENT_NAME_ID", "nunique"),
        "maint_monitor_reset_count": ("is_monitor_reset", "sum"),
        "maint_overdue_count": ("is_overdue", "sum"),
        "maint_due_now_count": ("is_due_now", "sum"),
        "maint_notice_status_count": ("is_notice_or_status", "sum"),
        "maint_unavailable_count": ("maintenance_unavailable", "sum"),
        "maint_remaining_negative_count": ("remaining_negative", "sum"),
        "maint_remaining_zero_count": ("remaining_zero", "sum"),
        "maint_remaining_hours_min": ("remaining_hours", "min"),
        "maint_remaining_hours_mean": ("remaining_hours", "mean"),
        "maint_remaining_hours_max": ("remaining_hours", "max"),
        "maint_interval_hours_min": ("INTERVAL_HOURS", "min"),
        "maint_interval_hours_mean": ("INTERVAL_HOURS", "mean"),
        "maint_interval_hours_max": ("INTERVAL_HOURS", "max"),
        "maint_threshold_hours_mean": ("THRESHOLD_HOURS", "mean"),
        "maint_smr_min": ("smr_hours", "min"),
        "maint_smr_max": ("smr_hours", "max"),
        "maint_previous_reset_smr_mean": ("previous_reset_smr_hours", "mean"),
    }
    for col in indicator_cols:
        agg_spec[col] = (col, "sum")
    agg = df.groupby(keys, as_index=False).agg(**agg_spec)
    agg["maint_smr_range"] = agg["maint_smr_max"] - agg["maint_smr_min"]
    agg["maint_overdue_rate"] = safe_divide(agg["maint_overdue_count"], agg["maint_event_count"])
    agg["maint_due_now_rate"] = safe_divide(agg["maint_due_now_count"], agg["maint_event_count"])
    agg["maint_reset_rate"] = safe_divide(agg["maint_monitor_reset_count"], agg["maint_event_count"])
    return agg


def _prepare_fluid() -> pd.DataFrame:
    base_cols = [
        "FULL_MODEL", "SERIAL", "sample_drawn_date", "LABS_SAMPLE_NUMBER",
        "sample_result_severity_order", "TELEMETRY_SMR_NUMERIC",
    ]
    analytes = [
        "Ag_Silver_PPM", "Al_Aluminum_PPM", "Cr_Chromium_PPM", "Cu_Copper_PPM",
        "Fe_Iron_PPM", "K_Potassium_PPM", "Li_Lithium_PPM", "Na_Sodium_PPM",
        "Ni_Nickel_PPM", "Pb_Lead_PPM", "Si_Silicon_PPM", "Sn_Tin_PPM",
        "Soot_Soot_PERCENT", "Ti_Titanium_PPM", "V_Vanadium_PPM", "Water_Water_PERCENT",
    ]
    df = pd.read_csv(cfg.FLUID_SAMPLES_PATH, usecols=base_cols + analytes, low_memory=False)
    df = _prepare_common(df, date_col="sample_drawn_date", model_col="FULL_MODEL", serial_col="SERIAL")
    for col in ["sample_result_severity_order", "TELEMETRY_SMR_NUMERIC"] + analytes:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for severity in range(6):
        df[f"fluid_severity_{severity}_count"] = df["sample_result_severity_order"].eq(severity).astype("int16")
    df.attrs["analytes"] = analytes
    return df


def _aggregate_fluid(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    analytes = [
        "Ag_Silver_PPM", "Al_Aluminum_PPM", "Cr_Chromium_PPM", "Cu_Copper_PPM",
        "Fe_Iron_PPM", "K_Potassium_PPM", "Li_Lithium_PPM", "Na_Sodium_PPM",
        "Ni_Nickel_PPM", "Pb_Lead_PPM", "Si_Silicon_PPM", "Sn_Tin_PPM",
        "Soot_Soot_PERCENT", "Ti_Titanium_PPM", "V_Vanadium_PPM", "Water_Water_PERCENT",
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "fluid_sample_count": ("LABS_SAMPLE_NUMBER", "size"),
        "fluid_active_days": ("sample_drawn_date", "nunique"),
        "fluid_telemetry_smr_min": ("TELEMETRY_SMR_NUMERIC", "min"),
        "fluid_telemetry_smr_max": ("TELEMETRY_SMR_NUMERIC", "max"),
    }
    for severity in range(6):
        col = f"fluid_severity_{severity}_count"
        agg_spec[col] = (col, "sum")
    for col in analytes:
        short = col.lower().replace("_", "")
        agg_spec[f"fluid_{short}_mean"] = (col, "mean")
        agg_spec[f"fluid_{short}_max"] = (col, "max")
    agg = df.groupby(keys, as_index=False).agg(**agg_spec)
    agg["fluid_telemetry_smr_range"] = agg["fluid_telemetry_smr_max"] - agg["fluid_telemetry_smr_min"]
    rated = sum(agg[f"fluid_severity_{i}_count"] for i in range(5))
    agg["fluid_rated_sample_count"] = rated
    agg["fluid_nonzero_rated_count"] = sum(agg[f"fluid_severity_{i}_count"] for i in range(1, 5))
    agg["fluid_nonzero_rated_rate"] = safe_divide(agg["fluid_nonzero_rated_count"], agg["fluid_rated_sample_count"])
    return agg


def _prepare_operation() -> pd.DataFrame:
    usecols = [
        "full_model", "SERIAL", "LOCAL_DATE",
        "smr_hours", "smr_delta_clean_since_prev_obs_hours",
        "actual_working_hours_clean", "actual_work_day_flag", "inactive_actual_work_day_flag",
        "fuel_active_flag", "activity_fuel_consumption_value_raw",
        "engine_running_hours_clean", "engine_running_day_flag",
        "engine_idling_hours_clean", "engine_idle_share_daily",
        "throttle_full_hours_clean", "throttle_full_engine_share_daily",
        "throttle_average_dial_position_clean", "high_throttle_day_flag", "long_engine_day_flag",
        "traveling_hours_clean", "moving_back_forth_hours_clean", "steering_hours_clean",
        "working_hours_clean", "auto_quick_shift_hours_clean", "manual_variable_shift_hours_clean",
        "travel_day_flag", "movement_day_flag",
        "smr_missing_flag", "smr_decrease_flag", "smr_large_jump_flag",
        "actual_work_missing_flag", "engine_seconds_invalid_flag", "movement_invalid_flag",
        "travel_invalid_flag", "work_idle_sum_exceeds_engine_flag", "idle_exceeds_engine_flag",
        "throttle_full_exceeds_engine_flag", "days_since_last_actual_work_day_through_current_day",
    ]
    df = pd.read_csv(cfg.OPERATION_PATH, usecols=usecols, low_memory=False)
    df = _prepare_common(df, date_col="LOCAL_DATE", model_col="full_model", serial_col="SERIAL")
    numeric_cols = [c for c in usecols if c not in {"full_model", "SERIAL", "LOCAL_DATE"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _aggregate_operation(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    sum_mean_max_cols = [
        "smr_delta_clean_since_prev_obs_hours", "actual_working_hours_clean",
        "activity_fuel_consumption_value_raw", "engine_running_hours_clean",
        "engine_idling_hours_clean", "throttle_full_hours_clean", "traveling_hours_clean",
        "moving_back_forth_hours_clean", "steering_hours_clean", "working_hours_clean",
        "auto_quick_shift_hours_clean", "manual_variable_shift_hours_clean",
    ]
    mean_max_cols = [
        "engine_idle_share_daily", "throttle_full_engine_share_daily",
        "throttle_average_dial_position_clean", "days_since_last_actual_work_day_through_current_day",
    ]
    flag_cols = [
        "actual_work_day_flag", "inactive_actual_work_day_flag", "fuel_active_flag",
        "engine_running_day_flag", "high_throttle_day_flag", "long_engine_day_flag",
        "travel_day_flag", "movement_day_flag", "smr_missing_flag", "smr_decrease_flag",
        "smr_large_jump_flag", "actual_work_missing_flag", "engine_seconds_invalid_flag",
        "movement_invalid_flag", "travel_invalid_flag", "work_idle_sum_exceeds_engine_flag",
        "idle_exceeds_engine_flag", "throttle_full_exceeds_engine_flag",
    ]
    agg_spec: dict[str, tuple[str, str]] = {
        "operation_day_count": ("LOCAL_DATE", "nunique"),
        "operation_smr_min": ("smr_hours", "min"),
        "operation_smr_max": ("smr_hours", "max"),
    }
    for col in sum_mean_max_cols:
        base = f"operation_{col}"
        agg_spec[f"{base}_sum"] = (col, "sum")
        agg_spec[f"{base}_mean"] = (col, "mean")
        agg_spec[f"{base}_max"] = (col, "max")
    for col in mean_max_cols:
        base = f"operation_{col}"
        agg_spec[f"{base}_mean"] = (col, "mean")
        agg_spec[f"{base}_max"] = (col, "max")
    for col in flag_cols:
        agg_spec[f"operation_{col}_count"] = (col, "sum")
    agg = df.groupby(keys, as_index=False).agg(**agg_spec)
    agg["operation_smr_range"] = agg["operation_smr_max"] - agg["operation_smr_min"]
    agg["operation_work_day_rate"] = safe_divide(agg["operation_actual_work_day_flag_count"], agg["operation_day_count"])
    agg["operation_inactive_day_rate"] = safe_divide(agg["operation_inactive_actual_work_day_flag_count"], agg["operation_day_count"])
    agg["operation_engine_day_rate"] = safe_divide(agg["operation_engine_running_day_flag_count"], agg["operation_day_count"])
    agg["operation_high_throttle_day_rate"] = safe_divide(agg["operation_high_throttle_day_flag_count"], agg["operation_day_count"])
    agg["operation_long_engine_day_rate"] = safe_divide(agg["operation_long_engine_day_flag_count"], agg["operation_day_count"])
    agg["operation_fuel_per_engine_hour"] = safe_divide(
        agg["operation_activity_fuel_consumption_value_raw_sum"],
        agg["operation_engine_running_hours_clean_sum"],
    )
    agg["operation_idle_to_engine_ratio"] = safe_divide(
        agg["operation_engine_idling_hours_clean_sum"],
        agg["operation_engine_running_hours_clean_sum"],
    )
    agg["operation_work_to_engine_ratio"] = safe_divide(
        agg["operation_actual_working_hours_clean_sum"],
        agg["operation_engine_running_hours_clean_sum"],
    )
    return agg


def _merge_source(
    base: pd.DataFrame,
    table: pd.DataFrame,
    source_name: str,
    *,
    anchor: bool,
) -> pd.DataFrame:
    keys = ["machine_key", "period_start"] + (["anchor_date"] if anchor else [])
    if table.empty:
        out = base.copy()
        out[f"has_{source_name}_data"] = 0
        return out
    feature_cols = [c for c in table.columns if c not in keys]
    out = base.merge(table, on=keys, how="left", validate="one_to_one")
    first_feature = feature_cols[0]
    out[f"has_{source_name}_data"] = out[first_feature].notna().astype("int8")
    return out


def _fill_count_like(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count_like = [
        c for c in result.columns
        if c.endswith("_count")
        or c.endswith("_sum")
        or c.startswith("has_")
    ]
    for col in count_like:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result


def build_condition_features(*, force_rebuild: bool = False) -> FeatureBuildResult:
    cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    segment_path = cfg.CACHE_DIR / "condition_segment_features_90d.pkl"
    anchor_path = cfg.CACHE_DIR / "condition_anchor_features_90d.pkl"
    roster_path = cfg.CACHE_DIR / "machine_roster.csv"
    audit_path = cfg.CACHE_DIR / "source_feature_audit.csv"
    if (
        cfg.REUSE_CONDITION_FEATURE_CACHE
        and not force_rebuild
        and segment_path.exists()
        and anchor_path.exists()
        and roster_path.exists()
        and audit_path.exists()
    ):
        segment = pd.read_pickle(segment_path)
        anchor = pd.read_pickle(anchor_path)
        roster = pd.read_csv(roster_path)
        audit = pd.read_csv(audit_path)
        for frame in (segment, anchor):
            for col in ["period_start", "period_end", "anchor_date"]:
                if col in frame.columns:
                    frame[col] = pd.to_datetime(frame[col], errors="coerce")
        return FeatureBuildResult(segment, anchor, roster, audit)

    roster = build_machine_roster()
    plan = build_anchor_plan()
    segment = build_segment_grid(roster)
    anchor = build_anchor_grid(roster, plan)
    audit_rows: list[dict[str, object]] = []

    source_jobs: list[tuple[str, Callable[[], pd.DataFrame], Callable[[pd.DataFrame, list[str]], pd.DataFrame], str]] = [
        ("fault", _prepare_fault, _aggregate_fault, "event_date"),
        ("maintenance", _prepare_maintenance, _aggregate_maintenance, "event_date"),
        ("fluid", _prepare_fluid, _aggregate_fluid, "sample_drawn_date"),
        ("operation", _prepare_operation, _aggregate_operation, "LOCAL_DATE"),
    ]
    for source_name, prepare, aggregate, date_col in source_jobs:
        print(f"Preparing and aggregating {source_name} features...", flush=True)
        prepared = prepare()
        fixed_table, anchor_table = _aggregate_fixed_and_anchors(
            prepared,
            date_col=date_col,
            aggregate=aggregate,
            anchor_plan=plan,
        )
        if not anchor_table.empty and "anchor_date" not in anchor_table.columns:
            raise RuntimeError(f"Anchor date missing from {source_name} aggregate")
        segment = _merge_source(segment, fixed_table, source_name, anchor=False)
        anchor = _merge_source(anchor, anchor_table, source_name, anchor=True)
        audit_rows.append(
            {
                "source": source_name,
                "prepared_rows": int(len(prepared)),
                "fixed_aggregate_rows": int(len(fixed_table)),
                "anchor_aggregate_rows": int(len(anchor_table)),
                "fixed_feature_columns": int(max(0, len(fixed_table.columns) - 2)),
                "anchor_feature_columns": int(max(0, len(anchor_table.columns) - 3)),
            }
        )
        del prepared, fixed_table, anchor_table
        gc.collect()

    segment = _fill_count_like(segment)
    anchor = _fill_count_like(anchor)
    segment.to_pickle(segment_path)
    anchor.to_pickle(anchor_path)
    roster.to_csv(roster_path, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(audit_path, index=False)
    return FeatureBuildResult(segment, anchor, roster, audit)
