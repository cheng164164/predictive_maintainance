"""Reviewed shared feature schema used by every forecast horizon.

The list began with the historical reduced schema and was expanded after a
cross-horizon importance and domain review. It intentionally retains the full
L00-L04 action-level counts, aggregate fault severity/volume, actionable
maintenance status, severe fluid indicators, selected operation intensity and
utilization measures, source-availability flags, and prior-claim history.

The list is frozen outside config.py so experiments remain reproducible while
config.py stays focused on user controls. Step 04 validates every name against
the Step 03 full feature schema and restores the original dataframe order.
"""

from __future__ import annotations

# Selection method used for the reviewed schema:
#   1. Start with the prior 80-feature shared schema.
#   2. Add raw features represented among the top 30 full-model importance
#      positions for at least one evaluated horizon (30, 60, or 180 days).
#   3. Add domain-important fault severity, maintenance-status, fluid-severity,
#      source-availability, and selected operation features.
#   4. Verify the same list across all horizons with chronological splits.
REDUCED_FEATURES = (
    'full_model',
    'segment_season',
    'segment_year',
    'fault_event_count',
    'fault_active_days',
    'fault_unique_code_count',
    'fault_occurrence_sum',
    'fault_occurrence_max',
    'fault_log_occurrence_sum',
    'fault_log_occurrence_mean',
    'fault_engine_on_hours_sum',
    'fault_engine_on_hours_max',
    'fault_smr_min',
    'fault_smr_max',
    'fault_mechanical_count',
    'fault_electrical_count',
    'fault_action_level_mean',
    'fault_action_level_max',
    'fault_evidence_score_sum',
    'fault_evidence_score_mean',
    'fault_evidence_score_max',
    'fault_action_l00_count',
    'fault_action_l01_count',
    'fault_action_l02_count',
    'fault_action_l03_count',
    'fault_action_l04_count',
    'fault_strength_weak_count',
    'fault_strength_medium_count',
    'fault_strength_strong_count',
    'fault_group_context_count',
    'fault_group_event_count',
    'fault_logical_mon_count',
    'fault_component_urea_scr_system_count',
    'fault_component_engine_count',
    'fault_component_power_train_system_count',
    'fault_component_work_equipment_system_count',
    'fault_component_cooling_system_count',
    'fault_component_control_system_count',
    'fault_smr_range',
    'fault_events_per_active_day',
    'fault_strong_rate',
    'fault_medium_or_strong_rate',
    'fault_action_ge3_rate',
    'has_fault_data',
    'maint_event_count',
    'maint_active_days',
    'maint_unique_event_count',
    'maint_monitor_reset_count',
    'maint_overdue_count',
    'maint_due_now_count',
    'maint_remaining_negative_count',
    'maint_remaining_zero_count',
    'maint_remaining_hours_min',
    'maint_remaining_hours_mean',
    'maint_remaining_hours_max',
    'maint_interval_hours_min',
    'maint_interval_hours_mean',
    'maint_interval_hours_max',
    'maint_threshold_hours_mean',
    'maint_smr_min',
    'maint_smr_max',
    'maint_previous_reset_smr_mean',
    'maint_type_filter_count',
    'maint_type_oil_count',
    'maint_type_breather_count',
    'maint_type_cleaning_count',
    'maint_type_coolant_count',
    'maint_component_engine_count',
    'maint_component_final_drive_count',
    'maint_smr_range',
    'maint_overdue_rate',
    'maint_due_now_rate',
    'maint_reset_rate',
    'has_maintenance_data',
    'fluid_sample_count',
    'fluid_active_days',
    'fluid_telemetry_smr_min',
    'fluid_telemetry_smr_max',
    'fluid_severity_2_count',
    'fluid_severity_3_count',
    'fluid_severity_4_count',
    'fluid_severity_5_count',
    'fluid_agsilverppm_mean',
    'fluid_agsilverppm_max',
    'fluid_alaluminumppm_mean',
    'fluid_alaluminumppm_max',
    'fluid_crchromiumppm_mean',
    'fluid_crchromiumppm_max',
    'fluid_cucopperppm_mean',
    'fluid_cucopperppm_max',
    'fluid_feironppm_mean',
    'fluid_feironppm_max',
    'fluid_kpotassiumppm_mean',
    'fluid_kpotassiumppm_max',
    'fluid_nasodiumppm_mean',
    'fluid_nasodiumppm_max',
    'fluid_ninickelppm_mean',
    'fluid_ninickelppm_max',
    'fluid_pbleadppm_mean',
    'fluid_pbleadppm_max',
    'fluid_sisiliconppm_mean',
    'fluid_sisiliconppm_max',
    'fluid_sntinppm_mean',
    'fluid_sntinppm_max',
    'fluid_sootsootpercent_mean',
    'fluid_sootsootpercent_max',
    'fluid_tititaniumppm_mean',
    'fluid_tititaniumppm_max',
    'fluid_vvanadiumppm_mean',
    'fluid_vvanadiumppm_max',
    'fluid_waterwaterpercent_mean',
    'fluid_waterwaterpercent_max',
    'fluid_telemetry_smr_range',
    'fluid_rated_sample_count',
    'fluid_nonzero_rated_count',
    'fluid_nonzero_rated_rate',
    'has_fluid_data',
    'operation_smr_min',
    'operation_smr_max',
    'operation_smr_delta_clean_since_prev_obs_hours_mean',
    'operation_actual_working_hours_clean_mean',
    'operation_activity_fuel_consumption_value_raw_mean',
    'operation_activity_fuel_consumption_value_raw_max',
    'operation_engine_running_hours_clean_mean',
    'operation_engine_running_hours_clean_max',
    'operation_throttle_full_hours_clean_max',
    'operation_steering_hours_clean_max',
    'operation_working_hours_clean_max',
    'operation_engine_idle_share_daily_max',
    'operation_throttle_full_engine_share_daily_mean',
    'operation_throttle_full_engine_share_daily_max',
    'operation_throttle_average_dial_position_clean_mean',
    'operation_high_throttle_day_rate',
    'operation_fuel_per_engine_hour',
    'prior_claim_count_30d',
    'prior_claim_count_90d',
    'prior_claim_count_180d',
    'prior_claim_count_365d',
    'prior_claim_count_ever',
    'days_since_prior_claim',
)

# Features explicitly retained after the action-level review. These assertions
# make accidental removal visible during import and code review.
REQUIRED_REVIEW_FEATURES = (
    "fault_action_l00_count",
    "fault_action_l01_count",
    "fault_action_l02_count",
    "fault_action_l03_count",
    "fault_action_l04_count",
    "fault_action_level_max",
    "has_fault_data",
    "has_maintenance_data",
    "has_fluid_data",
)

_missing_review_features = [
    feature for feature in REQUIRED_REVIEW_FEATURES if feature not in REDUCED_FEATURES
]
if _missing_review_features:
    raise RuntimeError(
        "Reviewed feature schema is missing required features: "
        f"{_missing_review_features}"
    )
