"""
Configuration for the standalone machine-learning training and validation project.

Normal usage from this folder:

    python main.py

The scripts do not import the feature-selection project. They reuse the same
chronological split strategy, but assume feature selection has already been
reviewed and that the prepared features below are the approved feature sets.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# The resolver accepts either a file or a directory. If this is a directory, the
# loader searches for snapshot_dataframe.parquet, snapshot_dataframe.csv, or the
# first parquet/csv file in the folder.
INPUT_DATA_PATH = PROJECT_DIR.parent / "data_preparation" / "output" / "snapshot_dataframe"

# All generated CSV, JSON, model, and validation outputs are written here.
OUTPUT_DIR = PROJECT_DIR / "output"

# -----------------------------------------------------------------------------
# Dataset columns
# -----------------------------------------------------------------------------
TARGET_COL = "claim_next_90d"
DATE_COL = "snapshot_date"
ID_COLS = ["model_id"]
SECONDARY_SORT_COLS = ["model_id"]

# Optional source columns to drop immediately after loading, before splitting.
SOURCE_COLUMNS_TO_DROP_BEFORE_MODELING = []

# -----------------------------------------------------------------------------
# Snapshot row filtering and target-window clipping
# -----------------------------------------------------------------------------
# The raw snapshot builder may create a full machine-date panel. That can produce
# many artificial inactive rows where every model feature is zero/blank. These
# rows are removed before split/CV/final training so the model learns from active
# machine snapshots instead of from empty machine-date combinations.
SNAPSHOT_TRAINING_FILTERS_ENABLED = True

# claim_next_90d needs a complete 90-day future observation window. The current
# warranty source ends on 2026-06-26, so the latest valid snapshot date is
# 2026-06-26 minus 90 days = 2026-03-28. Snapshot dates after that are clipped.
DROP_SNAPSHOTS_AFTER_FULL_TARGET_WINDOW = True
SNAPSHOT_CUTOFF_REFERENCE_END_DATE = "2026-06-26"
SNAPSHOT_TARGET_HORIZON_DAYS = 90

# Frozen-feature run: use only the generalized numeric sparsity check to remove
# completely empty model rows. With the minimum set to 1, rows with zero nonzero
# numeric model features are removed, while rows with at least one real feature
# value are retained.
DROP_ALL_ZERO_SNAPSHOT_ROWS = False
DROP_EXTREME_SPARSE_SNAPSHOT_ROWS = True
SPARSE_ROW_MIN_NONZERO_FEATURE_COUNT = 5
SPARSE_ROW_NONZERO_EPSILON = 1e-12
SPARSE_ROW_NUMERIC_ONLY = True

# Optional override for the model-feature sparsity calculation. Leave empty to
# use the source features required by the configured MODEL_VARIANTS_TO_RUN.
SNAPSHOT_SPARSITY_FEATURE_COLUMNS = []

# Empty-row detection is intentionally separate from model-feature sparsity.
# For the base snapshot, these columns directly indicate whether any source
# record exists inside the 90-day observation window. If neither column exists
# (for example, when running an older frozen-feature snapshot), the utility
# automatically falls back to cleaned numeric model features while excluding
# sentinel/recency fields.
SNAPSHOT_ACTIVITY_FEATURE_COLUMNS = [
    "source_record_count_window",
    "has_any_source_window",
]
SNAPSHOT_ACTIVITY_EXCLUDE_COLUMNS = [
    "days_since_prior_claim_before_window",
    "fault_days_since_latest_in_window",
    "fluid_days_since_latest_sample_window",
    "maintenance_days_since_latest_event_window",
]

# Keep this False for normal modeling. Set True temporarily only when debugging
# the filtered snapshot, because it adds a diagnostic column to the dataframe.
SNAPSHOT_FILTER_ADD_DIAGNOSTIC_COLUMNS = False

# -----------------------------------------------------------------------------
# Outer chronological split: train / validation / test
# -----------------------------------------------------------------------------
# These are treated as weights and normalized internally if they do not sum to 1.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# -----------------------------------------------------------------------------
# Expanding-window CV inside training_main only
# -----------------------------------------------------------------------------
CV_N_SPLITS = 1
CV_VALIDATION_WINDOW_DAYS = 45
CV_GAP_DAYS = 90
CV_MIN_TRAIN_ROWS = 1000
CV_MIN_VALIDATION_ROWS = 200
CV_MIN_POSITIVES_IN_TRAIN = 10
CV_MIN_POSITIVES_IN_VALIDATION = 5

# CV threshold metrics are reported for context only. Hyperparameter selection
# uses mean PR-AUC / average precision, not F2.
CV_DEFAULT_THRESHOLD = 0.50
CV_MAX_FLAGGED_RATE_FOR_BEST_F2 = 0.2

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
ONE_HOT_ENCODE_CATEGORICAL = True

# Missing source columns are skipped instead of stopping the run. Every prepared
# feature mapped to a missing source column is removed from that model run and is
# reported in the CV/final-model outputs. Set this to True only when you want a
# missing source column to fail the workflow immediately.
ERROR_ON_MISSING_SOURCE_FEATURES = False

# If a source column exists but a selected prepared column is not generated in a
# particular training period (most commonly a one-hot category absent from that
# training fold), add that prepared column as zero. This setting does NOT restore
# prepared features whose entire source column is missing; those are skipped.
ADD_MISSING_PREPARED_FEATURES_AS_ZERO = True

# -----------------------------------------------------------------------------
# 9999 sentinel cleaning
# -----------------------------------------------------------------------------
# Do not treat 9999 as a real number of days. Sentinel values are replaced with
# missing before splitting and preprocessing. Existing base-feature availability
# flags are preserved and checked; older frozen-feature indicators are created
# when they are not already present. The numeric imputer then handles the cleaned
# recency value.
SENTINEL_CLEANING_ENABLED = True
SENTINEL_VALUE = 9999
SENTINEL_REPLACE_WITH = None  # None means pandas NA/NaN
SENTINEL_COLUMNS_TO_CLEAN = {
    # Frozen-feature recency columns.
    "days_since_last_fault": "has_prior_fault",
    "days_since_last_severe_fault": "has_prior_severe_fault",
    "days_since_last_claim": "has_prior_claim",
    "days_since_last_reset": "has_prior_reset",
    "days_since_last_smr": "has_prior_smr",

    # Base-feature recency columns. The current-window indicators already exist
    # in the snapshot and are preserved/validated by the cleaning utility.
    "days_since_prior_claim_before_window": "has_prior_claim_before_window",
    "fault_days_since_latest_in_window": "has_fault_window",
    "fluid_days_since_latest_sample_window": "has_fluid_window",
    "maintenance_days_since_latest_event_window": "has_maintenance_window",
}

# -----------------------------------------------------------------------------
# Feature variants
# -----------------------------------------------------------------------------
# Model A = all 93 prepared features that survived feature-selection review.
# Model B = Model A minus high-confidence drop candidates.
# Model C = Model B minus additional review/drop candidates, including the
# redundant has_fault_90d flag.
MODEL_A_PREPARED_FEATURES = ['num__days_since_last_fault',
 'num__smr_latest_before_snapshot',
 'num__days_since_last_claim',
 'num__fluid_sample_count_365d',
 'num__days_since_last_reset',
 'num__prior_claim_count_365d',
 'num__Cu_Copper_PPM',
 'num__monitor_reset_count_180d',
 'num__prior_claim_count_180d',
 'num__Fe_Iron_PPM',
 'num__days_since_last_severe_fault',
 'num__avg_remaining_hours',
 'num__maintenance_events_180d',
 'num__max_log_occurrence_90d',
 'num__maintenance_events_90d',
 'num__Soot_Soot_PERCENT',
 'num__avg_working_hours_per_actual_work_day_90d',
 'num__avg_event_evidence_score_90d',
 'num__fault_smr_delta_90d',
 'num__travel_share_of_working_hours_90d',
 'num__Sn_Tin_PPM',
 'num__smr_latest_hours',
 'num__Pb_Lead_PPM',
 'num__Ag_Silver_PPM',
 'num__monitor_reset_count_90d',
 'num__top_component_fault_ratio_90d',
 'num__unique_fault_code_count_90d',
 'num__smr_delta_30d',
 'num__engine_idling_share_90d',
 'num__min_remaining_hours',
 'num__faults_per_100_hours',
 'num__occurrence_severity_score_90d',
 'num__Ti_Titanium_PPM',
 'num__moving_back_forth_to_travel_ratio_90d',
 'num__manual_variable_shift_hours_sum_90d',
 'num__engine_fault_count_90d',
 'num__K_Potassium_PPM',
 'num__days_since_last_smr',
 'num__engine_observed_day_count_90d',
 'num__urea_scr_system_reset_count_180d',
 'num__engine_running_day_ratio_90d',
 'num__smr_delta_90d',
 'num__Soot_Soot_Abs_cm',
 'num__repeat_fault_ratio_90d',
 'num__max_context_evidence_score_90d',
 'num__mechanical_fault_count_90d',
 'num__working_hours_sum_7d',
 'num__unique_claim_type_count_365d',
 'num__unique_component_count_90d',
 'num__moderate_fault_count_90d',
 'num__Fuel_Fuel_PERCENT',
 'num__prior_claim_count_90d',
 'num__action_L03_count_90d',
 'num__strong_fault_count_90d',
 'num__electrical_fault_count_90d',
 'num__smr_since_last_reset',
 'num__fault_count_30d',
 'num__fuel_actual_work_conflict_count_90d',
 'num__action_L04_count_90d',
 'num__fault_count_7d',
 'num__smr_delta_7d',
 'num__fault_growth_rate',
 'num__auto_quick_shift_hours_sum_90d',
 'num__V_Vanadium_PPM',
 'num__action_L01_count_90d',
 'cat__full_model_D71EXI_24',
 'num__workequipment_fault_count_90d',
 'num__Ni_Nickel_PPM',
 'num__maintenance_due_or_overdue_ratio',
 'num__max_action_level_90d',
 'num__working_hours_sum_90d',
 'num__electrical_fault_count_30d',
 'num__powertrain_fault_count_90d',
 'num__actual_work_day_ratio_change_30d_vs_90d',
 'num__has_fault_90d',
 'cat__full_model_D71PX_24',
 'num__fault_count_previous_30d',
 'num__mechanical_fault_count_30d',
 'num__Water_Water_PERCENT',
 'num__current_actual_work_streak_days',
 'num__Li_Lithium_PPM',
 'num__action_L02_count_90d',
 'num__due_now_item_count',
 'num__throttle_full_share_change_30d_vs_90d',
 'num__final_drive_overdue_item_count',
 'num__cooling_fault_count_90d',
 'cat__full_model_D71EX_24',
 'cat__full_model_D71PX_24E0',
 'num__working_hours_rate_change_30d_vs_90d',
 'cat__full_model_D71PXI_24',
 'num__urea_scr_system_overdue_item_count',
 'num__overdue_item_count',
 'num__transmission_overdue_item_count']

MODEL_B_DROP_PREPARED_FEATURES = ['num__transmission_overdue_item_count',
 'num__overdue_item_count',
 'num__urea_scr_system_overdue_item_count',
 'num__cooling_fault_count_90d',
 'num__final_drive_overdue_item_count',
 'num__Li_Lithium_PPM',
 'num__current_actual_work_streak_days',
 'num__action_L02_count_90d',
 'num__due_now_item_count',
 'num__working_hours_rate_change_30d_vs_90d',
 'cat__full_model_D71PX_24E0',
 'cat__full_model_D71PXI_24',
 'num__maintenance_due_or_overdue_ratio',
 'num__powertrain_fault_count_90d']

MODEL_C_ADDITIONAL_DROP_PREPARED_FEATURES = ['num__actual_work_day_ratio_change_30d_vs_90d',
 'num__mechanical_fault_count_30d',
 'num__strong_fault_count_90d',
 'num__smr_since_last_reset',
 'num__auto_quick_shift_hours_sum_90d',
 'num__fault_growth_rate',
 'num__workequipment_fault_count_90d',
 'num__Water_Water_PERCENT',
 'num__Fuel_Fuel_PERCENT',
 'num__moving_back_forth_to_travel_ratio_90d',
 'num__fuel_actual_work_conflict_count_90d',
 'cat__full_model_D71EX_24',
 'cat__full_model_D71PX_24',
 'cat__full_model_D71EXI_24',
 'num__has_fault_90d']

PREPARED_TO_SOURCE_FEATURE = {'cat__full_model_D71EXI_24': 'full_model',
 'cat__full_model_D71EX_24': 'full_model',
 'cat__full_model_D71PXI_24': 'full_model',
 'cat__full_model_D71PX_24': 'full_model',
 'cat__full_model_D71PX_24E0': 'full_model',
 'num__Ag_Silver_PPM': 'Ag_Silver_PPM',
 'num__Cu_Copper_PPM': 'Cu_Copper_PPM',
 'num__Fe_Iron_PPM': 'Fe_Iron_PPM',
 'num__Fuel_Fuel_PERCENT': 'Fuel_Fuel_PERCENT',
 'num__K_Potassium_PPM': 'K_Potassium_PPM',
 'num__Li_Lithium_PPM': 'Li_Lithium_PPM',
 'num__Ni_Nickel_PPM': 'Ni_Nickel_PPM',
 'num__Pb_Lead_PPM': 'Pb_Lead_PPM',
 'num__Sn_Tin_PPM': 'Sn_Tin_PPM',
 'num__Soot_Soot_Abs_cm': 'Soot_Soot_Abs_cm',
 'num__Soot_Soot_PERCENT': 'Soot_Soot_PERCENT',
 'num__Ti_Titanium_PPM': 'Ti_Titanium_PPM',
 'num__V_Vanadium_PPM': 'V_Vanadium_PPM',
 'num__Water_Water_PERCENT': 'Water_Water_PERCENT',
 'num__action_L01_count_90d': 'action_L01_count_90d',
 'num__action_L02_count_90d': 'action_L02_count_90d',
 'num__action_L03_count_90d': 'action_L03_count_90d',
 'num__action_L04_count_90d': 'action_L04_count_90d',
 'num__actual_work_day_ratio_change_30d_vs_90d': 'actual_work_day_ratio_change_30d_vs_90d',
 'num__auto_quick_shift_hours_sum_90d': 'auto_quick_shift_hours_sum_90d',
 'num__avg_event_evidence_score_90d': 'avg_event_evidence_score_90d',
 'num__avg_remaining_hours': 'avg_remaining_hours',
 'num__avg_working_hours_per_actual_work_day_90d': 'avg_working_hours_per_actual_work_day_90d',
 'num__cooling_fault_count_90d': 'cooling_fault_count_90d',
 'num__current_actual_work_streak_days': 'current_actual_work_streak_days',
 'num__days_since_last_claim': 'days_since_last_claim',
 'num__days_since_last_fault': 'days_since_last_fault',
 'num__days_since_last_reset': 'days_since_last_reset',
 'num__days_since_last_severe_fault': 'days_since_last_severe_fault',
 'num__days_since_last_smr': 'days_since_last_smr',
 'num__due_now_item_count': 'due_now_item_count',
 'num__electrical_fault_count_30d': 'electrical_fault_count_30d',
 'num__electrical_fault_count_90d': 'electrical_fault_count_90d',
 'num__engine_fault_count_90d': 'engine_fault_count_90d',
 'num__engine_idling_share_90d': 'engine_idling_share_90d',
 'num__engine_observed_day_count_90d': 'engine_observed_day_count_90d',
 'num__engine_running_day_ratio_90d': 'engine_running_day_ratio_90d',
 'num__fault_count_30d': 'fault_count_30d',
 'num__fault_count_7d': 'fault_count_7d',
 'num__fault_count_previous_30d': 'fault_count_previous_30d',
 'num__fault_growth_rate': 'fault_growth_rate',
 'num__fault_smr_delta_90d': 'fault_smr_delta_90d',
 'num__faults_per_100_hours': 'faults_per_100_hours',
 'num__final_drive_overdue_item_count': 'final_drive_overdue_item_count',
 'num__fluid_sample_count_365d': 'fluid_sample_count_365d',
 'num__fuel_actual_work_conflict_count_90d': 'fuel_actual_work_conflict_count_90d',
 'num__has_fault_90d': 'has_fault_90d',
 'num__maintenance_due_or_overdue_ratio': 'maintenance_due_or_overdue_ratio',
 'num__maintenance_events_180d': 'maintenance_events_180d',
 'num__maintenance_events_90d': 'maintenance_events_90d',
 'num__manual_variable_shift_hours_sum_90d': 'manual_variable_shift_hours_sum_90d',
 'num__max_action_level_90d': 'max_action_level_90d',
 'num__max_context_evidence_score_90d': 'max_context_evidence_score_90d',
 'num__max_log_occurrence_90d': 'max_log_occurrence_90d',
 'num__mechanical_fault_count_30d': 'mechanical_fault_count_30d',
 'num__mechanical_fault_count_90d': 'mechanical_fault_count_90d',
 'num__min_remaining_hours': 'min_remaining_hours',
 'num__moderate_fault_count_90d': 'moderate_fault_count_90d',
 'num__monitor_reset_count_180d': 'monitor_reset_count_180d',
 'num__monitor_reset_count_90d': 'monitor_reset_count_90d',
 'num__moving_back_forth_to_travel_ratio_90d': 'moving_back_forth_to_travel_ratio_90d',
 'num__occurrence_severity_score_90d': 'occurrence_severity_score_90d',
 'num__overdue_item_count': 'overdue_item_count',
 'num__powertrain_fault_count_90d': 'powertrain_fault_count_90d',
 'num__prior_claim_count_180d': 'prior_claim_count_180d',
 'num__prior_claim_count_365d': 'prior_claim_count_365d',
 'num__prior_claim_count_90d': 'prior_claim_count_90d',
 'num__repeat_fault_ratio_90d': 'repeat_fault_ratio_90d',
 'num__smr_delta_30d': 'smr_delta_30d',
 'num__smr_delta_7d': 'smr_delta_7d',
 'num__smr_delta_90d': 'smr_delta_90d',
 'num__smr_latest_before_snapshot': 'smr_latest_before_snapshot',
 'num__smr_latest_hours': 'smr_latest_hours',
 'num__smr_since_last_reset': 'smr_since_last_reset',
 'num__strong_fault_count_90d': 'strong_fault_count_90d',
 'num__throttle_full_share_change_30d_vs_90d': 'throttle_full_share_change_30d_vs_90d',
 'num__top_component_fault_ratio_90d': 'top_component_fault_ratio_90d',
 'num__transmission_overdue_item_count': 'transmission_overdue_item_count',
 'num__travel_share_of_working_hours_90d': 'travel_share_of_working_hours_90d',
 'num__unique_claim_type_count_365d': 'unique_claim_type_count_365d',
 'num__unique_component_count_90d': 'unique_component_count_90d',
 'num__unique_fault_code_count_90d': 'unique_fault_code_count_90d',
 'num__urea_scr_system_overdue_item_count': 'urea_scr_system_overdue_item_count',
 'num__urea_scr_system_reset_count_180d': 'urea_scr_system_reset_count_180d',
 'num__workequipment_fault_count_90d': 'workequipment_fault_count_90d',
 'num__working_hours_rate_change_30d_vs_90d': 'working_hours_rate_change_30d_vs_90d',
 'num__working_hours_sum_7d': 'working_hours_sum_7d',
 'num__working_hours_sum_90d': 'working_hours_sum_90d'}

MODEL_B_PREPARED_FEATURES = [
    f for f in MODEL_A_PREPARED_FEATURES if f not in set(MODEL_B_DROP_PREPARED_FEATURES)
]
MODEL_C_PREPARED_FEATURES = [
    f
    for f in MODEL_A_PREPARED_FEATURES
    if f not in set(MODEL_B_DROP_PREPARED_FEATURES)
    and f not in set(MODEL_C_ADDITIONAL_DROP_PREPARED_FEATURES)
]

# Model D = Model C plus protected machine-context features. This keeps the
# lean feature set from Model C while retaining broader model/product context.
# Do not use unique equipment serial IDs as one-hot model inputs unless you have
# explicitly validated that this does not create memorization/leakage.
MACHINE_CONTEXT_SOURCE_FEATURES = ["full_model"]
MACHINE_CONTEXT_PREPARED_FEATURE_PREFIXES = ["cat__full_model_"]
PROTECTED_MACHINE_CONTEXT_PREPARED_FEATURES = [
    f
    for f in MODEL_A_PREPARED_FEATURES
    if PREPARED_TO_SOURCE_FEATURE.get(f) in set(MACHINE_CONTEXT_SOURCE_FEATURES)
    or any(f.startswith(prefix) for prefix in MACHINE_CONTEXT_PREPARED_FEATURE_PREFIXES)
]
MODEL_D_PREPARED_FEATURES = [
    f
    for f in MODEL_A_PREPARED_FEATURES
    if f in set(MODEL_C_PREPARED_FEATURES)
    or f in set(PROTECTED_MACHINE_CONTEXT_PREPARED_FEATURES)
]

# Model E = the transparent 90-day base feature set produced by the revised
# snapshot builder. Numeric source columns map directly to num__* prepared
# features. Categorical wildcard entries are expanded after OneHotEncoder is fit,
# so Model E automatically keeps every observed category without hardcoding the
# current category values.
BASE_NUMERIC_FEATURES = [
    "prior_claim_count_before_window",
    "days_since_prior_claim_before_window",
    "has_any_source_window",
    "source_record_count_window",
    "has_fault_window",
    "fault_count_window",
    "fault_unique_code_count_window",
    "fault_l03plus_count_window",
    "fault_l04plus_count_window",
    "fault_max_action_level_window",
    "fault_max_evidence_score_window",
    "fault_mean_evidence_score_window",
    "fault_max_log_occurrence_window",
    "fault_days_since_latest_in_window",
    "fault_mechanical_count_window",
    "fault_electrical_count_window",
    "has_fluid_window",
    "fluid_sample_count_window",
    "fluid_max_severity_window",
    "fluid_latest_severity_window",
    "fluid_days_since_latest_sample_window",
    "fluid_max_cu_ppm_window",
    "fluid_max_fe_ppm_window",
    "fluid_max_pb_ppm_window",
    "fluid_max_soot_percent_window",
    "fluid_max_water_percent_window",
    "has_maintenance_window",
    "maintenance_event_count_window",
    "maintenance_monitor_reset_count_window",
    "maintenance_overdue_count_window",
    "maintenance_due_now_count_window",
    "maintenance_min_remaining_hours_window",
    "maintenance_days_since_latest_event_window",
    "has_operation_window",
    "operation_day_count_window",
    "operation_working_hours_sum_window",
    "operation_working_hours_mean_window",
    "operation_working_hours_max_window",
    "operation_engine_running_hours_sum_window",
    "operation_idle_hours_sum_window",
    "operation_idle_share_window",
    "operation_latest_smr_window",
    "operation_smr_delta_window",
    "operation_high_throttle_day_count_window",
]

BASE_CATEGORICAL_FEATURES = [
    "full_model",
    "fault_dominant_component_window",
    "maintenance_dominant_component_window",
]

MODEL_E_CATEGORICAL_PREPARED_PATTERNS = [
    f"cat__{source}_*" for source in BASE_CATEGORICAL_FEATURES
]
MODEL_E_PREPARED_FEATURES = [
    *(f"num__{source}" for source in BASE_NUMERIC_FEATURES),
    *MODEL_E_CATEGORICAL_PREPARED_PATTERNS,
]

for _source_col in BASE_NUMERIC_FEATURES:
    PREPARED_TO_SOURCE_FEATURE[f"num__{_source_col}"] = _source_col
for _source_col in BASE_CATEGORICAL_FEATURES:
    PREPARED_TO_SOURCE_FEATURE[f"cat__{_source_col}_*"] = _source_col



def _append_sentinel_indicator_features(prepared_features):
    """Add has_prior_* prepared features after selected cleaned days-since features."""
    if not SENTINEL_CLEANING_ENABLED:
        return list(prepared_features)
    source_to_indicator_prepared = {
        source: f"num__{indicator}"
        for source, indicator in SENTINEL_COLUMNS_TO_CLEAN.items()
    }
    out = []
    for feature in prepared_features:
        if feature not in out:
            out.append(feature)
        source = PREPARED_TO_SOURCE_FEATURE.get(feature)
        indicator_prepared = source_to_indicator_prepared.get(source)
        if indicator_prepared and indicator_prepared not in out:
            out.append(indicator_prepared)
    return out

if SENTINEL_CLEANING_ENABLED:
    for _source_col, _indicator_col in SENTINEL_COLUMNS_TO_CLEAN.items():
        PREPARED_TO_SOURCE_FEATURE[f"num__{_indicator_col}"] = _indicator_col
    MODEL_A_PREPARED_FEATURES = _append_sentinel_indicator_features(MODEL_A_PREPARED_FEATURES)
    MODEL_B_PREPARED_FEATURES = _append_sentinel_indicator_features(MODEL_B_PREPARED_FEATURES)
    MODEL_C_PREPARED_FEATURES = _append_sentinel_indicator_features(MODEL_C_PREPARED_FEATURES)
    MODEL_D_PREPARED_FEATURES = _append_sentinel_indicator_features(MODEL_D_PREPARED_FEATURES)

FEATURE_SETS = {
    "A": MODEL_A_PREPARED_FEATURES,
    "B": MODEL_B_PREPARED_FEATURES,
    "C": MODEL_C_PREPARED_FEATURES,
    "D": MODEL_D_PREPARED_FEATURES,
    "E": MODEL_E_PREPARED_FEATURES,
}

# Run CV and validation training for these variants. Test evaluation should be
# done only for the final selected variant unless EVALUATE_TEST_FOR_ALL_VARIANTS
# is intentionally set to True.
# For coarse hyperparameter tuning, run the current preferred variant first.
# Change to ["A", "B", "C", "D", "E"] to compare all variants.
MODEL_VARIANTS_TO_RUN = ["C"]

# If True, choose the final test-evaluated variant using validation F2 after the
# max flagged-rate constraint. If False, FINAL_MODEL_VARIANT is used.
AUTO_SELECT_FINAL_VARIANT_BY_VALIDATION_F2 = True
FINAL_MODEL_ALGORITHM = "xgboost"
FINAL_MODEL_VARIANT = "C"
EVALUATE_TEST_FOR_ALL_VARIANTS = False


# -----------------------------------------------------------------------------
# Optional reduced source snapshot export
# -----------------------------------------------------------------------------
# When enabled, 00_split_data.py saves a source-level dataframe reduced to the
# context columns plus the feature columns required by the selected model variant.
# Included context columns: ID_COLS, DATE_COL, and TARGET_COL.
# The train/validation/test split column is intentionally excluded.
# Categorical features are kept in their original source format instead of being
# one-hot encoded in this export.
SAVE_REDUCED_SNAPSHOT_DATAFRAME = True
REDUCED_SNAPSHOT_MODEL_VARIANT = "C"
REDUCED_SNAPSHOT_OUTPUT_FILENAME = "06_snapshot_dataframe_model_C_reduced_snapshot.csv"

# -----------------------------------------------------------------------------
# Model algorithms
# -----------------------------------------------------------------------------
# When HYPERPARAMETER_TUNING_ENABLED=False, each algorithm listed here is run
# once using its default settings. When tuning is enabled, the current workflow
# tunes XGBoost only using HYPERPARAMETER_SEARCH_GRID.
MODEL_ALGORITHMS_TO_RUN = ["xgboost"]
HYPERPARAMETER_TUNING_ALGORITHM = "xgboost"

# -----------------------------------------------------------------------------
# XGBoost settings
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
USE_SCALE_POS_WEIGHT = True

# Project default parameters used when hyperparameter tuning is disabled.
# These are also the base parameters that HYPERPARAMETER_SEARCH_GRID overrides
# when hyperparameter tuning is enabled.
XGB_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# Default/non-tuned comparison models. These intentionally keep mostly native
# defaults plus reproducibility settings.
LIGHTGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "verbose": -1,
}


RANDOM_FOREST_DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# ----------------------------------------------------------------------------
# Hyperparameter tuning control
# ----------------------------------------------------------------------------
# Simple switch:
#   False -> run one CV pass using XGB_DEFAULT_PARAMS and the original
#            USE_SCALE_POS_WEIGHT behavior.
#   True  -> expand HYPERPARAMETER_SEARCH_GRID and select the parameter set with
#            the highest mean CV average_precision for each model variant.
HYPERPARAMETER_TUNING_ENABLED = False

# Coarse XGBoost grid search. Used only when HYPERPARAMETER_TUNING_ENABLED=True.
# You can configure this as either:
#   1) a dict of parameter ranges/lists, which creates a Cartesian product, or
#   2) a list of explicit parameter dictionaries.
#
# scale_pos_weight options:
#   - numeric values such as 1, 3, 5, 7 use that exact class weight.
#   - "auto" computes neg/pos separately inside each CV training fold.
#   - omit scale_pos_weight and set USE_SCALE_POS_WEIGHT=True to use "auto".
#
# This is intentionally coarse for a first pass. For faster runs, reduce values
# or run only MODEL_VARIANTS_TO_RUN = ["C"] for the base-feature experiment.
HYPERPARAMETER_SEARCH_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [3, 4],
    "learning_rate": [0.03],
    "min_child_weight": [5, 20],
    "subsample": [0.85],
    "colsample_bytree": [0.85],
    "scale_pos_weight": [1, 3, "auto"],
}

# Backward-compatible alias. You can ignore this and use
# HYPERPARAMETER_SEARCH_GRID going forward.
HYPERPARAMETER_GRID = HYPERPARAMETER_SEARCH_GRID

# -----------------------------------------------------------------------------
# Validation threshold selection and reporting
# -----------------------------------------------------------------------------
THRESHOLD_BETA = 2.0
MAX_FLAGGED_RATE = 0.15
THRESHOLD_GRID_SIZE = 1001
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

# Snapshot-row top-K reports answer questions such as: if we flag the top 5%
# highest-risk snapshot rows, what recall and precision do we get?
TOP_K_RATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

# Optional machine-level top-K reporting. This collapses repeated snapshots for
# the same machine into one machine-level risk score before ranking. It is often
# more actionable than snapshot-row top-K for inspection/worklist planning.
ENABLE_MACHINE_LEVEL_TOP_K = True
MACHINE_ID_COL = "model_id"
MACHINE_TOP_K_RATES = TOP_K_RATES
MACHINE_PROBABILITY_AGGREGATION = "max"  # allowed: "max", "mean", "latest"
MACHINE_TARGET_AGGREGATION = "max"       # allowed: "max", "latest"

# -----------------------------------------------------------------------------
# Output / persistence
# -----------------------------------------------------------------------------
SAVE_CV_PREDICTIONS = True
SAVE_VALIDATION_AND_TEST_PREDICTIONS = True
SAVE_FINAL_MODEL_ARTIFACTS = True
