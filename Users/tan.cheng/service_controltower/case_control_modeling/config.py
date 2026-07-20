"""
Configuration for window-based case-control predictive-maintenance modeling.

Normal usage from this folder:

    python main.py

The scripts assume this folder lives beside an enriched_data/ folder:

    project_root/
        enriched_data/
            warranty.csv
            fault_codes.csv
            fluid_samples.csv
            maintenance.csv
            operation.csv
        case_control_modeling/
            config.py
            main.py
            ...

All outputs are written under case_control_modeling/output/.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR.parent / "enriched_data"

# Convenience fallback for local smoke testing when files are attached directly
# to a ChatGPT / notebook workspace. In your project, enriched_data/ should be used.
if not SOURCE_DIR.exists() and Path("/mnt/data").exists():
    SOURCE_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "output"

# Candidate file names. The first existing file in SOURCE_DIR is used.
WARRANTY_FILE_CANDIDATES = ["warranty.csv", "warranty(4).csv", "warranty(3).csv"]
FAULT_CODES_FILE_CANDIDATES = ["fault_codes.csv", "fault_codes(2).csv", "fault_codes(1).csv"]
FLUID_SAMPLES_FILE_CANDIDATES = ["fluid_samples.csv", "fluid_samples(4).csv", "fluid_samples(3).csv"]
MAINTENANCE_FILE_CANDIDATES = ["maintenance.csv", "maintenance(3).csv", "maintenance(2).csv"]
OPERATION_FILE_CANDIDATES = ["operation.csv", "operation_partial.csv", "operation_partial(2).csv", "operation_partial(1).csv"]

# -----------------------------------------------------------------------------
# Date / data cleaning boundaries
# -----------------------------------------------------------------------------
# Used to remove impossible future source records, especially from fluid samples.
# Set to None to disable.
MAX_VALID_EVENT_DATE = "2026-06-26"
MIN_VALID_EVENT_DATE = "2015-01-01"

# Optional claim-date bounds. Usually leave as None for first pass.
MIN_CLAIM_DATE = None
MAX_CLAIM_DATE = "2026-06-26"

# -----------------------------------------------------------------------------
# Claim episode construction
# -----------------------------------------------------------------------------
# Multiple warranty rows for the same machine close together are grouped into
# one claim episode so that the model does not treat closely related claim rows
# as independent failures.
CLAIM_EPISODE_GAP_DAYS = 30

# Optional target filter. For first concept validation, keep all claims.
# Later, you can set KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS=True to reduce target noise.
KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS = False
INVALID_CRITICAL_PART_VALUES = {"", "0", "0000", "000000", "nan", "none", "null"}

# -----------------------------------------------------------------------------
# Positive claim selection mode
# -----------------------------------------------------------------------------
# Controls which warranty claim events are allowed to become positive case rows.
#
# "first"
#   Use only the first claim event for each machine. This matches a first-failure
#   modeling design and prevents repeat-claim history from influencing the target.
#
# "multiple"
#   Use the first claim event for each machine, plus later claim events only when
#   the later claim is at least lead_max_days after the immediately previous
#   claim event for the same machine. The lead_max_days value is read from each
#   WINDOW_CONFIGS entry, so every selected claim has a clean monitoring window
#   of the same configured length before the claim.
#
# Note: claim events come from 01_build_claim_episodes.py. If CLAIM_EPISODE_GAP_DAYS
# is greater than zero, very close warranty rows are first grouped into claim
# episodes. Set CLAIM_EPISODE_GAP_DAYS = 0 if you want almost one event per raw
# warranty claim date instead of episode grouping.
POSITIVE_CLAIM_SELECTION_MODE = "first"  # allowed: "first", "multiple"

# -----------------------------------------------------------------------------
# Window-based case-control design
# -----------------------------------------------------------------------------
# Each positive training row is one claim episode with an observation window
# before the claim:
#     window_start = claim_date - lead_max_days
#     window_end   = claim_date - lead_min_days
#
# Example: lead_max_days=120 and lead_min_days=30 means the model sees evidence
# from 120 to 30 days before the claim. Evidence after window_end is intentionally
# excluded to preserve action lead time.
WINDOW_CONFIGS = [
    {"name": "lead_120_to_30", "lead_max_days": 120, "lead_min_days": 30},
    {"name": "lead_180_to_30", "lead_max_days": 180, "lead_min_days": 30},
]

CONTROLS_PER_POSITIVE_CASE = 3
RANDOM_STATE = 42

# Negative-control eligibility.
# A control uses the same calendar observation window as the positive case.
# It must not have a claim soon after the window_end.
CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END = 180
CONTROL_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START = 30
CONTROL_MATCH_ON_FULL_MODEL = True
CONTROL_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW = True
REQUIRE_POSITIVE_SOURCE_COVERAGE_OVERLAP_WINDOW = True

# If True, controls are preferentially sampled from machines with latest SMR
# closest to the positive case at window_end. If operation data is unavailable,
# the script falls back to random sampling.
MATCH_CONTROLS_BY_LATEST_OPERATION_SMR = False

# Optional limit for quick debugging. Default uses 1000 positives per window for fast concept validation.
# Set to None for the full experiment after the workflow is validated.
MAX_POSITIVE_CASES_PER_WINDOW = None

# -----------------------------------------------------------------------------
# Preliminary base feature set for concept validation
# -----------------------------------------------------------------------------
# These are intentionally not too many. They focus on strong, explainable signals
# from each source inside the observation window, plus prior warranty context.
BASE_NUMERIC_FEATURES = [
    # Prior warranty context before the observation window
    "prior_claim_count_before_window",
    "days_since_prior_claim_before_window",

    # Source availability / coverage flags
    "has_any_source_window",
    "source_record_count_window",

    # Fault-code signals inside the observation window
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

    # Fluid-sample signals inside the observation window
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

    # Maintenance signals inside the observation window
    "has_maintenance_window",
    "maintenance_event_count_window",
    "maintenance_monitor_reset_count_window",
    "maintenance_overdue_count_window",
    "maintenance_due_now_count_window",
    "maintenance_min_remaining_hours_window",
    "maintenance_days_since_latest_event_window",

    # Operation / usage signals inside the observation window
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

# -----------------------------------------------------------------------------
# Optional granular component-level feature add-on
# -----------------------------------------------------------------------------
# Set ENABLE_COMPONENT_FEATURES=False to run the original compact base feature set.
# When enabled, the dataset builder adds component-specific features from the
# fault-code and maintenance sources, such as engine / hydraulic / powertrain
# fault counts and maintenance due/overdue/reset counts.
ENABLE_COMPONENT_FEATURES = False

# Component groups are keyword based. The code lowercases source text and searches
# these keywords across fault related/applicable component text, fault error text,
# maintenance related components, maintenance event name, and maintenance type.
COMPONENT_FEATURE_GROUPS = {
    "engine": ["engine", "eng", "kccv", "aircleaner", "fuel", "oil filter", "engine oil"],
    "hydraulic": ["hydraulic", "hyd", "hst"],
    "power_train": ["power train", "powertrain", "transmission", "hst"],
    "work_equipment": ["work equipment", "workequipment", "we", "final case", "final drive"],
    "urea_scr": ["urea", "scr", "adblue", "def"],
    "cooling": ["cooling", "coolant", "radiator"],
    "control": ["control", "monitor", "mon", "controller", "sensor", "solenoid"],
    "transmission": ["transmission", "t/m", "tm", "hst filter"],
    "final_drive": ["final drive", "final case"],
}

# Fault-code component add-on features generated for each group:
#   fault_component_<group>_count_window
#   fault_component_<group>_l03plus_count_window
#   fault_component_<group>_max_action_level_window
#   fault_component_<group>_max_evidence_score_window
#
# Maintenance component add-on features generated for each group:
#   maintenance_component_<group>_count_window
#   maintenance_component_<group>_overdue_count_window
#   maintenance_component_<group>_due_now_count_window
#   maintenance_component_<group>_monitor_reset_count_window
#   maintenance_component_<group>_min_remaining_hours_window

def _component_numeric_features():
    out = []
    for group in COMPONENT_FEATURE_GROUPS:
        out.extend([
            # f"fault_component_{group}_count_window",
            f"fault_component_{group}_l03plus_count_window",
            # f"fault_component_{group}_max_action_level_window",
            f"fault_component_{group}_max_evidence_score_window",
            f"maintenance_component_{group}_count_window",
            f"maintenance_component_{group}_overdue_count_window",
            f"maintenance_component_{group}_due_now_count_window",
            # f"maintenance_component_{group}_monitor_reset_count_window",
            # f"maintenance_component_{group}_min_remaining_hours_window",
        ])
    return out

COMPONENT_NUMERIC_FEATURES = _component_numeric_features() if ENABLE_COMPONENT_FEATURES else []
COMPONENT_CATEGORICAL_FEATURES = []

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + COMPONENT_NUMERIC_FEATURES
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + COMPONENT_CATEGORICAL_FEATURES

# -----------------------------------------------------------------------------
# Train / validation / test split
# -----------------------------------------------------------------------------
# Split is applied at case-control-group level, not row level, so each positive
# case and its sampled controls stay together. The split is chronological by
# SPLIT_DATE_COL, normally window_end, which is the prediction date for the
# completed observation window.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_DATE_COL = "window_end"

# -----------------------------------------------------------------------------
# Smoke run / validation evaluation
# -----------------------------------------------------------------------------
# The smoke run trains on the train split and evaluates on the validation split.
SMOKE_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]

# -----------------------------------------------------------------------------
# Cross validation
# -----------------------------------------------------------------------------
# CV is performed only within the train split. Validation/test splits remain
# untouched for holdout evaluation and final reporting.
CV_N_SPLITS = 4
CV_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]

# User requested comparison models. For binary targets, logistic_regression is the
# classification-safe linear baseline. linear_regression is also supported as a
# ranking baseline with clipped predictions.
# MODELS_TO_RUN = ["linear_regression", "logistic_regression", "linear_svm", "random_forest", "xgboost"]
MODELS_TO_RUN = ["xgboost"]
SKIP_MISSING_OPTIONAL_ALGORITHMS = True

# Model parameters. Keep simple for concept validation.
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
}
LINEAR_SVM_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 5000,
}
RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# XGBoost class-importance control.
# scale_pos_weight is the standard XGBoost binary-class importance parameter:
#     scale_pos_weight = negative rows / positive rows
# Modes:
#   "auto"  -> compute neg/pos inside each smoke/CV training split.
#   "fixed" -> use XGBOOST_FIXED_SCALE_POS_WEIGHT.
#   "none"  -> do not add scale_pos_weight.
# For case-control data with 3 controls per case, auto usually resolves to about 3.0.
XGBOOST_CLASS_IMPORTANCE_MODE = "auto"  # allowed: "auto", "fixed", "none"
XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0

# CV prediction files can be large. Keep True for first diagnosis; set False later.
SAVE_CV_PREDICTIONS = True
SAVE_SMOKE_PREDICTIONS = True

# -----------------------------------------------------------------------------
# Validation prediction report
# -----------------------------------------------------------------------------
# Step 05 trains each configured model on the chronological training split and
# scores the chronological validation split. The exported files are intended for
# inspection at both machine-window level and machine summary level.
VALIDATION_SCORE_THRESHOLD = 0.50
VALIDATION_TOP_K_RATES = SMOKE_TOP_K_RATES
VALIDATION_INCLUDE_FEATURE_COLUMNS = True
VALIDATION_SAVE_MODEL_ARTIFACTS = False
