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
    {"lead_max_days": 90, "lead_min_days": 30},
    # {"lead_max_days": 120, "lead_min_days": 60},
    # {"lead_max_days": 120, "lead_min_days": 30},
    # {"lead_max_days": 180, "lead_min_days": 30},
]

# Number of matched controls sampled for each positive case in the case-control
# training design. This is the only matched-control ratio parameter.
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
    # "fault_dominant_component_window",
    # "maintenance_dominant_component_window",
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
# As-of-date population validation / test evaluation
# -----------------------------------------------------------------------------
# Training still uses the matched case-control dataset. Validation and final test
# can additionally use realistic as-of-date population evaluation datasets.
#
# A population as-of row mimics the future production scoring scenario:
#
#     as_of_date / window_end = a historical scoring date
#     window_start            = as_of_date - (lead_max_days - lead_min_days)
#     features                = source signals observed in that lookback window
#     evaluation label        = whether a claim occurs within the next N days
#
# Unlike the old random-negative evaluation, these rows are not forced to be
# negative. Some rows naturally become positives at 30, 60, 90, 120, 180, or
# 365 days depending on days_to_next_claim_on_or_after_window_end. This makes
# horizon trend evaluation meaningful.
ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION = True
ADD_ASOF_POPULATION_EVALUATION_TO_TEST = False

# Snapshot dates are sampled every N days across the validation/test date range.
# Each snapshot can include all eligible machines or a random cap. Use a cap for
# faster experiments and None for a fuller production-like snapshot.
ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS = 30
ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT = 1000  # set None to include all eligible machines
ASOF_EVALUATION_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW = True
ASOF_EVALUATION_REQUIRE_FUTURE_OBSERVABILITY = True
ASOF_EVALUATION_EXCLUDE_CLAIMS_DURING_OBSERVATION_WINDOW = False
ASOF_EVALUATION_MAX_ROWS_PER_SPLIT = None

# Optional down-sampling if the generated population evaluation set is too large.
# The default None uses all rows after the per-snapshot cap above.
VALIDATION_ASOF_EVALUATION_MAX_ROWS = None
TEST_ASOF_EVALUATION_MAX_ROWS = None

# Legacy random-negative evaluation is kept off by default. It can still be
# enabled for comparison, but the recommended validation/test design is now the
# as-of-date population evaluation above.
ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION = False
ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST = False
VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE = 10
POPULATION_RANDOM_NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END = CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END
POPULATION_RANDOM_NEGATIVE_REQUIRE_FUTURE_OBSERVABILITY = True
POPULATION_RANDOM_NEGATIVE_EXCLUDE_CLAIMS_DURING_OBSERVATION_WINDOW = True
POPULATION_RANDOM_NEGATIVE_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW = True
POPULATION_RANDOM_NEGATIVE_MAX_ATTEMPTS_MULTIPLIER = 80

# -----------------------------------------------------------------------------
# Validation report options
# -----------------------------------------------------------------------------
# Used by 04_fit_validate_model_report.py. The model is fitted on the full
# chronological training split and evaluated only on validation views. The test
# split is not evaluated until 07_final_test_evaluation.py.
VALIDATION_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
VALIDATION_SCORE_THRESHOLD = 0.50
VALIDATION_INCLUDE_FEATURE_COLUMNS = True
VALIDATION_SAVE_MODEL_ARTIFACTS = False

# -----------------------------------------------------------------------------
# Evaluation-only future-claim target
# -----------------------------------------------------------------------------
# Training always uses the original case-control `target` column. These settings
# only change validation / cross-validation / final-test metrics and prediction
# output columns. Use this when a window should be counted as a successful risk
# hit if the machine has a claim within a broader future horizon, for example
# the next 90 or 120 days, instead of only the original case-control label.
#
# Allowed modes:
#   "training_target"       -> evaluate with the original `target` column.
#   "claim_within_horizon"  -> evaluate with eval_target_claim_within_next_Nd.
#
# EVALUATION_CLAIM_HORIZON_DAYS can be either one integer or a list of integers.
# When it is a list, 04_fit_validate_model_report.py evaluates every horizon
# and writes horizon-trend summaries so you can review how positives, average
# precision, and top-k metrics change as the definition of "near future" is
# relaxed from short to longer horizons. Training target generation is unchanged.
EVALUATION_TARGET_MODE = "claim_within_horizon"
EVALUATION_CLAIM_HORIZON_DAYS = [30, 60, 90, 120, 180, 365]
EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END = True

# Backward-compatible aliases for older helper functions.
HOLDOUT_TOP_K_RATES = VALIDATION_TOP_K_RATES
SMOKE_TOP_K_RATES = VALIDATION_TOP_K_RATES

# -----------------------------------------------------------------------------
# Cross validation
# -----------------------------------------------------------------------------
# CV is performed only within the train split. Validation/test splits remain
# untouched by CV.
CV_N_SPLITS = 4
CV_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]

# User requested comparison models. For binary targets, logistic_regression is the
# classification-safe linear baseline. linear_regression is also supported as a
# ranking baseline with clipped predictions.
# MODELS_TO_RUN = ["linear_regression", "logistic_regression", "linear_svm", "random_forest", "xgboost"]
MODELS_TO_RUN = ["xgboost"]
SKIP_MISSING_OPTIONAL_ALGORITHMS = True

# Model parameters. Phase 1 intentionally keeps these simple and does not tune
# explicit L1/L2 regularization or early stopping.
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

# XGBoost learning-curve / early-stopping controls.
# Normal validation and hyperparameter-tuning runs can enable these, but the
# Phase 1 design sweep forcibly disables learning-curve artifacts and early
# stopping so the sweep remains fast and easy to review.
XGBOOST_ENABLE_LEARNING_CURVE = True
XGBOOST_LEARNING_CURVE_EVAL_VIEW = "matched_validation"
XGBOOST_USE_EARLY_STOPPING = False
XGBOOST_EARLY_STOPPING_ROUNDS = 0
XGBOOST_FIT_VERBOSE = False

# XGBoost class-importance control.
# Modes:
#   "auto"  -> compute negative rows / positive rows inside each training split.
#   "fixed" -> use XGBOOST_FIXED_SCALE_POS_WEIGHT.
#   "none"  -> do not add scale_pos_weight.
XGBOOST_CLASS_IMPORTANCE_MODE = "auto"  # allowed: "auto", "fixed", "none"
XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0

# Validation report output controls from 04_fit_validate_model_report.py.
# Detailed predictions / interpretation are useful for one-off validation and
# final review, but Phase 1 design sweep disables them to produce compact summaries.
VALIDATION_SAVE_DETAILED_OUTPUTS = True
VALIDATION_INCLUDE_FEATURE_COLUMNS = True
VALIDATION_SAVE_MODEL_ARTIFACTS = False

# Model interpretation outputs from 04_fit_validate_model_report.py.
SAVE_FEATURE_IMPORTANCE = True
SAVE_SHAP_VALUES = True
SHAP_EVALUATION_VIEWS = ["matched_validation", "population_like_validation"]
SHAP_MAX_ROWS = 1000
SHAP_TOP_SCORE_ROWS = 500
SHAP_RANDOM_ROWS = 500
SHAP_MAX_FEATURES_IN_ROW_OUTPUT = 50

# CV prediction files can be large.
SAVE_CV_PREDICTIONS = True


# -----------------------------------------------------------------------------
# Phase 1 design sweep grid
# -----------------------------------------------------------------------------
# 05_run_design_sweep.py expands this grid automatically and creates experiment
# ids from parameter values. Keep this grid focused on data-design and light
# class-weighting choices. Do not include test-set settings here.
DESIGN_SWEEP_GRID = {
    "CONTROLS_PER_POSITIVE_CASE": [3, 5, 10],
    # For validation, as-of population evaluation is now the recommended
    # production-like view. This cap controls runtime; set None in a one-off
    # validation run if you want all eligible machines per snapshot.
    "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT": [500, 1000],
    # scale_pos_weight is a shortcut interpreted by 05_run_design_sweep.py:
    #   "none" -> XGBOOST_CLASS_IMPORTANCE_MODE = "none"
    #   "auto" -> XGBOOST_CLASS_IMPORTANCE_MODE = "auto"
    #   number -> XGBOOST_CLASS_IMPORTANCE_MODE = "fixed" and that number is used.
    "scale_pos_weight": ["none", 3.0, "auto"],
}
DESIGN_SWEEP_FIXED_OVERRIDES = {
    "ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION": True,
    "ADD_ASOF_POPULATION_EVALUATION_TO_TEST": False,
    "ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION": False,
    "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST": False,
    "XGBOOST_ENABLE_LEARNING_CURVE": False,
    "XGBOOST_USE_EARLY_STOPPING": False,
    "XGBOOST_EARLY_STOPPING_ROUNDS": 0,
    "SAVE_FEATURE_IMPORTANCE": False,
    "SAVE_SHAP_VALUES": False,
    "VALIDATION_SAVE_DETAILED_OUTPUTS": False,
    "VALIDATION_INCLUDE_FEATURE_COLUMNS": False,
    "VALIDATION_SAVE_MODEL_ARTIFACTS": False,
}
# Phase 1 design sweep skips CV. It directly fits on the training split and
# evaluates validation views for compact comparison.
DESIGN_SWEEP_RUN_CROSS_VALIDATION = False
DESIGN_SWEEP_RUN_STEPS = [
    "02_build_case_control_dataset",
    "04_fit_validate_model_report",
]

# -----------------------------------------------------------------------------
# Phase 3 XGBoost hyperparameter tuning grid
# -----------------------------------------------------------------------------
# After Phase 1 selects a promising data design, set the fixed design below and
# run 06_tune_xgboost_hyperparameters.py. This script expands the coarse grid,
# fits on the training split, and chooses only from validation metrics.
HYPERPARAMETER_TUNING_DATA_DESIGN = {
    "CONTROLS_PER_POSITIVE_CASE": 5,
    "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT": 1000,
    "scale_pos_weight": 3.0,
}
HYPERPARAMETER_TUNING_GRID = {
    "max_depth": [2, 3],
    "min_child_weight": [5, 10],
    "subsample": [0.85],
    "colsample_bytree": [0.85],
    "gamma": [0, 1],
    "reg_lambda": [1, 10],
    "reg_alpha": [0, 0.1],
    "learning_rate": [0.03],
    "n_estimators": [800],
    # 0 disables early stopping. Positive values enable it with that round count.
    "early_stopping_rounds": [0, 50],
}
HYPERPARAMETER_TUNING_MAX_EXPERIMENTS = None
HYPERPARAMETER_TUNING_RUN_CROSS_VALIDATION = False

# -----------------------------------------------------------------------------
# Final locked-parameter test evaluation
# -----------------------------------------------------------------------------
# 07_final_test_evaluation.py is the only script intended to evaluate the test
# split. Edit these values after Phase 1 and Phase 3 are complete.
FINAL_CONTROLS_PER_POSITIVE_CASE = 5
FINAL_VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE = 10
FINAL_ADD_ASOF_POPULATION_EVALUATION_TO_TEST = True
FINAL_TEST_ASOF_EVALUATION_MAX_ROWS = None
FINAL_ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST = False
FINAL_TEST_RANDOM_NEGATIVES_PER_POSITIVE = 20
FINAL_XGBOOST_CLASS_IMPORTANCE_MODE = "fixed"
FINAL_XGBOOST_FIXED_SCALE_POS_WEIGHT = 3.0
FINAL_XGBOOST_PARAMS = {
    "n_estimators": 800,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 1,
    "reg_lambda": 10,
    "reg_alpha": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
FINAL_FIT_ON = "train_plus_validation"  # allowed: "train", "train_plus_validation"
FINAL_XGBOOST_USE_EARLY_STOPPING = False
FINAL_XGBOOST_EARLY_STOPPING_ROUNDS = 0
FINAL_SAVE_MODEL_ARTIFACT = True
FINAL_TEST_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
FINAL_TEST_SCORE_THRESHOLD = 0.50
FINAL_INCLUDE_FEATURE_COLUMNS = True

# Final test uses the same evaluation-only target convention by default. These
# settings do not affect final model fitting.
FINAL_EVALUATION_TARGET_MODE = EVALUATION_TARGET_MODE
FINAL_EVALUATION_CLAIM_HORIZON_DAYS = 120
