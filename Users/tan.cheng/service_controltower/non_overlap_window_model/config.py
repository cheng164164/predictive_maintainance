"""Central configuration for the exact 90-day future-event risk model.

Expected local layout
---------------------
project_root/
  enriched_data/
    fault_codes.csv
    fluid_samples.csv
    maintenance.csv
    operation.csv
    warranty.csv
    Physical Failure Events.csv
  future_claim_predictive_model_90d/
    config.py
    01_validate_inputs.py
    ...

Users run the numbered scripts without command-line arguments. Every path,
target choice, time window, algorithm, threshold, and anchor-evaluation control
is defined in this file.
"""
from __future__ import annotations

from pathlib import Path


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_ROOT / "enriched_data"


def _resolve_source_file(preferred_name: str, aliases: tuple[str, ...]) -> Path:
    preferred = DATA_DIR / preferred_name
    if preferred.exists():
        return preferred
    matches: list[Path] = []
    for pattern in aliases:
        matches.extend(sorted(DATA_DIR.glob(pattern)))
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else preferred


FAULT_CODES_PATH = _resolve_source_file("fault_codes.csv", ("fault_codes*.csv", "*fault*code*.csv"))
FLUID_SAMPLES_PATH = _resolve_source_file("fluid_samples.csv", ("fluid_samples*.csv", "*fluid*sample*.csv"))
MAINTENANCE_PATH = _resolve_source_file("maintenance.csv", ("maintenance*.csv", "*maint*.csv"))
OPERATION_PATH = _resolve_source_file("operation.csv", ("operation*.csv", "*operation*partial*.csv"))
WARRANTY_PATH = _resolve_source_file("warranty.csv", ("warranty*.csv", "*warranty*claim*.csv"))
PHYSICAL_FAILURE_PATH = _resolve_source_file(
    "Physical Failure Events.csv",
    ("physical_failure_events*.csv", "physical_failure_event*.csv", "*Physical*Failure*Events*.csv"),
)


# =============================================================================
# 2. TARGET-SOURCE SWITCH
# =============================================================================
# Choose exactly one target source. No command-line argument is needed.
#   "warranty"         -> original warranty.csv machine-day events
#   "physical_failure" -> Physical Failure Events.csv machine-day events
TARGET_SOURCE = "physical_failure"

# Warranty cleaning mode:
#   "all_rows_machine_day"  reproduces the prior 90-day comparison experiment.
#   "original_script_cleaned" applies claim type, part, and minimum-SMR filters.
WARRANTY_FILTER_MODE = "all_rows_machine_day"
WARRANTY_ALLOWED_CLAIM_TYPES = (
    "Standard Warranty",
    "Advantage Claim",
    "Parts & Components Claim",
    "Remanufactured Components Claim",
)
WARRANTY_MIN_FAILURE_SMR = 25.0
WARRANTY_INVALID_PART_CODES = (
    "", "0", "00", "000", "0000", "00000", "000000",
    "N/A", "NA", "NONE", "NULL", "UNKNOWN",
)


# =============================================================================
# 3. 90-DAY MODEL DESIGN
# =============================================================================
LOOKBACK_DAYS = 90
SEGMENT_STRIDE_DAYS = 90
HORIZON_DAYS = 90
ANALYSIS_START = "2021-05-01"
ANALYSIS_END = "2026-04-30"

# Non-overlapping native chronological evaluation.
NATIVE_VALIDATION_PERIODS = 1
NATIVE_LOCKED_TEST_PERIODS = 2

# Fixed common outcome cutoff used in the prior two-target experiment. Keeping
# this date makes warranty and physical-failure runs use identical native
# train/validation/test periods. Set to None to use the selected target file's
# latest event date.
LABEL_OBSERVATION_END_OVERRIDE = "2026-06-26"

# Feature variants:
#   reviewed140_with_history: condition features plus prior selected-target events
#   reviewed134_condition_only: removes the six prior-event-history features
MODEL_VARIANT = "reviewed140_with_history"

# Candidate screen and final selected model. For the fastest smoke run, leave
# RUN_CANDIDATE_SCREEN=False; XGBoost is still trained and validated.
RUN_CANDIDATE_SCREEN = False
CANDIDATE_ALGORITHMS = (
    "logistic_regression",
    "hist_gradient_boosting",
    "extra_trees",
    "xgboost",
    "lightgbm",
)
SELECTED_ALGORITHM = "xgboost"

# Compatibility names used by shared modeling utilities.
MODEL_VARIANTS = (MODEL_VARIANT,)
PRIMARY_VARIANT = MODEL_VARIANT
PRIMARY_ALGORITHM = SELECTED_ALGORITHM

# Threshold is learned from the native validation period by Platt calibration
# plus F1 maximization. It is then frozen for locked-test and anchor scoring.
THRESHOLD_TUNING_METRIC = "f1"


# =============================================================================
# 4. DEPLOYMENT-STYLE MULTI-ANCHOR FLEET VALIDATION
# =============================================================================
VALIDATION_FOLDS = (
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
    ("2025H2", "2025-07-01", "2026-01-01"),
    ("2026Q1", "2026-01-01", "2026-05-01"),
)
ANCHOR_DATES_BY_FOLD = {
    "2024H1": ("2024-02-03", "2024-04-17", "2024-06-01"),
    "2024H2": ("2024-08-03", "2024-10-24", "2024-12-10"),
    "2025H1": ("2025-02-14", "2025-04-03", "2025-05-22"),
    "2025H2": ("2025-07-19", "2025-10-27", "2025-11-28"),
    "2026Q1": ("2026-01-14", "2026-02-22", "2026-04-28"),
}
TOP_K_RATES = (0.01, 0.05, 0.10)
TOP_N_COUNTS = (5, 10, 20)
WRITE_FULL_ANCHOR_FEATURES = True
WRITE_FULL_RANKED_ANCHOR_FILES = True


# =============================================================================
# 5. REPRODUCIBILITY AND COMPUTE
# =============================================================================
RANDOM_SEED = 20260731
N_JOBS = 8
REUSE_CONDITION_FEATURE_CACHE = True
FORCE_REBUILD_CONDITION_FEATURES = False


# =============================================================================
# 6. OUTPUT AND ARTIFACT PATHS
# =============================================================================
# Target-specific folders prevent one target run from overwriting the other.
TARGET_RUN_NAME = TARGET_SOURCE
OUTPUT_ROOT = PROJECT_DIR / "outputs" / TARGET_RUN_NAME
ARTIFACT_ROOT = PROJECT_DIR / "artifacts" / TARGET_RUN_NAME
CACHE_DIR = PROJECT_DIR / "cache_90d_condition_features"
LOG_ROOT = PROJECT_DIR / "logs" / TARGET_RUN_NAME

STEP_01_OUTPUT_DIR = OUTPUT_ROOT / "01_validate_inputs"
STEP_02_OUTPUT_DIR = OUTPUT_ROOT / "02_define_data_splits"
STEP_03_OUTPUT_DIR = OUTPUT_ROOT / "03_build_features_targets"
STEP_04_OUTPUT_DIR = OUTPUT_ROOT / "04_select_features"
STEP_05_OUTPUT_DIR = OUTPUT_ROOT / "05_train_candidate_models"
STEP_06_OUTPUT_DIR = OUTPUT_ROOT / "06_validate_candidate_models"
STEP_07_OUTPUT_DIR = OUTPUT_ROOT / "07_calibrate_selected_model"
STEP_08_OUTPUT_DIR = OUTPUT_ROOT / "08_finalize_model"
STEP_09_OUTPUT_DIR = OUTPUT_ROOT / "09_evaluate_locked_test"
STEP_10_OUTPUT_DIR = OUTPUT_ROOT / "10_consolidate_results"
STEP_11_OUTPUT_DIR = OUTPUT_ROOT / "11_score_latest"
STEP_12_OUTPUT_DIR = OUTPUT_ROOT / "12_generate_report"
STEP_13_OUTPUT_DIR = OUTPUT_ROOT / "13_validate_outputs"
STEP_14_OUTPUT_DIR = OUTPUT_ROOT / "14_anchor_fleet_validation"

SPLIT_DEFINITION_PATH = STEP_02_OUTPUT_DIR / "native_split_definition.json"
SPLIT_PLAN_PATH = STEP_02_OUTPUT_DIR / "machine_period_split_plan.csv.gz"
FULL_DATASET_PATH = STEP_03_OUTPUT_DIR / "segment_dataset_90d.pkl"
ANCHOR_DATASET_PATH = STEP_03_OUTPUT_DIR / "anchor_dataset_90d.pkl"
MODEL_FEATURE_LIST_PATH = STEP_04_OUTPUT_DIR / "model_feature_list.csv"
CANDIDATE_ARTIFACT_PATH = ARTIFACT_ROOT / "candidate_models.joblib"
SELECTED_MODEL_ARTIFACT_PATH = ARTIFACT_ROOT / "selected_calibrated_model.joblib"
FINAL_MODEL_ARTIFACT_PATH = ARTIFACT_ROOT / "final_model.joblib"


def split_dataset_path(split_name: str) -> Path:
    valid = {"training", "validation", "locked_test", "purged_gap", "label_incomplete"}
    if split_name not in valid:
        raise ValueError(f"split_name must be one of {sorted(valid)}")
    return STEP_03_OUTPUT_DIR / f"{split_name}.csv.gz"


# =============================================================================
# 7. PIPELINE EXECUTION RANGE
# =============================================================================
# run_all_steps.py runs this inclusive range. Step 14 is deliberately separate
# because it retrains one fold-specific model per anchor-validation fold.
PIPELINE_START_STEP = 1
PIPELINE_END_STEP = 13
RUN_ANCHOR_VALIDATION_IN_PIPELINE = False
