"""Configuration for case-control predictive-maintenance modeling.

Normal usage from this folder:

    python main.py

Expected project layout:

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

All generated artifacts are written under case_control_modeling/output/.
"""
from __future__ import annotations

from pathlib import Path

from feature_definitions import (
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    FROZEN_CATEGORICAL_FEATURES,
    FROZEN_NUMERIC_FEATURES,
)


# =============================================================================
# 1. Paths and source files
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR.parent / "enriched_data"

OUTPUT_DIR = PROJECT_DIR / "output"

# Central assets are intentionally outside per-experiment output folders. The
# design-sweep script temporarily changes OUTPUT_DIR, but every experiment must
# reuse the same machine assignments and validation/test row identities.
FIXED_SPLIT_ASSET_DIR = OUTPUT_DIR / "fixed_split_assets"

WARRANTY_FILE_CANDIDATES = ["warranty.csv", "warranty(5).csv", "warranty(4).csv", "warranty(3).csv"]
FAULT_CODES_FILE_CANDIDATES = ["fault_codes.csv", "fault_codes(3).csv", "fault_codes(2).csv", "fault_codes(1).csv"]
FLUID_SAMPLES_FILE_CANDIDATES = ["fluid_samples.csv", "fluid_samples(5).csv", "fluid_samples(4).csv", "fluid_samples(3).csv"]
MAINTENANCE_FILE_CANDIDATES = ["maintenance.csv", "maintenance(4).csv", "maintenance(3).csv", "maintenance(2).csv"]
OPERATION_FILE_CANDIDATES = [
    "operation.csv",
    "operation_partial.csv",
    "operation_partial(3).csv",
    "operation_partial(2).csv",
    "operation_partial(1).csv",
]


# =============================================================================
# 2. Data eligibility and claim construction
# =============================================================================
MIN_VALID_EVENT_DATE = "2015-01-01"
MAX_VALID_EVENT_DATE = "2026-06-26"
MIN_CLAIM_DATE = None
MAX_CLAIM_DATE = "2026-06-26"

CLAIM_EPISODE_GAP_DAYS = 30

# One switch controls whether warranty rows are filtered before claim episodes,
# training labels, validation labels, and future-claim timing are created:
#   "none"            - keep every date-eligible warranty row.
#   "failure_focused" - remove claim categories that are most likely to reflect
#                       inspections, administrative adjustments, or routine wear
#                       replacement rather than an unexpected machine failure.
WARRANTY_CLAIM_FILTER_MODE = "failure_focused"
WARRANTY_FAILURE_FOCUSED_EXCLUDED_CLAIM_TYPES = {
    "Inspection Claim KF": "inspection_or_check",
    "Policy Adjustment Claim": "policy_adjustment",
    "Commercial Adjustment Policy Claim": "commercial_adjustment",
    "Replacement Undercarriage": "routine_wear_component_replacement",
}

# This older optional rule remains independent of the claim-type filter. Keep it
# False unless the project explicitly requires every retained claim to identify a
# valid critical part number.
KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS = False
INVALID_CRITICAL_PART_VALUES = {"", "0", "0000", "000000", "nan", "none", "null"}

# "first" keeps only the first eligible claim per machine.
# "multiple" also keeps later eligible claims when the gap from the immediately
# previous eligible claim is at least lead_max_days for the active window.
POSITIVE_CLAIM_SELECTION_MODE = "multiple"


# =============================================================================
# 3. Modeling window and negative sampling
# =============================================================================
WINDOW_CONFIGS = [
    {"lead_max_days": 90, "lead_min_days": 30},
    # {"lead_max_days": 120, "lead_min_days": 30},
]

# One switch controls the TRAINING negative-case design:
#   "controlled" - same calendar window as the positive case, same full model.
#   "random"     - random eligible window from another machine of the same model.
#   "mixed"      - an even split of controlled and random negatives. When the
#                  requested total is odd, controlled negatives receive one extra.
# Validation/test rows are locked separately and do not change when this switch
# is varied during experiments.
NEGATIVE_SAMPLING_MODE = "mixed"
NEGATIVES_PER_POSITIVE_CASE = 3

# The same exclusion rules are applied to controlled and random negatives.
NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END = 180
NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START = 30
REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW = True

# Optional development cap. Set to None for the full experiment.
MAX_POSITIVE_CASES_PER_WINDOW = None
RANDOM_STATE = 42


# =============================================================================
# 4. Feature set
# =============================================================================
# One switch controls all modeling features:
#   "base"   - compact explainable case-control window features.
#   "frozen" - the previously frozen snapshot feature set.
FEATURE_SET = "base"
FLUID_SAMPLE_LOOKBACK_DAYS = 365

# Stable feature-name lists are defined in feature_definitions.py. Keeping
# them outside this file makes the experiment switches below easier to scan.


def refresh_derived_config() -> None:
    """Refresh feature lists after a sweep/final script changes FEATURE_SET."""
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES
    mode = str(FEATURE_SET).strip().lower()
    if mode == "base":
        NUMERIC_FEATURES = list(BASE_NUMERIC_FEATURES)
        CATEGORICAL_FEATURES = list(BASE_CATEGORICAL_FEATURES)
    elif mode == "frozen":
        NUMERIC_FEATURES = list(FROZEN_NUMERIC_FEATURES)
        CATEGORICAL_FEATURES = list(FROZEN_CATEGORICAL_FEATURES)
    else:
        raise ValueError("FEATURE_SET must be 'base' or 'frozen'.")


refresh_derived_config()


# =============================================================================
# 5. Fixed random train/validation/test split
# =============================================================================
# Every eligible machine is assigned to exactly one split. Assignments are
# deterministic, stratified by full_model and claim-history status, saved to CSV,
# and reused on reruns. Delete the assignment CSV only when a new holdout design
# is intentionally required.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
FIXED_SPLIT_RANDOM_STATE = 42

# Validation/test cohort design. This switch does not change training.
#
#   "random"
#       Positive windows are claim-anchored. Each negative machine receives one
#       independently sampled eligible no-claim window. Negatives are not matched
#       to positives by full model or calendar date.
#
#   "controlled"
#       Each positive claim window is matched to negative machines from the same
#       full model using the exact same calendar window.
#
#   "as_of_anchor"
#       Deployment-like fleet snapshot. One deterministic random as-of date is
#       selected and shared by all validation/test machines. Every row uses the
#       same calendar feature window ending on that date. The design target is 1
#       when the machine has a claim within lead_min_days after the as-of date;
#       otherwise it is 0. Ratio cohorts use nested randomly ranked negatives.
HOLDOUT_NEGATIVE_SAMPLING_MODE = "random"

# The as-of-date design searches daily candidate dates and requires at least this
# many positive machines in both validation and test when feasible. If no date
# reaches the requested minimum, the best feasible date is used and documented.
HOLDOUT_AS_OF_MIN_POSITIVE_MACHINES = 10

# Step 02 creates a fixed machine-level master pool and nested ratio datasets.
# The 1:1 and 5:1 cohorts use exactly the same positives; larger ratios only add
# negatives. The largest ratio is the reference validation/test dataset used by
# Steps 04 and 07.
HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS = [3]
HOLDOUT_RANDOM_STATE = 42

# When EVALUATION_TARGET_MODE="claim_within_horizon", Step 08 can handle the
# prevalence-ratio experiment in either of two ways:
#
#   "horizon_specific_random"
#       Recommended for prevalence sensitivity. For each split and each claim
#       horizon, randomly select a fixed positive set and a nested negative pool
#       large enough for every configured ratio. Thus 1:1, 2:1, ..., N:1 are
#       exact at every horizon. Samples may differ between horizons because the
#       true class definition changes with the horizon.
#
#   "reuse_same_rows"
#       Backward-compatible behavior. Reuse the same holdout rows for every
#       horizon and only recompute labels. The realized negative:positive ratio
#       is then allowed to change as the horizon grows.
HOLDOUT_HORIZON_RATIO_MODE = "reuse_same_rows"

# Step 08 ranking diagnostics. Percentage cutoffs show operational performance
# at a fixed fraction of the population. Fixed counts make the workload identical
# across prevalence ratios and are the preferred extreme-top degradation check.
HOLDOUT_TOP_N_COUNTS = [10, 20, 50]

# Machine-level stratified bootstrap settings for Step 08. Positives and negatives
# are resampled separately with replacement so each bootstrap replicate preserves
# the evaluated negative:positive ratio.
HOLDOUT_BOOTSTRAP_ENABLED = True
HOLDOUT_BOOTSTRAP_N_RESAMPLES = 1000
HOLDOUT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
HOLDOUT_BOOTSTRAP_RANDOM_STATE = 20260727


# =============================================================================
# 5B. Multi-anchor natural-prevalence fleet validation/test
# =============================================================================
# This optional deployment-style evaluation is independent of the ratio-sampled
# holdouts above. Each anchor is one fleet scoring date. All eligible validation
# or test machines at that date are retained at the natural observed prevalence,
# ranked within that date, and labeled by claims occurring after the anchor.
MULTI_ANCHOR_FLEET_ENABLED = True

# Explicit dates take precedence. Leave these lists empty to choose deterministic
# anchors automatically. Validation anchors are selected from the earlier feasible
# period; test anchors are selected from the later feasible period.
MULTI_ANCHOR_FLEET_VALIDATION_DATES = []
MULTI_ANCHOR_FLEET_TEST_DATES = []
MULTI_ANCHOR_FLEET_VALIDATION_ANCHOR_COUNT = 3
MULTI_ANCHOR_FLEET_TEST_ANCHOR_COUNT = 2

# Automatic anchor selection requirements. Anchors must have complete follow-up
# for the largest evaluation horizon and enough positive machines in the relevant
# split. Dates are kept apart to avoid nearly duplicate fleet snapshots.
MULTI_ANCHOR_FLEET_MIN_POSITIVE_MACHINES = 5
MULTI_ANCHOR_FLEET_MIN_ELIGIBLE_MACHINES = 50
MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS = 60
MULTI_ANCHOR_FLEET_VALIDATION_PERIOD_FRACTION = 0.70
MULTI_ANCHOR_FLEET_TEST_START_GAP_DAYS = 30
MULTI_ANCHOR_FLEET_RANDOM_STATE = 20260728

# None keeps the complete eligible fleet. A positive integer is only a development
# cap: all positives are kept first, then deterministic negatives fill the cap.
MULTI_ANCHOR_FLEET_MAX_MACHINES_PER_ANCHOR = None

# Step 10 reports ranking and actual claim timing at these fixed workloads.
MULTI_ANCHOR_FLEET_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
MULTI_ANCHOR_FLEET_TOP_N_COUNTS = [10, 20, 50]

# Anchor-fleet horizons are independently configurable. Leave empty to reuse
# EVALUATION_CLAIM_HORIZON_DAYS. Step 10 scores each snapshot once, then reports
# Top-K and Top-N trends separately for every horizon listed here.
MULTI_ANCHOR_FLEET_EVALUATION_HORIZONS = [30]

MULTI_ANCHOR_FLEET_BOOTSTRAP_ENABLED = True
MULTI_ANCHOR_FLEET_BOOTSTRAP_N_RESAMPLES = 200
MULTI_ANCHOR_FLEET_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95




# =============================================================================
# 5C. Negative-only score-distribution validation
# =============================================================================
# This optional diagnostic scores a validation/test cohort containing no positive
# rows under the design horizon (lead_min_days). It is intended to reveal whether
# the model assigns near-zero scores to ordinary negatives and how often it assigns
# high risk to machines whose claims occur only after the design horizon.
NEGATIVE_ONLY_EVALUATION_ENABLED = True
NEGATIVE_ONLY_EVALUATION_SPLITS = ["validation"]

# Random machine snapshots retained per split. Each machine contributes at most
# one row. Set to None to use the largest feasible balanced cohort.
NEGATIVE_ONLY_SAMPLE_SIZE_PER_SPLIT = 500

# Portion that must satisfy the full NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END rule.
# The remaining rows are design-horizon negatives whose next claim occurs after
# lead_min_days but no later than NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END.
NEGATIVE_ONLY_STRICT_FRACTION = 0.50

# Delayed-claim negatives are distributed as evenly as feasible across lead-time
# intervals such as 31-60, 61-90, ..., 151-180 days.
NEGATIVE_ONLY_DELAYED_CLAIM_BIN_WIDTH_DAYS = 30
NEGATIVE_ONLY_RANDOM_STATE = 20260729
NEGATIVE_ONLY_SCORE_THRESHOLD = 0.50
NEGATIVE_ONLY_SCORE_HISTOGRAM_BINS = 20
NEGATIVE_ONLY_SCORE_REVIEW_THRESHOLDS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Empty means reuse EVALUATION_CLAIM_HORIZON_DAYS. These columns show when a row
# that is negative at lead_min_days becomes positive under a relaxed horizon.
NEGATIVE_ONLY_EVALUATION_HORIZONS = [30, 60, 90]


# =============================================================================
# 5D. Positive-only score-distribution and threshold diagnostics
# =============================================================================
# Step 13 scores every positive row in the fixed validation/test evaluation
# cohorts for each configured future-claim horizon. It reports score
# distributions, actual days to claim, and the share of positives captured at
# candidate thresholds. No existing training or holdout design is changed.
POSITIVE_ONLY_EVALUATION_ENABLED = True
POSITIVE_ONLY_EVALUATION_SPLITS = ["validation", "test"]

# Empty means reuse EVALUATION_CLAIM_HORIZON_DAYS. The same model score is reused
# and only the true future-claim label changes by horizon.
POSITIVE_ONLY_EVALUATION_HORIZONS = [30, 60, 90]

# Thresholds used to show how many known positive machines would be captured.
POSITIVE_ONLY_SCORE_THRESHOLD = 0.50
POSITIVE_ONLY_SCORE_REVIEW_THRESHOLDS = [0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
POSITIVE_ONLY_SCORE_HISTOGRAM_BINS = 20

# When Step 12 outputs are available, Step 13 combines the positive and negative
# score distributions. It proposes threshold candidates that keep the empirical
# negative false-positive rate below these levels. The 5% and 1% rows are useful
# starting candidates for high-risk and critical-risk tiers, respectively.
POSITIVE_ONLY_THRESHOLD_FALSE_POSITIVE_TARGETS = [0.10, 0.05, 0.01]

# Because a case-control validation sample does not necessarily have production
# prevalence, Step 13 also estimates threshold precision at these hypothetical
# deployment positive rates using the observed TPR and FPR.
POSITIVE_ONLY_ASSUMED_DEPLOYMENT_PREVALENCES = [0.01, 0.05, 0.10, 0.20]

# None uses OUTPUT_DIR/12_negative_only_score_diagnostics. Set a directory only
# when Step 12 results were written elsewhere.
POSITIVE_ONLY_NEGATIVE_DIAGNOSTICS_DIR = None


# =============================================================================
# 6. Evaluation target, validation, cross-validation, and models
# =============================================================================
# One switch controls how CV, validation, and final-test metrics define a
# positive row. Model training always continues to use the original case-control
# `target` column.
#
#   "training_target"
#       Evaluate against the original case-control target.
#
#   "claim_within_horizon"
#       Evaluate whether the machine has a claim within each configured number
#       of days after window_end. Sample identities and model scores stay fixed,
#       while labels are recomputed separately for every listed horizon. Step 04,
#       Step 07, and Step 08 write horizon-specific results. With
#       HOLDOUT_NEGATIVE_SAMPLING_MODE="as_of_anchor", the active shared-date
#       deployment cohort is reused for all horizons. With random/controlled
#       holdouts, Step 02 also creates the separate outcome-independent horizon
#       cohort used by the standard validation and final-test reports.
#
# Rerun Step 02 when switching into claim_within_horizon so future-claim timing
# and any required horizon cohort are materialized. Changing the horizon list
# afterward does not change locked sample identities; metric code derives every
# label directly from days-to-next-claim.
EVALUATION_TARGET_MODE = "training_target"
EVALUATION_CLAIM_HORIZON_DAYS = [30, 60, 90]
EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END = True

VALIDATION_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
VALIDATION_SCORE_THRESHOLD = 0.50
VALIDATION_SAVE_DETAILED_OUTPUTS = True
VALIDATION_INCLUDE_FEATURE_COLUMNS = True
VALIDATION_SAVE_MODEL_ARTIFACTS = False

CV_N_SPLITS = 4
CV_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
SAVE_CV_PREDICTIONS = True

MODELS_TO_RUN = ["xgboost"]
SKIP_MISSING_OPTIONAL_ALGORITHMS = True

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
}
LINEAR_SVM_PARAMS = {"class_weight": "balanced", "max_iter": 5000}
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
    "n_estimators": 120,
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

XGBOOST_ENABLE_LEARNING_CURVE = False
XGBOOST_LEARNING_CURVE_EVAL_VIEW = "validation"
XGBOOST_USE_EARLY_STOPPING = False
XGBOOST_EARLY_STOPPING_ROUNDS = 0
XGBOOST_FIT_VERBOSE = False

# "auto" computes negative/positive rows from each training split.
# "fixed" uses XGBOOST_FIXED_SCALE_POS_WEIGHT. "none" omits the parameter.
XGBOOST_CLASS_IMPORTANCE_MODE = "none"
XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0

SAVE_FEATURE_IMPORTANCE = True
SAVE_SHAP_VALUES = False
SHAP_EVALUATION_VIEWS = ["validation"]
SHAP_MAX_ROWS = 1000
SHAP_TOP_SCORE_ROWS = 500
SHAP_RANDOM_ROWS = 500
SHAP_MAX_FEATURES_IN_ROW_OUTPUT = 50


# =============================================================================
# 7. Phase 1 data-design sweep
# =============================================================================
# Keep the default grid small. Add values only when you intentionally want a
# larger experiment matrix.
DESIGN_SWEEP_GRID = {
    "NEGATIVE_SAMPLING_MODE": ["controlled", "random", "mixed"],
    "NEGATIVES_PER_POSITIVE_CASE": [3],
    "FEATURE_SET": ["base", "frozen"],
    "scale_pos_weight": ["none", "auto"],
}
DESIGN_SWEEP_FIXED_OVERRIDES = {
    "XGBOOST_ENABLE_LEARNING_CURVE": False,
    "XGBOOST_USE_EARLY_STOPPING": False,
    "XGBOOST_EARLY_STOPPING_ROUNDS": 0,
    "SAVE_FEATURE_IMPORTANCE": False,
    "SAVE_SHAP_VALUES": False,
    "VALIDATION_SAVE_DETAILED_OUTPUTS": False,
    "VALIDATION_INCLUDE_FEATURE_COLUMNS": False,
    "VALIDATION_SAVE_MODEL_ARTIFACTS": False,
}
DESIGN_SWEEP_RUN_STEPS = [
    "02_build_case_control_dataset",
    "04_fit_validate_model_report",
]


# =============================================================================
# 8. XGBoost hyperparameter tuning
# =============================================================================
HYPERPARAMETER_TUNING_DATA_DESIGN = {
    "NEGATIVE_SAMPLING_MODE": "mixed",
    "NEGATIVES_PER_POSITIVE_CASE": 3,
    "FEATURE_SET": "frozen",
    "scale_pos_weight": "auto",
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
    "early_stopping_rounds": [0, 50],
}
HYPERPARAMETER_TUNING_MAX_EXPERIMENTS = None
HYPERPARAMETER_TUNING_RUN_CROSS_VALIDATION = False


# =============================================================================
# 9. Final locked test evaluation
# =============================================================================
FINAL_NEGATIVE_SAMPLING_MODE = "mixed"
FINAL_NEGATIVES_PER_POSITIVE_CASE = 3
FINAL_FEATURE_SET = "frozen"
FINAL_XGBOOST_CLASS_IMPORTANCE_MODE = "auto"
FINAL_XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0
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
FINAL_FIT_ON = "train_plus_validation"  # "train" or "train_plus_validation"
FINAL_XGBOOST_USE_EARLY_STOPPING = False
FINAL_XGBOOST_EARLY_STOPPING_ROUNDS = 0
FINAL_SAVE_MODEL_ARTIFACT = True
FINAL_TEST_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
FINAL_TEST_SCORE_THRESHOLD = 0.50
FINAL_INCLUDE_FEATURE_COLUMNS = True
# Final test intentionally reuses EVALUATION_TARGET_MODE and
# EVALUATION_CLAIM_HORIZON_DAYS from Section 6. There is no separate final-test
# target-mode switch.
