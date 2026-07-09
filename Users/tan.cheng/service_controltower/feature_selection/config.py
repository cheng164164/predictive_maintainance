"""
Configuration for the feature-selection analysis project.

This file is intentionally the only place where normal run-time settings should
be edited. The analysis is designed to run without command-line arguments:

    python main.py

The workflow is non-cross-validation based:
1. Split the full unified snapshot dataframe chronologically into
   training/validation/test.
2. Split the training portion again into feature_train and
   feature_selection_holdout.
3. Use feature_train for training-only feature diagnostics/rankings.
4. Use feature_selection_holdout for permutation and SHAP review.
5. Preserve validation_holdout and test_holdout for later model validation and
   final reporting; they are not used for feature ranking in this script.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Default assumes the unified snapshot dataset is one folder above this project.
# Replace this with the production unified snapshot file path when ready.
INPUT_DATA_PATH = PROJECT_DIR.parent / "snapshot_dataframe_mini.csv"

# All generated CSV, JSON, Excel, markdown, and plot outputs are written here.
OUTPUT_DIR = PROJECT_DIR / "output"

# -----------------------------------------------------------------------------
# Dataset columns
# -----------------------------------------------------------------------------
TARGET_COL = "claim_next_45d"
DATE_COL = "snapshot_date"

# Columns that identify a row or define the target/time axis should not be used
# as model features. full_model is intentionally not excluded by default because
# it may be useful as a machine-context categorical feature. Add it here if you
# do not want it one-hot encoded and evaluated.
ID_COLS = ["model_id"]
EXCLUDE_FEATURE_COLS = [TARGET_COL, DATE_COL] + ID_COLS

# Optional manual include/exclude behavior.
# If CANDIDATE_FEATURE_COLS is empty, all columns except EXCLUDE_FEATURE_COLS are
# used as feature candidates. MANUAL_DROP_FEATURE_COLS can remove specific
# columns from that automatically discovered feature list.
CANDIDATE_FEATURE_COLS = []
MANUAL_DROP_FEATURE_COLS = []

# -----------------------------------------------------------------------------
# Chronological splits
# -----------------------------------------------------------------------------
# Full dataset chronological split: train / validation / test.
# The values are treated as ratio weights. The requested 0.75/0.15/0.15 sums to
# 1.05, so the script normalizes the shares internally and records the effective
# normalized shares in 00_run_summary.json.
TRAIN_RATIO = 0.75
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Inner split inside the training set only.
# feature_train is used to fit training-only feature ranking methods and the
# temporary XGBoost model. feature_selection_holdout is used for permutation and
# SHAP review only.
FEATURE_TRAIN_RATIO_WITHIN_TRAIN = 0.75
FEATURE_SELECTION_HOLDOUT_RATIO_WITHIN_TRAIN = 0.25

# Sort by DATE_COL first, then by these columns if they are present. Stable
# secondary sorting makes repeated runs deterministic when many rows share the
# same snapshot date.
SECONDARY_SORT_COLS = ["model_id"]

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
# Numeric features are imputed before model-based methods. Raw missingness is
# still reported before this step so you can review missingness patterns.
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
ONE_HOT_ENCODE_CATEGORICAL = True

# -----------------------------------------------------------------------------
# Feature grouping for grouped correlation analysis
# -----------------------------------------------------------------------------
# Correlation is calculated within feature groups only. This greatly reduces the
# number of pairwise correlations versus a global all-vs-all correlation matrix.
# Rules are evaluated in order. The first matching rule assigns the group.
# Matching is case-insensitive and checks the raw source feature name.
FEATURE_GROUP_RULES = [
    {
        "group": "fault_codes",
        "patterns": [
            r"fault",
            r"dtc",
            r"diagnostic",
            r"trouble",
            r"action_level",
            r"action_",
            r"occurrence",
            r"severity",
        ],
    },
    {
        "group": "maintenance",
        "patterns": [
            r"maint",
            r"maintenance",
            r"service",
            r"pm_",
            r"reset",
            r"overdue",
            r"repair",
        ],
    },
    {
        "group": "operation",
        "patterns": [
            r"operation",
            r"operating",
            r"engine",
            r"idle",
            r"idling",
            r"throttle",
            r"travel",
            r"work",
            r"working",
            r"steer",
            r"shift",
            r"load",
            r"fuel",
        ],
    },
    {
        "group": "fluid_oil",
        "patterns": [
            r"fluid",
            r"oil",
            r"sample",
            r"contamination",
            r"viscosity",
            r"coolant",
            r"lab",
        ],
    },
    {
        "group": "smr_usage",
        "patterns": [
            r"smr",
            r"hour",
            r"hours",
            r"usage",
            r"meter",
            r"age",
        ],
    },
    {
        "group": "machine_context",
        "patterns": [
            r"full_model",
            r"model",
            r"serial",
            r"region",
            r"dealer",
            r"customer",
            r"location",
        ],
    },
]
DEFAULT_FEATURE_GROUP = "other"

# -----------------------------------------------------------------------------
# Statistical feature-ranking settings
# -----------------------------------------------------------------------------
MUTUAL_INFO_RANDOM_STATE = 42

# Mutual information calculation mode:
# - "quantile_binned": fast MI estimate using quantile bins for continuous
#   features.
# - "sklearn": uses sklearn.feature_selection.mutual_info_classif directly,
#   which can be slower on larger/wider datasets.
MUTUAL_INFO_MODE = "quantile_binned"
MUTUAL_INFO_N_BINS = 10

# -----------------------------------------------------------------------------
# XGBoost settings
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
XGB_PARAMS = {
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

# -----------------------------------------------------------------------------
# Permutation and SHAP settings
# -----------------------------------------------------------------------------
# Permutation importance uses F2 as requested. F2 emphasizes recall more than
# precision. Because sklearn permutation_importance calls model.predict for an
# F-beta scorer, this uses the model's default classification threshold.
PERMUTATION_SCORING = "f2"
PERMUTATION_N_REPEATS = 10
PERMUTATION_RANDOM_STATE = 42
PERMUTATION_N_JOBS = 1

# SHAP can be expensive on large datasets. This row cap affects only the SHAP
# reporting sample from feature_selection_holdout, not model training.
SHAP_MAX_ROWS = 3000
SHAP_RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
GENERATE_EXCEL_REPORT = True
GENERATE_PLOTS = True
REPORT_TOP_N = 50

# File names
EXCEL_REPORT_NAME = "feature_selection_report.xlsx"
MARKDOWN_REPORT_NAME = "feature_selection_summary.md"
