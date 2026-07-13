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

# Default production path, assuming this project folder sits beside
# data_preparation/ under the service_controltower project root.
INPUT_DATA_PATH = PROJECT_DIR.parent / "data_preparation" / "output" / "snapshot_dataframe.csv"

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
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Inner split inside the training set only.
# feature_train is used to fit training-only feature ranking methods and the
# temporary XGBoost model. feature_selection_holdout is used for permutation and
# SHAP review only.
FEATURE_TRAIN_RATIO_WITHIN_TRAIN = 0.8
FEATURE_SELECTION_HOLDOUT_RATIO_WITHIN_TRAIN = 0.2

# Sort by DATE_COL first, then by these columns if they are present. Stable
# secondary sorting makes repeated runs deterministic when many rows share the
# same snapshot date.
SECONDARY_SORT_COLS = ["model_id"]

# -----------------------------------------------------------------------------
# Preprocessing / early feature screening
# -----------------------------------------------------------------------------
# Numeric features are imputed before model-based methods. Raw missingness is
# still reported before this step so you can review missingness patterns.
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
ONE_HOT_ENCODE_CATEGORICAL = True

# Early feature screening is fitted only on feature_train. These filters remove
# obviously unusable source columns before correlation, statistical tests,
# XGBoost, permutation, and SHAP are run. The reports still show exactly what was
# removed so you can review the decisions.
DROP_FEATURES_WITH_HIGH_MISSINGNESS = True
HIGH_MISSINGNESS_THRESHOLD = 0.1  # drop when feature_train missing rate is > 10%
DROP_ZERO_VARIANCE_FEATURES = True

# -----------------------------------------------------------------------------
# Correlation pruning settings
# -----------------------------------------------------------------------------
# Correlation pruning is based only on feature_train and only considers prepared
# features that appear in at least one grouped correlation pair with absolute
# correlation >= CORRELATION_REVIEW_THRESHOLD. The correlation step writes an
# editable confirmation file. Downstream steps drop only rows marked
# drop_confirmed=True in that file.
CORRELATION_REVIEW_THRESHOLD = 0.90
CORRELATION_AUTO_DROP_THRESHOLD = 0.95

# Editable file consumed by downstream steps. By default, step 02 creates this
# file only when it does not already exist so your manual edits are preserved
# across reruns of 02_correlation_analysis.py. Set
# CORRELATION_OVERWRITE_CONFIRMED_DROP_FILE=True only when you intentionally want
# to reset this editable file from the latest recommendations.
CORRELATION_CONFIRMED_DROP_FILE_NAME = "06_correlation_confirmed_drop_features_EDIT_ME.csv"
CORRELATION_OVERWRITE_CONFIRMED_DROP_FILE = False

# Fresh, non-editable template regenerated by step 02 every run for comparison
# against your manually edited confirmed-drop file.
CORRELATION_CONFIRMED_DROP_TEMPLATE_FILE_NAME = "06_correlation_confirmed_drop_features_TEMPLATE.csv"

CORRELATION_RESTRICT_DROPS_TO_REVIEWED_FEATURES = True
APPLY_CONFIRMED_CORRELATION_PRUNING_TO_DOWNSTREAM_STEPS = True

# -----------------------------------------------------------------------------
# Feature grouping for grouped correlation analysis
# -----------------------------------------------------------------------------
# Correlation is calculated within feature groups only. This greatly reduces the
# number of pairwise correlations versus a global all-vs-all correlation matrix.
# Rules are evaluated in order. The first matching rule assigns the group.
# Matching is case-insensitive and checks the raw source feature name.
FEATURE_GROUP_RULES = [
    {
        # Machine/master-data context. Put this first so any machine_* columns
        # carried through from the canonical backbone stay together instead of
        # being captured by generic operation/fault keywords.
        "group": "machine_context",
        "patterns": [
            r"^full_model$",
            r"^machine_",
            r"serial",
            r"region",
            r"dealer",
            r"customer",
            r"location",
        ],
    },
    {
        # Fluid/oil lab-result features. These patterns intentionally match the
        # exact lab-style frozen feature names from build_snapshot_dataframe.py,
        # for example Fe_Iron_PPM, Fuel_Fuel_PERCENT, Soot_Soot_Abs, and
        # Water_Water_PERCENT. Avoid broad words like "fuel" or "oil" here;
        # otherwise operation features such as fuel_actual_work_conflict_count
        # or maintenance features such as oil_reset_count could be mis-grouped.
        "group": "fluid_oil",
        "patterns": [
            r"^fluid_sample_",
            r"fluid_sample",
            r"days_since_last_fluid_sample",
            r"^(ag_silver|al_aluminum|cr_chromium|cu_copper|fe_iron|ni_nickel|pb_lead|sn_tin|ti_titanium|v_vanadium)_ppm$",
            r"^(k_potassium|li_lithium|na_sodium|si_silicon)_ppm$",
            r"^sediment_sediment_mg_per_l$",
            r"^solids_solids_percent$",
            r"^water_water_percent$",
            r"^fuel_fuel_percent$",
            r"^gly_glycol_percent$",
            r"^ethyleneglycol_ethylene_glycol_percent$",
            r"^polypropyleneglycol_polypropylene_glycol_percent$",
            r"^soot_soot_",
        ],
    },
    {
        # Fault-code and event-history features from fault_features_for_model().
        # Includes count/recency features, action-level features, evidence
        # scores, component fault counts, and the fault-specific SMR helper
        # features created in the fault source snapshot.
        "group": "fault_codes",
        "patterns": [
            r"fault",
            r"^action_",
            r"action_level",
            r"occurrence",
            r"evidence",
            r"mechanical",
            r"electrical",
            r"component",
            r"severity_score",
            r"max_log_occurrence",
            r"sum_log_occurrence",
        ],
    },
    {
        # Maintenance-monitor / PM features from maintenance_features_for_model().
        # This rule is evaluated before operation, so features like
        # engine_reset_count_180d remain maintenance rather than being captured
        # by an engine keyword.
        "group": "maintenance",
        "patterns": [
            r"maint",
            r"maintenance",
            r"service",
            r"pm_",
            r"monitor_",
            r"reset",
            r"overdue",
            r"due_now",
            r"due_or_overdue",
            r"remaining_hours",
            r"active_maintenance",
        ],
    },
    {
        # Warranty/prior-claim features. claim_next_45d is excluded as the target,
        # but prior claim counts/amounts/recency are candidate features and should
        # not be left in the default other group.
        "group": "warranty_prior",
        "patterns": [
            r"prior_claim",
            r"days_since_last_claim",
            r"has_prior_claim",
            r"unique_claim_type",
            r"warranty",
            r"claim",
        ],
    },
    {
        # Operation/utilization features from operation_features_for_model().
        # These patterns are source-specific enough to capture work, engine,
        # throttle, travel, steering, shift, and observed-day features while
        # avoiding fluid lab fuel and maintenance reset features above.
        "group": "operation",
        "patterns": [
            r"operation",
            r"actual_work",
            r"working_hours",
            r"work_idle",
            r"work_day",
            r"engine_running",
            r"engine_idling",
            r"engine_observed",
            r"engine_seconds",
            r"throttle",
            r"travel",
            r"moving_back_forth",
            r"steering",
            r"shift",
            r"fuel_actual_work",
            r"high_throttle",
            r"long_engine",
            r"has_operation",
            r"has_travel_data",
        ],
    },
    {
        # SMR/usage-meter features that are not source-specific maintenance,
        # fault, operation, or fluid sample features. Rules above intentionally
        # capture fault_smr_delta_90d, smr_since_last_reset, and
        # fluid_sample_latest_smr before this generic SMR group is reached.
        "group": "smr_usage",
        "patterns": [
            r"^smr_latest_hours$",
            r"^smr_delta_",
            r"^days_since_last_smr$",
            r"^smr_",
            r"usage",
            r"meter",
            r"machine_age",
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
