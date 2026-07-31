"""Configuration for the rolling-origin machine-risk model.

Expected local layout::

    project_root/
      enriched_data/
        fault_codes.csv
        fluid_samples.csv
        maintenance.csv
        operation.csv
        warranty.csv
        Physical Failure Events.csv
      rolling_origin_model/
        config.py
        ...scripts...

Set ``TARGET_SOURCE`` below to ``"physical_failure"`` or ``"warranty"``.
It can also be overridden without editing this file by setting the environment
variable ``RISK_TARGET_SOURCE`` or by using ``run_all.py --target-source ...``.
All artifacts are written to target-specific folders so one run cannot overwrite
another.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_ROOT / "enriched_data"


def _first_existing(*names: str) -> Path:
    """Return the first existing candidate under enriched_data.

    The first name is the preferred local filename. The additional names keep
    compatibility with the uploaded smoke-run filenames.
    """
    candidates = [DATA_DIR / name for name in names]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# ---------------------------------------------------------------------------
# Source files: all are expected under project_root/enriched_data/.
# ---------------------------------------------------------------------------
FAULT_FILE = _first_existing("fault_codes.csv", "fault_codes(5).csv")
FLUID_FILE = _first_existing("fluid_samples.csv", "fluid_samples(9).csv")
MAINTENANCE_FILE = _first_existing("maintenance.csv", "maintenance(7).csv")
OPERATION_RAW_FILE = _first_existing("operation.csv", "operation_partial(5).csv")
OPERATION_CLEAN_FILE = DATA_DIR / "operation_clean.csv"
OPERATION_ROSTER_FILE = DATA_DIR / "operation_machine_roster.csv"
OPERATION_FILE = OPERATION_CLEAN_FILE if OPERATION_CLEAN_FILE.exists() else OPERATION_RAW_FILE
WARRANTY_FILE = _first_existing("warranty.csv", "warranty(9).csv")
PHYSICAL_FAILURE_FILE = _first_existing(
    "Physical Failure Events.csv",
    "physical_failure_events.csv",
)

# ---------------------------------------------------------------------------
# Target switch.
# ---------------------------------------------------------------------------
SUPPORTED_TARGET_SOURCES = ("physical_failure", "warranty")
TARGET_SOURCE = os.getenv("RISK_TARGET_SOURCE", "physical_failure").strip().lower()
if TARGET_SOURCE not in SUPPORTED_TARGET_SOURCES:
    raise ValueError(
        f"TARGET_SOURCE must be one of {SUPPORTED_TARGET_SOURCES}; got {TARGET_SOURCE!r}."
    )

TARGET_SOURCE_CONFIG = {
    "physical_failure": {
        "file": PHYSICAL_FAILURE_FILE,
        "display_name": "Physical Failure Events",
        "history_noun": "target events",
    },
    "warranty": {
        "file": WARRANTY_FILE,
        "display_name": "Original Warranty Claims",
        "history_noun": "warranty claims",
    },
}
TARGET_FILE = Path(TARGET_SOURCE_CONFIG[TARGET_SOURCE]["file"])
TARGET_DISPLAY_NAME = str(TARGET_SOURCE_CONFIG[TARGET_SOURCE]["display_name"])
TARGET_HISTORY_NOUN = str(TARGET_SOURCE_CONFIG[TARGET_SOURCE]["history_noun"])

# The same generic column names are used for both targets. Since artifacts are
# isolated by target source, this makes the training/evaluation code identical.
TARGET_COLUMN = "target_event_0_90"
FUTURE_TARGET_COUNT_COLUMN = "future_target_event_count_90d"
HISTORY_FEATURES = (
    "days_since_prior_target_event",
    "prior_target_event_count_90d",
    "prior_target_event_count_365d",
    "prior_target_event_count_730d",
)

OUTPUT_ROOT = PROJECT_DIR / "outputs"
MODEL_ROOT = PROJECT_DIR / "models"
CHART_ROOT = PROJECT_DIR / "charts"
OUTPUT_DIR = OUTPUT_ROOT / TARGET_SOURCE
MODEL_DIR = MODEL_ROOT / TARGET_SOURCE
CHART_DIR = CHART_ROOT / TARGET_SOURCE

# ---------------------------------------------------------------------------
# Modeling frame.
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 90
HORIZON_DAYS = 90
SNAPSHOT_FREQUENCY = "MS"
TRAIN_SNAPSHOT_START = "2022-01-01"
TRAIN_SNAPSHOT_END = "2026-04-01"
LATEST_SCORE_DATE = "2026-07-01"
DENSE_ALL_MACHINE_GRID = True

# Fault-code feature design.
TOP_CODE_CANDIDATE_COUNT = 40
TOP_FAILURE_CODE_FEATURE_COUNT = 5
SERIOUS_ACTION_LEVEL_MIN = 2
SERIOUS_EVIDENCE_GROUP = "EVENT"
FAULT_RECENCY_HALF_LIFE_DAYS = 30.0

# Fluid LOCF design.
FLUID_LOCF_EXPIRY_DAYS = 365
FLUID_UNKNOWN_SEVERITY_VALUES = (5,)

# Honest rolling-origin evaluation.
PURGE_DAYS = 90
CALIBRATION_MONTHS = 3
VALIDATION_FOLDS = [
    ("2024H1", "2024-01-01", "2024-07-01"),
    ("2024H2", "2024-07-01", "2025-01-01"),
    ("2025H1", "2025-01-01", "2025-07-01"),
    ("2025H2", "2025-07-01", "2026-01-01"),
    ("2026Q1", "2026-01-01", "2026-05-01"),
]
NULL_REPLICATES_PER_FOLD = 5

XGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "early_stopping_rounds": 75,
}

RANDOM_SEED = 20260731
N_JOBS = 8

MODEL_VARIANTS = (
    "base27",
    "base27_plus_history",
)
PRODUCTION_VARIANT = "base27_plus_history"
ALGORITHMS = (
    "xgboost",
    "lightgbm",
    "hist_gradient_boosting",
    "logistic_regression",
)

WRITE_PICKLE = True
WRITE_COMPRESSED_CSV = True
WRITE_OOF_PREDICTIONS = True
RUN_ABLATION = True
RUN_SHUFFLED_LABEL_NULL = True

# ---------------------------------------------------------------------------
# Deployment-style multi-anchor fleet evaluation.
# ---------------------------------------------------------------------------
MULTI_ANCHOR_FLEET_ENABLED = True
MULTI_ANCHOR_FLEET_MODEL_VARIANTS = ("base27", "base27_plus_history")
MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD = 3
MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS = 30
MULTI_ANCHOR_FLEET_RANDOM_STATE = 20260731
MULTI_ANCHOR_FLEET_TOP_K_RATES = (0.01, 0.05, 0.10)
MULTI_ANCHOR_FLEET_TOP_N_COUNTS = (5, 10, 20)
MULTI_ANCHOR_FLEET_OUTPUT_SUBDIR = "10_multi_anchor_fleet_evaluation"
MULTI_ANCHOR_FLEET_WRITE_PER_ANCHOR_FILES = True
MULTI_ANCHOR_FLEET_EVALUATE_CALIBRATION_PERIOD = False

# Use the exact same deployment anchors for both target-source runs. Set this
# to None to return to reproducible random date sampling inside each fold.
# The original warranty extract ends on 2026-06-26, so its 2026-04-28
# anchor is right-censored before the complete 90-day outcome horizon. Keeping
# this True reproduces the prior same-date comparison, while every output row is
# marked with outcome_window_complete. Set False for strict production validation.
MULTI_ANCHOR_ALLOW_INCOMPLETE_FIXED_DATES = True
MULTI_ANCHOR_FLEET_FIXED_DATES = {
    "2024H1": ("2024-02-03", "2024-04-17", "2024-06-01"),
    "2024H2": ("2024-08-03", "2024-10-24", "2024-12-10"),
    "2025H1": ("2025-02-14", "2025-04-03", "2025-05-22"),
    "2025H2": ("2025-07-19", "2025-10-27", "2025-11-28"),
    "2026Q1": ("2026-01-14", "2026-02-22", "2026-04-28"),
}

OPERATION_CACHE_FORCE_REBUILD = False
