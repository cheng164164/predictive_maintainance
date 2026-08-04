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

All user-controlled settings are kept in this file. Set ``TARGET_SOURCE`` to
``"physical_failure"`` or ``"warranty"``. Feature variants may be configured as
one string or as multiple strings; both forms are normalized automatically.
"""
from __future__ import annotations

from inspect import Parameter
from pathlib import Path
from typing import Iterable
import json

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
DATA_DIR = PROJECT_ROOT / "enriched_data"


def _first_existing(*names: str) -> Path:
    """Return the first existing candidate under enriched_data."""
    candidates = [DATA_DIR / name for name in names]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# ---------------------------------------------------------------------------
# Source files: all are expected under project_root/enriched_data/.
# ---------------------------------------------------------------------------
FAULT_FILE = _first_existing(
    "fault_codes.csv",
    "fault_codes(6).csv",
    "fault_codes(5).csv",
)
FLUID_FILE = _first_existing(
    "fluid_samples.csv",
    "fluid_samples(10).csv",
    "fluid_samples(9).csv",
)
MAINTENANCE_FILE = _first_existing(
    "maintenance.csv",
    "maintenance(8).csv",
    "maintenance(7).csv",
)
OPERATION_RAW_FILE = _first_existing(
    "operation.csv",
    "operation_partial(8).csv",
    "operation_partial(5).csv",
)
OPERATION_CLEAN_FILE = DATA_DIR / "operation_clean.csv"
OPERATION_ROSTER_FILE = DATA_DIR / "operation_machine_roster.csv"
OPERATION_FILE = OPERATION_CLEAN_FILE if OPERATION_CLEAN_FILE.exists() else OPERATION_RAW_FILE
WARRANTY_FILE = _first_existing("warranty.csv", "warranty(9).csv")
PHYSICAL_FAILURE_FILE = _first_existing(
    "Physical Failure Events.csv",
    "Physical Failure Events(2).csv",
    "physical_failure_events.csv",
    "physical_failure_event.csv",
)

# ---------------------------------------------------------------------------
# Target switch.
# ---------------------------------------------------------------------------
SUPPORTED_TARGET_SOURCES = ("physical_failure", "warranty")
TARGET_SOURCE = "physical_failure"
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
# Production scoring of an incoming three-month source refresh.
#
# The incoming directory is expected to contain the most recent source files
# for the complete active fleet. The production scoring script merges those
# files with the retained historical sources only where longer memory is
# required (for example prior target history, PM resets, and fluid LOCF).
# ---------------------------------------------------------------------------
INCOMING_DATA_DIR = PROJECT_ROOT / "incoming_data"
PRODUCTION_SCORING_OUTPUT_DIR = OUTPUT_DIR / "production_scoring"
PRODUCTION_SCORING_WORK_DIR = OUTPUT_DIR / "production_scoring_work"
INCOMING_SOURCE_HISTORY_DAYS = 90
INCOMING_SCORE_DATE = None  # Prefer --score-date YYYY-MM-DD in production.
INCOMING_REQUIRE_COMPLETE_FLEET_OPERATION = False
INCOMING_ALLOW_MISSING_TARGET_REFRESH = False
INCOMING_SOURCE_FILE_ALIASES = {
    "fault": ("fault_codes.csv", "fault_codes(6).csv", "fault_codes(5).csv"),
    "fluid": ("fluid_samples.csv", "fluid_samples(10).csv", "fluid_samples(9).csv"),
    "maintenance": ("maintenance.csv", "maintenance(8).csv", "maintenance(7).csv"),
    "operation": ("operation.csv", "operation_partial(8).csv", "operation_partial(5).csv"),
    "physical_failure": (
        "Physical Failure Events.csv",
        "Physical Failure Events(2).csv",
        "physical_failure_events.csv",
    ),
    "warranty": ("warranty.csv", "warranty(9).csv"),
}

# ---------------------------------------------------------------------------
# Modeling frame.
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 90
HORIZON_DAYS = 90
SNAPSHOT_FREQUENCY = "MS"
TRAIN_SNAPSHOT_START = "2022-01-01"
TRAIN_SNAPSHOT_END = "2026-04-01"
DENSE_ALL_MACHINE_GRID = True
SNAPSHOT_BUILD_RESUME = True

# Retention windows used when preparing a temporary scoring bundle. Fault and
# operation features consume only the incoming lookback window. Maintenance
# and target history retain enough prior data for capped recency/count
# features. Fluid history is intentionally not truncated because the
# worsening-trend feature may need the sample immediately preceding the
# current LOCF sample.
SCORING_HISTORY_RETENTION_DAYS = {
    "fault": LOOKBACK_DAYS,
    "operation": LOOKBACK_DAYS,
    "maintenance": 1096,
    "physical_failure": 1096,
    "warranty": 1096,
    "fluid": None,
}

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
# Baseline retained for repeatable comparison in 02_tune_xgboost.py.
# For XGBoost’s tree booster, the defaults are:
# Parameter	Default	Meaning
# reg_alpha	0	No L1 regularization
# reg_lambda	1	L2 regularization is still applied
# early_stopping_rounds	None	Early stopping is disabled
XGB_TUNING_BASELINE_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "early_stopping_rounds": 75,
}

# Approved XGBoost settings selected through rolling-origin tuning. Re-run
# 02_tune_xgboost.py whenever the development dataset or feature design changes,
# then review the validation evidence before promoting new values.
XGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    # Keep aucpr last: XGBoost early stopping monitors the last metric on the
    # last evaluation set, which is the reserved calibration period.
    "eval_metric": ["logloss", "aucpr"],
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

# Controlled tuning grid: all other XGBoost parameters remain fixed so the
# effects of L1, L2, and early stopping can be audited directly.
XGB_TUNING_OUTPUT_SUBDIR = "02_xgboost_tuning"
XGB_TUNING_VARIANT = "base27_plus_history"
XGB_TUNING_REG_ALPHA_VALUES = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
XGB_TUNING_REG_LAMBDA_VALUES = (1.0, 2.0, 5.0, 10.0)
XGB_TUNING_EARLY_STOPPING_ROUNDS_VALUES = (30, 50, 75)
XGB_TUNING_BASE_PARAMS = {
    key: value
    for key, value in XGB_TUNING_BASELINE_PARAMS.items()
    if key not in {"reg_alpha", "reg_lambda", "early_stopping_rounds"}
}

# Future retraining workflow: 02_tune_xgboost.py writes this file. Each later
# pipeline subprocess imports config.py again and therefore uses the newly
# selected parameters automatically. The explicit XGB_PARAMS block above is a
# stable fallback and documents the currently approved settings.
XGB_TUNED_PARAMS_FILE = MODEL_DIR / "xgboost_tuned_params.json"
XGB_LOAD_TUNED_PARAMS_FILE = True
if XGB_LOAD_TUNED_PARAMS_FILE and XGB_TUNED_PARAMS_FILE.exists():
    try:
        _tuned_payload = json.loads(XGB_TUNED_PARAMS_FILE.read_text(encoding="utf-8"))
        _tuned_params = _tuned_payload.get("params", _tuned_payload)
        if not isinstance(_tuned_params, dict):
            raise TypeError("Tuned parameter payload must contain a dictionary.")
        XGB_PARAMS.update(_tuned_params)
    except Exception as exc:
        raise ValueError(
            f"Could not load tuned XGBoost parameters from {XGB_TUNED_PARAMS_FILE}: {exc}"
        ) from exc

RANDOM_SEED = 20260731
N_JOBS = -1

# Final-model diagnostic exports.
FINAL_MODEL_DIAGNOSTICS_ENABLED = True
FEATURE_IMPORTANCE_TOP_N = 25
SHAP_SAMPLE_SIZE = 5000
SHAP_TOP_N = 25
OVERFIT_AUCPR_GAP_WARNING = 0.10
OVERFIT_AUCPR_DETERIORATION_WARNING = 0.02

# ---------------------------------------------------------------------------
# Probability calibration and operating-threshold selection.
#
# The model is fitted on the historical fit set. The reserved calibration set
# is then used to fit the probability calibrator and select one operating
# threshold. Both are frozen before forward validation or anchor scoring.
# ---------------------------------------------------------------------------
PROBABILITY_CALIBRATION_ENABLED = True
SUPPORTED_CALIBRATION_METHODS = ("platt", "none")
PROBABILITY_CALIBRATION_METHOD = "platt"
if PROBABILITY_CALIBRATION_METHOD not in SUPPORTED_CALIBRATION_METHODS:
    raise ValueError(
        "PROBABILITY_CALIBRATION_METHOD must be one of "
        f"{SUPPORTED_CALIBRATION_METHODS}; got "
        f"{PROBABILITY_CALIBRATION_METHOD!r}."
    )

SUPPORTED_THRESHOLD_METRICS = ("f1", "f2")
THRESHOLD_SELECTION_METRIC = "f1"
if THRESHOLD_SELECTION_METRIC not in SUPPORTED_THRESHOLD_METRICS:
    raise ValueError(
        f"THRESHOLD_SELECTION_METRIC must be one of {SUPPORTED_THRESHOLD_METRICS}; "
        f"got {THRESHOLD_SELECTION_METRIC!r}."
    )

# Candidate thresholds are generated from probability quantiles. This keeps
# runtime stable even when many machine snapshots have identical scores.
THRESHOLD_GRID_SIZE = 1001
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
THRESHOLD_TIE_BREAKER = "higher_precision"
CALIBRATION_ECE_BINS = 10

# The two standalone diagnostic scripts use these model sets. None means use
# exactly MODEL_VARIANTS / ALGORITHMS after those settings are normalized.
CALIBRATION_MODEL_VARIANTS = None
CALIBRATION_ALGORITHMS = None
CALIBRATION_OUTPUT_SUBDIR = "04_probability_calibration"
THRESHOLD_OUTPUT_SUBDIR = "05_operating_threshold_selection"
# ---------------------------------------------------------------------------
# Feature/model variant selection.
#
# ONE option:     MODEL_VARIANTS = "base27_plus_history"
# MULTIPLE:       MODEL_VARIANTS = ("base27", "base27_plus_history")
# Lists also work: MODEL_VARIANTS = ["base27", "base27_plus_history"]
# ---------------------------------------------------------------------------
SUPPORTED_MODEL_VARIANTS = (
    "base27",
    "base27_plus_history",
    "base27_plus_history_liveness",
    "enhanced_condition",
    "enhanced_plus_history",
)


def normalize_model_variants(
    value: str | Iterable[str],
    setting_name: str,
) -> tuple[str, ...]:
    """Normalize a single variant or a collection into a validated tuple.

    A plain string is treated as one complete model-variant name. This prevents
    Python from iterating over the characters in ``"base27_plus_history"``.
    """
    if isinstance(value, str):
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f"{setting_name} must be a variant string or an iterable of strings."
            ) from exc

    cleaned: list[str] = []
    for item in raw_values:
        if not isinstance(item, str):
            raise TypeError(
                f"{setting_name} contains a non-string value: {item!r}."
            )
        variant = item.strip().lower()
        if not variant:
            raise ValueError(f"{setting_name} contains an empty variant name.")
        if variant not in cleaned:
            cleaned.append(variant)

    if not cleaned:
        raise ValueError(f"{setting_name} cannot be empty.")

    unknown = [name for name in cleaned if name not in SUPPORTED_MODEL_VARIANTS]
    if unknown:
        raise ValueError(
            f"{setting_name} contains unsupported variants {unknown}. "
            f"Supported variants are {SUPPORTED_MODEL_VARIANTS}."
        )
    return tuple(cleaned)


# Select one string or multiple strings here.
MODEL_VARIANTS = "base27_plus_history"
MODEL_VARIANTS = normalize_model_variants(MODEL_VARIANTS, "MODEL_VARIANTS")

# Used by final scoring when multiple variants are trained.
PRODUCTION_VARIANT = "base27_plus_history"
PRODUCTION_VARIANT = PRODUCTION_VARIANT.strip().lower()
if PRODUCTION_VARIANT not in SUPPORTED_MODEL_VARIANTS:
    raise ValueError(
        f"PRODUCTION_VARIANT must be one of {SUPPORTED_MODEL_VARIANTS}; "
        f"got {PRODUCTION_VARIANT!r}."
    )
if PRODUCTION_VARIANT not in MODEL_VARIANTS:
    raise ValueError(
        "PRODUCTION_VARIANT must also be included in MODEL_VARIANTS. "
        f"MODEL_VARIANTS={MODEL_VARIANTS}, PRODUCTION_VARIANT={PRODUCTION_VARIANT!r}."
    )

SUPPORTED_ALGORITHMS = (
    "xgboost",
    "lightgbm",
    "catboost",
    "hist_gradient_boosting",
    "logistic_regression",
)


def normalize_algorithms(
    value: str | Iterable[str],
    setting_name: str,
) -> tuple[str, ...]:
    """Normalize one algorithm name or an iterable into a validated tuple."""
    if isinstance(value, str):
        raw_values = (value,)
    else:
        try:
            raw_values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f"{setting_name} must be an algorithm string or an iterable of strings."
            ) from exc

    cleaned: list[str] = []
    for item in raw_values:
        if not isinstance(item, str):
            raise TypeError(
                f"{setting_name} contains a non-string value: {item!r}."
            )
        algorithm = item.strip().lower()
        if not algorithm:
            raise ValueError(f"{setting_name} contains an empty algorithm name.")
        if algorithm not in cleaned:
            cleaned.append(algorithm)

    if not cleaned:
        raise ValueError(f"{setting_name} cannot be empty.")

    unknown = [name for name in cleaned if name not in SUPPORTED_ALGORITHMS]
    if unknown:
        raise ValueError(
            f"{setting_name} contains unsupported algorithms {unknown}. "
            f"Supported algorithms are {SUPPORTED_ALGORITHMS}."
        )
    return tuple(cleaned)


# Algorithms used by the regular rolling-origin validation.
ALGORITHMS = (
    "xgboost",
    # "lightgbm",
    # "catboost",
    # "hist_gradient_boosting",
    # "logistic_regression",
)
ALGORITHMS = normalize_algorithms(ALGORITHMS, "ALGORITHMS")

if CALIBRATION_MODEL_VARIANTS is None:
    CALIBRATION_MODEL_VARIANTS = MODEL_VARIANTS
else:
    CALIBRATION_MODEL_VARIANTS = normalize_model_variants(
        CALIBRATION_MODEL_VARIANTS, "CALIBRATION_MODEL_VARIANTS"
    )

if CALIBRATION_ALGORITHMS is None:
    CALIBRATION_ALGORITHMS = ALGORITHMS
else:
    CALIBRATION_ALGORITHMS = normalize_algorithms(
        CALIBRATION_ALGORITHMS, "CALIBRATION_ALGORITHMS"
    )

WRITE_PICKLE = True
WRITE_COMPRESSED_CSV = True
WRITE_OOF_PREDICTIONS = True

# ---------------------------------------------------------------------------
# Deployment-style multi-anchor fleet evaluation.
# ---------------------------------------------------------------------------
MULTI_ANCHOR_FLEET_ENABLED = True

# None means: automatically use exactly MODEL_VARIANTS.
# You may also set one string or multiple strings independently, for example:
# MULTI_ANCHOR_FLEET_MODEL_VARIANTS = "base27_plus_history"
# MULTI_ANCHOR_FLEET_MODEL_VARIANTS = ("base27", "base27_plus_history")
MULTI_ANCHOR_FLEET_MODEL_VARIANTS = None
if MULTI_ANCHOR_FLEET_MODEL_VARIANTS is None:
    MULTI_ANCHOR_FLEET_MODEL_VARIANTS = MODEL_VARIANTS
else:
    MULTI_ANCHOR_FLEET_MODEL_VARIANTS = normalize_model_variants(
        MULTI_ANCHOR_FLEET_MODEL_VARIANTS,
        "MULTI_ANCHOR_FLEET_MODEL_VARIANTS",
    )

# None means use the same algorithms as ALGORITHMS. A single string or a tuple/list
# can also be configured independently for anchor validation.
# Example: MULTI_ANCHOR_FLEET_ALGORITHMS = ("xgboost", "lightgbm")
MULTI_ANCHOR_FLEET_ALGORITHMS = ("xgboost",)
if MULTI_ANCHOR_FLEET_ALGORITHMS is None:
    MULTI_ANCHOR_FLEET_ALGORITHMS = ALGORITHMS
else:
    MULTI_ANCHOR_FLEET_ALGORITHMS = normalize_algorithms(
        MULTI_ANCHOR_FLEET_ALGORITHMS,
        "MULTI_ANCHOR_FLEET_ALGORITHMS",
    )

# Continue to the next configured algorithm if an optional dependency is missing
# or one algorithm fails. Completed algorithms are checkpointed to disk.
MULTI_ANCHOR_FLEET_CONTINUE_ON_ALGORITHM_ERROR = True
MULTI_ANCHOR_FLEET_RESUME = True

# Use more deterministic anchor dates than the earlier three-date design.
# Evenly spaced anchors reduce random-date sensitivity and improve the
# anchor-cluster bootstrap used for tier policy confidence intervals.
MULTI_ANCHOR_FLEET_ANCHOR_MODE = "evenly_spaced"
SUPPORTED_MULTI_ANCHOR_MODES = ("evenly_spaced", "reproducible_random_dates")
MULTI_ANCHOR_FLEET_ANCHORS_PER_FOLD = 5
MULTI_ANCHOR_FLEET_MIN_DAYS_BETWEEN_ANCHORS = 25
MULTI_ANCHOR_FLEET_RANDOM_STATE = 20260731
MULTI_ANCHOR_FLEET_TOP_K_RATES = (0.01, 0.05, 0.10)
# Coarse fixed Top-N grid used for routine capacity checks.
TIER_TOP_N_GRID_START = 5
TIER_TOP_N_GRID_MAX = 200
TIER_TOP_N_GRID_STEP = 5
# Critical precision may be unattainable at the minimum coarse boundary N=5.
# In that case, evaluate N=1,2,3,4 under the same confidence rule. If no
# confidence-qualified boundary exists anywhere, Critical falls back to the N
# in 1..5 with the highest pooled validation precision. This fallback is
# explicitly flagged because it does not carry the 95% confidence guarantee.
TIER_CRITICAL_FINE_SCAN_ENABLED = True
TIER_CRITICAL_FINE_SCAN_START = 1
TIER_CRITICAL_FINE_SCAN_MAX = TIER_TOP_N_GRID_START - 1
TIER_CRITICAL_FINE_SCAN_STEP = 1
TIER_CRITICAL_MAX_PRECISION_FALLBACK_ENABLED = True
TIER_CRITICAL_FALLBACK_MIN_N = 1
TIER_CRITICAL_FALLBACK_MAX_N = TIER_TOP_N_GRID_START
TIER_CRITICAL_FALLBACK_SELECTION_METRIC = "micro_precision"
MULTI_ANCHOR_FLEET_TOP_N_COUNTS = tuple(
    range(TIER_TOP_N_GRID_START, TIER_TOP_N_GRID_MAX + 1, TIER_TOP_N_GRID_STEP)
)
MULTI_ANCHOR_FLEET_OUTPUT_SUBDIR = "06_multi_anchor_validation"
MULTI_ANCHOR_FLEET_WRITE_PER_ANCHOR_FILES = True
MULTI_ANCHOR_FLEET_WRITE_FEATURE_DATASET = False
MULTI_ANCHOR_FLEET_EVALUATE_CALIBRATION_PERIOD = False

MULTI_ANCHOR_ALLOW_INCOMPLETE_FIXED_DATES = False
# Set to a fold/date mapping to force common dates across target sources.
# None uses the deterministic anchor mode above.
MULTI_ANCHOR_FLEET_FIXED_DATES = None


# ---------------------------------------------------------------------------
# Precision-governed risk-tier policy.
#
# Top-N boundaries are cumulative: Critical is ranks 1..N_critical; High
# extends through N_high; Medium extends through N_medium. The non-overlapping
# tier bands are evaluated separately and exported for audit.
# ---------------------------------------------------------------------------
TIER_POLICY_ENABLED = True
TIER_POLICY_OUTPUT_SUBDIR = "07_risk_tier_policy"
TIER_POLICY_VALIDATION_ALGORITHM = "xgboost"
TIER_PRECISION_TARGETS = {
    "CRITICAL": 0.85,
    "HIGH": 0.75,
    "MEDIUM": 0.60,
}
TIER_CONFIDENCE_LEVEL = 0.95
TIER_BOOTSTRAP_REPLICATES = 1000
TIER_MIN_ANCHOR_COUNT = 5
TIER_MIN_ANCHOR_PASS_RATE = 0.50
# Require each non-overlapping tier band, not only the cumulative Top-N pool,
# to meet its own precision target. Tiers are selected sequentially in severity
# order so each enabled band is audited against the preceding enabled boundary.
TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION = True
TIER_COMBINATION_SELECTION_OBJECTIVE = "strict_sequential_severity_capacity"
# High and Medium must satisfy their targets at the lower 95% anchor-cluster
# bootstrap confidence bound. Critical first follows the same strict rule, but
# may use the explicitly marked maximum-precision N=1..5 fallback configured
# above when no confidence-qualified Critical boundary exists.
TIER_SELECTION_POLICY = "strict_confidence_with_critical_precision_fallback"
SUPPORTED_TIER_SELECTION_POLICIES = (
    "strict_confidence_with_critical_precision_fallback",
)
# A candidate must exceed the historical lower-tail boundary score. With 0.10,
# approximately 90% of accepted historical anchor boundaries were stronger.
TIER_SCORE_GATE_QUANTILE = 0.10
# point_quantile uses the empirical lower-tail boundary score.
# conservative_upper_ci uses the upper 95% bootstrap bound of that quantile,
# making candidate confirmation more conservative under threshold uncertainty.
TIER_SCORE_GATE_CONFIDENCE_MODE = "conservative_upper_ci"
SUPPORTED_TIER_SCORE_GATE_CONFIDENCE_MODES = (
    "point_quantile",
    "conservative_upper_ci",
)
SUPPORTED_TIER_SCORE_CONFIRMATION_RULES = (
    "both_calibrated_and_risk_index",
    "either_calibrated_or_risk_index",
    "calibrated_probability_only",
    "risk_index_only",
    "all_three_scores",
)
TIER_SCORE_CONFIRMATION_RULE = "risk_index_only"
FINAL_TIER_THRESHOLD_OUTPUT_SUBDIR = "08_final_tier_thresholds"

# Customer-facing risk_index definition. It is the calibrated probability of
# the configured target event within HORIZON_DAYS, expressed as a percentage.
# A tiny clip keeps the value strictly inside (0, 100) without materially
# changing the probability interpretation.
RISK_INDEX_PROBABILITY_EPSILON = 1e-6
RISK_INDEX_DEFINITION = (
    "100 x calibrated probability of the configured target event within "
    f"the next {HORIZON_DAYS} days"
)

# ---------------------------------------------------------------------------
# Approved production model and tier policy.
#
# These values are the release-controlled settings used by both the standard
# latest-scoring script and the incoming-data scoring script. Retraining may
# generate new candidate parameters and thresholds, but they should be copied
# here only after validation and stakeholder approval. This makes deployment
# deterministic and prevents an unreviewed training run from silently changing
# customer-facing risk tiers.
# ---------------------------------------------------------------------------
PRODUCTION_MODEL_SETTINGS = {
    "variant": "base27_plus_history",
    "algorithm": "xgboost",
    "model_file": "xgboost_base27_plus_history.json",
    "metadata_file": "model_metadata_base27_plus_history.json",
    "best_iteration": 76,
    "calibration_method": "platt",
    "platt_coefficient": 0.7762747621032015,
    "platt_intercept": -0.4873194660663577,
    "operating_threshold": 0.247760287346681,
    "threshold_metric": "f1",
    "risk_horizon_days": HORIZON_DAYS,
}

# Raw fault codes corresponding to the five supervised code features in the
# approved model. Keeping the raw values here avoids trying to reverse the
# sanitized feature names when a future incoming extract contains only 90 days.
PRODUCTION_SELECTED_FAULT_CODES = (
    "B@BCNS",
    "CA234",
    "B@BCQA",
    "6091NX",
    "CA2249",
)

# Cumulative Top-N boundaries and conservative score gates approved from the
# tuned High=75% policy. A machine must first fall inside a cumulative Top-N
# boundary and then pass the configured risk-index gate. Failed candidates are
# demoted and tested against the next lower enabled tier.
PRODUCTION_TIER_POLICY = {
    "policy_version": 4,
    "algorithm": "xgboost",
    "variant": "base27_plus_history",
    "candidate_selection": "nested cumulative fixed Top-N boundaries",
    "confirmation_rule": "risk_index_only",
    "confidence_level": TIER_CONFIDENCE_LEVEL,
    "risk_index_definition": RISK_INDEX_DEFINITION,
    "risk_horizon_days": HORIZON_DAYS,
    "tiers": {
        "CRITICAL": {
            "selected_top_n": 1,
            "enabled": True,
            "required_precision": 0.85,
            "selection_status": "confidence_qualified_fine_scan",
            "confidence_guarantee_met": True,
            "final_raw_score_gate": 0.9243537187576294,
            "final_calibrated_probability_gate": 0.8108767050233519,
            "final_risk_index_gate": 81.08767050233519,
        },
        "HIGH": {
            "selected_top_n": 40,
            "enabled": True,
            "required_precision": 0.75,
            "selection_status": "confidence_qualified",
            "confidence_guarantee_met": True,
            "final_raw_score_gate": 0.7755436897277832,
            "final_calibrated_probability_gate": 0.716102567154952,
            "final_risk_index_gate": 71.6102567154952,
        },
        "MEDIUM": {
            "selected_top_n": 100,
            "enabled": True,
            "required_precision": 0.60,
            "selection_status": "confidence_qualified",
            "confidence_guarantee_met": True,
            "final_raw_score_gate": 0.650837779045105,
            "final_calibrated_probability_gate": 0.619007594066622,
            "final_risk_index_gate": 61.900759406662196,
        },
    },
}

# Production scoring validates that the saved model metadata agrees with the
# release-controlled values above. Set False only for controlled migration or
# diagnostic work; ordinary production runs should keep this True.
VALIDATE_PRODUCTION_CONFIG_AGAINST_MODEL_ARTIFACTS = True
PRODUCTION_CONFIGURATION_ABSOLUTE_TOLERANCE = 1e-10
PRODUCTION_FAIL_ON_MISSING_MODEL_FEATURES = True

OPERATION_CACHE_FORCE_REBUILD = False

# ---------------------------------------------------------------------------
# Development workflow controls used by ``run_development.py``.
#
# The complete workflow always retains rolling-origin validation, calibration,
# threshold selection, multi-anchor fleet validation, tier-policy construction,
# and final-model fitting. Hyperparameter tuning may be disabled when the
# approved XGBoost parameters are intentionally being reused.
# ---------------------------------------------------------------------------
RUN_OPERATION_PREPARATION_STEP = True
RUN_XGBOOST_TUNING_STEP = True
RUN_ROLLING_ORIGIN_VALIDATION_STEP = True
RUN_PROBABILITY_CALIBRATION_STEP = True
RUN_THRESHOLD_SELECTION_STEP = True
RUN_MULTI_ANCHOR_VALIDATION_STEP = True
RUN_TIER_POLICY_STEP = True
RUN_FINAL_MODEL_TRAINING_STEP = True
RUN_PRODUCTION_SETTINGS_EXPORT_STEP = True
