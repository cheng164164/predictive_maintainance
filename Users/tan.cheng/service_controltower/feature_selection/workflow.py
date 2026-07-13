"""
Shared workflow functions for the step-by-step feature-selection project.

The numbered scripts in this folder are intentionally thin wrappers around this
module. Keeping the implementation here avoids copy/paste drift while still
letting you run each feature-selection method separately:

    python 00_prepare_data.py
    python 01_unsupervised_selection.py
    python 02_correlation_analysis.py
    ...

main.py also calls the same functions in order when you want to run everything
in one command.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

import config
from fs_utils import (
    build_consensus_report,
    build_correlation_pruning_reports,
    chronological_split,
    drop_prepared_features,
    ensure_dir,
    get_candidate_features,
    grouped_correlation_pairs,
    identify_high_missing_features,
    identify_zero_variance_features,
    inner_training_split,
    load_confirmed_correlation_drop_features,
    merge_with_feature_map,
    prepare_features,
    prepared_feature_diagnostics,
    raw_feature_inventory,
    raw_missing_value_report,
    read_snapshot,
    run_anova,
    run_chi2_minmax,
    run_mutual_info,
    run_permutation_importance,
    run_shap_importance,
    save_excel_report,
    save_top_bar_plot,
    summarize_split,
    threshold_free_metrics,
    train_xgboost_classifier,
    write_json,
    write_markdown_summary,
    xgboost_importance,
)


STEP_DIRS = {
    "prepare": "00_prepare_data",
    "unsupervised": "01_unsupervised_selection",
    "correlation": "02_correlation_analysis",
    "statistical": "03_statistical_tests",
    "xgboost": "04_xgboost_importance",
    "permutation": "05_permutation_importance",
    "shap": "06_shap_analysis",
    "consensus": "07_consensus_report",
}


@dataclass
class WorkflowContext:
    """Reusable data objects shared by the numbered feature-selection steps.

    The context is rebuilt by each standalone step script. That makes every step
    runnable by itself, without requiring pickle files or hidden state from a
    previous script. The tradeoff is that preprocessing is repeated, but the
    leakage boundary stays simple and transparent.
    """

    df: pd.DataFrame
    split: object
    inner_split: object
    split_summary: pd.DataFrame
    initial_candidate_features: List[str]
    candidate_features_after_missing_filter: List[str]
    candidate_features: List[str]
    raw_inventory_before_zero_variance_filter: pd.DataFrame
    high_missing_features_dropped: pd.DataFrame
    zero_variance_features_dropped: pd.DataFrame
    preselection_filter_summary: pd.DataFrame
    prepared: object
    raw_missing: pd.DataFrame
    source_feature_groups: pd.DataFrame
    feature_group_summary: pd.DataFrame
    notes: List[str]
    effective_split_ratios: Dict[str, float]


@dataclass
class StepResult:
    """Small return object used by main.py to summarize step outputs."""

    step_name: str
    output_dir: Path
    output_files: List[Path]
    notes: List[str]


def step_output_dir(step_key: str) -> Path:
    """Return and create the configured output subfolder for one step."""

    if step_key not in STEP_DIRS:
        raise KeyError(f"Unknown step key: {step_key}")
    out_dir = Path(config.OUTPUT_DIR) / STEP_DIRS[step_key]
    ensure_dir(out_dir)
    return out_dir


def plots_dir(step_key: str) -> Path:
    """Return and create the plots subfolder for a method step."""

    out_dir = step_output_dir(step_key) / "plots"
    ensure_dir(out_dir)
    return out_dir


def _output_names(paths: Sequence[Path], base_dir: Path) -> List[str]:
    """Convert absolute output paths to names relative to the step folder."""

    names: List[str] = []
    for path in paths:
        try:
            names.append(str(path.relative_to(base_dir)))
        except ValueError:
            names.append(str(path))
    return names


def _write_step_summary(
    step_name: str,
    step_dir: Path,
    output_files: Sequence[Path],
    notes: Sequence[str],
    extra: dict | None = None,
) -> Path:
    """Write a compact JSON manifest for the current step run."""

    summary = {
        "step_name": step_name,
        "input_data_path": str(Path(config.INPUT_DATA_PATH)),
        "output_dir": str(step_dir),
        "output_files": _output_names(output_files, step_dir),
        "notes": list(notes),
    }
    if extra:
        summary.update(extra)
    path = step_dir / "step_run_summary.json"
    write_json(summary, path)
    return path


def _target_series(df: pd.DataFrame) -> pd.Series:
    """Return the binary target as integers with a clear error on missing values."""

    y = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
    if y.isna().any():
        missing_count = int(y.isna().sum())
        raise ValueError(
            f"TARGET_COL='{config.TARGET_COL}' contains {missing_count} missing or non-numeric values."
        )
    return y.astype(int)


def _correlation_dir() -> Path:
    """Return the output directory used by the correlation step."""

    return Path(config.OUTPUT_DIR) / STEP_DIRS["correlation"]


def _correlation_confirmed_drop_path() -> Path:
    """Return the editable confirmed correlation-drop file path."""

    return _correlation_dir() / getattr(
        config,
        "CORRELATION_CONFIRMED_DROP_FILE_NAME",
        "06_correlation_confirmed_drop_features_EDIT_ME.csv",
    )


def _eligible_correlated_features_path() -> Path:
    """Return the file containing features that appear in abs_corr >= review threshold pairs."""

    return _correlation_dir() / "03_correlated_pairs_ge_090_feature_train.csv"


def _apply_confirmed_correlation_pruning(
    ctx: WorkflowContext,
    step_dir: Path,
    output_files: List[Path],
    notes: List[str],
) -> None:
    """Apply user-confirmed correlation drops to prepared matrices for downstream steps."""

    if not getattr(config, "APPLY_CONFIRMED_CORRELATION_PRUNING_TO_DOWNSTREAM_STEPS", True):
        notes.append("Confirmed correlation pruning is disabled in config.py.")
        return

    confirmed_path = _correlation_confirmed_drop_path()
    confirmed = load_confirmed_correlation_drop_features(confirmed_path)

    if confirmed.empty:
        notes.append(
            "No confirmed correlation-pruning rows were found. Run 02_correlation_analysis.py first "
            f"and edit {confirmed_path.name} if you want manual correlation drops."
        )
        used_path = step_dir / "00_confirmed_correlation_drop_features_used.csv"
        confirmed.to_csv(used_path, index=False)
        output_files.append(used_path)
        return

    ignored = pd.DataFrame()
    if getattr(config, "CORRELATION_RESTRICT_DROPS_TO_REVIEWED_FEATURES", True):
        eligible_path = _eligible_correlated_features_path()
        if eligible_path.exists():
            eligible_pairs = pd.read_csv(eligible_path)
            eligible_features = set()
            for col in ["feature_1", "feature_2"]:
                if col in eligible_pairs.columns:
                    eligible_features.update(eligible_pairs[col].dropna().astype(str).tolist())
            eligible_mask = confirmed["prepared_feature"].isin(eligible_features)
            ignored = confirmed.loc[~eligible_mask].copy()
            confirmed = confirmed.loc[eligible_mask].copy()
        else:
            notes.append(
                "Confirmed correlation-pruning file exists, but the >= review-threshold pair file is missing; "
                "no correlation drops were applied because pruning is restricted to reviewed features."
            )
            confirmed = confirmed.iloc[0:0].copy()

    used_path = step_dir / "00_confirmed_correlation_drop_features_used.csv"
    confirmed.to_csv(used_path, index=False)
    output_files.append(used_path)

    if not ignored.empty:
        ignored_path = step_dir / "00_ignored_correlation_drop_features_not_in_ge_090_pairs.csv"
        ignored.to_csv(ignored_path, index=False)
        output_files.append(ignored_path)
        notes.append(
            f"Ignored {len(ignored)} confirmed correlation drop rows because they did not appear in "
            "the abs_corr >= review-threshold pair file."
        )

    ctx.prepared, audit = drop_prepared_features(
        prepared=ctx.prepared,
        drop_features=confirmed.get("prepared_feature", []),
    )
    audit_path = step_dir / "00_correlation_pruning_applied_audit.csv"
    audit.to_csv(audit_path, index=False)
    output_files.append(audit_path)

    mapping_path = step_dir / "00_prepared_feature_mapping_after_correlation_pruning.csv"
    ctx.prepared.feature_map.to_csv(mapping_path, index=False)
    output_files.append(mapping_path)

    applied_count = int(audit["drop_applied"].sum()) if not audit.empty else 0
    notes.append(
        f"Applied confirmed correlation pruning from {confirmed_path}: dropped {applied_count} prepared features."
    )


def _base_notes(split_summary: pd.DataFrame, effective_split_ratios: dict) -> List[str]:
    """Build notes shared by all step-level summaries."""

    notes = [
        "Early quality filters are applied before downstream methods: high missingness and zero variance.",
        "No final importance-based keep/drop threshold is applied in this project.",
        "The full dataset is split chronologically into training/validation/test.",
        "The training set is split again into feature_train and feature_selection_holdout.",
        "validation_holdout and test_holdout are not used for feature-selection ranking.",
        "Raw missing-value counts are reported before preprocessing/imputation.",
        "High-missingness and zero-variance filters are fitted only on feature_train to avoid holdout leakage.",
        "Correlation analysis is calculated within configured feature groups only.",
        f"Permutation importance uses config.PERMUTATION_SCORING='{config.PERMUTATION_SCORING}'.",
    ]

    zero_positive_splits = split_summary[split_summary["target_positive_count"].fillna(0) == 0]
    for _, row in zero_positive_splits.iterrows():
        notes.append(
            f"Warning: split '{row['split']}' has zero positive target rows in this run. "
            "Feature-selection/model-validation metrics for that split may not be meaningful."
        )

    notes.append(
        "Configured full split ratio weights are "
        f"{config.TRAIN_RATIO}/{config.VALIDATION_RATIO}/{config.TEST_RATIO}; "
        f"effective normalized shares are {effective_split_ratios}."
    )
    return notes


def build_workflow_context() -> WorkflowContext:
    """Load data, create splits, select candidates, and fit preprocessing.

    This function performs the common setup needed by every feature-selection
    step. The key leakage-control behavior is that preprocessing is fitted only
    on feature_train, then applied to feature_selection_holdout, validation, and
    test. The returned context is held in memory only; each step writes its own
    review files into its own output subfolder.
    """

    df = read_snapshot(Path(config.INPUT_DATA_PATH), config.DATE_COL)

    # Drop source columns that should not enter any feature-selection step. This
    # is intentionally done immediately after loading so the columns are excluded
    # from the split files, missingness report, feature inventory, preprocessing,
    # and every downstream ranking method. It also protects older snapshot CSVs
    # that were built before these columns were removed in data_preparation.
    source_columns_to_drop = [
        c for c in getattr(config, "SOURCE_COLUMNS_TO_DROP_BEFORE_FEATURE_SELECTION", [])
        if c in df.columns
    ]
    if source_columns_to_drop:
        df = df.drop(columns=source_columns_to_drop)

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{config.TARGET_COL}' was not found in input data.")

    split = chronological_split(
        df=df,
        date_col=config.DATE_COL,
        train_ratio=config.TRAIN_RATIO,
        validation_ratio=config.VALIDATION_RATIO,
        test_ratio=config.TEST_RATIO,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )
    inner_split = inner_training_split(
        training_df=split.train,
        date_col=config.DATE_COL,
        feature_train_ratio=config.FEATURE_TRAIN_RATIO_WITHIN_TRAIN,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )

    split_summary = pd.DataFrame(
        [
            summarize_split("training_main", split.train, config.TARGET_COL, config.DATE_COL, config.ID_COLS),
            summarize_split(
                "feature_train",
                inner_split.feature_train,
                config.TARGET_COL,
                config.DATE_COL,
                config.ID_COLS,
            ),
            summarize_split(
                "feature_selection_holdout",
                inner_split.feature_selection_holdout,
                config.TARGET_COL,
                config.DATE_COL,
                config.ID_COLS,
            ),
            summarize_split("validation_holdout", split.validation, config.TARGET_COL, config.DATE_COL, config.ID_COLS),
            summarize_split("test_holdout", split.test, config.TARGET_COL, config.DATE_COL, config.ID_COLS),
        ]
    )

    ratio_sum = config.TRAIN_RATIO + config.VALIDATION_RATIO + config.TEST_RATIO
    effective_split_ratios = {
        "training_main": config.TRAIN_RATIO / ratio_sum,
        "validation_holdout": config.VALIDATION_RATIO / ratio_sum,
        "test_holdout": config.TEST_RATIO / ratio_sum,
    }
    notes = _base_notes(split_summary, effective_split_ratios)
    if source_columns_to_drop:
        notes.append(
            "Dropped source columns before feature selection: "
            + ", ".join(source_columns_to_drop)
        )

    initial_candidate_features = get_candidate_features(
        df=df,
        exclude_cols=config.EXCLUDE_FEATURE_COLS,
        candidate_feature_cols=config.CANDIDATE_FEATURE_COLS,
        manual_drop_cols=config.MANUAL_DROP_FEATURE_COLS,
    )

    raw_missing = raw_missing_value_report(
        split_frames={
            "training_main": split.train,
            "feature_train": inner_split.feature_train,
            "feature_selection_holdout": inner_split.feature_selection_holdout,
            "validation_holdout": split.validation,
            "test_holdout": split.test,
        },
        features=initial_candidate_features,
    )

    high_missing_features_dropped = pd.DataFrame()
    candidate_features_after_missing_filter = list(initial_candidate_features)
    if getattr(config, "DROP_FEATURES_WITH_HIGH_MISSINGNESS", True):
        high_missing_features_dropped = identify_high_missing_features(
            raw_missing=raw_missing,
            split_name="feature_train",
            threshold=getattr(config, "HIGH_MISSINGNESS_THRESHOLD", 0.90),
        )
        high_missing_drop_set = set(high_missing_features_dropped.get("feature", []))
        candidate_features_after_missing_filter = [
            f for f in initial_candidate_features if f not in high_missing_drop_set
        ]
        if high_missing_drop_set:
            notes.append(
                f"Dropped {len(high_missing_drop_set)} raw source features before preprocessing because "
                f"feature_train missing rate was greater than "
                f"{getattr(config, 'HIGH_MISSINGNESS_THRESHOLD', 0.90):.0%}."
            )

    raw_inventory_before_zero_variance_filter = raw_feature_inventory(
        inner_split.feature_train,
        candidate_features_after_missing_filter,
        config.TARGET_COL,
    )

    zero_variance_features_dropped = pd.DataFrame()
    candidate_features = list(candidate_features_after_missing_filter)
    if getattr(config, "DROP_ZERO_VARIANCE_FEATURES", True):
        zero_variance_features_dropped = identify_zero_variance_features(
            raw_inventory_before_zero_variance_filter
        )
        zero_variance_drop_set = set(zero_variance_features_dropped.get("feature", []))
        candidate_features = [
            f for f in candidate_features_after_missing_filter if f not in zero_variance_drop_set
        ]
        if zero_variance_drop_set:
            notes.append(
                f"Dropped {len(zero_variance_drop_set)} raw source features before preprocessing because "
                "they were constant or zero-variance in feature_train."
            )

    if not candidate_features:
        raise ValueError(
            "No candidate features remain after high-missingness and zero-variance filters. "
            "Loosen the thresholds in config.py or review the input snapshot dataframe."
        )

    preselection_filter_summary = pd.DataFrame(
        [
            {
                "stage": "initial_candidate_features",
                "feature_count": len(initial_candidate_features),
                "features_removed_at_stage": 0,
            },
            {
                "stage": "after_high_missingness_filter",
                "feature_count": len(candidate_features_after_missing_filter),
                "features_removed_at_stage": len(high_missing_features_dropped),
                "threshold": getattr(config, "HIGH_MISSINGNESS_THRESHOLD", 0.90),
            },
            {
                "stage": "after_zero_variance_filter",
                "feature_count": len(candidate_features),
                "features_removed_at_stage": len(zero_variance_features_dropped),
            },
        ]
    )

    prepared = prepare_features(
        feature_train=inner_split.feature_train,
        feature_holdout=inner_split.feature_selection_holdout,
        validation=split.validation,
        test=split.test,
        features=candidate_features,
        numeric_impute_strategy=config.NUMERIC_IMPUTE_STRATEGY,
        categorical_impute_strategy=config.CATEGORICAL_IMPUTE_STRATEGY,
        one_hot_encode_categorical=config.ONE_HOT_ENCODE_CATEGORICAL,
        feature_group_rules=config.FEATURE_GROUP_RULES,
        default_feature_group=config.DEFAULT_FEATURE_GROUP,
    )

    source_feature_groups = (
        prepared.feature_map.groupby(["source_feature", "feature_group"], dropna=False)
        .agg(prepared_feature_count=("prepared_feature", "count"))
        .reset_index()
        .sort_values(["feature_group", "source_feature"])
    )
    feature_group_summary = (
        prepared.feature_map.groupby("feature_group", dropna=False)
        .agg(
            source_feature_count=("source_feature", "nunique"),
            prepared_feature_count=("prepared_feature", "count"),
        )
        .reset_index()
        .sort_values("feature_group")
    )

    other_group_count = int((source_feature_groups["feature_group"] == config.DEFAULT_FEATURE_GROUP).sum())
    if other_group_count:
        notes.append(
            f"Warning: {other_group_count} raw source features were assigned to the default "
            f"'{config.DEFAULT_FEATURE_GROUP}' feature group. Review the feature-group outputs."
        )

    return WorkflowContext(
        df=df,
        split=split,
        inner_split=inner_split,
        split_summary=split_summary,
        initial_candidate_features=list(initial_candidate_features),
        candidate_features_after_missing_filter=list(candidate_features_after_missing_filter),
        candidate_features=list(candidate_features),
        raw_inventory_before_zero_variance_filter=raw_inventory_before_zero_variance_filter,
        high_missing_features_dropped=high_missing_features_dropped,
        zero_variance_features_dropped=zero_variance_features_dropped,
        preselection_filter_summary=preselection_filter_summary,
        prepared=prepared,
        raw_missing=raw_missing,
        source_feature_groups=source_feature_groups,
        feature_group_summary=feature_group_summary,
        notes=notes,
        effective_split_ratios=effective_split_ratios,
    )


def run_prepare_data() -> StepResult:
    """Step 00: write split, candidate-feature, missingness, and mapping outputs."""

    step_name = "00_prepare_data"
    step_dir = step_output_dir("prepare")
    ctx = build_workflow_context()
    output_files: List[Path] = []

    split_summary_path = step_dir / "01_split_summary.csv"
    ctx.split_summary.to_csv(split_summary_path, index=False)
    output_files.append(split_summary_path)

    split_assignments_path = step_dir / "01_split_assignments.csv"
    ctx.split.split_assignments.to_csv(split_assignments_path, index=False)
    output_files.append(split_assignments_path)

    inner_assignments_path = step_dir / "01_inner_training_split_assignments.csv"
    ctx.inner_split.split_assignments.to_csv(inner_assignments_path, index=False)
    output_files.append(inner_assignments_path)

    initial_candidate_path = step_dir / "02_initial_candidate_features_before_filters.csv"
    pd.DataFrame({"raw_candidate_feature": ctx.initial_candidate_features}).to_csv(initial_candidate_path, index=False)
    output_files.append(initial_candidate_path)

    final_candidate_path = step_dir / "03_candidate_features_after_preselection_filters.csv"
    pd.DataFrame({"raw_candidate_feature": ctx.candidate_features}).to_csv(final_candidate_path, index=False)
    output_files.append(final_candidate_path)

    raw_missing_path = step_dir / "04_raw_missing_values_before_preprocessing.csv"
    ctx.raw_missing.to_csv(raw_missing_path, index=False)
    output_files.append(raw_missing_path)

    high_missing_path = step_dir / "05_dropped_high_missing_features_feature_train.csv"
    ctx.high_missing_features_dropped.to_csv(high_missing_path, index=False)
    output_files.append(high_missing_path)

    zero_variance_path = step_dir / "06_dropped_zero_variance_features_feature_train.csv"
    ctx.zero_variance_features_dropped.to_csv(zero_variance_path, index=False)
    output_files.append(zero_variance_path)

    filter_summary_path = step_dir / "07_preselection_filter_summary.csv"
    ctx.preselection_filter_summary.to_csv(filter_summary_path, index=False)
    output_files.append(filter_summary_path)

    feature_map_path = step_dir / "08_prepared_feature_mapping.csv"
    ctx.prepared.feature_map.to_csv(feature_map_path, index=False)
    output_files.append(feature_map_path)

    feature_groups_path = step_dir / "09_prepared_feature_groups_for_correlation.csv"
    ctx.prepared.feature_map[
        ["prepared_feature", "source_feature", "feature_group", "preprocessor_feature_name"]
    ].to_csv(feature_groups_path, index=False)
    output_files.append(feature_groups_path)

    source_feature_groups_path = step_dir / "10_source_feature_groups_for_review.csv"
    ctx.source_feature_groups.to_csv(source_feature_groups_path, index=False)
    output_files.append(source_feature_groups_path)

    feature_group_summary_path = step_dir / "11_feature_group_summary_for_review.csv"
    ctx.feature_group_summary.to_csv(feature_group_summary_path, index=False)
    output_files.append(feature_group_summary_path)

    run_summary_extra = {
        "rows": int(len(ctx.df)),
        "columns": int(len(ctx.df.columns)),
        "initial_raw_candidate_feature_count": int(len(ctx.initial_candidate_features)),
        "candidate_feature_count_after_preselection_filters": int(len(ctx.candidate_features)),
        "high_missing_feature_drop_count": int(len(ctx.high_missing_features_dropped)),
        "zero_variance_feature_drop_count": int(len(ctx.zero_variance_features_dropped)),
        "prepared_feature_count_after_encoding": int(ctx.prepared.X_feature_train.shape[1]),
        "numeric_input_feature_count": int(len(ctx.prepared.numeric_input_cols)),
        "categorical_input_feature_count": int(len(ctx.prepared.categorical_input_cols)),
        "effective_normalized_split_ratios": ctx.effective_split_ratios,
        "feature_group_counts": ctx.prepared.feature_map["feature_group"].value_counts(dropna=False).to_dict(),
    }
    summary_path = _write_step_summary(step_name, step_dir, output_files, ctx.notes, run_summary_extra)
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, ctx.notes)


def run_unsupervised_selection() -> StepResult:
    """Step 01: write raw inventory and zero-variance screening outputs.

    High-missingness screening is reported in step 00. This step focuses on the
    next unsupervised source-level filter: dropping raw columns with no observed
    variation in feature_train. The prepared diagnostics are calculated after
    both early filters have been applied, so they reflect the smaller feature set
    that step 02 and later methods will use.
    """

    step_name = "01_unsupervised_selection"
    step_dir = step_output_dir("unsupervised")
    ctx = build_workflow_context()
    output_files: List[Path] = []

    raw_inventory_before_zero_path = step_dir / "01_raw_feature_inventory_after_high_missing_filter_feature_train.csv"
    ctx.raw_inventory_before_zero_variance_filter.to_csv(raw_inventory_before_zero_path, index=False)
    output_files.append(raw_inventory_before_zero_path)

    zero_variance_path = step_dir / "02_dropped_zero_variance_features_feature_train.csv"
    ctx.zero_variance_features_dropped.to_csv(zero_variance_path, index=False)
    output_files.append(zero_variance_path)

    raw_inventory_after_filters = raw_feature_inventory(
        ctx.inner_split.feature_train,
        ctx.candidate_features,
        config.TARGET_COL,
    )
    raw_inventory_after_filters_path = step_dir / "03_raw_feature_inventory_after_preselection_filters_feature_train.csv"
    raw_inventory_after_filters.to_csv(raw_inventory_after_filters_path, index=False)
    output_files.append(raw_inventory_after_filters_path)

    prep_diag = prepared_feature_diagnostics(ctx.prepared.X_feature_train)
    prep_diag = merge_with_feature_map(prep_diag, ctx.prepared.feature_map)
    prep_diag_path = step_dir / "04_prepared_feature_diagnostics_after_preselection_filters_feature_train.csv"
    prep_diag.to_csv(prep_diag_path, index=False)
    output_files.append(prep_diag_path)

    constant_features = prep_diag[prep_diag["is_constant_exact"] == True].copy()
    constant_path = step_dir / "05_constant_prepared_features_after_preselection_filters_feature_train.csv"
    constant_features.to_csv(constant_path, index=False)
    output_files.append(constant_path)

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        ctx.notes,
        {
            "initial_raw_candidate_feature_count": int(len(ctx.initial_candidate_features)),
            "after_high_missing_filter_raw_feature_count": int(len(ctx.candidate_features_after_missing_filter)),
            "zero_variance_feature_drop_count": int(len(ctx.zero_variance_features_dropped)),
            "candidate_feature_count_after_preselection_filters": int(len(ctx.candidate_features)),
            "prepared_feature_count_after_encoding": int(ctx.prepared.X_feature_train.shape[1]),
            "exact_constant_prepared_feature_count_after_filters": int(len(constant_features)),
        },
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, ctx.notes)

def run_correlation_analysis() -> StepResult:
    """Step 02: write correlation pairs plus auto/manual pruning review files."""

    step_name = "02_correlation_analysis"
    step_dir = step_output_dir("correlation")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)

    corr_pairs = grouped_correlation_pairs(
        X=ctx.prepared.X_feature_train,
        feature_map=ctx.prepared.feature_map,
    )
    corr_pairs_path = step_dir / "01_grouped_correlation_pairs_feature_train.csv"
    corr_pairs.to_csv(corr_pairs_path, index=False)
    output_files.append(corr_pairs_path)

    corr_summary = (
        corr_pairs.groupby("feature_group", dropna=False)
        .agg(
            correlation_pair_count=("abs_correlation", "count"),
            max_abs_correlation=("abs_correlation", "max"),
            mean_abs_correlation=("abs_correlation", "mean"),
        )
        .reset_index()
        .sort_values("correlation_pair_count", ascending=False)
        if not corr_pairs.empty
        else pd.DataFrame(columns=["feature_group", "correlation_pair_count", "max_abs_correlation", "mean_abs_correlation"])
    )
    corr_summary_path = step_dir / "02_grouped_correlation_summary_feature_train.csv"
    corr_summary.to_csv(corr_summary_path, index=False)
    output_files.append(corr_summary_path)

    raw_inventory_after_filters = raw_feature_inventory(
        ctx.inner_split.feature_train,
        ctx.candidate_features,
        config.TARGET_COL,
    )
    pruning_reports = build_correlation_pruning_reports(
        corr_pairs=corr_pairs,
        X=ctx.prepared.X_feature_train,
        feature_map=ctx.prepared.feature_map,
        raw_inventory=raw_inventory_after_filters,
        review_threshold=getattr(config, "CORRELATION_REVIEW_THRESHOLD", 0.90),
        auto_drop_threshold=getattr(config, "CORRELATION_AUTO_DROP_THRESHOLD", 0.95),
    )

    correlated_pairs_path = step_dir / "03_correlated_pairs_ge_090_feature_train.csv"
    pruning_reports["correlated_pairs_ge_review_threshold"].to_csv(correlated_pairs_path, index=False)
    output_files.append(correlated_pairs_path)

    auto_prune_path = step_dir / "04_correlation_auto_prune_ge_095_feature_train.csv"
    pruning_reports["auto_prune_report"].to_csv(auto_prune_path, index=False)
    output_files.append(auto_prune_path)

    manual_review_path = step_dir / "05_correlation_manual_review_choices_090_to_lt_095_feature_train.csv"
    pruning_reports["manual_review_report"].to_csv(manual_review_path, index=False)
    output_files.append(manual_review_path)

    confirmed_template = pruning_reports["confirmed_drop_template"]

    confirmed_template_path = step_dir / getattr(
        config,
        "CORRELATION_CONFIRMED_DROP_FILE_NAME",
        "06_correlation_confirmed_drop_features_EDIT_ME.csv",
    )
    refreshed_template_path = step_dir / getattr(
        config,
        "CORRELATION_CONFIRMED_DROP_TEMPLATE_FILE_NAME",
        "06_correlation_confirmed_drop_features_TEMPLATE.csv",
    )
    overwrite_confirmed_drop_file = bool(
        getattr(config, "CORRELATION_OVERWRITE_CONFIRMED_DROP_FILE", False)
    )

    # Always write a fresh generated template when it is separate from the
    # user-editable confirmation file. This gives you the latest recommendations
    # without destroying manual edits.
    if refreshed_template_path != confirmed_template_path:
        confirmed_template.to_csv(refreshed_template_path, index=False)
        output_files.append(refreshed_template_path)

    editable_file_existed = confirmed_template_path.exists()
    if overwrite_confirmed_drop_file or not editable_file_existed:
        confirmed_template.to_csv(confirmed_template_path, index=False)
        output_files.append(confirmed_template_path)
        confirmed_file_action = "overwritten" if editable_file_existed else "created"
    else:
        output_files.append(confirmed_template_path)
        confirmed_file_action = "preserved_existing_manual_edits"

    quality_path = step_dir / "07_correlation_pruning_feature_quality_scores.csv"
    pruning_reports["feature_quality"].to_csv(quality_path, index=False)
    output_files.append(quality_path)

    auto_drop_count = int(
        (pruning_reports["auto_prune_report"].get("recommendation_action", pd.Series(dtype=str)) == "drop").sum()
    ) if not pruning_reports["auto_prune_report"].empty else 0
    manual_choice_count = int(len(pruning_reports["manual_review_report"]))
    generated_template_confirmed_count = int(
        confirmed_template.get("drop_confirmed", pd.Series(dtype=bool)).apply(bool).sum()
    ) if not confirmed_template.empty else 0
    editable_confirmed_count = None
    if confirmed_template_path.exists():
        try:
            editable_confirmed_count = int(len(load_confirmed_correlation_drop_features(confirmed_template_path)))
        except Exception as exc:
            notes.append(f"Could not count confirmed rows in {confirmed_template_path.name}: {exc}")

    notes.append(
        "Correlation pruning review uses only feature_train and only features in pairs with "
        f"abs_corr >= {getattr(config, 'CORRELATION_REVIEW_THRESHOLD', 0.90):.2f}."
    )
    notes.append(
        f"Auto-pruning candidates use abs_corr >= {getattr(config, 'CORRELATION_AUTO_DROP_THRESHOLD', 0.95):.2f}; "
        f"manual-review choices use {getattr(config, 'CORRELATION_REVIEW_THRESHOLD', 0.90):.2f} <= abs_corr "
        f"< {getattr(config, 'CORRELATION_AUTO_DROP_THRESHOLD', 0.95):.2f}."
    )
    if confirmed_file_action == "preserved_existing_manual_edits":
        notes.append(
            f"Preserved existing {confirmed_template_path.name}; rerunning step 02 did not overwrite manual edits. "
            f"A fresh comparison template was written to {refreshed_template_path.name}."
        )
    elif confirmed_file_action == "overwritten":
        notes.append(
            f"Overwrote {confirmed_template_path.name} because "
            "CORRELATION_OVERWRITE_CONFIRMED_DROP_FILE=True in config.py."
        )
    else:
        notes.append(f"Created editable confirmed-drop file: {confirmed_template_path.name}.")

    notes.append(
        f"Edit {confirmed_template_path.name}: keep auto rows as drop_confirmed=True, "
        "set manual rows to True only after review, then run 03_statistical_tests.py."
    )

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        notes,
        {
            "grouped_correlation_pair_count": int(len(corr_pairs)),
            "correlated_pair_count_ge_review_threshold": int(len(pruning_reports["correlated_pairs_ge_review_threshold"])),
            "auto_drop_feature_count_ge_auto_threshold": auto_drop_count,
            "manual_review_pair_count_review_to_lt_auto_threshold": manual_choice_count,
            "generated_template_confirmed_drop_count": generated_template_confirmed_count,
            "editable_confirmed_drop_count": editable_confirmed_count,
            "confirmed_drop_file_action": confirmed_file_action,
            "confirmed_drop_file_overwrite_enabled": overwrite_confirmed_drop_file,
            "review_threshold": getattr(config, "CORRELATION_REVIEW_THRESHOLD", 0.90),
            "auto_drop_threshold": getattr(config, "CORRELATION_AUTO_DROP_THRESHOLD", 0.95),
        },
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)

def run_statistical_tests() -> StepResult:
    """Step 03: run ANOVA, mutual information, and chi-squared filters."""

    step_name = "03_statistical_tests"
    step_dir = step_output_dir("statistical")
    ensure_dir(step_dir / "plots")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)
    _apply_confirmed_correlation_pruning(ctx, step_dir, output_files, notes)
    y_feature_train = _target_series(ctx.inner_split.feature_train)

    anova_report = run_anova(ctx.prepared.X_feature_train, y_feature_train)
    anova_report = merge_with_feature_map(anova_report, ctx.prepared.feature_map)
    anova_path = step_dir / "01_anova_f_classif_feature_train.csv"
    anova_report.to_csv(anova_path, index=False)
    output_files.append(anova_path)

    mi_report = run_mutual_info(
        ctx.prepared.X_feature_train,
        y_feature_train,
        random_state=config.MUTUAL_INFO_RANDOM_STATE,
        mode=config.MUTUAL_INFO_MODE,
        n_bins=config.MUTUAL_INFO_N_BINS,
    )
    mi_report = merge_with_feature_map(mi_report, ctx.prepared.feature_map)
    mi_path = step_dir / "02_mutual_info_classif_feature_train.csv"
    mi_report.to_csv(mi_path, index=False)
    output_files.append(mi_path)

    chi2_report = run_chi2_minmax(ctx.prepared.X_feature_train, y_feature_train)
    chi2_report = merge_with_feature_map(chi2_report, ctx.prepared.feature_map)
    chi2_path = step_dir / "03_chi2_minmax_scaled_feature_train.csv"
    chi2_report.to_csv(chi2_path, index=False)
    output_files.append(chi2_path)

    if config.GENERATE_PLOTS:
        plot_specs = [
            (anova_report, "prepared_feature", "anova_f_score", "Top ANOVA F scores", "plot_top_anova_f_score.png"),
            (mi_report, "prepared_feature", "mutual_info_score", "Top mutual information scores", "plot_top_mutual_info.png"),
            (chi2_report, "prepared_feature", "chi2_score_minmax_scaled", "Top chi2 scores, minmax scaled", "plot_top_chi2.png"),
        ]
        for df_plot, feature_col, value_col, title, file_name in plot_specs:
            plot_path = step_dir / "plots" / file_name
            save_top_bar_plot(df_plot, feature_col, value_col, title, plot_path, config.REPORT_TOP_N)
            if plot_path.exists():
                output_files.append(plot_path)

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        notes,
        {"prepared_feature_count_after_correlation_pruning": int(ctx.prepared.X_feature_train.shape[1])},
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)


def _train_xgboost_for_context(ctx: WorkflowContext):
    """Train the temporary XGBoost model used by XGB, permutation, and SHAP steps."""

    y_feature_train = _target_series(ctx.inner_split.feature_train)
    model = train_xgboost_classifier(
        ctx.prepared.X_feature_train,
        y_feature_train,
        config.XGB_PARAMS,
    )
    return model


def run_xgboost_importance() -> StepResult:
    """Step 04: train temporary XGBoost and write built-in importance reports."""

    step_name = "04_xgboost_importance"
    step_dir = step_output_dir("xgboost")
    ensure_dir(step_dir / "plots")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)
    _apply_confirmed_correlation_pruning(ctx, step_dir, output_files, notes)

    xgb_report = pd.DataFrame()
    xgb_metrics = {}
    try:
        model = _train_xgboost_for_context(ctx)
        y_feature_holdout = _target_series(ctx.inner_split.feature_selection_holdout)
        xgb_report = xgboost_importance(model, ctx.prepared.X_feature_train.columns)
        xgb_report = merge_with_feature_map(xgb_report, ctx.prepared.feature_map)
        xgb_metrics = threshold_free_metrics(model, ctx.prepared.X_feature_holdout, y_feature_holdout)
    except Exception as exc:
        xgb_report = pd.DataFrame({"warning": [f"XGBoost training or importance failed: {exc}"]})
        xgb_metrics = {"warning": str(exc), "traceback": traceback.format_exc()}
        notes.append(f"XGBoost report contains warning: {exc}")

    xgb_path = step_dir / "01_xgboost_importance_feature_train.csv"
    xgb_report.to_csv(xgb_path, index=False)
    output_files.append(xgb_path)

    xgb_metrics_path = step_dir / "02_xgboost_threshold_free_metrics_feature_selection_holdout.json"
    write_json(xgb_metrics, xgb_metrics_path)
    output_files.append(xgb_metrics_path)

    if config.GENERATE_PLOTS:
        plot_path = step_dir / "plots" / "plot_top_xgb_total_gain.png"
        save_top_bar_plot(xgb_report, "prepared_feature", "xgb_total_gain", "Top XGBoost total gain", plot_path, config.REPORT_TOP_N)
        if plot_path.exists():
            output_files.append(plot_path)

    summary_path = _write_step_summary(step_name, step_dir, output_files, notes, {"xgb_metrics": xgb_metrics})
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)


def run_permutation_step() -> StepResult:
    """Step 05: compute permutation importance on feature_selection_holdout."""

    step_name = "05_permutation_importance"
    step_dir = step_output_dir("permutation")
    ensure_dir(step_dir / "plots")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)
    _apply_confirmed_correlation_pruning(ctx, step_dir, output_files, notes)

    try:
        model = _train_xgboost_for_context(ctx)
        y_feature_holdout = _target_series(ctx.inner_split.feature_selection_holdout)
        permutation_report = run_permutation_importance(
            model=model,
            X=ctx.prepared.X_feature_holdout,
            y=y_feature_holdout,
            scoring=config.PERMUTATION_SCORING,
            n_repeats=config.PERMUTATION_N_REPEATS,
            random_state=config.PERMUTATION_RANDOM_STATE,
            n_jobs=config.PERMUTATION_N_JOBS,
        )
        permutation_report = merge_with_feature_map(permutation_report, ctx.prepared.feature_map)
    except Exception as exc:
        permutation_report = pd.DataFrame({"warning": [f"Permutation importance failed: {exc}"]})
        notes.append(f"Permutation importance warning: {exc}")

    permutation_path = step_dir / "01_permutation_importance_f2_feature_selection_holdout.csv"
    permutation_report.to_csv(permutation_path, index=False)
    output_files.append(permutation_path)

    if config.GENERATE_PLOTS:
        plot_path = step_dir / "plots" / "plot_top_permutation_importance_f2.png"
        save_top_bar_plot(
            permutation_report,
            "prepared_feature",
            "permutation_importance_mean",
            "Top permutation importance using F2",
            plot_path,
            config.REPORT_TOP_N,
        )
        if plot_path.exists():
            output_files.append(plot_path)

    summary_path = _write_step_summary(step_name, step_dir, output_files, notes)
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)


def run_shap_step() -> StepResult:
    """Step 06: compute SHAP mean absolute importance on feature_selection_holdout."""

    step_name = "06_shap_analysis"
    step_dir = step_output_dir("shap")
    ensure_dir(step_dir / "plots")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)
    _apply_confirmed_correlation_pruning(ctx, step_dir, output_files, notes)

    try:
        model = _train_xgboost_for_context(ctx)
        shap_report = run_shap_importance(
            model=model,
            X=ctx.prepared.X_feature_holdout,
            max_rows=config.SHAP_MAX_ROWS,
            random_state=config.SHAP_RANDOM_STATE,
        )
        shap_report = merge_with_feature_map(shap_report, ctx.prepared.feature_map)
    except Exception as exc:
        shap_report = pd.DataFrame({"warning": [f"SHAP analysis failed: {exc}"]})
        notes.append(f"SHAP analysis warning: {exc}")

    shap_path = step_dir / "01_shap_importance_feature_selection_holdout.csv"
    shap_report.to_csv(shap_path, index=False)
    output_files.append(shap_path)

    if config.GENERATE_PLOTS:
        plot_path = step_dir / "plots" / "plot_top_shap_importance.png"
        save_top_bar_plot(shap_report, "prepared_feature", "mean_abs_shap", "Top mean absolute SHAP", plot_path, config.REPORT_TOP_N)
        if plot_path.exists():
            output_files.append(plot_path)

    summary_path = _write_step_summary(step_name, step_dir, output_files, notes)
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)


def _read_csv_report(path: Path, report_name: str) -> pd.DataFrame:
    """Read a method report if it exists; otherwise return a warning table."""

    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({"warning": [f"Missing {report_name} report. Run the corresponding step first: {path}"]})


def run_consensus_report() -> StepResult:
    """Step 07: combine existing method reports into consensus/Excel summaries."""

    step_name = "07_consensus_report"
    step_dir = step_output_dir("consensus")
    ctx = build_workflow_context()
    output_files: List[Path] = []
    notes = list(ctx.notes)
    _apply_confirmed_correlation_pruning(ctx, step_dir, output_files, notes)

    prepare_dir = step_output_dir("prepare")
    unsup_dir = step_output_dir("unsupervised")
    corr_dir = step_output_dir("correlation")
    stat_dir = step_output_dir("statistical")
    xgb_dir = step_output_dir("xgboost")
    perm_dir = step_output_dir("permutation")
    shap_dir = step_output_dir("shap")

    raw_missing = _read_csv_report(prepare_dir / "04_raw_missing_values_before_preprocessing.csv", "raw missing")
    high_missing_features = _read_csv_report(prepare_dir / "05_dropped_high_missing_features_feature_train.csv", "high missing features")
    zero_variance_features = _read_csv_report(prepare_dir / "06_dropped_zero_variance_features_feature_train.csv", "zero variance features")
    preselection_summary = _read_csv_report(prepare_dir / "07_preselection_filter_summary.csv", "preselection filter summary")
    source_feature_groups = _read_csv_report(prepare_dir / "10_source_feature_groups_for_review.csv", "source feature groups")
    feature_group_summary = _read_csv_report(prepare_dir / "11_feature_group_summary_for_review.csv", "feature group summary")
    raw_inventory = _read_csv_report(unsup_dir / "03_raw_feature_inventory_after_preselection_filters_feature_train.csv", "raw inventory")
    prep_diag = _read_csv_report(unsup_dir / "04_prepared_feature_diagnostics_after_preselection_filters_feature_train.csv", "prepared diagnostics")
    constant_features = _read_csv_report(unsup_dir / "05_constant_prepared_features_after_preselection_filters_feature_train.csv", "constant features")
    corr_pairs = _read_csv_report(corr_dir / "01_grouped_correlation_pairs_feature_train.csv", "correlation")
    corr_pairs_ge_090 = _read_csv_report(corr_dir / "03_correlated_pairs_ge_090_feature_train.csv", "correlation pairs >= review threshold")
    corr_auto_prune = _read_csv_report(corr_dir / "04_correlation_auto_prune_ge_095_feature_train.csv", "correlation auto prune")
    corr_manual_review = _read_csv_report(corr_dir / "05_correlation_manual_review_choices_090_to_lt_095_feature_train.csv", "correlation manual review")
    corr_confirmed_template = _read_csv_report(corr_dir / getattr(config, "CORRELATION_CONFIRMED_DROP_FILE_NAME", "06_correlation_confirmed_drop_features_EDIT_ME.csv"), "confirmed correlation drop template")
    anova_report = _read_csv_report(stat_dir / "01_anova_f_classif_feature_train.csv", "ANOVA")
    mi_report = _read_csv_report(stat_dir / "02_mutual_info_classif_feature_train.csv", "mutual information")
    chi2_report = _read_csv_report(stat_dir / "03_chi2_minmax_scaled_feature_train.csv", "chi2")
    xgb_report = _read_csv_report(xgb_dir / "01_xgboost_importance_feature_train.csv", "XGBoost")
    permutation_report = _read_csv_report(perm_dir / "01_permutation_importance_f2_feature_selection_holdout.csv", "permutation")
    shap_report = _read_csv_report(shap_dir / "01_shap_importance_feature_selection_holdout.csv", "SHAP")

    reports = {
        "anova": anova_report,
        "mutual_info": mi_report,
        "chi2": chi2_report,
        "xgboost": xgb_report,
        "permutation": permutation_report,
        "shap": shap_report,
    }
    consensus = build_consensus_report(ctx.prepared.feature_map, reports)
    consensus_path = step_dir / "01_consensus_rank_review_no_threshold.csv"
    consensus.to_csv(consensus_path, index=False)
    output_files.append(consensus_path)

    if config.GENERATE_EXCEL_REPORT:
        excel_corr_pairs = corr_pairs.head(min(len(corr_pairs), 1_000_000)) if not corr_pairs.empty else corr_pairs
        excel_tables = {
            "split_summary": ctx.split_summary,
            "candidate_features_after_filters": pd.DataFrame({"raw_candidate_feature": ctx.candidate_features}),
            "preselection_summary": preselection_summary,
            "raw_missing_before_prep": raw_missing,
            "dropped_high_missing": high_missing_features,
            "dropped_zero_variance": zero_variance_features,
            "raw_inventory_after_filters": raw_inventory,
            "prepared_mapping": ctx.prepared.feature_map,
            "source_feature_groups": source_feature_groups,
            "feature_group_summary": feature_group_summary,
            "prepared_diagnostics": prep_diag,
            "constant_prepared_after_filters": constant_features,
            "grouped_correlation_pairs": excel_corr_pairs,
            "corr_pairs_ge_090": corr_pairs_ge_090,
            "corr_auto_prune_ge_095": corr_auto_prune,
            "corr_manual_review_090_095": corr_manual_review,
            "corr_confirmed_drop_template": corr_confirmed_template,
            "anova": anova_report,
            "mutual_info": mi_report,
            "chi2": chi2_report,
            "xgb_importance": xgb_report,
            "permutation_f2": permutation_report,
            "shap": shap_report,
            "consensus_review": consensus,
        }
        excel_path = step_dir / config.EXCEL_REPORT_NAME
        save_excel_report(excel_tables, excel_path)
        output_files.append(excel_path)

    markdown_path = step_dir / config.MARKDOWN_REPORT_NAME
    all_output_names = _output_names(output_files, step_dir)
    write_markdown_summary(
        path=markdown_path,
        input_path=Path(config.INPUT_DATA_PATH),
        split_summary=ctx.split_summary,
        output_files=all_output_names,
        notes=notes,
    )
    output_files.append(markdown_path)

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        notes,
        {
            "consensus_feature_count": int(len(consensus)),
            "method_report_paths_read": {
                "anova": str(stat_dir / "01_anova_f_classif_feature_train.csv"),
                "mutual_info": str(stat_dir / "02_mutual_info_classif_feature_train.csv"),
                "chi2": str(stat_dir / "03_chi2_minmax_scaled_feature_train.csv"),
                "xgboost": str(xgb_dir / "01_xgboost_importance_feature_train.csv"),
                "permutation": str(perm_dir / "01_permutation_importance_f2_feature_selection_holdout.csv"),
                "shap": str(shap_dir / "01_shap_importance_feature_selection_holdout.csv"),
            },
        },
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, notes)


def run_all_steps() -> List[StepResult]:
    """Run every numbered step in order, matching the command sequence."""

    results = [
        run_prepare_data(),
        run_unsupervised_selection(),
        run_correlation_analysis(),
        run_statistical_tests(),
        run_xgboost_importance(),
        run_permutation_step(),
        run_shap_step(),
        run_consensus_report(),
    ]

    root_summary = {
        "input_data_path": str(Path(config.INPUT_DATA_PATH)),
        "output_dir": str(Path(config.OUTPUT_DIR)),
        "steps": [
            {
                "step_name": result.step_name,
                "output_dir": str(result.output_dir),
                "output_files": _output_names(result.output_files, result.output_dir),
            }
            for result in results
        ],
    }
    root_summary_path = Path(config.OUTPUT_DIR) / "00_all_steps_run_summary.json"
    ensure_dir(Path(config.OUTPUT_DIR))
    write_json(root_summary, root_summary_path)
    print("All feature-selection steps completed.")
    print(f"Combined summary: {root_summary_path}")
    return results
