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
    chronological_split,
    ensure_dir,
    get_candidate_features,
    grouped_correlation_pairs,
    inner_training_split,
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
    candidate_features: List[str]
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


def _base_notes(split_summary: pd.DataFrame, effective_split_ratios: dict) -> List[str]:
    """Build notes shared by all step-level summaries."""

    notes = [
        "No final keep/drop feature threshold is applied in this project.",
        "The full dataset is split chronologically into training/validation/test.",
        "The training set is split again into feature_train and feature_selection_holdout.",
        "validation_holdout and test_holdout are not used for feature-selection ranking.",
        "Raw missing-value counts are reported before preprocessing/imputation.",
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

    candidate_features = get_candidate_features(
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
        features=candidate_features,
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
        candidate_features=list(candidate_features),
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

    candidate_path = step_dir / "02_candidate_features.csv"
    pd.DataFrame({"raw_candidate_feature": ctx.candidate_features}).to_csv(candidate_path, index=False)
    output_files.append(candidate_path)

    raw_missing_path = step_dir / "03_raw_missing_values_before_preprocessing.csv"
    ctx.raw_missing.to_csv(raw_missing_path, index=False)
    output_files.append(raw_missing_path)

    feature_map_path = step_dir / "04_prepared_feature_mapping.csv"
    ctx.prepared.feature_map.to_csv(feature_map_path, index=False)
    output_files.append(feature_map_path)

    feature_groups_path = step_dir / "05_prepared_feature_groups_for_correlation.csv"
    ctx.prepared.feature_map[
        ["prepared_feature", "source_feature", "feature_group", "preprocessor_feature_name"]
    ].to_csv(feature_groups_path, index=False)
    output_files.append(feature_groups_path)

    source_feature_groups_path = step_dir / "06_source_feature_groups_for_review.csv"
    ctx.source_feature_groups.to_csv(source_feature_groups_path, index=False)
    output_files.append(source_feature_groups_path)

    feature_group_summary_path = step_dir / "07_feature_group_summary_for_review.csv"
    ctx.feature_group_summary.to_csv(feature_group_summary_path, index=False)
    output_files.append(feature_group_summary_path)

    run_summary_extra = {
        "rows": int(len(ctx.df)),
        "columns": int(len(ctx.df.columns)),
        "raw_candidate_feature_count": int(len(ctx.candidate_features)),
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
    """Step 01: write raw inventory, prepared diagnostics, and constant features."""

    step_name = "01_unsupervised_selection"
    step_dir = step_output_dir("unsupervised")
    ctx = build_workflow_context()
    output_files: List[Path] = []

    raw_inventory = raw_feature_inventory(
        ctx.inner_split.feature_train,
        ctx.candidate_features,
        config.TARGET_COL,
    )
    raw_inventory_path = step_dir / "01_raw_feature_inventory_feature_train.csv"
    raw_inventory.to_csv(raw_inventory_path, index=False)
    output_files.append(raw_inventory_path)

    prep_diag = prepared_feature_diagnostics(ctx.prepared.X_feature_train)
    prep_diag = merge_with_feature_map(prep_diag, ctx.prepared.feature_map)
    prep_diag_path = step_dir / "02_prepared_feature_diagnostics_feature_train.csv"
    prep_diag.to_csv(prep_diag_path, index=False)
    output_files.append(prep_diag_path)

    constant_features = prep_diag[prep_diag["is_constant_exact"] == True].copy()
    constant_path = step_dir / "03_constant_features_exact_feature_train.csv"
    constant_features.to_csv(constant_path, index=False)
    output_files.append(constant_path)

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        ctx.notes,
        {
            "raw_candidate_feature_count": int(len(ctx.candidate_features)),
            "prepared_feature_count": int(ctx.prepared.X_feature_train.shape[1]),
            "exact_constant_prepared_feature_count": int(len(constant_features)),
        },
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, ctx.notes)


def run_correlation_analysis() -> StepResult:
    """Step 02: write grouped within-source correlation pairs for feature_train."""

    step_name = "02_correlation_analysis"
    step_dir = step_output_dir("correlation")
    ctx = build_workflow_context()
    output_files: List[Path] = []

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

    summary_path = _write_step_summary(
        step_name,
        step_dir,
        output_files,
        ctx.notes,
        {"grouped_correlation_pair_count": int(len(corr_pairs))},
    )
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, ctx.notes)


def run_statistical_tests() -> StepResult:
    """Step 03: run ANOVA, mutual information, and chi-squared filters."""

    step_name = "03_statistical_tests"
    step_dir = step_output_dir("statistical")
    ensure_dir(step_dir / "plots")
    ctx = build_workflow_context()
    y_feature_train = _target_series(ctx.inner_split.feature_train)
    output_files: List[Path] = []

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

    summary_path = _write_step_summary(step_name, step_dir, output_files, ctx.notes)
    output_files.append(summary_path)

    print(f"{step_name} completed. Outputs: {step_dir}")
    return StepResult(step_name, step_dir, output_files, ctx.notes)


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

    prepare_dir = step_output_dir("prepare")
    unsup_dir = step_output_dir("unsupervised")
    corr_dir = step_output_dir("correlation")
    stat_dir = step_output_dir("statistical")
    xgb_dir = step_output_dir("xgboost")
    perm_dir = step_output_dir("permutation")
    shap_dir = step_output_dir("shap")

    raw_missing = _read_csv_report(prepare_dir / "03_raw_missing_values_before_preprocessing.csv", "raw missing")
    source_feature_groups = _read_csv_report(prepare_dir / "06_source_feature_groups_for_review.csv", "source feature groups")
    feature_group_summary = _read_csv_report(prepare_dir / "07_feature_group_summary_for_review.csv", "feature group summary")
    raw_inventory = _read_csv_report(unsup_dir / "01_raw_feature_inventory_feature_train.csv", "raw inventory")
    prep_diag = _read_csv_report(unsup_dir / "02_prepared_feature_diagnostics_feature_train.csv", "prepared diagnostics")
    constant_features = _read_csv_report(unsup_dir / "03_constant_features_exact_feature_train.csv", "constant features")
    corr_pairs = _read_csv_report(corr_dir / "01_grouped_correlation_pairs_feature_train.csv", "correlation")
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
            "candidate_features": pd.DataFrame({"raw_candidate_feature": ctx.candidate_features}),
            "raw_missing_before_prep": raw_missing,
            "raw_inventory": raw_inventory,
            "prepared_mapping": ctx.prepared.feature_map,
            "source_feature_groups": source_feature_groups,
            "feature_group_summary": feature_group_summary,
            "prepared_diagnostics": prep_diag,
            "constant_features": constant_features,
            "grouped_correlation_pairs": excel_corr_pairs,
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
