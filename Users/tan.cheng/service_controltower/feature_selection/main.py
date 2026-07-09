"""
Run the config-driven feature-selection analysis workflow.

Run from this folder with:

    python main.py

The script intentionally accepts no command-line arguments. Edit config.py to
change input/output paths, split ratios, model settings, preprocessing settings,
or report settings.
"""
from __future__ import annotations

import traceback
from pathlib import Path

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


def main() -> None:
    """
    Orchestrate the full non-CV feature-selection reporting workflow.

    The function follows the split design agreed for the project:
    - full dataset -> training_main / validation_holdout / test_holdout
    - training_main -> feature_train / feature_selection_holdout

    Feature-ranking reports are created from feature_train, except permutation
    and SHAP, which use feature_selection_holdout. The script does not choose a
    final feature subset; it produces review files only.
    """

    # ------------------------------------------------------------------
    # Set up output locations and run-level notes.
    # ------------------------------------------------------------------
    output_dir = Path(config.OUTPUT_DIR)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "plots")

    output_files = []
    notes = [
        "No final keep/drop feature threshold is applied in this run.",
        "The full dataset is split chronologically into training/validation/test.",
        "The training set is split again into feature_train and feature_selection_holdout for permutation and SHAP reporting.",
        "The validation_holdout and test_holdout splits are not used for feature-selection ranking.",
        "Raw missing-value counts are saved before imputation/preprocessing.",
        "Correlation analysis is calculated within configured feature groups only.",
        "Permutation importance uses F2 scoring from config.PERMUTATION_SCORING.",
    ]

    # ------------------------------------------------------------------
    # Load input data and create chronological splits.
    # ------------------------------------------------------------------
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

    # Summarize every split so the user can quickly inspect row counts, date
    # windows, target prevalence, and unique machine counts.
    split_summary = pd.DataFrame(
        [
            summarize_split(
                "training_main",
                split.train,
                config.TARGET_COL,
                config.DATE_COL,
                config.ID_COLS,
            ),
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
            summarize_split(
                "validation_holdout",
                split.validation,
                config.TARGET_COL,
                config.DATE_COL,
                config.ID_COLS,
            ),
            summarize_split(
                "test_holdout",
                split.test,
                config.TARGET_COL,
                config.DATE_COL,
                config.ID_COLS,
            ),
        ]
    )

    zero_positive_splits = split_summary[split_summary["target_positive_count"].fillna(0) == 0]
    for _, row in zero_positive_splits.iterrows():
        notes.append(
            f"Warning: split '{row['split']}' has zero positive target rows in this run. "
            "Feature-selection/model-validation metrics for that split may not be meaningful."
        )

    ratio_sum = config.TRAIN_RATIO + config.VALIDATION_RATIO + config.TEST_RATIO
    effective_split_ratios = {
        "training_main": config.TRAIN_RATIO / ratio_sum,
        "validation_holdout": config.VALIDATION_RATIO / ratio_sum,
        "test_holdout": config.TEST_RATIO / ratio_sum,
    }
    notes.append(
        "Configured full split ratio weights are "
        f"{config.TRAIN_RATIO}/{config.VALIDATION_RATIO}/{config.TEST_RATIO}; "
        f"effective normalized shares are {effective_split_ratios}."
    )

    split_summary_path = output_dir / "01_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    output_files.append(split_summary_path.name)

    split_assignments_path = output_dir / "01_split_assignments.csv"
    split.split_assignments.to_csv(split_assignments_path, index=False)
    output_files.append(split_assignments_path.name)

    inner_assignments_path = output_dir / "01_inner_training_split_assignments.csv"
    inner_split.split_assignments.to_csv(inner_assignments_path, index=False)
    output_files.append(inner_assignments_path.name)

    # ------------------------------------------------------------------
    # Select raw candidate features and save pre-preprocessing diagnostics.
    # ------------------------------------------------------------------
    candidate_features = get_candidate_features(
        df=df,
        exclude_cols=config.EXCLUDE_FEATURE_COLS,
        candidate_feature_cols=config.CANDIDATE_FEATURE_COLS,
        manual_drop_cols=config.MANUAL_DROP_FEATURE_COLS,
    )

    candidate_path = output_dir / "02_candidate_features.csv"
    pd.DataFrame({"raw_candidate_feature": candidate_features}).to_csv(candidate_path, index=False)
    output_files.append(candidate_path.name)

    # Missingness must be reported before SimpleImputer runs; otherwise the
    # original data-quality signal is hidden by preprocessing.
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
    raw_missing_path = output_dir / "03_raw_missing_values_before_preprocessing.csv"
    raw_missing.to_csv(raw_missing_path, index=False)
    output_files.append(raw_missing_path.name)

    raw_inventory = raw_feature_inventory(
        inner_split.feature_train,
        candidate_features,
        config.TARGET_COL,
    )
    raw_inventory_path = output_dir / "04_raw_feature_inventory_feature_train.csv"
    raw_inventory.to_csv(raw_inventory_path, index=False)
    output_files.append(raw_inventory_path.name)

    # ------------------------------------------------------------------
    # Fit preprocessing on feature_train and transform all splits.
    # ------------------------------------------------------------------
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

    feature_map_path = output_dir / "05_prepared_feature_mapping.csv"
    prepared.feature_map.to_csv(feature_map_path, index=False)
    output_files.append(feature_map_path.name)

    # This smaller mapping is dedicated to reviewing which group each prepared
    # feature belongs to before grouped correlation is calculated.
    feature_groups_path = output_dir / "05_prepared_feature_groups_for_correlation.csv"
    prepared.feature_map[
        ["prepared_feature", "source_feature", "feature_group", "preprocessor_feature_name"]
    ].to_csv(feature_groups_path, index=False)
    output_files.append(feature_groups_path.name)

    prep_diag = prepared_feature_diagnostics(prepared.X_feature_train)
    prep_diag = merge_with_feature_map(prep_diag, prepared.feature_map)
    prep_diag_path = output_dir / "06_prepared_feature_diagnostics_feature_train.csv"
    prep_diag.to_csv(prep_diag_path, index=False)
    output_files.append(prep_diag_path.name)

    constant_features = prep_diag[prep_diag["is_constant_exact"] == True].copy()
    constant_path = output_dir / "06_constant_features_exact_feature_train.csv"
    constant_features.to_csv(constant_path, index=False)
    output_files.append(constant_path.name)

    # ------------------------------------------------------------------
    # Grouped correlation report on feature_train only.
    # ------------------------------------------------------------------
    corr_pairs = grouped_correlation_pairs(
        X=prepared.X_feature_train,
        feature_map=prepared.feature_map,
    )
    corr_pairs_path = output_dir / "07_grouped_correlation_pairs_feature_train.csv"
    corr_pairs.to_csv(corr_pairs_path, index=False)
    output_files.append(corr_pairs_path.name)

    # ------------------------------------------------------------------
    # Supervised statistical filters fitted on feature_train only.
    # ------------------------------------------------------------------
    y_feature_train = inner_split.feature_train[config.TARGET_COL].astype(int)
    y_feature_holdout = inner_split.feature_selection_holdout[config.TARGET_COL].astype(int)

    anova_report = run_anova(prepared.X_feature_train, y_feature_train)
    anova_report = merge_with_feature_map(anova_report, prepared.feature_map)
    anova_path = output_dir / "08_anova_f_classif_feature_train.csv"
    anova_report.to_csv(anova_path, index=False)
    output_files.append(anova_path.name)

    mi_report = run_mutual_info(
        prepared.X_feature_train,
        y_feature_train,
        random_state=config.MUTUAL_INFO_RANDOM_STATE,
        mode=config.MUTUAL_INFO_MODE,
        n_bins=config.MUTUAL_INFO_N_BINS,
    )
    mi_report = merge_with_feature_map(mi_report, prepared.feature_map)
    mi_path = output_dir / "09_mutual_info_classif_feature_train.csv"
    mi_report.to_csv(mi_path, index=False)
    output_files.append(mi_path.name)

    chi2_report = run_chi2_minmax(prepared.X_feature_train, y_feature_train)
    chi2_report = merge_with_feature_map(chi2_report, prepared.feature_map)
    chi2_path = output_dir / "10_chi2_minmax_scaled_feature_train.csv"
    chi2_report.to_csv(chi2_path, index=False)
    output_files.append(chi2_path.name)

    # ------------------------------------------------------------------
    # XGBoost feature importance fitted on feature_train only.
    # ------------------------------------------------------------------
    model = None
    xgb_report = pd.DataFrame()
    xgb_metrics = {}
    try:
        model = train_xgboost_classifier(
            prepared.X_feature_train,
            y_feature_train,
            config.XGB_PARAMS,
        )
        xgb_report = xgboost_importance(model, prepared.X_feature_train.columns)
        xgb_report = merge_with_feature_map(xgb_report, prepared.feature_map)
        xgb_metrics = threshold_free_metrics(model, prepared.X_feature_holdout, y_feature_holdout)
    except Exception as exc:
        xgb_report = pd.DataFrame({"warning": [f"XGBoost training or importance failed: {exc}"]})
        xgb_metrics = {"warning": str(exc), "traceback": traceback.format_exc()}
        notes.append(f"XGBoost report contains warning: {exc}")

    xgb_path = output_dir / "11_xgboost_importance_feature_train.csv"
    xgb_report.to_csv(xgb_path, index=False)
    output_files.append(xgb_path.name)

    xgb_metrics_path = output_dir / "11_xgboost_threshold_free_metrics_feature_selection_holdout.json"
    write_json(xgb_metrics, xgb_metrics_path)
    output_files.append(xgb_metrics_path.name)

    # ------------------------------------------------------------------
    # Permutation and SHAP review on feature_selection_holdout only.
    # ------------------------------------------------------------------
    if model is not None:
        permutation_report = run_permutation_importance(
            model=model,
            X=prepared.X_feature_holdout,
            y=y_feature_holdout,
            scoring=config.PERMUTATION_SCORING,
            n_repeats=config.PERMUTATION_N_REPEATS,
            random_state=config.PERMUTATION_RANDOM_STATE,
            n_jobs=config.PERMUTATION_N_JOBS,
        )
        permutation_report = merge_with_feature_map(permutation_report, prepared.feature_map)

        shap_report = run_shap_importance(
            model=model,
            X=prepared.X_feature_holdout,
            max_rows=config.SHAP_MAX_ROWS,
            random_state=config.SHAP_RANDOM_STATE,
        )
        shap_report = merge_with_feature_map(shap_report, prepared.feature_map)
    else:
        permutation_report = pd.DataFrame({"warning": ["Skipped because XGBoost model was not available."]})
        shap_report = pd.DataFrame({"warning": ["Skipped because XGBoost model was not available."]})

    permutation_path = output_dir / "12_permutation_importance_f2_feature_selection_holdout.csv"
    permutation_report.to_csv(permutation_path, index=False)
    output_files.append(permutation_path.name)

    shap_path = output_dir / "13_shap_importance_feature_selection_holdout.csv"
    shap_report.to_csv(shap_path, index=False)
    output_files.append(shap_path.name)

    # ------------------------------------------------------------------
    # Consensus review table, not a final feature decision.
    # ------------------------------------------------------------------
    consensus = build_consensus_report(
        feature_map=prepared.feature_map,
        reports={
            "anova": anova_report,
            "mutual_info": mi_report,
            "chi2": chi2_report,
            "xgboost": xgb_report,
            "permutation": permutation_report,
            "shap": shap_report,
        },
    )
    consensus_path = output_dir / "14_consensus_rank_review_no_threshold.csv"
    consensus.to_csv(consensus_path, index=False)
    output_files.append(consensus_path.name)

    # ------------------------------------------------------------------
    # Plots for convenient quick review.
    # ------------------------------------------------------------------
    if config.GENERATE_PLOTS:
        plot_specs = [
            (anova_report, "prepared_feature", "anova_f_score", "Top ANOVA F scores", "plot_top_anova_f_score.png"),
            (mi_report, "prepared_feature", "mutual_info_score", "Top mutual information scores", "plot_top_mutual_info.png"),
            (chi2_report, "prepared_feature", "chi2_score_minmax_scaled", "Top chi2 scores, minmax scaled", "plot_top_chi2.png"),
            (xgb_report, "prepared_feature", "xgb_total_gain", "Top XGBoost total gain", "plot_top_xgb_total_gain.png"),
            (permutation_report, "prepared_feature", "permutation_importance_mean", "Top permutation importance using F2", "plot_top_permutation_importance_f2.png"),
            (shap_report, "prepared_feature", "mean_abs_shap", "Top mean absolute SHAP", "plot_top_shap_importance.png"),
        ]
        for df_plot, feature_col, value_col, title, file_name in plot_specs:
            plot_path = output_dir / "plots" / file_name
            save_top_bar_plot(
                df=df_plot,
                feature_col=feature_col,
                value_col=value_col,
                title=title,
                path=plot_path,
                top_n=config.REPORT_TOP_N,
            )
            if plot_path.exists():
                output_files.append(str(Path("plots") / file_name))

    # ------------------------------------------------------------------
    # Combined Excel report for convenient workbook-style review.
    # ------------------------------------------------------------------
    if config.GENERATE_EXCEL_REPORT:
        # Full correlation pairs are saved to CSV. For Excel, keep the table under
        # Excel's row limit if a production run creates a very large grouped-pair
        # report.
        excel_corr_pairs = corr_pairs.head(min(len(corr_pairs), 1_000_000))
        excel_tables = {
            "split_summary": split_summary,
            "candidate_features": pd.DataFrame({"raw_candidate_feature": candidate_features}),
            "raw_missing_before_prep": raw_missing,
            "raw_inventory": raw_inventory,
            "prepared_mapping": prepared.feature_map,
            "feature_groups_corr": prepared.feature_map[
                ["prepared_feature", "source_feature", "feature_group", "preprocessor_feature_name"]
            ],
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
        excel_path = output_dir / config.EXCEL_REPORT_NAME
        save_excel_report(excel_tables, excel_path)
        output_files.append(excel_path.name)

    # ------------------------------------------------------------------
    # Run summary files.
    # ------------------------------------------------------------------
    run_summary = {
        "input_data_path": str(Path(config.INPUT_DATA_PATH)),
        "output_dir": str(output_dir),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_col": config.TARGET_COL,
        "date_col": config.DATE_COL,
        "configured_split_ratio_weights": {
            "training_main": config.TRAIN_RATIO,
            "validation_holdout": config.VALIDATION_RATIO,
            "test_holdout": config.TEST_RATIO,
        },
        "effective_normalized_split_ratios": effective_split_ratios,
        "raw_candidate_feature_count": int(len(candidate_features)),
        "prepared_feature_count_after_encoding": int(prepared.X_feature_train.shape[1]),
        "numeric_input_feature_count": int(len(prepared.numeric_input_cols)),
        "categorical_input_feature_count": int(len(prepared.categorical_input_cols)),
        "feature_group_counts": prepared.feature_map["feature_group"].value_counts(dropna=False).to_dict(),
        "grouped_correlation_pair_count": int(len(corr_pairs)),
        "permutation_scoring": config.PERMUTATION_SCORING,
        "xgb_feature_selection_holdout_metrics": xgb_metrics,
        "notes": notes,
    }
    run_summary_path = output_dir / "00_run_summary.json"
    write_json(run_summary, run_summary_path)
    output_files.append(run_summary_path.name)

    markdown_path = output_dir / config.MARKDOWN_REPORT_NAME
    write_markdown_summary(
        path=markdown_path,
        input_path=Path(config.INPUT_DATA_PATH),
        split_summary=split_summary,
        output_files=output_files,
        notes=notes,
    )
    output_files.append(markdown_path.name)

    print("Feature selection analysis completed.")
    print(f"Output directory: {output_dir}")
    print(f"Main Excel report: {output_dir / config.EXCEL_REPORT_NAME}")
    print(f"Summary report: {markdown_path}")


if __name__ == "__main__":
    main()
