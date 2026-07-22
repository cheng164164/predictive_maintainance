"""Step 02: build window-based case-control datasets and train/validation/test splits."""
from __future__ import annotations

import pandas as pd

import config
from cc_utils import (
    build_case_control_base_rows,
    build_machine_master,
    build_population_random_negative_base_rows,
    build_asof_population_evaluation_base_rows,
    build_window_features,
    annotate_future_claim_outcomes,
    ensure_dir,
    load_sources,
    select_positive_claims_for_window_config,
    split_case_control_train_validation_test,
    validate_dataset_features,
    window_config_name,
    window_dataset_id,
    write_json,
)


def _load_episodes() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "01_claim_episodes" / "claim_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Claim episode file not found: {path}. Run 01_build_claim_episodes.py first."
        )
    episodes = pd.read_csv(path, parse_dates=["claim_date", "episode_end_date"])
    return episodes


def _split_counts(df: pd.DataFrame) -> dict:
    rows = {}
    if "split" not in df.columns:
        return rows
    for split_name, sub in df.groupby("split", dropna=False):
        prefix = str(split_name)
        rows[f"{prefix}_rows"] = int(len(sub))
        rows[f"{prefix}_positive_rows"] = int(sub["target"].sum()) if "target" in sub.columns else None
        rows[f"{prefix}_control_rows"] = int((sub["target"] == 0).sum()) if "target" in sub.columns else None
        rows[f"{prefix}_groups"] = int(sub["case_control_group_id"].nunique(dropna=True)) if "case_control_group_id" in sub.columns else None
    return rows


def run() -> None:
    step_dir = config.OUTPUT_DIR / "02_case_control_datasets"
    ensure_dir(step_dir)

    print("Loading sources for case-control dataset build...")
    sources = load_sources(config, include_operation=True)
    episodes = _load_episodes()
    machine_master = build_machine_master(sources)
    machine_master.to_csv(step_dir / "machine_master_source_coverage.csv", index=False)

    dataset_index_rows = []
    for window_config in config.WINDOW_CONFIGS:
        window_name = window_config_name(window_config)
        dataset_id = window_dataset_id(window_config, config)
        dataset_dir = step_dir / dataset_id
        ensure_dir(dataset_dir)
        print(f"Building dataset: {dataset_id}")

        selected_episodes, claim_selection_audit = select_positive_claims_for_window_config(
            episodes=episodes,
            window_config=window_config,
            config=config,
        )
        claim_selection_audit.to_csv(dataset_dir / "positive_claim_selection_audit.csv", index=False)
        selected_episodes.to_csv(dataset_dir / "selected_positive_claim_events.csv", index=False)

        base_rows, control_audit = build_case_control_base_rows(
            episodes=selected_episodes,
            machine_master=machine_master,
            sources=sources,
            window_config=window_config,
            config=config,
            claim_history_episodes=episodes,
        )
        base_rows.to_csv(dataset_dir / "case_control_base_rows.csv", index=False)
        control_audit.to_csv(dataset_dir / "control_sampling_audit.csv", index=False)

        if base_rows.empty:
            print(f"  [WARN] No rows generated for {dataset_id}")
            continue

        # Use all claim episodes for prior-claim history features. The selected
        # episodes only determine which claims become positive case rows.
        full_df = build_window_features(base_rows, sources=sources, episodes=episodes)
        full_df = annotate_future_claim_outcomes(full_df, claim_history_episodes=episodes, config=config)
        validate_dataset_features(full_df, config)
        full_df, split_summary = split_case_control_train_validation_test(full_df, config)

        full_path = dataset_dir / "case_control_dataset_with_split.csv"
        train_path = dataset_dir / "case_control_training_dataset.csv"
        validation_path = dataset_dir / "case_control_validation_dataset.csv"
        test_path = dataset_dir / "case_control_test_dataset.csv"

        full_df.to_csv(full_path, index=False)
        full_df[full_df["split"].eq("train")].to_csv(train_path, index=False)
        full_df[full_df["split"].eq("validation")].to_csv(validation_path, index=False)
        full_df[full_df["split"].eq("test")].to_csv(test_path, index=False)
        split_summary.to_csv(dataset_dir / "split_summary.csv", index=False)

        # ------------------------------------------------------------------
        # As-of-date population validation/test evaluation datasets.
        # ------------------------------------------------------------------
        # These files preserve the original matched case-control train/validation/test
        # files while adding realistic production-like evaluation datasets.  An
        # as-of evaluation row is a machine window ending at a historical scoring
        # date; future-claim horizon columns decide whether it is positive for
        # 30/60/90/120/etc. day evaluation.
        population_paths = {}
        population_summary_rows = []
        population_audit_rows = []

        asof_split_specs = [
            ("validation", "ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION"),
        ]
        # Test as-of evaluation is only created when explicitly enabled, normally
        # by 07_final_test_evaluation.py after parameters are locked.
        if bool(getattr(config, "ADD_ASOF_POPULATION_EVALUATION_TO_TEST", False)):
            asof_split_specs.append(("test", "ADD_ASOF_POPULATION_EVALUATION_TO_TEST"))

        for split_name, add_attr in asof_split_specs:
            if not bool(getattr(config, add_attr, False)):
                continue
            reference_split = full_df[full_df["split"].eq(split_name)].copy()
            asof_base, asof_audit = build_asof_population_evaluation_base_rows(
                reference_split_df=reference_split,
                machine_master=machine_master,
                sources=sources,
                claim_history_episodes=episodes,
                window_config=window_config,
                split_name=split_name,
                config=config,
            )
            asof_base_path = dataset_dir / f"{split_name}_asof_population_base_rows.csv"
            asof_audit_path = dataset_dir / f"{split_name}_asof_population_audit.csv"
            asof_base.to_csv(asof_base_path, index=False)
            asof_audit.to_csv(asof_audit_path, index=False)
            if not asof_audit.empty:
                population_audit_rows.extend(asof_audit.to_dict(orient="records"))

            if not asof_base.empty:
                asof_df = build_window_features(asof_base, sources=sources, episodes=episodes)
                asof_df = annotate_future_claim_outcomes(asof_df, claim_history_episodes=episodes, config=config)
                # The original `target` column is not used for horizon-sweep metrics,
                # but setting it to the maximum configured horizon makes the file
                # easier to inspect and keeps downstream summaries meaningful.
                eval_horizons = [int(h) for h in getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", [])] if isinstance(getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", []), (list, tuple, set)) else [int(getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", 0) or 0)]
                eval_horizons = [h for h in eval_horizons if h > 0]
                if eval_horizons:
                    primary_eval_col = f"eval_target_claim_within_next_{max(eval_horizons)}d"
                    if primary_eval_col in asof_df.columns:
                        asof_df["target"] = asof_df[primary_eval_col].astype(int)
                validate_dataset_features(asof_df, config)
                asof_df["split"] = split_name
            else:
                asof_df = pd.DataFrame(columns=full_df.columns)

            asof_path = dataset_dir / f"{split_name}_asof_population_evaluation_dataset.csv"
            asof_df.to_csv(asof_path, index=False)
            population_paths[f"{split_name}_asof_population_dataset_path"] = str(asof_path)
            population_paths[f"{split_name}_asof_population_base_rows_path"] = str(asof_base_path)
            population_paths[f"{split_name}_asof_population_audit_path"] = str(asof_audit_path)

            horizon_cols = [c for c in asof_df.columns if c.startswith("eval_target_claim_within_next_")]
            summary = {
                "split": split_name,
                "matched_split_rows": int(len(reference_split)),
                "asof_population_rows": int(len(asof_df)),
                "asof_population_machines": int(asof_df["machine_key"].nunique(dropna=True)) if len(asof_df) and "machine_key" in asof_df.columns else 0,
                "asof_population_as_of_dates": int(pd.to_datetime(asof_df.get("window_end"), errors="coerce").nunique()) if len(asof_df) and "window_end" in asof_df.columns else 0,
                "asof_population_path": str(asof_path),
            }
            for col in horizon_cols:
                summary[f"{col}_positive_rows"] = int(pd.to_numeric(asof_df[col], errors="coerce").fillna(0).sum()) if len(asof_df) else 0
                summary[f"{col}_positive_rate"] = float(pd.to_numeric(asof_df[col], errors="coerce").fillna(0).mean()) if len(asof_df) else None
            population_summary_rows.append(summary)

        # Optional legacy random-negative evaluation. Kept off by default.
        legacy_random_specs = []
        if bool(getattr(config, "ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION", False)):
            legacy_random_specs.append(("validation", "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE"))
        if bool(getattr(config, "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST", False)):
            legacy_random_specs.append(("test", "TEST_RANDOM_NEGATIVES_PER_POSITIVE"))
        for split_name, ratio_attr in legacy_random_specs:
            reference_split = full_df[full_df["split"].eq(split_name)].copy()
            neg_ratio = int(getattr(config, ratio_attr, 0))
            pop_base, pop_audit = build_population_random_negative_base_rows(
                reference_split_df=reference_split,
                machine_master=machine_master,
                sources=sources,
                claim_history_episodes=episodes,
                window_config=window_config,
                split_name=split_name,
                negatives_per_positive=neg_ratio,
                config=config,
            )
            pop_base_path = dataset_dir / f"{split_name}_population_random_negative_base_rows.csv"
            pop_audit_path = dataset_dir / f"{split_name}_population_random_negative_audit.csv"
            pop_base.to_csv(pop_base_path, index=False)
            pop_audit.to_csv(pop_audit_path, index=False)
            if not pop_audit.empty:
                population_audit_rows.extend(pop_audit.to_dict(orient="records"))
            if not pop_base.empty:
                pop_df = build_window_features(pop_base, sources=sources, episodes=episodes)
                pop_df = annotate_future_claim_outcomes(pop_df, claim_history_episodes=episodes, config=config)
                validate_dataset_features(pop_df, config)
                pop_df["split"] = split_name
            else:
                pop_df = pd.DataFrame(columns=full_df.columns)
            split_cases = reference_split[reference_split["target"].astype(int).eq(1)].copy()
            population_like_df = pd.concat([split_cases, pop_df], ignore_index=True, sort=False)
            with_extra_df = pd.concat([reference_split, pop_df], ignore_index=True, sort=False)
            population_like_path = dataset_dir / f"{split_name}_population_like_evaluation_dataset.csv"
            with_extra_path = dataset_dir / f"{split_name}_dataset_with_population_negatives.csv"
            pop_df_path = dataset_dir / f"{split_name}_population_random_negative_feature_rows.csv"
            population_like_df.to_csv(population_like_path, index=False)
            with_extra_df.to_csv(with_extra_path, index=False)
            pop_df.to_csv(pop_df_path, index=False)
            population_paths[f"{split_name}_population_like_dataset_path"] = str(population_like_path)
            population_paths[f"{split_name}_with_population_negatives_path"] = str(with_extra_path)
            population_paths[f"{split_name}_population_random_negative_feature_rows_path"] = str(pop_df_path)
            population_paths[f"{split_name}_population_random_negative_audit_path"] = str(pop_audit_path)

        if population_summary_rows:
            pd.DataFrame(population_summary_rows).to_csv(dataset_dir / "population_evaluation_dataset_summary.csv", index=False)
        if population_audit_rows:
            pd.DataFrame(population_audit_rows).to_csv(dataset_dir / "population_evaluation_audit_all_splits.csv", index=False)

        group_split = (
            full_df[["case_control_group_id", "split", config.SPLIT_DATE_COL]]
            .drop_duplicates()
            .sort_values([config.SPLIT_DATE_COL, "case_control_group_id"], kind="mergesort")
        )
        group_split.to_csv(dataset_dir / "case_control_group_split_assignments.csv", index=False)

        positive_rows = int((full_df["target"] == 1).sum())
        control_rows = int((full_df["target"] == 0).sum())
        train_df = full_df[full_df["split"].eq("train")]
        validation_df = full_df[full_df["split"].eq("validation")]
        test_df = full_df[full_df["split"].eq("test")]

        summary = {
            "dataset_id": dataset_id,
            "window_config": window_config,
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "controls_per_positive_case_requested": int(getattr(config, "CONTROLS_PER_POSITIVE_CASE", 3)),
            "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "control_rows": control_rows,
            "positive_rate": positive_rows / len(full_df) if len(full_df) else None,
            "unique_case_control_groups": int(full_df["case_control_group_id"].nunique(dropna=True)),
            "unique_machines": int(full_df["machine_key"].nunique(dropna=True)),
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "base_numeric_feature_count": len(getattr(config, "BASE_NUMERIC_FEATURES", [])),
            "component_features_enabled": bool(getattr(config, "ENABLE_COMPONENT_FEATURES", False)),
            "component_feature_groups": getattr(config, "COMPONENT_FEATURE_GROUPS", {}),
            "component_numeric_feature_count": len(getattr(config, "COMPONENT_NUMERIC_FEATURES", [])),
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
            "split_date_col": str(config.SPLIT_DATE_COL),
            "train_ratio": float(config.TRAIN_RATIO),
            "validation_ratio": float(config.VALIDATION_RATIO),
            "test_ratio": float(config.TEST_RATIO),
            "control_no_claim_days_after_window_end": int(config.CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END),
            "control_exclude_prior_claim_days_before_window_start": int(config.CONTROL_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START),
            "control_match_on_full_model": bool(config.CONTROL_MATCH_ON_FULL_MODEL),
            "control_require_source_coverage_overlap_window": bool(config.CONTROL_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW),
            "require_positive_source_coverage_overlap_window": bool(config.REQUIRE_POSITIVE_SOURCE_COVERAGE_OVERLAP_WINDOW),
            "max_positive_cases_per_window": config.MAX_POSITIVE_CASES_PER_WINDOW,
            "population_evaluation_paths": population_paths,
            "population_evaluation_summary": population_summary_rows,
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "evaluation_include_claim_on_window_end": bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True)),
            "positive_claim_selection_mode": getattr(config, "POSITIVE_CLAIM_SELECTION_MODE", "first"),
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "claim_events_available_before_selection": int(len(episodes)),
            "claim_events_selected_before_source_coverage_filter": int(len(selected_episodes)),
            "claim_events_excluded_by_selection_rule": int((~claim_selection_audit.get("selected_as_positive_claim", pd.Series(dtype=bool)).astype(bool)).sum()) if not claim_selection_audit.empty else 0,
            "positive_claim_selection_audit_path": str(dataset_dir / "positive_claim_selection_audit.csv"),
            "selected_positive_claim_events_path": str(dataset_dir / "selected_positive_claim_events.csv"),
        }
        summary.update(_split_counts(full_df))
        write_json(summary, dataset_dir / "dataset_summary.json")

        # Simple feature-level target-rate sanity table for fast inspection.
        availability_cols = [
            "has_any_source_window",
            "has_fault_window",
            "has_fluid_window",
            "has_maintenance_window",
            "has_operation_window",
        ]
        avail_rows = []
        for col in availability_cols:
            if col in full_df.columns:
                for value, sub in full_df.groupby(col, dropna=False):
                    avail_rows.append({
                        "feature": col,
                        "value": value,
                        "rows": int(len(sub)),
                        "positive_rate": float(sub["target"].mean()) if len(sub) else None,
                    })
        pd.DataFrame(avail_rows).to_csv(dataset_dir / "source_availability_target_rates.csv", index=False)

        dataset_index_row = {
            "dataset_id": dataset_id,
            "window_name": window_name,
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "control_rows": control_rows,
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
            "dataset_dir": str(dataset_dir),
            **population_paths,
            "positive_claim_selection_mode": getattr(config, "POSITIVE_CLAIM_SELECTION_MODE", "first"),
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "claim_events_available_before_selection": int(len(episodes)),
            "claim_events_selected_before_source_coverage_filter": int(len(selected_episodes)),
        }
        dataset_index_row.update(_split_counts(full_df))
        dataset_index_rows.append(dataset_index_row)

        print(
            f"  saved {full_path} rows={len(full_df):,} "
            f"positives={positive_rows:,} controls={control_rows:,}; "
            f"train={len(train_df):,} validation={len(validation_df):,} test={len(test_df):,}"
        )

    dataset_index = pd.DataFrame(dataset_index_rows)
    dataset_index.to_csv(step_dir / "dataset_index.csv", index=False)
    write_json(
        {
            "step": "02_build_case_control_dataset",
            "output_dir": str(step_dir),
            "dataset_count": int(len(dataset_index)),
            "window_configs": config.WINDOW_CONFIGS,
            "controls_per_positive_case": int(getattr(config, "CONTROLS_PER_POSITIVE_CASE", 3)),
            "validation_random_negatives_per_positive": int(getattr(config, "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE", 0)),
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "positive_claim_selection_mode": getattr(config, "POSITIVE_CLAIM_SELECTION_MODE", "first"),
            "positive_claim_selection_rule": (
                "first claim event per machine only"
                if str(getattr(config, "POSITIVE_CLAIM_SELECTION_MODE", "first")).lower().startswith("first")
                else "first claim event plus later events at least lead_max_days after the previous event"
            ),
            "component_features_enabled": bool(getattr(config, "ENABLE_COMPONENT_FEATURES", False)),
            "component_feature_groups": getattr(config, "COMPONENT_FEATURE_GROUPS", {}),
            "split": {
                "split_date_col": str(config.SPLIT_DATE_COL),
                "train_ratio": float(config.TRAIN_RATIO),
                "validation_ratio": float(config.VALIDATION_RATIO),
                "test_ratio": float(config.TEST_RATIO),
                "group_column": "case_control_group_id",
            },
        },
        step_dir / "run_summary.json",
    )
    print(f"02_build_case_control_dataset completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
