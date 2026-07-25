"""Step 02: build fixed machine-level train/validation/test datasets.

Training negatives follow the configurable design switch. Matched validation
and test base rows are created once in a central fixed-split asset folder and
reused by every experiment. Claim-within-horizon evaluation uses a second fixed,
outcome-independent machine/window cohort whose labels are recomputed by horizon.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd

import config
from cc_utils import (
    annotate_future_claim_outcomes,
    build_case_control_base_rows,
    build_fixed_horizon_evaluation_base_rows,
    build_machine_master,
    build_or_load_fixed_machine_split_assignments,
    build_window_features,
    configured_evaluation_horizons,
    ensure_dir,
    future_claim_target_col,
    load_sources,
    negative_sampling_mode,
    negatives_per_positive,
    normalize_negative_sampling_mode,
    read_json,
    select_positive_claims_for_window_config,
    validate_dataset_features,
    validate_fixed_dataset_splits,
    validate_negative_count,
    window_config_name,
    window_dataset_id,
    write_json,
)


_BASE_ROW_DATE_COLUMNS = [
    "window_start",
    "window_end",
    "linked_case_window_start",
    "linked_case_window_end",
    "future_claim_date",
    "control_no_claim_start",
    "control_no_claim_end",
]

_MIN_FIXED_HORIZON_FOLLOWUP_DAYS = 365


def _load_episodes() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "01_claim_episodes" / "claim_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Claim episode file not found: {path}. Run 01_build_claim_episodes.py first."
        )
    return pd.read_csv(path, parse_dates=["claim_date", "episode_end_date"])


def _split_counts(df: pd.DataFrame) -> dict:
    out: dict[str, int | float | None] = {}
    for split_name in ["train", "validation", "test"]:
        sub = df[df["split"].eq(split_name)] if "split" in df.columns else df.iloc[0:0]
        target = pd.to_numeric(sub.get("target", pd.Series(dtype=float)), errors="coerce")
        out[f"{split_name}_rows"] = int(len(sub))
        out[f"{split_name}_positive_rows"] = int(target.fillna(0).sum()) if len(sub) else 0
        out[f"{split_name}_negative_rows"] = int(target.eq(0).sum()) if len(sub) else 0
        out[f"{split_name}_positive_rate"] = float(target.mean()) if len(sub) else None
        out[f"{split_name}_machines"] = (
            int(sub["machine_key"].nunique(dropna=True)) if "machine_key" in sub.columns else 0
        )
        out[f"{split_name}_groups"] = (
            int(sub["case_control_group_id"].nunique(dropna=True))
            if "case_control_group_id" in sub.columns
            else 0
        )
    return out


def _frame_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    """Return a stable, order-independent SHA-256 fingerprint for selected columns."""
    work = df.reindex(columns=columns).copy()
    for col in columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(work[col]):
            values = pd.to_datetime(work[col], errors="coerce")
            work[col] = values.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("<NA>")
        else:
            work[col] = work[col].astype("string").fillna("<NA>")
    if columns:
        work = work.sort_values(columns, kind="mergesort").reset_index(drop=True)
    payload = work.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_locked_base_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    for col in _BASE_ROW_DATE_COLUMNS:
        if col in rows.columns:
            rows[col] = pd.to_datetime(rows[col], errors="coerce")
    if "target" in rows.columns:
        rows["target"] = pd.to_numeric(rows["target"], errors="raise").astype(int)
    return rows


def _holdout_expected_metadata(
    window_config: Mapping,
    selected_episodes: pd.DataFrame,
    all_episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    split_assignments: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
) -> dict:
    selected_with_split = selected_episodes.merge(
        split_assignments[["machine_key", "split"]],
        on="machine_key",
        how="left",
        validate="many_to_one",
    )
    selected_holdout = selected_with_split[
        selected_with_split["split"].isin(["validation", "test"])
    ].copy()
    holdout_mode = normalize_negative_sampling_mode(
        config.FIXED_HOLDOUT_NEGATIVE_SAMPLING_MODE,
        "FIXED_HOLDOUT_NEGATIVE_SAMPLING_MODE",
    )
    holdout_count = validate_negative_count(
        config.FIXED_HOLDOUT_NEGATIVES_PER_POSITIVE_CASE,
        "FIXED_HOLDOUT_NEGATIVES_PER_POSITIVE_CASE",
    )
    return {
        "lock_version": 2,
        "window_name": window_config_name(window_config),
        "window_config": {
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
        },
        "fixed_split_random_state": int(config.FIXED_SPLIT_RANDOM_STATE),
        "fixed_holdout_negative_sampling_mode": holdout_mode,
        "fixed_holdout_negatives_per_positive_case": holdout_count,
        "negative_no_claim_days_after_window_end": int(
            config.NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END
        ),
        "negative_exclude_prior_claim_days_before_window_start": int(
            config.NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START
        ),
        "require_source_coverage_overlap_window": bool(
            config.REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW
        ),
        "machine_assignment_fingerprint": _frame_fingerprint(
            split_assignments,
            ["machine_key", "split", "full_model", "has_eligible_claim_history"],
        ),
        "selected_holdout_claim_fingerprint": _frame_fingerprint(
            selected_holdout,
            ["claim_episode_id", "machine_key", "claim_date", "split"],
        ),
        "all_claim_history_fingerprint": _frame_fingerprint(
            all_episodes,
            ["claim_episode_id", "machine_key", "claim_date"],
        ),
        "machine_coverage_fingerprint": _frame_fingerprint(
            machine_master,
            ["machine_key", "full_model", "first_source_date", "last_source_date"],
        ),
    }


def _validate_locked_holdout_rows(
    rows: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> None:
    required = {
        "split",
        "machine_key",
        "case_machine_key",
        "case_control_group_id",
        "target",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Locked validation/test base rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("Locked validation/test base rows are empty.")
    invalid = sorted(set(rows["split"].dropna().astype(str)) - {"validation", "test"})
    if invalid:
        raise ValueError(f"Locked holdout rows contain unexpected split labels: {invalid}")

    assignment_map = split_assignments[["machine_key", "split"]].copy()
    assignment_map["machine_key"] = assignment_map["machine_key"].astype(str)
    check = rows.copy()
    check["machine_key"] = check["machine_key"].astype(str)
    check = check.merge(
        assignment_map.rename(columns={"split": "assigned_split"}),
        on="machine_key",
        how="left",
        validate="many_to_one",
    )
    bad = check[check["assigned_split"].isna() | check["split"].ne(check["assigned_split"])]
    if not bad.empty:
        examples = bad[["machine_key", "split", "assigned_split"]].head(10).to_dict("records")
        raise ValueError(
            "Locked holdout rows do not match the fixed machine assignments. "
            f"Examples: {examples}"
        )

    group_splits = rows.groupby("case_control_group_id", dropna=False)["split"].nunique()
    if (group_splits > 1).any():
        raise ValueError("A locked case-control group spans validation and test.")
    machine_splits = rows.groupby("machine_key", dropna=False)["split"].nunique()
    if (machine_splits > 1).any():
        raise ValueError("A machine appears in both locked validation and test rows.")

    for split_name, ratio in [
        ("validation", float(config.VALIDATION_RATIO)),
        ("test", float(config.TEST_RATIO)),
    ]:
        if ratio <= 0:
            continue
        sub = rows[rows["split"].eq(split_name)]
        if sub.empty:
            raise ValueError(f"Locked holdout split '{split_name}' is empty.")
        classes = set(pd.to_numeric(sub["target"], errors="coerce").dropna().astype(int))
        if classes != {0, 1}:
            raise ValueError(
                f"Locked holdout split '{split_name}' must contain both positive and negative rows."
            )


def _build_or_load_fixed_holdout_rows(
    window_config: Mapping,
    selected_episodes: pd.DataFrame,
    all_episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    split_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, Path]:
    window_name = window_config_name(window_config)
    lock_dir = Path(config.FIXED_SPLIT_ASSET_DIR) / window_name
    ensure_dir(lock_dir)
    rows_path = lock_dir / "fixed_validation_test_base_rows.csv"
    audit_path = lock_dir / "fixed_validation_test_negative_sampling_audit.csv"
    metadata_path = lock_dir / "fixed_validation_test_metadata.json"
    expected = _holdout_expected_metadata(
        window_config,
        selected_episodes,
        all_episodes,
        machine_master,
        split_assignments,
        sources,
    )

    existing_parts = [rows_path.exists(), metadata_path.exists()]
    if any(existing_parts) and not all(existing_parts):
        raise ValueError(
            "The fixed validation/test lock is incomplete. Delete the entire lock folder "
            f"only if you intentionally want to rebuild the holdout: {lock_dir}"
        )

    if rows_path.exists():
        metadata = read_json(metadata_path)
        if metadata.get("lock_version") != expected["lock_version"]:
            # Version 2 removes the prior 4,000-hour eligibility rule. Holdout
            # rows created by the older design cannot be reused, but the fixed
            # machine split itself remains valid and is preserved.
            for stale_path in [rows_path, audit_path, metadata_path]:
                stale_path.unlink(missing_ok=True)
            return _build_or_load_fixed_holdout_rows(
                window_config=window_config,
                selected_episodes=selected_episodes,
                all_episodes=all_episodes,
                machine_master=machine_master,
                sources=sources,
                split_assignments=split_assignments,
            )
        mismatches = {
            key: {"stored": metadata.get(key), "current": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "The eligible data or fixed-holdout settings changed after validation/test "
                "rows were locked. Keep the existing data/settings for comparable experiments, "
                f"or intentionally delete {lock_dir} to create a new holdout. "
                f"Changed fields: {sorted(mismatches)}"
            )
        rows = _read_locked_base_rows(rows_path)
        actual_fingerprint = _frame_fingerprint(
            rows,
            [
                "case_control_group_id",
                "row_role",
                "machine_key",
                "split",
                "window_start",
                "window_end",
                "target",
                "negative_sampling_type",
            ],
        )
        if metadata.get("holdout_row_fingerprint") != actual_fingerprint:
            raise ValueError(
                f"The locked validation/test row file was modified: {rows_path}"
            )
        audit = pd.read_csv(audit_path, low_memory=False) if audit_path.exists() else pd.DataFrame()
        _validate_locked_holdout_rows(rows, split_assignments)
        return rows, audit, True, rows_path

    holdout_mode = expected["fixed_holdout_negative_sampling_mode"]
    holdout_count = expected["fixed_holdout_negatives_per_positive_case"]
    rows, audit = build_case_control_base_rows(
        episodes=selected_episodes,
        machine_master=machine_master,
        sources=sources,
        window_config=window_config,
        config=config,
        claim_history_episodes=all_episodes,
        split_assignments=split_assignments,
        included_splits=["validation", "test"],
        sampling_mode_override=holdout_mode,
        negatives_per_positive_override=holdout_count,
        random_state_override=int(config.FIXED_SPLIT_RANDOM_STATE),
        apply_positive_case_cap=False,
    )
    _validate_locked_holdout_rows(rows, split_assignments)
    rows.to_csv(rows_path, index=False)
    audit.to_csv(audit_path, index=False)
    metadata = {
        **expected,
        "holdout_rows": int(len(rows)),
        "validation_rows": int(rows["split"].eq("validation").sum()),
        "test_rows": int(rows["split"].eq("test").sum()),
        "holdout_row_fingerprint": _frame_fingerprint(
            rows,
            [
                "case_control_group_id",
                "row_role",
                "machine_key",
                "split",
                "window_start",
                "window_end",
                "target",
                "negative_sampling_type",
            ],
        ),
    }
    write_json(metadata, metadata_path)
    return rows, audit, False, rows_path


def _fixed_horizon_expected_metadata(
    window_config: Mapping,
    machine_master: pd.DataFrame,
    split_assignments: pd.DataFrame,
    minimum_followup_days: int,
) -> dict:
    holdout_assignments = split_assignments[
        split_assignments["split"].isin(["validation", "test"])
    ].copy()
    holdout_machine_keys = set(holdout_assignments["machine_key"].astype(str))
    holdout_coverage = machine_master[
        machine_master["machine_key"].astype(str).isin(holdout_machine_keys)
    ].copy()
    return {
        "lock_version": 1,
        "strategy": "one_deterministic_outcome_independent_window_per_holdout_machine",
        "window_name": window_config_name(window_config),
        "window_config": {
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
        },
        "fixed_split_random_state": int(config.FIXED_SPLIT_RANDOM_STATE),
        "minimum_followup_days": int(minimum_followup_days),
        "machine_assignment_fingerprint": _frame_fingerprint(
            holdout_assignments,
            ["machine_key", "split", "full_model", "has_eligible_claim_history"],
        ),
        "machine_coverage_fingerprint": _frame_fingerprint(
            holdout_coverage,
            ["machine_key", "full_model", "first_source_date", "last_source_date"],
        ),
    }


def _validate_fixed_horizon_rows(
    rows: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> None:
    required = {
        "evaluation_sample_id",
        "split",
        "machine_key",
        "window_start",
        "window_end",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(
            f"Fixed horizon-evaluation rows are missing required columns: {missing}"
        )
    if rows.empty:
        raise ValueError("Fixed horizon-evaluation rows are empty.")
    if rows["evaluation_sample_id"].duplicated().any():
        raise ValueError("Fixed horizon-evaluation sample IDs are not unique.")
    if rows["machine_key"].duplicated().any():
        raise ValueError(
            "Fixed horizon evaluation must contain at most one row per holdout machine."
        )
    for split_name, configured_ratio in [
        ("validation", float(config.VALIDATION_RATIO)),
        ("test", float(config.TEST_RATIO)),
    ]:
        if configured_ratio > 0 and not rows["split"].astype(str).eq(split_name).any():
            raise ValueError(
                f"No eligible fixed horizon-evaluation windows were found for {split_name}. "
                "Review fixed_validation_test_horizon_sampling_audit.csv; machines need "
                "enough source history and complete future claim follow-up."
            )

    assignment_map = split_assignments[["machine_key", "split"]].copy()
    assignment_map["machine_key"] = assignment_map["machine_key"].astype(str)
    check = rows.copy()
    check["machine_key"] = check["machine_key"].astype(str)
    check = check.merge(
        assignment_map.rename(columns={"split": "assigned_split"}),
        on="machine_key",
        how="left",
        validate="one_to_one",
    )
    mismatch = check[check["split"].astype(str) != check["assigned_split"].astype(str)]
    if not mismatch.empty:
        raise ValueError(
            "Fixed horizon-evaluation rows do not match machine split assignments."
        )


def _build_or_load_fixed_horizon_rows(
    window_config: Mapping,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    split_assignments: pd.DataFrame,
    minimum_followup_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, Path]:
    window_name = window_config_name(window_config)
    lock_dir = Path(config.FIXED_SPLIT_ASSET_DIR) / window_name
    ensure_dir(lock_dir)
    rows_path = lock_dir / "fixed_validation_test_horizon_base_rows.csv"
    audit_path = lock_dir / "fixed_validation_test_horizon_sampling_audit.csv"
    metadata_path = lock_dir / "fixed_validation_test_horizon_metadata.json"
    expected = _fixed_horizon_expected_metadata(
        window_config,
        machine_master,
        split_assignments,
        minimum_followup_days,
    )

    existing_parts = [rows_path.exists(), metadata_path.exists()]
    if any(existing_parts) and not all(existing_parts):
        raise ValueError(
            "The fixed horizon-evaluation lock is incomplete. Delete its three "
            f"files only if you intentionally want to rebuild it: {lock_dir}"
        )

    if rows_path.exists():
        metadata = read_json(metadata_path)
        mismatches = {
            key: {"stored": metadata.get(key), "current": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "The machine population or fixed horizon-evaluation design changed "
                "after samples were locked. Keep the existing data/settings for "
                f"comparable experiments, or intentionally delete {lock_dir}. "
                f"Changed fields: {sorted(mismatches)}"
            )
        rows = _read_locked_base_rows(rows_path)
        actual_fingerprint = _frame_fingerprint(
            rows,
            [
                "evaluation_sample_id",
                "machine_key",
                "split",
                "window_start",
                "window_end",
            ],
        )
        if metadata.get("sample_fingerprint") != actual_fingerprint:
            raise ValueError(f"The fixed horizon-evaluation row file was modified: {rows_path}")
        audit = pd.read_csv(audit_path, low_memory=False) if audit_path.exists() else pd.DataFrame()
        _validate_fixed_horizon_rows(rows, split_assignments)
        return rows, audit, True, rows_path

    rows, audit = build_fixed_horizon_evaluation_base_rows(
        machine_master=machine_master,
        sources=sources,
        split_assignments=split_assignments,
        window_config=window_config,
        config=config,
        minimum_followup_days=minimum_followup_days,
    )
    _validate_fixed_horizon_rows(rows, split_assignments)
    rows.to_csv(rows_path, index=False)
    audit.to_csv(audit_path, index=False)
    metadata = {
        **expected,
        "selected_rows": int(len(rows)),
        "validation_rows": int(rows["split"].eq("validation").sum()),
        "test_rows": int(rows["split"].eq("test").sum()),
        "sample_fingerprint": _frame_fingerprint(
            rows,
            [
                "evaluation_sample_id",
                "machine_key",
                "split",
                "window_start",
                "window_end",
            ],
        ),
    }
    write_json(metadata, metadata_path)
    return rows, audit, False, rows_path


def _horizon_label_summary(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ["validation", "test"]:
        sub = df[df["split"].eq(split_name)].copy()
        previous_positive_count = -1
        for horizon in horizons:
            col = future_claim_target_col(horizon)
            y = pd.to_numeric(sub.get(col, pd.Series(0, index=sub.index)), errors="coerce").fillna(0).astype(int)
            positive_count = int(y.sum())
            rows.append(
                {
                    "split": split_name,
                    "horizon_days": int(horizon),
                    "rows": int(len(sub)),
                    "positive_rows": positive_count,
                    "negative_rows": int(len(sub) - positive_count),
                    "positive_rate": float(y.mean()) if len(y) else None,
                    "positive_count_non_decreasing": (
                        True if previous_positive_count < 0 else positive_count >= previous_positive_count
                    ),
                }
            )
            previous_positive_count = positive_count
    return pd.DataFrame(rows)


def run() -> None:
    config.refresh_derived_config()
    step_dir = config.OUTPUT_DIR / "02_case_control_datasets"
    ensure_dir(step_dir)
    ensure_dir(Path(config.FIXED_SPLIT_ASSET_DIR))

    print("Loading sources for case-control dataset build...")
    sources = load_sources(config, include_operation=True)
    episodes = _load_episodes()
    machine_master = build_machine_master(sources)
    machine_master.to_csv(step_dir / "machine_master_source_coverage.csv", index=False)

    assignment_path = Path(config.FIXED_SPLIT_ASSET_DIR) / "fixed_machine_split_assignments.csv"
    split_assignments, machine_split_summary = build_or_load_fixed_machine_split_assignments(
        machine_master=machine_master,
        claim_history_episodes=episodes,
        config=config,
        assignment_path=assignment_path,
    )
    # Save convenient copies in the current run output while retaining the central lock.
    split_assignments.to_csv(step_dir / "fixed_machine_split_assignments.csv", index=False)
    machine_split_summary.to_csv(step_dir / "fixed_machine_split_summary.csv", index=False)

    dataset_index_rows: list[dict] = []
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
        claim_selection_audit.to_csv(
            dataset_dir / "positive_claim_selection_audit.csv", index=False
        )
        selected_episodes.to_csv(
            dataset_dir / "selected_positive_claim_events.csv", index=False
        )

        train_rows, train_negative_audit = build_case_control_base_rows(
            episodes=selected_episodes,
            machine_master=machine_master,
            sources=sources,
            window_config=window_config,
            config=config,
            claim_history_episodes=episodes,
            split_assignments=split_assignments,
            included_splits=["train"],
        )
        holdout_rows, holdout_negative_audit, holdout_reused, holdout_path = (
            _build_or_load_fixed_holdout_rows(
                window_config=window_config,
                selected_episodes=selected_episodes,
                all_episodes=episodes,
                machine_master=machine_master,
                sources=sources,
                split_assignments=split_assignments,
            )
        )
        base_rows = pd.concat([train_rows, holdout_rows], ignore_index=True, sort=False)
        if not base_rows.empty:
            base_rows = base_rows.sort_values(
                ["split", "case_control_group_id", "target", "machine_key"],
                ascending=[True, True, False, True],
                kind="mergesort",
            ).reset_index(drop=True)
        base_rows.to_csv(dataset_dir / "case_control_base_rows.csv", index=False)
        train_rows.to_csv(dataset_dir / "training_base_rows.csv", index=False)
        holdout_rows.to_csv(dataset_dir / "fixed_validation_test_base_rows.csv", index=False)

        train_negative_audit = train_negative_audit.copy()
        train_negative_audit["dataset_partition"] = "train"
        holdout_negative_audit = holdout_negative_audit.copy()
        if not holdout_negative_audit.empty:
            holdout_negative_audit["dataset_partition"] = "validation_test_locked"
        negative_audit = pd.concat(
            [train_negative_audit, holdout_negative_audit], ignore_index=True, sort=False
        )
        negative_audit.to_csv(dataset_dir / "negative_sampling_audit.csv", index=False)
        train_negative_audit.to_csv(
            dataset_dir / "training_negative_sampling_audit.csv", index=False
        )
        holdout_negative_audit.to_csv(
            dataset_dir / "fixed_validation_test_negative_sampling_audit.csv", index=False
        )

        if base_rows.empty:
            print(f"  [WARN] No eligible rows generated for {dataset_id}")
            continue

        full_df = build_window_features(
            base_rows,
            sources=sources,
            episodes=episodes,
            config=config,
        )
        if str(config.EVALUATION_TARGET_MODE).strip().lower() == "claim_within_horizon":
            full_df = annotate_future_claim_outcomes(
                full_df,
                claim_history_episodes=episodes,
                config=config,
            )
        validate_dataset_features(full_df, config)
        split_summary = validate_fixed_dataset_splits(full_df, config)

        full_path = dataset_dir / "case_control_dataset_with_fixed_split.csv"
        train_path = dataset_dir / "case_control_training_dataset.csv"
        validation_path = dataset_dir / "case_control_validation_dataset.csv"
        test_path = dataset_dir / "case_control_test_dataset.csv"

        train_df = full_df[full_df["split"].eq("train")].copy()
        validation_df = full_df[full_df["split"].eq("validation")].copy()
        test_df = full_df[full_df["split"].eq("test")].copy()
        full_df.to_csv(full_path, index=False)
        train_df.to_csv(train_path, index=False)
        validation_df.to_csv(validation_path, index=False)
        test_df.to_csv(test_path, index=False)
        split_summary.to_csv(dataset_dir / "fixed_split_dataset_summary.csv", index=False)

        # --------------------------------------------------------------
        # Fixed outcome-independent horizon evaluation cohort.
        # --------------------------------------------------------------
        # The matched holdout remains the validation set for the original
        # training target and for early-stopping monitoring. When evaluating
        # claim-within-horizon outcomes, use a second fixed set of machine/window
        # identities that was sampled without inspecting future claims. This
        # allows the same rows to receive different true labels at 30/60/90/etc.
        horizon_base_lock_path = ""
        horizon_base_reused = False
        horizon_full_path = ""
        validation_horizon_path = ""
        test_horizon_path = ""
        horizon_summary_path = ""
        horizon_rows = 0
        horizon_validation_rows = 0
        horizon_test_rows = 0
        evaluation_mode = str(config.EVALUATION_TARGET_MODE).strip().lower()
        if evaluation_mode == "claim_within_horizon":
            horizons = configured_evaluation_horizons(config)
            if not horizons:
                raise ValueError(
                    "EVALUATION_CLAIM_HORIZON_DAYS must contain at least one positive horizon."
                )
            minimum_followup_days = max(
                _MIN_FIXED_HORIZON_FOLLOWUP_DAYS,
                int(max(horizons)),
            )
            (
                horizon_base_rows,
                horizon_sampling_audit,
                horizon_base_reused,
                horizon_base_lock,
            ) = _build_or_load_fixed_horizon_rows(
                window_config=window_config,
                machine_master=machine_master,
                sources=sources,
                split_assignments=split_assignments,
                minimum_followup_days=minimum_followup_days,
            )
            horizon_base_lock_path = str(horizon_base_lock)
            horizon_base_rows.to_csv(
                dataset_dir / "fixed_validation_test_horizon_base_rows.csv",
                index=False,
            )
            horizon_sampling_audit.to_csv(
                dataset_dir / "fixed_validation_test_horizon_sampling_audit.csv",
                index=False,
            )

            horizon_df = build_window_features(
                horizon_base_rows,
                sources=sources,
                episodes=episodes,
                config=config,
            )
            horizon_df = annotate_future_claim_outcomes(
                horizon_df,
                claim_history_episodes=episodes,
                config=config,
                horizons=horizons,
            )
            primary_horizon = int(max(horizons))
            primary_target_col = future_claim_target_col(primary_horizon)
            horizon_df["target"] = pd.to_numeric(
                horizon_df[primary_target_col], errors="coerce"
            ).fillna(0).astype(int)
            horizon_df["target_definition"] = (
                f"inspection_only_{primary_target_col}; metrics_recompute_each_horizon"
            )
            validate_dataset_features(horizon_df, config)

            horizon_validation_df = horizon_df[
                horizon_df["split"].eq("validation")
            ].copy()
            horizon_test_df = horizon_df[horizon_df["split"].eq("test")].copy()
            horizon_full_file = dataset_dir / "fixed_horizon_evaluation_dataset.csv"
            validation_horizon_file = (
                dataset_dir / "fixed_validation_horizon_evaluation_dataset.csv"
            )
            test_horizon_file = dataset_dir / "fixed_test_horizon_evaluation_dataset.csv"
            horizon_summary_file = dataset_dir / "fixed_horizon_label_summary.csv"
            horizon_df.to_csv(horizon_full_file, index=False)
            horizon_validation_df.to_csv(validation_horizon_file, index=False)
            horizon_test_df.to_csv(test_horizon_file, index=False)

            horizon_label_summary = _horizon_label_summary(horizon_df, horizons)
            if not horizon_label_summary.empty and not horizon_label_summary[
                "positive_count_non_decreasing"
            ].all():
                raise ValueError(
                    "Future-claim positive counts must be non-decreasing as the horizon grows."
                )
            horizon_label_summary.to_csv(horizon_summary_file, index=False)

            horizon_full_path = str(horizon_full_file)
            validation_horizon_path = str(validation_horizon_file)
            test_horizon_path = str(test_horizon_file)
            horizon_summary_path = str(horizon_summary_file)
            horizon_rows = int(len(horizon_df))
            horizon_validation_rows = int(len(horizon_validation_df))
            horizon_test_rows = int(len(horizon_test_df))

            for split_name, split_summary_df in horizon_label_summary.groupby("split"):
                counts = split_summary_df["positive_rows"].tolist()
                if len(set(counts)) == 1:
                    print(
                        f"  [WARN] {split_name} horizon cohort has the same positive count "
                        f"at all configured horizons: {counts[0]}. The labels were recomputed "
                        "correctly, but this sample happens not to contain claims in the "
                        "intervals between the configured cutoffs."
                    )

        group_split = (
            full_df[["case_control_group_id", "case_machine_key", "split"]]
            .drop_duplicates()
            .sort_values(["split", "case_control_group_id"], kind="mergesort")
        )
        group_split.to_csv(
            dataset_dir / "case_control_group_split_assignments.csv", index=False
        )

        negative_type_summary = (
            full_df[full_df["target"].eq(0)]
            .groupby(["split", "negative_sampling_type"], dropna=False)
            .agg(rows=("target", "size"), machines=("machine_key", "nunique"))
            .reset_index()
        )
        negative_type_summary.to_csv(
            dataset_dir / "negative_sampling_type_summary.csv", index=False
        )

        positive_rows = int(full_df["target"].eq(1).sum())
        negative_rows = int(full_df["target"].eq(0).sum())
        holdout_mode = normalize_negative_sampling_mode(
            config.FIXED_HOLDOUT_NEGATIVE_SAMPLING_MODE,
            "FIXED_HOLDOUT_NEGATIVE_SAMPLING_MODE",
        )
        holdout_count = validate_negative_count(
            config.FIXED_HOLDOUT_NEGATIVES_PER_POSITIVE_CASE,
            "FIXED_HOLDOUT_NEGATIVES_PER_POSITIVE_CASE",
        )
        summary = {
            "dataset_id": dataset_id,
            "window_config": dict(window_config),
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "negative_sampling_mode": negative_sampling_mode(config),
            "negatives_per_positive_case_requested": negatives_per_positive(config),
            "training_negative_sampling_mode": negative_sampling_mode(config),
            "training_negatives_per_positive_case_requested": negatives_per_positive(config),
            "fixed_holdout_negative_sampling_mode": holdout_mode,
            "fixed_holdout_negatives_per_positive_case": holdout_count,
            "fixed_holdout_base_rows_path": str(holdout_path),
            "fixed_holdout_reused": bool(holdout_reused),
            "feature_set": config.FEATURE_SET,
            "evaluation_target_mode": config.EVALUATION_TARGET_MODE,
            "evaluation_claim_horizon_days": list(config.EVALUATION_CLAIM_HORIZON_DAYS),
            "fixed_horizon_evaluation_base_rows_path": horizon_base_lock_path,
            "fixed_horizon_evaluation_base_rows_reused": bool(horizon_base_reused),
            "fixed_horizon_evaluation_dataset_path": horizon_full_path,
            "validation_horizon_evaluation_dataset_path": validation_horizon_path,
            "test_horizon_evaluation_dataset_path": test_horizon_path,
            "fixed_horizon_label_summary_path": horizon_summary_path,
            "fixed_horizon_evaluation_rows": horizon_rows,
            "fixed_horizon_validation_rows": horizon_validation_rows,
            "fixed_horizon_test_rows": horizon_test_rows,
            "numeric_features": list(config.NUMERIC_FEATURES),
            "categorical_features": list(config.CATEGORICAL_FEATURES),
            "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "positive_rate": positive_rows / len(full_df) if len(full_df) else None,
            "unique_case_control_groups": int(
                full_df["case_control_group_id"].nunique(dropna=True)
            ),
            "unique_machines": int(full_df["machine_key"].nunique(dropna=True)),
                "fixed_machine_split_strategy": (
                "deterministic_random_machine_level_stratified_by_full_model_and_claim_history"
            ),
            "fixed_machine_split_assignment_path": str(assignment_path),
            "fixed_split_asset_dir": str(config.FIXED_SPLIT_ASSET_DIR),
            "train_ratio": float(config.TRAIN_RATIO),
            "validation_ratio": float(config.VALIDATION_RATIO),
            "test_ratio": float(config.TEST_RATIO),
            "negative_no_claim_days_after_window_end": int(
                config.NEGATIVE_NO_CLAIM_DAYS_AFTER_WINDOW_END
            ),
            "negative_exclude_prior_claim_days_before_window_start": int(
                config.NEGATIVE_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START
            ),
            "require_source_coverage_overlap_window": bool(
                config.REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW
            ),
            "positive_claim_selection_mode": config.POSITIVE_CLAIM_SELECTION_MODE,
            "claim_events_available_before_selection": int(len(episodes)),
            "claim_events_selected_before_source_coverage_filter": int(
                len(selected_episodes)
            ),
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
        }
        summary.update(_split_counts(full_df))
        write_json(summary, dataset_dir / "dataset_summary.json")

        availability_cols = [
            "has_any_source_window",
            "has_fault_window",
            "has_fluid_window",
            "has_maintenance_window",
            "has_operation_window",
        ]
        availability_rows = []
        for col in availability_cols:
            if col not in full_df.columns:
                continue
            for value, sub in full_df.groupby(col, dropna=False):
                availability_rows.append(
                    {
                        "feature": col,
                        "value": value,
                        "rows": int(len(sub)),
                        "positive_rate": float(sub["target"].mean()) if len(sub) else None,
                    }
                )
        pd.DataFrame(availability_rows).to_csv(
            dataset_dir / "source_availability_target_rates.csv", index=False
        )

        index_row = {
            "dataset_id": dataset_id,
            "window_name": window_name,
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "negative_sampling_mode": negative_sampling_mode(config),
            "negatives_per_positive_case_requested": negatives_per_positive(config),
            "training_negative_sampling_mode": negative_sampling_mode(config),
            "training_negatives_per_positive_case_requested": negatives_per_positive(config),
            "fixed_holdout_negative_sampling_mode": holdout_mode,
            "fixed_holdout_negatives_per_positive_case": holdout_count,
            "fixed_holdout_base_rows_path": str(holdout_path),
            "fixed_holdout_reused": bool(holdout_reused),
            "feature_set": config.FEATURE_SET,
                "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
            "dataset_dir": str(dataset_dir),
            "fixed_machine_split_assignment_path": str(assignment_path),
            "positive_claim_selection_mode": config.POSITIVE_CLAIM_SELECTION_MODE,
            "evaluation_target_mode": config.EVALUATION_TARGET_MODE,
            "evaluation_claim_horizon_days": ",".join(
                str(int(x)) for x in config.EVALUATION_CLAIM_HORIZON_DAYS
            ),
            "fixed_horizon_evaluation_base_rows_path": horizon_base_lock_path,
            "fixed_horizon_evaluation_base_rows_reused": bool(horizon_base_reused),
            "fixed_horizon_evaluation_dataset_path": horizon_full_path,
            "validation_horizon_evaluation_dataset_path": validation_horizon_path,
            "test_horizon_evaluation_dataset_path": test_horizon_path,
            "fixed_horizon_label_summary_path": horizon_summary_path,
            "fixed_horizon_evaluation_rows": horizon_rows,
            "fixed_horizon_validation_rows": horizon_validation_rows,
            "fixed_horizon_test_rows": horizon_test_rows,
        }
        index_row.update(_split_counts(full_df))
        dataset_index_rows.append(index_row)

        print(
            f"  rows={len(full_df):,}; positives={positive_rows:,}; negatives={negative_rows:,}; "
            f"train={len(train_df):,}; validation={len(validation_df):,}; test={len(test_df):,}; "
            f"holdout={'reused' if holdout_reused else 'created'}"
        )

    dataset_index = pd.DataFrame(dataset_index_rows)
    dataset_index.to_csv(step_dir / "dataset_index.csv", index=False)
    write_json(
        {
            "step": "02_build_case_control_dataset",
            "output_dir": str(step_dir),
            "dataset_count": int(len(dataset_index)),
            "window_configs": config.WINDOW_CONFIGS,
                "training_negative_sampling_mode": negative_sampling_mode(config),
            "training_negatives_per_positive_case": negatives_per_positive(config),
            "fixed_holdout_negative_sampling_mode": config.FIXED_HOLDOUT_NEGATIVE_SAMPLING_MODE,
            "fixed_holdout_negatives_per_positive_case": int(
                config.FIXED_HOLDOUT_NEGATIVES_PER_POSITIVE_CASE
            ),
            "fixed_split_asset_dir": str(config.FIXED_SPLIT_ASSET_DIR),
            "feature_set": config.FEATURE_SET,
            "evaluation_target_mode": config.EVALUATION_TARGET_MODE,
            "evaluation_claim_horizon_days": list(config.EVALUATION_CLAIM_HORIZON_DAYS),
            "positive_claim_selection_mode": config.POSITIVE_CLAIM_SELECTION_MODE,
            "split": {
                "strategy": (
                    "fixed deterministic random machine-level split; stratified by full_model "
                    "and eligible claim-history status"
                ),
                "assignment_path": str(assignment_path),
                "train_ratio": float(config.TRAIN_RATIO),
                "validation_ratio": float(config.VALIDATION_RATIO),
                "test_ratio": float(config.TEST_RATIO),
                "leakage_rule": "each physical machine appears in exactly one split",
                "holdout_lock_rule": (
                    "validation/test base-row identities are persisted centrally and reused "
                    "across all experiment output folders"
                ),
            },
        },
        step_dir / "run_summary.json",
    )
    print(f"02_build_case_control_dataset completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
