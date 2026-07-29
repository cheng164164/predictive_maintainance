"""Step 02: build training data and fixed validation/test ratio cohorts.

Training negatives follow ``NEGATIVE_SAMPLING_MODE``. Validation and test use the
separate ``HOLDOUT_NEGATIVE_SAMPLING_MODE`` switch:

- ``random`` selects independent eligible no-claim windows;
- ``controlled`` matches controls to the positive machine's full model and exact
  calendar feature window;
- each physical machine belongs to only one train/validation/test split;
- each holdout machine contributes at most one row;
- positive rows use the configured claim-relative feature window;
- ratio datasets are nested for fair 1:1, 2:1, ..., N:1 comparisons.

Feature engineering is performed once on the maximum-ratio master pool. Smaller
ratio datasets are created by subsetting already-engineered rows.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import config
from cc_utils import (
    annotate_future_claim_outcomes,
    build_case_control_base_rows,
    build_fixed_horizon_evaluation_base_rows,
    build_machine_master,
    build_or_load_fixed_machine_split_assignments,
    build_holdout_ratio_base_pool,
    build_window_features,
    configured_evaluation_horizons,
    configured_holdout_negative_ratios,
    ensure_dir,
    future_claim_target_col,
    load_sources,
    holdout_negative_sampling_mode,
    holdout_machine_selection_scope,
    negative_sampling_mode,
    negatives_per_positive,
    holdout_ratio_subset,
    read_json,
    select_positive_claims_for_window_config,
    validate_dataset_features,
    validate_fixed_dataset_splits,
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
    "as_of_anchor_date",
    "as_of_actual_next_claim_date",
    "next_claim_date_on_or_after_window_end",
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
    """Return a stable, order-independent SHA-256 fingerprint."""
    work = df.reindex(columns=columns).copy()
    for col in columns:
        if "date" in col.lower() or pd.api.types.is_datetime64_any_dtype(work[col]):
            values = pd.to_datetime(work[col], errors="coerce")
            work[col] = values.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("<NA>")
        else:
            # CSV round-trips convert empty strings to missing values by default.
            # Canonicalize both representations so the lock fingerprint remains
            # stable without weakening the tamper/change check.
            values = work[col].astype("string").str.strip()
            work[col] = values.fillna("<NA>").replace("", "<NA>")
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
    for col in [
        "holdout_positive_rank",
        "holdout_negative_rank",
        "holdout_control_rank_within_positive",
        "matched_holdout_positive_rank",
    ]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def _random_holdout_expected_metadata(
    window_config: Mapping,
    selected_episodes: pd.DataFrame,
    all_episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> dict:
    """Build immutable holdout-lock metadata for the configured design.

    The random-mode payload intentionally preserves the previous version-2 keys
    exactly, so existing fixed random holdout assets remain reusable and the
    default smoke-run point estimates do not change.
    """
    mode = holdout_negative_sampling_mode(config)
    selection_scope = holdout_machine_selection_scope(config, window_config)
    selected_with_split = selected_episodes.merge(
        split_assignments[["machine_key", "split"]],
        on="machine_key",
        how="left",
        validate="many_to_one",
    )
    selected_holdout = selected_with_split[
        selected_with_split["split"].isin(["validation", "test"])
    ].copy()
    ratios = configured_holdout_negative_ratios(config)
    common = {
        "window_name": window_config_name(window_config),
        "window_config": {
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
        },
        "holdout_negative_to_positive_ratios": ratios,
        "holdout_max_negative_to_positive_ratio": int(max(ratios)),
        "holdout_random_state": int(config.HOLDOUT_RANDOM_STATE),
        "fixed_split_random_state": int(config.FIXED_SPLIT_RANDOM_STATE),
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
    shared_across_windows = selection_scope != window_config_name(window_config)
    if mode == "random":
        payload = {
            "lock_version": 2,
            "strategy": "fixed_random_machine_level_joint_class_assignment_nested_ratio_pool",
            "negative_candidate_policy": (
                "any_machine_with_an_eligible_no_claim_window_not_selected_as_positive"
            ),
            **common,
        }
        # Preserve the historical version-2 payload exactly for a normal
        # single-window run. Multi-window experiments use a version-3 lock so
        # each window shares one deterministic machine-ranking scope.
        if shared_across_windows:
            payload.update({
                "lock_version": 3,
                "strategy": (
                    "fixed_random_machine_level_joint_class_assignment_nested_ratio_pool_"
                    "shared_machine_ranking_across_windows"
                ),
                "holdout_machine_selection_scope": selection_scope,
            })
        return payload
    if mode == "as_of_anchor":
        return {
            "lock_version": 1,
            "strategy": "fixed_shared_as_of_anchor_calendar_snapshot_nested_ratio_pool",
            "holdout_negative_sampling_mode": "as_of_anchor",
            "negative_candidate_policy": (
                "same_anchor_date_no_claim_within_lead_min_days_unique_machine"
            ),
            "as_of_min_positive_machines": int(
                getattr(config, "HOLDOUT_AS_OF_MIN_POSITIVE_MACHINES", 10)
            ),
            "as_of_required_followup_days": int(
                max(
                    [int(window_config["lead_min_days"]),
                     *configured_evaluation_horizons(config)]
                )
            ),
            **common,
        }
    payload = {
        "lock_version": 1,
        "strategy": "fixed_controlled_same_window_same_full_model_nested_ratio_pool",
        "holdout_negative_sampling_mode": "controlled",
        "negative_candidate_policy": (
            "same_split_same_full_model_exact_positive_calendar_window_unique_machine"
        ),
        **common,
    }
    if shared_across_windows:
        payload.update({
            "lock_version": 2,
            "strategy": (
                "fixed_controlled_same_window_same_full_model_nested_ratio_pool_"
                "shared_machine_ranking_across_windows"
            ),
            "holdout_machine_selection_scope": selection_scope,
        })
    return payload

def _validate_random_holdout_master_rows(
    rows: pd.DataFrame,
    split_assignments: pd.DataFrame,
) -> None:
    required = {
        "split",
        "machine_key",
        "case_control_group_id",
        "target",
        "window_start",
        "window_end",
        "holdout_sampling_design",
        "holdout_positive_rank",
        "holdout_negative_rank",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Random holdout master rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("Random validation/test master pool is empty.")
    invalid = sorted(set(rows["split"].dropna().astype(str)) - {"validation", "test"})
    if invalid:
        raise ValueError(f"Random holdout contains unexpected split labels: {invalid}")
    if rows["machine_key"].astype(str).duplicated().any():
        examples = rows.loc[
            rows["machine_key"].astype(str).duplicated(keep=False), "machine_key"
        ].astype(str).unique().tolist()[:10]
        raise ValueError(
            "Each random holdout machine must contribute at most one row. "
            f"Examples: {examples}"
        )
    if rows["case_control_group_id"].astype(str).duplicated().any():
        raise ValueError("Random holdout row/group IDs must be unique.")

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
    bad = check[
        check["assigned_split"].isna()
        | check["split"].astype(str).ne(check["assigned_split"].astype(str))
    ]
    if not bad.empty:
        raise ValueError("Random holdout rows do not match fixed machine assignments.")

    max_ratio = int(max(configured_holdout_negative_ratios(config)))
    for split_name, configured_split_ratio in [
        ("validation", float(config.VALIDATION_RATIO)),
        ("test", float(config.TEST_RATIO)),
    ]:
        if configured_split_ratio <= 0:
            continue
        sub = rows[rows["split"].eq(split_name)].copy()
        if sub.empty:
            raise ValueError(f"Random holdout split {split_name!r} is empty.")
        positive_rows = int(pd.to_numeric(sub["target"], errors="coerce").eq(1).sum())
        negative_rows = int(pd.to_numeric(sub["target"], errors="coerce").eq(0).sum())
        if positive_rows < 1 or negative_rows < 1:
            raise ValueError(f"Random holdout split {split_name!r} must contain both classes.")
        if negative_rows != positive_rows * max_ratio:
            raise ValueError(
                f"Random holdout split {split_name!r} does not have the configured "
                f"maximum ratio {max_ratio}:1. positives={positive_rows}, negatives={negative_rows}."
            )


def _build_or_load_fixed_random_holdout_rows(
    window_config: Mapping,
    selected_episodes: pd.DataFrame,
    all_episodes: pd.DataFrame,
    machine_master: pd.DataFrame,
    sources: Mapping[str, pd.DataFrame],
    split_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, Path]:
    """Build or load the fixed holdout pool for the configured negative design.

    The function name is retained for backward compatibility. Random mode keeps
    the previous lock filenames and fingerprint columns exactly; controlled mode
    uses a separate set of lock assets in the same window folder.
    """
    mode = holdout_negative_sampling_mode(config)
    window_name = window_config_name(window_config)
    lock_dir = Path(config.FIXED_SPLIT_ASSET_DIR) / window_name
    ensure_dir(lock_dir)
    prefix = f"fixed_{mode}_validation_test"
    rows_path = lock_dir / f"{prefix}_master_base_rows.csv"
    audit_path = lock_dir / f"{prefix}_sampling_audit.csv"
    pool_summary_path = lock_dir / f"{prefix}_pool_summary.csv"
    metadata_path = lock_dir / f"{prefix}_metadata.json"
    expected = _random_holdout_expected_metadata(
        window_config,
        selected_episodes,
        all_episodes,
        machine_master,
        split_assignments,
    )
    fingerprint_columns = [
        "case_control_group_id",
        "machine_key",
        "split",
        "target",
        "window_start",
        "window_end",
        "claim_episode_id",
        "holdout_positive_rank",
        "holdout_negative_rank",
    ]
    if mode == "controlled":
        fingerprint_columns.extend(
            ["holdout_match_id", "holdout_control_rank_within_positive"]
        )

    # Preserve the prior one-time random-lock migration. Controlled mode has its
    # own version-1 assets and never touches random locks.
    if mode == "random" and rows_path.exists() and metadata_path.exists():
        stored_metadata = read_json(metadata_path)
        if stored_metadata.get("lock_version") != expected["lock_version"]:
            print(
                f"  Rebuilding outdated random holdout lock in {lock_dir} "
                f"(version {stored_metadata.get('lock_version')} -> "
                f"{expected['lock_version']}).",
                flush=True,
            )
            for old_path in [rows_path, audit_path, pool_summary_path, metadata_path]:
                old_path.unlink(missing_ok=True)

    existing_parts = [rows_path.exists(), metadata_path.exists()]
    if any(existing_parts) and not all(existing_parts):
        raise ValueError(
            f"The fixed {mode} validation/test lock is incomplete. Delete its files "
            f"only when intentionally redesigning the holdout: {lock_dir}"
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
                f"The source population or {mode} holdout settings changed after the "
                "validation/test master pool was locked. Keep the current data/settings "
                f"or intentionally delete the {mode}-holdout lock files in {lock_dir}. "
                f"Changed fields: {sorted(mismatches)}"
            )
        rows = _read_locked_base_rows(rows_path)
        actual_fingerprint = _frame_fingerprint(rows, fingerprint_columns)
        if metadata.get("holdout_master_row_fingerprint") != actual_fingerprint:
            raise ValueError(f"The locked {mode} holdout file was modified: {rows_path}")
        audit = pd.read_csv(audit_path, low_memory=False) if audit_path.exists() else pd.DataFrame()
        pool_summary = (
            pd.read_csv(pool_summary_path, low_memory=False)
            if pool_summary_path.exists()
            else pd.DataFrame()
        )
        _validate_random_holdout_master_rows(rows, split_assignments)
        return rows, audit, pool_summary, True, rows_path

    rows, audit, pool_summary = build_holdout_ratio_base_pool(
        episodes=selected_episodes,
        machine_master=machine_master,
        sources=sources,
        window_config=window_config,
        config=config,
        claim_history_episodes=all_episodes,
        split_assignments=split_assignments,
        included_splits=("validation", "test"),
        random_state_override=int(config.HOLDOUT_RANDOM_STATE),
    )
    _validate_random_holdout_master_rows(rows, split_assignments)
    rows.to_csv(rows_path, index=False)
    audit.to_csv(audit_path, index=False)
    pool_summary.to_csv(pool_summary_path, index=False)
    metadata = {
        **expected,
        "holdout_master_rows": int(len(rows)),
        "validation_rows": int(rows["split"].eq("validation").sum()),
        "test_rows": int(rows["split"].eq("test").sum()),
        "holdout_master_row_fingerprint": _frame_fingerprint(
            rows, fingerprint_columns
        ),
    }
    write_json(metadata, metadata_path)
    return rows, audit, pool_summary, False, rows_path

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
            "The fixed horizon-evaluation lock is incomplete. Delete its files "
            f"only if intentionally rebuilding it: {lock_dir}"
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
                f"after samples were locked. Changed fields: {sorted(mismatches)}"
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
            raise ValueError(f"The fixed horizon-evaluation file was modified: {rows_path}")
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


def _horizon_label_summary(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for split_name in ["validation", "test"]:
        sub = df[df["split"].eq(split_name)].copy()
        previous_positive_count = -1
        for horizon in horizons:
            col = future_claim_target_col(horizon)
            y = pd.to_numeric(
                sub.get(col, pd.Series(0, index=sub.index)), errors="coerce"
            ).fillna(0).astype(int)
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
                        True
                        if previous_positive_count < 0
                        else positive_count >= previous_positive_count
                    ),
                }
            )
            previous_positive_count = positive_count
    return pd.DataFrame(rows)


def _ratio_slug(ratio: int) -> str:
    return f"ratio_{int(ratio)}_to_1"


def _materialize_holdout_ratio_datasets(
    holdout_master_df: pd.DataFrame,
    dataset_dir: Path,
    dataset_id: str,
) -> tuple[pd.DataFrame, Path, Path]:
    """Materialize nested validation/test ratio datasets for either holdout mode."""
    ratios = configured_holdout_negative_ratios(config)
    mode = holdout_negative_sampling_mode(config)
    index_rows: list[dict] = []
    reference_validation_path: Path | None = None
    reference_test_path: Path | None = None
    for split_name in ["validation", "test"]:
        for ratio in ratios:
            subset = holdout_ratio_subset(
                holdout_master_df,
                split_name=split_name,
                negative_to_positive_ratio=ratio,
            )
            path = dataset_dir / f"case_control_{split_name}_{_ratio_slug(ratio)}.csv"
            subset.to_csv(path, index=False)
            positive_rows = int(subset["target"].eq(1).sum())
            negative_rows = int(subset["target"].eq(0).sum())
            nested_rules = subset.get(
                "holdout_nested_negative_rule", pd.Series(dtype=str)
            ).dropna().astype(str).unique().tolist()
            anchor_values = (
                pd.to_datetime(subset["as_of_anchor_date"], errors="coerce")
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
                if "as_of_anchor_date" in subset.columns
                else []
            )
            as_of_horizon_values = (
                pd.to_numeric(
                    subset["as_of_prediction_horizon_days"], errors="coerce"
                )
                .dropna()
                .drop_duplicates()
                .sort_values()
                .astype(int)
                .tolist()
                if "as_of_prediction_horizon_days" in subset.columns
                else []
            )
            index_rows.append(
                {
                    "dataset_id": dataset_id,
                    "split": split_name,
                    "holdout_negative_sampling_mode": mode,
                    "negative_to_positive_ratio_requested": int(ratio),
                    "rows": int(len(subset)),
                    "positive_rows": positive_rows,
                    "negative_rows": negative_rows,
                    "actual_negative_to_positive_ratio": (
                        float(negative_rows / positive_rows) if positive_rows else np.nan
                    ),
                    "machines": int(subset["machine_key"].nunique(dropna=True)),
                    "as_of_anchor_date": (
                        pd.Timestamp(anchor_values[0]).strftime("%Y-%m-%d")
                        if len(anchor_values) == 1
                        else ""
                    ),
                    "as_of_prediction_horizon_days": (
                        int(as_of_horizon_values[0])
                        if len(as_of_horizon_values) == 1
                        else np.nan
                    ),
                    "dataset_path": str(path),
                    "nested_negative_rule": (
                        nested_rules[0] if nested_rules else "fixed nested holdout negatives"
                    ),
                }
            )
            if ratio == max(ratios):
                if split_name == "validation":
                    reference_validation_path = path
                else:
                    reference_test_path = path
    if reference_validation_path is None or reference_test_path is None:
        raise ValueError("Failed to create reference validation/test ratio datasets.")
    ratio_index = pd.DataFrame(index_rows).sort_values(
        ["split", "negative_to_positive_ratio_requested"], kind="mergesort"
    )
    mode_path = dataset_dir / f"{mode}_holdout_ratio_index.csv"
    ratio_index.to_csv(mode_path, index=False)
    ratio_index.to_csv(dataset_dir / "holdout_ratio_index.csv", index=False)
    return ratio_index, reference_validation_path, reference_test_path

def run() -> None:
    config.refresh_derived_config()
    step_dir = config.OUTPUT_DIR / "02_case_control_datasets"
    ensure_dir(step_dir)
    ensure_dir(Path(config.FIXED_SPLIT_ASSET_DIR))

    print("Loading sources for case-control dataset build...", flush=True)
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
    split_assignments.to_csv(step_dir / "fixed_machine_split_assignments.csv", index=False)
    machine_split_summary.to_csv(step_dir / "fixed_machine_split_summary.csv", index=False)

    dataset_index_rows: list[dict] = []
    holdout_mode = holdout_negative_sampling_mode(config)
    for window_config in config.WINDOW_CONFIGS:
        window_name = window_config_name(window_config)
        dataset_id = window_dataset_id(window_config, config)
        dataset_dir = step_dir / dataset_id
        ensure_dir(dataset_dir)
        print(f"Building dataset: {dataset_id}", flush=True)

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
        (
            holdout_master_rows,
            holdout_sampling_audit,
            holdout_pool_summary,
            holdout_reused,
            holdout_lock_path,
        ) = _build_or_load_fixed_random_holdout_rows(
            window_config=window_config,
            selected_episodes=selected_episodes,
            all_episodes=episodes,
            machine_master=machine_master,
            sources=sources,
            split_assignments=split_assignments,
        )

        base_rows = pd.concat(
            [train_rows, holdout_master_rows], ignore_index=True, sort=False
        )
        if base_rows.empty:
            print(f"  [WARN] No eligible rows generated for {dataset_id}", flush=True)
            continue
        base_rows = base_rows.sort_values(
            ["split", "target", "machine_key", "window_end"],
            ascending=[True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        base_rows.to_csv(dataset_dir / "case_control_base_rows.csv", index=False)
        train_rows.to_csv(dataset_dir / "training_base_rows.csv", index=False)
        holdout_master_rows.to_csv(
            dataset_dir / f"fixed_{holdout_mode}_validation_test_master_base_rows.csv",
            index=False,
        )
        # Generic alias always points to the active maximum-ratio holdout pool.
        holdout_master_rows.to_csv(
            dataset_dir / "fixed_validation_test_base_rows.csv", index=False
        )
        train_negative_audit = train_negative_audit.copy()
        train_negative_audit["dataset_partition"] = "train"
        holdout_sampling_audit = holdout_sampling_audit.copy()
        if not holdout_sampling_audit.empty:
            holdout_sampling_audit["dataset_partition"] = (
                f"validation_test_{holdout_mode}_pool"
            )
        pd.concat(
            [train_negative_audit, holdout_sampling_audit],
            ignore_index=True,
            sort=False,
        ).to_csv(dataset_dir / "negative_sampling_audit.csv", index=False)
        train_negative_audit.to_csv(
            dataset_dir / "training_negative_sampling_audit.csv", index=False
        )
        holdout_sampling_audit.to_csv(
            dataset_dir / f"fixed_{holdout_mode}_validation_test_sampling_audit.csv",
            index=False,
        )
        holdout_sampling_audit.to_csv(
            dataset_dir / "fixed_validation_test_sampling_audit.csv", index=False
        )
        holdout_pool_summary.to_csv(
            dataset_dir / f"fixed_{holdout_mode}_validation_test_pool_summary.csv",
            index=False,
        )
        holdout_pool_summary.to_csv(
            dataset_dir / "fixed_validation_test_pool_summary.csv", index=False
        )

        print(
            f"  Engineering {config.FEATURE_SET} features once for "
            f"{len(base_rows):,} train + max-ratio holdout rows...",
            flush=True,
        )
        full_df = build_window_features(
            base_rows,
            sources=sources,
            episodes=episodes,
            config=config,
        )
        evaluation_mode = str(config.EVALUATION_TARGET_MODE).strip().lower()
        evaluation_horizons = (
            configured_evaluation_horizons(config)
            if evaluation_mode == "claim_within_horizon"
            else []
        )
        # Always retain actual next-claim timing for prediction review. Horizon
        # target columns are materialized only when horizon evaluation is active.
        full_df = annotate_future_claim_outcomes(
            full_df,
            claim_history_episodes=episodes,
            config=config,
            horizons=evaluation_horizons,
        )
        validate_dataset_features(full_df, config)
        split_summary = validate_fixed_dataset_splits(full_df, config)

        full_path = dataset_dir / "case_control_dataset_with_fixed_split.csv"
        train_path = dataset_dir / "case_control_training_dataset.csv"
        holdout_master_path = dataset_dir / f"{holdout_mode}_holdout_master_dataset.csv"
        train_df = full_df[full_df["split"].eq("train")].copy()
        holdout_master_df = full_df[
            full_df["split"].isin(["validation", "test"])
        ].copy()
        full_df.to_csv(full_path, index=False)
        train_df.to_csv(train_path, index=False)
        holdout_master_df.to_csv(holdout_master_path, index=False)
        # Generic alias always points to the active holdout design selected in config.
        holdout_master_df.to_csv(dataset_dir / "holdout_master_dataset.csv", index=False)
        split_summary.to_csv(dataset_dir / "fixed_split_dataset_summary.csv", index=False)

        ratio_index, validation_path, test_path = _materialize_holdout_ratio_datasets(
            holdout_master_df=holdout_master_df,
            dataset_dir=dataset_dir,
            dataset_id=dataset_id,
        )
        ratio_index_path = dataset_dir / f"{holdout_mode}_holdout_ratio_index.csv"

        # Optional outcome-independent horizon cohort remains separate because the
        # case-control ratio pool is intentionally outcome-conditioned.
        horizon_base_lock_path = ""
        horizon_base_reused = False
        horizon_full_path = ""
        validation_horizon_path = ""
        test_horizon_path = ""
        horizon_summary_path = ""
        horizon_rows = 0
        horizon_validation_rows = 0
        horizon_test_rows = 0
        if evaluation_mode == "claim_within_horizon" and holdout_mode == "as_of_anchor":
            # The active as-of holdout is already an outcome-reviewable fleet
            # snapshot. Reuse its fixed rows and dynamically relabel them for
            # every configured horizon instead of creating a second cohort.
            horizons = configured_evaluation_horizons(config)
            horizon_full_path = str(holdout_master_path)
            validation_horizon_path = str(validation_path)
            test_horizon_path = str(test_path)
            horizon_summary_file = dataset_dir / "as_of_anchor_horizon_label_summary.csv"
            _horizon_label_summary(holdout_master_df, horizons).to_csv(
                horizon_summary_file, index=False
            )
            horizon_summary_path = str(horizon_summary_file)
            horizon_rows = int(len(holdout_master_df))
            horizon_validation_rows = int(holdout_master_df["split"].eq("validation").sum())
            horizon_test_rows = int(holdout_master_df["split"].eq("test").sum())
        elif evaluation_mode == "claim_within_horizon":
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
            horizon_label_summary.to_csv(horizon_summary_file, index=False)
            horizon_full_path = str(horizon_full_file)
            validation_horizon_path = str(validation_horizon_file)
            test_horizon_path = str(test_horizon_file)
            horizon_summary_path = str(horizon_summary_file)
            horizon_rows = int(len(horizon_df))
            horizon_validation_rows = int(len(horizon_validation_df))
            horizon_test_rows = int(len(horizon_test_df))

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
        ratios = configured_holdout_negative_ratios(config)
        active_anchor_dates = (
            pd.to_datetime(holdout_master_df["as_of_anchor_date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
            if "as_of_anchor_date" in holdout_master_df.columns
            else []
        )
        active_as_of_anchor_date = (
            pd.Timestamp(active_anchor_dates[0]).strftime("%Y-%m-%d")
            if len(active_anchor_dates) == 1
            else ""
        )
        active_as_of_horizon_values = (
            pd.to_numeric(
                holdout_master_df["as_of_prediction_horizon_days"], errors="coerce"
            )
            .dropna()
            .drop_duplicates()
            .sort_values()
            .astype(int)
            .tolist()
            if "as_of_prediction_horizon_days" in holdout_master_df.columns
            else []
        )
        active_as_of_prediction_horizon_days = (
            int(active_as_of_horizon_values[0])
            if len(active_as_of_horizon_values) == 1
            else None
        )
        summary = {
            "dataset_id": dataset_id,
            "window_config": dict(window_config),
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "training_negative_sampling_mode": negative_sampling_mode(config),
            "training_negatives_per_positive_case_requested": negatives_per_positive(config),
            "holdout_negative_sampling_mode": holdout_mode,
            "holdout_negative_to_positive_ratios": ratios,
            "holdout_reference_ratio": int(max(ratios)),
            "holdout_random_state": int(config.HOLDOUT_RANDOM_STATE),
            "as_of_anchor_date": active_as_of_anchor_date,
            "as_of_prediction_horizon_days": active_as_of_prediction_horizon_days,
            "holdout_master_base_rows_path": str(holdout_lock_path),
            "holdout_master_reused": bool(holdout_reused),
            "holdout_master_dataset_path": str(holdout_master_path),
            "holdout_ratio_index_path": str(ratio_index_path),
            # Backward-compatible random_* aliases.
            "random_holdout_negative_to_positive_ratios": ratios,
            "random_holdout_reference_ratio": int(max(ratios)),
            "random_holdout_random_state": int(config.HOLDOUT_RANDOM_STATE),
            "random_holdout_master_base_rows_path": str(holdout_lock_path),
            "random_holdout_master_reused": bool(holdout_reused),
            "random_holdout_master_dataset_path": str(holdout_master_path),
            "random_holdout_ratio_index_path": str(ratio_index_path),
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

        index_row = {
            "dataset_id": dataset_id,
            "window_name": window_name,
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "training_negative_sampling_mode": negative_sampling_mode(config),
            "training_negatives_per_positive_case_requested": negatives_per_positive(config),
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
            "holdout_negative_sampling_mode": holdout_mode,
            "holdout_negative_to_positive_ratios": ",".join(str(x) for x in ratios),
            "holdout_reference_ratio": int(max(ratios)),
            "as_of_anchor_date": active_as_of_anchor_date,
            "as_of_prediction_horizon_days": active_as_of_prediction_horizon_days,
            "holdout_master_base_rows_path": str(holdout_lock_path),
            "holdout_master_reused": bool(holdout_reused),
            "holdout_master_dataset_path": str(holdout_master_path),
            "holdout_ratio_index_path": str(ratio_index_path),
            # Backward-compatible random_* aliases.
            "random_holdout_negative_to_positive_ratios": ",".join(
                str(x) for x in ratios
            ),
            "random_holdout_reference_ratio": int(max(ratios)),
            "random_holdout_master_base_rows_path": str(holdout_lock_path),
            "random_holdout_master_reused": bool(holdout_reused),
            "random_holdout_master_dataset_path": str(holdout_master_path),
            "random_holdout_ratio_index_path": str(ratio_index_path),
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

        validation_reference = ratio_index[
            (ratio_index["split"].eq("validation"))
            & (
                ratio_index["negative_to_positive_ratio_requested"].eq(max(ratios))
            )
        ].iloc[0]
        test_reference = ratio_index[
            (ratio_index["split"].eq("test"))
            & (
                ratio_index["negative_to_positive_ratio_requested"].eq(max(ratios))
            )
        ].iloc[0]
        print(
            f"  train rows={len(train_df):,}; {holdout_mode} validation reference="
            f"{int(validation_reference['positive_rows']):,}+"
            f"{int(validation_reference['negative_rows']):,}; {holdout_mode} test reference="
            f"{int(test_reference['positive_rows']):,}+"
            f"{int(test_reference['negative_rows']):,}; "
            f"holdout={'reused' if holdout_reused else 'created'}",
            flush=True,
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
            "holdout_negative_sampling_mode": holdout_mode,
            "holdout_negative_to_positive_ratios": configured_holdout_negative_ratios(config),
            "holdout_reference_ratio": int(
                max(configured_holdout_negative_ratios(config))
            ),
            # Backward-compatible random_* aliases.
            "random_holdout_negative_to_positive_ratios": configured_holdout_negative_ratios(config),
            "random_holdout_reference_ratio": int(
                max(configured_holdout_negative_ratios(config))
            ),
            "holdout_random_state": int(config.HOLDOUT_RANDOM_STATE),
            "fixed_split_asset_dir": str(config.FIXED_SPLIT_ASSET_DIR),
            "feature_set": config.FEATURE_SET,
            "evaluation_target_mode": config.EVALUATION_TARGET_MODE,
            "evaluation_claim_horizon_days": list(config.EVALUATION_CLAIM_HORIZON_DAYS),
            "positive_claim_selection_mode": config.POSITIVE_CLAIM_SELECTION_MODE,
            "split": {
                "strategy": (
                    "fixed deterministic random machine-level split; stratified by "
                    "full_model and eligible claim-history status"
                ),
                "assignment_path": str(assignment_path),
                "train_ratio": float(config.TRAIN_RATIO),
                "validation_ratio": float(config.VALIDATION_RATIO),
                "test_ratio": float(config.TEST_RATIO),
                "leakage_rule": "each physical machine appears in exactly one split",
                "holdout_rule": (
                    "random: one independent no-claim window per negative machine; "
                    "controlled: same full model and exact positive calendar window; "
                    "both designs use unique machines and nested negative-ratio subsets"
                ),
            },
        },
        step_dir / "run_summary.json",
    )
    print(f"02_build_case_control_dataset completed. Outputs: {step_dir}", flush=True)


if __name__ == "__main__":
    run()
