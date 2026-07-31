"""Step 13: positive-only score distributions and threshold diagnostics.

This optional evaluation scores every positive machine snapshot present in the fixed
validation/test evaluation cohorts.  A row is positive separately for each configured
future-claim horizon, so the same machine can enter the 60-, 90-, or 180-day positive
set even when it is negative at 30 days.

The step reports:

1. machine-level positive predictions with actual next-claim timing;
2. positive-score distributions and threshold capture rates; and
3. optional positive-versus-negative threshold diagnostics using the fixed negative
   cohort created by Step 12.

The raw classifier output is treated as a risk score.  Candidate high/critical
thresholds are selected on validation data by limiting the empirical false-positive
rate among negative reference machines.  They are not presented as calibrated failure
probabilities.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from cc_utils import (
    annotate_future_claim_outcomes,
    configured_evaluation_horizons,
    ensure_dir,
    fit_model_pipeline,
    make_model_pipeline,
    predict_score,
    validate_dataset_features,
    write_json,
)

DATE_COLUMNS = [
    "window_start",
    "window_end",
    "claim_date",
    "future_claim_date",
    "next_claim_date_on_or_after_window_end",
    "as_of_actual_next_claim_date",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    for column in DATE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _existing_path(value) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    return path if path.exists() else None


def _configured_splits() -> list[str]:
    raw = getattr(config, "POSITIVE_ONLY_EVALUATION_SPLITS", ["validation", "test"])
    if isinstance(raw, str):
        raw = [raw]
    result: list[str] = []
    for value in raw:
        split_name = str(value).strip().lower()
        if split_name not in {"validation", "test"}:
            raise ValueError(
                "POSITIVE_ONLY_EVALUATION_SPLITS may contain only 'validation' and 'test'."
            )
        if split_name not in result:
            result.append(split_name)
    if not result:
        raise ValueError("POSITIVE_ONLY_EVALUATION_SPLITS cannot be empty.")
    return result


def _configured_horizons() -> list[int]:
    raw = getattr(config, "POSITIVE_ONLY_EVALUATION_HORIZONS", None)
    if raw in (None, [], (), ""):
        horizons = configured_evaluation_horizons(config)
    elif isinstance(raw, (int, float, np.integer, np.floating, str)):
        horizons = [int(raw)]
    else:
        horizons = [int(value) for value in raw]
    result = sorted({int(value) for value in horizons if int(value) >= 0})
    if not result:
        raise ValueError("POSITIVE_ONLY_EVALUATION_HORIZONS cannot be empty.")
    return result


def _review_thresholds() -> list[float]:
    raw = getattr(
        config,
        "POSITIVE_ONLY_SCORE_REVIEW_THRESHOLDS",
        [0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    )
    values = sorted({float(value) for value in raw})
    if not values or any(value < 0 or value > 1 for value in values):
        raise ValueError(
            "POSITIVE_ONLY_SCORE_REVIEW_THRESHOLDS must contain values in [0, 1]."
        )
    return values


def _target_false_positive_rates() -> list[float]:
    raw = getattr(
        config,
        "POSITIVE_ONLY_THRESHOLD_FALSE_POSITIVE_TARGETS",
        [0.10, 0.05, 0.01],
    )
    values = sorted({float(value) for value in raw}, reverse=True)
    if not values or any(value <= 0 or value >= 1 for value in values):
        raise ValueError(
            "POSITIVE_ONLY_THRESHOLD_FALSE_POSITIVE_TARGETS must be between 0 and 1."
        )
    return values


def _assumed_prevalences() -> list[float]:
    raw = getattr(
        config,
        "POSITIVE_ONLY_ASSUMED_DEPLOYMENT_PREVALENCES",
        [0.01, 0.05, 0.10, 0.20],
    )
    values = sorted({float(value) for value in raw})
    if not values or any(value <= 0 or value >= 1 for value in values):
        raise ValueError(
            "POSITIVE_ONLY_ASSUMED_DEPLOYMENT_PREVALENCES must be between 0 and 1."
        )
    return values


def _load_dataset_index() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}. Run Steps 01 and 02 first.")
    return pd.read_csv(path, low_memory=False)


def _load_episodes() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "01_claim_episodes" / "claim_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Claim episodes not found: {path}. Run Step 01 first.")
    return pd.read_csv(path, parse_dates=["claim_date", "episode_end_date"], low_memory=False)


def _evaluation_path(dataset_row: pd.Series, split_name: str) -> Path:
    mode = str(getattr(config, "EVALUATION_TARGET_MODE", "training_target")).strip().lower()
    if mode == "claim_within_horizon":
        horizon_path = _existing_path(
            dataset_row.get(f"{split_name}_horizon_evaluation_dataset_path")
        )
        if horizon_path is not None:
            return horizon_path
    standard_path = _existing_path(dataset_row.get(f"{split_name}_dataset_path"))
    if standard_path is None:
        raise FileNotFoundError(
            f"No usable {split_name} dataset path was found for dataset "
            f"{dataset_row.get('dataset_id')}."
        )
    return standard_path


def _first_positive_horizon(days_to_claim: pd.Series, horizons: Sequence[int]) -> pd.Series:
    days = pd.to_numeric(days_to_claim, errors="coerce")
    result = pd.Series(pd.NA, index=days.index, dtype="Int64")
    include_cutoff = bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True))
    eligible = days.ge(0) if include_cutoff else days.gt(0)
    for horizon in sorted(int(value) for value in horizons):
        mask = result.isna() & eligible & days.le(int(horizon))
        result.loc[mask] = int(horizon)
    return result


def _score_distribution_summary(positives: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["dataset_id", "algorithm", "split", "evaluation_horizon_days"]
    for keys, group in positives.groupby(group_columns, dropna=False, sort=True):
        scores = pd.to_numeric(group["risk_score"], errors="coerce").dropna()
        row = dict(zip(group_columns, keys))
        row.update({
            "positive_rows": int(len(group)),
            "unique_positive_machines": int(group["machine_key"].astype(str).nunique()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "minimum_score": float(scores.min()),
            "p01_score": float(scores.quantile(0.01)),
            "p05_score": float(scores.quantile(0.05)),
            "p10_score": float(scores.quantile(0.10)),
            "p25_score": float(scores.quantile(0.25)),
            "median_score": float(scores.median()),
            "p75_score": float(scores.quantile(0.75)),
            "p90_score": float(scores.quantile(0.90)),
            "p95_score": float(scores.quantile(0.95)),
            "p99_score": float(scores.quantile(0.99)),
            "maximum_score": float(scores.max()),
            "captured_at_configured_threshold": int((scores >= threshold).sum()),
            "positive_capture_rate_at_configured_threshold": float((scores >= threshold).mean()),
            "median_days_to_claim": float(
                pd.to_numeric(group["days_to_next_claim_on_or_after_window_end"], errors="coerce").median()
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_capture_summary(positives: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_columns = ["dataset_id", "algorithm", "split", "evaluation_horizon_days"]
    for keys, group in positives.groupby(group_columns, dropna=False, sort=True):
        scores = pd.to_numeric(group["risk_score"], errors="coerce")
        days = pd.to_numeric(
            group["days_to_next_claim_on_or_after_window_end"], errors="coerce"
        )
        for threshold in _review_thresholds():
            selected = scores.ge(threshold)
            row = dict(zip(group_columns, keys))
            row.update({
                "score_threshold": float(threshold),
                "positive_rows": int(len(group)),
                "captured_positive_rows": int(selected.sum()),
                "missed_positive_rows": int((~selected).sum()),
                "positive_capture_rate": float(selected.mean()),
                "mean_days_to_claim_captured": float(days[selected].mean()) if selected.any() else np.nan,
                "median_days_to_claim_captured": float(days[selected].median()) if selected.any() else np.nan,
                "mean_days_to_claim_missed": float(days[~selected].mean()) if (~selected).any() else np.nan,
                "median_days_to_claim_missed": float(days[~selected].median()) if (~selected).any() else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _histogram_table(positives: pd.DataFrame) -> pd.DataFrame:
    bins = max(2, int(getattr(config, "POSITIVE_ONLY_SCORE_HISTOGRAM_BINS", 20)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    group_columns = ["dataset_id", "algorithm", "split", "evaluation_horizon_days"]
    for keys, group in positives.groupby(group_columns, dropna=False, sort=True):
        scores = pd.to_numeric(group["risk_score"], errors="coerce").dropna().to_numpy()
        counts, _ = np.histogram(scores, bins=edges)
        for index, count in enumerate(counts):
            row = dict(zip(group_columns, keys))
            row.update({
                "score_bin_lower": float(edges[index]),
                "score_bin_upper": float(edges[index + 1]),
                "positive_rows": int(count),
                "fraction_of_positive_rows": float(count / len(scores)) if len(scores) else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _threshold_for_max_fpr(negative_scores: pd.Series, target_fpr: float) -> float:
    scores = pd.to_numeric(negative_scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(scores) == 0:
        return np.nan
    candidates = np.concatenate(
        [np.unique(scores), [np.nextafter(float(np.max(scores)), np.inf)]]
    )
    candidates.sort()
    valid: list[float] = []
    for threshold in candidates:
        fpr = float(np.mean(scores >= threshold))
        if fpr <= float(target_fpr) + 1e-12:
            valid.append(float(threshold))
    return min(valid) if valid else float(np.nextafter(float(np.max(scores)), np.inf))


def _ppv_from_rates(tpr: float, fpr: float, prevalence: float) -> float:
    denominator = prevalence * tpr + (1.0 - prevalence) * fpr
    return float(prevalence * tpr / denominator) if denominator > 0 else np.nan


def _threshold_separation_rows(
    positive_group: pd.DataFrame,
    negative_group: pd.DataFrame,
    dataset_id: str,
    algorithm: str,
    split_name: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_scores = pd.to_numeric(positive_group["risk_score"], errors="coerce").dropna()
    negative_scores_all = pd.to_numeric(negative_group["risk_score"], errors="coerce").dropna()
    references: list[tuple[str, pd.Series]] = [("all_current_negatives", negative_scores_all)]
    if "negative_case_type" in negative_group.columns:
        strict_mask = negative_group["negative_case_type"].astype(str).eq("strict_negative")
        strict_scores = pd.to_numeric(
            negative_group.loc[strict_mask, "risk_score"], errors="coerce"
        ).dropna()
        if len(strict_scores):
            references.append(("strict_negatives", strict_scores))

    fixed_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for reference_name, negative_scores in references:
        for threshold in _review_thresholds():
            tpr = float((positive_scores >= threshold).mean()) if len(positive_scores) else np.nan
            fpr = float((negative_scores >= threshold).mean()) if len(negative_scores) else np.nan
            selected_positive = int((positive_scores >= threshold).sum())
            selected_negative = int((negative_scores >= threshold).sum())
            denominator = selected_positive + selected_negative
            row = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "evaluation_horizon_days": int(horizon),
                "negative_reference": reference_name,
                "threshold_source": "configured_review_threshold",
                "score_threshold": float(threshold),
                "positive_rows": int(len(positive_scores)),
                "negative_rows": int(len(negative_scores)),
                "captured_positive_rows": selected_positive,
                "false_positive_rows": selected_negative,
                "positive_capture_rate_tpr": tpr,
                "negative_false_positive_rate_fpr": fpr,
                "diagnostic_sample_precision": float(selected_positive / denominator) if denominator else np.nan,
                "positive_to_negative_selection_ratio": float(selected_positive / selected_negative) if selected_negative else np.inf if selected_positive else np.nan,
                "likelihood_ratio_tpr_over_fpr": float(tpr / fpr) if fpr > 0 else np.inf if tpr > 0 else np.nan,
            }
            for prevalence in _assumed_prevalences():
                label = str(prevalence).replace(".", "p")
                row[f"estimated_precision_at_prevalence_{label}"] = _ppv_from_rates(
                    tpr, fpr, prevalence
                )
            fixed_rows.append(row)

        for target_fpr in _target_false_positive_rates():
            threshold = _threshold_for_max_fpr(negative_scores, target_fpr)
            tpr = float((positive_scores >= threshold).mean()) if len(positive_scores) else np.nan
            fpr = float((negative_scores >= threshold).mean()) if len(negative_scores) else np.nan
            selected_positive = int((positive_scores >= threshold).sum())
            selected_negative = int((negative_scores >= threshold).sum())
            denominator = selected_positive + selected_negative
            risk_tier = (
                "critical_candidate" if target_fpr <= 0.01
                else "high_candidate" if target_fpr <= 0.05
                else "review_candidate"
            )
            row = {
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "split": split_name,
                "evaluation_horizon_days": int(horizon),
                "negative_reference": reference_name,
                "risk_tier_candidate": risk_tier,
                "target_max_negative_fpr": float(target_fpr),
                "score_threshold": float(threshold),
                "positive_rows": int(len(positive_scores)),
                "negative_rows": int(len(negative_scores)),
                "captured_positive_rows": selected_positive,
                "false_positive_rows": selected_negative,
                "positive_capture_rate_tpr": tpr,
                "actual_negative_fpr": fpr,
                "diagnostic_sample_precision": float(selected_positive / denominator) if denominator else np.nan,
                "likelihood_ratio_tpr_over_fpr": float(tpr / fpr) if fpr > 0 else np.inf if tpr > 0 else np.nan,
            }
            for prevalence in _assumed_prevalences():
                label = str(prevalence).replace(".", "p")
                row[f"estimated_precision_at_prevalence_{label}"] = _ppv_from_rates(
                    tpr, fpr, prevalence
                )
            candidate_rows.append(row)
    return pd.DataFrame(fixed_rows), pd.DataFrame(candidate_rows)


def _validation_selected_threshold_rows(
    scored_by_split: dict[str, pd.DataFrame],
    negative_reference: pd.DataFrame,
    dataset_id: str,
    algorithm: str,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Select thresholds on validation negatives and apply unchanged to each split."""
    if "validation" not in scored_by_split or negative_reference is None or negative_reference.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    validation_eval = scored_by_split["validation"]
    validation_negative_base = negative_reference[
        negative_reference["split"].astype(str).eq("validation")
    ].copy()
    if validation_negative_base.empty:
        return pd.DataFrame()

    for horizon in horizons:
        target_column = f"eval_target_claim_within_next_{int(horizon)}d"
        validation_labels = pd.to_numeric(
            validation_eval[target_column], errors="coerce"
        ).fillna(0).astype(int)
        validation_positive_scores = pd.to_numeric(
            validation_eval.loc[validation_labels.eq(1), "risk_score"], errors="coerce"
        ).dropna()
        negative_label_column = f"true_label_{int(horizon)}d"
        if negative_label_column in validation_negative_base.columns:
            validation_negative_labels = pd.to_numeric(
                validation_negative_base[negative_label_column], errors="coerce"
            ).fillna(0).astype(int)
        elif target_column in validation_negative_base.columns:
            validation_negative_labels = pd.to_numeric(
                validation_negative_base[target_column], errors="coerce"
            ).fillna(0).astype(int)
        else:
            validation_negative_labels = pd.Series(
                0, index=validation_negative_base.index, dtype=int
            )
        validation_current_negatives = validation_negative_base.loc[
            validation_negative_labels.eq(0)
        ].copy()
        references: list[tuple[str, pd.DataFrame]] = [
            ("all_current_negatives", validation_current_negatives)
        ]
        if "negative_case_type" in validation_current_negatives.columns:
            strict = validation_current_negatives[
                validation_current_negatives["negative_case_type"]
                .astype(str)
                .eq("strict_negative")
            ].copy()
            if not strict.empty:
                references.append(("strict_negatives", strict))

        for reference_name, validation_negative_group in references:
            validation_negative_scores = pd.to_numeric(
                validation_negative_group["risk_score"], errors="coerce"
            ).dropna()
            for target_fpr in _target_false_positive_rates():
                threshold = _threshold_for_max_fpr(
                    validation_negative_scores, target_fpr
                )
                risk_tier = (
                    "critical_candidate" if target_fpr <= 0.01
                    else "high_candidate" if target_fpr <= 0.05
                    else "review_candidate"
                )
                for evaluation_split, evaluation_df in scored_by_split.items():
                    labels = pd.to_numeric(
                        evaluation_df[target_column], errors="coerce"
                    ).fillna(0).astype(int)
                    positive_scores = pd.to_numeric(
                        evaluation_df.loc[labels.eq(1), "risk_score"], errors="coerce"
                    ).dropna()
                    negative_base = negative_reference[
                        negative_reference["split"].astype(str).eq(evaluation_split)
                    ].copy()
                    if negative_base.empty:
                        continue
                    if negative_label_column in negative_base.columns:
                        negative_labels = pd.to_numeric(
                            negative_base[negative_label_column], errors="coerce"
                        ).fillna(0).astype(int)
                    elif target_column in negative_base.columns:
                        negative_labels = pd.to_numeric(
                            negative_base[target_column], errors="coerce"
                        ).fillna(0).astype(int)
                    else:
                        negative_labels = pd.Series(0, index=negative_base.index, dtype=int)
                    negative_group = negative_base.loc[negative_labels.eq(0)].copy()
                    if reference_name == "strict_negatives":
                        negative_group = negative_group[
                            negative_group["negative_case_type"]
                            .astype(str)
                            .eq("strict_negative")
                        ].copy()
                    negative_scores = pd.to_numeric(
                        negative_group["risk_score"], errors="coerce"
                    ).dropna()
                    if len(positive_scores) == 0 or len(negative_scores) == 0:
                        continue
                    tpr = float((positive_scores >= threshold).mean())
                    fpr = float((negative_scores >= threshold).mean())
                    selected_positive = int((positive_scores >= threshold).sum())
                    selected_negative = int((negative_scores >= threshold).sum())
                    denominator = selected_positive + selected_negative
                    row = {
                        "dataset_id": dataset_id,
                        "algorithm": algorithm,
                        "threshold_selection_split": "validation",
                        "evaluation_split": evaluation_split,
                        "evaluation_horizon_days": int(horizon),
                        "negative_reference": reference_name,
                        "risk_tier_candidate": risk_tier,
                        "target_max_validation_negative_fpr": float(target_fpr),
                        "validation_selected_score_threshold": float(threshold),
                        "validation_positive_rows": int(len(validation_positive_scores)),
                        "validation_negative_rows_used_for_threshold": int(len(validation_negative_scores)),
                        "evaluation_positive_rows": int(len(positive_scores)),
                        "evaluation_negative_rows": int(len(negative_scores)),
                        "captured_positive_rows": selected_positive,
                        "false_positive_rows": selected_negative,
                        "positive_capture_rate_tpr": tpr,
                        "negative_false_positive_rate_fpr": fpr,
                        "diagnostic_sample_precision": float(selected_positive / denominator) if denominator else np.nan,
                        "likelihood_ratio_tpr_over_fpr": float(tpr / fpr) if fpr > 0 else np.inf if tpr > 0 else np.nan,
                    }
                    for prevalence in _assumed_prevalences():
                        label = str(prevalence).replace(".", "p")
                        row[f"estimated_precision_at_prevalence_{label}"] = _ppv_from_rates(
                            tpr, fpr, prevalence
                        )
                    rows.append(row)
    return pd.DataFrame(rows)



def _load_negative_reference(
    dataset_id: str,
    algorithm: str,
    model,
    feature_columns: Sequence[str],
    episodes: pd.DataFrame,
    horizons: Sequence[int],
) -> Optional[pd.DataFrame]:
    override = getattr(config, "POSITIVE_ONLY_NEGATIVE_DIAGNOSTICS_DIR", None)
    negative_dir = (
        Path(override)
        if override not in (None, "")
        else config.OUTPUT_DIR / "12_negative_only_score_diagnostics"
    )
    feature_path = negative_dir / f"{dataset_id}__negative_only_feature_dataset.csv"
    if feature_path.exists():
        frame = _read_csv(feature_path)
        validate_dataset_features(frame, config)
        frame = annotate_future_claim_outcomes(
            frame,
            claim_history_episodes=episodes,
            config=config,
            horizons=horizons,
        )
        frame["risk_score"] = np.asarray(
            predict_score(model, frame[list(feature_columns)], algorithm), dtype=float
        )
        frame["dataset_id"] = dataset_id
        frame["algorithm"] = algorithm
        return frame

    prediction_path = negative_dir / (
        f"{dataset_id}__{algorithm}__negative_only_machine_predictions.csv"
    )
    if prediction_path.exists():
        return _read_csv(prediction_path)
    all_path = negative_dir / "negative_only_machine_predictions_all.csv"
    if all_path.exists():
        frame = _read_csv(all_path)
        return frame[
            frame["dataset_id"].astype(str).eq(dataset_id)
            & frame["algorithm"].astype(str).eq(algorithm)
        ].copy()
    return None


def _save_distribution_plot(
    positives: pd.DataFrame,
    negatives: Optional[pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    bins = max(2, int(getattr(config, "POSITIVE_ONLY_SCORE_HISTOGRAM_BINS", 20)))
    plt.figure(figsize=(9, 5.5))
    if negatives is not None and not negatives.empty:
        plt.hist(
            pd.to_numeric(negatives["risk_score"], errors="coerce").dropna(),
            bins=bins,
            range=(0, 1),
            alpha=0.45,
            density=True,
            label="negative reference",
        )
    plt.hist(
        pd.to_numeric(positives["risk_score"], errors="coerce").dropna(),
        bins=bins,
        range=(0, 1),
        alpha=0.55,
        density=True,
        label="future-claim positives",
    )
    plt.xlabel("Model risk score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run(
    dataset_index_path: str | Path | None = None,
    step_dir: str | Path | None = None,
) -> None:
    config.refresh_derived_config()
    if not bool(getattr(config, "POSITIVE_ONLY_EVALUATION_ENABLED", True)):
        print("13_positive_only_score_diagnostics skipped: disabled in config.", flush=True)
        return

    output_dir = (
        Path(step_dir)
        if step_dir is not None
        else config.OUTPUT_DIR / "13_positive_only_score_diagnostics"
    )
    ensure_dir(output_dir)
    dataset_index = (
        pd.read_csv(Path(dataset_index_path), low_memory=False)
        if dataset_index_path is not None
        else _load_dataset_index()
    )
    episodes = _load_episodes()
    horizons = _configured_horizons()
    splits = _configured_splits()
    configured_threshold = float(
        getattr(config, "POSITIVE_ONLY_SCORE_THRESHOLD", 0.50)
    )

    all_positive_predictions: list[pd.DataFrame] = []
    all_distributions: list[pd.DataFrame] = []
    all_capture_tables: list[pd.DataFrame] = []
    all_histograms: list[pd.DataFrame] = []
    all_threshold_separation: list[pd.DataFrame] = []
    all_threshold_candidates: list[pd.DataFrame] = []
    all_validation_selected_thresholds: list[pd.DataFrame] = []
    run_summaries: list[dict] = []

    for _, dataset_row in dataset_index.iterrows():
        dataset_id = str(dataset_row["dataset_id"])
        train_df = _read_csv(dataset_row["training_dataset_path"])
        fit_eval_path = _existing_path(dataset_row.get("validation_dataset_path"))
        if fit_eval_path is None:
            raise FileNotFoundError(
                f"Matched validation dataset missing for model fitting: {dataset_id}"
            )
        fit_eval_df = _read_csv(fit_eval_path)
        validate_dataset_features(train_df, config)
        validate_dataset_features(fit_eval_df, config)
        feature_columns = list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)
        X_train = train_df[feature_columns]
        y_train = pd.to_numeric(train_df["target"], errors="raise").astype(int)
        X_fit_eval = fit_eval_df[feature_columns]
        y_fit_eval = pd.to_numeric(fit_eval_df["target"], errors="raise").astype(int)

        for algorithm in config.MODELS_TO_RUN:
            print(
                f"Positive-only diagnostics dataset={dataset_id} algorithm={algorithm}",
                flush=True,
            )
            model = make_model_pipeline(algorithm, config)
            if model is None:
                continue
            fit_metadata = fit_model_pipeline(
                model,
                algorithm,
                X_train,
                y_train,
                config,
                X_eval=X_fit_eval,
                y_eval=y_fit_eval,
                eval_name="standard_validation_for_positive_only_fit",
            )
            negative_reference = _load_negative_reference(
                dataset_id,
                algorithm,
                model,
                feature_columns,
                episodes,
                horizons,
            )
            scored_evaluation_by_split: dict[str, pd.DataFrame] = {}

            for split_name in splits:
                evaluation_df = _read_csv(_evaluation_path(dataset_row, split_name))
                validate_dataset_features(evaluation_df, config)
                evaluation_df = annotate_future_claim_outcomes(
                    evaluation_df,
                    claim_history_episodes=episodes,
                    config=config,
                    horizons=horizons,
                )
                evaluation_df["dataset_id"] = dataset_id
                evaluation_df["algorithm"] = algorithm
                evaluation_df["split"] = split_name
                evaluation_df["risk_score"] = np.asarray(
                    predict_score(model, evaluation_df[feature_columns], algorithm), dtype=float
                )
                evaluation_df["predicted_label_at_configured_threshold"] = (
                    evaluation_df["risk_score"] >= configured_threshold
                ).astype(int)
                evaluation_df["score_rank_within_split"] = (
                    evaluation_df["risk_score"]
                    .rank(method="first", ascending=False)
                    .astype(int)
                )
                evaluation_df["score_percentile_within_split"] = (
                    evaluation_df["risk_score"].rank(method="average", pct=True)
                )
                evaluation_df["first_configured_positive_horizon_days"] = (
                    _first_positive_horizon(
                        evaluation_df["days_to_next_claim_on_or_after_window_end"],
                        horizons,
                    )
                )
                scored_evaluation_by_split[split_name] = evaluation_df.copy()

                split_negative = None
                if negative_reference is not None and not negative_reference.empty:
                    split_negative = negative_reference[
                        negative_reference["split"].astype(str).eq(split_name)
                    ].copy()

                split_positive_rows = 0
                for horizon in horizons:
                    target_column = f"eval_target_claim_within_next_{int(horizon)}d"
                    labels = pd.to_numeric(
                        evaluation_df[target_column], errors="coerce"
                    ).fillna(0).astype(int)
                    positives = evaluation_df.loc[labels.eq(1)].copy()
                    if positives.empty:
                        continue
                    positives["evaluation_horizon_days"] = int(horizon)
                    positives["true_label"] = 1
                    positives["predicted_label"] = positives[
                        "predicted_label_at_configured_threshold"
                    ].astype(int)
                    positives["prediction_correct"] = positives["predicted_label"].eq(1).astype(int)
                    positives["evaluation_population_rows"] = int(len(evaluation_df))
                    positives["evaluation_positive_rows"] = int(labels.sum())
                    positives["evaluation_positive_rate"] = float(labels.mean())
                    split_positive_rows += len(positives)

                    file_path = output_dir / (
                        f"{dataset_id}__{algorithm}__{split_name}__horizon_{int(horizon)}d"
                        "__positive_machine_predictions.csv"
                    )
                    positives.to_csv(file_path, index=False)
                    all_positive_predictions.append(positives)
                    all_distributions.append(
                        _score_distribution_summary(positives, configured_threshold)
                    )
                    all_capture_tables.append(_threshold_capture_summary(positives))
                    all_histograms.append(_histogram_table(positives))

                    negative_for_horizon = None
                    if split_negative is not None and not split_negative.empty:
                        negative_label_column = f"true_label_{int(horizon)}d"
                        if negative_label_column in split_negative.columns:
                            neg_labels = pd.to_numeric(
                                split_negative[negative_label_column], errors="coerce"
                            ).fillna(0).astype(int)
                        elif target_column in split_negative.columns:
                            neg_labels = pd.to_numeric(
                                split_negative[target_column], errors="coerce"
                            ).fillna(0).astype(int)
                        else:
                            neg_labels = pd.Series(0, index=split_negative.index, dtype=int)
                        negative_for_horizon = split_negative.loc[neg_labels.eq(0)].copy()
                        if not negative_for_horizon.empty:
                            fixed, candidates = _threshold_separation_rows(
                                positives,
                                negative_for_horizon,
                                dataset_id,
                                algorithm,
                                split_name,
                                int(horizon),
                            )
                            all_threshold_separation.append(fixed)
                            all_threshold_candidates.append(candidates)

                    _save_distribution_plot(
                        positives,
                        negative_for_horizon,
                        output_dir / (
                            f"{dataset_id}__{algorithm}__{split_name}__horizon_{int(horizon)}d"
                            "__positive_vs_negative_score_distribution.png"
                        ),
                        title=(
                            f"{split_name.title()} score distribution: claims within "
                            f"{int(horizon)} days"
                        ),
                    )

                run_summaries.append({
                    "dataset_id": dataset_id,
                    "algorithm": algorithm,
                    "split": split_name,
                    "evaluation_rows": int(len(evaluation_df)),
                    "positive_prediction_rows_across_horizons": int(split_positive_rows),
                    "configured_threshold": configured_threshold,
                    "negative_reference_available": bool(
                        split_negative is not None and not split_negative.empty
                    ),
                    "fit_metadata": fit_metadata,
                })

            if negative_reference is not None and not negative_reference.empty:
                validation_selected = _validation_selected_threshold_rows(
                    scored_evaluation_by_split,
                    negative_reference,
                    dataset_id,
                    algorithm,
                    horizons,
                )
                if not validation_selected.empty:
                    all_validation_selected_thresholds.append(validation_selected)

    def concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        kept = [frame for frame in frames if frame is not None and not frame.empty]
        return pd.concat(kept, ignore_index=True, sort=False) if kept else pd.DataFrame()

    outputs = {
        "positive_only_machine_predictions_all.csv": concat(all_positive_predictions),
        "positive_only_score_distribution_summary.csv": concat(all_distributions),
        "positive_only_threshold_capture_summary.csv": concat(all_capture_tables),
        "positive_only_score_histogram.csv": concat(all_histograms),
        "positive_negative_threshold_separation_summary.csv": concat(all_threshold_separation),
        "positive_only_threshold_candidates.csv": concat(all_threshold_candidates),
        "positive_only_validation_selected_thresholds_applied.csv": concat(
            all_validation_selected_thresholds
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    write_json(
        {
            "step": "13_positive_only_score_diagnostics",
            "output_dir": str(output_dir),
            "evaluation_splits": splits,
            "evaluation_horizons": horizons,
            "configured_score_threshold": configured_threshold,
            "review_thresholds": _review_thresholds(),
            "threshold_false_positive_targets": _target_false_positive_rates(),
            "assumed_deployment_prevalences": _assumed_prevalences(),
            "run_summaries": run_summaries,
            "notes": [
                "Positive-only distributions measure positive capture, not precision by themselves.",
                "Threshold candidate rows use validation/test negative references from Step 12 when available.",
                "High and critical candidates are based on empirical negative false-positive-rate limits.",
                "Raw classifier scores are risk scores and are not calibrated deployment probabilities.",
                "Use validation to choose thresholds; reserve test results for final confirmation.",
            ],
        },
        output_dir / "run_summary.json",
    )
    print(
        f"13_positive_only_score_diagnostics completed. Outputs: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    run()
