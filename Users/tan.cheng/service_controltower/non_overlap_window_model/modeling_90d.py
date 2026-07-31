"""Shared 90-day model fitting, calibration, ranking, and evaluation utilities.

All controls are read from config.py. This module has no command-line interface.
"""
from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    fbeta_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config as cfg
from reduced_features import REDUCED_FEATURES

HISTORY_FEATURES = (
    "prior_claim_count_30d",
    "prior_claim_count_90d",
    "prior_claim_count_180d",
    "prior_claim_count_365d",
    "prior_claim_count_ever",
    "days_since_prior_claim",
)

def safe_metric(function, y: np.ndarray, score: np.ndarray) -> float:
    return float(function(y, score)) if np.unique(y).size == 2 else float("nan")

def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)

def fit_platt(raw_score: np.ndarray, y: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, max_iter=500, random_state=cfg.RANDOM_SEED)
    calibrator.fit(probability_to_logit(raw_score), y)
    return calibrator

def apply_platt(calibrator: LogisticRegression, raw_score: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(probability_to_logit(raw_score))[:, 1]

def choose_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    table = pd.DataFrame(
        {"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1}
    ).sort_values(["f1", "precision", "recall", "threshold"], ascending=False)
    return float(table.iloc[0]["threshold"])

def threshold_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "flagged_count": int(prediction.sum()),
        "flagged_rate": float(prediction.mean()),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def annotate_target_outcomes(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add prior target history and the next-90-day outcome to snapshot rows."""

    out = frame.copy().reset_index(drop=True)
    out["period_end"] = pd.to_datetime(out["period_end"], errors="raise").dt.normalize()
    event_map = {
        key: np.sort(group["event_date"].values.astype("datetime64[D]"))
        for key, group in events.groupby("machine_key", sort=False)
    }
    result = {
        "prior_claim_count_30d": np.zeros(len(out), dtype=np.int32),
        "prior_claim_count_90d": np.zeros(len(out), dtype=np.int32),
        "prior_claim_count_180d": np.zeros(len(out), dtype=np.int32),
        "prior_claim_count_365d": np.zeros(len(out), dtype=np.int32),
        "prior_claim_count_ever": np.zeros(len(out), dtype=np.int32),
        "days_since_prior_claim": np.full(len(out), np.nan),
        "target_90d": np.zeros(len(out), dtype=np.int8),
        "target_claim_count_90d": np.zeros(len(out), dtype=np.int32),
        "target_first_claim_date_90d": np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]"),
        "next_target_event_date": np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]"),
        "days_to_next_target_event": np.full(len(out), np.nan),
    }
    for machine_key, index in out.groupby("machine_key", sort=False).groups.items():
        loc = np.asarray(list(index), dtype=int)
        as_of = out.loc[loc, "period_end"].values.astype("datetime64[D]")
        dates = event_map.get(machine_key, np.array([], dtype="datetime64[D]"))
        if len(dates) == 0:
            continue
        right = np.searchsorted(dates, as_of, side="right")
        result["prior_claim_count_ever"][loc] = right
        for days in (30, 90, 180, 365):
            left = np.searchsorted(dates, as_of - np.timedelta64(days, "D"), side="right")
            result[f"prior_claim_count_{days}d"][loc] = right - left
        has_prior = right > 0
        if has_prior.any():
            previous = dates[np.maximum(right - 1, 0)]
            delta = (as_of - previous) / np.timedelta64(1, "D")
            result["days_since_prior_claim"][loc[has_prior]] = delta[has_prior]

        target_end = as_of + np.timedelta64(int(cfg.HORIZON_DAYS), "D")
        future_right = np.searchsorted(dates, target_end, side="right")
        count = future_right - right
        result["target_claim_count_90d"][loc] = count
        result["target_90d"][loc] = (count > 0).astype(np.int8)
        has_future = right < len(dates)
        if has_future.any():
            next_dates = dates[np.minimum(right, len(dates) - 1)]
            next_idx = loc[has_future]
            result["next_target_event_date"][next_idx] = next_dates[has_future].astype("datetime64[ns]")
            score_date = as_of + np.timedelta64(1, "D")
            result["days_to_next_target_event"][next_idx] = (
                (next_dates[has_future] - score_date[has_future]) / np.timedelta64(1, "D")
            )
        in_horizon = count > 0
        if in_horizon.any():
            first_dates = dates[right[in_horizon]]
            result["target_first_claim_date_90d"][loc[in_horizon]] = first_dates.astype("datetime64[ns]")

    for column, values in result.items():
        out[column] = values
    observation_end = pd.Timestamp(events["event_date"].max()).normalize()
    out["target_observation_end"] = observation_end
    out["target_window_start"] = out["period_end"] + pd.Timedelta(days=1)
    out["target_window_end"] = out["period_end"] + pd.Timedelta(days=int(cfg.HORIZON_DAYS))
    out["label_complete_90d"] = out["target_window_end"].le(observation_end).astype("int8")
    return out

def feature_list(frame: pd.DataFrame, variant: str) -> list[str]:
    missing = [feature for feature in REDUCED_FEATURES if feature not in frame.columns]
    if missing:
        raise KeyError(f"Condition dataset is missing reviewed features: {missing[:30]}")
    features = list(REDUCED_FEATURES)
    if variant == "reviewed134_condition_only":
        features = [feature for feature in features if feature not in HISTORY_FEATURES]
    elif variant != "reviewed140_with_history":
        raise ValueError(f"Unknown model variant: {variant}")
    usable = [
        feature for feature in features
        if frame[feature].notna().any() and frame[feature].nunique(dropna=False) > 1
    ]
    return usable

def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [feature for feature in ("full_model", "segment_season") if feature in features]
    numeric = [feature for feature in features if feature not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        )
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )

def candidate_models(y: np.ndarray) -> dict[str, object]:
    positive = max(int(y.sum()), 1)
    ratio = float((len(y) - positive) / positive)
    weight = max(1.0, math.sqrt(ratio))
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                C=0.20,
                class_weight="balanced",
                random_state=cfg.RANDOM_SEED,
            ),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=35,
            l2_regularization=7.0,
            random_state=cfg.RANDOM_SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.04,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=8.0,
            reg_alpha=0.20,
            scale_pos_weight=weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_SEED,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=220,
            num_leaves=23,
            learning_rate=0.035,
            min_child_samples=45,
            colsample_bytree=0.82,
            reg_lambda=8.0,
            reg_alpha=0.20,
            scale_pos_weight=weight,
            verbosity=-1,
            n_jobs=cfg.N_JOBS,
            random_state=cfg.RANDOM_SEED,
        ),
    }

def define_native_split(period_table: pd.DataFrame, common_observation_end: pd.Timestamp) -> dict[str, list[pd.Timestamp]]:
    periods = (
        period_table[["period_start", "period_end"]]
        .drop_duplicates()
        .sort_values("period_start")
        .reset_index(drop=True)
    )
    periods["target_window_end"] = periods["period_end"] + pd.Timedelta(days=int(cfg.HORIZON_DAYS))
    complete = periods[periods["target_window_end"].le(common_observation_end)].copy()
    if len(complete) < cfg.NATIVE_LOCKED_TEST_PERIODS + cfg.NATIVE_VALIDATION_PERIODS + 4:
        raise ValueError("Not enough complete 90-day periods for native split design.")
    locked = complete.tail(int(cfg.NATIVE_LOCKED_TEST_PERIODS))
    locked_start = pd.Timestamp(locked["period_start"].min())
    validation_candidates = complete[complete["target_window_end"].lt(locked_start)]
    validation = validation_candidates.tail(int(cfg.NATIVE_VALIDATION_PERIODS))
    validation_start = pd.Timestamp(validation["period_start"].min())
    training = complete[complete["target_window_end"].lt(validation_start)]
    return {
        "training": training["period_start"].tolist(),
        "validation": validation["period_start"].tolist(),
        "locked_test": locked["period_start"].tolist(),
    }

def select_periods(frame: pd.DataFrame, starts: list[pd.Timestamp]) -> pd.DataFrame:
    return frame[frame["period_start"].isin(pd.to_datetime(starts))].copy()

def evaluate_scores(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, threshold: float) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": safe_metric(roc_auc_score, y, raw),
        "average_precision": safe_metric(average_precision_score, y, raw),
        "brier_score": float(brier_score_loss(y, calibrated)),
        "log_loss": float(log_loss(y, calibrated, labels=[0, 1])),
    }
    metrics.update(threshold_metrics(y, calibrated, threshold))
    return metrics

def selection_metrics_for_group(
    ranked: pd.DataFrame,
    *,
    selection_type: str,
    selection_value: float | int,
) -> dict[str, float | int]:
    n = len(ranked)
    if selection_type == "top_k_fraction":
        count = max(1, int(math.ceil(n * float(selection_value))))
    elif selection_type == "top_n":
        count = min(int(selection_value), n)
    else:
        raise ValueError(selection_type)
    selected = ranked.head(count)
    positives = int(ranked["true_label"].sum())
    tp = int(selected["true_label"].sum())
    precision = float(tp / count) if count else float("nan")
    recall = float(tp / positives) if positives else float("nan")
    prevalence = float(ranked["true_label"].mean())
    return {
        "selection_type": selection_type,
        "selection_value": selection_value,
        "selected_count": int(count),
        "true_positive_count": tp,
        "positive_machines": positives,
        "eligible_machines": int(n),
        "precision": precision,
        "recall": recall,
        "positive_rate": prevalence,
        "lift_vs_fleet": float(precision / prevalence) if prevalence > 0 else float("nan"),
    }

def rank_predictions(
    frame: pd.DataFrame,
    raw: np.ndarray,
    calibrated: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["raw_score"] = np.asarray(raw, dtype=float)
    out["calibrated_probability"] = np.asarray(calibrated, dtype=float)
    out["true_label"] = out["target_90d"].astype(int)
    out["prediction_threshold"] = float(threshold)
    out["predicted_label"] = out["calibrated_probability"].ge(threshold).astype("int8")
    out = out.sort_values(["raw_score", "machine_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    out["rank_within_anchor"] = np.arange(1, len(out) + 1)
    out["score_top_fraction"] = out["rank_within_anchor"] / len(out)
    out["risk_index"] = 100.0 * (len(out) - out["rank_within_anchor"]) / max(len(out) - 1, 1)
    # User-facing score aliases: probability scale and 0-100 ranking scale.
    out["risk_score"] = out["calibrated_probability"]
    out["risk_score_0_100"] = out["risk_index"]
    out["prediction_outcome"] = np.select(
        [
            out["true_label"].eq(1) & out["predicted_label"].eq(1),
            out["true_label"].eq(0) & out["predicted_label"].eq(1),
            out["true_label"].eq(1) & out["predicted_label"].eq(0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    for rate in cfg.TOP_K_RATES:
        label = int(round(rate * 100))
        out[f"selected_top_{label}pct"] = out["rank_within_anchor"].le(max(1, math.ceil(len(out) * rate))).astype("int8")
    for count in cfg.TOP_N_COUNTS:
        out[f"selected_top_n_{count}"] = out["rank_within_anchor"].le(min(count, len(out))).astype("int8")
    return out

def pooled_selection_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    group_cols = ["target_source", "variant", "algorithm", "evaluation_scope", "selection_type", "selection_value"]
    for keys, group in rows.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        selected = int(group["selected_count"].sum())
        tp = int(group["true_positive_count"].sum())
        positives = int(group["positive_machines"].sum())
        eligible = int(group["eligible_machines"].sum())
        precision = tp / selected if selected else np.nan
        recall = tp / positives if positives else np.nan
        prevalence = positives / eligible if eligible else np.nan
        row.update(
            {
                "anchor_or_period_count": int(len(group)),
                "total_selected": selected,
                "total_true_positive": tp,
                "total_positive": positives,
                "total_eligible": eligible,
                "micro_precision": precision,
                "micro_recall": recall,
                "natural_positive_rate": prevalence,
                "micro_lift_vs_fleet": precision / prevalence if prevalence and prevalence > 0 else np.nan,
                "minimum_group_precision": float(group["precision"].min()),
                "maximum_group_precision": float(group["precision"].max()),
                "mean_group_precision": float(group["precision"].mean()),
            }
        )
        output.append(row)
    return pd.DataFrame(output)

def fit_preprocessed_model(
    fit: pd.DataFrame,
    features: list[str],
    algorithm: str,
) -> tuple[ColumnTransformer, object]:
    y = fit["target_90d"].to_numpy(dtype=int)
    preprocessor = make_preprocessor(features)
    transformed = preprocessor.fit_transform(fit[features])
    model = candidate_models(y)[algorithm]
    model.fit(transformed, y)
    return preprocessor, model

def score_model(preprocessor: ColumnTransformer, model: object, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(preprocessor.transform(frame[features]))[:, 1]

def native_experiment(
    data: pd.DataFrame,
    *,
    target_source: str,
    split_dates: dict[str, list[pd.Timestamp]],
    output_dir: Path,
    skip_candidate_screen: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []

    for variant in cfg.MODEL_VARIANTS:
        features = feature_list(data, variant)
        train = select_periods(data, split_dates["training"])
        validation = select_periods(data, split_dates["validation"])
        locked = select_periods(data, split_dates["locked_test"])
        algorithms = (
            cfg.CANDIDATE_ALGORITHMS
            if (variant == cfg.PRIMARY_VARIANT and not skip_candidate_screen)
            else (cfg.PRIMARY_ALGORITHM,)
        )
        preprocessor = make_preprocessor(features)
        X_train = preprocessor.fit_transform(train[features])
        X_validation = preprocessor.transform(validation[features])
        X_locked = preprocessor.transform(locked[features])
        y_train = train["target_90d"].to_numpy(dtype=int)
        y_validation = validation["target_90d"].to_numpy(dtype=int)
        y_locked = locked["target_90d"].to_numpy(dtype=int)
        model_defs = candidate_models(y_train)

        for algorithm in algorithms:
            print(f"Native fit target={target_source} variant={variant} algorithm={algorithm}", flush=True)
            model = model_defs[algorithm]
            model.fit(X_train, y_train)
            raw_validation = model.predict_proba(X_validation)[:, 1]
            calibrator = fit_platt(raw_validation, y_validation)
            calibrated_validation = apply_platt(calibrator, raw_validation)
            threshold = choose_f1_threshold(y_validation, calibrated_validation)
            raw_locked = model.predict_proba(X_locked)[:, 1]
            calibrated_locked = apply_platt(calibrator, raw_locked)

            for split_name, split_frame, y, raw, calibrated in (
                ("validation", validation, y_validation, raw_validation, calibrated_validation),
                ("locked_test", locked, y_locked, raw_locked, calibrated_locked),
            ):
                row = {
                    "target_source": target_source,
                    "variant": variant,
                    "algorithm": algorithm,
                    "evaluation_scope": f"native_{split_name}",
                    "feature_count": len(features),
                    "training_rows": len(train),
                    "training_positives": int(y_train.sum()),
                    **evaluate_scores(y, raw, calibrated, threshold),
                }
                metrics_rows.append(row)
                prediction = split_frame[
                    [
                        "machine_key", "full_model", "period_start", "period_end",
                        "target_window_start", "target_window_end", "target_first_claim_date_90d",
                        "target_claim_count_90d", "label_complete_90d", "days_to_next_target_event",
                    ]
                ].copy()
                prediction["target_source"] = target_source
                prediction["variant"] = variant
                prediction["algorithm"] = algorithm
                prediction["evaluation_scope"] = f"native_{split_name}"
                prediction["true_label"] = y
                prediction["raw_score"] = raw
                prediction["calibrated_probability"] = calibrated
                prediction["prediction_threshold"] = threshold
                prediction["predicted_label"] = (calibrated >= threshold).astype("int8")
                prediction["rank_within_period"] = prediction.groupby("period_end")["raw_score"].rank(method="first", ascending=False).astype(int)
                prediction_parts.append(prediction)

                for period_end, period in prediction.groupby("period_end", sort=True):
                    ranked = period.sort_values(["raw_score", "machine_key"], ascending=[False, True])
                    for rate in cfg.TOP_K_RATES:
                        selection_rows.append(
                            {
                                "target_source": target_source,
                                "variant": variant,
                                "algorithm": algorithm,
                                "evaluation_scope": f"native_{split_name}",
                                "group_date": period_end,
                                **selection_metrics_for_group(
                                    ranked,
                                    selection_type="top_k_fraction",
                                    selection_value=rate,
                                ),
                            }
                        )
                    for count in cfg.TOP_N_COUNTS:
                        selection_rows.append(
                            {
                                "target_source": target_source,
                                "variant": variant,
                                "algorithm": algorithm,
                                "evaluation_scope": f"native_{split_name}",
                                "group_date": period_end,
                                **selection_metrics_for_group(
                                    ranked,
                                    selection_type="top_n",
                                    selection_value=count,
                                ),
                            }
                        )

            artifact = {
                "target_source": target_source,
                "variant": variant,
                "algorithm": algorithm,
                "lookback_days": cfg.LOOKBACK_DAYS,
                "horizon_days": cfg.HORIZON_DAYS,
                "features": features,
                "preprocessor": preprocessor,
                "model": model,
                "calibrator": calibrator,
                "threshold": threshold,
                "split_dates": {k: [str(pd.Timestamp(v).date()) for v in values] for k, values in split_dates.items()},
            }
            joblib.dump(artifact, output_dir / f"{variant}__{algorithm}.joblib")

    metrics = pd.DataFrame(metrics_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    metrics.to_csv(output_dir / "native_metrics.csv", index=False)
    selections.to_csv(output_dir / "native_top_k_top_n_by_period.csv", index=False)
    predictions.to_csv(output_dir / "native_predictions.csv.gz", index=False, compression="gzip")
    return metrics, selections, predictions

def anchor_experiment(
    segment_data: pd.DataFrame,
    anchor_data: pd.DataFrame,
    *,
    target_source: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir = output_dir / "ranked_predictions"
    ranked_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    compact_parts: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []

    for fold_name, validation_start, validation_end in cfg.VALIDATION_FOLDS:
        validation_start_ts = pd.Timestamp(validation_start)
        eligible_periods = (
            segment_data[
                (segment_data["period_end"] + pd.Timedelta(days=int(cfg.HORIZON_DAYS))).lt(validation_start_ts)
            ][["period_start", "period_end"]]
            .drop_duplicates()
            .sort_values("period_start")
        )
        if len(eligible_periods) < 3:
            raise ValueError(f"Insufficient segment history before fold {fold_name}")
        calibration_start = pd.Timestamp(eligible_periods.iloc[-1]["period_start"])
        fit_starts = eligible_periods.iloc[:-1]["period_start"].tolist()
        fit = select_periods(segment_data, fit_starts)
        calibration = segment_data[segment_data["period_start"].eq(calibration_start)].copy()
        fold_anchors = anchor_data[anchor_data["fold"].eq(fold_name)].copy()

        for variant in cfg.MODEL_VARIANTS:
            features = feature_list(segment_data, variant)
            preprocessor, model = fit_preprocessed_model(fit, features, cfg.PRIMARY_ALGORITHM)
            raw_calibration = score_model(preprocessor, model, calibration, features)
            y_calibration = calibration["target_90d"].to_numpy(dtype=int)
            calibrator = fit_platt(raw_calibration, y_calibration)
            calibrated_calibration = apply_platt(calibrator, raw_calibration)
            threshold = choose_f1_threshold(y_calibration, calibrated_calibration)

            fold_audit.append(
                {
                    "target_source": target_source,
                    "variant": variant,
                    "fold": fold_name,
                    "validation_start": validation_start,
                    "validation_end_exclusive": validation_end,
                    "fit_periods": len(fit_starts),
                    "fit_rows": len(fit),
                    "fit_positives": int(fit["target_90d"].sum()),
                    "calibration_period_start": calibration_start,
                    "calibration_rows": len(calibration),
                    "calibration_positives": int(y_calibration.sum()),
                    "feature_count": len(features),
                    "threshold": threshold,
                }
            )

            for anchor_date, anchor in fold_anchors.groupby("anchor_date", sort=True):
                raw = score_model(preprocessor, model, anchor, features)
                calibrated = apply_platt(calibrator, raw)
                ranked = rank_predictions(anchor, raw, calibrated, threshold)
                y = ranked["true_label"].to_numpy(dtype=int)
                row = {
                    "target_source": target_source,
                    "variant": variant,
                    "algorithm": cfg.PRIMARY_ALGORITHM,
                    "evaluation_scope": "same_date_anchor_fleet",
                    "fold": fold_name,
                    "anchor_id": str(ranked["anchor_id"].iloc[0]),
                    "anchor_date": pd.Timestamp(anchor_date),
                    "outcome_window_complete": bool(ranked["label_complete_90d"].all()),
                    "feature_count": len(features),
                    **evaluate_scores(
                        y,
                        ranked["raw_score"].to_numpy(),
                        ranked["calibrated_probability"].to_numpy(),
                        threshold,
                    ),
                }
                metric_rows.append(row)

                for rate in cfg.TOP_K_RATES:
                    selection_rows.append(
                        {
                            "target_source": target_source,
                            "variant": variant,
                            "algorithm": cfg.PRIMARY_ALGORITHM,
                            "evaluation_scope": "same_date_anchor_fleet",
                            "fold": fold_name,
                            "anchor_id": row["anchor_id"],
                            "anchor_date": anchor_date,
                            "outcome_window_complete": row["outcome_window_complete"],
                            **selection_metrics_for_group(
                                ranked,
                                selection_type="top_k_fraction",
                                selection_value=rate,
                            ),
                        }
                    )
                for count in cfg.TOP_N_COUNTS:
                    selection_rows.append(
                        {
                            "target_source": target_source,
                            "variant": variant,
                            "algorithm": cfg.PRIMARY_ALGORITHM,
                            "evaluation_scope": "same_date_anchor_fleet",
                            "fold": fold_name,
                            "anchor_id": row["anchor_id"],
                            "anchor_date": anchor_date,
                            "outcome_window_complete": row["outcome_window_complete"],
                            **selection_metrics_for_group(
                                ranked,
                                selection_type="top_n",
                                selection_value=count,
                            ),
                        }
                    )

                file_dir = ranked_dir / variant / fold_name
                file_dir.mkdir(parents=True, exist_ok=True)
                ranked_file = file_dir / f"{pd.Timestamp(anchor_date):%Y-%m-%d}__ranked_fleet.csv.gz"
                ranked.to_csv(ranked_file, index=False, compression="gzip")
                compact_cols = [
                    "target_source", "fold", "anchor_id", "anchor_date", "machine_key", "full_model",
                    "period_start", "period_end", "target_window_start", "target_window_end",
                    "label_complete_90d", "true_label", "predicted_label", "prediction_outcome",
                    "raw_score", "calibrated_probability", "risk_score", "risk_score_0_100", "prediction_threshold", "risk_index",
                    "rank_within_anchor", "score_top_fraction", "target_claim_count_90d",
                    "target_first_claim_date_90d", "next_target_event_date", "days_to_next_target_event",
                ] + [f"selected_top_{int(round(rate * 100))}pct" for rate in cfg.TOP_K_RATES] + [
                    f"selected_top_n_{count}" for count in cfg.TOP_N_COUNTS
                ]
                compact = ranked[[c for c in compact_cols if c in ranked.columns]].copy()
                compact["variant"] = variant
                compact["algorithm"] = cfg.PRIMARY_ALGORITHM
                compact["ranked_file"] = str(ranked_file.relative_to(cfg.PROJECT_DIR))
                compact_parts.append(compact)

    metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    compact = pd.concat(compact_parts, ignore_index=True, sort=False)
    metrics.to_csv(output_dir / "anchor_metrics_by_anchor.csv", index=False)
    selections.to_csv(output_dir / "anchor_top_k_top_n_by_anchor.csv", index=False)
    compact.to_csv(output_dir / "anchor_all_ranked_predictions_compact.csv.gz", index=False, compression="gzip")
    pd.DataFrame(fold_audit).to_csv(output_dir / "anchor_fold_training_audit.csv", index=False)
    return metrics, selections, compact
