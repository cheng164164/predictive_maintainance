"""
Utility functions for standalone machine-learning training and validation.

The functions in this module intentionally do not import feature-selection code.
They repeat the needed split/preprocessing/training utilities so that the ML
workflow can be run directly when new snapshot data arrives.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_assignments: pd.DataFrame


@dataclass
class FoldDefinition:
    fold_id: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    gap_start_date: pd.Timestamp
    gap_end_date: pd.Timestamp
    validation_start_date: pd.Timestamp
    validation_end_date: pd.Timestamp
    train_index: np.ndarray
    validation_index: np.ndarray


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: Optional[pd.DataFrame]
    feature_map: pd.DataFrame
    numeric_input_cols: List[str]
    categorical_input_cols: List[str]
    preprocessor: ColumnTransformer
    selected_prepared_features: List[str]
    missing_selected_prepared_features: List[str]
    extra_prepared_features_dropped: List[str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_path(path: Path) -> Path:
    """Resolve a configured snapshot path that may be either a file or folder."""

    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        # Common fallback if config points to a stem without extension.
        for suffix in [".parquet", ".pq", ".csv", ".xlsx", ".xls"]:
            candidate = Path(str(path) + suffix)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Input data path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Input data path is neither a file nor a directory: {path}")

    preferred_names = [
        "snapshot_dataframe.parquet",
        "snapshot_dataframe.pq",
        "snapshot_dataframe.csv",
        "snapshot_dataframe.xlsx",
        "snapshot_dataframe.xls",
    ]
    for name in preferred_names:
        candidate = path / name
        if candidate.is_file():
            return candidate

    for pattern in ["*.parquet", "*.pq", "*.csv", "*.xlsx", "*.xls"]:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No supported snapshot file found inside directory: {path}. "
        "Expected parquet, pq, csv, xlsx, or xls."
    )


def read_snapshot(path: Path, date_col: str) -> pd.DataFrame:
    """Load snapshot dataframe from CSV, Parquet, or Excel and parse the date column."""

    resolved = resolve_input_path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(resolved)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(resolved)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved)
    else:
        raise ValueError(f"Unsupported input extension '{suffix}' for {resolved}")

    if date_col not in df.columns:
        raise ValueError(f"DATE_COL='{date_col}' was not found in input data: {resolved}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"DATE_COL='{date_col}' could not be parsed as dates.")
    if df[date_col].isna().any():
        missing = int(df[date_col].isna().sum())
        raise ValueError(f"DATE_COL='{date_col}' has {missing} missing/unparseable dates.")
    return df


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    secondary_sort_cols: Optional[Sequence[str]] = None,
) -> Tuple[SplitResult, Dict[str, float]]:
    """Split rows chronologically into train, validation, and test."""

    ratio_sum = train_ratio + validation_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    effective = {
        "training_main": train_ratio / ratio_sum,
        "validation_holdout": validation_ratio / ratio_sum,
        "test_holdout": test_ratio / ratio_sum,
    }

    sort_cols = [date_col]
    for col in secondary_sort_cols or []:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)

    sorted_df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    n_rows = len(sorted_df)
    train_end = int(np.floor(n_rows * effective["training_main"]))
    validation_end = int(np.floor(n_rows * (effective["training_main"] + effective["validation_holdout"])))

    train = sorted_df.iloc[:train_end].copy()
    validation = sorted_df.iloc[train_end:validation_end].copy()
    test = sorted_df.iloc[validation_end:].copy()

    audit_cols = [c for c in [date_col] + list(secondary_sort_cols or []) if c in sorted_df.columns]
    assignments = sorted_df[audit_cols].copy()
    assignments["row_position_chronological"] = np.arange(n_rows)
    assignments["split"] = "test_holdout"
    if train_end > 0:
        assignments.loc[: train_end - 1, "split"] = "training_main"
    if validation_end > train_end:
        assignments.loc[train_end: validation_end - 1, "split"] = "validation_holdout"

    return SplitResult(train=train, validation=validation, test=test, split_assignments=assignments), effective


def target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    y = pd.to_numeric(df[target_col], errors="coerce")
    if y.isna().any():
        raise ValueError(f"TARGET_COL='{target_col}' contains missing or non-numeric values.")
    return y.astype(int)


def summarize_split(name: str, df: pd.DataFrame, target_col: str, date_col: str, id_cols: Sequence[str]) -> dict:
    out = {
        "split": name,
        "rows": int(len(df)),
        "date_min": None,
        "date_max": None,
        "target_positive_count": None,
        "target_positive_rate": None,
    }
    if len(df):
        out["date_min"] = df[date_col].min()
        out["date_max"] = df[date_col].max()
    if target_col in df.columns and len(df):
        y = target_series(df, target_col)
        out["target_positive_count"] = int((y == 1).sum())
        out["target_positive_rate"] = float((y == 1).mean())
    for col in id_cols or []:
        if col in df.columns:
            out[f"unique_{col}"] = int(df[col].nunique(dropna=True))
    return out


def make_split_summary(split: SplitResult, target_col: str, date_col: str, id_cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            summarize_split("training_main", split.train, target_col, date_col, id_cols),
            summarize_split("validation_holdout", split.validation, target_col, date_col, id_cols),
            summarize_split("test_holdout", split.test, target_col, date_col, id_cols),
        ]
    )


def source_features_for_prepared_features(
    prepared_features: Sequence[str], prepared_to_source: Dict[str, str]
) -> List[str]:
    missing = [f for f in prepared_features if f not in prepared_to_source]
    if missing:
        raise ValueError(
            "Prepared features are missing from PREPARED_TO_SOURCE_FEATURE mapping: "
            + ", ".join(missing[:20])
        )
    out = []
    for f in prepared_features:
        source = prepared_to_source[f]
        if source not in out:
            out.append(source)
    return out


def validate_source_features(
    df: pd.DataFrame,
    source_features: Sequence[str],
    error_on_missing: bool = True,
) -> Tuple[List[str], List[str]]:
    present = [c for c in source_features if c in df.columns]
    missing = [c for c in source_features if c not in df.columns]
    if missing and error_on_missing:
        raise ValueError(
            "The input snapshot is missing source columns required by the selected feature set: "
            + ", ".join(missing)
        )
    return present, missing


def _safe_feature_name(name: str, existing: set) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not safe:
        safe = "feature"
    base = safe
    i = 2
    while safe in existing:
        safe = f"{base}_{i}"
        i += 1
    existing.add(safe)
    return safe


def _make_one_hot_encoder():
    """Create a dense OneHotEncoder compatible with old/new sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_transform_prepared_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    source_features: Sequence[str],
    selected_prepared_features: Sequence[str],
    numeric_impute_strategy: str,
    categorical_impute_strategy: str,
    one_hot_encode_categorical: bool,
    add_missing_prepared_features_as_zero: bool = True,
) -> PreparedData:
    """
    Fit preprocessing on train_df and transform validation/test consistently.

    The final returned matrices contain exactly selected_prepared_features in the
    configured order. Missing one-hot columns can be added as all-zero columns.
    """

    source_features = list(source_features)
    selected_prepared_features = list(selected_prepared_features)
    X_train_raw = train_df[source_features].copy()

    numeric_cols = [c for c in source_features if pd.api.types.is_numeric_dtype(X_train_raw[c])]
    categorical_cols = [c for c in source_features if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=numeric_impute_strategy))])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        if not one_hot_encode_categorical:
            raise ValueError("Categorical features are present but one-hot encoding is disabled.")
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No source features available for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    train_arr = preprocessor.fit_transform(train_df[source_features])
    val_arr = preprocessor.transform(validation_df[source_features])
    test_arr = preprocessor.transform(test_df[source_features]) if test_df is not None else None

    original_names = list(preprocessor.get_feature_names_out())
    existing = set()
    safe_names = [_safe_feature_name(x, existing) for x in original_names]

    def as_df(arr, index):
        return pd.DataFrame(arr, columns=safe_names, index=index).astype(float)

    X_train_all = as_df(train_arr, train_df.index)
    X_val_all = as_df(val_arr, validation_df.index)
    X_test_all = as_df(test_arr, test_df.index) if test_arr is not None and test_df is not None else None

    missing_selected = [f for f in selected_prepared_features if f not in X_train_all.columns]
    if missing_selected and not add_missing_prepared_features_as_zero:
        raise ValueError(
            "Selected prepared features were not generated by the preprocessor: "
            + ", ".join(missing_selected)
        )

    for feature in missing_selected:
        X_train_all[feature] = 0.0
        X_val_all[feature] = 0.0
        if X_test_all is not None:
            X_test_all[feature] = 0.0

    extra_dropped = [f for f in X_train_all.columns if f not in set(selected_prepared_features)]

    X_train = X_train_all[selected_prepared_features].copy()
    X_val = X_val_all[selected_prepared_features].copy()
    X_test = X_test_all[selected_prepared_features].copy() if X_test_all is not None else None

    feature_map = pd.DataFrame(
        {
            "prepared_feature": safe_names,
            "preprocessor_feature_name": original_names,
            "selected_for_model": [name in set(selected_prepared_features) for name in safe_names],
        }
    )
    if missing_selected:
        extra_rows = pd.DataFrame(
            {
                "prepared_feature": missing_selected,
                "preprocessor_feature_name": missing_selected,
                "selected_for_model": True,
                "note": "selected prepared feature was missing from this fit and added as zero",
            }
        )
        feature_map = pd.concat([feature_map, extra_rows], ignore_index=True)

    return PreparedData(
        X_train=X_train,
        X_validation=X_val,
        X_test=X_test,
        feature_map=feature_map,
        numeric_input_cols=list(numeric_cols),
        categorical_input_cols=list(categorical_cols),
        preprocessor=preprocessor,
        selected_prepared_features=selected_prepared_features,
        missing_selected_prepared_features=missing_selected,
        extra_prepared_features_dropped=extra_dropped,
    )


def build_expanding_window_folds(
    training_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    n_splits: int,
    validation_window_days: int,
    gap_days: int,
    min_train_rows: int,
    min_validation_rows: int,
    min_positives_in_train: int,
    min_positives_in_validation: int,
) -> Tuple[List[FoldDefinition], pd.DataFrame]:
    """Build expanding-window CV folds inside training_main only."""

    if n_splits <= 0:
        raise ValueError("CV_N_SPLITS must be positive.")
    if validation_window_days <= 0:
        raise ValueError("CV_VALIDATION_WINDOW_DAYS must be positive.")
    if gap_days < 0:
        raise ValueError("CV_GAP_DAYS cannot be negative.")

    df = training_df.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    dates = pd.to_datetime(df[date_col])
    y = target_series(df, target_col)
    last_date = dates.max().normalize()

    folds: List[FoldDefinition] = []
    audit_rows = []
    one_day = pd.Timedelta(days=1)

    # Chronological windows from oldest validation fold to newest validation fold.
    for fold_id in range(1, n_splits + 1):
        windows_after = n_splits - fold_id
        validation_end_exclusive = last_date + one_day - pd.Timedelta(days=windows_after * validation_window_days)
        validation_start = validation_end_exclusive - pd.Timedelta(days=validation_window_days)
        validation_end_inclusive = validation_end_exclusive - one_day
        gap_end = validation_start - one_day
        gap_start = validation_start - pd.Timedelta(days=gap_days)
        train_end_exclusive = gap_start

        train_mask = dates < train_end_exclusive
        val_mask = (dates >= validation_start) & (dates < validation_end_exclusive)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())

        train_pos = int((y.iloc[train_idx] == 1).sum()) if len(train_idx) else 0
        val_pos = int((y.iloc[val_idx] == 1).sum()) if len(val_idx) else 0

        status = "used"
        reasons = []
        if len(train_idx) < min_train_rows:
            reasons.append(f"train_rows_lt_{min_train_rows}")
        if len(val_idx) < min_validation_rows:
            reasons.append(f"validation_rows_lt_{min_validation_rows}")
        if train_pos < min_positives_in_train:
            reasons.append(f"train_positives_lt_{min_positives_in_train}")
        if val_pos < min_positives_in_validation:
            reasons.append(f"validation_positives_lt_{min_positives_in_validation}")
        if reasons:
            status = "skipped"

        audit_rows.append(
            {
                "fold_id": fold_id,
                "status": status,
                "skip_reasons": ";".join(reasons),
                "train_start_date": dates.iloc[train_idx].min() if len(train_idx) else pd.NaT,
                "train_end_date": dates.iloc[train_idx].max() if len(train_idx) else pd.NaT,
                "gap_start_date": gap_start,
                "gap_end_date": gap_end,
                "validation_start_date": validation_start,
                "validation_end_date": validation_end_inclusive,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_positive_count": train_pos,
                "validation_positive_count": val_pos,
                "train_positive_rate": float(train_pos / len(train_idx)) if len(train_idx) else np.nan,
                "validation_positive_rate": float(val_pos / len(val_idx)) if len(val_idx) else np.nan,
            }
        )

        if status == "used":
            folds.append(
                FoldDefinition(
                    fold_id=fold_id,
                    train_start_date=dates.iloc[train_idx].min(),
                    train_end_date=dates.iloc[train_idx].max(),
                    gap_start_date=gap_start,
                    gap_end_date=gap_end,
                    validation_start_date=validation_start,
                    validation_end_date=validation_end_inclusive,
                    train_index=train_idx,
                    validation_index=val_idx,
                )
            )

    return folds, pd.DataFrame(audit_rows)


def compute_scale_pos_weight(y: pd.Series) -> float:
    y = pd.Series(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(neg / pos, 1.0)


def make_xgb_params(default_params: dict, override_params: Optional[dict], y: pd.Series, use_scale_pos_weight: bool) -> dict:
    params = dict(default_params)
    if override_params:
        params.update(override_params)
    if use_scale_pos_weight:
        params["scale_pos_weight"] = compute_scale_pos_weight(y)
    return params


def train_xgboost_classifier(X: pd.DataFrame, y: pd.Series, params: dict):
    y = pd.Series(y).astype(int)
    if y.nunique(dropna=True) < 2:
        raise ValueError("XGBoost training target has fewer than two classes.")
    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


def threshold_free_metrics(y_true: pd.Series, probability: np.ndarray) -> dict:
    y_true = pd.Series(y_true).astype(int)
    result = {
        "rows": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()) if len(y_true) else np.nan,
    }
    if y_true.nunique(dropna=True) < 2:
        result["warning"] = "Only one target class present; threshold-free metrics are undefined."
        return result
    result["average_precision"] = float(average_precision_score(y_true, probability))
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
    except Exception as exc:
        result["roc_auc_warning"] = str(exc)
    try:
        result["log_loss"] = float(log_loss(y_true, probability, labels=[0, 1]))
    except Exception as exc:
        result["log_loss_warning"] = str(exc)
    return result


def metrics_at_threshold(y_true: pd.Series, probability: np.ndarray, threshold: float, beta: float = 2.0) -> dict:
    y_true = pd.Series(y_true).astype(int)
    pred = (np.asarray(probability) >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()) if len(y_true) else np.nan,
        "flagged_count": int(pred.sum()),
        "flagged_rate": float(pred.mean()) if len(pred) else np.nan,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        f"f_beta_{beta:g}": float(fbeta_score(y_true, pred, beta=beta, zero_division=0)),
    }


def threshold_grid(min_threshold: float, max_threshold: float, grid_size: int) -> np.ndarray:
    if grid_size < 2:
        raise ValueError("THRESHOLD_GRID_SIZE must be at least 2.")
    return np.linspace(float(min_threshold), float(max_threshold), int(grid_size))


def threshold_search(
    y_true: pd.Series,
    probability: np.ndarray,
    thresholds: Sequence[float],
    beta: float,
    max_flagged_rate: Optional[float],
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        row = metrics_at_threshold(y_true, probability, float(thr), beta=beta)
        row["within_max_flagged_rate"] = True
        if max_flagged_rate is not None:
            row["within_max_flagged_rate"] = bool(row["flagged_rate"] <= max_flagged_rate)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_threshold(search_df: pd.DataFrame, beta: float) -> dict:
    metric_col = f"f_beta_{beta:g}"
    if metric_col not in search_df.columns:
        metric_col = "f2"
    candidates = search_df[search_df["within_max_flagged_rate"]].copy()
    if candidates.empty:
        candidates = search_df.copy()
        candidates["selection_warning"] = "No thresholds satisfied max flagged-rate constraint. Selected from all thresholds."
    else:
        candidates["selection_warning"] = ""
    candidates = candidates.sort_values(
        [metric_col, "recall", "precision", "flagged_rate", "threshold"],
        ascending=[False, False, False, True, True],
    )
    return candidates.iloc[0].to_dict()


def top_k_metrics(y_true: pd.Series, probability: np.ndarray, top_k_rates: Sequence[float]) -> pd.DataFrame:
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    prob = pd.Series(probability).reset_index(drop=True)
    order = prob.sort_values(ascending=False).index.to_numpy()
    rows = []
    n = len(y_true)
    total_pos = int((y_true == 1).sum())
    for rate in top_k_rates:
        k = int(np.ceil(n * float(rate)))
        k = max(1, min(k, n)) if n else 0
        flagged = np.zeros(n, dtype=int)
        if k:
            flagged[order[:k]] = 1
        tp = int(((flagged == 1) & (y_true.to_numpy() == 1)).sum())
        fp = int(((flagged == 1) & (y_true.to_numpy() == 0)).sum())
        precision = tp / k if k else np.nan
        recall = tp / total_pos if total_pos else np.nan
        rows.append(
            {
                "top_k_rate": float(rate),
                "rows": int(n),
                "flagged_count": int(k),
                "flagged_rate_actual": float(k / n) if n else np.nan,
                "positive_count": total_pos,
                "true_positive": tp,
                "false_positive": fp,
                "precision_at_k": float(precision),
                "recall_at_k": float(recall),
                "lift_vs_random": float(precision / (total_pos / n)) if n and total_pos else np.nan,
                "min_probability_in_top_k": float(prob.iloc[order[k - 1]]) if k else np.nan,
            }
        )
    return pd.DataFrame(rows)


def machine_level_top_k_metrics(
    prediction_df: pd.DataFrame,
    probability_col: str,
    target_col: str,
    machine_id_col: str,
    top_k_rates: Sequence[float],
    date_col: Optional[str] = None,
    probability_aggregation: str = "max",
    target_aggregation: str = "max",
) -> pd.DataFrame:
    """
    Collapse snapshot-level predictions to one row per machine, then calculate
    top-K precision/recall/lift at the machine level.

    Recommended default:
        machine_probability = max(snapshot_probability)
        machine_target = max(snapshot_target)

    This answers: if the business reviews the top X% highest-risk machines,
    how many positive machines are captured?
    """

    required = [machine_id_col, probability_col, target_col]
    if date_col is not None:
        required.append(date_col)
    missing = [c for c in required if c not in prediction_df.columns]
    if missing:
        raise ValueError(f"Missing columns for machine-level top-K metrics: {missing}")

    probability_aggregation = str(probability_aggregation).lower()
    target_aggregation = str(target_aggregation).lower()
    valid_probability_aggs = {"max", "mean", "latest"}
    valid_target_aggs = {"max", "latest"}
    if probability_aggregation not in valid_probability_aggs:
        raise ValueError(
            f"Unsupported probability_aggregation='{probability_aggregation}'. "
            f"Allowed values: {sorted(valid_probability_aggs)}"
        )
    if target_aggregation not in valid_target_aggs:
        raise ValueError(
            f"Unsupported target_aggregation='{target_aggregation}'. "
            f"Allowed values: {sorted(valid_target_aggs)}"
        )
    if (probability_aggregation == "latest" or target_aggregation == "latest") and date_col is None:
        raise ValueError("date_col is required when using latest aggregation.")

    cols = [machine_id_col, probability_col, target_col]
    if date_col is not None:
        cols.append(date_col)
    work = prediction_df[cols].copy()
    work = work.dropna(subset=[machine_id_col])
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int)
    work[probability_col] = pd.to_numeric(work[probability_col], errors="coerce")
    work = work.dropna(subset=[probability_col])
    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    if work.empty:
        return pd.DataFrame(
            columns=[
                "top_k_rate",
                "machine_count",
                "source_snapshot_count",
                "positive_machine_count",
                "machine_positive_rate",
                "flagged_machine_count",
                "flagged_machine_rate_actual",
                "true_positive_machines",
                "false_positive_machines",
                "machine_precision_at_k",
                "machine_recall_at_k",
                "machine_lift_vs_random",
                "min_probability_in_top_k",
            ]
        )

    grouped = work.groupby(machine_id_col, dropna=False)
    machine_df = pd.DataFrame({machine_id_col: grouped.size().index})
    machine_df["source_snapshot_count"] = grouped.size().to_numpy()

    if probability_aggregation == "max":
        machine_df["machine_probability"] = grouped[probability_col].max().to_numpy()
    elif probability_aggregation == "mean":
        machine_df["machine_probability"] = grouped[probability_col].mean().to_numpy()
    else:
        latest_prob = (
            work.sort_values([machine_id_col, date_col], kind="mergesort")
            .groupby(machine_id_col, dropna=False)
            .tail(1)
            .set_index(machine_id_col)[probability_col]
        )
        machine_df["machine_probability"] = machine_df[machine_id_col].map(latest_prob).to_numpy()

    if target_aggregation == "max":
        machine_df["machine_target"] = grouped[target_col].max().to_numpy()
    else:
        latest_target = (
            work.sort_values([machine_id_col, date_col], kind="mergesort")
            .groupby(machine_id_col, dropna=False)
            .tail(1)
            .set_index(machine_id_col)[target_col]
        )
        machine_df["machine_target"] = machine_df[machine_id_col].map(latest_target).fillna(0).astype(int).to_numpy()

    if date_col is not None:
        machine_df["machine_first_snapshot_date"] = grouped[date_col].min().to_numpy()
        machine_df["machine_last_snapshot_date"] = grouped[date_col].max().to_numpy()

    machine_df["machine_target"] = machine_df["machine_target"].astype(int)
    machine_df = machine_df.sort_values(
        ["machine_probability", machine_id_col], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    total_machines = len(machine_df)
    source_snapshot_count = int(machine_df["source_snapshot_count"].sum())
    positive_machines = int(machine_df["machine_target"].sum())
    machine_positive_rate = positive_machines / total_machines if total_machines else np.nan

    rows = []
    for rate in top_k_rates:
        k = int(np.ceil(total_machines * float(rate)))
        k = max(1, min(k, total_machines)) if total_machines else 0
        flagged = machine_df.head(k)
        tp = int(flagged["machine_target"].sum()) if k else 0
        fp = int(k - tp)
        precision = tp / k if k else np.nan
        recall = tp / positive_machines if positive_machines else np.nan
        lift = precision / machine_positive_rate if machine_positive_rate and machine_positive_rate > 0 else np.nan
        min_prob = float(flagged["machine_probability"].min()) if k else np.nan
        rows.append(
            {
                "top_k_rate": float(rate),
                "machine_count": int(total_machines),
                "source_snapshot_count": int(source_snapshot_count),
                "positive_machine_count": int(positive_machines),
                "machine_positive_rate": float(machine_positive_rate),
                "flagged_machine_count": int(k),
                "flagged_machine_rate_actual": float(k / total_machines) if total_machines else np.nan,
                "true_positive_machines": int(tp),
                "false_positive_machines": int(fp),
                "machine_precision_at_k": float(precision),
                "machine_recall_at_k": float(recall),
                "machine_lift_vs_random": float(lift),
                "min_probability_in_top_k": min_prob,
                "probability_aggregation": probability_aggregation,
                "target_aggregation": target_aggregation,
                "machine_id_col": machine_id_col,
            }
        )
    return pd.DataFrame(rows)


def xgboost_importance(model, feature_names: Sequence[str]) -> pd.DataFrame:
    booster = model.get_booster()
    rows = pd.DataFrame({"prepared_feature": list(feature_names)})
    for imp_type in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        raw_scores = booster.get_score(importance_type=imp_type)
        rows[f"xgb_{imp_type}"] = rows["prepared_feature"].map(raw_scores).fillna(0.0)
    for col in [c for c in rows.columns if c.startswith("xgb_")]:
        rows[f"rank_{col}"] = rows[col].rank(ascending=False, method="min")
    return rows.sort_values(["xgb_total_gain", "xgb_gain"], ascending=False).reset_index(drop=True)


def make_param_grid(default_params: dict, hyperparameter_grid: Sequence[dict]) -> List[dict]:
    """Return parameter overrides to evaluate. Empty grid means one default run."""

    if not hyperparameter_grid:
        return [{}]
    return [dict(x) for x in hyperparameter_grid]


def param_set_id(override: dict, idx: int) -> str:
    if not override:
        return "default"
    parts = [f"{k}={override[k]}" for k in sorted(override)]
    label = "__".join(parts)
    safe = re.sub(r"[^0-9A-Za-z_=.\-]+", "_", label)
    return f"grid_{idx:03d}__{safe}"


def prediction_frame(
    df: pd.DataFrame,
    date_col: str,
    id_cols: Sequence[str],
    y_true: pd.Series,
    probability: np.ndarray,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    cols = [c for c in [date_col] + list(id_cols or []) if c in df.columns]
    out = df[cols].copy().reset_index(drop=True)
    out["y_true"] = pd.Series(y_true).astype(int).reset_index(drop=True)
    out["probability"] = np.asarray(probability)
    if threshold is not None:
        out["selected_threshold"] = float(threshold)
        out["predicted_flag"] = (out["probability"] >= float(threshold)).astype(int)
    return out


def save_model_artifacts(model, prepared: PreparedData, output_dir: Path, metadata: dict) -> None:
    ensure_dir(output_dir)
    model_path = output_dir / "final_xgboost_model.json"
    model.save_model(str(model_path))
    joblib.dump(prepared.preprocessor, output_dir / "final_preprocessor.joblib")
    write_json(metadata, output_dir / "final_model_metadata.json")
    pd.DataFrame({"prepared_feature": prepared.selected_prepared_features}).to_csv(
        output_dir / "selected_prepared_features.csv", index=False
    )
    prepared.feature_map.to_csv(output_dir / "feature_map_from_final_preprocessor.csv", index=False)
