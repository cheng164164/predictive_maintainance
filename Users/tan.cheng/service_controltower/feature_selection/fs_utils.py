"""
Utility functions for the feature-selection analysis workflow.

The project keeps most reusable logic in this module so that main.py can read as
an end-to-end orchestration script. Each function is intentionally small and has
an explicit responsibility: loading data, splitting chronologically,
preprocessing, creating feature reports, training XGBoost, or saving outputs.

Important leakage rule followed by this module:
- Methods that learn from features or targets for ranking are fitted on
  feature_train only.
- Permutation and SHAP are computed on feature_selection_holdout only.
- validation_holdout and test_holdout are transformed only for future use; they
  are not used to rank features in this workflow.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    log_loss,
    make_scorer,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


@dataclass
class SplitResult:
    """
    Container for the outer chronological split.

    Attributes
    ----------
    train:
        Older portion of the dataset used for all feature-selection work.
    validation:
        Middle/future holdout reserved for later model validation.
    test:
        Final future holdout reserved for final model reporting.
    split_assignments:
        Row-level audit table showing where each chronological row was assigned.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_assignments: pd.DataFrame


@dataclass
class InnerSplitResult:
    """
    Container for the split performed inside training_main.

    Attributes
    ----------
    feature_train:
        Earlier part of training_main used to fit preprocessing, statistical
        rankings, and the temporary XGBoost model.
    feature_selection_holdout:
        Later part of training_main used for permutation and SHAP reporting.
    split_assignments:
        Row-level audit table showing the inner split assignment.
    """

    feature_train: pd.DataFrame
    feature_selection_holdout: pd.DataFrame
    split_assignments: pd.DataFrame


@dataclass
class PreparedData:
    """
    Container for preprocessed feature matrices and feature mapping metadata.

    The prepared matrices are numeric-only because categorical features are
    one-hot encoded when present. The feature_map links each prepared column back
    to the raw source feature and its configured feature group.
    """

    X_feature_train: pd.DataFrame
    X_feature_holdout: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    feature_map: pd.DataFrame
    numeric_input_cols: List[str]
    categorical_input_cols: List[str]
    preprocessor: ColumnTransformer


def ensure_dir(path: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path to create. Parent directories are created as needed.
    """

    path.mkdir(parents=True, exist_ok=True)


def write_json(obj: dict, path: Path) -> None:
    """
    Save a Python dictionary to a pretty-printed JSON file.

    default=str is used so that dates, numpy scalars, and other report objects
    that are not natively JSON serializable can still be written safely.
    """

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_snapshot(path: Path, date_col: str) -> pd.DataFrame:
    """
    Load the unified snapshot dataset from CSV, Parquet, or Excel.

    The date column is parsed immediately because every downstream split depends
    on chronological ordering. The function raises clear errors for missing
    files, unsupported extensions, missing date columns, or unparseable dates.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported input extension '{path.suffix}'. Use csv, parquet, or xlsx."
        )

    if date_col not in df.columns:
        raise ValueError(f"DATE_COL='{date_col}' was not found in input data.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().all():
        raise ValueError(f"DATE_COL='{date_col}' could not be parsed as dates.")

    return df


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    secondary_sort_cols: Optional[Sequence[str]] = None,
) -> SplitResult:
    """
    Split rows chronologically into training, validation, and test partitions.

    The ratio values are treated as weights. This means values such as
    0.75/0.15/0.15 are accepted even though they sum to 1.05; the function
    normalizes them internally before calculating row boundaries.

    Chronological splitting is used instead of random splitting because the
    predictive-maintenance use case is future prediction: older snapshots should
    train the model and newer snapshots should validate/test it.
    """

    ratio_sum = train_ratio + validation_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        train_ratio = train_ratio / ratio_sum
        validation_ratio = validation_ratio / ratio_sum
        test_ratio = test_ratio / ratio_sum

    sort_cols = [date_col]
    for col in secondary_sort_cols or []:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)

    sorted_df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    n_rows = len(sorted_df)
    train_end = int(np.floor(n_rows * train_ratio))
    validation_end = int(np.floor(n_rows * (train_ratio + validation_ratio)))

    train = sorted_df.iloc[:train_end].copy()
    validation = sorted_df.iloc[train_end:validation_end].copy()
    test = sorted_df.iloc[validation_end:].copy()

    audit_cols = [c for c in [date_col] + list(secondary_sort_cols or []) if c in sorted_df.columns]
    assignments = sorted_df[audit_cols].copy()
    assignments["row_position_chronological"] = np.arange(n_rows)
    assignments["split"] = "test_holdout"
    assignments.loc[: train_end - 1, "split"] = "training_main"
    assignments.loc[train_end: validation_end - 1, "split"] = "validation_holdout"

    return SplitResult(
        train=train,
        validation=validation,
        test=test,
        split_assignments=assignments,
    )


def inner_training_split(
    training_df: pd.DataFrame,
    date_col: str,
    feature_train_ratio: float,
    secondary_sort_cols: Optional[Sequence[str]] = None,
) -> InnerSplitResult:
    """
    Split training_main into feature_train and feature_selection_holdout.

    feature_train is the only split used to fit preprocessing and target-aware
    feature ranking methods. feature_selection_holdout is used only for
    permutation and SHAP review, keeping the outer validation/test sets clean.
    """

    if not 0 < feature_train_ratio < 1:
        raise ValueError("feature_train_ratio must be between 0 and 1.")

    sort_cols = [date_col]
    for col in secondary_sort_cols or []:
        if col in training_df.columns and col not in sort_cols:
            sort_cols.append(col)

    sorted_df = training_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    n_rows = len(sorted_df)
    feature_train_end = int(np.floor(n_rows * feature_train_ratio))

    feature_train = sorted_df.iloc[:feature_train_end].copy()
    feature_holdout = sorted_df.iloc[feature_train_end:].copy()

    audit_cols = [c for c in [date_col] + list(secondary_sort_cols or []) if c in sorted_df.columns]
    assignments = sorted_df[audit_cols].copy()
    assignments["row_position_within_training"] = np.arange(n_rows)
    assignments["inner_split"] = "feature_selection_holdout"
    assignments.loc[: feature_train_end - 1, "inner_split"] = "feature_train"

    return InnerSplitResult(
        feature_train=feature_train,
        feature_selection_holdout=feature_holdout,
        split_assignments=assignments,
    )


def summarize_split(
    name: str,
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    id_cols: Optional[Sequence[str]] = None,
) -> dict:
    """
    Produce a compact row/date/target summary for one split.

    This report is useful for quickly checking whether each split contains enough
    rows, enough positive target examples, and enough machine diversity to make
    the downstream feature-selection reports meaningful.
    """

    out = {
        "split": name,
        "rows": int(len(df)),
        "date_min": None,
        "date_max": None,
        "target_positive_count": None,
        "target_positive_rate": None,
    }
    if len(df) > 0:
        out["date_min"] = df[date_col].min()
        out["date_max"] = df[date_col].max()
    if target_col in df.columns and len(df) > 0:
        y = pd.to_numeric(df[target_col], errors="coerce")
        out["target_positive_count"] = int((y == 1).sum())
        out["target_positive_rate"] = float((y == 1).mean())
    for col in id_cols or []:
        if col in df.columns:
            out[f"unique_{col}"] = int(df[col].nunique(dropna=True))
    return out


def get_candidate_features(
    df: pd.DataFrame,
    exclude_cols: Sequence[str],
    candidate_feature_cols: Optional[Sequence[str]] = None,
    manual_drop_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Determine the raw feature columns that should enter feature analysis.

    If candidate_feature_cols is configured, the function validates and uses
    exactly that list. Otherwise, it uses all dataframe columns except excluded
    columns, then removes any manually dropped columns. No preprocessing or
    ranking happens here; this is only raw column selection.
    """

    if candidate_feature_cols:
        missing = [c for c in candidate_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Configured CANDIDATE_FEATURE_COLS not found: {missing}")
        features = list(candidate_feature_cols)
    else:
        exclude = set(exclude_cols)
        features = [c for c in df.columns if c not in exclude]

    drop = set(manual_drop_cols or [])
    features = [c for c in features if c not in drop]
    return features


def raw_missing_value_report(
    split_frames: Dict[str, pd.DataFrame],
    features: Sequence[str],
) -> pd.DataFrame:
    """
    Count missing and non-missing raw values before preprocessing/imputation.

    The report is intentionally generated before SimpleImputer is fitted. This
    preserves visibility into source-data quality, which would otherwise be
    hidden after preprocessing fills missing values.
    """

    rows = []
    for split_name, df in split_frames.items():
        n_rows = len(df)
        for feature in features:
            if feature not in df.columns:
                rows.append(
                    {
                        "split": split_name,
                        "feature": feature,
                        "raw_dtype": None,
                        "rows": n_rows,
                        "missing_count": n_rows,
                        "missing_percent": 100.0 if n_rows else np.nan,
                        "non_missing_count": 0,
                        "non_missing_percent": 0.0 if n_rows else np.nan,
                        "unique_non_missing_count": 0,
                        "all_missing": True,
                        "warning": "feature column not found in this split",
                    }
                )
                continue

            s = df[feature]
            missing_count = int(s.isna().sum())
            non_missing_count = int(s.notna().sum())
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "raw_dtype": str(s.dtype),
                    "rows": n_rows,
                    "missing_count": missing_count,
                    "missing_percent": float(missing_count / n_rows * 100.0) if n_rows else np.nan,
                    "non_missing_count": non_missing_count,
                    "non_missing_percent": float(non_missing_count / n_rows * 100.0) if n_rows else np.nan,
                    "unique_non_missing_count": int(s.nunique(dropna=True)),
                    "all_missing": bool(non_missing_count == 0),
                    "warning": None,
                }
            )
    return pd.DataFrame(rows)


def raw_feature_inventory(
    df: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Create a raw feature diagnostic table on feature_train.

    This complements raw_missing_value_report by adding numeric distribution
    statistics, zero/negative counts, unique counts, and target-class means where
    possible. It is descriptive only and does not apply any keep/drop threshold.
    """

    rows = []
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else None
    for col in features:
        s = df[col]
        row = {
            "feature": col,
            "raw_dtype": str(s.dtype),
            "non_null_count": int(s.notna().sum()),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "unique_count": int(s.nunique(dropna=True)),
            "is_numeric_raw": bool(pd.api.types.is_numeric_dtype(s)),
        }
        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            row.update(
                {
                    "mean": float(sn.mean()) if sn.notna().any() else np.nan,
                    "variance": float(sn.var()) if sn.notna().sum() > 1 else np.nan,
                    "std": float(sn.std()) if sn.notna().sum() > 1 else np.nan,
                    "min": float(sn.min()) if sn.notna().any() else np.nan,
                    "p25": float(sn.quantile(0.25)) if sn.notna().any() else np.nan,
                    "median": float(sn.median()) if sn.notna().any() else np.nan,
                    "p75": float(sn.quantile(0.75)) if sn.notna().any() else np.nan,
                    "max": float(sn.max()) if sn.notna().any() else np.nan,
                    "zero_count": int((sn == 0).sum()),
                    "zero_rate": float((sn == 0).mean()),
                    "negative_count": int((sn < 0).sum()),
                    "negative_rate": float((sn < 0).mean()),
                }
            )
            if y is not None and y.notna().any() and y.nunique(dropna=True) > 1:
                try:
                    row["mean_when_target_0"] = float(sn[y == 0].mean())
                    row["mean_when_target_1"] = float(sn[y == 1].mean())
                except Exception:
                    row["mean_when_target_0"] = np.nan
                    row["mean_when_target_1"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def identify_high_missing_features(
    raw_missing: pd.DataFrame,
    split_name: str,
    threshold: float,
) -> pd.DataFrame:
    """Return source features whose raw missing rate exceeds a threshold.

    Parameters
    ----------
    raw_missing:
        Output of raw_missing_value_report across one or more dataset splits.
    split_name:
        Split used to make the feature-screening decision. In this project this
        should be "feature_train" so future validation/test periods do not
        influence which columns are removed.
    threshold:
        Missing-rate threshold expressed as a fraction, for example 0.90 means
        drop features with more than 90% missing values in the decision split.

    Returns
    -------
    pandas.DataFrame
        One row per feature exceeding the threshold, with the drop reason added.
    """

    if raw_missing.empty:
        return pd.DataFrame()

    decision_rows = raw_missing[raw_missing["split"] == split_name].copy()
    if decision_rows.empty:
        raise ValueError(f"Missingness report does not contain split '{split_name}'.")

    decision_rows["missing_rate"] = decision_rows["missing_percent"] / 100.0
    out = decision_rows[decision_rows["missing_rate"] > float(threshold)].copy()
    if out.empty:
        return out

    out["drop_reason"] = (
        "feature_train_missing_rate_gt_" + str(float(threshold))
    )
    return out.sort_values(["missing_rate", "feature"], ascending=[False, True]).reset_index(drop=True)


def identify_zero_variance_features(raw_inventory: pd.DataFrame) -> pd.DataFrame:
    """Return raw source features with no observed variation in feature_train.

    Numeric columns are flagged when their non-missing values have zero variance
    or, equivalently, one or fewer unique non-missing values. Non-numeric columns
    are flagged when they have one or fewer unique non-missing values. This
    source-level filter runs before preprocessing so downstream correlation and
    model-based reports start from a smaller, cleaner feature set.
    """

    if raw_inventory.empty:
        return raw_inventory.copy()

    out = raw_inventory.copy()
    unique_count = pd.to_numeric(out.get("unique_count"), errors="coerce").fillna(0)
    variance = pd.to_numeric(out.get("variance"), errors="coerce")
    is_numeric = out.get("is_numeric_raw", False)
    if not isinstance(is_numeric, pd.Series):
        is_numeric = pd.Series(bool(is_numeric), index=out.index)

    out["is_zero_variance_or_constant"] = (
        unique_count <= 1
    ) | (
        is_numeric.astype(bool) & variance.notna() & (variance == 0)
    )
    out["drop_reason"] = "zero_variance_or_constant_in_feature_train"

    return (
        out[out["is_zero_variance_or_constant"]]
        .sort_values(["unique_count", "feature"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _safe_feature_name(name: str, existing: set) -> str:
    """
    Convert a preprocessor-generated feature name into a safe unique column name.

    OneHotEncoder can create names with characters that are inconvenient in CSV,
    Excel, XGBoost, or downstream scripts. This helper keeps names alphanumeric
    with underscores and appends a numeric suffix if a collision occurs.
    """

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not cleaned:
        cleaned = "feature"
    if cleaned[0].isdigit():
        cleaned = "f_" + cleaned
    base = cleaned
    i = 2
    while cleaned in existing:
        cleaned = f"{base}_{i}"
        i += 1
    existing.add(cleaned)
    return cleaned


def _extract_source_feature(
    transformer_feature_name: str,
    categorical_cols: Optional[Sequence[str]] = None,
) -> str:
    """
    Map a prepared feature name back to its raw source column.

    Examples
    --------
    num__fault_count_90d -> fault_count_90d
    cat__full_model_PC200 -> full_model

    This mapping is important because one raw categorical feature may expand into
    many prepared one-hot columns, but for reporting/grouping we still want to
    know the original source column.
    """

    if transformer_feature_name.startswith("num__"):
        return transformer_feature_name.replace("num__", "", 1)
    if transformer_feature_name.startswith("cat__"):
        tail = transformer_feature_name.replace("cat__", "", 1)
        for col in sorted(list(categorical_cols or []), key=len, reverse=True):
            if tail == col or tail.startswith(f"{col}_"):
                return col
        return tail
    return transformer_feature_name


def assign_feature_group(
    source_feature: str,
    group_rules: Sequence[dict],
    default_group: str = "other",
) -> str:
    """
    Assign one raw source feature to a configured feature group.

    Grouping is rule-based and intentionally transparent. The function evaluates
    FEATURE_GROUP_RULES in order and returns the first group whose regex pattern
    matches the source feature name. This is used only for grouped correlation
    reporting; it does not remove or select features by itself.
    """

    feature_name = str(source_feature).lower()
    for rule in group_rules:
        group = rule.get("group", default_group)
        for pattern in rule.get("patterns", []):
            if re.search(pattern, feature_name, flags=re.IGNORECASE):
                return group
    return default_group


def add_feature_groups_to_map(
    feature_map: pd.DataFrame,
    group_rules: Sequence[dict],
    default_group: str,
) -> pd.DataFrame:
    """
    Add a feature_group column to the prepared-to-raw feature mapping table.

    The prepared feature map is the central lookup table used across reports. By
    adding feature_group once here, downstream outputs such as correlation,
    ANOVA, XGBoost importance, permutation, SHAP, and consensus reports can all
    show the same group labels.
    """

    out = feature_map.copy()
    out["feature_group"] = out["source_feature"].apply(
        lambda x: assign_feature_group(x, group_rules, default_group)
    )
    return out


def prepare_features(
    feature_train: pd.DataFrame,
    feature_holdout: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    numeric_impute_strategy: str,
    categorical_impute_strategy: str,
    one_hot_encode_categorical: bool,
    feature_group_rules: Optional[Sequence[dict]] = None,
    default_feature_group: str = "other",
) -> PreparedData:
    """
    Fit preprocessing on feature_train and transform all splits consistently.

    Leakage control is the main reason this function exists. The imputer and
    one-hot encoder are fitted only on feature_train, then reused to transform
    feature_selection_holdout, validation_holdout, and test_holdout. That avoids
    learning missing-value medians, category levels, or encoded feature names
    from future holdout data.
    """

    X_train_raw = feature_train[list(features)].copy()

    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(X_train_raw[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy=numeric_impute_strategy))]
        )
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        if not one_hot_encode_categorical:
            raise ValueError(
                "Categorical features are present but ONE_HOT_ENCODE_CATEGORICAL=False."
            )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_impute_strategy)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No candidate features were available after exclusions.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    train_arr = preprocessor.fit_transform(feature_train[list(features)])
    holdout_arr = preprocessor.transform(feature_holdout[list(features)])
    validation_arr = preprocessor.transform(validation[list(features)])
    test_arr = preprocessor.transform(test[list(features)])

    original_names = list(preprocessor.get_feature_names_out())
    existing = set()
    safe_names = [_safe_feature_name(x, existing) for x in original_names]

    feature_map = pd.DataFrame(
        {
            "prepared_feature": safe_names,
            "preprocessor_feature_name": original_names,
            "source_feature": [
                _extract_source_feature(x, categorical_cols) for x in original_names
            ],
        }
    )
    feature_map = add_feature_groups_to_map(
        feature_map,
        group_rules=feature_group_rules or [],
        default_group=default_feature_group,
    )

    def as_df(arr: np.ndarray, index: pd.Index) -> pd.DataFrame:
        """Return a numeric dataframe with stable prepared feature names."""

        return pd.DataFrame(arr, columns=safe_names, index=index).astype(float)

    return PreparedData(
        X_feature_train=as_df(train_arr, feature_train.index),
        X_feature_holdout=as_df(holdout_arr, feature_holdout.index),
        X_validation=as_df(validation_arr, validation.index),
        X_test=as_df(test_arr, test.index),
        feature_map=feature_map,
        numeric_input_cols=list(numeric_cols),
        categorical_input_cols=list(categorical_cols),
        preprocessor=preprocessor,
    )


def prepared_feature_diagnostics(X: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize prepared numeric features after imputation/encoding.

    This report identifies exact constant prepared features, near-zero variance
    candidates, zero-heavy columns, and basic ranges after preprocessing. It is a
    diagnostic report only; no automatic dropping is applied.
    """

    rows = []
    n_rows = len(X)
    for col in X.columns:
        s = X[col]
        rows.append(
            {
                "prepared_feature": col,
                "non_null_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_rate": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "variance": float(s.var()) if n_rows > 1 else np.nan,
                "std": float(s.std()) if n_rows > 1 else np.nan,
                "min": float(s.min()) if n_rows > 0 else np.nan,
                "median": float(s.median()) if n_rows > 0 else np.nan,
                "max": float(s.max()) if n_rows > 0 else np.nan,
                "zero_count": int((s == 0).sum()),
                "zero_rate": float((s == 0).mean()) if n_rows > 0 else np.nan,
                "is_constant_exact": bool(s.nunique(dropna=False) <= 1),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["is_constant_exact", "variance"], ascending=[False, True]
    )


def correlation_pairs(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all pairwise Pearson correlations for the supplied columns.

    This helper is intentionally generic. The main workflow calls it on one
    feature group at a time through grouped_correlation_pairs, which avoids the
    huge all-vs-all correlation matrix produced by global correlation analysis.
    """

    if X.shape[1] < 2:
        return pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
        )

    corr = X.corr(method="pearson", numeric_only=True)
    upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pairs = corr.where(upper_mask).stack(dropna=True).reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def grouped_correlation_pairs(
    X: pd.DataFrame,
    feature_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate pairwise correlations within feature groups only.

    The feature_map must contain prepared_feature and feature_group columns. For
    each group with at least two prepared features, all within-group pairs are
    calculated and saved. No row cap is applied; the reduced pair volume comes
    from grouping before correlation, not from truncating the report.
    """

    required_cols = {"prepared_feature", "feature_group"}
    if not required_cols.issubset(feature_map.columns):
        missing = sorted(required_cols - set(feature_map.columns))
        raise ValueError(f"feature_map is missing required columns: {missing}")

    all_pairs = []
    for group_name, group_map in feature_map.groupby("feature_group", dropna=False):
        group_features = [c for c in group_map["prepared_feature"].tolist() if c in X.columns]
        if len(group_features) < 2:
            continue
        group_pairs = correlation_pairs(X[group_features])
        if group_pairs.empty:
            continue
        group_pairs.insert(0, "feature_group", group_name)
        group_pairs.insert(1, "n_features_in_group", len(group_features))
        all_pairs.append(group_pairs)

    if not all_pairs:
        return pd.DataFrame(
            columns=[
                "feature_group",
                "n_features_in_group",
                "feature_1",
                "feature_2",
                "correlation",
                "abs_correlation",
            ]
        )


    return (
        pd.concat(all_pairs, ignore_index=True)
        .sort_values(["feature_group", "abs_correlation"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _feature_name_priority(name: str) -> float:
    """Return a deterministic interpretability score used for correlation pruning.

    The score is intentionally simple and transparent. It is not a model-based
    importance score; it only helps choose a representative when two or more
    features are highly correlated in feature_train.
    """

    s = str(name).lower()
    score = 0.0

    positive_patterns = [
        ("severity_score", 30.0),
        ("fault_count", 24.0),
        ("occurrence", 18.0),
        ("working_hours", 16.0),
        ("engine_running", 15.0),
        ("actual_work", 14.0),
        ("remaining_hours", 13.0),
        ("days_since", 12.0),
        ("count", 10.0),
        ("sum_", 8.0),
        ("max_", 8.0),
        ("mean_", 6.0),
        ("smr", 5.0),
    ]
    negative_patterns = [
        ("has_", -8.0),
        ("flag", -8.0),
        ("valid", -8.0),
        ("observed_day_count", -6.0),
        ("missing", -6.0),
        ("unknown", -4.0),
    ]

    for pattern, value in positive_patterns:
        if pattern in s:
            score += value
    for pattern, value in negative_patterns:
        if pattern in s:
            score += value

    return score


def prepared_feature_quality_table(
    X: pd.DataFrame,
    feature_map: pd.DataFrame,
    raw_inventory: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build transparent feature-quality scores for correlation pruning.

    The output is used only to choose/recommend which feature to keep when a
    correlation component or pair contains redundant features. The score favors
    lower raw missingness, more observed variation, and more interpretable source
    feature names. It does not use the target variable.
    """

    required_cols = {"prepared_feature", "source_feature", "feature_group"}
    if not required_cols.issubset(feature_map.columns):
        missing = sorted(required_cols - set(feature_map.columns))
        raise ValueError(f"feature_map is missing required columns: {missing}")

    fmap = feature_map[
        ["prepared_feature", "source_feature", "feature_group"]
    ].drop_duplicates("prepared_feature").copy()

    prep_diag = prepared_feature_diagnostics(X)[
        ["prepared_feature", "unique_count", "variance", "zero_rate", "is_constant_exact"]
    ].rename(
        columns={
            "unique_count": "prepared_unique_count",
            "variance": "prepared_variance",
            "zero_rate": "prepared_zero_rate",
        }
    )
    out = fmap.merge(prep_diag, on="prepared_feature", how="left")

    if raw_inventory is not None and not raw_inventory.empty and "feature" in raw_inventory.columns:
        raw_cols = [
            c
            for c in [
                "feature",
                "missing_rate",
                "unique_count",
                "variance",
                "is_numeric_raw",
            ]
            if c in raw_inventory.columns
        ]
        raw = raw_inventory[raw_cols].drop_duplicates("feature").copy()
        raw = raw.rename(
            columns={
                "feature": "source_feature",
                "missing_rate": "raw_missing_rate",
                "unique_count": "raw_unique_count",
                "variance": "raw_variance",
            }
        )
        out = out.merge(raw, on="source_feature", how="left")
    else:
        out["raw_missing_rate"] = np.nan
        out["raw_unique_count"] = np.nan
        out["raw_variance"] = np.nan
        out["is_numeric_raw"] = np.nan

    raw_missing = pd.to_numeric(out.get("raw_missing_rate"), errors="coerce").fillna(0.0)
    raw_unique = pd.to_numeric(out.get("raw_unique_count"), errors="coerce").fillna(0.0)
    prep_unique = pd.to_numeric(out.get("prepared_unique_count"), errors="coerce").fillna(0.0)
    prep_variance = pd.to_numeric(out.get("prepared_variance"), errors="coerce").fillna(0.0)

    name_priority = out.apply(
        lambda row: _feature_name_priority(row.get("source_feature", ""))
        + 0.25 * _feature_name_priority(row.get("prepared_feature", "")),
        axis=1,
    )

    out["feature_name_priority_score"] = name_priority.astype(float)
    out["correlation_pruning_score"] = (
        -100.0 * raw_missing
        + 2.0 * np.log1p(raw_unique)
        + 1.0 * np.log1p(prep_unique)
        + 0.01 * np.log1p(np.maximum(prep_variance, 0.0))
        + out["feature_name_priority_score"]
    )

    return out.sort_values(
        ["correlation_pruning_score", "prepared_feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _connected_components_from_pairs(pairs: pd.DataFrame) -> List[List[str]]:
    """Return connected components from feature_1/feature_2 pair rows."""

    adjacency: Dict[str, set] = {}
    if pairs.empty:
        return []

    for _, row in pairs.iterrows():
        f1 = str(row["feature_1"])
        f2 = str(row["feature_2"])
        adjacency.setdefault(f1, set()).add(f2)
        adjacency.setdefault(f2, set()).add(f1)

    components: List[List[str]] = []
    seen = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency.get(node, [])):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _feature_quality_lookup(feature_quality: pd.DataFrame) -> Dict[str, dict]:
    """Create a prepared_feature -> row dictionary from feature-quality output."""

    if feature_quality.empty or "prepared_feature" not in feature_quality.columns:
        return {}
    return feature_quality.set_index("prepared_feature").to_dict(orient="index")


def _pick_keep_feature(features: Sequence[str], feature_quality: pd.DataFrame) -> str:
    """Choose one representative feature from a correlated feature set."""

    lookup = _feature_quality_lookup(feature_quality)

    def sort_key(feature: str) -> tuple:
        q = lookup.get(feature, {})
        score = q.get("correlation_pruning_score", -np.inf)
        missing = q.get("raw_missing_rate", np.inf)
        prep_unique = q.get("prepared_unique_count", -np.inf)
        return (
            float(score) if pd.notna(score) else -np.inf,
            -float(missing) if pd.notna(missing) else -np.inf,
            float(prep_unique) if pd.notna(prep_unique) else -np.inf,
            str(feature),
        )

    return max(list(features), key=sort_key)


def _max_abs_corr_involving_feature(pairs: pd.DataFrame, feature: str) -> float:
    mask = (pairs["feature_1"] == feature) | (pairs["feature_2"] == feature)
    vals = pd.to_numeric(pairs.loc[mask, "abs_correlation"], errors="coerce")
    return float(vals.max()) if vals.notna().any() else np.nan


def _abs_corr_between_features(pairs: pd.DataFrame, feature_a: str, feature_b: str) -> float:
    mask = (
        ((pairs["feature_1"] == feature_a) & (pairs["feature_2"] == feature_b))
        | ((pairs["feature_1"] == feature_b) & (pairs["feature_2"] == feature_a))
    )
    vals = pd.to_numeric(pairs.loc[mask, "abs_correlation"], errors="coerce")
    return float(vals.max()) if vals.notna().any() else np.nan


def build_correlation_auto_prune_report(
    corr_pairs: pd.DataFrame,
    feature_quality: pd.DataFrame,
    auto_drop_threshold: float,
) -> pd.DataFrame:
    """Build auto-pruning decisions for abs(correlation) >= threshold.

    The function treats highly correlated pairs as a graph and keeps one
    representative per connected component. All other features in that component
    become auto-drop candidates. Only features appearing in pairs above the
    threshold can be auto-dropped.
    """

    columns = [
        "correlation_component_id",
        "component_size",
        "feature_group",
        "prepared_feature",
        "source_feature",
        "recommendation_action",
        "drop_confirmed",
        "kept_prepared_feature",
        "kept_source_feature",
        "max_abs_correlation_in_component",
        "max_abs_correlation_to_kept_feature",
        "correlation_pruning_score",
        "raw_missing_rate",
        "prepared_unique_count",
        "recommendation_reason",
    ]
    if corr_pairs.empty:
        return pd.DataFrame(columns=columns)

    high_pairs = corr_pairs[
        pd.to_numeric(corr_pairs["abs_correlation"], errors="coerce") >= float(auto_drop_threshold)
    ].copy()
    if high_pairs.empty:
        return pd.DataFrame(columns=columns)

    q_lookup = _feature_quality_lookup(feature_quality)
    rows = []
    for i, component in enumerate(_connected_components_from_pairs(high_pairs), start=1):
        component_pairs = high_pairs[
            high_pairs["feature_1"].isin(component) & high_pairs["feature_2"].isin(component)
        ].copy()
        kept = _pick_keep_feature(component, feature_quality)
        kept_source = q_lookup.get(kept, {}).get("source_feature")
        component_id = f"C{i:04d}"
        for feature in component:
            q = q_lookup.get(feature, {})
            is_keep = feature == kept
            rows.append(
                {
                    "correlation_component_id": component_id,
                    "component_size": len(component),
                    "feature_group": q.get("feature_group"),
                    "prepared_feature": feature,
                    "source_feature": q.get("source_feature"),
                    "recommendation_action": "keep" if is_keep else "drop",
                    "drop_confirmed": bool(not is_keep),
                    "kept_prepared_feature": kept,
                    "kept_source_feature": kept_source,
                    "max_abs_correlation_in_component": _max_abs_corr_involving_feature(component_pairs, feature),
                    "max_abs_correlation_to_kept_feature": np.nan if is_keep else _abs_corr_between_features(component_pairs, feature, kept),
                    "correlation_pruning_score": q.get("correlation_pruning_score"),
                    "raw_missing_rate": q.get("raw_missing_rate"),
                    "prepared_unique_count": q.get("prepared_unique_count"),
                    "recommendation_reason": (
                        f"Kept as representative of component with abs_corr >= {auto_drop_threshold:.2f}."
                        if is_keep
                        else f"Auto-drop: redundant with kept feature '{kept}' at abs_corr >= {auto_drop_threshold:.2f}."
                    ),
                }
            )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["correlation_component_id", "recommendation_action", "prepared_feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_correlation_manual_review_report(
    corr_pairs: pd.DataFrame,
    feature_quality: pd.DataFrame,
    review_threshold: float,
    auto_drop_threshold: float,
    auto_drop_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build pair-level manual review choices for 0.90 <= abs(corr) < 0.95.

    The default thresholds come from config.py. Each row represents one pairwise
    choice and includes a recommended drop feature, but drop_confirmed is False
    so the user can review before applying it downstream.
    """

    columns = [
        "manual_pair_id",
        "feature_group",
        "feature_1",
        "source_feature_1",
        "feature_1_pruning_score",
        "feature_1_raw_missing_rate",
        "feature_2",
        "source_feature_2",
        "feature_2_pruning_score",
        "feature_2_raw_missing_rate",
        "correlation",
        "abs_correlation",
        "recommended_keep_feature",
        "recommended_drop_feature",
        "recommendation_reason",
        "drop_confirmed",
        "confirmed_drop_feature",
        "review_notes",
    ]
    if corr_pairs.empty:
        return pd.DataFrame(columns=columns)

    abs_corr = pd.to_numeric(corr_pairs["abs_correlation"], errors="coerce")
    review_pairs = corr_pairs[
        (abs_corr >= float(review_threshold)) & (abs_corr < float(auto_drop_threshold))
    ].copy()
    if review_pairs.empty:
        return pd.DataFrame(columns=columns)

    q_lookup = _feature_quality_lookup(feature_quality)
    auto_drop_set = set(auto_drop_features or [])
    rows = []

    for i, row in enumerate(
        review_pairs.sort_values("abs_correlation", ascending=False).itertuples(index=False),
        start=1,
    ):
        f1 = str(getattr(row, "feature_1"))
        f2 = str(getattr(row, "feature_2"))
        q1 = q_lookup.get(f1, {})
        q2 = q_lookup.get(f2, {})
        s1 = q1.get("correlation_pruning_score", -np.inf)
        s2 = q2.get("correlation_pruning_score", -np.inf)

        if f1 in auto_drop_set and f2 not in auto_drop_set:
            keep_feature, drop_feature = f2, f1
            reason = "Recommended because this feature is already auto-dropped in an abs_corr >= auto threshold component."
        elif f2 in auto_drop_set and f1 not in auto_drop_set:
            keep_feature, drop_feature = f1, f2
            reason = "Recommended because this feature is already auto-dropped in an abs_corr >= auto threshold component."
        elif float(s1 if pd.notna(s1) else -np.inf) >= float(s2 if pd.notna(s2) else -np.inf):
            keep_feature, drop_feature = f1, f2
            reason = "Recommended drop has lower correlation-pruning score based on missingness, variation, and name heuristics."
        else:
            keep_feature, drop_feature = f2, f1
            reason = "Recommended drop has lower correlation-pruning score based on missingness, variation, and name heuristics."

        rows.append(
            {
                "manual_pair_id": f"M{i:05d}",
                "feature_group": getattr(row, "feature_group", None),
                "feature_1": f1,
                "source_feature_1": q1.get("source_feature"),
                "feature_1_pruning_score": q1.get("correlation_pruning_score"),
                "feature_1_raw_missing_rate": q1.get("raw_missing_rate"),
                "feature_2": f2,
                "source_feature_2": q2.get("source_feature"),
                "feature_2_pruning_score": q2.get("correlation_pruning_score"),
                "feature_2_raw_missing_rate": q2.get("raw_missing_rate"),
                "correlation": getattr(row, "correlation"),
                "abs_correlation": getattr(row, "abs_correlation"),
                "recommended_keep_feature": keep_feature,
                "recommended_drop_feature": drop_feature,
                "recommendation_reason": reason,
                "drop_confirmed": False,
                "confirmed_drop_feature": "",
                "review_notes": "Set drop_confirmed=True in the confirmed-drop template if you agree with this recommendation.",
            }
        )

    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def build_correlation_confirmed_drop_template(
    auto_prune_report: pd.DataFrame,
    manual_review_report: pd.DataFrame,
    feature_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Create the user-editable confirmed-drop template for downstream steps."""

    columns = [
        "prepared_feature",
        "source_feature",
        "feature_group",
        "drop_confirmed",
        "decision_source",
        "pair_or_component_count",
        "max_abs_correlation",
        "recommendation_reason",
        "review_notes",
    ]
    q_lookup = _feature_quality_lookup(feature_quality)
    rows = []

    if auto_prune_report is not None and not auto_prune_report.empty:
        auto_drops = auto_prune_report[
            auto_prune_report["recommendation_action"] == "drop"
        ].copy()
        for _, row in auto_drops.iterrows():
            feature = row["prepared_feature"]
            q = q_lookup.get(feature, {})
            rows.append(
                {
                    "prepared_feature": feature,
                    "source_feature": q.get("source_feature", row.get("source_feature")),
                    "feature_group": q.get("feature_group", row.get("feature_group")),
                    "drop_confirmed": True,
                    "decision_source": "auto_ge_auto_threshold",
                    "pair_or_component_count": 1,
                    "max_abs_correlation": row.get("max_abs_correlation_in_component"),
                    "recommendation_reason": row.get("recommendation_reason"),
                    "review_notes": "Auto-confirmed. Change drop_confirmed to False if you want to keep it.",
                }
            )

    if manual_review_report is not None and not manual_review_report.empty:
        manual = manual_review_report.copy()
        manual = manual[manual["recommended_drop_feature"].notna()]
        manual = manual[manual["recommended_drop_feature"].astype(str).str.len() > 0]
        if not manual.empty:
            grouped = (
                manual.groupby("recommended_drop_feature", dropna=False)
                .agg(
                    pair_or_component_count=("manual_pair_id", "count"),
                    max_abs_correlation=("abs_correlation", "max"),
                    example_keep_feature=("recommended_keep_feature", "first"),
                    recommendation_reason=("recommendation_reason", "first"),
                )
                .reset_index()
            )
            existing_features = {r["prepared_feature"] for r in rows}
            for _, row in grouped.iterrows():
                feature = row["recommended_drop_feature"]
                if feature in existing_features:
                    continue
                q = q_lookup.get(feature, {})
                rows.append(
                    {
                        "prepared_feature": feature,
                        "source_feature": q.get("source_feature"),
                        "feature_group": q.get("feature_group"),
                        "drop_confirmed": False,
                        "decision_source": "manual_recommended_review_090_to_lt_auto_threshold",
                        "pair_or_component_count": row.get("pair_or_component_count"),
                        "max_abs_correlation": row.get("max_abs_correlation"),
                        "recommendation_reason": (
                            str(row.get("recommendation_reason"))
                            + f" Example keep feature: {row.get('example_keep_feature')}."
                        ),
                        "review_notes": "Manual review needed. Change drop_confirmed to True if you agree.",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["drop_confirmed", "max_abs_correlation", "prepared_feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_correlation_pruning_reports(
    corr_pairs: pd.DataFrame,
    X: pd.DataFrame,
    feature_map: pd.DataFrame,
    raw_inventory: Optional[pd.DataFrame],
    review_threshold: float,
    auto_drop_threshold: float,
) -> Dict[str, pd.DataFrame]:
    """Build all correlation-pruning reports used by step 02."""

    feature_quality = prepared_feature_quality_table(
        X=X,
        feature_map=feature_map,
        raw_inventory=raw_inventory,
    )

    if corr_pairs.empty:
        correlated_pairs = corr_pairs.copy()
    else:
        correlated_pairs = corr_pairs[
            pd.to_numeric(corr_pairs["abs_correlation"], errors="coerce") >= float(review_threshold)
        ].copy()
        correlated_pairs = correlated_pairs.sort_values(
            ["abs_correlation", "feature_group", "feature_1", "feature_2"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

    auto_prune = build_correlation_auto_prune_report(
        corr_pairs=corr_pairs,
        feature_quality=feature_quality,
        auto_drop_threshold=auto_drop_threshold,
    )
    auto_drop_features = auto_prune.loc[
        auto_prune.get("recommendation_action", pd.Series(dtype=str)) == "drop",
        "prepared_feature",
    ].tolist() if not auto_prune.empty else []

    manual_review = build_correlation_manual_review_report(
        corr_pairs=corr_pairs,
        feature_quality=feature_quality,
        review_threshold=review_threshold,
        auto_drop_threshold=auto_drop_threshold,
        auto_drop_features=auto_drop_features,
    )
    confirmed_template = build_correlation_confirmed_drop_template(
        auto_prune_report=auto_prune,
        manual_review_report=manual_review,
        feature_quality=feature_quality,
    )

    return {
        "feature_quality": feature_quality,
        "correlated_pairs_ge_review_threshold": correlated_pairs,
        "auto_prune_report": auto_prune,
        "manual_review_report": manual_review,
        "confirmed_drop_template": confirmed_template,
    }


def _normalize_confirmation_value(value) -> bool:
    """Normalize editable CSV values such as True/yes/1/drop to bool."""

    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y", "drop", "confirmed"}


def load_confirmed_correlation_drop_features(path: Path) -> pd.DataFrame:
    """Load a user-edited correlation drop file.

    If a drop_confirmed column exists, only truthy rows are used. If the file is
    a plain one-column list without drop_confirmed, every listed feature is
    treated as confirmed. The output always contains a prepared_feature column.
    """

    columns = ["prepared_feature", "drop_confirmed", "decision_source", "load_source"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=columns)

    candidate_cols = [
        "prepared_feature",
        "drop_feature",
        "feature",
        "recommended_drop_feature",
        "confirmed_drop_feature",
    ]
    feature_col = next((c for c in candidate_cols if c in df.columns), None)
    if feature_col is None:
        raise ValueError(
            f"Confirmed correlation drop file does not contain one of these columns: {candidate_cols}. Path: {path}"
        )

    if "drop_confirmed" in df.columns:
        confirmed_mask = df["drop_confirmed"].apply(_normalize_confirmation_value)
    else:
        confirmed_mask = pd.Series(True, index=df.index)

    out = df.loc[confirmed_mask].copy()
    out["prepared_feature"] = out[feature_col].astype(str).str.strip()

    if "confirmed_drop_feature" in df.columns and feature_col != "confirmed_drop_feature":
        extra = df.loc[confirmed_mask, ["confirmed_drop_feature"]].copy()
        extra["prepared_feature"] = extra["confirmed_drop_feature"].astype(str).str.strip()
        extra = extra[extra["prepared_feature"].notna()]
        extra = extra[~extra["prepared_feature"].isin(["", "nan", "None"])]
        if not extra.empty:
            for col in out.columns:
                if col not in extra.columns:
                    extra[col] = np.nan
            out = pd.concat([out, extra[out.columns]], ignore_index=True)

    out = out[out["prepared_feature"].notna()]
    out = out[~out["prepared_feature"].isin(["", "nan", "None"])]
    out["drop_confirmed"] = True
    if "decision_source" not in out.columns:
        out["decision_source"] = "manual_confirmed"
    out["load_source"] = str(path)

    return out.drop_duplicates("prepared_feature").reset_index(drop=True)


def drop_prepared_features(
    prepared: PreparedData,
    drop_features: Sequence[str],
) -> Tuple[PreparedData, pd.DataFrame]:
    """Return a PreparedData object with selected prepared columns removed."""

    requested = sorted({str(f).strip() for f in drop_features if str(f).strip()})
    existing = [f for f in requested if f in prepared.X_feature_train.columns]
    missing = [f for f in requested if f not in prepared.X_feature_train.columns]

    audit_rows = []
    for feature in existing:
        audit_rows.append(
            {
                "prepared_feature": feature,
                "drop_requested": True,
                "drop_applied": True,
                "status": "dropped",
            }
        )
    for feature in missing:
        audit_rows.append(
            {
                "prepared_feature": feature,
                "drop_requested": True,
                "drop_applied": False,
                "status": "not_found_in_current_prepared_matrix",
            }
        )
    audit = pd.DataFrame(
        audit_rows,
        columns=["prepared_feature", "drop_requested", "drop_applied", "status"],
    )

    if not existing:
        return prepared, audit

    new_feature_map = prepared.feature_map[
        ~prepared.feature_map["prepared_feature"].isin(existing)
    ].copy().reset_index(drop=True)

    return PreparedData(
        X_feature_train=prepared.X_feature_train.drop(columns=existing),
        X_feature_holdout=prepared.X_feature_holdout.drop(columns=existing),
        X_validation=prepared.X_validation.drop(columns=existing),
        X_test=prepared.X_test.drop(columns=existing),
        feature_map=new_feature_map,
        numeric_input_cols=list(prepared.numeric_input_cols),
        categorical_input_cols=list(prepared.categorical_input_cols),
        preprocessor=prepared.preprocessor,
    ), audit


def _check_binary_target(y: pd.Series) -> Tuple[bool, str]:
    """
    Confirm that a target vector contains at least two classes.

    Many feature-ranking methods and model metrics are undefined when a split has
    only positives or only negatives. Instead of failing silently, the caller can
    return a warning report when this function returns False.
    """

    vals = sorted(pd.Series(y).dropna().unique().tolist())
    if len(vals) < 2:
        return False, f"Target has fewer than 2 classes in this split: {vals}"
    return True, "ok"


def run_anova(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Run ANOVA F-test feature ranking on feature_train.

    ANOVA is a univariate supervised filter method. It scores each prepared
    feature independently by how strongly the feature differs across target
    classes. It does not account for interactions or redundancy.
    """

    ok, msg = _check_binary_target(y)
    if not ok:
        return pd.DataFrame({"warning": [msg]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, p_values = f_classif(X, y)
    out = pd.DataFrame(
        {
            "prepared_feature": X.columns,
            "anova_f_score": scores,
            "anova_p_value": p_values,
        }
    )
    out["anova_f_score"] = out["anova_f_score"].replace([np.inf, -np.inf], np.nan)
    out["rank_anova"] = out["anova_f_score"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    return out.sort_values(["rank_anova", "prepared_feature"]).reset_index(drop=True)


def run_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    mode: str = "quantile_binned",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Run mutual-information feature ranking on feature_train.

    Mutual information is a univariate supervised filter method that can capture
    nonlinear dependency between a feature and the binary target. The default
    quantile-binned mode is faster and more stable for larger feature sets; the
    sklearn mode calls mutual_info_classif directly.
    """

    ok, msg = _check_binary_target(y)
    if not ok:
        return pd.DataFrame({"warning": [msg]})

    discrete_mask = np.array([X[c].nunique(dropna=True) <= 2 for c in X.columns], dtype=bool)

    if mode == "sklearn":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = mutual_info_classif(
                X,
                y,
                discrete_features=discrete_mask,
                random_state=random_state,
            )
        method = "sklearn_mutual_info_classif"
    elif mode == "quantile_binned":
        scores = []
        y_arr = pd.Series(y).astype(int).to_numpy()
        for col in X.columns:
            s = pd.Series(X[col]).replace([np.inf, -np.inf], np.nan).fillna(0)
            if s.nunique(dropna=True) <= 1:
                scores.append(0.0)
            elif s.nunique(dropna=True) <= 2:
                scores.append(float(mutual_info_score(y_arr, s.astype(str).to_numpy())))
            else:
                try:
                    bins = pd.qcut(
                        s.rank(method="first"),
                        q=min(n_bins, s.nunique()),
                        labels=False,
                        duplicates="drop",
                    )
                    scores.append(float(mutual_info_score(y_arr, bins.astype(str).to_numpy())))
                except Exception:
                    scores.append(np.nan)
        scores = np.asarray(scores, dtype=float)
        method = f"quantile_binned_{n_bins}"
    else:
        raise ValueError("MUTUAL_INFO_MODE must be either 'quantile_binned' or 'sklearn'.")

    out = pd.DataFrame(
        {
            "prepared_feature": X.columns,
            "mutual_info_score": scores,
            "is_discrete_for_mi": discrete_mask,
            "mutual_info_method": method,
        }
    )
    out["rank_mutual_info"] = out["mutual_info_score"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    return out.sort_values(["rank_mutual_info", "prepared_feature"]).reset_index(drop=True)


def run_chi2_minmax(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Run chi-squared feature ranking after min-max scaling prepared features.

    Chi-squared requires non-negative feature values. MinMaxScaler is fitted only
    on feature_train within this function to make numeric prepared features
    non-negative for the ranking calculation. This is a reporting transformation,
    not a model-training preprocessing step.
    """

    ok, msg = _check_binary_target(y)
    if not ok:
        return pd.DataFrame({"warning": [msg]})
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, p_values = chi2(X_scaled, y)
    out = pd.DataFrame(
        {
            "prepared_feature": X.columns,
            "chi2_score_minmax_scaled": scores,
            "chi2_p_value_minmax_scaled": p_values,
        }
    )
    out["rank_chi2"] = out["chi2_score_minmax_scaled"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    return out.sort_values(["rank_chi2", "prepared_feature"]).reset_index(drop=True)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute XGBoost scale_pos_weight from the training target distribution.

    The value is negative_count / positive_count, with safeguards for no-positive
    splits. This helps XGBoost account for imbalanced binary targets.
    """

    y = pd.Series(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        return 1.0
    return max(neg / pos, 1.0)


def train_xgboost_classifier(X: pd.DataFrame, y: pd.Series, xgb_params: dict):
    """
    Train an XGBoost binary classifier on feature_train.

    The function injects scale_pos_weight based on y, then fits the model using
    the configured XGB_PARAMS. It raises a clear error if the training target has
    fewer than two classes.
    """

    ok, msg = _check_binary_target(y)
    if not ok:
        raise ValueError(msg)
    from xgboost import XGBClassifier

    params = dict(xgb_params)
    params["scale_pos_weight"] = compute_scale_pos_weight(y)
    model = XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


def xgboost_importance(model, feature_names: Sequence[str]) -> pd.DataFrame:
    """
    Extract multiple built-in XGBoost feature-importance measures.

    The report includes weight, gain, cover, total_gain, and total_cover. These
    are model-based rankings from the temporary model trained on feature_train.
    They should be reviewed alongside filters, permutation, SHAP, and domain
    knowledge rather than used as the only selection criterion.
    """

    booster = model.get_booster()
    rows = pd.DataFrame({"prepared_feature": list(feature_names)})
    for imp_type in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        raw_scores = booster.get_score(importance_type=imp_type)
        rows[f"xgb_{imp_type}"] = rows["prepared_feature"].map(raw_scores).fillna(0.0)
    for col in [c for c in rows.columns if c.startswith("xgb_")]:
        rows[f"rank_{col}"] = rows[col].rank(ascending=False, method="min")
    return rows.sort_values(["xgb_total_gain", "xgb_gain"], ascending=False).reset_index(drop=True)


def threshold_free_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compute threshold-free holdout metrics for the temporary XGBoost model.

    Average precision, ROC-AUC, and log loss use predicted probabilities rather
    than a hard classification threshold. The metrics are reported on
    feature_selection_holdout for context, not as final model validation.
    """

    result = {
        "rows": int(len(y)),
        "positive_count": int((pd.Series(y) == 1).sum()),
        "positive_rate": float((pd.Series(y) == 1).mean()) if len(y) > 0 else None,
    }
    ok, msg = _check_binary_target(y)
    if not ok:
        result["warning"] = msg
        return result
    prob = model.predict_proba(X)[:, 1]
    result["average_precision"] = float(average_precision_score(y, prob))
    try:
        result["roc_auc"] = float(roc_auc_score(y, prob))
    except Exception as exc:
        result["roc_auc_warning"] = str(exc)
    try:
        result["log_loss"] = float(log_loss(y, prob, labels=[0, 1]))
    except Exception as exc:
        result["log_loss_warning"] = str(exc)
    return result


def make_permutation_scorer(scoring: str):
    """
    Convert the config scoring value into a sklearn-compatible scorer.

    The special value "f2" returns an F-beta scorer with beta=2 and
    zero_division=0. Other values are passed through to sklearn, so settings such
    as "average_precision" remain valid if you change the config later.
    """

    if scoring.lower() == "f2":
        return make_scorer(fbeta_score, beta=2, zero_division=0)
    return scoring


def run_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute permutation importance on feature_selection_holdout.

    For each prepared feature, sklearn repeatedly shuffles that feature and
    measures the drop in the configured score. With PERMUTATION_SCORING="f2",
    larger positive importance means the feature helps preserve F2 performance on
    the inner holdout. Negative values can occur when shuffling improves the
    score by chance.
    """

    ok, msg = _check_binary_target(y)
    if not ok:
        return pd.DataFrame({"warning": [msg]})
    scorer = make_permutation_scorer(scoring)
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    out = pd.DataFrame(
        {
            "prepared_feature": X.columns,
            "permutation_scoring": scoring,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )
    out["rank_permutation"] = out["permutation_importance_mean"].rank(
        ascending=False, method="min", na_option="bottom"
    )
    return out.sort_values(["rank_permutation", "prepared_feature"]).reset_index(drop=True)


def run_shap_importance(model, X: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    """
    Compute mean absolute SHAP importance on feature_selection_holdout.

    SHAP explains how each feature contributes to model predictions. For feature
    review, local SHAP values are aggregated into mean absolute SHAP per feature.
    The optional max_rows cap limits reporting cost on large datasets without
    changing model training.
    """

    try:
        import shap
    except Exception as exc:
        return pd.DataFrame({"warning": [f"SHAP import failed: {exc}"]})

    if len(X) == 0:
        return pd.DataFrame({"warning": ["No rows available for SHAP importance."]})

    X_sample = X
    if max_rows and len(X) > max_rows:
        X_sample = X.sample(n=max_rows, random_state=random_state)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_arr = shap_values[-1]
        else:
            shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, -1]
        mean_abs = np.abs(shap_arr).mean(axis=0)
        mean_signed = shap_arr.mean(axis=0)
        out = pd.DataFrame(
            {
                "prepared_feature": X.columns,
                "mean_abs_shap": mean_abs,
                "mean_signed_shap": mean_signed,
                "shap_rows_used": len(X_sample),
            }
        )
        out["rank_shap"] = out["mean_abs_shap"].rank(
            ascending=False, method="min", na_option="bottom"
        )
        return out.sort_values(["rank_shap", "prepared_feature"]).reset_index(drop=True)
    except Exception as exc:
        return pd.DataFrame({"warning": [f"SHAP calculation failed: {exc}"]})


def merge_with_feature_map(report: pd.DataFrame, feature_map: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw source-feature metadata to a prepared-feature report.

    Many reports are calculated at the prepared feature level. This helper joins
    the prepared-to-raw map so each report includes source_feature and
    feature_group, making review easier.
    """

    if "prepared_feature" not in report.columns:
        return report
    return report.merge(feature_map, on="prepared_feature", how="left")


def build_consensus_report(
    feature_map: pd.DataFrame,
    reports: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine ranking/score columns from all feature-selection methods.

    This report is for review only. It calculates an average rank across methods
    that produced rank columns, but it does not apply a selection threshold or
    label features as keep/drop.
    """

    out = feature_map.copy()
    for _name, report in reports.items():
        if "prepared_feature" not in report.columns:
            continue
        cols = ["prepared_feature"] + [
            c
            for c in report.columns
            if c.startswith("rank_")
            or c.endswith("score")
            or c.endswith("importance_mean")
            or c in {"mean_abs_shap", "xgb_total_gain", "xgb_gain"}
        ]
        tmp = report[cols].copy()
        out = out.merge(tmp, on="prepared_feature", how="left")
    rank_cols = [c for c in out.columns if c.startswith("rank_")]
    if rank_cols:
        out["mean_rank_across_available_methods"] = out[rank_cols].mean(axis=1, skipna=True)
        out["rank_method_count"] = out[rank_cols].notna().sum(axis=1)
        out["consensus_rank"] = out["mean_rank_across_available_methods"].rank(
            ascending=True, method="min", na_option="bottom"
        )
        out = out.sort_values(["consensus_rank", "prepared_feature"]).reset_index(drop=True)
    return out


def save_top_bar_plot(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    title: str,
    path: Path,
    top_n: int,
) -> None:
    """
    Save a horizontal top-N bar chart for quick visual review.

    The function is defensive: if the requested columns are missing or the table
    is empty, it simply skips the plot. This allows the run to complete even when
    a method returns a warning table because a split has only one target class.
    """

    if df.empty or feature_col not in df.columns or value_col not in df.columns:
        return
    plot_df = (
        df[[feature_col, value_col]]
        .dropna()
        .sort_values(value_col, ascending=False)
        .head(top_n)
    )
    if plot_df.empty:
        return
    import matplotlib.pyplot as plt

    height = max(5, min(18, 0.28 * len(plot_df) + 2))
    plt.figure(figsize=(12, height))
    plt.barh(plot_df[feature_col][::-1], plot_df[value_col][::-1])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_excel_report(tables: Dict[str, pd.DataFrame], path: Path) -> None:
    """
    Save selected CSV-style reports into one Excel workbook.

    Excel sheet names are sanitized and truncated to 31 characters. Timezone or
    datetime columns are converted to strings to avoid Excel writer issues.
    """

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in tables.items():
            if df is None:
                continue
            safe_sheet = re.sub(r"[^A-Za-z0-9_]+", "_", sheet_name)[:31]
            if not safe_sheet:
                safe_sheet = "sheet"
            df_to_write = df.copy()
            for col in df_to_write.columns:
                if pd.api.types.is_datetime64_any_dtype(df_to_write[col]):
                    df_to_write[col] = df_to_write[col].astype(str)
            df_to_write.to_excel(writer, sheet_name=safe_sheet, index=False)


def write_markdown_summary(
    path: Path,
    input_path: Path,
    split_summary: pd.DataFrame,
    output_files: List[str],
    notes: Sequence[str],
) -> None:
    """
    Write a concise human-readable markdown summary of the run.

    The summary is useful when reviewing outputs outside Python. It lists the
    input file, split summary, warnings/notes, and generated output files.
    """

    lines = []
    lines.append("# Feature Selection Analysis Summary")
    lines.append("")
    lines.append(f"Input data: `{input_path}`")
    lines.append("")
    lines.append("## Split summary")
    lines.append("")
    try:
        lines.append(split_summary.to_markdown(index=False))
    except Exception:
        lines.append(split_summary.to_string(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for file_name in output_files:
        lines.append(f"- `{file_name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
