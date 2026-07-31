"""Shared modeling, calibration, validation, ranking, and scoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config


BASE_FAULT_FEATURES = [
    'fault_count_90d',
    'fault_serious_count_90d',
    'fault_severity_weighted_90d',
    'fault_days_since_last',
    'fault_serious_30d',
    'fault_severity_recency_90d',
    'fault_count_7d',
    'fault_velocity_ratio',
    'fault_distinct_serious_systems',
    'fault_single_system_concentration',
    'fault_hst_serious_flag',
    'fault_l04_count_90d',
    'fault_l03_count_90d',
    'fault_evidence_weighted_90d',
    'fault_distinct_codes_90d',
]
BASE_FLUID_FEATURES = [
    'fluid_current_severity',
    'fluid_recent_abnormal_count_90d',
    'fluid_contaminant_flag',
    'fluid_worsening_trend',
    'fluid_sample_staleness_days',
]
BASE_PM_FEATURES = [
    'pm_days_since_last_reset',
    'pm_overdue_count_90d',
]
BASE_STATIC_FEATURES = BASE_FAULT_FEATURES + BASE_FLUID_FEATURES + BASE_PM_FEATURES
HISTORY_FEATURES = list(config.HISTORY_FEATURES)
LIVENESS_FEATURES = [
    'smr_delta_30d',
    'engine_hours_30d',
    'work_hours_30d',
    'active_days_30d',
    'days_since_last_sensor_reading',
    'operation_history_days',
]
TARGET_AND_LEAKAGE_COLUMNS = {
    config.TARGET_COLUMN,
    config.FUTURE_TARGET_COUNT_COLUMN,
    'target_warranty_0_90',
    'target_service_repair_0_90',
    'target_tsi_0_90',
}


@dataclass
class PlattCalibration:
    coefficient: float
    intercept: float

    def apply(self, probabilities: Sequence[float]) -> np.ndarray:
        raw = np.asarray(probabilities, dtype=float)
        eps = 1e-6
        clipped = np.clip(raw, eps, 1.0 - eps)
        logit = np.log(clipped / (1.0 - clipped))
        calibrated_logit = self.coefficient * logit + self.intercept
        calibrated_logit = np.clip(calibrated_logit, -40, 40)
        return 1.0 / (1.0 + np.exp(-calibrated_logit))


def ensure_directories() -> None:
    for directory in [config.OUTPUT_DIR, config.MODEL_DIR, config.CHART_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_snapshot_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load a snapshot table from pickle when present, otherwise compressed CSV."""
    if path is None:
        pickle_path = config.OUTPUT_DIR / 'snapshot_features.pkl'
        csv_path = config.OUTPUT_DIR / 'snapshot_features.csv.gz'
        if pickle_path.exists():
            path = pickle_path
        elif csv_path.exists():
            path = csv_path
        else:
            raise FileNotFoundError(
                'Snapshot dataset not found. Run 01_build_snapshot_dataset.py first.'
            )
    path = Path(path)
    if path.suffix == '.pkl':
        dataframe = pd.read_pickle(path)
    else:
        dataframe = pd.read_csv(path, low_memory=False)
    dataframe['snapshot_date'] = pd.to_datetime(
        dataframe['snapshot_date'], errors='raise'
    )
    code_columns = fault_code_feature_columns(dataframe)
    if code_columns:
        dataframe[code_columns] = dataframe[code_columns].fillna(0)
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    return dataframe


def load_latest_snapshot_dataframe() -> pd.DataFrame:
    pickle_path = config.OUTPUT_DIR / 'latest_snapshot_features.pkl'
    csv_path = config.OUTPUT_DIR / 'latest_snapshot_features.csv.gz'
    if pickle_path.exists():
        return load_snapshot_dataframe(pickle_path)
    if csv_path.exists():
        return load_snapshot_dataframe(csv_path)
    raise FileNotFoundError(
        'Latest snapshot dataset not found. Run 01_build_snapshot_dataset.py first.'
    )


def fault_code_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return sorted(
        column for column in dataframe.columns if column.startswith('fault_code_')
    )


def select_top_failure_codes(
    fit_dataframe: pd.DataFrame,
    count: int | None = None,
    minimum_support: int = 30,
) -> tuple[list[str], list[dict]]:
    """Select label-correlated code features using fit data only.

    Candidate code columns are generated without the target. Their supervised ranking
    is repeated independently inside each fold to avoid validation leakage.
    """
    count = count or config.TOP_FAILURE_CODE_FEATURE_COUNT
    target = fit_dataframe[config.TARGET_COLUMN].astype(int).to_numpy()
    scored: list[tuple[float, float, int, str, float, float]] = []
    for column in fault_code_feature_columns(fit_dataframe):
        values = fit_dataframe[column].fillna(0).to_numpy()
        present = values > 0
        support = int(present.sum())
        if support < minimum_support or int((~present).sum()) < minimum_support:
            continue
        positive_rate_present = float(target[present].mean())
        positive_rate_absent = float(target[~present].mean())
        lift = positive_rate_present - positive_rate_absent
        ranking_score = lift * math.sqrt(support)
        scored.append(
            (
                ranking_score,
                lift,
                support,
                column,
                positive_rate_present,
                positive_rate_absent,
            )
        )
    scored.sort(reverse=True)
    chosen = [row[3] for row in scored[:count]]
    detail = [
        {
            'feature': row[3],
            'selection_score': row[0],
            'absolute_rate_difference': row[1],
            'support_rows': row[2],
            'positive_rate_when_present': row[4],
            'positive_rate_when_absent': row[5],
        }
        for row in scored[:count]
    ]
    return chosen, detail


def feature_list_for_variant(
    dataframe: pd.DataFrame,
    fit_dataframe: pd.DataFrame,
    variant: str,
) -> tuple[list[str], list[dict]]:
    top_codes, top_code_detail = select_top_failure_codes(fit_dataframe)
    if variant == 'base27':
        features = BASE_STATIC_FEATURES + top_codes
    elif variant == 'base27_plus_history':
        features = BASE_STATIC_FEATURES + top_codes + HISTORY_FEATURES
    elif variant == 'base27_plus_history_liveness':
        features = (
            BASE_STATIC_FEATURES + top_codes + HISTORY_FEATURES + LIVENESS_FEATURES
        )
    elif variant == 'enhanced_condition':
        numeric = dataframe.select_dtypes(include=[np.number]).columns
        features = [
            column
            for column in numeric
            if column not in TARGET_AND_LEAKAGE_COLUMNS
            and column not in HISTORY_FEATURES
        ]
    elif variant == 'enhanced_plus_history':
        numeric = dataframe.select_dtypes(include=[np.number]).columns
        features = [
            column for column in numeric if column not in TARGET_AND_LEAKAGE_COLUMNS
        ]
    else:
        raise ValueError(f'Unknown model variant: {variant}')

    # Remove duplicates while preserving order and remove columns that contain no
    # usable variation in the actual fit data.
    unique_features = list(dict.fromkeys(features))
    usable = [
        column
        for column in unique_features
        if column in dataframe.columns
        and fit_dataframe[column].notna().any()
        and fit_dataframe[column].nunique(dropna=False) > 1
    ]
    return usable, top_code_detail


def rolling_origin_split(
    dataframe: pd.DataFrame,
    fold: tuple[str, str, str],
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fit, calibration, and validation sets with a 90-day purge."""
    fold_name, validation_start, validation_end = fold
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)
    fit_pool_end = validation_start_ts - pd.Timedelta(days=config.PURGE_DAYS)
    train_pool = dataframe[dataframe['snapshot_date'] < fit_pool_end].copy()
    validation = dataframe[
        dataframe['snapshot_date'].ge(validation_start_ts)
        & dataframe['snapshot_date'].lt(validation_end_ts)
    ].copy()
    available_dates = sorted(train_pool['snapshot_date'].dropna().unique())
    if len(available_dates) <= config.CALIBRATION_MONTHS:
        raise ValueError(f'Insufficient training dates for fold {fold_name}')
    calibration_dates = available_dates[-config.CALIBRATION_MONTHS :]
    calibration = train_pool[
        train_pool['snapshot_date'].isin(calibration_dates)
    ].copy()
    fit = train_pool[
        ~train_pool['snapshot_date'].isin(calibration_dates)
    ].copy()
    return fold_name, fit, calibration, validation


def _xgboost_model() -> XGBClassifier:
    parameters = dict(config.XGB_PARAMS)
    return XGBClassifier(
        **parameters,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_SEED,
    )


def make_algorithm(name: str):
    if name == 'xgboost':
        return _xgboost_model()
    if name == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError('lightgbm is not installed') from exc
        return LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=5,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            objective='binary',
            random_state=config.RANDOM_SEED,
            n_jobs=config.N_JOBS,
            verbosity=-1,
        )
    if name == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError('catboost is not installed') from exc
        return CatBoostClassifier(
            iterations=1500,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            l2_leaf_reg=5,
            random_seed=config.RANDOM_SEED,
            thread_count=config.N_JOBS,
            verbose=False,
            od_type='Iter',
            od_wait=75,
            allow_writing_files=False,
        )
    if name == 'hist_gradient_boosting':
        return HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.06,
            max_leaf_nodes=31,
            max_depth=6,
            min_samples_leaf=40,
            l2_regularization=2.0,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            random_state=config.RANDOM_SEED,
        )
    if name == 'logistic_regression':
        return Pipeline(
            steps=[
                (
                    'impute',
                    SimpleImputer(strategy='median', add_indicator=True),
                ),
                ('scale', StandardScaler()),
                (
                    'model',
                    LogisticRegression(
                        C=0.2,
                        penalty='l2',
                        solver='liblinear',
                        max_iter=500,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f'Unknown algorithm: {name}')


def fit_algorithm(
    name: str,
    model,
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_calibration: pd.DataFrame,
    y_calibration: np.ndarray,
):
    """Fit one algorithm and return the fitted model and effective iteration."""
    if name == 'xgboost':
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_calibration, y_calibration)],
            verbose=False,
        )
        return model, int(getattr(model, 'best_iteration', -1))
    if name == 'lightgbm':
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_calibration, y_calibration)],
            eval_metric='average_precision',
            callbacks=[early_stopping(75, verbose=False), log_evaluation(0)],
        )
        return model, int(getattr(model, 'best_iteration_', -1))
    if name == 'catboost':
        model.fit(
            x_fit,
            y_fit,
            eval_set=(x_calibration, y_calibration),
            verbose=False,
        )
        return model, int(model.get_best_iteration())
    model.fit(x_fit, y_fit)
    return model, int(getattr(model, 'n_iter_', -1))


def fit_platt_calibration(
    labels: Sequence[int], probabilities: Sequence[float]
) -> PlattCalibration:
    labels_array = np.asarray(labels, dtype=int)
    if np.unique(labels_array).size < 2:
        return PlattCalibration(1.0, 0.0)
    eps = 1e-6
    clipped = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
    calibrator.fit(logits, labels_array)
    return PlattCalibration(
        coefficient=float(calibrator.coef_[0, 0]),
        intercept=float(calibrator.intercept_[0]),
    )


def choose_f2_threshold(
    labels: Sequence[int], calibrated_probabilities: Sequence[float]
) -> tuple[float, float]:
    labels_array = np.asarray(labels, dtype=int)
    probabilities = np.asarray(calibrated_probabilities, dtype=float)
    if np.unique(labels_array).size < 2:
        return 0.5, 0.0
    candidates = np.unique(np.quantile(probabilities, np.linspace(0, 1, 401)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = fbeta_score(
            labels_array,
            probabilities >= threshold,
            beta=2,
            zero_division=0,
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def top_k_metrics(
    labels: Sequence[int], scores: Sequence[float], top_fraction: float
) -> dict:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    review_count = max(1, int(np.ceil(len(labels_array) * top_fraction)))
    selected = np.argpartition(-scores_array, review_count - 1)[:review_count]
    true_positives = int(labels_array[selected].sum())
    precision = true_positives / review_count
    recall = true_positives / max(1, int(labels_array.sum()))
    base_rate = float(labels_array.mean())
    return {
        'top_pct': top_fraction,
        'reviewed_rows': review_count,
        'precision': precision,
        'recall': recall,
        'lift_vs_base': precision / max(base_rate, 1e-12),
    }


def evaluate_predictions(
    labels: Sequence[int],
    raw_scores: Sequence[float],
    calibrated_probabilities: Sequence[float],
    threshold: float,
) -> dict:
    labels_array = np.asarray(labels, dtype=int)
    raw_scores_array = np.asarray(raw_scores, dtype=float)
    calibrated_array = np.asarray(calibrated_probabilities, dtype=float)
    predictions = calibrated_array >= threshold
    top10 = top_k_metrics(labels_array, raw_scores_array, 0.10)
    return {
        'roc_auc': float(roc_auc_score(labels_array, raw_scores_array)),
        'average_precision': float(
            average_precision_score(labels_array, raw_scores_array)
        ),
        'precision': float(
            precision_score(labels_array, predictions, zero_division=0)
        ),
        'recall': float(recall_score(labels_array, predictions, zero_division=0)),
        'f1': float(f1_score(labels_array, predictions, zero_division=0)),
        'f2': float(
            fbeta_score(labels_array, predictions, beta=2, zero_division=0)
        ),
        'flagged_rate': float(predictions.mean()),
        'brier': float(brier_score_loss(labels_array, calibrated_array)),
        'log_loss': float(log_loss(labels_array, calibrated_array)),
        'precision_top10': float(top10['precision']),
        'recall_top10': float(top10['recall']),
        'lift_top10': float(top10['lift_vs_base']),
    }


def summarize_oof_predictions(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    grouping_column: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name, group in predictions.groupby(grouping_column, sort=False):
        labels = group[config.TARGET_COLUMN].astype(int).to_numpy()
        raw_scores = group['score'].to_numpy()
        calibrated = group['calibrated_probability'].to_numpy()
        predicted = group['prediction'].astype(int).to_numpy()
        group_folds = fold_metrics[fold_metrics[grouping_column].eq(group_name)]
        top10 = top_k_metrics(labels, raw_scores, 0.10)
        rows.append(
            {
                grouping_column: group_name,
                'mean_roc_auc': float(group_folds['roc_auc'].mean()),
                'std_roc_auc': float(group_folds['roc_auc'].std(ddof=1)),
                'min_fold_auc': float(group_folds['roc_auc'].min()),
                'latest_fold_auc': float(group_folds.iloc[-1]['roc_auc']),
                'pooled_roc_auc': float(roc_auc_score(labels, raw_scores)),
                'pooled_average_precision': float(
                    average_precision_score(labels, raw_scores)
                ),
                'pooled_precision': float(
                    precision_score(labels, predicted, zero_division=0)
                ),
                'pooled_recall': float(
                    recall_score(labels, predicted, zero_division=0)
                ),
                'pooled_f1': float(f1_score(labels, predicted, zero_division=0)),
                'pooled_f2': float(
                    fbeta_score(labels, predicted, beta=2, zero_division=0)
                ),
                'pooled_brier': float(brier_score_loss(labels, calibrated)),
                'pooled_log_loss': float(log_loss(labels, calibrated)),
                'precision_top10': float(top10['precision']),
                'recall_top10': float(top10['recall']),
                'lift_top10': float(top10['lift_vs_base']),
            }
        )
    return pd.DataFrame(rows).sort_values('mean_roc_auc', ascending=False)


def risk_decile_table(
    predictions: pd.DataFrame, grouping_column: str = 'variant'
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name, group in predictions.groupby(grouping_column, sort=False):
        scores = group['score'].to_numpy()
        labels = group[config.TARGET_COLUMN].astype(int).to_numpy()
        ranks = pd.Series(scores).rank(method='first', pct=True)
        deciles = np.ceil(ranks * 10).clip(1, 10).astype(int).to_numpy()
        base_rate = labels.mean()
        for decile in range(1, 11):
            mask = deciles == decile
            failure_rate = float(labels[mask].mean())
            rows.append(
                {
                    grouping_column: group_name,
                    'risk_decile': decile,
                    'rows': int(mask.sum()),
                    'failures': int(labels[mask].sum()),
                    'failure_rate': failure_rate,
                    'lift_vs_base': failure_rate / max(base_rate, 1e-12),
                }
            )
    return pd.DataFrame(rows)


def top_k_table(
    predictions: pd.DataFrame,
    grouping_column: str = 'variant',
    fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_name, group in predictions.groupby(grouping_column, sort=False):
        labels = group[config.TARGET_COLUMN].astype(int).to_numpy()
        scores = group['score'].to_numpy()
        for fraction in fractions:
            row = top_k_metrics(labels, scores, fraction)
            row[grouping_column] = group_name
            rows.append(row)
    columns = [grouping_column, 'top_pct', 'reviewed_rows', 'precision', 'recall', 'lift_vs_base']
    return pd.DataFrame(rows)[columns]


def friendly_feature_name(column: str) -> str:
    history_noun = config.TARGET_HISTORY_NOUN
    exact = {
        'days_since_prior_target_event': f'days since prior {history_noun}',
        'prior_target_event_count_90d': f'prior {history_noun} in 90d',
        'prior_target_event_count_365d': f'prior {history_noun} in 365d',
        'prior_target_event_count_730d': f'prior {history_noun} in 730d',
        'pm_days_since_last_reset': 'PM: days since last reset',
        'pm_overdue_count_90d': 'PM: overdue count in 90d',
        'fluid_sample_staleness_days': 'fluid: sample staleness days',
        'fluid_current_severity': 'fluid: current severity',
    }
    if column in exact:
        return exact[column]
    result = column
    if result.startswith('fault_code_'):
        code = result.removeprefix('fault_code_').removesuffix('_90d')
        return f'fault code {code.replace("_", " ")} in 90d'
    replacements = [
        ('fault_', 'fault: '),
        ('fluid_', 'fluid: '),
        ('pm_', 'PM: '),
        ('_90d', ' in 90d'),
        ('_365d', ' in 365d'),
        ('_730d', ' in 730d'),
        ('_', ' '),
    ]
    for old, new_value in replacements:
        result = result.replace(old, new_value)
    return result.strip()


def _series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors='coerce').fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def add_reconstructed_risk_tiers(scores: pd.DataFrame) -> pd.DataFrame:
    """Add a conservative approximation of the note's evidence-based tier ladder.

    Exact component degradation gates and the original machine master were not
    available, so this field is explicitly named ``risk_level_reconstructed`` and
    should not be treated as a validated reproduction of the production tier logic.
    """
    result = scores.copy()
    strong_fault = (
        _series(result, 'fault_l04_count_90d').ge(1)
        | _series(result, 'fault_serious_30d').ge(3)
    )
    strong_fluid = (
        _series(result, 'fluid_current_severity').ge(2)
        | _series(result, 'fluid_contaminant_flag').ge(1)
    )
    overdue = _series(result, 'pm_overdue_count_90d').ge(1)
    recent_history = _series(
        result, 'prior_target_event_count_365d'
    ).ge(1)
    evidence_count = (
        strong_fault.astype(int)
        + strong_fluid.astype(int)
        + overdue.astype(int)
        + recent_history.astype(int)
    )
    corroborated = evidence_count.ge(2)
    severe_crescendo = (
        _series(result, 'fault_l04_count_90d').ge(3)
        | (
            _series(result, 'fault_l04_count_90d').ge(1)
            & _series(result, 'fault_serious_30d').ge(3)
            & _series(result, 'has_accelerating_faults').ge(1)
        )
    )
    down = (
        _series(result, 'operation_history_days').ge(30)
        & _series(result, 'days_since_last_sensor_reading', 366).le(21)
        & _series(result, 'smr_delta_30d').lt(5)
        & severe_crescendo
    )
    concentrated_l03 = (
        _series(result, 'fault_l03_count_90d').ge(3)
        & _series(result, 'fault_single_system_concentration').ge(0.70)
    )
    high = (
        _series(result, 'fault_l04_count_90d').ge(3)
        | concentrated_l03
        | (result['risk_index'].ge(90) & corroborated)
    )
    medium = (
        (result['risk_index'].ge(55) & corroborated)
        | strong_fault
        | strong_fluid
        | overdue
    )
    result['has_current_condition_evidence'] = (
        strong_fault | strong_fluid | overdue
    ).astype('int8')
    result['corroborated_evidence'] = corroborated.astype('int8')
    result['operating_status_reconstructed'] = np.where(down, 'DOWN', 'ACTIVE')
    result['risk_level_reconstructed'] = np.select(
        [down, high, medium],
        ['CRITICAL', 'HIGH', 'MEDIUM'],
        default='LOW',
    )
    return result
