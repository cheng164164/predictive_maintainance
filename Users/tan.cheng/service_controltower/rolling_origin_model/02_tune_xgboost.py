"""Tune XGBoost L1/L2 regularization and early stopping with rolling-origin folds.

The tuning design keeps all non-regularization model parameters fixed so the
comparison isolates reg_alpha (L1), reg_lambda (L2), and early stopping. Each
candidate is evaluated on the same purged rolling-origin folds. The reserved
calibration period in each fold is used for early stopping, probability
calibration, and F1 threshold selection; the forward validation period remains
untouched until scoring.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from xgboost import XGBClassifier

import config
from modeling_utils import (
    choose_configured_threshold,
    ensure_directories,
    evaluate_predictions,
    feature_list_for_variant,
    fit_probability_calibration,
    load_snapshot_dataframe,
    rolling_origin_split,
)


def _candidate_id(alpha: float, lambda_: float, early_stopping: int) -> str:
    """Create a stable identifier for one XGBoost tuning candidate."""
    def token(value: float) -> str:
        """Convert a parameter value into a stable filename-safe token."""
        return str(value).replace('.', 'p')
    return f"alpha_{token(alpha)}__lambda_{token(lambda_)}__es_{early_stopping}"


def _candidate_grid() -> list[dict]:
    """Generate the controlled L1, L2, and early-stopping search grid."""
    candidates = []
    for alpha, lambda_, early_stopping in itertools.product(
        config.XGB_TUNING_REG_ALPHA_VALUES,
        config.XGB_TUNING_REG_LAMBDA_VALUES,
        config.XGB_TUNING_EARLY_STOPPING_ROUNDS_VALUES,
    ):
        candidates.append(
            {
                'candidate_id': _candidate_id(alpha, lambda_, early_stopping),
                'reg_alpha': float(alpha),
                'reg_lambda': float(lambda_),
                'early_stopping_rounds': int(early_stopping),
            }
        )

    baseline = {
        'candidate_id': 'baseline_current_config',
        'reg_alpha': float(config.XGB_TUNING_BASELINE_PARAMS.get('reg_alpha', 0.0)),
        'reg_lambda': float(config.XGB_TUNING_BASELINE_PARAMS.get('reg_lambda', 0.0)),
        'early_stopping_rounds': int(
            config.XGB_TUNING_BASELINE_PARAMS.get('early_stopping_rounds', 0)
        ),
    }
    by_signature = {
        (row['reg_alpha'], row['reg_lambda'], row['early_stopping_rounds']): row
        for row in candidates
    }
    baseline_signature = (
        baseline['reg_alpha'],
        baseline['reg_lambda'],
        baseline['early_stopping_rounds'],
    )
    if baseline_signature not in by_signature:
        candidates.insert(0, baseline)
    else:
        existing = by_signature[baseline_signature]
        existing['is_baseline'] = True
        existing['baseline_alias'] = baseline['candidate_id']
    for row in candidates:
        row.setdefault('is_baseline', row['candidate_id'] == 'baseline_current_config')
    return candidates


def _build_model(candidate: dict) -> XGBClassifier:
    """Instantiate an XGBoost classifier from one candidate parameter set."""
    params = dict(config.XGB_TUNING_BASE_PARAMS)
    params.update(
        reg_alpha=float(candidate['reg_alpha']),
        reg_lambda=float(candidate['reg_lambda']),
        early_stopping_rounds=int(candidate['early_stopping_rounds']),
    )
    return XGBClassifier(
        **params,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_SEED,
    )


def _fit_one_fold(
    dataframe: pd.DataFrame,
    fold: tuple[str, str, str],
    variant: str,
    candidate: dict,
    keep_predictions: bool = False,
) -> tuple[dict, pd.DataFrame | None]:
    """Fit and evaluate one tuning candidate on one rolling-origin fold."""
    fold_name, fit, calibration, validation = rolling_origin_split(dataframe, fold)
    features, _ = feature_list_for_variant(dataframe, fit, variant)
    x_fit = fit[features].astype(float)
    y_fit = fit[config.TARGET_COLUMN].astype(int).to_numpy()
    x_cal = calibration[features].astype(float)
    y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
    x_val = validation[features].astype(float)
    y_val = validation[config.TARGET_COLUMN].astype(int).to_numpy()

    model = _build_model(candidate)
    started = time.time()
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_fit, y_fit), (x_cal, y_cal)],
        verbose=False,
    )
    elapsed = time.time() - started
    raw_cal = model.predict_proba(x_cal)[:, 1]
    raw_val = model.predict_proba(x_val)[:, 1]
    calibrator = fit_probability_calibration(y_cal, raw_cal)
    calibrated_cal = calibrator.apply(raw_cal)
    calibrated_val = calibrator.apply(raw_val)
    threshold, threshold_score, _ = choose_configured_threshold(y_cal, calibrated_cal)
    metrics = evaluate_predictions(y_val, raw_val, calibrated_val, threshold)

    evals_result = model.evals_result()
    train_metrics = evals_result.get('validation_0', {})
    cal_metrics = evals_result.get('validation_1', {})
    best_iteration = int(getattr(model, 'best_iteration', -1))
    best_index = max(0, best_iteration)
    last_index = max(0, len(next(iter(cal_metrics.values()), [])) - 1)

    row = {
        **candidate,
        'variant': variant,
        'fold': fold_name,
        'fit_rows': int(len(fit)),
        'calibration_rows': int(len(calibration)),
        'validation_rows': int(len(validation)),
        'validation_positive_rate': float(y_val.mean()),
        'n_features': int(len(features)),
        'best_iteration': best_iteration,
        'fit_seconds': elapsed,
        'operating_threshold': float(threshold),
        'calibration_threshold_score': float(threshold_score),
        **metrics,
    }
    for metric_name in ('aucpr', 'logloss'):
        train_values = train_metrics.get(metric_name, [])
        cal_values = cal_metrics.get(metric_name, [])
        if train_values and cal_values:
            idx = min(best_index, len(train_values) - 1, len(cal_values) - 1)
            row[f'train_{metric_name}_at_best'] = float(train_values[idx])
            row[f'calibration_{metric_name}_at_best'] = float(cal_values[idx])
            row[f'{metric_name}_generalization_gap_at_best'] = float(
                train_values[idx] - cal_values[idx]
            )
            row[f'calibration_{metric_name}_at_last'] = float(cal_values[-1])

    predictions = None
    if keep_predictions:
        predictions = validation[
            ['machine_key', 'snapshot_date', config.TARGET_COLUMN]
        ].copy()
        predictions['candidate_id'] = candidate['candidate_id']
        predictions['fold'] = fold_name
        predictions['raw_score'] = raw_val
        predictions['calibrated_probability'] = calibrated_val
        predictions['operating_threshold'] = threshold
        predictions['prediction'] = (calibrated_val >= threshold).astype('int8')
    return row, predictions


def _summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level tuning metrics into candidate-level statistics."""
    summary = (
        fold_metrics.groupby(
            [
                'candidate_id',
                'reg_alpha',
                'reg_lambda',
                'early_stopping_rounds',
                'is_baseline',
            ],
            as_index=False,
        )
        .agg(
            fold_count=('fold', 'nunique'),
            mean_average_precision=('average_precision', 'mean'),
            std_average_precision=('average_precision', 'std'),
            min_average_precision=('average_precision', 'min'),
            mean_roc_auc=('roc_auc', 'mean'),
            std_roc_auc=('roc_auc', 'std'),
            mean_f1=('f1', 'mean'),
            mean_f2=('f2', 'mean'),
            mean_precision=('precision', 'mean'),
            mean_recall=('recall', 'mean'),
            mean_brier=('brier', 'mean'),
            mean_log_loss=('log_loss', 'mean'),
            mean_precision_top10=('precision_top10', 'mean'),
            mean_recall_top10=('recall_top10', 'mean'),
            mean_best_iteration=('best_iteration', 'mean'),
            max_best_iteration=('best_iteration', 'max'),
            mean_fit_seconds=('fit_seconds', 'mean'),
        )
    )
    return summary.sort_values(
        ['mean_average_precision', 'mean_roc_auc', 'mean_f1', 'mean_brier'],
        ascending=[False, False, False, True],
        kind='mergesort',
    ).reset_index(drop=True)


def _pooled_metrics(predictions: pd.DataFrame) -> dict:
    """Calculate pooled out-of-fold classification and ranking metrics."""
    labels = predictions[config.TARGET_COLUMN].astype(int).to_numpy()
    raw = predictions['raw_score'].to_numpy(float)
    calibrated = predictions['calibrated_probability'].to_numpy(float)
    predicted = predictions['prediction'].astype(int).to_numpy()
    return {
        'pooled_rows': int(len(predictions)),
        'pooled_positive_rate': float(labels.mean()),
        'pooled_average_precision': float(average_precision_score(labels, raw)),
        'pooled_roc_auc': float(roc_auc_score(labels, raw)),
        'pooled_precision': float(precision_score(labels, predicted, zero_division=0)),
        'pooled_recall': float(recall_score(labels, predicted, zero_division=0)),
        'pooled_f1': float(f1_score(labels, predicted, zero_division=0)),
        'pooled_f2': float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        'pooled_brier': float(brier_score_loss(labels, calibrated)),
        'pooled_log_loss': float(log_loss(labels, calibrated)),
        'pooled_flagged_rate': float(predicted.mean()),
    }


def _comparison_table(
    summary: pd.DataFrame,
    pooled: pd.DataFrame,
    baseline_id: str,
    best_id: str,
) -> pd.DataFrame:
    """Build the baseline-versus-selected tuning comparison table."""
    summary_indexed = summary.set_index('candidate_id')
    pooled_indexed = pooled.set_index('candidate_id')
    rows = []
    for metric in [
        'mean_average_precision',
        'mean_roc_auc',
        'mean_f1',
        'mean_f2',
        'mean_precision',
        'mean_recall',
        'mean_brier',
        'mean_log_loss',
        'mean_precision_top10',
        'mean_recall_top10',
        'mean_best_iteration',
        'pooled_average_precision',
        'pooled_roc_auc',
        'pooled_f1',
        'pooled_f2',
        'pooled_precision',
        'pooled_recall',
        'pooled_brier',
        'pooled_log_loss',
    ]:
        source = pooled_indexed if metric.startswith('pooled_') else summary_indexed
        baseline = float(source.loc[baseline_id, metric])
        tuned = float(source.loc[best_id, metric])
        rows.append(
            {
                'metric': metric,
                'baseline': baseline,
                'tuned': tuned,
                'absolute_change': tuned - baseline,
                'relative_change_pct': (
                    100.0 * (tuned - baseline) / abs(baseline)
                    if baseline != 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run XGBoost regularization tuning and persist the selected parameters."""
    ensure_directories()
    output_dir = config.OUTPUT_DIR / config.XGB_TUNING_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = load_snapshot_dataframe()
    variant = config.XGB_TUNING_VARIANT
    candidates = _candidate_grid()
    print(
        f'XGBoost tuning: {len(candidates)} candidates x '
        f'{len(config.VALIDATION_FOLDS)} folds',
        flush=True,
    )

    fold_rows = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            f"[{candidate_index:02d}/{len(candidates):02d}] "
            f"alpha={candidate['reg_alpha']}, lambda={candidate['reg_lambda']}, "
            f"early_stopping={candidate['early_stopping_rounds']}",
            flush=True,
        )
        for fold in config.VALIDATION_FOLDS:
            row, _ = _fit_one_fold(dataframe, fold, variant, candidate)
            fold_rows.append(row)
            print(
                f"  {row['fold']}: AP={row['average_precision']:.4f}, "
                f"AUC={row['roc_auc']:.4f}, F1={row['f1']:.4f}, "
                f"best_iter={row['best_iteration']}",
                flush=True,
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = _summarize(fold_metrics)
    best_row = summary.iloc[0].to_dict()
    best_id = str(best_row['candidate_id'])
    baseline_candidates = summary[summary['is_baseline'].astype(bool)]
    if baseline_candidates.empty:
        baseline_signature = (
            float(config.XGB_TUNING_BASELINE_PARAMS.get('reg_alpha', 0.0)),
            float(config.XGB_TUNING_BASELINE_PARAMS.get('reg_lambda', 0.0)),
            int(config.XGB_TUNING_BASELINE_PARAMS.get('early_stopping_rounds', 0)),
        )
        mask = (
            summary['reg_alpha'].eq(baseline_signature[0])
            & summary['reg_lambda'].eq(baseline_signature[1])
            & summary['early_stopping_rounds'].eq(baseline_signature[2])
        )
        baseline_id = str(summary.loc[mask, 'candidate_id'].iloc[0])
    else:
        baseline_id = str(baseline_candidates.iloc[0]['candidate_id'])

    selected_candidates = {
        candidate['candidate_id']: candidate
        for candidate in candidates
        if candidate['candidate_id'] in {baseline_id, best_id}
    }
    prediction_frames = []
    for candidate_id in [baseline_id, best_id]:
        candidate = selected_candidates[candidate_id]
        for fold in config.VALIDATION_FOLDS:
            _, predictions = _fit_one_fold(
                dataframe, fold, variant, candidate, keep_predictions=True
            )
            prediction_frames.append(predictions)
    oof = pd.concat(prediction_frames, ignore_index=True)
    pooled_rows = []
    for candidate_id, group in oof.groupby('candidate_id', sort=False):
        pooled_rows.append({'candidate_id': candidate_id, **_pooled_metrics(group)})
    pooled = pd.DataFrame(pooled_rows)
    comparison = _comparison_table(summary, pooled, baseline_id, best_id)

    best_params = dict(config.XGB_TUNING_BASE_PARAMS)
    best_params.update(
        reg_alpha=float(best_row['reg_alpha']),
        reg_lambda=float(best_row['reg_lambda']),
        early_stopping_rounds=int(best_row['early_stopping_rounds']),
    )
    payload = {
        'selected_candidate_id': best_id,
        'selection_objective': 'mean_average_precision',
        'variant': variant,
        'source_data_scope': 'configured_development_dataset',
        'params': best_params,
        'baseline_candidate_id': baseline_id,
        'baseline_params': dict(config.XGB_TUNING_BASELINE_PARAMS),
        'selected_summary': {
            key: (value.item() if hasattr(value, 'item') else value)
            for key, value in best_row.items()
        },
    }

    fold_metrics.to_csv(output_dir / 'xgboost_tuning_fold_metrics.csv', index=False)
    summary.to_csv(output_dir / 'xgboost_tuning_candidate_summary.csv', index=False)
    pooled.to_csv(output_dir / 'xgboost_tuning_pooled_metrics.csv', index=False)
    comparison.to_csv(output_dir / 'xgboost_tuning_baseline_vs_tuned.csv', index=False)
    oof.to_csv(
        output_dir / 'xgboost_tuning_baseline_and_best_oof_predictions.csv.gz',
        index=False,
        compression='gzip',
    )
    (config.MODEL_DIR / 'xgboost_tuned_params.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )

    print('\nSelected candidate')
    print(json.dumps(payload, indent=2))
    print('\nBaseline versus tuned')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    main()
