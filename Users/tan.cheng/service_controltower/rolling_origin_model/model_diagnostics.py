"""XGBoost diagnostics for final-model explainability and overfit review."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import config


def _booster_iteration_range(model) -> tuple[int, int]:
    """Return the safe tree iteration range used for final-model diagnostics."""
    best_iteration = int(getattr(model, 'best_iteration', -1))
    if best_iteration >= 0:
        return (0, best_iteration + 1)
    return (0, 0)


def save_feature_importance(model, features: list[str], variant: str) -> pd.DataFrame:
    """Save multiple native XGBoost importance definitions and a gain plot."""
    booster = model.get_booster()
    importance_types = ('gain', 'total_gain', 'weight', 'cover', 'total_cover')
    data = {'feature': features}
    for importance_type in importance_types:
        scores = booster.get_score(importance_type=importance_type)
        values = np.asarray([float(scores.get(feature, 0.0)) for feature in features])
        data[f'importance_{importance_type}'] = values
        total = float(values.sum())
        data[f'importance_{importance_type}_normalized'] = (
            values / total if total > 0 else np.zeros_like(values)
        )
    importance = pd.DataFrame(data).sort_values(
        ['importance_gain', 'importance_weight'], ascending=[False, False]
    )
    csv_path = config.MODEL_DIR / f'feature_importance_{variant}.csv'
    importance.to_csv(csv_path, index=False)

    top_n = int(getattr(config, 'FEATURE_IMPORTANCE_TOP_N', 25))
    plot_data = importance.head(top_n).sort_values('importance_gain')
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot_data))))
    ax.barh(plot_data['feature'], plot_data['importance_gain_normalized'])
    ax.set_xlabel('Normalized gain importance')
    ax.set_ylabel('Feature')
    ax.set_title(f'XGBoost Feature Importance - {variant}')
    fig.tight_layout()
    fig.savefig(
        config.CHART_DIR / f'feature_importance_gain_{variant}.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(fig)
    return importance


def _sample_for_shap(
    dataframe: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a deterministic representative sample for SHAP calculations."""
    sample_size = min(int(getattr(config, 'SHAP_SAMPLE_SIZE', 5000)), len(dataframe))
    if sample_size <= 0:
        raise ValueError('SHAP_SAMPLE_SIZE must be positive.')
    if len(dataframe) > sample_size:
        sampled = dataframe.sample(
            n=sample_size,
            random_state=int(config.RANDOM_SEED),
            replace=False,
        ).sort_index()
    else:
        sampled = dataframe.copy()
    return sampled, sampled[features].astype(float)


def save_shap_diagnostics(
    model,
    reference_dataframe: pd.DataFrame,
    features: list[str],
    variant: str,
) -> pd.DataFrame:
    """Calculate exact TreeSHAP contributions and save tables and plots."""
    sampled, x_sample = _sample_for_shap(reference_dataframe, features)
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(x_sample, feature_names=features)
    iteration_range = _booster_iteration_range(model)
    kwargs = {'pred_contribs': True}
    if iteration_range[1] > 0:
        kwargs['iteration_range'] = iteration_range
    contributions = booster.predict(dmatrix, **kwargs)
    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]

    summary = pd.DataFrame(
        {
            'feature': features,
            'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),
            'mean_signed_shap': np.mean(shap_values, axis=0),
            'median_abs_shap': np.median(np.abs(shap_values), axis=0),
            'p90_abs_shap': np.quantile(np.abs(shap_values), 0.90, axis=0),
            'positive_shap_rate': np.mean(shap_values > 0, axis=0),
        }
    ).sort_values('mean_abs_shap', ascending=False)
    total = float(summary['mean_abs_shap'].sum())
    summary['mean_abs_shap_normalized'] = (
        summary['mean_abs_shap'] / total if total > 0 else 0.0
    )
    summary.to_csv(config.MODEL_DIR / f'shap_summary_{variant}.csv', index=False)

    identifiers = sampled[
        [column for column in ('machine_key', 'snapshot_date', config.TARGET_COLUMN) if column in sampled]
    ].reset_index(drop=True)
    contribution_frame = pd.DataFrame(
        shap_values,
        columns=[f'shap__{feature}' for feature in features],
    )
    contribution_frame.insert(0, 'shap_base_value_log_odds', base_values)
    detailed = pd.concat([identifiers, contribution_frame], axis=1)
    detailed.to_csv(
        config.MODEL_DIR / f'shap_sample_contributions_{variant}.csv.gz',
        index=False,
        compression='gzip',
    )

    top_n = int(getattr(config, 'SHAP_TOP_N', 25))
    plot_data = summary.head(top_n).sort_values('mean_abs_shap')
    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot_data))))
    ax.barh(plot_data['feature'], plot_data['mean_abs_shap'])
    ax.set_xlabel('Mean absolute SHAP value (log-odds contribution)')
    ax.set_ylabel('Feature')
    ax.set_title(f'Global SHAP Importance - {variant}')
    fig.tight_layout()
    fig.savefig(
        config.CHART_DIR / f'shap_importance_{variant}.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(fig)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values,
        x_sample,
        feature_names=features,
        max_display=min(top_n, len(features)),
        show=False,
    )
    plt.title(f'SHAP Summary - {variant}')
    plt.tight_layout()
    plt.savefig(
        config.CHART_DIR / f'shap_beeswarm_{variant}.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close()

    metadata = {
        'variant': variant,
        'sample_rows': int(len(sampled)),
        'feature_count': int(len(features)),
        'method': 'exact TreeSHAP via XGBoost pred_contribs',
        'contribution_scale': 'raw margin/log-odds',
        'mean_base_value_log_odds': float(np.mean(base_values)),
        'best_iteration': int(getattr(model, 'best_iteration', -1)),
    }
    (config.MODEL_DIR / f'shap_metadata_{variant}.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    return summary


def save_learning_curve(model, variant: str) -> dict:
    """Save training/calibration learning curves and an overfit diagnostic."""
    results = model.evals_result()
    train = results.get('validation_0', {})
    calibration = results.get('validation_1', {})
    metrics = sorted(set(train).intersection(calibration))
    if not metrics:
        raise ValueError(
            'Learning curves require eval_set=[training, calibration] during fit.'
        )
    iteration_count = min(len(train[metric]) for metric in metrics)
    iteration_count = min(
        iteration_count, *(len(calibration[metric]) for metric in metrics)
    )
    curve = pd.DataFrame({'iteration': np.arange(iteration_count, dtype=int)})
    for metric in metrics:
        curve[f'train_{metric}'] = np.asarray(train[metric][:iteration_count], dtype=float)
        curve[f'calibration_{metric}'] = np.asarray(
            calibration[metric][:iteration_count], dtype=float
        )
    best_iteration = int(getattr(model, 'best_iteration', iteration_count - 1))
    best_iteration = min(max(best_iteration, 0), iteration_count - 1)
    curve['is_best_iteration'] = curve['iteration'].eq(best_iteration)
    curve.to_csv(config.MODEL_DIR / f'learning_curve_{variant}.csv', index=False)

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        ax.plot(curve['iteration'], curve[f'train_{metric}'], label='Training')
        ax.plot(
            curve['iteration'],
            curve[f'calibration_{metric}'],
            label='Calibration',
        )
        ax.axvline(best_iteration, linestyle='--', label='Best iteration')
        ax.set_xlabel('Boosting iteration')
        ax.set_ylabel(metric)
        ax.set_title(metric.upper())
        ax.legend()
    fig.suptitle(f'XGBoost Learning Curves - {variant}')
    fig.tight_layout()
    fig.savefig(
        config.CHART_DIR / f'learning_curve_{variant}.png',
        dpi=180,
        bbox_inches='tight',
    )
    plt.close(fig)

    diagnostic = {
        'variant': variant,
        'best_iteration': best_iteration,
        'iterations_recorded': int(iteration_count),
        'early_stopping_rounds': int(model.get_params().get('early_stopping_rounds', 0)),
    }
    if 'aucpr' in metrics:
        train_best = float(curve.loc[best_iteration, 'train_aucpr'])
        cal_best = float(curve.loc[best_iteration, 'calibration_aucpr'])
        cal_last = float(curve.iloc[-1]['calibration_aucpr'])
        diagnostic.update(
            train_aucpr_at_best=train_best,
            calibration_aucpr_at_best=cal_best,
            aucpr_generalization_gap_at_best=train_best - cal_best,
            calibration_aucpr_last=cal_last,
            calibration_aucpr_deterioration_after_best=cal_best - cal_last,
        )
    if 'logloss' in metrics:
        train_best = float(curve.loc[best_iteration, 'train_logloss'])
        cal_best = float(curve.loc[best_iteration, 'calibration_logloss'])
        cal_last = float(curve.iloc[-1]['calibration_logloss'])
        diagnostic.update(
            train_logloss_at_best=train_best,
            calibration_logloss_at_best=cal_best,
            logloss_generalization_gap_at_best=cal_best - train_best,
            calibration_logloss_last=cal_last,
            calibration_logloss_deterioration_after_best=cal_last - cal_best,
        )
    max_auc_gap = float(getattr(config, 'OVERFIT_AUCPR_GAP_WARNING', 0.10))
    max_auc_deterioration = float(
        getattr(config, 'OVERFIT_AUCPR_DETERIORATION_WARNING', 0.02)
    )
    generalization_gap_warning = bool(
        diagnostic.get('aucpr_generalization_gap_at_best', 0.0) > max_auc_gap
    )
    progressive_overfitting_warning = bool(
        diagnostic.get('calibration_aucpr_deterioration_after_best', 0.0)
        > max_auc_deterioration
    )
    diagnostic['generalization_gap_warning'] = generalization_gap_warning
    diagnostic['progressive_overfitting_warning'] = progressive_overfitting_warning
    diagnostic['early_stopping_controlled'] = bool(
        best_iteration < iteration_count - 1 and not progressive_overfitting_warning
    )
    # Backward-compatible generic flag refers to progressive validation
    # deterioration. A separate generalization-gap warning is retained because
    # a train/calibration gap can exist even when early stopping is effective.
    diagnostic['overfitting_warning'] = progressive_overfitting_warning
    if progressive_overfitting_warning:
        diagnostic['diagnostic_status'] = 'validation_metric_deterioration_detected'
    elif generalization_gap_warning:
        diagnostic['diagnostic_status'] = (
            'early_stopping_controlled_with_generalization_gap'
        )
    else:
        diagnostic['diagnostic_status'] = 'no_material_overfitting_signal'
    (config.MODEL_DIR / f'learning_curve_diagnostics_{variant}.json').write_text(
        json.dumps(diagnostic, indent=2), encoding='utf-8'
    )
    return diagnostic


def save_final_xgboost_diagnostics(
    model,
    reference_dataframe: pd.DataFrame,
    features: list[str],
    variant: str,
) -> dict:
    """Run all requested final-model diagnostic exports."""
    importance = save_feature_importance(model, features, variant)
    shap_summary = save_shap_diagnostics(
        model, reference_dataframe, features, variant
    )
    learning = save_learning_curve(model, variant)
    return {
        'feature_importance_rows': int(len(importance)),
        'shap_summary_rows': int(len(shap_summary)),
        'learning_curve': learning,
    }
