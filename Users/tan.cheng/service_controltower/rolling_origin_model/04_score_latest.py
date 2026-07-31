"""Score the latest dense fleet snapshot with the trained final XGBoost models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

import config
from modeling_utils import (
    PlattCalibration,
    add_reconstructed_risk_tiers,
    ensure_directories,
    friendly_feature_name,
    load_latest_snapshot_dataframe,
)


def top_positive_contributions(
    model: XGBClassifier,
    features: list[str],
    matrix: pd.DataFrame,
    top_n: int = 3,
) -> list[str]:
    contributions = model.get_booster().predict(
        DMatrix(matrix, feature_names=features), pred_contribs=True
    )[:, :-1]
    result: list[str] = []
    for row in contributions:
        order = np.argsort(row)[::-1]
        factors: list[str] = []
        for index in order:
            if row[index] <= 0:
                continue
            factors.append(
                f'{friendly_feature_name(features[index])} (+{row[index]:.2f})'
            )
            if len(factors) >= top_n:
                break
        result.append('; '.join(factors))
    return result


def main() -> None:
    ensure_directories()
    latest = load_latest_snapshot_dataframe()
    tier_audit_columns = [
        'fault_count_90d',
        'fault_serious_count_90d',
        'fault_serious_30d',
        'fault_l03_count_90d',
        'fault_l04_count_90d',
        'fault_single_system_concentration',
        'fault_severity_recency_90d',
        'has_accelerating_faults',
        'fluid_current_severity',
        'fluid_contaminant_flag',
        'pm_overdue_count_90d',
        'operation_history_days',
        'days_since_last_sensor_reading',
        'smr_delta_30d',
        'prior_target_event_count_365d',
    ]

    summary_rows: list[dict] = []
    for variant in config.MODEL_VARIANTS:
        model_path = config.MODEL_DIR / f'xgboost_{variant}.json'
        metadata_path = config.MODEL_DIR / f'model_metadata_{variant}.json'
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f'Missing model artifacts for {variant}. Run 03_train_final_xgboost.py.'
            )
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        features = metadata['features']
        for feature in features:
            if feature not in latest.columns:
                latest[feature] = 0.0
        model = XGBClassifier()
        model.load_model(model_path)
        raw_score = model.predict_proba(latest[features])[:, 1]
        platt = PlattCalibration(
            coefficient=float(metadata['platt_coefficient']),
            intercept=float(metadata['platt_intercept']),
        )
        probability = platt.apply(raw_score)
        scores = latest[['machine_key', 'snapshot_date', 'full_model']].copy()
        scores['raw_model_score'] = raw_score
        scores['failure_probability'] = probability
        scores['risk_index'] = scores['raw_model_score'].rank(
            method='average', pct=True
        ) * 100.0
        scores['risk_score_0_5'] = scores['risk_index'] / 20.0
        scores['model_flag_f2'] = (
            probability >= float(metadata['f2_threshold'])
        ).astype('int8')
        for column in tier_audit_columns:
            scores[column] = latest[column].to_numpy() if column in latest else 0.0
        scores = add_reconstructed_risk_tiers(scores)
        scores['top_risk_factors'] = top_positive_contributions(
            model, features, latest[features]
        )
        tier_order = pd.Categorical(
            scores['risk_level_reconstructed'],
            categories=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
            ordered=True,
        )
        scores = (
            scores.assign(_tier_order=tier_order)
            .sort_values(['_tier_order', 'risk_index'], ascending=[True, False])
            .drop(columns='_tier_order')
        )
        output_path = config.OUTPUT_DIR / f'latest_scores_{variant}.csv'
        scores.to_csv(output_path, index=False)
        counts = scores['risk_level_reconstructed'].value_counts().to_dict()
        summary_rows.append(
            {
                'variant': variant,
                'score_date': str(latest['snapshot_date'].iloc[0].date()),
                'score_rows': len(scores),
                'mean_failure_probability': scores['failure_probability'].mean(),
                'f2_flagged_rows': int(scores['model_flag_f2'].sum()),
                'critical_reconstructed': int(counts.get('CRITICAL', 0)),
                'high_reconstructed': int(counts.get('HIGH', 0)),
                'medium_reconstructed': int(counts.get('MEDIUM', 0)),
                'low_reconstructed': int(counts.get('LOW', 0)),
                'output_file': output_path.name,
            }
        )
        print(
            f'{variant}: rows={len(scores):,}, reconstructed tiers={counts}',
            flush=True,
        )

    pd.DataFrame(summary_rows).to_csv(
        config.OUTPUT_DIR / 'latest_scoring_summary.csv', index=False
    )
    production_file = config.OUTPUT_DIR / (
        f'latest_scores_{config.PRODUCTION_VARIANT}.csv'
    )
    print(f'Production candidate output: {production_file}')
    print(
        'Note: risk_level_reconstructed is an approximation because the exact '
        'component-score gates from the original implementation were not supplied.'
    )


if __name__ == '__main__':
    main()
