"""Train, calibrate, diagnose, and save the final XGBoost model."""
from __future__ import annotations

import json

import pandas as pd

import config
from modeling_utils import (
    choose_configured_threshold,
    ensure_directories,
    evaluate_predictions,
    feature_list_for_variant,
    fit_algorithm,
    fit_probability_calibration,
    load_snapshot_dataframe,
    make_algorithm,
)
from risk_tier_utils import finalize_tier_policy_for_model
from model_diagnostics import save_final_xgboost_diagnostics
from snapshot_builder import safe_name


def selected_raw_fault_codes(features: list[str]) -> list[str]:
    """Map selected sanitized fault-code features back to their raw code values.

    The source profile records every raw candidate code used when the snapshot
    table was built. Persisting the selected raw codes in model metadata keeps
    future 90-day scoring extracts schema-compatible with the trained model.
    """
    profile_path = config.OUTPUT_DIR / 'source_and_snapshot_profile.json'
    if not profile_path.exists():
        return []
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    feature_to_raw = {
        f"fault_code_{safe_name(code)}_90d": str(code)
        for code in profile.get('candidate_serious_codes', [])
    }
    return [
        feature_to_raw[feature]
        for feature in features
        if feature in feature_to_raw
    ]


def main() -> None:
    """Train, calibrate, diagnose, and save each configured final model variant."""
    ensure_directories()
    dataframe = load_snapshot_dataframe()
    calibration_dates = sorted(dataframe['snapshot_date'].unique())[
        -config.CALIBRATION_MONTHS :
    ]
    calibration = dataframe[
        dataframe['snapshot_date'].isin(calibration_dates)
    ].copy()
    fit = dataframe[
        ~dataframe['snapshot_date'].isin(calibration_dates)
    ].copy()
    fit_training = fit

    summaries: list[dict] = []
    for variant in config.MODEL_VARIANTS:
        print(f'Training final variant: {variant}', flush=True)
        features, top_code_detail = feature_list_for_variant(
            dataframe, fit, variant
        )
        model = make_algorithm('xgboost_note')
        model, best_iteration = fit_algorithm(
            'xgboost_note',
            model,
            fit_training[features],
            fit_training[config.TARGET_COLUMN].astype(int).to_numpy(),
            calibration[features],
            calibration[config.TARGET_COLUMN].astype(int).to_numpy(),
        )
        raw_cal = model.predict_proba(calibration[features])[:, 1]
        y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
        calibrator = fit_probability_calibration(y_cal, raw_cal)
        calibrated = calibrator.apply(raw_cal)
        threshold, calibration_threshold_score, _ = choose_configured_threshold(
            y_cal, calibrated
        )
        metrics = evaluate_predictions(y_cal, raw_cal, calibrated, threshold)

        model_path = config.MODEL_DIR / f'xgboost_{variant}.json'
        model.save_model(model_path)

        tier_policy = None
        tier_policy_path = None
        if bool(getattr(config, 'TIER_POLICY_ENABLED', False)):
            validation_policy_dir = (
                config.OUTPUT_DIR
                / config.MULTI_ANCHOR_FLEET_OUTPUT_SUBDIR
                / config.TIER_POLICY_OUTPUT_SUBDIR
            )
            calibration_scores = calibration[
                ['machine_key', 'snapshot_date', 'full_model', config.TARGET_COLUMN]
            ].copy()
            calibration_scores['raw_model_score'] = raw_cal
            calibration_scores['failure_probability'] = calibrated
            calibration_scores['true_label_90d'] = y_cal
            tier_policy = finalize_tier_policy_for_model(
                calibration_scores=calibration_scores,
                validation_policy_dir=validation_policy_dir,
                algorithm=config.TIER_POLICY_VALIDATION_ALGORITHM,
                variant=variant,
                model_dir=config.MODEL_DIR,
                output_dir=config.OUTPUT_DIR / config.FINAL_TIER_THRESHOLD_OUTPUT_SUBDIR,
            )
            tier_policy_path = config.MODEL_DIR / f'tier_policy_{variant}.json'
        diagnostics = None
        if bool(getattr(config, 'FINAL_MODEL_DIAGNOSTICS_ENABLED', True)):
            diagnostics = save_final_xgboost_diagnostics(
                model=model,
                reference_dataframe=calibration,
                features=features,
                variant=variant,
            )
        metadata = {
            'variant': variant,
            'features': features,
            'top_code_selection': top_code_detail,
            'selected_raw_fault_codes': selected_raw_fault_codes(features),
            'fit_snapshot_date_min': str(fit_training['snapshot_date'].min().date()),
            'fit_snapshot_date_max': str(fit_training['snapshot_date'].max().date()),
            'fit_rows': int(len(fit_training)),
            'calibration_dates': [
                str(pd.Timestamp(value).date()) for value in calibration_dates
            ],
            'best_iteration': best_iteration,
            'calibration_method': (
                config.PROBABILITY_CALIBRATION_METHOD
                if config.PROBABILITY_CALIBRATION_ENABLED
                else 'none'
            ),
            'platt_coefficient': calibrator.coefficient,
            'platt_intercept': calibrator.intercept,
            'operating_threshold': threshold,
            'threshold_metric': config.THRESHOLD_SELECTION_METRIC,
            'calibration_threshold_score': calibration_threshold_score,
            'target_source': config.TARGET_SOURCE,
            'target_display_name': config.TARGET_DISPLAY_NAME,
            'raw_warranty_csv_used': config.TARGET_SOURCE == 'warranty',
            'tier_policy_enabled': bool(tier_policy is not None),
            'tier_policy_file': tier_policy_path.name if tier_policy_path else None,
            'tier_confirmation_rule': (
                tier_policy.get('confirmation_rule') if tier_policy else None
            ),
            'tier_selection_policy': config.TIER_SELECTION_POLICY,
            'tier_require_non_overlapping_band_precision': bool(
                config.TIER_REQUIRE_NON_OVERLAPPING_BAND_PRECISION
            ),
            'tier_combination_selection_objective': (
                config.TIER_COMBINATION_SELECTION_OBJECTIVE
            ),
            'tier_score_gate_confidence_mode': (
                config.TIER_SCORE_GATE_CONFIDENCE_MODE
            ),
            'risk_index_definition': config.RISK_INDEX_DEFINITION,
            'risk_horizon_days': int(config.HORIZON_DAYS),
            'final_model_diagnostics': diagnostics,
            **{f'calibration_{key}': value for key, value in metrics.items()},
        }
        (config.MODEL_DIR / f'model_metadata_{variant}.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8'
        )
        summaries.append(
            {
                'variant': variant,
                'n_features': len(features),
                'best_iteration': best_iteration,
                'calibration_roc_auc': metrics['roc_auc'],
                'calibration_average_precision': metrics['average_precision'],
                'calibration_precision': metrics['precision'],
                'calibration_recall': metrics['recall'],
                'calibration_f2': metrics['f2'],
                'calibration_brier': metrics['brier'],
                'operating_threshold': threshold,
                'threshold_metric': config.THRESHOLD_SELECTION_METRIC,
                'calibration_threshold_score': calibration_threshold_score,
                'model_file': model_path.name,
                'tier_policy_file': tier_policy_path.name if tier_policy_path else None,
                'critical_top_n': (
                    int(tier_policy['tiers']['CRITICAL']['selected_top_n'])
                    if tier_policy else 0
                ),
                'high_top_n': (
                    int(tier_policy['tiers']['HIGH']['selected_top_n'])
                    if tier_policy else 0
                ),
                'medium_top_n': (
                    int(tier_policy['tiers']['MEDIUM']['selected_top_n'])
                    if tier_policy else 0
                ),
                'critical_selection_status': (
                    tier_policy['tiers']['CRITICAL']['selection_status']
                    if tier_policy else None
                ),
                'high_selection_status': (
                    tier_policy['tiers']['HIGH']['selection_status']
                    if tier_policy else None
                ),
                'medium_selection_status': (
                    tier_policy['tiers']['MEDIUM']['selection_status']
                    if tier_policy else None
                ),
                'tier_score_gate_confidence_mode': (
                    tier_policy.get('score_gate_confidence_mode')
                    if tier_policy else None
                ),
            }
        )
        print(
            f"  features={len(features)}, calibration AUC={metrics['roc_auc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, best_iteration={best_iteration}",
            flush=True,
        )

    pd.DataFrame(summaries).to_csv(
        config.MODEL_DIR / 'final_model_summary.csv', index=False
    )
    print('Final model training complete.')


if __name__ == '__main__':
    main()
