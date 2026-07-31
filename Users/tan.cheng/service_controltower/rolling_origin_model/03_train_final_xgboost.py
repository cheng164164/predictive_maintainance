"""Train final XGBoost models after the out-of-time validation design is frozen."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

import config
from modeling_utils import (
    choose_f2_threshold,
    ensure_directories,
    evaluate_predictions,
    feature_list_for_variant,
    fit_algorithm,
    fit_platt_calibration,
    load_snapshot_dataframe,
    make_algorithm,
)


def main() -> None:
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

    summaries: list[dict] = []
    for variant in config.MODEL_VARIANTS:
        print(f'Training final variant: {variant}', flush=True)
        features, top_code_detail = feature_list_for_variant(
            dataframe, fit, variant
        )
        model = make_algorithm('xgboost')
        model, best_iteration = fit_algorithm(
            'xgboost',
            model,
            fit[features],
            fit[config.TARGET_COLUMN].astype(int).to_numpy(),
            calibration[features],
            calibration[config.TARGET_COLUMN].astype(int).to_numpy(),
        )
        raw_cal = model.predict_proba(calibration[features])[:, 1]
        y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
        platt = fit_platt_calibration(y_cal, raw_cal)
        calibrated = platt.apply(raw_cal)
        threshold, calibration_f2 = choose_f2_threshold(y_cal, calibrated)
        metrics = evaluate_predictions(y_cal, raw_cal, calibrated, threshold)

        model_path = config.MODEL_DIR / f'xgboost_{variant}.json'
        model.save_model(model_path)
        importance = pd.DataFrame(
            {
                'feature': features,
                'importance_gain': model.feature_importances_,
            }
        ).sort_values('importance_gain', ascending=False)
        importance.to_csv(
            config.MODEL_DIR / f'feature_importance_{variant}.csv', index=False
        )
        metadata = {
            'variant': variant,
            'features': features,
            'top_code_selection': top_code_detail,
            'fit_snapshot_date_min': str(fit['snapshot_date'].min().date()),
            'fit_snapshot_date_max': str(fit['snapshot_date'].max().date()),
            'calibration_dates': [
                str(pd.Timestamp(value).date()) for value in calibration_dates
            ],
            'best_iteration': best_iteration,
            'platt_coefficient': platt.coefficient,
            'platt_intercept': platt.intercept,
            'f2_threshold': threshold,
            'calibration_f2_selected': calibration_f2,
            'target_source': config.TARGET_SOURCE,
            'target_display_name': config.TARGET_DISPLAY_NAME,
            'raw_warranty_csv_used': config.TARGET_SOURCE == 'warranty',
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
                'f2_threshold': threshold,
                'model_file': model_path.name,
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
