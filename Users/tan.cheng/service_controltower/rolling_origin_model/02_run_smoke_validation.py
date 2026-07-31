"""Run honest rolling-origin smoke validation across algorithms and XGBoost variants."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import config
from modeling_utils import (
    BASE_FAULT_FEATURES,
    BASE_FLUID_FEATURES,
    BASE_PM_FEATURES,
    BASE_STATIC_FEATURES,
    choose_f2_threshold,
    ensure_directories,
    evaluate_predictions,
    feature_list_for_variant,
    fit_algorithm,
    fit_platt_calibration,
    load_snapshot_dataframe,
    make_algorithm,
    risk_decile_table,
    rolling_origin_split,
    summarize_oof_predictions,
    top_k_table,
)

warnings.filterwarnings('ignore')


def run_algorithm_comparison(dataframe: pd.DataFrame) -> None:
    """Compare algorithms using the same fold-specific 27-feature base model."""
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped: list[dict] = []

    for algorithm_name in config.ALGORITHMS:
        print(f'\nAlgorithm: {algorithm_name}', flush=True)
        try:
            make_algorithm(algorithm_name)
        except ImportError as exc:
            print(f'  skipped: {exc}', flush=True)
            skipped.append({'algorithm': algorithm_name, 'reason': str(exc)})
            continue

        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            features, top_code_detail = feature_list_for_variant(
                dataframe, fit, 'base27'
            )
            x_fit = fit[features].astype(float)
            y_fit = fit[config.TARGET_COLUMN].astype(int).to_numpy()
            x_cal = calibration[features].astype(float)
            y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            x_val = validation[features].astype(float)
            y_val = validation[config.TARGET_COLUMN].astype(int).to_numpy()

            model = make_algorithm(algorithm_name)
            started = time.time()
            model, best_iteration = fit_algorithm(
                algorithm_name,
                model,
                x_fit,
                y_fit,
                x_cal,
                y_cal,
            )
            raw_cal = model.predict_proba(x_cal)[:, 1]
            raw_val = model.predict_proba(x_val)[:, 1]
            platt = fit_platt_calibration(y_cal, raw_cal)
            calibrated_cal = platt.apply(raw_cal)
            calibrated_val = platt.apply(raw_val)
            threshold, calibration_f2 = choose_f2_threshold(
                y_cal, calibrated_cal
            )
            metrics = evaluate_predictions(
                y_val, raw_val, calibrated_val, threshold
            )
            row = {
                'algorithm': algorithm_name,
                'fold': fold_name,
                'train_rows': len(fit),
                'calibration_rows': len(calibration),
                'validation_rows': len(validation),
                'validation_base_rate': float(y_val.mean()),
                'n_features': len(features),
                'threshold': threshold,
                'calibration_f2': calibration_f2,
                'best_iteration': best_iteration,
                'seconds': time.time() - started,
                'top_codes': '|'.join(item['feature'] for item in top_code_detail),
                'features': '|'.join(features),
                **metrics,
            }
            metric_rows.append(row)
            predictions = validation[
                ['machine_key', 'snapshot_date', config.TARGET_COLUMN]
            ].copy()
            predictions['algorithm'] = algorithm_name
            predictions['fold'] = fold_name
            predictions['score'] = raw_val
            predictions['calibrated_probability'] = calibrated_val
            predictions['prediction'] = (calibrated_val >= threshold).astype(int)
            prediction_frames.append(predictions)
            print(
                f"  {fold_name}: AUC={metrics['roc_auc']:.4f}, "
                f"AP={metrics['average_precision']:.4f}, "
                f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                f"F2={metrics['f2']:.3f}, top10 P={metrics['precision_top10']:.3f}",
                flush=True,
            )

    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_oof_predictions(
        fold_metrics, oof_predictions, grouping_column='algorithm'
    )
    fold_metrics.to_csv(
        config.OUTPUT_DIR / 'algorithm_fold_metrics.csv', index=False
    )
    summary.to_csv(config.OUTPUT_DIR / 'algorithm_summary.csv', index=False)
    if config.WRITE_OOF_PREDICTIONS:
        oof_predictions.to_csv(
            config.OUTPUT_DIR / 'algorithm_oof_predictions.csv.gz',
            index=False,
            compression='gzip',
        )
    if skipped:
        pd.DataFrame(skipped).to_csv(
            config.OUTPUT_DIR / 'algorithm_skipped.csv', index=False
        )
    print('\nAlgorithm summary')
    print(summary.to_string(index=False))


def run_xgboost_variant_comparison(dataframe: pd.DataFrame) -> None:
    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for variant in config.MODEL_VARIANTS:
        print(f'\nXGBoost variant: {variant}', flush=True)
        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            features, top_code_detail = feature_list_for_variant(
                dataframe, fit, variant
            )
            x_fit = fit[features].astype(float)
            y_fit = fit[config.TARGET_COLUMN].astype(int).to_numpy()
            x_cal = calibration[features].astype(float)
            y_cal = calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            x_val = validation[features].astype(float)
            y_val = validation[config.TARGET_COLUMN].astype(int).to_numpy()

            model = make_algorithm('xgboost')
            started = time.time()
            model, best_iteration = fit_algorithm(
                'xgboost',
                model,
                x_fit,
                y_fit,
                x_cal,
                y_cal,
            )
            raw_cal = model.predict_proba(x_cal)[:, 1]
            raw_val = model.predict_proba(x_val)[:, 1]
            platt = fit_platt_calibration(y_cal, raw_cal)
            calibrated_cal = platt.apply(raw_cal)
            calibrated_val = platt.apply(raw_val)
            threshold, calibration_f2 = choose_f2_threshold(
                y_cal, calibrated_cal
            )
            metrics = evaluate_predictions(
                y_val, raw_val, calibrated_val, threshold
            )
            metric_rows.append(
                {
                    'variant': variant,
                    'fold': fold_name,
                    'train_rows': len(fit),
                    'calibration_rows': len(calibration),
                    'validation_rows': len(validation),
                    'validation_base_rate': float(y_val.mean()),
                    'n_features': len(features),
                    'threshold': threshold,
                    'calibration_f2': calibration_f2,
                    'best_iteration': best_iteration,
                    'seconds': time.time() - started,
                    'top_codes': '|'.join(
                        item['feature'] for item in top_code_detail
                    ),
                    'features': '|'.join(features),
                    **metrics,
                }
            )
            predictions = validation[
                ['machine_key', 'snapshot_date', config.TARGET_COLUMN]
            ].copy()
            predictions['variant'] = variant
            predictions['fold'] = fold_name
            predictions['score'] = raw_val
            predictions['calibrated_probability'] = calibrated_val
            predictions['prediction'] = (calibrated_val >= threshold).astype(int)
            prediction_frames.append(predictions)
            print(
                f"  {fold_name}: AUC={metrics['roc_auc']:.4f}, "
                f"AP={metrics['average_precision']:.4f}, "
                f"top10 P={metrics['precision_top10']:.3f}, "
                f"features={len(features)}",
                flush=True,
            )

    fold_metrics = pd.DataFrame(metric_rows)
    oof_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_oof_predictions(
        fold_metrics, oof_predictions, grouping_column='variant'
    )
    fold_metrics.to_csv(
        config.OUTPUT_DIR / 'xgboost_variant_fold_metrics.csv', index=False
    )
    summary.to_csv(
        config.OUTPUT_DIR / 'xgboost_variant_summary.csv', index=False
    )
    top_k_table(oof_predictions, grouping_column='variant').to_csv(
        config.OUTPUT_DIR / 'top_k_ranking_metrics.csv', index=False
    )
    risk_decile_table(oof_predictions, grouping_column='variant').to_csv(
        config.OUTPUT_DIR / 'risk_decile_lift.csv', index=False
    )
    if config.WRITE_OOF_PREDICTIONS:
        oof_predictions.to_csv(
            config.OUTPUT_DIR / 'xgboost_variant_oof_predictions.csv.gz',
            index=False,
            compression='gzip',
        )
    print('\nXGBoost variant summary')
    print(summary.to_string(index=False))


def run_ablation(dataframe: pd.DataFrame) -> None:
    groups = {
        'full_base27': None,
        'drop_fault': 'drop_fault',
        'drop_fluid': 'drop_fluid',
        'drop_pm': 'drop_pm',
        'fault_only20': 'fault_only',
        'fluid_only5': 'fluid_only',
        'pm_only2': 'pm_only',
    }
    rows: list[dict] = []
    for group_name, mode in groups.items():
        print(f'\nAblation: {group_name}', flush=True)
        for fold in config.VALIDATION_FOLDS:
            fold_name, fit, calibration, validation = rolling_origin_split(
                dataframe, fold
            )
            base_features, top_detail = feature_list_for_variant(
                dataframe, fit, 'base27'
            )
            top_codes = [item['feature'] for item in top_detail]
            fault_features = [
                feature
                for feature in BASE_FAULT_FEATURES + top_codes
                if feature in base_features
            ]
            fluid_features = [
                feature for feature in BASE_FLUID_FEATURES if feature in base_features
            ]
            pm_features = [
                feature for feature in BASE_PM_FEATURES if feature in base_features
            ]
            if mode is None:
                features = base_features
            elif mode == 'drop_fault':
                features = fluid_features + pm_features
            elif mode == 'drop_fluid':
                features = fault_features + pm_features
            elif mode == 'drop_pm':
                features = fault_features + fluid_features
            elif mode == 'fault_only':
                features = fault_features
            elif mode == 'fluid_only':
                features = fluid_features
            elif mode == 'pm_only':
                features = pm_features
            else:
                raise AssertionError(mode)

            model = XGBClassifier(
                **dict(config.XGB_PARAMS),
                n_jobs=config.N_JOBS,
                random_state=config.RANDOM_SEED,
            )
            model.fit(
                fit[features],
                fit[config.TARGET_COLUMN],
                eval_set=[
                    (calibration[features], calibration[config.TARGET_COLUMN])
                ],
                verbose=False,
            )
            score = model.predict_proba(validation[features])[:, 1]
            labels = validation[config.TARGET_COLUMN].astype(int).to_numpy()
            rows.append(
                {
                    'group': group_name,
                    'fold': fold_name,
                    'n_features': len(features),
                    'roc_auc': roc_auc_score(labels, score),
                    'average_precision': average_precision_score(labels, score),
                }
            )
    fold_metrics = pd.DataFrame(rows)
    summary = (
        fold_metrics.groupby('group', as_index=False)
        .agg(
            mean_roc_auc=('roc_auc', 'mean'),
            std_roc_auc=('roc_auc', 'std'),
            mean_average_precision=('average_precision', 'mean'),
            mean_features=('n_features', 'mean'),
        )
        .sort_values('mean_roc_auc', ascending=False)
    )
    full_auc = float(
        summary.loc[summary['group'].eq('full_base27'), 'mean_roc_auc'].iloc[0]
    )
    summary['delta_vs_full'] = summary['mean_roc_auc'] - full_auc
    fold_metrics.to_csv(
        config.OUTPUT_DIR / 'feature_group_ablation_fold_metrics.csv', index=False
    )
    summary.to_csv(
        config.OUTPUT_DIR / 'feature_group_ablation_summary.csv', index=False
    )


def run_shuffled_label_null(dataframe: pd.DataFrame) -> None:
    rows: list[dict] = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    for fold in config.VALIDATION_FOLDS:
        fold_name, fit, calibration, validation = rolling_origin_split(
            dataframe, fold
        )
        features, _ = feature_list_for_variant(dataframe, fit, 'base27')
        for replicate in range(config.NULL_REPLICATES_PER_FOLD):
            shuffled_fit = rng.permutation(
                fit[config.TARGET_COLUMN].astype(int).to_numpy()
            )
            shuffled_cal = rng.permutation(
                calibration[config.TARGET_COLUMN].astype(int).to_numpy()
            )
            model = XGBClassifier(
                **dict(config.XGB_PARAMS),
                n_jobs=config.N_JOBS,
                random_state=config.RANDOM_SEED + replicate,
            )
            model.fit(
                fit[features],
                shuffled_fit,
                eval_set=[(calibration[features], shuffled_cal)],
                verbose=False,
            )
            score = model.predict_proba(validation[features])[:, 1]
            labels = validation[config.TARGET_COLUMN].astype(int).to_numpy()
            rows.append(
                {
                    'fold': fold_name,
                    'replicate': replicate + 1,
                    'roc_auc': roc_auc_score(labels, score),
                    'average_precision': average_precision_score(labels, score),
                }
            )
    metrics = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                'null_replicates': len(metrics),
                'mean_null_auc': metrics['roc_auc'].mean(),
                'std_null_auc': metrics['roc_auc'].std(ddof=1),
                'min_null_auc': metrics['roc_auc'].min(),
                'max_null_auc': metrics['roc_auc'].max(),
                'mean_null_ap': metrics['average_precision'].mean(),
            }
        ]
    )
    metrics.to_csv(
        config.OUTPUT_DIR / 'shuffled_label_null_metrics.csv', index=False
    )
    summary.to_csv(
        config.OUTPUT_DIR / 'shuffled_label_null_summary.csv', index=False
    )


def main() -> None:
    ensure_directories()
    dataframe = load_snapshot_dataframe()
    print(
        f'Loaded {len(dataframe):,} snapshots, '
        f'{dataframe["machine_key"].nunique():,} machines, '
        f'positive rate={dataframe[config.TARGET_COLUMN].mean():.2%}',
        flush=True,
    )
    run_algorithm_comparison(dataframe)
    run_xgboost_variant_comparison(dataframe)
    if config.RUN_ABLATION:
        run_ablation(dataframe)
    if config.RUN_SHUFFLED_LABEL_NULL:
        run_shuffled_label_null(dataframe)
    print('\nSmoke validation complete.')


if __name__ == '__main__':
    main()
