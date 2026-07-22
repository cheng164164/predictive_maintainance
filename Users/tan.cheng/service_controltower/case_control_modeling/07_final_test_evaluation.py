"""Step 07: final locked-parameter test evaluation and model save.

Run this only after Phase 1 data-design selection and Phase 3 hyperparameter
tuning are complete. This is the first script that evaluates the test split.

It applies the FINAL_* settings from config.py, rebuilds the dataset in a final
output folder, fits the final XGBoost model, evaluates test views, and saves the
fitted model artifact.
"""
from __future__ import annotations

import importlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config
from cc_utils import (
    ensure_dir,
    fit_model_pipeline,
    get_evaluation_target,
    future_claim_lead_time_summary,
    make_model_pipeline,
    metrics_at_threshold,
    predict_score,
    threshold_free_metrics,
    top_k_metrics,
    validate_dataset_features,
    write_json,
)


DATE_COLUMNS = [
    "window_start",
    "window_end",
    "future_claim_date",
    "control_no_claim_start",
    "control_no_claim_end",
]


def _read_dataset(path_value) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _existing_path(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    path = Path(str(value))
    return str(path) if path.exists() else None


def _safe_name(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text).strip()).strip("_") or "view"


def _test_views(dataset_row: pd.Series) -> list[tuple[str, str]]:
    candidates = [
        ("matched_test", dataset_row.get("test_dataset_path")),
        # Recommended final production-like test view.
        ("asof_population_test", dataset_row.get("test_asof_population_dataset_path")),
        # Legacy views retained only if older files are present.
        ("test_with_population_negatives", dataset_row.get("test_with_population_negatives_path")),
        ("population_like_test", dataset_row.get("test_population_like_dataset_path")),
    ]
    out = []
    seen = set()
    for name, value in candidates:
        path = _existing_path(value)
        if path and path not in seen:
            out.append((name, path))
            seen.add(path)
    return out


def _add_prediction_columns(pred: pd.DataFrame, score: np.ndarray, threshold: float, top_k_rates: list[float]) -> pd.DataFrame:
    out = pred.copy().reset_index(drop=True)
    out["score"] = np.asarray(score, dtype=float)
    out["predicted_label"] = (out["score"] >= threshold).astype(int)
    out["score_rank_overall"] = out["score"].rank(method="first", ascending=False).astype(int)
    n = len(out)
    out["score_percentile"] = 1.0 - ((out["score_rank_overall"] - 1) / n) if n else np.nan
    for rate in top_k_rates:
        k = max(1, int(np.ceil(n * float(rate)))) if n else 0
        out[f"top_{int(round(float(rate) * 100))}pct_flag"] = (out["score_rank_overall"] <= k).astype(int) if k else 0
    return out


def _select_output_columns(pred: pd.DataFrame, include_features: bool) -> pd.DataFrame:
    priority_cols = [
        "dataset_id",
        "algorithm",
        "evaluation_view",
        "window_name",
        "case_control_group_id",
        "row_role",
        "target",
        "evaluation_target",
        "evaluation_target_col",
        "evaluation_target_mode",
        "evaluation_horizon_days",
        "machine_key",
        "full_model",
        "serial",
        "window_start",
        "window_end",
        "future_claim_date",
        "next_claim_date_on_or_after_window_end",
        "days_to_next_claim_on_or_after_window_end",
        "has_future_claim_on_or_after_window_end",
        "future_claim_lead_time_bucket",
        "claim_episode_id",
        "score",
        "score_rank_overall",
        "score_percentile",
        "predicted_label",
        "top_1pct_flag",
        "top_5pct_flag",
        "top_10pct_flag",
        "top_20pct_flag",
    ]
    priority_cols = [c for c in priority_cols if c in pred.columns]
    feature_cols = []
    if include_features:
        feature_cols = [c for c in list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES) if c in pred.columns]
    extra_cols = [c for c in pred.columns if c not in priority_cols and c not in feature_cols]
    return pred[priority_cols + feature_cols + extra_cols]


def _run_step(module_name: str) -> None:
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"Module {module_name} does not define run().")
    module.run()


def _snapshot(keys: set[str]) -> dict:
    return {k: getattr(config, k, None) for k in keys if hasattr(config, k)}


def _restore(values: dict) -> None:
    for key, value in values.items():
        setattr(config, key, value)


def _apply_final_config(final_dataset_output_dir: Path) -> dict:
    config.OUTPUT_DIR = final_dataset_output_dir
    config.CONTROLS_PER_POSITIVE_CASE = int(config.FINAL_CONTROLS_PER_POSITIVE_CASE)
    config.ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION = False
    config.ADD_ASOF_POPULATION_EVALUATION_TO_TEST = bool(getattr(config, "FINAL_ADD_ASOF_POPULATION_EVALUATION_TO_TEST", True))
    config.TEST_ASOF_EVALUATION_MAX_ROWS = getattr(config, "FINAL_TEST_ASOF_EVALUATION_MAX_ROWS", None)
    config.ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION = False
    config.VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE = int(config.FINAL_VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE)
    config.ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST = bool(config.FINAL_ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST)
    # 02_build_case_control_dataset.py reads this only when ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST is true.
    config.TEST_RANDOM_NEGATIVES_PER_POSITIVE = int(config.FINAL_TEST_RANDOM_NEGATIVES_PER_POSITIVE)
    config.XGBOOST_CLASS_IMPORTANCE_MODE = str(config.FINAL_XGBOOST_CLASS_IMPORTANCE_MODE)
    config.XGBOOST_FIXED_SCALE_POS_WEIGHT = float(config.FINAL_XGBOOST_FIXED_SCALE_POS_WEIGHT)
    config.XGBOOST_PARAMS = dict(config.FINAL_XGBOOST_PARAMS)
    config.XGBOOST_USE_EARLY_STOPPING = bool(config.FINAL_XGBOOST_USE_EARLY_STOPPING)
    config.XGBOOST_EARLY_STOPPING_ROUNDS = int(config.FINAL_XGBOOST_EARLY_STOPPING_ROUNDS)
    return {
        "OUTPUT_DIR": str(final_dataset_output_dir),
        "CONTROLS_PER_POSITIVE_CASE": config.CONTROLS_PER_POSITIVE_CASE,
        "FINAL_FIT_ON": config.FINAL_FIT_ON,
        "ADD_ASOF_POPULATION_EVALUATION_TO_TEST": getattr(config, "ADD_ASOF_POPULATION_EVALUATION_TO_TEST", False),
        "TEST_ASOF_EVALUATION_MAX_ROWS": getattr(config, "TEST_ASOF_EVALUATION_MAX_ROWS", None),
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST": config.ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST,
        "TEST_RANDOM_NEGATIVES_PER_POSITIVE": config.TEST_RANDOM_NEGATIVES_PER_POSITIVE,
        "XGBOOST_CLASS_IMPORTANCE_MODE": config.XGBOOST_CLASS_IMPORTANCE_MODE,
        "XGBOOST_FIXED_SCALE_POS_WEIGHT": config.XGBOOST_FIXED_SCALE_POS_WEIGHT,
        "XGBOOST_USE_EARLY_STOPPING": config.XGBOOST_USE_EARLY_STOPPING,
        "XGBOOST_EARLY_STOPPING_ROUNDS": config.XGBOOST_EARLY_STOPPING_ROUNDS,
        "XGBOOST_PARAMS": config.XGBOOST_PARAMS,
        "FINAL_EVALUATION_TARGET_MODE": getattr(config, "FINAL_EVALUATION_TARGET_MODE", getattr(config, "EVALUATION_TARGET_MODE", "training_target")),
        "FINAL_EVALUATION_CLAIM_HORIZON_DAYS": getattr(config, "FINAL_EVALUATION_CLAIM_HORIZON_DAYS", getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None)),
    }


def _score_test_views(dataset_row: pd.Series, model, algorithm: str, output_dir: Path) -> tuple[list[dict], list[pd.DataFrame]]:
    dataset_id = str(dataset_row["dataset_id"])
    threshold = float(getattr(config, "FINAL_TEST_SCORE_THRESHOLD", 0.50))
    top_k_rates = [float(x) for x in getattr(config, "FINAL_TEST_TOP_K_RATES", [0.01, 0.05, 0.10, 0.20])]
    include_features = bool(getattr(config, "FINAL_INCLUDE_FEATURE_COLUMNS", True))
    metric_rows = []
    topk_rows = []

    for view_name, path in _test_views(dataset_row):
        view_safe = _safe_name(view_name)
        test_df = _read_dataset(path)
        validate_dataset_features(test_df, config)
        X_test = test_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
        y_test_training = test_df["target"].astype(int)
        y_test, eval_target_col, eval_target_mode, eval_horizon_days = get_evaluation_target(test_df, config, prefix="FINAL")
        score = predict_score(model, X_test, algorithm)

        free = threshold_free_metrics(y_test, score)
        thresh = metrics_at_threshold(y_test, score, threshold=threshold)
        lead_summary = future_claim_lead_time_summary(test_df, y_test)
        metric_row = {
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "evaluation_view": view_name,
            "status": "used",
            "test_rows": int(len(test_df)),
            "test_training_target_positive_rows": int(y_test_training.sum()),
            "test_positive_rows": int(y_test.sum()),
            "test_positive_rate": float(y_test.mean()) if len(y_test) else np.nan,
            "evaluation_target_col": eval_target_col,
            "evaluation_target_mode": eval_target_mode,
            "evaluation_horizon_days": eval_horizon_days,
            "evaluation_path": path,
        }
        metric_row.update({f"threshold_free_{k}": v for k, v in free.items()})
        metric_row.update({f"threshold_{str(threshold).replace('.', 'p')}_{k}": v for k, v in thresh.items()})
        metric_row.update({f"lead_time_{k}": v for k, v in lead_summary.items()})
        metric_rows.append(metric_row)

        pred = test_df.copy()
        pred.insert(0, "evaluation_target", y_test.to_numpy())
        pred.insert(0, "evaluation_target_col", eval_target_col)
        pred.insert(0, "evaluation_target_mode", eval_target_mode)
        pred.insert(0, "evaluation_horizon_days", eval_horizon_days)
        pred.insert(0, "evaluation_view", view_name)
        pred.insert(0, "algorithm", algorithm)
        pred.insert(0, "dataset_id", dataset_id)
        pred = _add_prediction_columns(pred, score, threshold, top_k_rates)
        pred = _select_output_columns(pred, include_features=include_features)
        pred = pred.sort_values(
            ["dataset_id", "algorithm", "evaluation_view", "score", "machine_key", "window_end"],
            ascending=[True, True, True, False, True, True],
            kind="mergesort",
        )
        pred.to_csv(output_dir / f"{dataset_id}__{algorithm}__{view_safe}__test_window_predictions.csv", index=False)
        _select_output_columns(pred, include_features=False).to_csv(
            output_dir / f"{dataset_id}__{algorithm}__{view_safe}__test_window_predictions_compact.csv",
            index=False,
        )

        topk = top_k_metrics(y_test, score, top_k_rates)
        topk.insert(0, "evaluation_target_col", eval_target_col)
        topk.insert(0, "evaluation_target_mode", eval_target_mode)
        topk.insert(0, "evaluation_horizon_days", eval_horizon_days)
        topk.insert(0, "evaluation_view", view_name)
        topk.insert(0, "algorithm", algorithm)
        topk.insert(0, "dataset_id", dataset_id)
        topk_rows.append(topk)
    return metric_rows, topk_rows


def run() -> None:
    original_output_dir = config.OUTPUT_DIR
    final_root = original_output_dir / "07_final_test_evaluation"
    final_dataset_output_dir = final_root / "_final_dataset"
    model_output_dir = final_root / "model_and_test_results"
    ensure_dir(final_root)
    ensure_dir(model_output_dir)

    restore_keys = {
        "OUTPUT_DIR",
        "CONTROLS_PER_POSITIVE_CASE",
        "ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION",
        "ADD_ASOF_POPULATION_EVALUATION_TO_TEST",
        "TEST_ASOF_EVALUATION_MAX_ROWS",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST",
        "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE",
        "TEST_RANDOM_NEGATIVES_PER_POSITIVE",
        "XGBOOST_CLASS_IMPORTANCE_MODE",
        "XGBOOST_FIXED_SCALE_POS_WEIGHT",
        "XGBOOST_PARAMS",
        "XGBOOST_USE_EARLY_STOPPING",
        "XGBOOST_EARLY_STOPPING_ROUNDS",
    }
    original_values = _snapshot(restore_keys)

    all_metrics = []
    all_topk = []
    run_summaries = []
    applied_final = {}
    try:
        applied_final = _apply_final_config(final_dataset_output_dir)
        write_json(applied_final, final_root / "final_applied_config.json")

        for step in ["01_build_claim_episodes", "02_build_case_control_dataset"]:
            print(f"  building final test dataset: {step}")
            _run_step(step)

        dataset_index_path = final_dataset_output_dir / "02_case_control_datasets" / "dataset_index.csv"
        dataset_index = pd.read_csv(dataset_index_path)
        for _, dataset_row in dataset_index.iterrows():
            dataset_id = str(dataset_row["dataset_id"])
            train_df = _read_dataset(dataset_row["training_dataset_path"])
            valid_df = _read_dataset(dataset_row["validation_dataset_path"])
            validate_dataset_features(train_df, config)
            validate_dataset_features(valid_df, config)

            fit_on = str(getattr(config, "FINAL_FIT_ON", "train_plus_validation")).lower()
            use_early_stopping = bool(getattr(config, "FINAL_XGBOOST_USE_EARLY_STOPPING", False))
            if use_early_stopping:
                # Keep validation as the early-stopping/eval set and do not touch test.
                fit_df = train_df.copy()
                fit_eval_df = valid_df.copy()
                effective_fit_on = "train_with_validation_eval_for_early_stopping"
            elif fit_on == "train_plus_validation":
                fit_df = pd.concat([train_df, valid_df], ignore_index=True, sort=False)
                fit_eval_df = None
                effective_fit_on = "train_plus_validation_no_eval_set"
            else:
                fit_df = train_df.copy()
                fit_eval_df = valid_df.copy()
                effective_fit_on = "train_with_validation_learning_curve_eval"

            X_fit = fit_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
            y_fit = fit_df["target"].astype(int)
            X_eval = None
            y_eval = None
            if fit_eval_df is not None:
                X_eval = fit_eval_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
                y_eval = fit_eval_df["target"].astype(int)

            algorithm = "xgboost"
            model = make_model_pipeline(algorithm, config)
            if model is None:
                raise RuntimeError("XGBoost is not installed but FINAL model algorithm is xgboost.")
            fit_metadata = fit_model_pipeline(
                model,
                algorithm,
                X_fit,
                y_fit,
                config,
                X_eval=X_eval,
                y_eval=y_eval,
                eval_name="validation" if X_eval is not None else "none",
            )

            model_path = ""
            if bool(getattr(config, "FINAL_SAVE_MODEL_ARTIFACT", True)):
                import joblib
                model_dir = model_output_dir / "models"
                ensure_dir(model_dir)
                model_artifact = model_dir / f"{dataset_id}__final_xgboost_model.joblib"
                joblib.dump(model, model_artifact)
                model_path = str(model_artifact)

            metric_rows, topk_rows = _score_test_views(dataset_row, model, algorithm, model_output_dir)
            for row in metric_rows:
                row["fit_rows"] = int(len(fit_df))
                row["fit_positive_rows"] = int(y_fit.sum())
                row["effective_fit_on"] = effective_fit_on
                row["model_artifact_path"] = model_path
                for key, value in fit_metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        row[f"fit_{key}"] = value
            all_metrics.extend(metric_rows)
            all_topk.extend(topk_rows)
            run_summaries.append({
                "dataset_id": dataset_id,
                "fit_rows": int(len(fit_df)),
                "fit_positive_rows": int(y_fit.sum()),
                "effective_fit_on": effective_fit_on,
                "model_artifact_path": model_path,
                "test_views": [v for v, _ in _test_views(dataset_row)],
            })

    except Exception:
        traceback.print_exc()
        raise
    finally:
        _restore(original_values)
        config.OUTPUT_DIR = original_output_dir

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(model_output_dir / "final_test_metrics_all_datasets.csv", index=False)
    if all_topk:
        pd.concat(all_topk, ignore_index=True).to_csv(model_output_dir / "final_test_top_k_all_datasets.csv", index=False)

    write_json(
        {
            "step": "07_final_test_evaluation",
            "final_root": str(final_root),
            "final_dataset_output_dir": str(final_dataset_output_dir),
            "model_output_dir": str(model_output_dir),
            "applied_final_config": applied_final,
            "run_summaries": run_summaries,
            "notes": [
                "This is the only script intended to evaluate the test split.",
                "Final parameters are read from FINAL_* values in config.py.",
                "Final model fitting still uses the original case-control target; test metrics can use the evaluation-only future-claim horizon target.",
                "If final early stopping is enabled, the model fits on train and monitors validation; otherwise FINAL_FIT_ON controls whether train+validation are used.",
            ],
        },
        final_root / "run_summary.json",
    )
    print(f"Final test evaluation completed. Outputs: {final_root}")


if __name__ == "__main__":
    run()
