"""Step 06: coarse XGBoost hyperparameter grid search on validation only.

Use this after Phase 1 design sweep has selected a promising data-design setup.
This script:
  1. applies HYPERPARAMETER_TUNING_DATA_DESIGN from config.py,
  2. builds one dataset for that design,
  3. expands HYPERPARAMETER_TUNING_GRID,
  4. fits on the training split and evaluates validation views only,
  5. collects validation metrics across the grid.

The test split is not evaluated here.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import re
import traceback
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import config
from cc_utils import ensure_dir, write_json


def _banner(title: str) -> None:
    line = "=" * 88
    print("\n" + line)
    print(title)
    print(line, flush=True)


DATASET_BUILD_STEPS = ["01_build_claim_episodes", "02_build_case_control_dataset"]


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "p")
    text = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", text).strip("_")
    return text or "value"


def _short_hash(obj: Any, length: int = 8) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def _safe_id(text: str, max_len: int = 96, hash_payload: Any | None = None) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text).strip()).strip("_") or "experiment"
    if hash_payload is not None:
        suffix = f"__h{_short_hash(hash_payload)}"
        base_len = max(1, max_len - len(suffix))
        return (clean[:base_len].rstrip("_") or "experiment") + suffix
    if len(clean) > max_len:
        suffix = f"__h{_short_hash(clean)}"
        base_len = max(1, max_len - len(suffix))
        clean = (clean[:base_len].rstrip("_") or "experiment") + suffix
    return clean


def _expand_grid(grid: Mapping[str, Any]) -> list[dict]:
    if not grid:
        raise ValueError("HYPERPARAMETER_TUNING_GRID is empty.")
    keys = list(grid.keys())
    values = []
    for key in keys:
        value = grid[key]
        if isinstance(value, (list, tuple, set)):
            vals = list(value)
        else:
            vals = [value]
        if not vals:
            raise ValueError(f"HYPERPARAMETER_TUNING_GRID[{key!r}] has no values.")
        values.append(vals)
    out = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        out.append(params)
    max_n = getattr(config, "HYPERPARAMETER_TUNING_MAX_EXPERIMENTS", None)
    if max_n is not None:
        out = out[: int(max_n)]
    return out


def _hyperparam_experiment_id(params: Mapping[str, Any], experiment_number: int) -> str:
    """Return a short folder-safe ID for one hyperparameter experiment.

    Full hyperparameter values are saved in hyperparameter_experiment_config.json
    and summary CSV columns. The folder name stays compact to avoid path-length
    errors on Azure/AML mounts.
    """
    short = {
        "max_depth": "d",
        "min_child_weight": "c",
        "subsample": "sub",
        "colsample_bytree": "col",
        "gamma": "g",
        "reg_lambda": "l2",
        "reg_alpha": "l1",
        "learning_rate": "lr",
        "n_estimators": "n",
        "early_stopping_rounds": "es",
    }
    preferred = ["max_depth", "min_child_weight", "gamma", "reg_lambda", "reg_alpha", "early_stopping_rounds"]
    parts = [f"hp{experiment_number:03d}"]
    for key in preferred:
        if key in params:
            parts.append(f"{short.get(key, key)}{_safe_token(params[key])}")
    return _safe_id("__".join(parts), max_len=64, hash_payload=None)


def _apply_scale_pos_weight(value: Any, applied: dict) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip().lower() in {"none", "off", "false"}:
        config.XGBOOST_CLASS_IMPORTANCE_MODE = "none"
        config.XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0
    elif isinstance(value, str) and value.strip().lower() == "auto":
        config.XGBOOST_CLASS_IMPORTANCE_MODE = "auto"
        config.XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0
    else:
        config.XGBOOST_CLASS_IMPORTANCE_MODE = "fixed"
        config.XGBOOST_FIXED_SCALE_POS_WEIGHT = float(value)
    applied["XGBOOST_CLASS_IMPORTANCE_MODE"] = config.XGBOOST_CLASS_IMPORTANCE_MODE
    applied["XGBOOST_FIXED_SCALE_POS_WEIGHT"] = config.XGBOOST_FIXED_SCALE_POS_WEIGHT


def _apply_data_design(design: Mapping[str, Any]) -> dict:
    applied = {}
    for key, value in design.items():
        if key == "scale_pos_weight":
            _apply_scale_pos_weight(value, applied)
        elif key.isupper():
            setattr(config, key, value)
            applied[key] = value
        else:
            canonical = key.upper()
            setattr(config, canonical, value)
            applied[canonical] = value
    config.ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION = True
    config.ADD_ASOF_POPULATION_EVALUATION_TO_TEST = False
    config.ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION = False
    config.ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST = False
    applied["ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION"] = True
    applied["ADD_ASOF_POPULATION_EVALUATION_TO_TEST"] = False
    applied["ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION"] = False
    applied["ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST"] = False
    return applied


def _apply_hyperparameters(params: Mapping[str, Any], base_xgb_params: Mapping[str, Any]) -> dict:
    applied = {}
    xgb_update = {}
    early_rounds = int(params.get("early_stopping_rounds", 0) or 0)
    for key, value in params.items():
        if key == "early_stopping_rounds":
            continue
        xgb_update[key] = value
    config.XGBOOST_PARAMS = dict(base_xgb_params)
    config.XGBOOST_PARAMS.update(xgb_update)
    config.XGBOOST_USE_EARLY_STOPPING = early_rounds > 0
    config.XGBOOST_EARLY_STOPPING_ROUNDS = early_rounds
    applied.update({f"xgb_{k}": v for k, v in xgb_update.items()})
    applied["XGBOOST_USE_EARLY_STOPPING"] = config.XGBOOST_USE_EARLY_STOPPING
    applied["XGBOOST_EARLY_STOPPING_ROUNDS"] = early_rounds
    return applied


def _snapshot(keys: set[str]) -> dict:
    return {k: getattr(config, k, None) for k in keys if hasattr(config, k)}


def _restore(values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        setattr(config, key, value)


def _run_step(module_name: str) -> None:
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"Module {module_name} does not define run().")
    module.run()


def _read_with_metadata(path: Path, experiment_id: str, data_design: Mapping[str, Any], params: Mapping[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "hyperparameter_experiment_id", experiment_id)
    for key, value in data_design.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            df[f"design_{key}"] = value
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            df[f"hyper_{key}"] = value
    return df




def _topk_label(rate: float) -> str:
    pct = int(round(float(rate) * 100))
    return f"top_{pct}pct"


def _build_tuning_review_summary(metrics: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    key_cols = ["hyperparameter_experiment_id", "dataset_id", "algorithm", "evaluation_view", "evaluation_horizon_days"]
    key_cols = [c for c in key_cols if c in metrics.columns]
    design_cols = [c for c in metrics.columns if c.startswith("design_")]
    hyper_cols = [c for c in metrics.columns if c.startswith("hyper_")]
    metric_cols = [
        c for c in [
            "status",
            "train_rows",
            "evaluation_rows",
            "evaluation_positive_rate",
            "evaluation_target_col",
            "evaluation_target_mode",
            "lead_time_future_claim_observed_rows",
            "lead_time_future_claim_never_observed_rows",
            "lead_time_evaluation_positive_days_median",
            "threshold_free_average_precision",
            "threshold_free_roc_auc",
            "fit_xgboost_class_importance_mode",
            "fit_xgboost_scale_pos_weight",
            "fit_xgboost_early_stopping_enabled",
            "fit_xgboost_early_stopping_rounds",
            "fit_xgboost_best_iteration",
            "fit_xgboost_best_score",
        ] if c in metrics.columns
    ]
    base = metrics[key_cols + metric_cols + design_cols + hyper_cols].drop_duplicates(key_cols).copy()
    if not topk.empty:
        tk = topk.copy()
        tk["top_k_label"] = tk["top_k_rate"].map(_topk_label)
        if {"precision_at_k", "flagged_count"}.issubset(tk.columns):
            tk["positive_hits_at_k"] = tk["precision_at_k"] * tk["flagged_count"]
        value_cols = [c for c in ["precision_at_k", "recall_at_k", "lift_vs_random", "flagged_count", "positive_hits_at_k"] if c in tk.columns]
        wide_parts = []
        for value_col in value_cols:
            pivot = tk.pivot_table(index=key_cols, columns="top_k_label", values=value_col, aggfunc="first")
            pivot.columns = [f"{label}_{value_col}" for label in pivot.columns]
            wide_parts.append(pivot.reset_index())
        if wide_parts:
            wide = wide_parts[0]
            for extra in wide_parts[1:]:
                wide = wide.merge(extra, on=key_cols, how="outer")
            base = base.merge(wide, on=key_cols, how="left")
    eval_priority = {"asof_population_validation": 0, "population_like_validation": 1, "validation_with_population_negatives": 2, "matched_validation": 3}
    base["evaluation_view_priority"] = base["evaluation_view"].map(eval_priority).fillna(9).astype(int)
    base["evaluation_horizon_days_sort"] = pd.to_numeric(base.get("evaluation_horizon_days", float("nan")), errors="coerce").fillna(-1)
    sort_cols = ["evaluation_view_priority", "evaluation_horizon_days_sort"] + [
        c for c in ["top_5pct_precision_at_k", "threshold_free_average_precision", "top_10pct_precision_at_k", "threshold_free_roc_auc"]
        if c in base.columns
    ]
    out = base.sort_values(sort_cols, ascending=[True, True] + [False] * (len(sort_cols)-2), kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["evaluation_view_priority", "evaluation_horizon_days_sort"], errors="ignore")

def run() -> None:
    original_output_dir = config.OUTPUT_DIR
    tuning_root = original_output_dir / "06_xgboost_hyperparameter_tuning"
    dataset_output_dir = tuning_root / "_dataset_for_tuning"
    experiments_root = tuning_root / "experiments"
    ensure_dir(tuning_root)
    ensure_dir(experiments_root)

    data_design = dict(getattr(config, "HYPERPARAMETER_TUNING_DATA_DESIGN", {}) or {})
    grid = getattr(config, "HYPERPARAMETER_TUNING_GRID", {}) or {}
    param_grid = _expand_grid(grid)
    if not param_grid:
        raise ValueError("No hyperparameter experiments were generated.")

    _banner(f"Starting XGBoost hyperparameter tuning: {len(param_grid)} experiments")
    print(f"Tuning root: {tuning_root}", flush=True)
    print(f"Data design: {data_design}", flush=True)

    restore_keys = {
        "OUTPUT_DIR",
        "CONTROLS_PER_POSITIVE_CASE",
        "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE",
        "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT",
        "ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS",
        "VALIDATION_ASOF_EVALUATION_MAX_ROWS",
        "ADD_ASOF_POPULATION_EVALUATION_TO_VALIDATION",
        "ADD_ASOF_POPULATION_EVALUATION_TO_TEST",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST",
        "XGBOOST_CLASS_IMPORTANCE_MODE",
        "XGBOOST_FIXED_SCALE_POS_WEIGHT",
        "XGBOOST_PARAMS",
        "XGBOOST_USE_EARLY_STOPPING",
        "XGBOOST_EARLY_STOPPING_ROUNDS",
        "SAVE_SHAP_VALUES",
        "SHAP_EVALUATION_VIEWS",
    }
    original_values = _snapshot(restore_keys)
    base_xgb_params = dict(config.XGBOOST_PARAMS)

    combined_metrics = []
    combined_topk = []
    combined_group = []
    run_rows = []

    try:
        # Build the dataset once for the selected data design.
        config.OUTPUT_DIR = dataset_output_dir
        applied_design = _apply_data_design(data_design)
        write_json({"data_design": applied_design}, dataset_output_dir / "hyperparameter_tuning_data_design.json")
        _banner("Building fixed dataset for hyperparameter tuning")
        for step_num, step in enumerate(DATASET_BUILD_STEPS, start=1):
            print(f"\n>>> [Dataset build step {step_num}/{len(DATASET_BUILD_STEPS)}] {step}", flush=True)
            _run_step(step)
        if bool(getattr(config, "HYPERPARAMETER_TUNING_RUN_CROSS_VALIDATION", False)):
            print("  running 03_cross_validation for tuning dataset")
            _run_step("03_cross_validation")
        dataset_index_path = dataset_output_dir / "02_case_control_datasets" / "dataset_index.csv"
        if not dataset_index_path.exists():
            raise FileNotFoundError(f"Dataset index was not created: {dataset_index_path}")

        validate_module = importlib.import_module("04_fit_validate_model_report")
        validate_module = importlib.reload(validate_module)

        for exp_num, params in enumerate(param_grid, start=1):
            experiment_id = _hyperparam_experiment_id(params, exp_num)
            exp_dir = experiments_root / experiment_id
            ensure_dir(exp_dir)
            _banner(f"[Hyperparameter tuning {exp_num}/{len(param_grid)}] {experiment_id}")
            print(f"Parameters: {params}", flush=True)

            # Restore selected data-design settings, then apply params.
            _apply_data_design(data_design)
            applied_params = _apply_hyperparameters(params, base_xgb_params)
            write_json(
                {
                    "experiment_id": experiment_id,
                    "data_design": applied_design,
                    "hyperparameters": params,
                    "applied_params": applied_params,
                    "dataset_index_path": str(dataset_index_path),
                },
                exp_dir / "hyperparameter_experiment_config.json",
            )

            status = "completed"
            error = ""
            try:
                validate_module.run(dataset_index_path=dataset_index_path, step_dir=exp_dir / "04_fit_validate_model_report")
            except Exception as exc:
                status = "failed"
                error = str(exc)
                traceback.print_exc()

            metrics_path = exp_dir / "04_fit_validate_model_report" / "validation_metrics_all_datasets.csv"
            topk_path = exp_dir / "04_fit_validate_model_report" / "validation_top_k_all_datasets.csv"
            group_path = exp_dir / "04_fit_validate_model_report" / "validation_group_ranking_metrics_all_datasets.csv"
            m = _read_with_metadata(metrics_path, experiment_id, applied_design, params)
            t = _read_with_metadata(topk_path, experiment_id, applied_design, params)
            g = _read_with_metadata(group_path, experiment_id, applied_design, params)
            if not m.empty:
                combined_metrics.append(m)
            if not t.empty:
                combined_topk.append(t)
            if not g.empty:
                combined_group.append(g)
            run_rows.append({
                "hyperparameter_experiment_id": experiment_id,
                "experiment_number": exp_num,
                "experiment_count": len(param_grid),
                "status": status,
                "error": error,
                "output_dir": str(exp_dir),
                **{f"design_{k}": v for k, v in applied_design.items() if isinstance(v, (str, int, float, bool)) or v is None},
                **{f"hyper_{k}": v for k, v in params.items() if isinstance(v, (str, int, float, bool)) or v is None},
            })

    finally:
        _restore(original_values)
        config.OUTPUT_DIR = original_output_dir

    pd.DataFrame(run_rows).to_csv(tuning_root / "hyperparameter_tuning_run_summary.csv", index=False)
    if combined_metrics:
        pd.concat(combined_metrics, ignore_index=True).to_csv(tuning_root / "hyperparameter_tuning_validation_metrics.csv", index=False)
    if combined_topk:
        pd.concat(combined_topk, ignore_index=True).to_csv(tuning_root / "hyperparameter_tuning_validation_top_k.csv", index=False)
    metrics_all = pd.concat(combined_metrics, ignore_index=True) if combined_metrics else pd.DataFrame()
    topk_all = pd.concat(combined_topk, ignore_index=True) if combined_topk else pd.DataFrame()
    if combined_group:
        pd.concat(combined_group, ignore_index=True).to_csv(tuning_root / "hyperparameter_tuning_validation_group_ranking.csv", index=False)
    review = _build_tuning_review_summary(metrics_all, topk_all)
    if not review.empty:
        review.to_csv(tuning_root / "hyperparameter_tuning_summary_for_review.csv", index=False)

    write_json(
        {
            "tuning_root": str(tuning_root),
            "dataset_output_dir": str(dataset_output_dir),
            "experiment_count": len(param_grid),
            "data_design": data_design,
            "grid": grid,
            "test_split_used": False,
            "summary_files": [
                "hyperparameter_tuning_run_summary.csv",
                "hyperparameter_tuning_validation_metrics.csv",
                "hyperparameter_tuning_validation_top_k.csv",
                "hyperparameter_tuning_validation_group_ranking.csv",
                "hyperparameter_tuning_summary_for_review.csv",
            ],
        },
        tuning_root / "hyperparameter_tuning_summary.json",
    )
    _banner(f"Hyperparameter tuning completed. Outputs: {tuning_root}")
    if 'review' in locals() and not review.empty:
        cols = [c for c in [
            "hyperparameter_experiment_id", "dataset_id", "evaluation_view", "algorithm",
            "top_5pct_precision_at_k", "top_5pct_lift_vs_random",
            "threshold_free_average_precision", "threshold_free_roc_auc",
            "fit_xgboost_best_iteration",
        ] if c in review.columns]
        print("Top rows from hyperparameter_tuning_summary_for_review.csv:", flush=True)
        print(review[cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
