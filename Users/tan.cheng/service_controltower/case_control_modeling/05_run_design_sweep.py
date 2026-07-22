"""Step 05: Phase 1 data-design sweep on validation only.

This sweep is intentionally compact:
  - no cross-validation is run,
  - no learning curves are saved,
  - no SHAP / feature-importance artifacts are saved,
  - no detailed prediction files are saved,
  - each experiment directly fits on the training split and evaluates validation views.

Use this step to choose data-design parameters such as controls-per-positive,
validation population-negative ratio, and XGBoost class weighting. Keep model
hyperparameter tuning for 06_tune_xgboost_hyperparameters.py.
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


PREP_STEPS = ["01_build_claim_episodes"]
DEFAULT_RUN_STEPS = ["02_build_case_control_dataset", "04_fit_validate_model_report"]

PARAMETER_ALIASES = {
    "controls_per_positive_case": "CONTROLS_PER_POSITIVE_CASE",
    "validation_random_negatives_per_positive": "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE",
    "asof_evaluation_max_machines_per_snapshot": "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT",
    "asof_evaluation_snapshot_frequency_days": "ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS",
    "xgboost_class_importance_mode": "XGBOOST_CLASS_IMPORTANCE_MODE",
    "xgboost_fixed_scale_pos_weight": "XGBOOST_FIXED_SCALE_POS_WEIGHT",
    "positive_claim_selection_mode": "POSITIVE_CLAIM_SELECTION_MODE",
    "random_state": "RANDOM_STATE",
}


def _banner(title: str) -> None:
    line = "=" * 88
    print("\n" + line)
    print(title)
    print(line, flush=True)


def _safe_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(".", "p")
    text = re.sub(r"[^a-zA-Z0-9_.=-]+", "_", text).strip("_")
    return text or "value"


def _short_hash(obj: Any, length: int = 8) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def _safe_id(text: str, max_len: int = 96, hash_payload: Any | None = None) -> str:
    text = str(text).strip() or "experiment"
    clean = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_") or "experiment"
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
        raise ValueError("DESIGN_SWEEP_GRID is empty.")
    keys = list(grid.keys())
    value_lists = []
    for key in keys:
        value = grid[key]
        if isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = [value]
        if not values:
            raise ValueError(f"DESIGN_SWEEP_GRID[{key!r}] has no values.")
        value_lists.append(values)
    experiments = []
    for combo in itertools.product(*value_lists):
        exp = dict(zip(keys, combo))
        experiments.append(_normalize_experiment(exp))
    return experiments


def _normalize_experiment(exp: Mapping[str, Any]) -> dict:
    out = dict(exp)
    for key, canonical in list(PARAMETER_ALIASES.items()):
        if key in out and canonical not in out:
            out[canonical] = out[key]

    # Convenience shortcut for XGBoost scale_pos_weight.
    if "scale_pos_weight" in out:
        value = out["scale_pos_weight"]
        if isinstance(value, str) and value.strip().lower() in {"none", "off", "false"}:
            out["XGBOOST_CLASS_IMPORTANCE_MODE"] = "none"
            out["XGBOOST_FIXED_SCALE_POS_WEIGHT"] = 1.0
        elif isinstance(value, str) and value.strip().lower() == "auto":
            out["XGBOOST_CLASS_IMPORTANCE_MODE"] = "auto"
            out["XGBOOST_FIXED_SCALE_POS_WEIGHT"] = 1.0
        else:
            out["XGBOOST_CLASS_IMPORTANCE_MODE"] = "fixed"
            out["XGBOOST_FIXED_SCALE_POS_WEIGHT"] = float(value)

    parts = []
    preferred = [
        "CONTROLS_PER_POSITIVE_CASE",
        "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT",
        "ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS",
        "scale_pos_weight",
        "RANDOM_STATE",
        "POSITIVE_CLAIM_SELECTION_MODE",
    ]
    short_names = {
        "CONTROLS_PER_POSITIVE_CASE": "ctrl",
        "ASOF_EVALUATION_MAX_MACHINES_PER_SNAPSHOT": "asofm",
        "ASOF_EVALUATION_SNAPSHOT_FREQUENCY_DAYS": "asoffreq",
        "scale_pos_weight": "spw",
        "RANDOM_STATE": "seed",
        "POSITIVE_CLAIM_SELECTION_MODE": "claim",
    }
    for key in preferred:
        if key in out:
            parts.append(f"{short_names[key]}{_safe_token(out[key])}")

    # Store only a short human-readable label here. The final experiment_id is
    # assigned later with a numeric prefix, such as exp001__ctrl3__valneg10__spwnone.
    # Do NOT include fixed policy flags in folder names; full config details are
    # saved in experiment_config.json and the combined summary CSVs.
    out["experiment_label"] = _safe_id("__".join(parts) or "design", max_len=48, hash_payload=None)
    out.pop("experiment_id", None)
    return out


def _assign_numbered_experiment_ids(experiments: list[dict], prefix: str = "exp") -> list[dict]:
    """Assign short, stable folder-safe experiment IDs.

    The ID is intentionally short to avoid OS path-length errors. Full parameter
    details remain available in experiment_config.json and summary CSV columns.
    """
    out = []
    used = set()
    for i, exp in enumerate(experiments, start=1):
        item = dict(exp)
        label = _safe_id(item.get("experiment_label", "design"), max_len=40, hash_payload=None)
        base = f"{prefix}{i:03d}__{label}"
        exp_id = _safe_id(base, max_len=60, hash_payload=None)
        if exp_id in used:
            exp_id = _safe_id(f"{prefix}{i:03d}__{label}__h{_short_hash(item)}", max_len=60, hash_payload=None)
        used.add(exp_id)
        item["experiment_id"] = exp_id
        out.append(item)
    return out


def _snapshot_config(keys: list[str]) -> dict:
    return {k: getattr(config, k, None) for k in keys if hasattr(config, k)}


def _apply_config(exp: Mapping[str, Any]) -> dict:
    applied = {}
    skip = {"experiment_id", "description", "scale_pos_weight"}
    for key, value in exp.items():
        if key in skip or key.islower():
            continue
        setattr(config, key, value)
        applied[key] = value
    return applied


def _apply_phase1_policy(applied: dict) -> None:
    """Force Phase 1 to be a compact data-design sweep."""
    # No CV, no learning curves, no early stopping, no SHAP, no feature-importance,
    # and no detailed validation prediction files during the design sweep.
    config.XGBOOST_ENABLE_LEARNING_CURVE = False
    config.XGBOOST_USE_EARLY_STOPPING = False
    config.XGBOOST_EARLY_STOPPING_ROUNDS = 0
    config.SAVE_FEATURE_IMPORTANCE = False
    config.SAVE_SHAP_VALUES = False
    config.VALIDATION_SAVE_DETAILED_OUTPUTS = False
    config.VALIDATION_INCLUDE_FEATURE_COLUMNS = False
    config.VALIDATION_SAVE_MODEL_ARTIFACTS = False

    applied["XGBOOST_ENABLE_LEARNING_CURVE"] = False
    applied["XGBOOST_USE_EARLY_STOPPING"] = False
    applied["XGBOOST_EARLY_STOPPING_ROUNDS"] = 0
    applied["SAVE_FEATURE_IMPORTANCE"] = False
    applied["SAVE_SHAP_VALUES"] = False
    applied["VALIDATION_SAVE_DETAILED_OUTPUTS"] = False
    applied["VALIDATION_INCLUDE_FEATURE_COLUMNS"] = False
    applied["VALIDATION_SAVE_MODEL_ARTIFACTS"] = False

    # Keep Phase 1 from explicitly tuning L1/L2. XGBoost defaults still apply.
    if isinstance(getattr(config, "XGBOOST_PARAMS", None), dict):
        config.XGBOOST_PARAMS = dict(config.XGBOOST_PARAMS)
        config.XGBOOST_PARAMS.pop("reg_alpha", None)
        config.XGBOOST_PARAMS.pop("reg_lambda", None)
    applied["phase1_regularization_tuning"] = "disabled"
    applied["phase1_early_stopping"] = "disabled"
    applied["phase1_cv"] = "disabled"
    applied["phase1_learning_curve_artifacts"] = "disabled"


def _run_step(module_name: str) -> None:
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"Module {module_name} does not define run().")
    module.run()


def _read_if_exists(path: Path, experiment_id: str, applied: Mapping[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "experiment_id", experiment_id)
    for key, value in applied.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            df[f"config_{key}"] = value
    return df


def _collect_experiment_outputs(exp_dir: Path, experiment_id: str, applied: Mapping[str, Any]) -> dict:
    paths = {
        "validation_metrics": exp_dir / "04_fit_validate_model_report" / "validation_metrics_all_datasets.csv",
        "validation_top_k": exp_dir / "04_fit_validate_model_report" / "validation_top_k_all_datasets.csv",
        "validation_group_ranking": exp_dir / "04_fit_validate_model_report" / "validation_group_ranking_metrics_all_datasets.csv",
        "validation_horizon_trend_summary_for_review": exp_dir / "04_fit_validate_model_report" / "validation_horizon_trend_summary_for_review.csv",
        "dataset_index": exp_dir / "02_case_control_datasets" / "dataset_index.csv",
    }
    collected = {}
    for name, path in paths.items():
        df = _read_if_exists(path, experiment_id, applied)
        if not df.empty:
            collected[name] = df
    return collected


def _topk_label(rate: float) -> str:
    pct = int(round(float(rate) * 100))
    return f"top_{pct}pct"


def _build_review_summary(metrics: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    # Include evaluation_horizon_days in the key because Step 04 can now compute
    # validation metrics for a sweep of future-claim horizons using the same score.
    key_cols = ["experiment_id", "dataset_id", "algorithm", "evaluation_view", "evaluation_horizon_days"]
    key_cols = [c for c in key_cols if c in metrics.columns]
    config_cols = [c for c in metrics.columns if c.startswith("config_")]
    metric_cols = [
        c for c in [
            "status",
            "train_rows",
            "evaluation_rows",
            "train_positive_rows",
            "evaluation_positive_rows",
            "evaluation_positive_rate",
            "evaluation_target_col",
            "evaluation_target_mode",
            "lead_time_future_claim_observed_rows",
            "lead_time_future_claim_never_observed_rows",
            "lead_time_evaluation_positive_days_median",
            "threshold_free_average_precision",
            "threshold_free_roc_auc",
            "threshold_free_positive_rate",
            "fit_xgboost_class_importance_mode",
            "fit_xgboost_scale_pos_weight",
        ] if c in metrics.columns
    ]
    base = metrics[key_cols + metric_cols + config_cols].drop_duplicates(key_cols).copy()

    if not topk.empty:
        tk = topk.copy()
        tk["top_k_label"] = tk["top_k_rate"].map(_topk_label)
        if {"precision_at_k", "flagged_count"}.issubset(tk.columns):
            tk["positive_hits_at_k"] = tk["precision_at_k"] * tk["flagged_count"]
        value_cols = [
            c for c in [
                "precision_at_k",
                "recall_at_k",
                "lift_vs_random",
                "flagged_count",
                "positive_hits_at_k",
            ] if c in tk.columns
        ]
        wide_parts = []
        for value_col in value_cols:
            pivot = tk.pivot_table(
                index=key_cols,
                columns="top_k_label",
                values=value_col,
                aggfunc="first",
            )
            pivot.columns = [f"{label}_{value_col}" for label in pivot.columns]
            wide_parts.append(pivot.reset_index())
        if wide_parts:
            wide = wide_parts[0]
            for extra in wide_parts[1:]:
                wide = wide.merge(extra, on=key_cols, how="outer")
            base = base.merge(wide, on=key_cols, how="left")

    preferred_sort = [
        "top_5pct_precision_at_k",
        "threshold_free_average_precision",
        "top_10pct_precision_at_k",
        "threshold_free_roc_auc",
    ]
    eval_priority = {
        "population_like_validation": 0,
        "validation_with_population_negatives": 1,
        "matched_validation": 2,
    }
    base["evaluation_view_priority"] = base["evaluation_view"].map(eval_priority).fillna(9).astype(int)
    base["evaluation_horizon_days_sort"] = pd.to_numeric(base.get("evaluation_horizon_days", float("nan")), errors="coerce").fillna(-1)
    sort_cols = ["evaluation_view_priority", "evaluation_horizon_days_sort"] + [c for c in preferred_sort if c in base.columns]
    ascending = [True, True] + [False] * (len(sort_cols) - 2)
    out = base.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["evaluation_view_priority", "evaluation_horizon_days_sort"], errors="ignore")


def run() -> None:
    original_output_dir = config.OUTPUT_DIR
    sweep_root = original_output_dir / "05_design_sweep"
    ensure_dir(sweep_root)

    grid = getattr(config, "DESIGN_SWEEP_GRID", {}) or {}
    fixed_overrides = dict(getattr(config, "DESIGN_SWEEP_FIXED_OVERRIDES", {}) or {})
    raw_experiments = [_normalize_experiment({**exp, **fixed_overrides}) for exp in _expand_grid(grid)]
    experiments = _assign_numbered_experiment_ids(raw_experiments, prefix="exp")
    if not experiments:
        raise ValueError("No design-sweep experiments were generated from DESIGN_SWEEP_GRID.")

    # Design sweep must not run cross-validation.  Ignore any older config value
    # that still asks for CV and force direct train->validation evaluation.
    configured_steps = list(getattr(config, "DESIGN_SWEEP_RUN_STEPS", DEFAULT_RUN_STEPS) or DEFAULT_RUN_STEPS)
    run_steps = [s for s in configured_steps if s != "03_cross_validation"]
    if "02_build_case_control_dataset" not in run_steps:
        run_steps.insert(0, "02_build_case_control_dataset")
    if "04_fit_validate_model_report" not in run_steps:
        run_steps.append("04_fit_validate_model_report")

    restore_keys = sorted({
        "OUTPUT_DIR",
        "CONTROLS_PER_POSITIVE_CASE",
        "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_VALIDATION",
        "ADD_POPULATION_RANDOM_NEGATIVES_TO_TEST",
        "XGBOOST_CLASS_IMPORTANCE_MODE",
        "XGBOOST_FIXED_SCALE_POS_WEIGHT",
        "POSITIVE_CLAIM_SELECTION_MODE",
        "WINDOW_CONFIGS",
        "XGBOOST_PARAMS",
        "XGBOOST_ENABLE_LEARNING_CURVE",
        "XGBOOST_USE_EARLY_STOPPING",
        "XGBOOST_EARLY_STOPPING_ROUNDS",
        "SAVE_FEATURE_IMPORTANCE",
        "SAVE_SHAP_VALUES",
        "SHAP_EVALUATION_VIEWS",
        "VALIDATION_SAVE_DETAILED_OUTPUTS",
        "VALIDATION_INCLUDE_FEATURE_COLUMNS",
        "VALIDATION_SAVE_MODEL_ARTIFACTS",
        "RANDOM_STATE",
    } | {k for exp in experiments for k in exp if k.isupper()})
    original_values = _snapshot_config(restore_keys)

    combined = {
        "validation_metrics": [],
        "validation_top_k": [],
        "validation_group_ranking": [],
        "validation_horizon_trend_summary_for_review": [],
        "dataset_index": [],
    }
    run_rows = []

    _banner(f"Starting Phase 1 design sweep: {len(experiments)} experiments, CV disabled")
    print(f"Sweep root: {sweep_root}", flush=True)
    print(f"Run steps per experiment: {PREP_STEPS + run_steps}", flush=True)

    try:
        for exp_num, exp in enumerate(experiments, start=1):
            experiment_id = exp["experiment_id"]
            exp_dir = sweep_root / experiment_id
            ensure_dir(exp_dir)
            _banner(f"[Design sweep {exp_num}/{len(experiments)}] {experiment_id}")

            for key, value in original_values.items():
                setattr(config, key, value)
            config.OUTPUT_DIR = exp_dir
            applied = _apply_config(exp)
            _apply_phase1_policy(applied)
            applied["OUTPUT_DIR"] = str(exp_dir)
            write_json({"experiment_id": experiment_id, "applied_config": applied}, exp_dir / "experiment_config.json")

            status = "completed"
            error = ""
            step_list = PREP_STEPS + run_steps
            try:
                for step_num, step in enumerate(step_list, start=1):
                    print(f"\n>>> [Design sweep {exp_num}/{len(experiments)} | Step {step_num}/{len(step_list)}] {step}", flush=True)
                    _run_step(step)
            except Exception as exc:
                status = "failed"
                error = str(exc)
                traceback.print_exc()

            outputs = _collect_experiment_outputs(exp_dir, experiment_id, applied)
            for name, df in outputs.items():
                combined[name].append(df)
            run_rows.append({
                "experiment_id": experiment_id,
                "experiment_number": exp_num,
                "experiment_count": len(experiments),
                "status": status,
                "error": error,
                "output_dir": str(exp_dir),
                **{f"config_{k}": v for k, v in applied.items() if isinstance(v, (str, int, float, bool)) or v is None},
            })

    finally:
        for key, value in original_values.items():
            setattr(config, key, value)
        config.OUTPUT_DIR = original_output_dir

    run_summary = pd.DataFrame(run_rows)
    run_summary.to_csv(sweep_root / "design_sweep_run_summary.csv", index=False)

    combined_written = {}
    for name, frames in combined.items():
        if frames:
            out = pd.concat(frames, ignore_index=True)
            path = sweep_root / f"design_sweep_{name}.csv"
            out.to_csv(path, index=False)
            combined_written[name] = out

    review = _build_review_summary(
        combined_written.get("validation_metrics", pd.DataFrame()),
        combined_written.get("validation_top_k", pd.DataFrame()),
    )
    if not review.empty:
        review.to_csv(sweep_root / "design_sweep_validation_summary_for_review.csv", index=False)

    write_json(
        {
            "sweep_root": str(sweep_root),
            "experiment_count": len(experiments),
            "grid": grid,
            "fixed_overrides": fixed_overrides,
            "run_steps": run_steps,
            "cross_validation_enabled": False,
            "phase1_policy": [
                "direct fit on training split and validate on validation views",
                "no cross-validation",
                "no explicit L1/L2 tuning",
                "no early stopping",
                "no learning-curve artifacts",
                "no SHAP / feature-importance artifacts",
                "no detailed prediction files",
            ],
            "summary_files": [
                "design_sweep_run_summary.csv",
                "design_sweep_validation_summary_for_review.csv",
                "design_sweep_validation_metrics.csv",
                "design_sweep_validation_top_k.csv",
                "design_sweep_validation_group_ranking.csv",
                "design_sweep_validation_horizon_trend_summary_for_review.csv",
                "design_sweep_dataset_index.csv",
            ],
        },
        sweep_root / "design_sweep_summary.json",
    )
    _banner(f"Design sweep completed. Outputs: {sweep_root}")
    if not review.empty:
        cols = [c for c in [
            "experiment_id", "dataset_id", "evaluation_view", "algorithm",
            "top_5pct_precision_at_k", "top_5pct_lift_vs_random",
            "threshold_free_average_precision", "threshold_free_roc_auc",
        ] if c in review.columns]
        print("Top rows from design_sweep_validation_summary_for_review.csv:", flush=True)
        print(review[cols].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
