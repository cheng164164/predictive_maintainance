"""Step 04: fit models on training split and evaluate validation views only.

This step runs after 03_cross_validation.py. It fits configured models on the
full chronological training split and evaluates only the validation views:
matched validation, matched validation plus population negatives, and
population-like validation when available.

The test split is intentionally not loaded or scored here. Test evaluation is
reserved for 07_final_test_evaluation.py after data-design choices and model
hyperparameters are locked.

For XGBoost, this step can save:
- train vs validation learning curves from eval_set,
- model feature importance,
- booster gain/weight/cover importance,
- SHAP contribution values from XGBoost pred_contribs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from cc_utils import (
    ensure_dir,
    fit_model_pipeline,
    configured_evaluation_horizons,
    get_evaluation_target,
    future_claim_lead_time_summary,
    make_model_pipeline,
    metrics_at_threshold,
    model_feature_importance_frame,
    predict_score,
    summarize_xgboost_learning_curve,
    threshold_free_metrics,
    top_k_metrics,
    transform_with_fitted_preprocessor,
    validate_dataset_features,
    write_json,
    xgboost_booster_importance_frame,
    xgboost_learning_curve_frame,
)

DATE_COLUMNS = [
    "window_start",
    "window_end",
    "future_claim_date",
    "control_no_claim_start",
    "control_no_claim_end",
]


def _load_dataset_index(dataset_index_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(dataset_index_path) if dataset_index_path is not None else config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset index not found: {path}. Run 02_build_case_control_dataset.py first.")
    return pd.read_csv(path)


def _read_dataset(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _existing_path(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text if Path(text).exists() else None


def _evaluation_views(dataset_row: pd.Series) -> list[tuple[str, str]]:
    candidates = [
        ("matched_validation", dataset_row.get("validation_dataset_path")),
        # Recommended production-like validation view: rows are sampled from
        # realistic machine/as-of-date snapshots and labeled by future horizons.
        ("asof_population_validation", dataset_row.get("validation_asof_population_dataset_path")),
        # Legacy views retained only if older files are present.
        ("validation_with_population_negatives", dataset_row.get("validation_with_population_negatives_path")),
        ("population_like_validation", dataset_row.get("validation_population_like_dataset_path")),
    ]
    out: list[tuple[str, str]] = []
    seen = set()
    for name, path_value in candidates:
        path = _existing_path(path_value)
        if path and path not in seen:
            out.append((name, path))
            seen.add(path)
    return out


def _find_fit_eval_view(eval_views: list[tuple[str, str]]) -> tuple[str, str]:
    preferred = str(getattr(config, "XGBOOST_LEARNING_CURVE_EVAL_VIEW", "matched_validation"))
    for name, path in eval_views:
        if name == preferred:
            return name, path
    for name, path in eval_views:
        if "validation" in name:
            return name, path
    if not eval_views:
        raise ValueError("No evaluation views available for learning-curve monitoring.")
    return eval_views[0]


def _configured_threshold() -> float:
    return float(getattr(config, "VALIDATION_SCORE_THRESHOLD", 0.50))


def _configured_top_k_rates() -> list[float]:
    rates = getattr(config, "VALIDATION_TOP_K_RATES", None)
    if rates is None:
        rates = getattr(config, "HOLDOUT_TOP_K_RATES", [0.01, 0.05, 0.10, 0.20])
    return [float(x) for x in rates]


def _safe_name(text: str) -> str:
    import re
    text = str(text).strip() or "item"
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def _topk_label(rate: float) -> str:
    pct = int(round(float(rate) * 100))
    return f"top_{pct}pct"


def _is_scalar(value) -> bool:
    return isinstance(value, (str, int, float, bool, np.integer, np.floating)) or value is None or pd.isna(value)


def _fit_metadata_for_metrics(fit_metadata: Mapping) -> dict:
    out = {}
    for key, value in (fit_metadata or {}).items():
        if str(key).startswith("_"):
            continue
        if _is_scalar(value):
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            out[f"fit_{key}"] = value
    return out


def _add_prediction_columns(pred: pd.DataFrame, score: np.ndarray, threshold: float, top_k_rates: Iterable[float]) -> pd.DataFrame:
    out = pred.copy().reset_index(drop=True)
    out["score"] = np.asarray(score, dtype=float)
    out["predicted_label"] = (out["score"] >= threshold).astype(int)
    out["score_rank_overall"] = out["score"].rank(method="first", ascending=False).astype(int)
    n = len(out)
    if n:
        out["score_percentile"] = 1.0 - ((out["score_rank_overall"] - 1) / n)
    else:
        out["score_percentile"] = np.nan
    for rate in top_k_rates:
        k = max(1, int(np.ceil(n * float(rate)))) if n else 0
        col = f"top_{int(round(float(rate) * 100))}pct_flag"
        out[col] = (out["score_rank_overall"] <= k).astype(int) if k else 0
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
    if include_features:
        feature_cols = [c for c in list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES) if c in pred.columns]
    else:
        feature_cols = []
    extra_cols = [c for c in pred.columns if c not in priority_cols and c not in feature_cols]
    return pred[priority_cols + feature_cols + extra_cols]


def _summarize_machine_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["dataset_id", "algorithm", "evaluation_view", "machine_key"]
    for keys, g in pred.groupby(group_cols, dropna=False):
        g_score = g.sort_values("score", ascending=False, kind="mergesort")
        g_time = g.sort_values("window_end", kind="mergesort") if "window_end" in g.columns else g
        top = g_score.iloc[0]
        latest = g_time.iloc[-1]
        rows.append({
            "dataset_id": keys[0],
            "algorithm": keys[1],
            "evaluation_view": keys[2],
            "machine_key": keys[3],
            "full_model": top.get("full_model", ""),
            "serial": top.get("serial", ""),
            "window_rows": int(len(g)),
            "positive_window_rows": int(pd.to_numeric(g.get("evaluation_target", g.get("target", 0)), errors="coerce").fillna(0).sum()),
            "case_rows": int(g.get("row_role", pd.Series(dtype=str)).astype(str).eq("case").sum()),
            "control_rows": int(g.get("row_role", pd.Series(dtype=str)).astype(str).eq("control").sum()),
            "population_random_negative_rows": int(g.get("row_role", pd.Series(dtype=str)).astype(str).str.contains("population", case=False, na=False).sum()),
            "max_score": float(g["score"].max()),
            "mean_score": float(g["score"].mean()),
            "min_score": float(g["score"].min()),
            "latest_window_end": latest.get("window_end", pd.NaT),
            "latest_window_score": float(latest.get("score", np.nan)),
            "top_score_window_start": top.get("window_start", pd.NaT),
            "top_score_window_end": top.get("window_end", pd.NaT),
            "top_score": float(top.get("score", np.nan)),
            "top_score_target": int(top.get("target", 0)) if pd.notna(top.get("target", np.nan)) else np.nan,
            "top_score_evaluation_target": int(top.get("evaluation_target", top.get("target", 0))) if pd.notna(top.get("evaluation_target", top.get("target", np.nan))) else np.nan,
            "top_score_days_to_next_claim": top.get("days_to_next_claim_on_or_after_window_end", np.nan),
            "top_score_row_role": top.get("row_role", ""),
            "top_score_group_id": top.get("case_control_group_id", ""),
            "top_score_claim_episode_id": top.get("claim_episode_id", ""),
            "top_score_future_claim_date": top.get("future_claim_date", pd.NaT),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "algorithm", "evaluation_view", "max_score", "mean_score"],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    )


def _summarize_case_control_group_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "case_control_group_id" not in pred.columns:
        return pd.DataFrame()
    rows = []
    group_cols = ["dataset_id", "algorithm", "evaluation_view", "case_control_group_id"]
    for keys, g in pred.groupby(group_cols, dropna=False):
        ranked = g.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        ranked["rank_within_group"] = np.arange(1, len(ranked) + 1)
        group_target_col = "evaluation_target" if "evaluation_target" in ranked.columns else "target"
        cases = ranked[pd.to_numeric(ranked[group_target_col], errors="coerce").fillna(0).astype(int).eq(1)] if group_target_col in ranked.columns else ranked.iloc[0:0]
        controls = ranked[pd.to_numeric(ranked[group_target_col], errors="coerce").fillna(0).astype(int).eq(0)] if group_target_col in ranked.columns else ranked.iloc[0:0]
        case = cases.iloc[0] if len(cases) else None
        case_rank = int(case["rank_within_group"]) if case is not None else np.nan
        case_score = float(case["score"]) if case is not None else np.nan
        max_control_score = float(controls["score"].max()) if len(controls) else np.nan
        rows.append({
            "dataset_id": keys[0],
            "algorithm": keys[1],
            "evaluation_view": keys[2],
            "case_control_group_id": keys[3],
            "window_name": ranked.iloc[0].get("window_name", ""),
            "window_start": ranked.iloc[0].get("window_start", pd.NaT),
            "window_end": ranked.iloc[0].get("window_end", pd.NaT),
            "group_size": int(len(ranked)),
            "case_count": int(len(cases)),
            "control_count": int(len(controls)),
            "case_machine_key": case.get("machine_key", "") if case is not None else "",
            "case_full_model": case.get("full_model", "") if case is not None else "",
            "case_serial": case.get("serial", "") if case is not None else "",
            "case_score": case_score,
            "case_rank_within_group": case_rank,
            "case_is_top_score": bool(case_rank == 1) if pd.notna(case_rank) else False,
            "case_is_top_2": bool(case_rank <= 2) if pd.notna(case_rank) else False,
            "max_control_score": max_control_score,
            "case_score_minus_max_control_score": float(case_score - max_control_score) if pd.notna(case_score) and pd.notna(max_control_score) else np.nan,
            "top_rank_machine_key": ranked.iloc[0].get("machine_key", ""),
            "top_rank_row_role": ranked.iloc[0].get("row_role", ""),
            "top_rank_target": int(ranked.iloc[0].get("target", 0)) if pd.notna(ranked.iloc[0].get("target", np.nan)) else np.nan,
            "top_rank_evaluation_target": int(ranked.iloc[0].get("evaluation_target", ranked.iloc[0].get("target", 0))) if pd.notna(ranked.iloc[0].get("evaluation_target", ranked.iloc[0].get("target", np.nan))) else np.nan,
            "top_rank_days_to_next_claim": ranked.iloc[0].get("days_to_next_claim_on_or_after_window_end", np.nan),
            "top_rank_score": float(ranked.iloc[0].get("score", np.nan)),
            "future_claim_date": case.get("future_claim_date", pd.NaT) if case is not None else pd.NaT,
            "claim_episode_id": case.get("claim_episode_id", "") if case is not None else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_id", "algorithm", "evaluation_view", "window_end", "case_control_group_id"],
        kind="mergesort",
    )


def _group_ranking_metrics(group_summary: pd.DataFrame) -> dict:
    if group_summary.empty:
        return {}
    usable = group_summary[group_summary["case_count"] > 0].copy()
    if usable.empty:
        return {}
    ranks = pd.to_numeric(usable["case_rank_within_group"], errors="coerce")
    return {
        "case_control_groups": int(len(usable)),
        "mean_case_rank_within_group": float(ranks.mean()),
        "median_case_rank_within_group": float(ranks.median()),
        "case_top_score_rate": float(usable["case_is_top_score"].mean()),
        "case_top_2_rate": float(usable["case_is_top_2"].mean()),
        "mean_reciprocal_case_rank": float((1.0 / ranks.replace(0, np.nan)).mean()),
        "mean_case_score_minus_max_control_score": float(pd.to_numeric(usable["case_score_minus_max_control_score"], errors="coerce").mean()),
    }


def _plot_horizontal_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, top_n: int = 30) -> None:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return
    plot_df = df[[label_col, value_col]].dropna().sort_values(value_col, ascending=False).head(top_n)
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values(value_col, ascending=True)
    ensure_dir(path.parent)
    fig_height = max(4, min(12, 0.30 * len(plot_df) + 1.5))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_learning_curve_outputs(model, fit_metadata: Mapping, dataset_id: str, algorithm: str, output_dir: Path) -> dict:
    if str(algorithm).lower() != "xgboost":
        return {}
    curve = xgboost_learning_curve_frame(model, fit_metadata)
    if curve.empty:
        return {}
    curve_dir = output_dir / "learning_curves"
    plot_dir = output_dir / "plots" / "learning_curves"
    ensure_dir(curve_dir)
    ensure_dir(plot_dir)
    curve_path = curve_dir / f"{dataset_id}__{algorithm}__learning_curve.csv"
    curve.to_csv(curve_path, index=False)
    eval_label = str((fit_metadata or {}).get("xgboost_eval_name", "validation"))
    summary = summarize_xgboost_learning_curve(curve, eval_label=eval_label)
    summary_path = curve_dir / f"{dataset_id}__{algorithm}__learning_curve_summary.csv"
    if not summary.empty:
        summary.to_csv(summary_path, index=False)
    for metric, g in curve.groupby("metric", dropna=False):
        fig, ax = plt.subplots(figsize=(9, 5))
        for dataset_label, sub in g.groupby("dataset_label", dropna=False):
            sub = sub.sort_values("iteration")
            ax.plot(sub["iteration"], sub["value"], label=str(dataset_label))
        ax.set_title(f"{dataset_id} {algorithm} learning curve: {metric}")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel(str(metric))
        ax.legend()
        fig.tight_layout()
        plot_path = plot_dir / f"{dataset_id}__{algorithm}__learning_curve__{_safe_name(metric)}.png"
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    return {
        "learning_curve_path": str(curve_path),
        "learning_curve_summary_path": str(summary_path),
    }


def _save_feature_importance_outputs(model, algorithm: str, dataset_id: str, output_dir: Path) -> dict:
    if not bool(getattr(config, "SAVE_FEATURE_IMPORTANCE", True)):
        return {}
    importance_dir = output_dir / "feature_importance"
    plot_dir = output_dir / "plots" / "feature_importance"
    ensure_dir(importance_dir)
    ensure_dir(plot_dir)
    saved = {}

    imp = model_feature_importance_frame(model, algorithm)
    if not imp.empty:
        imp.insert(0, "dataset_id", dataset_id)
        path = importance_dir / f"{dataset_id}__{algorithm}__pipeline_feature_importance.csv"
        imp.to_csv(path, index=False)
        saved["pipeline_feature_importance_path"] = str(path)
        value_col = "absolute_value" if "absolute_value" in imp.columns else imp.columns[-1]
        _plot_horizontal_bar(
            imp,
            label_col="prepared_feature",
            value_col=value_col,
            title=f"{dataset_id} {algorithm} top pipeline feature importance",
            path=plot_dir / f"{dataset_id}__{algorithm}__pipeline_feature_importance_top30.png",
        )

    xgb_imp = xgboost_booster_importance_frame(model, algorithm)
    if not xgb_imp.empty:
        xgb_imp.insert(0, "dataset_id", dataset_id)
        path = importance_dir / f"{dataset_id}__{algorithm}__xgboost_booster_importance.csv"
        xgb_imp.to_csv(path, index=False)
        saved["xgboost_booster_importance_path"] = str(path)
        gain = xgb_imp[xgb_imp["importance_type"].eq("gain")].copy()
        _plot_horizontal_bar(
            gain,
            label_col="prepared_feature",
            value_col="importance_value",
            title=f"{dataset_id} {algorithm} XGBoost gain importance",
            path=plot_dir / f"{dataset_id}__{algorithm}__xgboost_gain_importance_top30.png",
        )
    return saved


def _configured_shap_views() -> set[str]:
    views = getattr(config, "SHAP_EVALUATION_VIEWS", ["matched_validation", "population_like_validation"])
    if views is None:
        return set()
    return {str(v) for v in views}


def _select_shap_indices(score: np.ndarray, n_rows: int) -> tuple[np.ndarray, dict[int, str]]:
    if n_rows <= 0:
        return np.array([], dtype=int), {}
    max_rows = int(getattr(config, "SHAP_MAX_ROWS", 1000) or 1000)
    top_rows = int(getattr(config, "SHAP_TOP_SCORE_ROWS", max_rows // 2) or 0)
    random_rows = int(getattr(config, "SHAP_RANDOM_ROWS", max_rows - top_rows) or 0)
    max_rows = max(1, min(max_rows, n_rows))
    order = np.argsort(np.asarray(score, dtype=float))[::-1]
    top_idx = list(order[: min(top_rows, n_rows, max_rows)])
    remaining = [int(i) for i in range(n_rows) if int(i) not in set(top_idx)]
    sample_kind = {int(i): "top_score" for i in top_idx}
    rng = np.random.default_rng(int(getattr(config, "RANDOM_STATE", 42)))
    random_n = min(random_rows, max_rows - len(top_idx), len(remaining))
    random_idx = []
    if random_n > 0:
        random_idx = [int(i) for i in rng.choice(remaining, size=random_n, replace=False)]
        for i in random_idx:
            sample_kind[i] = "random"
    indices = np.array(top_idx + random_idx, dtype=int)
    return indices, sample_kind


def _save_shap_outputs(
    model,
    algorithm: str,
    dataset_id: str,
    view_name: str,
    eval_df: pd.DataFrame,
    X_eval: pd.DataFrame,
    score: np.ndarray,
    output_dir: Path,
) -> dict:
    if str(algorithm).lower() != "xgboost" or not bool(getattr(config, "SAVE_SHAP_VALUES", True)):
        return {}
    if _configured_shap_views() and view_name not in _configured_shap_views():
        return {}
    if eval_df.empty:
        return {}
    try:
        from xgboost import DMatrix
    except Exception as exc:
        return {"shap_status": f"skipped_xgboost_import_failed: {exc}"}

    indices, sample_kind = _select_shap_indices(score, len(eval_df))
    if len(indices) == 0:
        return {"shap_status": "skipped_no_rows"}

    X_sample = X_eval.iloc[indices].copy()
    matrix, feature_names = transform_with_fitted_preprocessor(model, X_sample)
    try:
        booster = model.named_steps["model"].get_booster()
        dmat = DMatrix(matrix)
        pred_kwargs = {"pred_contribs": True, "validate_features": False}
        xgb_model = model.named_steps["model"]
        best_iteration = getattr(xgb_model, "best_iteration", None)
        if best_iteration is not None:
            try:
                best_int = int(best_iteration)
                if best_int >= 0:
                    pred_kwargs["iteration_range"] = (0, best_int + 1)
            except Exception:
                pass
        try:
            contrib = booster.predict(dmat, **pred_kwargs)
        except TypeError:
            pred_kwargs.pop("iteration_range", None)
            contrib = booster.predict(dmat, **pred_kwargs)
    except Exception as exc:
        return {"shap_status": f"failed: {exc}"}

    if contrib.ndim == 3:
        # Binary classifiers can occasionally return a 3D array depending on version.
        contrib = contrib[:, :, -1]
    shap_values = np.asarray(contrib[:, :-1], dtype=float)
    bias = np.asarray(contrib[:, -1], dtype=float)
    if shap_values.shape[1] != len(feature_names):
        feature_names = [f"f{i}" for i in range(shap_values.shape[1])]

    summary = pd.DataFrame({
        "prepared_feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "min_shap": shap_values.min(axis=0),
        "max_shap": shap_values.max(axis=0),
        "nonzero_rate": (np.abs(shap_values) > 1e-12).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False, kind="mergesort").reset_index(drop=True)
    summary.insert(0, "evaluation_view", view_name)
    summary.insert(0, "algorithm", algorithm)
    summary.insert(0, "dataset_id", dataset_id)

    max_features = int(getattr(config, "SHAP_MAX_FEATURES_IN_ROW_OUTPUT", 50) or 50)
    top_features = summary.head(max_features)["prepared_feature"].tolist()
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    col_map_rows = []
    shap_cols = {}
    for out_idx, fname in enumerate(top_features):
        source_idx = feature_to_idx[fname]
        col = f"shap_f{out_idx:04d}"
        shap_cols[col] = shap_values[:, source_idx]
        col_map_rows.append({
            "shap_column": col,
            "prepared_feature": fname,
            "source_feature_index": int(source_idx),
            "rank_by_mean_abs_shap": int(out_idx + 1),
        })

    meta_cols = [
        "machine_key", "full_model", "serial", "target", "evaluation_target", "row_role",
        "window_start", "window_end", "future_claim_date", "next_claim_date_on_or_after_window_end",
        "days_to_next_claim_on_or_after_window_end", "future_claim_lead_time_bucket",
        "case_control_group_id", "claim_episode_id",
    ]
    meta_cols = [c for c in meta_cols if c in eval_df.columns]
    meta = eval_df.iloc[indices][meta_cols].reset_index(drop=True).copy()
    meta.insert(0, "source_eval_row_index", indices.astype(int))
    meta.insert(1, "shap_sample_kind", [sample_kind.get(int(i), "sample") for i in indices])
    meta.insert(2, "score", np.asarray(score, dtype=float)[indices])
    wide = pd.concat([meta, pd.DataFrame(shap_cols)], axis=1)
    wide["shap_bias_term"] = bias

    shap_dir = output_dir / "shap_values"
    plot_dir = output_dir / "plots" / "shap"
    ensure_dir(shap_dir)
    ensure_dir(plot_dir)
    prefix = f"{dataset_id}__{algorithm}__{_safe_name(view_name)}"
    summary_path = shap_dir / f"{prefix}__shap_summary.csv"
    wide_path = shap_dir / f"{prefix}__shap_values_sample_wide.csv"
    map_path = shap_dir / f"{prefix}__shap_value_column_map.csv"
    summary.to_csv(summary_path, index=False)
    wide.to_csv(wide_path, index=False)
    pd.DataFrame(col_map_rows).to_csv(map_path, index=False)
    _plot_horizontal_bar(
        summary,
        label_col="prepared_feature",
        value_col="mean_abs_shap",
        title=f"{dataset_id} {algorithm} {view_name} mean absolute SHAP",
        path=plot_dir / f"{prefix}__mean_abs_shap_top30.png",
    )
    return {
        "shap_status": "saved",
        "shap_sample_rows": int(len(indices)),
        "shap_summary_path": str(summary_path),
        "shap_values_sample_wide_path": str(wide_path),
        "shap_column_map_path": str(map_path),
    }


def _configured_evaluation_horizon_sweep() -> list[Optional[int]]:
    """Return horizons to evaluate in Step 04.

    If EVALUATION_TARGET_MODE is training_target, there is no horizon sweep and
    the original target is evaluated once.  If EVALUATION_TARGET_MODE is
    claim_within_horizon, every configured horizon in
    EVALUATION_CLAIM_HORIZON_DAYS is evaluated with the same trained model and
    same scores.
    """
    mode = str(getattr(config, "EVALUATION_TARGET_MODE", "training_target")).strip().lower()
    if mode in {"training", "train", "target", "training_target", "original", "original_target"}:
        return [None]
    horizons = configured_evaluation_horizons(config)
    if not horizons:
        raise ValueError(
            "EVALUATION_CLAIM_HORIZON_DAYS must contain at least one horizon when "
            "EVALUATION_TARGET_MODE='claim_within_horizon'."
        )
    return [int(h) for h in horizons]


def _horizon_label(horizon: Optional[int]) -> str:
    return "training_target" if horizon is None else f"h{int(horizon)}d"


def _build_horizon_trend_summary(metrics: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    """Create one wide review table showing metric trends by horizon."""
    if metrics.empty:
        return pd.DataFrame()
    key_cols = ["dataset_id", "algorithm", "evaluation_view", "evaluation_horizon_days"]
    config_cols = [c for c in metrics.columns if c.startswith("config_") or c.startswith("fit_xgboost")]
    metric_cols = [
        c for c in [
            "status",
            "train_rows",
            "evaluation_rows",
            "train_positive_rows",
            "evaluation_training_target_positive_rows",
            "evaluation_positive_rows",
            "evaluation_positive_rate",
            "evaluation_target_col",
            "evaluation_target_mode",
            "lead_time_future_claim_observed_rows",
            "lead_time_future_claim_never_observed_rows",
            "lead_time_future_claim_days_median",
            "lead_time_future_claim_days_mean",
            "lead_time_evaluation_target_positive_rows",
            "lead_time_evaluation_target_positive_rate",
            "lead_time_evaluation_positive_days_median",
            "lead_time_evaluation_positive_days_max",
            "threshold_free_average_precision",
            "threshold_free_roc_auc",
            "threshold_free_positive_rate",
        ] if c in metrics.columns
    ]
    available_keys = [c for c in key_cols if c in metrics.columns]
    base = metrics[available_keys + metric_cols + config_cols].drop_duplicates(available_keys).copy()

    if not topk.empty:
        tk = topk.copy()
        tk["top_k_label"] = tk["top_k_rate"].map(_topk_label)
        # Add explicit positive hits at K for easier review.
        if {"precision_at_k", "flagged_count"}.issubset(tk.columns):
            tk["positive_hits_at_k"] = tk["precision_at_k"] * tk["flagged_count"]
        value_cols = [
            c for c in [
                "flagged_count",
                "positive_hits_at_k",
                "precision_at_k",
                "recall_at_k",
                "lift_vs_random",
                "min_score_in_top_k",
            ] if c in tk.columns
        ]
        wide_parts = []
        for value_col in value_cols:
            pivot = tk.pivot_table(
                index=available_keys,
                columns="top_k_label",
                values=value_col,
                aggfunc="first",
            )
            pivot.columns = [f"{label}_{value_col}" for label in pivot.columns]
            wide_parts.append(pivot.reset_index())
        if wide_parts:
            wide = wide_parts[0]
            for extra in wide_parts[1:]:
                wide = wide.merge(extra, on=available_keys, how="outer")
            base = base.merge(wide, on=available_keys, how="left")

    eval_priority = {
        "asof_population_validation": 0,
        "population_like_validation": 1,
        "validation_with_population_negatives": 2,
        "matched_validation": 3,
    }
    if "evaluation_view" in base.columns:
        base["evaluation_view_priority"] = base["evaluation_view"].map(eval_priority).fillna(9).astype(int)
    else:
        base["evaluation_view_priority"] = 9
    if "evaluation_horizon_days" in base.columns:
        base["evaluation_horizon_days_sort"] = pd.to_numeric(base["evaluation_horizon_days"], errors="coerce").fillna(-1)
    sort_cols = [c for c in ["dataset_id", "algorithm", "evaluation_view_priority", "evaluation_horizon_days_sort"] if c in base.columns]
    out = base.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out.drop(columns=[c for c in ["evaluation_view_priority", "evaluation_horizon_days_sort"] if c in out.columns])


def _train_score_one_dataset(dataset_row: pd.Series, output_dir: Path) -> dict:
    dataset_id = str(dataset_row["dataset_id"])
    train_path = dataset_row["training_dataset_path"]

    train_df = _read_dataset(train_path)
    validate_dataset_features(train_df, config)
    X_train = train_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
    y_train = train_df["target"].astype(int)

    eval_views = _evaluation_views(dataset_row)
    fit_eval_name, fit_eval_path = _find_fit_eval_view(eval_views)
    fit_eval_df = _read_dataset(fit_eval_path)
    validate_dataset_features(fit_eval_df, config)
    X_fit_eval = fit_eval_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
    # Learning-curve / early-stopping monitoring intentionally uses the original
    # training target so model fitting remains tied to the training objective.
    y_fit_eval = fit_eval_df["target"].astype(int)

    threshold = _configured_threshold()
    top_k_rates = _configured_top_k_rates()
    horizon_sweep = _configured_evaluation_horizon_sweep()
    include_features = bool(getattr(config, "VALIDATION_INCLUDE_FEATURE_COLUMNS", True))
    save_detailed_outputs = bool(getattr(config, "VALIDATION_SAVE_DETAILED_OUTPUTS", True))

    metric_rows = []
    topk_rows = []
    group_metric_rows = []
    run_summaries = []

    for algorithm in config.MODELS_TO_RUN:
        print(f"  Fit/validate report dataset={dataset_id} algorithm={algorithm}", flush=True)
        print(f"    Evaluation horizon sweep: {[ _horizon_label(h) for h in horizon_sweep ]}", flush=True)
        model = make_model_pipeline(algorithm, config)
        if model is None:
            metric_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "evaluation_view": "all",
                "status": "skipped_missing_dependency",
            })
            continue

        try:
            fit_metadata = fit_model_pipeline(
                model,
                algorithm,
                X_train,
                y_train,
                config,
                X_eval=X_fit_eval,
                y_eval=y_fit_eval,
                eval_name=fit_eval_name,
            )
            learning_outputs = _save_learning_curve_outputs(model, fit_metadata, dataset_id, algorithm, output_dir)
            importance_outputs = _save_feature_importance_outputs(model, algorithm, dataset_id, output_dir)

            model_artifact_path = ""
            if bool(getattr(config, "VALIDATION_SAVE_MODEL_ARTIFACTS", False)):
                try:
                    import joblib
                    model_dir = output_dir / "models"
                    ensure_dir(model_dir)
                    model_artifact = model_dir / f"{dataset_id}__{algorithm}__fitted_on_train.joblib"
                    joblib.dump(model, model_artifact)
                    model_artifact_path = str(model_artifact)
                except Exception as exc:
                    print(f"    warning: model artifact was not saved: {exc}", flush=True)

            for view_name, eval_path in eval_views:
                view_safe = _safe_name(view_name)
                eval_df = _read_dataset(eval_path)
                validate_dataset_features(eval_df, config)
                X_eval = eval_df[list(config.NUMERIC_FEATURES) + list(config.CATEGORICAL_FEATURES)]
                y_eval_training = eval_df["target"].astype(int)
                score = predict_score(model, X_eval, algorithm)

                # Save one prediction table per view. It includes the score,
                # future claim lead time, and all materialized horizon target
                # columns; metrics below are computed separately for each horizon.
                display_y, display_col, display_mode, display_horizon = get_evaluation_target(eval_df, config)
                pred_base = eval_df.copy()
                pred_base.insert(0, "evaluation_target", display_y.to_numpy())
                pred_base.insert(0, "evaluation_target_col", display_col)
                pred_base.insert(0, "evaluation_target_mode", display_mode)
                pred_base.insert(0, "evaluation_horizon_days", display_horizon)
                pred_base.insert(0, "evaluation_view", view_name)
                pred_base.insert(0, "algorithm", algorithm)
                pred_base.insert(0, "dataset_id", dataset_id)
                pred_base = _add_prediction_columns(pred_base, score, threshold=threshold, top_k_rates=top_k_rates)
                pred_base = _select_output_columns(pred_base, include_features=include_features)
                pred_base = pred_base.sort_values(
                    ["dataset_id", "algorithm", "evaluation_view", "score", "machine_key", "window_end"],
                    ascending=[True, True, True, False, True, True],
                    kind="mergesort",
                )

                compact = _select_output_columns(pred_base, include_features=False)
                machine_summary = _summarize_machine_predictions(pred_base)
                pred_file = output_dir / f"{dataset_id}__{algorithm}__{view_safe}__window_predictions.csv"
                compact_file = output_dir / f"{dataset_id}__{algorithm}__{view_safe}__window_predictions_compact.csv"
                machine_file = output_dir / f"{dataset_id}__{algorithm}__{view_safe}__machine_summary.csv"
                if save_detailed_outputs:
                    pred_base.to_csv(pred_file, index=False)
                    compact.to_csv(compact_file, index=False)
                    if not machine_summary.empty:
                        machine_summary.to_csv(machine_file, index=False)
                else:
                    pred_file = ""
                    compact_file = ""
                    machine_file = ""

                shap_outputs = _save_shap_outputs(
                    model=model,
                    algorithm=algorithm,
                    dataset_id=dataset_id,
                    view_name=view_name,
                    eval_df=pred_base,
                    X_eval=X_eval,
                    score=score,
                    output_dir=output_dir,
                )

                for eval_horizon in horizon_sweep:
                    horizon_label = _horizon_label(eval_horizon)
                    y_eval, eval_target_col, eval_target_mode, eval_horizon_days = get_evaluation_target(
                        eval_df,
                        config,
                        horizon_days=eval_horizon,
                    )
                    free = threshold_free_metrics(y_eval, score)
                    thresh = metrics_at_threshold(y_eval, score, threshold=threshold)
                    lead_summary = future_claim_lead_time_summary(eval_df, y_eval)

                    group_summary = pd.DataFrame()
                    group_metrics = {}
                    group_file = ""
                    if "matched" in view_name or "with_population" in view_name:
                        pred_for_group = pred_base.copy()
                        pred_for_group["evaluation_target"] = y_eval.to_numpy()
                        pred_for_group["evaluation_target_col"] = eval_target_col
                        pred_for_group["evaluation_target_mode"] = eval_target_mode
                        pred_for_group["evaluation_horizon_days"] = eval_horizon_days
                        group_summary = _summarize_case_control_group_predictions(pred_for_group)
                        group_metrics = _group_ranking_metrics(group_summary)
                        group_file = output_dir / f"{dataset_id}__{algorithm}__{view_safe}__{horizon_label}__case_control_group_summary.csv"
                        if save_detailed_outputs and not group_summary.empty:
                            group_summary.to_csv(group_file, index=False)
                        else:
                            group_file = ""

                    metric_row = {
                        "dataset_id": dataset_id,
                        "algorithm": algorithm,
                        "evaluation_view": view_name,
                        "status": "used",
                        "train_rows": int(len(train_df)),
                        "evaluation_rows": int(len(eval_df)),
                        "train_positive_rows": int(y_train.sum()),
                        "evaluation_training_target_positive_rows": int(y_eval_training.sum()),
                        "evaluation_positive_rows": int(y_eval.sum()),
                        "evaluation_positive_rate": float(y_eval.mean()) if len(y_eval) else np.nan,
                        "evaluation_target_col": eval_target_col,
                        "evaluation_target_mode": eval_target_mode,
                        "evaluation_horizon_days": eval_horizon_days,
                        "evaluation_path": eval_path,
                        "fit_eval_view_for_learning_curve": fit_eval_name,
                        "fit_eval_path_for_learning_curve": fit_eval_path,
                        "model_artifact_path": model_artifact_path,
                    }
                    metric_row.update(_fit_metadata_for_metrics(fit_metadata))
                    metric_row.update({f"threshold_free_{k}": v for k, v in free.items()})
                    metric_row.update({f"threshold_{str(threshold).replace('.', 'p')}_{k}": v for k, v in thresh.items()})
                    metric_row.update({f"lead_time_{k}": v for k, v in lead_summary.items()})
                    metric_row.update({f"group_rank_{k}": v for k, v in group_metrics.items()})
                    metric_row.update({f"learning_{k}": v for k, v in learning_outputs.items()})
                    metric_row.update({f"importance_{k}": v for k, v in importance_outputs.items()})
                    metric_row.update({f"shap_{k}": v for k, v in shap_outputs.items()})
                    metric_rows.append(metric_row)

                    topk = top_k_metrics(y_eval, score, top_k_rates)
                    topk.insert(0, "evaluation_target_col", eval_target_col)
                    topk.insert(0, "evaluation_target_mode", eval_target_mode)
                    topk.insert(0, "evaluation_horizon_days", eval_horizon_days)
                    topk.insert(0, "evaluation_view", view_name)
                    topk.insert(0, "algorithm", algorithm)
                    topk.insert(0, "dataset_id", dataset_id)
                    topk_rows.append(topk)

                    if group_metrics:
                        gm = {"dataset_id": dataset_id, "algorithm": algorithm, "evaluation_view": view_name, "evaluation_horizon_days": eval_horizon_days}
                        gm.update(group_metrics)
                        group_metric_rows.append(gm)

                    run_summaries.append({
                        "dataset_id": dataset_id,
                        "algorithm": algorithm,
                        "evaluation_view": view_name,
                        "train_rows": int(len(train_df)),
                        "evaluation_rows": int(len(eval_df)),
                        "evaluation_machines": int(eval_df["machine_key"].nunique(dropna=True)),
                        "evaluation_groups": int(eval_df["case_control_group_id"].nunique(dropna=True)) if "case_control_group_id" in eval_df.columns else 0,
                        "prediction_threshold": float(threshold),
                        "evaluation_target_col": eval_target_col,
                        "evaluation_target_mode": eval_target_mode,
                        "evaluation_horizon_days": eval_horizon_days,
                        "window_prediction_file": str(pred_file) if pred_file else "",
                        "compact_prediction_file": str(compact_file) if compact_file else "",
                        "machine_summary_file": str(machine_file) if machine_file else "",
                        "case_control_group_summary_file": str(group_file) if group_file else "",
                        **learning_outputs,
                        **importance_outputs,
                        **shap_outputs,
                    })

        except Exception as exc:
            print(f"    failed {algorithm}: {exc}", flush=True)
            metric_rows.append({
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "evaluation_view": "all",
                "status": "failed",
                "error": str(exc),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / f"{dataset_id}__validation_metrics_by_model.csv", index=False)
    if topk_rows:
        topk_df = pd.concat(topk_rows, ignore_index=True)
        topk_df.to_csv(output_dir / f"{dataset_id}__validation_top_k_by_model.csv", index=False)
    if group_metric_rows:
        pd.DataFrame(group_metric_rows).to_csv(output_dir / f"{dataset_id}__validation_group_ranking_metrics.csv", index=False)

    return {
        "dataset_id": dataset_id,
        "train_path": str(train_path),
        "train_rows": int(len(train_df)),
        "train_groups": int(train_df["case_control_group_id"].nunique()),
        "fit_eval_view_for_learning_curve": fit_eval_name,
        "fit_eval_path_for_learning_curve": fit_eval_path,
        "evaluation_horizon_sweep": [_horizon_label(h) for h in horizon_sweep],
        "evaluation_views": [{"view": v, "path": p} for v, p in eval_views],
        "models_attempted": list(config.MODELS_TO_RUN),
        "model_outputs": run_summaries,
    }

def run(
    dataset_index_path: str | Path | None = None,
    step_dir: str | Path | None = None,
) -> None:
    step_dir = Path(step_dir) if step_dir is not None else config.OUTPUT_DIR / "04_fit_validate_model_report"
    ensure_dir(step_dir)
    dataset_index = _load_dataset_index(dataset_index_path)

    summaries = []
    for _, dataset_row in dataset_index.iterrows():
        summaries.append(_train_score_one_dataset(dataset_row, step_dir))

    all_metrics = []
    all_topk = []
    all_group_metrics = []
    for dataset_id in dataset_index["dataset_id"]:
        p = step_dir / f"{dataset_id}__validation_metrics_by_model.csv"
        if p.exists():
            all_metrics.append(pd.read_csv(p, low_memory=False))
        t = step_dir / f"{dataset_id}__validation_top_k_by_model.csv"
        if t.exists():
            all_topk.append(pd.read_csv(t, low_memory=False))
        g = step_dir / f"{dataset_id}__validation_group_ranking_metrics.csv"
        if g.exists():
            all_group_metrics.append(pd.read_csv(g, low_memory=False))

    metrics_all = pd.DataFrame()
    topk_all = pd.DataFrame()
    if all_metrics:
        metrics_all = pd.concat(all_metrics, ignore_index=True)
        metrics_all.to_csv(step_dir / "validation_metrics_all_datasets.csv", index=False)
    if all_topk:
        topk_all = pd.concat(all_topk, ignore_index=True)
        topk_all.to_csv(step_dir / "validation_top_k_all_datasets.csv", index=False)
    if all_group_metrics:
        pd.concat(all_group_metrics, ignore_index=True).to_csv(step_dir / "validation_group_ranking_metrics_all_datasets.csv", index=False)

    trend = _build_horizon_trend_summary(metrics_all, topk_all)
    if not trend.empty:
        trend.to_csv(step_dir / "validation_horizon_trend_summary_for_review.csv", index=False)

    write_json(
        {
            "step": "04_fit_validate_model_report",
            "output_dir": str(step_dir),
            "dataset_index_path": str(dataset_index_path) if dataset_index_path is not None else str(config.OUTPUT_DIR / "02_case_control_datasets" / "dataset_index.csv"),
            "summaries": summaries,
            "models_to_run": config.MODELS_TO_RUN,
            "training_split": "train",
            "evaluation_split": "validation_only",
            "prediction_threshold": _configured_threshold(),
            "top_k_rates": _configured_top_k_rates(),
            "evaluation_target_mode": getattr(config, "EVALUATION_TARGET_MODE", "training_target"),
            "evaluation_claim_horizon_days": getattr(config, "EVALUATION_CLAIM_HORIZON_DAYS", None),
            "evaluation_horizon_sweep": [_horizon_label(h) for h in _configured_evaluation_horizon_sweep()],
            "evaluation_include_claim_on_window_end": bool(getattr(config, "EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END", True)),
            "xgboost_learning_curve_enabled": bool(getattr(config, "XGBOOST_ENABLE_LEARNING_CURVE", True)),
            "xgboost_early_stopping_enabled": bool(getattr(config, "XGBOOST_USE_EARLY_STOPPING", False)),
            "save_feature_importance": bool(getattr(config, "SAVE_FEATURE_IMPORTANCE", True)),
            "save_shap_values": bool(getattr(config, "SAVE_SHAP_VALUES", True)),
            "save_detailed_outputs": bool(getattr(config, "VALIDATION_SAVE_DETAILED_OUTPUTS", True)),
            "notes": [
                "Each configured model is fitted on the chronological training split and scored on validation views only.",
                "Training still uses the original case-control target; metrics can use the evaluation-only future-claim horizon target.",
                "When EVALUATION_CLAIM_HORIZON_DAYS is a list, validation metrics are computed once per horizon using the same trained model scores.",
                "validation_horizon_trend_summary_for_review.csv summarizes the trend across horizons for quick review.",
                "The test split is intentionally not loaded or scored in this step.",
                "For XGBoost, learning curves are saved from eval_set using the configured validation view.",
                "Feature importance and SHAP contribution files are saved for model interpretation.",
            ],
        },
        step_dir / "run_summary.json",
    )
    print(f"04_fit_validate_model_report completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
