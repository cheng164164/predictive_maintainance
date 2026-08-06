"""Self-contained scoring, calibration, risk-index, and tier logic for Foundry."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

TIER_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
ACTIVE_TIERS = TIER_ORDER[:-1]


def apply_platt_calibration(
    raw_probabilities: Sequence[float], coefficient: float, intercept: float
) -> np.ndarray:
    """Apply the promoted Platt transform to raw XGBoost probabilities."""
    raw = np.asarray(raw_probabilities, dtype=float)
    clipped = np.clip(raw, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    calibrated_logit = np.clip(float(coefficient) * logit + float(intercept), -40, 40)
    return 1.0 / (1.0 + np.exp(-calibrated_logit))


def probability_to_risk_index(
    probabilities: Sequence[float], epsilon: float
) -> np.ndarray:
    """Convert calibrated probabilities to the customer-facing open 0-100 scale."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Calibrated probabilities must be finite.")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("risk_index_epsilon must be between 0 and 0.5.")
    return 100.0 * np.clip(values, float(epsilon), 1.0 - float(epsilon))


def _feature_matrix(
    frame: pd.DataFrame, features: Sequence[str], fail_on_missing: bool
) -> pd.DataFrame:
    """Return the exact numeric feature schema expected by the serialized model."""
    prepared = frame.copy()
    missing = [feature for feature in features if feature not in prepared.columns]
    if missing and fail_on_missing:
        raise ValueError(f"Prepared feature dataset is missing model features: {missing}")
    for feature in missing:
        prepared[feature] = 0.0
    matrix = prepared[list(features)].apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _passes_gate(
    calibrated_probability: float,
    risk_index: float,
    raw_score: float,
    tier: Mapping[str, object],
    rule: str,
) -> bool:
    """Evaluate one machine against one promoted tier confirmation rule."""
    cal_gate = tier.get("final_calibrated_probability_gate")
    risk_gate = tier.get("final_risk_index_gate")
    raw_gate = tier.get("final_raw_score_gate")
    cal_pass = cal_gate is None or calibrated_probability >= float(cal_gate)
    risk_pass = risk_gate is None or risk_index >= float(risk_gate)
    raw_pass = raw_gate is None or raw_score >= float(raw_gate)
    if rule == "both_calibrated_and_risk_index":
        return bool(cal_pass and risk_pass)
    if rule == "either_calibrated_or_risk_index":
        return bool(cal_pass or risk_pass)
    if rule == "calibrated_probability_only":
        return bool(cal_pass)
    if rule == "risk_index_only":
        return bool(risk_pass)
    if rule == "all_three_scores":
        return bool(cal_pass and risk_pass and raw_pass)
    raise ValueError(f"Unsupported tier confirmation rule: {rule!r}")


def apply_tier_policy(scores: pd.DataFrame, policy: Mapping[str, object]) -> pd.DataFrame:
    """Assign nested Top-N candidates and apply demotion-only score confirmation."""
    out = scores.sort_values(
        ["failure_probability", "raw_model_score", "machine_key"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out["risk_rank_within_score_date"] = np.arange(1, len(out) + 1, dtype=int)
    tiers = policy["tiers"]
    boundaries = {
        tier: int((tiers.get(tier) or {}).get("selected_top_n") or 0)
        for tier in ACTIVE_TIERS
    }
    rank = out["risk_rank_within_score_date"]
    conditions: list[pd.Series] = []
    labels: list[str] = []
    for tier in ACTIVE_TIERS:
        if boundaries[tier] > 0:
            conditions.append(rank.le(boundaries[tier]))
            labels.append(tier)
    out["candidate_risk_tier"] = (
        np.select(conditions, labels, default="LOW") if conditions else "LOW"
    )

    rule = str(policy.get("confirmation_rule", "risk_index_only"))
    final_tiers: list[str] = []
    confirmed: list[int] = []
    demotions: list[int] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        candidate = str(row.candidate_risk_tier)
        if candidate == "LOW":
            final_tiers.append("LOW")
            confirmed.append(1)
            demotions.append(0)
            reasons.append("outside_all_selected_top_n_boundaries")
            continue
        chosen = "LOW"
        start = ACTIVE_TIERS.index(candidate)
        for tier_name in ACTIVE_TIERS[start:]:
            tier = tiers.get(tier_name) or {}
            if not bool(tier.get("enabled", False)) or boundaries[tier_name] <= 0:
                continue
            if _passes_gate(
                calibrated_probability=float(row.failure_probability),
                risk_index=float(row.risk_index),
                raw_score=float(row.raw_model_score),
                tier=tier,
                rule=rule,
            ):
                chosen = tier_name
                break
        final_tiers.append(chosen)
        confirmed.append(int(chosen == candidate))
        demotions.append(max(0, TIER_ORDER.index(chosen) - TIER_ORDER.index(candidate)))
        if chosen == candidate:
            reasons.append(f"{candidate.lower()}_candidate_passed_score_confirmation")
        elif chosen == "LOW":
            reasons.append(f"{candidate.lower()}_candidate_failed_all_lower_score_gates")
        else:
            reasons.append(
                f"{candidate.lower()}_candidate_demoted_to_{chosen.lower()}_after_score_gate"
            )

    out["risk_tier"] = final_tiers
    out["candidate_tier_confirmed"] = np.asarray(confirmed, dtype="int8")
    out["tier_demotion_steps"] = np.asarray(demotions, dtype="int8")
    out["tier_decision_reason"] = reasons
    out["tier_confirmation_rule"] = rule
    return out


def _top_positive_contributions(
    model: XGBClassifier,
    features: Sequence[str],
    matrix: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """Return the strongest positive XGBoost contribution features per row."""
    contributions = model.get_booster().predict(
        DMatrix(matrix, feature_names=list(features)), pred_contribs=True
    )[:, :-1]
    result: list[str] = []
    for row in contributions:
        order = np.argsort(row)[::-1]
        factors: list[str] = []
        for index in order:
            if row[index] <= 0:
                continue
            factors.append(f"{features[index]} (+{row[index]:.2f})")
            if len(factors) >= int(top_n):
                break
        result.append("; ".join(factors))
    return result


def score_feature_dataframe(
    model: XGBClassifier,
    settings: Mapping[str, object],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Generate raw/calibrated scores, risk index, and final tiers in Foundry."""
    identity = ("machine_key", "snapshot_date", "full_model")
    missing_identity = sorted(set(identity).difference(frame.columns))
    if missing_identity:
        raise ValueError(f"Prepared feature dataset is missing: {missing_identity}")
    features = tuple(str(value) for value in settings["features"])
    matrix = _feature_matrix(
        frame,
        features,
        fail_on_missing=bool(settings.get("fail_on_missing_features", True)),
    )

    best_iteration = settings.get("best_iteration")
    if best_iteration is None:
        raw = model.predict_proba(matrix)[:, 1]
    else:
        try:
            raw = model.predict_proba(
                matrix, iteration_range=(0, int(best_iteration) + 1)
            )[:, 1]
        except TypeError:
            raw = model.predict_proba(matrix)[:, 1]
    calibration_method = str(settings.get("calibration_method", "platt")).lower()
    if calibration_method == "platt":
        calibrated = apply_platt_calibration(
            raw,
            coefficient=float(settings["platt_coefficient"]),
            intercept=float(settings["platt_intercept"]),
        )
    elif calibration_method == "none":
        calibrated = np.asarray(raw, dtype=float)
    else:
        raise ValueError(
            f"Unsupported serialized calibration method: {calibration_method!r}"
        )

    scores = frame[list(identity)].copy()
    scores["raw_model_score"] = raw
    scores["failure_probability"] = calibrated
    scores["risk_index"] = probability_to_risk_index(
        calibrated, epsilon=float(settings.get("risk_index_epsilon", 1e-6))
    )
    scores["risk_score_0_5"] = scores["risk_index"] / 20.0
    scores["operating_threshold"] = float(settings["operating_threshold"])
    scores["threshold_metric"] = str(settings["threshold_metric"])
    scores["model_flag_at_threshold"] = scores["failure_probability"].ge(
        float(settings["operating_threshold"])
    ).astype("int8")
    scores["risk_index_horizon_days"] = int(settings["risk_horizon_days"])
    scores["risk_index_definition"] = str(settings["risk_index_definition"])
    if bool(settings.get("include_explanations", True)):
        scores["top_risk_factors"] = _top_positive_contributions(
            model,
            features,
            matrix,
            top_n=int(settings.get("explanation_top_n", 3)),
        )
    else:
        scores["top_risk_factors"] = ""
    return apply_tier_policy(scores, settings["tier_policy"])
