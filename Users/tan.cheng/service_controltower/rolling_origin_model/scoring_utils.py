"""Reusable production scoring utilities for machine-risk predictions.

This module is the single scoring implementation used by both the standard
latest-snapshot workflow and the incoming three-month refresh workflow. It
loads the approved model, validates release-controlled calibration and tier
settings, produces raw and calibrated probabilities, calculates ``risk_index``,
and applies the demotion-only tier policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBClassifier

import config
from modeling_utils import PlattCalibration, friendly_feature_name
from risk_tier_utils import apply_final_tier_policy


TIER_AUDIT_COLUMNS = (
    "fault_count_90d",
    "fault_serious_count_90d",
    "fault_serious_30d",
    "fault_l03_count_90d",
    "fault_l04_count_90d",
    "fault_single_system_concentration",
    "fault_severity_recency_90d",
    "has_accelerating_faults",
    "fluid_current_severity",
    "fluid_contaminant_flag",
    "pm_overdue_count_90d",
    "operation_history_days",
    "days_since_last_sensor_reading",
    "smr_delta_30d",
    "prior_target_event_count_365d",
)


@dataclass(frozen=True)
class ProductionArtifacts:
    """Loaded model, metadata, calibration object, and approved tier policy."""

    variant: str
    model: XGBClassifier
    metadata: dict
    features: tuple[str, ...]
    calibrator: PlattCalibration
    operating_threshold: float
    threshold_metric: str
    tier_policy: dict
    model_path: Path
    metadata_path: Path


def _assert_close(name: str, observed: object, expected: object) -> None:
    """Raise when a numeric model artifact value differs from approved config."""
    tolerance = float(config.PRODUCTION_CONFIGURATION_ABSOLUTE_TOLERANCE)
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"Production configuration mismatch for {name}: "
            f"artifact={observed!r}, config={expected!r}. "
            "Validate and explicitly promote the new release values in config.py."
        )


def _validate_model_metadata(metadata: Mapping[str, object], variant: str) -> None:
    """Validate saved model metadata against release-controlled config values."""
    approved = config.PRODUCTION_MODEL_SETTINGS
    if str(metadata.get("variant")) != variant:
        raise ValueError(
            f"Model metadata variant {metadata.get('variant')!r} does not match {variant!r}."
        )
    if str(approved["variant"]) != variant:
        raise ValueError(
            f"Approved production variant {approved['variant']!r} does not match {variant!r}."
        )
    if not bool(config.VALIDATE_PRODUCTION_CONFIG_AGAINST_MODEL_ARTIFACTS):
        return
    text_fields = ("calibration_method", "threshold_metric")
    for field in text_fields:
        if str(metadata.get(field)) != str(approved[field]):
            raise ValueError(
                f"Production configuration mismatch for {field}: "
                f"artifact={metadata.get(field)!r}, config={approved[field]!r}."
            )
    numeric_fields = (
        "best_iteration",
        "platt_coefficient",
        "platt_intercept",
        "operating_threshold",
    )
    for field in numeric_fields:
        _assert_close(field, metadata.get(field), approved[field])


def _validate_tier_policy_file(variant: str, approved_policy: Mapping[str, object]) -> None:
    """Cross-check the saved generated tier policy against approved config gates."""
    if not bool(config.VALIDATE_PRODUCTION_CONFIG_AGAINST_MODEL_ARTIFACTS):
        return
    policy_path = config.MODEL_DIR / f"tier_policy_{variant}.json"
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Saved tier policy not found for release validation: {policy_path}"
        )
    generated = json.loads(policy_path.read_text(encoding="utf-8"))
    for tier in ("CRITICAL", "HIGH", "MEDIUM"):
        approved_tier = approved_policy["tiers"][tier]
        generated_tier = generated["tiers"][tier]
        if int(generated_tier.get("selected_top_n") or 0) != int(
            approved_tier.get("selected_top_n") or 0
        ):
            raise ValueError(
                f"Approved {tier} Top-N differs from generated tier policy."
            )
        for field in (
            "final_raw_score_gate",
            "final_calibrated_probability_gate",
            "final_risk_index_gate",
        ):
            _assert_close(
                f"{tier}.{field}", generated_tier.get(field), approved_tier.get(field)
            )


def approved_tier_policy(variant: str) -> dict:
    """Return a validated deep copy of the release-controlled tier policy."""
    policy = deepcopy(config.PRODUCTION_TIER_POLICY)
    if str(policy.get("variant")) != variant:
        raise ValueError(
            f"Approved tier-policy variant {policy.get('variant')!r} does not match {variant!r}."
        )
    _validate_tier_policy_file(variant, policy)
    return policy


def load_production_artifacts(variant: str | None = None) -> ProductionArtifacts:
    """Load and validate all artifacts required for deterministic production scoring.

    Parameters
    ----------
    variant:
        Optional model variant override. Production callers should normally omit
        it so the release-controlled variant in ``config.py`` is used.

    Returns
    -------
    ProductionArtifacts
        Validated model, feature schema, Platt calibrator, threshold, and tier
        policy.
    """
    approved = config.PRODUCTION_MODEL_SETTINGS
    selected_variant = str(variant or approved["variant"])
    if selected_variant != str(config.PRODUCTION_VARIANT):
        raise ValueError(
            f"Production variant mismatch: requested={selected_variant!r}, "
            f"config.PRODUCTION_VARIANT={config.PRODUCTION_VARIANT!r}."
        )

    model_path = config.MODEL_DIR / str(approved["model_file"])
    metadata_path = config.MODEL_DIR / str(approved["metadata_file"])
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Missing production model artifacts. Expected:\n"
            f"  model: {model_path}\n  metadata: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_model_metadata(metadata, selected_variant)
    features = tuple(str(value) for value in metadata.get("features", ()))
    if not features:
        raise ValueError(f"No feature schema found in {metadata_path}.")

    model = XGBClassifier()
    model.load_model(model_path)
    calibrator = PlattCalibration(
        coefficient=float(approved["platt_coefficient"]),
        intercept=float(approved["platt_intercept"]),
    )
    policy = approved_tier_policy(selected_variant)
    return ProductionArtifacts(
        variant=selected_variant,
        model=model,
        metadata=metadata,
        features=features,
        calibrator=calibrator,
        operating_threshold=float(approved["operating_threshold"]),
        threshold_metric=str(approved["threshold_metric"]),
        tier_policy=policy,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def top_positive_contributions(
    model: XGBClassifier,
    features: Sequence[str],
    matrix: pd.DataFrame,
    top_n: int = 3,
) -> list[str]:
    """Return the strongest positive TreeSHAP-style contributions per machine."""
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
            factors.append(
                f"{friendly_feature_name(features[index])} (+{row[index]:.2f})"
            )
            if len(factors) >= top_n:
                break
        result.append("; ".join(factors))
    return result


def _ensure_feature_schema(
    snapshot: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Return a copy containing the exact model feature schema in stable order."""
    prepared = snapshot.copy()
    missing = [feature for feature in features if feature not in prepared.columns]
    if missing and bool(config.PRODUCTION_FAIL_ON_MISSING_MODEL_FEATURES):
        raise ValueError(
            "Scoring snapshot is missing approved model features: "
            f"{missing}. Rebuild the snapshot with the production source schema."
        )
    for feature in features:
        if feature not in prepared.columns:
            prepared[feature] = 0.0
    feature_matrix = prepared[list(features)].apply(pd.to_numeric, errors="coerce")
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prepared.loc[:, list(features)] = feature_matrix
    return prepared


def score_snapshot_dataframe(
    snapshot: pd.DataFrame,
    artifacts: ProductionArtifacts | None = None,
    include_explanations: bool = True,
) -> pd.DataFrame:
    """Score one fleet snapshot and assign calibrated probability-based tiers.

    The returned frame contains the raw XGBoost score, calibrated 90-day event
    probability, customer-facing risk index, provisional Top-N tier, final tier,
    score-gate audit columns, and optional top positive model contributions.
    """
    artifacts = artifacts or load_production_artifacts()
    required_identity = {"machine_key", "snapshot_date", "full_model"}
    missing_identity = sorted(required_identity.difference(snapshot.columns))
    if missing_identity:
        raise ValueError(
            f"Scoring snapshot is missing identity columns: {missing_identity}"
        )
    if snapshot.empty:
        raise ValueError("Scoring snapshot is empty.")
    if pd.to_datetime(snapshot["snapshot_date"]).nunique() != 1:
        raise ValueError("Production scoring expects exactly one snapshot date.")
    if snapshot["machine_key"].duplicated().any():
        duplicates = snapshot.loc[
            snapshot["machine_key"].duplicated(), "machine_key"
        ].head(5).tolist()
        raise ValueError(f"Duplicate machine rows in scoring snapshot: {duplicates}")

    prepared = _ensure_feature_schema(snapshot, artifacts.features)
    feature_matrix = prepared[list(artifacts.features)]
    raw_score = artifacts.model.predict_proba(feature_matrix)[:, 1]
    probability = artifacts.calibrator.apply(raw_score)

    scores = prepared[["machine_key", "snapshot_date", "full_model"]].copy()
    scores["raw_model_score"] = raw_score
    scores["failure_probability"] = probability
    scores["operating_threshold"] = artifacts.operating_threshold
    scores["threshold_metric"] = artifacts.threshold_metric
    scores["model_flag_at_threshold"] = (
        scores["failure_probability"].ge(artifacts.operating_threshold).astype("int8")
    )
    scores[f"model_flag_{artifacts.threshold_metric}"] = scores[
        "model_flag_at_threshold"
    ]
    for column in TIER_AUDIT_COLUMNS:
        scores[column] = (
            prepared[column].to_numpy() if column in prepared.columns else 0.0
        )
    if include_explanations:
        scores["top_risk_factors"] = top_positive_contributions(
            artifacts.model,
            artifacts.features,
            feature_matrix,
        )
    else:
        scores["top_risk_factors"] = ""

    scored = apply_final_tier_policy(scores, artifacts.tier_policy)
    tier_order = pd.Categorical(
        scored["risk_tier"],
        categories=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        ordered=True,
    )
    return (
        scored.assign(_tier_order=tier_order)
        .sort_values(["_tier_order", "risk_index"], ascending=[True, False])
        .drop(columns="_tier_order")
        .reset_index(drop=True)
    )


def scoring_summary(
    scores: pd.DataFrame,
    variant: str,
    output_file: Path | str,
) -> dict[str, object]:
    """Build a compact audit summary for one completed fleet scoring run."""
    counts = scores["risk_tier"].value_counts().to_dict()
    candidate_counts = scores["candidate_risk_tier"].value_counts().to_dict()
    return {
        "variant": variant,
        "score_date": str(pd.Timestamp(scores["snapshot_date"].iloc[0]).date()),
        "score_rows": int(len(scores)),
        "mean_failure_probability": float(scores["failure_probability"].mean()),
        "mean_risk_index": float(scores["risk_index"].mean()),
        "max_risk_index": float(scores["risk_index"].max()),
        "threshold_metric": str(scores["threshold_metric"].iloc[0]),
        "operating_threshold": float(scores["operating_threshold"].iloc[0]),
        "flagged_rows_at_threshold": int(scores["model_flag_at_threshold"].sum()),
        "critical_candidates": int(candidate_counts.get("CRITICAL", 0)),
        "high_candidates": int(candidate_counts.get("HIGH", 0)),
        "medium_candidates": int(candidate_counts.get("MEDIUM", 0)),
        "low_candidates": int(candidate_counts.get("LOW", 0)),
        "critical_confirmed": int(counts.get("CRITICAL", 0)),
        "high_confirmed": int(counts.get("HIGH", 0)),
        "medium_confirmed": int(counts.get("MEDIUM", 0)),
        "low_confirmed": int(counts.get("LOW", 0)),
        "demoted_candidates": int(scores["tier_demotion_steps"].gt(0).sum()),
        "tier_confirmation_rule": str(scores["tier_confirmation_rule"].iloc[0]),
        "output_file": str(output_file),
    }
