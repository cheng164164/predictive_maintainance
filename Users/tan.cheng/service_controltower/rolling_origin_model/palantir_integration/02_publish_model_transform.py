"""Publish local XGBoost artifacts as a versioned Palantir model asset."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Mapping

from transforms.api import Input, transform
from palantir_models.transforms import ModelOutput
import xgboost as xgb

from .foundry_paths import (
    MODEL_ASSET,
    MODEL_FILE_NAME,
    MODEL_FILES_DATASET,
    MODEL_METADATA_FILE_NAME,
    TIER_POLICY_FILE_NAME,
)
from .machine_risk_model_adapter import MachineRiskXGBoostAdapter


def _read_json(model_files, filename: str) -> dict:
    """Read one JSON object from the unstructured model-files dataset."""
    with model_files.filesystem().open(filename, "r") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {filename}.")
    return payload


def _validated_settings(
    metadata: Mapping[str, object],
    tier_policy: Mapping[str, object],
) -> dict[str, object]:
    """Validate uploaded artifact compatibility and build immutable settings."""
    variant = str(metadata["variant"])
    if str(tier_policy.get("variant")) != variant:
        raise ValueError(
            "Uploaded model metadata and tier policy use different variants: "
            f"{variant!r} versus {tier_policy.get('variant')!r}."
        )
    if str(tier_policy.get("algorithm", "")).lower() != "xgboost":
        raise ValueError("The uploaded tier policy is not an XGBoost policy.")
    metadata_horizon = int(metadata["risk_horizon_days"])
    policy_horizon = int(tier_policy["risk_horizon_days"])
    if metadata_horizon != policy_horizon:
        raise ValueError(
            "Uploaded model metadata and tier policy use different risk horizons: "
            f"{metadata_horizon} versus {policy_horizon}."
        )
    features = [str(value) for value in metadata.get("features", ())]
    if not features:
        raise ValueError("Uploaded model metadata contains no model features.")

    tiers = tier_policy.get("tiers")
    if not isinstance(tiers, Mapping):
        raise TypeError("Uploaded tier policy must contain a tiers mapping.")
    for tier_name in ("CRITICAL", "HIGH", "MEDIUM"):
        tier = tiers.get(tier_name)
        if not isinstance(tier, Mapping):
            raise ValueError(f"Uploaded tier policy is missing {tier_name}.")
        calibrated_gate = tier.get("final_calibrated_probability_gate")
        risk_gate = tier.get("final_risk_index_gate")
        if calibrated_gate is not None and risk_gate is not None:
            expected = 100.0 * float(calibrated_gate)
            if abs(expected - float(risk_gate)) > 1e-9:
                raise ValueError(
                    f"{tier_name} risk-index gate is not 100 x the calibrated gate."
                )

    return {
        "variant": variant,
        "features": features,
        "best_iteration": int(metadata["best_iteration"]),
        "calibration_method": str(metadata["calibration_method"]),
        "platt_coefficient": float(metadata["platt_coefficient"]),
        "platt_intercept": float(metadata["platt_intercept"]),
        "operating_threshold": float(metadata["operating_threshold"]),
        "threshold_metric": str(metadata["threshold_metric"]),
        "risk_horizon_days": metadata_horizon,
        "risk_index_definition": str(metadata["risk_index_definition"]),
        "risk_index_epsilon": 1e-6,
        "fail_on_missing_features": True,
        "include_explanations": True,
        "explanation_top_n": 3,
        "xgboost_version_at_publish": str(xgb.__version__),
        "tier_policy": dict(tier_policy),
    }


@transform(
    model_files=Input(MODEL_FILES_DATASET),
    model_output=ModelOutput(MODEL_ASSET),
)
def compute(model_files, model_output) -> None:
    """Load validated artifacts, create the adapter, and publish a model version."""
    with model_files.filesystem().open(MODEL_FILE_NAME, "rb") as source:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temporary:
            temporary.write(source.read())
            temporary_path = temporary.name

    try:
        model = xgb.XGBClassifier()
        model.load_model(temporary_path)
    finally:
        os.unlink(temporary_path)

    metadata = _read_json(model_files, MODEL_METADATA_FILE_NAME)
    tier_policy = _read_json(model_files, TIER_POLICY_FILE_NAME)
    settings = _validated_settings(metadata, tier_policy)
    model_output.publish(
        model_adapter=MachineRiskXGBoostAdapter(model=model, settings=settings)
    )
