"""Export reviewable production settings after retraining.

The script never edits ``config.py`` automatically. It reads the newly trained
model metadata and tier-policy artifact, writes a Python snippet containing the
candidate release values, and leaves explicit approval/promotion to the model
owner. This prevents an unreviewed retraining run from changing live risk tiers.
"""
from __future__ import annotations

import json
from pathlib import Path
from pprint import pformat

import config


def _required(path: Path) -> Path:
    """Return an existing artifact path or raise a descriptive error."""
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_candidate_settings(variant: str) -> tuple[dict, tuple[str, ...], dict]:
    """Extract model, raw fault-code, and tier settings from trained artifacts."""
    metadata_path = _required(config.MODEL_DIR / f"model_metadata_{variant}.json")
    policy_path = _required(config.MODEL_DIR / f"tier_policy_{variant}.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    model_settings = {
        "variant": variant,
        "algorithm": "xgboost",
        "model_file": f"xgboost_{variant}.json",
        "metadata_file": metadata_path.name,
        "best_iteration": int(metadata["best_iteration"]),
        "calibration_method": str(metadata["calibration_method"]),
        "platt_coefficient": float(metadata["platt_coefficient"]),
        "platt_intercept": float(metadata["platt_intercept"]),
        "operating_threshold": float(metadata["operating_threshold"]),
        "threshold_metric": str(metadata["threshold_metric"]),
        "risk_horizon_days": int(metadata.get("risk_horizon_days", config.HORIZON_DAYS)),
    }
    raw_codes = tuple(
        str(code) for code in metadata.get("selected_raw_fault_codes", ())
    )
    if not raw_codes:
        raise ValueError(
            "Model metadata does not contain selected_raw_fault_codes. "
            "Re-run 08_train_final_model.py with the updated scripts."
        )

    tier_policy = {
        "policy_version": int(policy.get("policy_version", 4)),
        "algorithm": str(policy["algorithm"]),
        "variant": str(policy["variant"]),
        "candidate_selection": str(policy["candidate_selection"]),
        "confirmation_rule": str(policy["confirmation_rule"]),
        "confidence_level": float(policy["confidence_level"]),
        "risk_index_definition": str(policy["risk_index_definition"]),
        "risk_horizon_days": int(policy["risk_horizon_days"]),
        "tiers": {},
    }
    for tier in ("CRITICAL", "HIGH", "MEDIUM"):
        source = policy["tiers"][tier]
        tier_policy["tiers"][tier] = {
            "selected_top_n": int(source.get("selected_top_n") or 0),
            "enabled": bool(source.get("enabled", False)),
            "required_precision": float(source["required_precision"]),
            "selection_status": str(source["selection_status"]),
            "confidence_guarantee_met": bool(source["confidence_guarantee_met"]),
            "final_raw_score_gate": source.get("final_raw_score_gate"),
            "final_calibrated_probability_gate": source.get(
                "final_calibrated_probability_gate"
            ),
            "final_risk_index_gate": source.get("final_risk_index_gate"),
        }
    return model_settings, raw_codes, tier_policy


def main() -> None:
    """Write a candidate config snippet for explicit production promotion."""
    variant = str(config.PRODUCTION_VARIANT)
    model_settings, raw_codes, tier_policy = build_candidate_settings(variant)
    output_path = config.OUTPUT_DIR / "production_config_candidate.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Generated candidate settings. Review validation evidence before copying\n"
        "# these values into the approved production section of config.py.\n\n"
        f"PRODUCTION_MODEL_SETTINGS = {pformat(model_settings, sort_dicts=False)}\n\n"
        f"PRODUCTION_SELECTED_FAULT_CODES = {pformat(raw_codes)}\n\n"
        f"PRODUCTION_TIER_POLICY = {pformat(tier_policy, sort_dicts=False)}\n"
    )
    output_path.write_text(content, encoding="utf-8")
    print(f"Production config candidate written: {output_path}")


if __name__ == "__main__":
    main()
