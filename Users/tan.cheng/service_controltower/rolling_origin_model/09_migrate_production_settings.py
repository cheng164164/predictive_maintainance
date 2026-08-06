"""Export or promote final model artifacts into production settings.

Run this script after ``08_train_final_model.py``. It reads the tuned XGBoost
parameters, final model metadata, selected raw fault codes, calibration values,
operating threshold, and final tier-policy artifact.

The mode is controlled by ``PRODUCTION_SETTINGS_MODE`` in ``config.py`` or the
``--mode`` command-line override:

``export_only``
    Write a reviewable JSON settings payload and a complete candidate config
    file without modifying the active ``config.py``.

``update_config``
    Atomically update only the approved top-level assignments in ``config.py``.
    The candidate is syntax-checked, a timestamped backup is created when
    enabled, and failed verification restores the original file.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping

import config

CONFIG_PATH = Path(config.__file__).resolve()
PROMOTED_ASSIGNMENTS = (
    "XGB_PARAMS",
    "PRODUCTION_MODEL_SETTINGS",
    "PRODUCTION_SELECTED_FAULT_CODES",
    "PRODUCTION_TIER_POLICY",
)


def parse_arguments() -> argparse.Namespace:
    """Parse an optional production-settings mode override."""
    parser = argparse.ArgumentParser(
        description=(
            "Export candidate production settings or atomically update config.py "
            "from the final reviewed model artifacts."
        )
    )
    parser.add_argument(
        "--mode",
        choices=tuple(config.SUPPORTED_PRODUCTION_SETTINGS_MODES),
        default=str(config.PRODUCTION_SETTINGS_MODE),
        help=(
            "export_only writes candidate files without changing config.py; "
            "update_config performs the verified atomic promotion."
        ),
    )
    return parser.parse_args()


def _required(path: Path) -> Path:
    """Return an existing artifact path or raise a descriptive error."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required promotion artifact does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one promotion artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    """Load a JSON object and reject non-dictionary top-level payloads."""
    payload = json.loads(_required(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}; got {type(payload).__name__}.")
    return payload


def _validate_xgboost_params(params: Mapping[str, object], source: str) -> dict:
    """Normalize and validate one candidate XGBoost parameter dictionary."""
    promoted = {str(key): value for key, value in params.items()}
    required = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "eval_metric",
        "tree_method",
        "early_stopping_rounds",
    }
    missing = sorted(required.difference(promoted))
    if missing:
        raise ValueError(
            f"XGBoost parameters from {source} are missing required keys: {missing}"
        )
    return promoted


def _write_config_parameter_fallback(params: Mapping[str, object]) -> Path:
    """Persist the config-based XGBoost fallback as a hashable audit artifact."""
    path = Path(config.OUTPUT_DIR) / "xgboost_params_from_config_fallback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "config.XGB_PARAMS",
        "reason": "xgboost_tuned_params.json was not available",
        "params": dict(params),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _promoted_xgboost_params() -> tuple[dict, Path, str]:
    """Return promoted XGBoost parameters, audit path, and source description.

    The tuned-parameter artifact is preferred. When it is absent and strict
    enforcement is disabled, the script reuses the approved ``XGB_PARAMS``
    dictionary already loaded from ``config.py``. This supports development
    runs that intentionally skip hyperparameter tuning while keeping the
    promotion step deterministic and auditable.
    """
    tuned_path = Path(config.XGB_TUNED_PARAMS_FILE)
    if tuned_path.exists():
        payload = _load_json(tuned_path)
        params = payload.get("params", payload)
        if not isinstance(params, dict) or not params:
            raise ValueError(f"No tuned parameter dictionary found in {tuned_path}.")
        promoted = _validate_xgboost_params(params, str(tuned_path))
        return promoted, tuned_path, "tuned_parameter_artifact"

    require_artifact = bool(
        getattr(config, "PRODUCTION_SETTINGS_REQUIRE_TUNED_PARAMS_FILE", False)
    )
    if require_artifact:
        raise FileNotFoundError(
            "Required tuned-parameter artifact does not exist: "
            f"{tuned_path}. Either run 02_tune_xgboost.py or set "
            "PRODUCTION_SETTINGS_REQUIRE_TUNED_PARAMS_FILE = False to reuse "
            "the approved XGB_PARAMS settings from config.py."
        )

    promoted = _validate_xgboost_params(
        dict(config.XGB_PARAMS),
        "config.XGB_PARAMS fallback",
    )
    fallback_path = _write_config_parameter_fallback(promoted)
    print(
        "WARNING: tuned XGBoost parameter artifact was not found; "
        "reusing approved config.XGB_PARAMS settings.",
        flush=True,
    )
    print(f"  missing artifact: {tuned_path}", flush=True)
    print(f"  fallback audit artifact: {fallback_path}", flush=True)
    return promoted, fallback_path, "config_xgb_params_fallback"


def _model_and_policy_artifacts(variant: str) -> tuple[dict, dict, Path, Path]:
    """Load final model metadata and tier policy for the promoted variant."""
    metadata_path = _required(config.MODEL_DIR / f"model_metadata_{variant}.json")
    policy_path = _required(config.MODEL_DIR / f"tier_policy_{variant}.json")
    metadata = _load_json(metadata_path)
    policy = _load_json(policy_path)
    if str(metadata.get("variant")) != variant:
        raise ValueError(
            f"Metadata variant {metadata.get('variant')!r} does not match {variant!r}."
        )
    if str(policy.get("variant")) != variant:
        raise ValueError(
            f"Tier-policy variant {policy.get('variant')!r} does not match {variant!r}."
        )
    return metadata, policy, metadata_path, policy_path


def _build_model_settings(variant: str, metadata: Mapping[str, object]) -> dict:
    """Build the approved production model/calibration settings dictionary."""
    return {
        "variant": variant,
        "algorithm": "xgboost",
        "model_file": f"xgboost_{variant}.json",
        "metadata_file": f"model_metadata_{variant}.json",
        "best_iteration": int(metadata["best_iteration"]),
        "calibration_method": str(metadata["calibration_method"]),
        "platt_coefficient": float(metadata["platt_coefficient"]),
        "platt_intercept": float(metadata["platt_intercept"]),
        "operating_threshold": float(metadata["operating_threshold"]),
        "threshold_metric": str(metadata["threshold_metric"]),
        "risk_horizon_days": int(
            metadata.get("risk_horizon_days", config.HORIZON_DAYS)
        ),
    }


def _build_raw_fault_codes(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the supervised raw fault codes required by production scoring."""
    raw_codes = tuple(
        str(value) for value in metadata.get("selected_raw_fault_codes", ())
    )
    if not raw_codes:
        raise ValueError(
            "Final model metadata contains no selected_raw_fault_codes. "
            "Re-run 08_train_final_model.py before promotion."
        )
    return raw_codes


def _tier_payload_changed(candidate: Mapping[str, object]) -> bool:
    """Return whether policy content changed apart from the version number."""
    current = dict(config.PRODUCTION_TIER_POLICY)
    current.pop("policy_version", None)
    comparison = dict(candidate)
    comparison.pop("policy_version", None)
    return current != comparison


def _build_tier_policy(policy: Mapping[str, object]) -> dict:
    """Build the compact release-controlled tier policy stored in config.py."""
    current_version = int(config.PRODUCTION_TIER_POLICY.get("policy_version", 1))
    source_version = int(policy.get("policy_version", 1))
    candidate: dict[str, object] = {
        "policy_version": max(current_version, source_version),
        "algorithm": str(policy["algorithm"]),
        "variant": str(policy["variant"]),
        "candidate_selection": str(policy["candidate_selection"]),
        "confirmation_rule": str(policy["confirmation_rule"]),
        "confidence_level": float(policy["confidence_level"]),
        "risk_index_definition": str(policy["risk_index_definition"]),
        "risk_horizon_days": int(policy["risk_horizon_days"]),
        "tiers": {},
    }
    tiers = candidate["tiers"]
    assert isinstance(tiers, dict)
    for tier in ("CRITICAL", "HIGH", "MEDIUM"):
        source = policy["tiers"][tier]
        calibrated_gate = source.get("final_calibrated_probability_gate")
        risk_gate = source.get("final_risk_index_gate")
        if calibrated_gate is not None and risk_gate is not None:
            expected = 100.0 * float(calibrated_gate)
            tolerance = max(
                float(config.PRODUCTION_CONFIGURATION_ABSOLUTE_TOLERANCE), 1e-9
            )
            if abs(expected - float(risk_gate)) > tolerance:
                raise ValueError(
                    f"{tier} risk-index gate is not 100 x its calibrated gate: "
                    f"{risk_gate!r} versus {expected!r}."
                )
        tiers[tier] = {
            "selected_top_n": int(source.get("selected_top_n") or 0),
            "enabled": bool(source.get("enabled", False)),
            "required_precision": float(source["required_precision"]),
            "selection_status": str(source["selection_status"]),
            "confidence_guarantee_met": bool(source["confidence_guarantee_met"]),
            "final_raw_score_gate": source.get("final_raw_score_gate"),
            "final_calibrated_probability_gate": calibrated_gate,
            "final_risk_index_gate": risk_gate,
        }

    if _tier_payload_changed(candidate):
        candidate["policy_version"] = max(source_version, current_version + 1)
    return candidate


def build_promoted_assignments() -> tuple[dict[str, object], dict[str, Path]]:
    """Build all config values and identify their final-model source artifacts."""
    variant = str(config.PRODUCTION_VARIANT)
    metadata, policy, metadata_path, policy_path = _model_and_policy_artifacts(variant)
    if bool(config.PRODUCTION_SETTINGS_MIGRATION_UPDATE_XGB_PARAMS):
        xgb_params, parameter_path, parameter_source = _promoted_xgboost_params()
    else:
        xgb_params = _validate_xgboost_params(
            dict(config.XGB_PARAMS),
            "config.XGB_PARAMS",
        )
        parameter_path = _write_config_parameter_fallback(xgb_params)
        parameter_source = "config_xgb_params_update_disabled"
    assignments: dict[str, object] = {
        "XGB_PARAMS": xgb_params,
        "PRODUCTION_MODEL_SETTINGS": _build_model_settings(variant, metadata),
        "PRODUCTION_SELECTED_FAULT_CODES": _build_raw_fault_codes(metadata),
        "PRODUCTION_TIER_POLICY": _build_tier_policy(policy),
    }
    sources = {
        f"xgboost_parameters__{parameter_source}": parameter_path,
        "model_metadata": metadata_path,
        "tier_policy": policy_path,
        "model_file": _required(config.MODEL_DIR / f"xgboost_{variant}.json"),
    }
    return assignments, sources


def _assignment_ranges(source_text: str) -> dict[str, tuple[int, int]]:
    """Locate line ranges for the promoted top-level assignments in config.py."""
    tree = ast.parse(source_text, filename=str(CONFIG_PATH))
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id in PROMOTED_ASSIGNMENTS:
                if node.end_lineno is None:
                    raise RuntimeError(
                        f"Python did not report end_lineno for {target.id}."
                    )
                ranges[target.id] = (node.lineno - 1, node.end_lineno)
    missing = sorted(set(PROMOTED_ASSIGNMENTS).difference(ranges))
    if missing:
        raise ValueError(f"Could not find config assignments for: {missing}")
    return ranges


def _literal(value: object) -> str:
    """Return a deterministic Python literal for a promoted scalar or sequence."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        body = ", ".join(_literal(item) for item in value)
        if len(value) == 1:
            body += ","
        return f"({body})"
    raise TypeError(f"Unsupported promoted literal type: {type(value).__name__}")


def _render_simple_dict(name: str, value: Mapping[str, object]) -> str:
    """Render a flat dictionary with one release-controlled setting per line."""
    lines = [f"{name} = {{\n"]
    for key, item in value.items():
        if name == "XGB_PARAMS" and key == "eval_metric":
            lines.extend(
                [
                    "    # Keep aucpr last: XGBoost early stopping monitors the last metric\n",
                    "    # on the last evaluation set, which is the reserved calibration period.\n",
                ]
            )
        lines.append(f"    {_literal(str(key))}: {_literal(item)},\n")
    lines.append("}\n")
    return "".join(lines)


def _render_fault_codes(value: tuple[str, ...]) -> str:
    """Render selected raw fault codes as a readable one-value-per-line tuple."""
    lines = ["PRODUCTION_SELECTED_FAULT_CODES = (\n"]
    lines.extend(f"    {_literal(code)},\n" for code in value)
    lines.append(")\n")
    return "".join(lines)


def _render_tier_policy(value: Mapping[str, object]) -> str:
    """Render the nested tier policy in the same clear structure as config.py."""
    lines = ["PRODUCTION_TIER_POLICY = {\n"]
    for key in (
        "policy_version",
        "algorithm",
        "variant",
        "candidate_selection",
        "confirmation_rule",
        "confidence_level",
        "risk_index_definition",
        "risk_horizon_days",
    ):
        lines.append(f"    {_literal(key)}: {_literal(value[key])},\n")
    lines.append('    "tiers": {\n')
    tiers = value["tiers"]
    if not isinstance(tiers, Mapping):
        raise TypeError("PRODUCTION_TIER_POLICY['tiers'] must be a mapping.")
    for tier_name in ("CRITICAL", "HIGH", "MEDIUM"):
        tier = tiers[tier_name]
        if not isinstance(tier, Mapping):
            raise TypeError(f"Tier {tier_name} must be a mapping.")
        lines.append(f"        {_literal(tier_name)}: {{\n")
        for key in (
            "selected_top_n",
            "enabled",
            "required_precision",
            "selection_status",
            "confidence_guarantee_met",
            "final_raw_score_gate",
            "final_calibrated_probability_gate",
            "final_risk_index_gate",
        ):
            lines.append(f"            {_literal(key)}: {_literal(tier[key])},\n")
        lines.append("        },\n")
    lines.extend(["    },\n", "}\n"])
    return "".join(lines)


def _render_assignment(name: str, value: object) -> list[str]:
    """Render one deterministic assignment while preserving config readability."""
    if name in {"XGB_PARAMS", "PRODUCTION_MODEL_SETTINGS"}:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping.")
        rendered = _render_simple_dict(name, value)
    elif name == "PRODUCTION_SELECTED_FAULT_CODES":
        rendered = _render_fault_codes(tuple(str(item) for item in value))
    elif name == "PRODUCTION_TIER_POLICY":
        if not isinstance(value, Mapping):
            raise TypeError("PRODUCTION_TIER_POLICY must be a mapping.")
        rendered = _render_tier_policy(value)
    else:  # Defensive guard for future edits to PROMOTED_ASSIGNMENTS.
        raise ValueError(f"No renderer is defined for {name}.")
    return rendered.splitlines(keepends=True)


def render_updated_config(assignments: Mapping[str, object]) -> str:
    """Return config.py text with only approved release blocks replaced."""
    source_text = CONFIG_PATH.read_text(encoding="utf-8")
    lines = source_text.splitlines(keepends=True)
    ranges = _assignment_ranges(source_text)
    replacements = sorted(
        (
            (ranges[name][0], ranges[name][1], _render_assignment(name, assignments[name]))
            for name in PROMOTED_ASSIGNMENTS
        ),
        reverse=True,
    )
    for start, end, replacement in replacements:
        lines[start:end] = replacement
    updated = "".join(lines)
    compile(updated, str(CONFIG_PATH), "exec")
    return updated


def _create_backup() -> Path | None:
    """Create a timestamped config backup when enabled and return its path."""
    if not bool(config.PRODUCTION_SETTINGS_MIGRATION_CREATE_BACKUP):
        return None
    backup_dir = Path(config.PRODUCTION_SETTINGS_MIGRATION_BACKUP_DIR)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"config.py.{timestamp}.bak"
    shutil.copy2(CONFIG_PATH, backup_path)
    return backup_path


def _atomic_write(updated_text: str) -> None:
    """Syntax-check and atomically replace config.py with updated text."""
    fd, temporary_name = tempfile.mkstemp(
        prefix="config.py.", suffix=".tmp", dir=str(CONFIG_PATH.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(updated_text)
            stream.flush()
            os.fsync(stream.fileno())
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(temporary_path)],
            check=True,
            cwd=CONFIG_PATH.parent,
        )
        os.replace(temporary_path, CONFIG_PATH)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _verify_promoted_config(assignments: Mapping[str, object]) -> None:
    """Import the migrated config in a fresh process and verify key equality."""
    verification_code = (
        "import json, config; "
        "print(json.dumps({"
        "'XGB_PARAMS': config.XGB_PARAMS, "
        "'PRODUCTION_MODEL_SETTINGS': config.PRODUCTION_MODEL_SETTINGS, "
        "'PRODUCTION_SELECTED_FAULT_CODES': list(config.PRODUCTION_SELECTED_FAULT_CODES), "
        "'PRODUCTION_TIER_POLICY': config.PRODUCTION_TIER_POLICY"
        "}, sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", verification_code],
        check=True,
        cwd=CONFIG_PATH.parent,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout.strip())
    expected = {
        "XGB_PARAMS": assignments["XGB_PARAMS"],
        "PRODUCTION_MODEL_SETTINGS": assignments["PRODUCTION_MODEL_SETTINGS"],
        "PRODUCTION_SELECTED_FAULT_CODES": list(
            assignments["PRODUCTION_SELECTED_FAULT_CODES"]
        ),
        "PRODUCTION_TIER_POLICY": assignments["PRODUCTION_TIER_POLICY"],
    }
    if observed != expected:
        raise RuntimeError(
            "The migrated config imports successfully but does not match the "
            "promoted artifact values."
        )


def _export_candidate_files(
    assignments: Mapping[str, object],
    sources: Mapping[str, Path],
) -> tuple[Path, Path]:
    """Write reviewable JSON and full candidate-config files without promotion."""
    json_path = Path(config.PRODUCTION_SETTINGS_EXPORT_JSON_FILE)
    candidate_config_path = Path(config.PRODUCTION_SETTINGS_EXPORT_CONFIG_FILE)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_config_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "export_only",
        "active_config_path": str(CONFIG_PATH),
        "candidate_config_path": str(candidate_config_path),
        "promoted_assignments": assignments,
        "sources": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in sources.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    candidate_text = render_updated_config(assignments)
    compile(candidate_text, str(candidate_config_path), "exec")
    candidate_config_path.write_text(candidate_text, encoding="utf-8")
    return json_path, candidate_config_path


def _write_audit(
    assignments: Mapping[str, object],
    sources: Mapping[str, Path],
    mode: str,
    backup_path: Path | None,
    export_json_path: Path | None,
    candidate_config_path: Path | None,
) -> Path:
    """Write an auditable JSON record of the export or config promotion."""
    audit_path = Path(config.PRODUCTION_SETTINGS_MIGRATION_AUDIT_FILE)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "active_config_path": str(CONFIG_PATH),
        "active_config_updated": mode == "update_config",
        "backup_path": str(backup_path) if backup_path else None,
        "export_json_path": str(export_json_path) if export_json_path else None,
        "candidate_config_path": (
            str(candidate_config_path) if candidate_config_path else None
        ),
        "promoted_assignments": list(PROMOTED_ASSIGNMENTS),
        "variant": assignments["PRODUCTION_MODEL_SETTINGS"]["variant"],
        "best_iteration": assignments["PRODUCTION_MODEL_SETTINGS"]["best_iteration"],
        "tier_policy_version": assignments["PRODUCTION_TIER_POLICY"]["policy_version"],
        "sources": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in sources.items()
        },
    }
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return audit_path


def main() -> None:
    """Export candidate settings or promote them according to the selected mode."""
    args = parse_arguments()
    mode = str(args.mode)
    assignments, sources = build_promoted_assignments()

    export_json_path, candidate_config_path = _export_candidate_files(
        assignments,
        sources,
    )

    if mode == "export_only":
        audit_path = _write_audit(
            assignments=assignments,
            sources=sources,
            mode=mode,
            backup_path=None,
            export_json_path=export_json_path,
            candidate_config_path=candidate_config_path,
        )
        print("Production settings export complete.")
        print("  active config updated: no")
        print(f"  candidate settings: {export_json_path}")
        print(f"  candidate config: {candidate_config_path}")
        print(f"  audit: {audit_path}")
        return

    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    updated_text = render_updated_config(assignments)
    backup_path = _create_backup()
    try:
        _atomic_write(updated_text)
        _verify_promoted_config(assignments)
    except Exception:
        # A failed fresh-process import or equality check must never leave a
        # partially promoted production configuration in place.
        _atomic_write(original_text)
        raise
    audit_path = _write_audit(
        assignments=assignments,
        sources=sources,
        mode=mode,
        backup_path=backup_path,
        export_json_path=export_json_path,
        candidate_config_path=candidate_config_path,
    )

    print("Production settings migration complete.")
    print("  active config updated: yes")
    print(f"  config: {CONFIG_PATH}")
    print(f"  backup: {backup_path or 'disabled'}")
    print(f"  candidate settings: {export_json_path}")
    print(f"  candidate config: {candidate_config_path}")
    print(f"  audit: {audit_path}")


if __name__ == "__main__":
    main()
