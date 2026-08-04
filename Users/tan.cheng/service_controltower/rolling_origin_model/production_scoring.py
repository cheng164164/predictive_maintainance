"""Prepare incoming source refreshes and execute production machine-risk scoring.

The expected operational input is the newest three months of source history for
all active machines. Fault and operation features use that incoming window
directly. Fluid, maintenance, and prior-target features are merged with retained
historical sources because the approved model contains longer-memory recency and
history features.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

import config
from scoring_utils import (
    load_production_artifacts,
    score_snapshot_dataframe,
    scoring_summary,
)
from snapshot_builder import SourceFileSet, build_snapshot_dataframe, load_sources


@dataclass(frozen=True)
class SourceSpec:
    """CSV schema, date field, retention rule, and deduplication keys."""

    key: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    date_column: str
    deduplication_keys: tuple[str, ...]
    retention_days: int | None


@dataclass(frozen=True)
class PreparedScoringBundle:
    """Prepared source files and manifest for one production scoring date."""

    score_date: pd.Timestamp
    source_files: SourceFileSet
    manifest_path: Path
    work_dir: Path


FAULT_COLUMNS = (
    "serial_number",
    "full_model",
    "event_date",
    "fault_code",
    "event_action_level",
    "logical_name",
    "occurrence_count",
    "log_occurrence_count",
    "smr_hours",
    "applicable_component",
    "failure_code_evidence_score",
    "failure_code_evidence_group",
    "action_level_num",
)
FLUID_COLUMNS = (
    "FULL_MODEL",
    "SERIAL",
    "sample_drawn_date",
    "sample_result_severity_order",
    "Water_Water_PERCENT",
    "Gly_Glycol_PERCENT",
    "EthyleneGlycol_Ethylene_Glycol_PERCENT",
    "Fuel_Fuel_PERCENT",
    "Si_Silicon_PPM",
    "Fe_Iron_PPM",
    "Cu_Copper_PPM",
    "TELEMETRY_SMR_NUMERIC",
)
MAINTENANCE_COLUMNS = (
    "full_model",
    "SERIAL",
    "EVENT_NAME_ID",
    "event_date",
    "is_monitor_reset",
    "is_overdue",
    "is_due_now",
    "remaining_hours",
    "INTERVAL_HOURS",
)
OPERATION_COLUMNS = (
    "full_model",
    "SERIAL",
    "LOCAL_DATE",
    "smr_hours",
    "smr_delta_clean_since_prev_obs_hours",
    "engine_running_hours_clean",
    "actual_working_hours_clean",
    "engine_idle_share_daily",
    "throttle_full_share_clean",
    "traveling_hours_clean",
    "engine_running_day_flag",
    "actual_work_day_flag",
)
PHYSICAL_FAILURE_REQUIRED_COLUMNS = ("machine", "event_date")
PHYSICAL_FAILURE_OPTIONAL_COLUMNS = ("event_source", "title")
WARRANTY_REQUIRED_COLUMNS = (
    "machine_id",
    "local_date",
)
WARRANTY_OPTIONAL_COLUMNS = (
    "claim_number",
    "claim_type_description",
)


def source_specs() -> dict[str, SourceSpec]:
    """Return source preparation rules for the active target configuration."""
    target_key = str(config.TARGET_SOURCE)
    target_spec = (
        SourceSpec(
            key="physical_failure",
            required_columns=PHYSICAL_FAILURE_REQUIRED_COLUMNS,
            optional_columns=PHYSICAL_FAILURE_OPTIONAL_COLUMNS,
            date_column="event_date",
            deduplication_keys=("machine", "event_date", "event_source", "title"),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["physical_failure"],
        )
        if target_key == "physical_failure"
        else SourceSpec(
            key="warranty",
            required_columns=WARRANTY_REQUIRED_COLUMNS,
            optional_columns=WARRANTY_OPTIONAL_COLUMNS,
            date_column="local_date",
            deduplication_keys=("machine_id", "local_date", "claim_number"),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["warranty"],
        )
    )
    return {
        "fault": SourceSpec(
            key="fault",
            required_columns=FAULT_COLUMNS,
            optional_columns=(),
            date_column="event_date",
            deduplication_keys=(
                "full_model",
                "serial_number",
                "event_date",
                "fault_code",
                "event_action_level",
                "occurrence_count",
            ),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["fault"],
        ),
        "fluid": SourceSpec(
            key="fluid",
            required_columns=FLUID_COLUMNS,
            optional_columns=(),
            date_column="sample_drawn_date",
            deduplication_keys=(
                "FULL_MODEL",
                "SERIAL",
                "sample_drawn_date",
                "TELEMETRY_SMR_NUMERIC",
                "sample_result_severity_order",
            ),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["fluid"],
        ),
        "maintenance": SourceSpec(
            key="maintenance",
            required_columns=MAINTENANCE_COLUMNS,
            optional_columns=(),
            date_column="event_date",
            deduplication_keys=(
                "full_model",
                "SERIAL",
                "EVENT_NAME_ID",
                "event_date",
                "is_monitor_reset",
                "is_overdue",
                "is_due_now",
            ),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["maintenance"],
        ),
        "operation": SourceSpec(
            key="operation",
            required_columns=OPERATION_COLUMNS,
            optional_columns=(),
            date_column="LOCAL_DATE",
            deduplication_keys=("full_model", "SERIAL", "LOCAL_DATE"),
            retention_days=config.SCORING_HISTORY_RETENTION_DAYS["operation"],
        ),
        target_key: target_spec,
    }


def _resolve_alias(directory: Path, source_key: str, required: bool) -> Path | None:
    """Resolve one incoming file using the configured canonical-name aliases."""
    aliases = tuple(config.INCOMING_SOURCE_FILE_ALIASES[source_key])
    for name in aliases:
        path = directory / name
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(
            f"Incoming {source_key} file not found under {directory}. "
            f"Accepted names: {aliases}"
        )
    return None


def resolve_incoming_files(incoming_dir: Path) -> dict[str, Path | None]:
    """Resolve required incoming source paths and the optional target refresh."""
    incoming_dir = Path(incoming_dir)
    if not incoming_dir.exists():
        raise FileNotFoundError(f"Incoming data directory does not exist: {incoming_dir}")
    target_key = str(config.TARGET_SOURCE)
    resolved: dict[str, Path | None] = {
        key: _resolve_alias(incoming_dir, key, required=True)
        for key in ("fault", "fluid", "maintenance", "operation")
    }
    resolved[target_key] = _resolve_alias(
        incoming_dir,
        target_key,
        required=not bool(config.INCOMING_ALLOW_MISSING_TARGET_REFRESH),
    )
    return resolved


def _read_header(path: Path) -> tuple[str, ...]:
    """Return CSV column names without loading data rows."""
    return tuple(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def _validate_source_schema(path: Path, spec: SourceSpec) -> tuple[str, ...]:
    """Validate required columns and return the columns retained for scoring."""
    header = _read_header(path)
    missing = sorted(set(spec.required_columns).difference(header))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    optional = [column for column in spec.optional_columns if column in header]
    return tuple(dict.fromkeys([*spec.required_columns, *optional]))


def _read_filtered_source(
    path: Path,
    spec: SourceSpec,
    score_date: pd.Timestamp,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Read required CSV columns and retain only rows usable at ``score_date``."""
    usecols = _validate_source_schema(path, spec)
    start_date = (
        score_date - pd.Timedelta(days=int(spec.retention_days))
        if spec.retention_days is not None
        else None
    )
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=list(usecols),
        chunksize=chunk_size,
        low_memory=False,
        encoding="utf-8-sig",
    ):
        parsed = pd.to_datetime(chunk[spec.date_column], errors="coerce", format="mixed")
        mask = parsed.notna() & parsed.lt(score_date)
        if start_date is not None:
            mask &= parsed.ge(start_date)
        if not mask.any():
            continue
        selected = chunk.loc[mask].copy()
        selected[spec.date_column] = parsed.loc[mask].dt.strftime("%Y-%m-%d")
        frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=usecols)
    return pd.concat(frames, ignore_index=True)


def _base_source_path(source_key: str) -> Path:
    """Return the retained historical source path for a logical source key."""
    mapping = {
        "fault": Path(config.FAULT_FILE),
        "fluid": Path(config.FLUID_FILE),
        "maintenance": Path(config.MAINTENANCE_FILE),
        "operation": Path(config.OPERATION_FILE),
        "physical_failure": Path(config.PHYSICAL_FAILURE_FILE),
        "warranty": Path(config.WARRANTY_FILE),
    }
    return mapping[source_key]


def _merge_historical_and_incoming(
    historical: pd.DataFrame,
    incoming: pd.DataFrame,
    spec: SourceSpec,
) -> pd.DataFrame:
    """Append only genuinely new incoming records to retained history.

    Duplicate rows already present inside either source are preserved because
    the training feature builder counted those rows. Only incoming records whose
    source-specific key is already present in retained history are suppressed.
    """
    if historical.empty:
        return incoming.reset_index(drop=True)
    if incoming.empty:
        return historical.reset_index(drop=True)
    keys = [key for key in spec.deduplication_keys if key in historical.columns]
    keys = [key for key in keys if key in incoming.columns]
    if not keys:
        return pd.concat([historical, incoming], ignore_index=True)
    historical_keys = pd.MultiIndex.from_frame(
        historical[keys].astype("string").fillna("<NA>")
    )
    incoming_keys = pd.MultiIndex.from_frame(
        incoming[keys].astype("string").fillna("<NA>")
    )
    new_incoming = incoming.loc[~incoming_keys.isin(historical_keys)]
    return pd.concat([historical, new_incoming], ignore_index=True)


def _machine_keys(frame: pd.DataFrame, model_column: str, serial_column: str) -> set[str]:
    """Create normalized machine keys for fleet-completeness validation."""
    model = frame[model_column].astype("string").str.strip().str.upper()
    serial = frame[serial_column].astype("string").str.extract(r"(\d+)", expand=False)
    return set((model + "-" + serial).dropna().tolist())


def load_retained_machine_roster() -> tuple[str, ...]:
    """Load the complete retained fleet roster used for dense production scoring."""
    roster_path = Path(config.OPERATION_ROSTER_FILE)
    if roster_path.exists():
        roster = pd.read_csv(roster_path, usecols=["machine_key"])
        return tuple(roster["machine_key"].dropna().astype(str).unique().tolist())

    operation_path = Path(config.OPERATION_FILE)
    if not operation_path.exists():
        raise FileNotFoundError(
            "A complete machine roster is required for production scoring. "
            f"Neither {roster_path} nor {operation_path} exists."
        )
    identities = pd.read_csv(
        operation_path,
        usecols=["full_model", "SERIAL"],
        low_memory=False,
    ).dropna(subset=["full_model", "SERIAL"])
    return tuple(sorted(_machine_keys(identities, "full_model", "SERIAL")))


def _sha256(path: Path) -> str:
    """Calculate a streaming SHA-256 digest for an audit artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def prepare_scoring_bundle(
    incoming_dir: Path,
    score_date: pd.Timestamp | str,
    work_dir: Path | None = None,
) -> PreparedScoringBundle:
    """Prepare deduplicated source files for one incoming production score date.

    Fault and operation are read from the incoming 90-day refresh. Fluid,
    maintenance, and target events are merged with retained historical sources
    so long-memory production features remain consistent with model training.
    """
    score_date = pd.Timestamp(score_date).normalize()
    if pd.isna(score_date):
        raise ValueError("A valid production score date is required.")
    incoming = resolve_incoming_files(Path(incoming_dir))
    specs = source_specs()
    target_key = str(config.TARGET_SOURCE)
    work_dir = Path(work_dir or config.PRODUCTION_SCORING_WORK_DIR / score_date.strftime("%Y%m%d"))
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    prepared_paths: dict[str, Path] = {}
    manifest_sources: dict[str, dict[str, object]] = {}
    incoming_frames: dict[str, pd.DataFrame] = {}

    for source_key in ("fault", "fluid", "maintenance", "operation", target_key):
        spec = specs[source_key]
        incoming_path = incoming.get(source_key)
        incoming_frame = (
            _read_filtered_source(Path(incoming_path), spec, score_date)
            if incoming_path is not None
            else pd.DataFrame(columns=spec.required_columns)
        )
        incoming_frames[source_key] = incoming_frame

        # Fault and operation are fully defined by the latest lookback refresh.
        # Longer-memory sources are merged with the retained historical extract.
        merge_history = source_key in {"fluid", "maintenance", target_key}
        base_path = _base_source_path(source_key)
        if merge_history:
            if not base_path.exists():
                raise FileNotFoundError(
                    f"Retained historical {source_key} source is required: {base_path}"
                )
            base_frame = _read_filtered_source(base_path, spec, score_date)
            combined = _merge_historical_and_incoming(
                base_frame, incoming_frame, spec
            )
        else:
            combined = incoming_frame
        if combined.empty and source_key == "operation":
            raise ValueError(
                f"Prepared {source_key} source is empty for score date {score_date.date()}."
            )
        output_path = work_dir / f"{source_key}.csv"
        combined.to_csv(output_path, index=False)
        prepared_paths[source_key] = output_path

        dates = pd.to_datetime(combined[spec.date_column], errors="coerce")
        manifest_sources[source_key] = {
            "incoming_file": str(incoming_path) if incoming_path else None,
            "historical_file": str(base_path) if merge_history else None,
            "prepared_file": str(output_path),
            "prepared_rows": int(len(combined)),
            "prepared_date_min": str(dates.min().date()) if dates.notna().any() else None,
            "prepared_date_max": str(dates.max().date()) if dates.notna().any() else None,
            "sha256": _sha256(output_path),
        }

    operation_dates = pd.to_datetime(
        incoming_frames["operation"]["LOCAL_DATE"], errors="coerce"
    )
    required_start = score_date - pd.Timedelta(
        days=int(config.INCOMING_SOURCE_HISTORY_DAYS)
    )
    if operation_dates.empty or operation_dates.min() > required_start:
        raise ValueError(
            "Incoming operation history does not cover the configured "
            f"{config.INCOMING_SOURCE_HISTORY_DAYS}-day feature window. "
            f"Required start <= {required_start.date()}."
        )
    if operation_dates.max() < score_date - pd.Timedelta(days=1):
        raise ValueError(
            "Incoming operation history is stale for the requested score "
            f"date {score_date.date()}; latest usable date is "
            f"{operation_dates.max().date()}."
        )
    operation_keys = _machine_keys(
        incoming_frames["operation"], "full_model", "SERIAL"
    )
    evidence_keys = set()
    evidence_keys |= _machine_keys(incoming_frames["fault"], "full_model", "serial_number")
    evidence_keys |= _machine_keys(incoming_frames["fluid"], "FULL_MODEL", "SERIAL")
    evidence_keys |= _machine_keys(
        incoming_frames["maintenance"], "full_model", "SERIAL"
    )
    missing_from_operation = sorted(evidence_keys.difference(operation_keys))
    if missing_from_operation and bool(config.INCOMING_REQUIRE_COMPLETE_FLEET_OPERATION):
        raise ValueError(
            "Incoming operation history does not contain the complete observed "
            f"fleet. Missing {len(missing_from_operation):,} machines; examples: "
            f"{missing_from_operation[:10]}"
        )
    if missing_from_operation:
        print(
            "WARNING: incoming operation history has no rows for "
            f"{len(missing_from_operation):,} machines observed in other sources. "
            "They remain in the scoring roster through the source union.",
            flush=True,
        )

    source_files = SourceFileSet(
        fault=prepared_paths["fault"],
        fluid=prepared_paths["fluid"],
        maintenance=prepared_paths["maintenance"],
        operation=prepared_paths["operation"],
        target=prepared_paths[target_key],
    )
    manifest = {
        "score_date": str(score_date.date()),
        "target_source": config.TARGET_SOURCE,
        "incoming_history_days": int(config.INCOMING_SOURCE_HISTORY_DAYS),
        "production_variant": config.PRODUCTION_MODEL_SETTINGS["variant"],
        "sources": manifest_sources,
        "operation_missing_observed_machine_count": len(missing_from_operation),
        "operation_missing_observed_machine_examples": missing_from_operation[:20],
    }
    manifest_path = work_dir / "scoring_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedScoringBundle(
        score_date=score_date,
        source_files=source_files,
        manifest_path=manifest_path,
        work_dir=work_dir,
    )


def build_incoming_scoring_snapshot(bundle: PreparedScoringBundle) -> pd.DataFrame:
    """Build one leakage-safe feature snapshot from a prepared source bundle."""
    artifacts = load_production_artifacts()
    selected_codes = tuple(config.PRODUCTION_SELECTED_FAULT_CODES)
    sources = load_sources(
        include_operation_features=True,
        source_files=bundle.source_files,
        candidate_serious_codes=selected_codes,
        machine_roster=load_retained_machine_roster(),
    )
    snapshot = build_snapshot_dataframe(
        sources,
        [bundle.score_date],
        include_targets=False,
        verbose=True,
    )
    missing_features = sorted(set(artifacts.features).difference(snapshot.columns))
    if missing_features:
        raise ValueError(
            "Incoming snapshot did not create all approved model features: "
            f"{missing_features}"
        )
    return snapshot


def run_incoming_scoring(
    incoming_dir: Path,
    score_date: pd.Timestamp | str,
    output_dir: Path | None = None,
    include_explanations: bool = True,
) -> tuple[pd.DataFrame, dict[str, object], PreparedScoringBundle]:
    """Prepare incoming sources, score the fleet, assign tiers, and save outputs."""
    bundle = prepare_scoring_bundle(incoming_dir, score_date)
    snapshot = build_incoming_scoring_snapshot(bundle)
    artifacts = load_production_artifacts()
    scores = score_snapshot_dataframe(
        snapshot,
        artifacts=artifacts,
        include_explanations=include_explanations,
    )

    output_dir = Path(output_dir or config.PRODUCTION_SCORING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_token = bundle.score_date.strftime("%Y%m%d")
    snapshot_path = output_dir / f"prepared_snapshot_{date_token}.csv.gz"
    score_path = output_dir / f"machine_risk_scores_{date_token}.csv"
    summary_path = output_dir / f"machine_risk_summary_{date_token}.csv"
    manifest_copy = output_dir / f"scoring_source_manifest_{date_token}.json"
    snapshot.to_csv(snapshot_path, index=False, compression="gzip")
    scores.to_csv(score_path, index=False)
    shutil.copy2(bundle.manifest_path, manifest_copy)

    summary = scoring_summary(scores, artifacts.variant, score_path.name)
    summary.update(
        {
            "source_manifest": manifest_copy.name,
            "prepared_snapshot": snapshot_path.name,
            "tier_policy_source": "config.PRODUCTION_TIER_POLICY",
            "model_file": artifacts.model_path.name,
            "metadata_file": artifacts.metadata_path.name,
        }
    )
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return scores, summary, bundle
