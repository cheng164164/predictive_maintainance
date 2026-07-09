"""
Submit the snapshot dataframe build as an Azure Machine Learning command job.

Option B design:
    - do NOT upload enriched_data/ as code
    - read input CSVs from Blob/Azure ML data input
    - use the project root as the AML code folder, protected by .amlignore
    - do NOT create a temporary .aml_job_source folder
    - write each run to its own datastore folder named by the AML job name
    - keep historical results in the datastore; do not clear old outputs

Run from service_controltower/ or service_controltower/data_preparation/:

    python data_preparation/submit_snapshot_build_aml_job.py
"""

from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import ManagedIdentityConfiguration, UserIdentityConfiguration
from azure.identity import (
    AzureCliCredential,
    DefaultAzureCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
)

import config as run_config


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def cfg(name: str, default=None):
    return getattr(run_config, name, default)


def require_value(name: str) -> str:
    value = str(cfg(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required config value: {name}")
    return value


def require_existing_path(path_value, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def project_root() -> Path:
    return Path(cfg("PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()


# -----------------------------------------------------------------------------
# AML code-folder handling
# -----------------------------------------------------------------------------
def default_amlignore_text() -> str:
    """Return ignore rules that keep AML code upload lightweight.

    Azure ML still needs a code folder for the command job. We use the project
    root, but .amlignore prevents large local folders from being hashed/uploaded.
    """
    return "\n".join(
        [
            "# Local environments and Git metadata",
            ".git/",
            ".venv/",
            "venv/",
            ".env",
            "__pycache__/",
            "*.pyc",
            ".ipynb_checkpoints/",
            "",
            "# Old temporary AML staging folder, if it exists from earlier versions",
            ".aml_job_source/",
            "",
            "# Data is passed as AML input, not uploaded as code",
            "enriched_data/",
            "",
            "# Outputs/results are written to AML output URI, not uploaded as code",
            "data_preparation/output/",
            "data_preparation/aml_run_results/",
            "outputs/",
            "",
            "# Large/generated files",
            "*.parquet",
            "*.zip",
            "*.log",
            "*.tmp",
        ]
    ) + "\n"


def ensure_amlignore(code_dir: Path) -> Path:
    """Ensure the selected AML code directory has an .amlignore file.

    This does not create a staging folder. It only ensures Azure ML does not scan
    .venv, enriched_data, outputs, and other large local folders.
    """
    amlignore = code_dir / ".amlignore"
    if not amlignore.exists():
        amlignore.write_text(default_amlignore_text(), encoding="utf-8")
        print(f"  created .amlignore at: {amlignore}", flush=True)
        return amlignore

    existing = amlignore.read_text(encoding="utf-8", errors="ignore")
    required_patterns = [
        ".venv/",
        ".git/",
        "enriched_data/",
        "data_preparation/output/",
        "data_preparation/aml_run_results/",
        ".aml_job_source/",
    ]
    missing = [p for p in required_patterns if p not in existing]
    if missing:
        with amlignore.open("a", encoding="utf-8") as f:
            f.write("\n# Added by submit_snapshot_build_aml_job.py to keep AML upload lightweight\n")
            for pattern in missing:
                f.write(pattern + "\n")
        print(f"  updated .amlignore with missing patterns: {missing}", flush=True)
    return amlignore


def resolve_code_dir() -> Path:
    """Use the project root as the AML code folder; do not stage/copy code."""
    root = project_root()
    code_dir = Path(cfg("AML_CODE_DIR", root)).expanduser().resolve()
    if not code_dir.exists():
        raise FileNotFoundError(f"AML_CODE_DIR not found: {code_dir}")

    # The build command expects these project-root relative paths.
    require_existing_path(code_dir / "data_preparation" / "build_snapshot_dataframe.py", "build script")
    require_existing_path(code_dir / "data_preparation" / "log_compute_device.py", "device logger")
    require_existing_path(code_dir / "requirements.txt", "requirements.txt")
    ensure_amlignore(code_dir)
    return code_dir


def relative_to_code_dir(path: Path, code_dir: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(code_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{label} must be inside code_dir. {label}={path}; code_dir={code_dir}") from exc


def build_aml_command(code_dir: Path, job_name: str, compute_target: str, compute_name: str, environment_name: str, require_gpu: bool) -> str:
    build_script = require_existing_path(code_dir / "data_preparation" / "build_snapshot_dataframe.py", "build script")
    log_device_script = require_existing_path(code_dir / "data_preparation" / "log_compute_device.py", "device logger")
    requirements_path = require_existing_path(code_dir / "requirements.txt", "requirements.txt")

    build_rel = relative_to_code_dir(build_script, code_dir, "build script")
    log_rel = relative_to_code_dir(log_device_script, code_dir, "device logger")
    req_rel = relative_to_code_dir(requirements_path, code_dir, "requirements.txt")

    # Important: Azure ML expressions such as ${{inputs.input_data}} and
    # ${{outputs.output_data}} are resolved inside the command string. They are
    # not reliably resolved when placed directly in environment_variables. The
    # shell exports below make the resolved paths visible to config.py and the
    # build/device scripts without requiring command-line arguments.
    mini_ids = csv_list(cfg("AML_MINI_RUN_MODEL_IDS", []))
    mini_enabled = str(bool(cfg("AML_MINI_RUN_ENABLED", False))).lower()
    mini_count = str(cfg("AML_MINI_RUN_MACHINE_COUNT", 2))

    commands: list[str] = [
        "export PYTHONUNBUFFERED=1",
        "export AML_IS_REMOTE_JOB=1",
        f"export AML_JOB_NAME={shlex.quote(job_name)}",
        f"export AML_COMPUTE_TARGET={shlex.quote(compute_target)}",
        f"export AML_SELECTED_COMPUTE_NAME={shlex.quote(compute_name)}",
        f"export AML_SELECTED_ENVIRONMENT={shlex.quote(environment_name)}",
        'export AML_INPUT_DIR="${{inputs.input_data}}"',
        'export SNAPSHOT_INPUT_DIR="$AML_INPUT_DIR"',
        'export AML_OUTPUT_DIR="${{outputs.output_data}}"',
        'export SNAPSHOT_OUTPUT_DIR="$AML_OUTPUT_DIR"',
        'export AML_SOURCE_SNAPSHOT_DIR="$AML_OUTPUT_DIR/source_snapshots"',
        'export SNAPSHOT_SOURCE_SNAPSHOT_DIR="$AML_SOURCE_SNAPSHOT_DIR"',
        'export AML_PROGRESS_LOG_PATH="$AML_OUTPUT_DIR/snapshot_build_progress_log.csv"',
        'export SNAPSHOT_PROGRESS_LOG_PATH="$AML_PROGRESS_LOG_PATH"',
        'export AML_ARTIFACT_MANIFEST_PATH="$AML_OUTPUT_DIR/snapshot_build_artifact_manifest.csv"',
        'export SNAPSHOT_ARTIFACT_MANIFEST_PATH="$AML_ARTIFACT_MANIFEST_PATH"',
        f"export AML_MINI_RUN_ENABLED={shlex.quote(mini_enabled)}",
        f"export SNAPSHOT_MINI_RUN_ENABLED={shlex.quote(mini_enabled)}",
        f"export AML_MINI_RUN_MACHINE_COUNT={shlex.quote(mini_count)}",
        f"export SNAPSHOT_MINI_RUN_MACHINE_COUNT={shlex.quote(mini_count)}",
        f"export AML_MINI_RUN_MODEL_IDS={shlex.quote(mini_ids)}",
        f"export SNAPSHOT_MINI_RUN_MODEL_IDS={shlex.quote(mini_ids)}",
        f"export MINI_RUN_ENABLED={shlex.quote(mini_enabled)}",
        f"export MINI_RUN_MACHINE_COUNT={shlex.quote(mini_count)}",
        f"export MINI_RUN_MODEL_IDS={shlex.quote(mini_ids)}",
        f"export REQUIRE_GPU={'1' if require_gpu else '0'}",
        f"export AML_REQUIRE_GPU={'1' if require_gpu else '0'}",
        'mkdir -p "$AML_OUTPUT_DIR" "$AML_SOURCE_SNAPSHOT_DIR"',
    ]

    commands.append(f"python {shlex.quote(log_rel)}")

    if bool(cfg("AML_INSTALL_REQUIREMENTS", True)):
        commands.append(
            "python -m pip install --disable-pip-version-check --no-input "
            f"-r {shlex.quote(req_rel)}"
        )

    # Log again after requirements are installed. This second log may include
    # torch CUDA information if the curated environment includes PyTorch.
    commands.append(f"python {shlex.quote(log_rel)}")
    commands.append(f"python {shlex.quote(build_rel)}")
    return " && ".join(commands)


# -----------------------------------------------------------------------------
# CPU/GPU target selection
# -----------------------------------------------------------------------------
def normalize_compute_target(value: str) -> str:
    target = str(value or "").strip().lower()
    if target in {"cpu", "standard", "standard_cpu"}:
        return "cpu"
    if target in {"gpu", "cuda", "nvidia"}:
        return "gpu"
    raise ValueError('AML_COMPUTE_TARGET must be either "cpu" or "gpu".')


def first_non_empty(*values) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def resolve_compute_settings() -> tuple[str, str, str, bool]:
    """Resolve compute target, compute name, environment, and GPU requirement.

    AML_COMPUTE_TARGET controls whether the job uses the CPU or GPU cluster.
    AML_COMPUTE_NAME and AML_ENVIRONMENT are optional direct overrides kept for
    backward compatibility. If they are blank, target-specific settings are used.
    """
    target = normalize_compute_target(str(cfg("AML_COMPUTE_TARGET", "gpu")))

    if target == "cpu":
        compute_name = first_non_empty(cfg("AML_COMPUTE_NAME", ""), cfg("AML_CPU_COMPUTE_NAME", ""))
        environment_name = first_non_empty(cfg("AML_ENVIRONMENT", ""), cfg("AML_CPU_ENVIRONMENT", ""))
        require_gpu = False
    else:
        compute_name = first_non_empty(cfg("AML_COMPUTE_NAME", ""), cfg("AML_GPU_COMPUTE_NAME", ""))
        environment_name = first_non_empty(cfg("AML_ENVIRONMENT", ""), cfg("AML_GPU_ENVIRONMENT", ""))
        require_gpu = bool(cfg("AML_REQUIRE_GPU", False))

    if not compute_name:
        raise ValueError(
            f"No compute name resolved for AML_COMPUTE_TARGET={target!r}. "
            "Set AML_CPU_COMPUTE_NAME or AML_GPU_COMPUTE_NAME in config.py."
        )
    if not environment_name:
        raise ValueError(
            f"No environment resolved for AML_COMPUTE_TARGET={target!r}. "
            "Set AML_CPU_ENVIRONMENT or AML_GPU_ENVIRONMENT in config.py."
        )

    if target == "cpu" and bool(cfg("AML_REQUIRE_GPU", False)):
        print(
            "  note: AML_COMPUTE_TARGET='cpu', so AML_REQUIRE_GPU is ignored for this run.",
            flush=True,
        )

    return target, compute_name, environment_name, require_gpu


# -----------------------------------------------------------------------------
# AML job naming/output helpers
# -----------------------------------------------------------------------------
def slugify_job_name(value: str) -> str:
    """Return an Azure-ML-safe job name component."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not value:
        value = "snapshot-build"
    if not value[0].isalpha():
        value = f"job-{value}"
    return value[:40].rstrip("-")


def generate_job_name(compute_target: str) -> str:
    explicit_name = str(cfg("AML_JOB_NAME", "")).strip()
    if explicit_name:
        return slugify_job_name(explicit_name)

    prefix = slugify_job_name(str(cfg("AML_JOB_NAME_PREFIX", "snapshot-build")))
    if bool(cfg("AML_INCLUDE_COMPUTE_TARGET_IN_JOB_NAME", True)):
        prefix = slugify_job_name(f"{prefix}-{compute_target}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return slugify_job_name(f"{prefix}-{timestamp}")


def build_run_display_name(base_display_name: str, job_name: str, compute_target: str) -> str:
    """Return a unique AML display name for easier filtering in Azure ML Studio."""
    base_display_name = str(base_display_name).strip() or "snapshot-build"
    job_name = str(job_name).strip()
    prefix = slugify_job_name(str(cfg("AML_JOB_NAME_PREFIX", "snapshot-build")))

    suffix = job_name
    for removable_prefix in (
        slugify_job_name(f"{prefix}-{compute_target}"),
        prefix,
    ):
        if suffix.startswith(removable_prefix + "-"):
            suffix = suffix[len(removable_prefix) + 1 :]
            break

    if suffix:
        return f"{base_display_name}-{compute_target}-{suffix}"
    return f"{base_display_name}-{compute_target}"


def normalize_base_uri(uri: str) -> str:
    uri = uri.strip()
    if not uri:
        raise ValueError("Output base URI is empty.")
    if not uri.endswith("/"):
        uri += "/"
    return uri


def output_base_uri() -> str:
    """Get the datastore folder under which job-name subfolders are written."""
    base = str(cfg("AML_OUTPUT_BASE_DATA_URI", "")).strip()
    if base:
        return normalize_base_uri(base)

    # Backward compatibility for older config.py files that only have
    # AML_OUTPUT_DATA_URI. If it points to .../latest/, use its parent as base.
    old = str(cfg("AML_OUTPUT_DATA_URI", "")).strip()
    if not old:
        raise ValueError("Missing AML_OUTPUT_BASE_DATA_URI or AML_OUTPUT_DATA_URI.")
    old = normalize_base_uri(old)
    if old.rstrip("/").endswith("/latest"):
        return old.rstrip("/").rsplit("/", 1)[0] + "/"
    return old


def join_uri(base_uri: str, *parts: str) -> str:
    out = normalize_base_uri(base_uri)
    clean_parts = [str(p).strip("/") for p in parts if str(p).strip("/")]
    if clean_parts:
        out += "/".join(clean_parts) + "/"
    return out


def build_data_identity():
    identity_mode = str(cfg("AML_DATA_IDENTITY", "user")).strip().lower()
    if identity_mode in {"user", "user_identity"}:
        return UserIdentityConfiguration()
    if identity_mode in {"managed", "managed_identity"}:
        return ManagedIdentityConfiguration()
    if identity_mode in {"none", "", "null"}:
        return None
    raise ValueError("AML_DATA_IDENTITY must be one of: user, managed, none")


def build_submit_credential():
    """Build the credential used to submit the AML job.

    Default is Azure CLI auth because it avoids the 300-second browser timeout
    seen with InteractiveBrowserCredential on remote compute. Run `az login`
    before submitting, or set AML_AUTH_MODE to another option.
    """
    auth_mode = str(cfg("AML_AUTH_MODE", "azure_cli")).strip().lower()
    if auth_mode in {"azure_cli", "cli"}:
        print("  submit auth mode: AzureCliCredential", flush=True)
        return AzureCliCredential()
    if auth_mode in {"managed", "managed_identity"}:
        print("  submit auth mode: ManagedIdentityCredential", flush=True)
        return ManagedIdentityCredential()
    if auth_mode in {"interactive", "browser"}:
        print("  submit auth mode: InteractiveBrowserCredential", flush=True)
        return InteractiveBrowserCredential()
    if auth_mode in {"default"}:
        print("  submit auth mode: DefaultAzureCredential without interactive browser", flush=True)
        return DefaultAzureCredential(exclude_interactive_browser_credential=True)
    raise ValueError("AML_AUTH_MODE must be one of: azure_cli, managed, interactive, default")


def csv_list(values) -> str:
    return ",".join(str(v).strip() for v in values if str(v).strip())


def write_last_job_info(job_name: str, output_uri: str, returned_job, compute_target: str, compute_name: str, environment_name: str) -> None:
    """Write a small local pointer so the downloader can fetch this run later."""
    root = project_root()
    default_path = root / "data_preparation" / "aml_last_submitted_job.json"
    path = Path(cfg("AML_LAST_JOB_INFO_PATH", default_path)).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "job_name": job_name,
        "returned_job_name": getattr(returned_job, "name", None),
        "output_uri": output_uri,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "studio_url": getattr(returned_job, "studio_url", None),
        "experiment_name": str(cfg("AML_EXPERIMENT_NAME", "")),
        "compute_target": compute_target,
        "compute": compute_name,
        "environment": environment_name,
        "mini_run_enabled": bool(cfg("AML_MINI_RUN_ENABLED", False)),
    }
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  last job info written: {path}", flush=True)


def main() -> None:
    subscription_id = require_value("AML_SUBSCRIPTION_ID")
    resource_group = require_value("AML_RESOURCE_GROUP")
    workspace_name = require_value("AML_WORKSPACE_NAME")
    compute_target, compute_name, environment_name, require_gpu = resolve_compute_settings()
    experiment_name = require_value("AML_EXPERIMENT_NAME")
    display_name_base = require_value("AML_DISPLAY_NAME")

    input_data_uri = require_value("AML_INPUT_DATA_URI")
    input_mode = str(cfg("AML_INPUT_MODE", "download")).strip()
    output_mode = str(cfg("AML_OUTPUT_MODE", "rw_mount")).strip()

    code_dir = resolve_code_dir()

    job_name = generate_job_name(compute_target)
    display_name = build_run_display_name(display_name_base, job_name, compute_target)
    job_output_uri = join_uri(output_base_uri(), job_name)
    command_line = build_aml_command(
        code_dir=code_dir,
        job_name=job_name,
        compute_target=compute_target,
        compute_name=compute_name,
        environment_name=environment_name,
        require_gpu=require_gpu,
    )

    print("Submitting Azure ML snapshot build job with:", flush=True)
    print(f"  workspace: {workspace_name}", flush=True)
    print(f"  resource group: {resource_group}", flush=True)
    print(f"  compute target: {compute_target}", flush=True)
    print(f"  compute: {compute_name}", flush=True)
    print(f"  environment: {environment_name}", flush=True)
    print(f"  code dir: {code_dir}", flush=True)
    print("  code upload note: .amlignore excludes .venv, enriched_data, outputs, and old results", flush=True)
    print(f"  input data uri: {input_data_uri}", flush=True)
    print(f"  input mode: {input_mode}", flush=True)
    print(f"  output base uri: {output_base_uri()}", flush=True)
    print(f"  output run uri: {job_output_uri}", flush=True)
    print(f"  output mode: {output_mode}", flush=True)
    print(f"  job name: {job_name}", flush=True)
    print(f"  display name: {display_name}", flush=True)
    print(f"  command: {command_line}", flush=True)
    print(f"  AML mini mode: {bool(cfg('AML_MINI_RUN_ENABLED', False))}", flush=True)
    print(f"  AML mini machine count: {cfg('AML_MINI_RUN_MACHINE_COUNT', 2)}", flush=True)
    print(f"  AML mini model ids: {cfg('AML_MINI_RUN_MODEL_IDS', [])}", flush=True)
    print(f"  AML_REQUIRE_GPU configured: {bool(cfg('AML_REQUIRE_GPU', False))}", flush=True)
    print(f"  REQUIRE_GPU resolved for this run: {require_gpu}", flush=True)
    print(f"  AML_DATA_IDENTITY: {cfg('AML_DATA_IDENTITY', 'user')}", flush=True)
    print("  Historical outputs: enabled; each run gets a job-name folder", flush=True)

    credential = build_submit_credential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    job = command(
        name=job_name,
        code=str(code_dir),
        command=command_line,
        inputs={
            "input_data": Input(type="uri_folder", path=input_data_uri, mode=input_mode),
        },
        outputs={
            "output_data": Output(type="uri_folder", path=job_output_uri, mode=output_mode),
        },
        environment=environment_name,
        compute=compute_name,
        experiment_name=experiment_name,
        display_name=display_name,
        identity=build_data_identity(),
        environment_variables={
            # Keep only constant values here. Runtime paths from Azure ML inputs
            # and outputs are exported inside the command string above so they
            # are resolved before Python starts.
            "PYTHONUNBUFFERED": "1",
        },
    )

    try:
        returned_job = ml_client.jobs.create_or_update(job)
    except Exception as exc:  # noqa: BLE001 - add practical guidance before re-raising
        print("\nAzure ML job submission failed.", flush=True)
        print("Most common fix from an Azure ML remote terminal:", flush=True)
        print("  az login --use-device-code", flush=True)
        print(f"  az account set --subscription {subscription_id}", flush=True)
        print("Then rerun this submit script.", flush=True)
        raise

    write_last_job_info(job_name, job_output_uri, returned_job, compute_target, compute_name, environment_name)

    print("Azure ML job submitted.", flush=True)
    print(f"  job name: {returned_job.name}", flush=True)
    print(f"  results datastore folder: {job_output_uri}", flush=True)
    if getattr(returned_job, "studio_url", None):
        print(f"  studio url: {returned_job.studio_url}", flush=True)

    if bool(cfg("AML_STREAM_LOGS", True)):
        print("Streaming Azure ML logs...", flush=True)
        ml_client.jobs.stream(returned_job.name)


if __name__ == "__main__":
    main()
