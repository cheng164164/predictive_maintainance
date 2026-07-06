"""
Submit the snapshot dataframe build as an Azure Machine Learning command job.

All parameters are read from config.py. Run without command-line arguments:

    python submit_snapshot_build_aml_job.py

The submitted job runs:
    1. log_compute_device.py
    2. pip install -r requirements.txt, if AML_INSTALL_REQUIREMENTS = True
    3. build_snapshot_dataframe.py

The device logger writes device_report.json to the configured OUTPUT_DIR and
prints whether the job can see CPU/GPU resources.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential

import config as run_config


def cfg(name: str, default=None):
    return getattr(run_config, name, default)


def require_value(name: str) -> str:
    value = str(cfg(name, "")).strip()
    if not value:
        raise ValueError(f"Missing required config value: {name}")
    return value


def require_existing_path(path_value, label: str) -> Path:
    path = Path(path_value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def relative_to_code_dir(path: Path, code_dir: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(code_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{label} must be inside AML_CODE_DIR so Azure ML can upload it. "
            f"{label}={path}; AML_CODE_DIR={code_dir}"
        ) from exc


def build_aml_command(code_dir: Path) -> str:
    build_script = require_existing_path(cfg("BUILD_SNAPSHOT_SCRIPT_PATH"), "BUILD_SNAPSHOT_SCRIPT_PATH")
    log_device_script = require_existing_path(cfg("LOG_DEVICE_SCRIPT_PATH"), "LOG_DEVICE_SCRIPT_PATH")
    requirements_path = require_existing_path(cfg("REQUIREMENTS_PATH"), "REQUIREMENTS_PATH")

    build_rel = relative_to_code_dir(build_script, code_dir, "BUILD_SNAPSHOT_SCRIPT_PATH")
    log_rel = relative_to_code_dir(log_device_script, code_dir, "LOG_DEVICE_SCRIPT_PATH")
    req_rel = relative_to_code_dir(requirements_path, code_dir, "REQUIREMENTS_PATH")

    commands: list[str] = []

    # Log first so GPU allocation/debug information appears even if package
    # installation fails later.
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


def main() -> None:
    subscription_id = require_value("AML_SUBSCRIPTION_ID")
    resource_group = require_value("AML_RESOURCE_GROUP")
    workspace_name = require_value("AML_WORKSPACE_NAME")
    compute_name = require_value("AML_COMPUTE_NAME")
    environment_name = require_value("AML_ENVIRONMENT")
    experiment_name = require_value("AML_EXPERIMENT_NAME")
    display_name = require_value("AML_DISPLAY_NAME")

    code_dir = require_existing_path(cfg("AML_CODE_DIR"), "AML_CODE_DIR")
    command_line = build_aml_command(code_dir)

    print("Submitting Azure ML snapshot build job with:", flush=True)
    print(f"  workspace: {workspace_name}", flush=True)
    print(f"  resource group: {resource_group}", flush=True)
    print(f"  compute: {compute_name}", flush=True)
    print(f"  environment: {environment_name}", flush=True)
    print(f"  code dir: {code_dir}", flush=True)
    print(f"  command: {command_line}", flush=True)
    print(f"  mini mode: {bool(cfg('MINI_RUN_ENABLED', False))}", flush=True)
    print(f"  AML_REQUIRE_GPU: {bool(cfg('AML_REQUIRE_GPU', False))}", flush=True)

    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    job = command(
        code=str(code_dir),
        command=command_line,
        environment=environment_name,
        compute=compute_name,
        experiment_name=experiment_name,
        display_name=display_name,
        environment_variables={
            "PYTHONUNBUFFERED": "1",
        },
    )

    returned_job = ml_client.jobs.create_or_update(job)

    print("Azure ML job submitted.", flush=True)
    print(f"  job name: {returned_job.name}", flush=True)
    if getattr(returned_job, "studio_url", None):
        print(f"  studio url: {returned_job.studio_url}", flush=True)

    if bool(cfg("AML_STREAM_LOGS", True)):
        print("Streaming Azure ML logs...", flush=True)
        ml_client.jobs.stream(returned_job.name)


if __name__ == "__main__":
    main()
