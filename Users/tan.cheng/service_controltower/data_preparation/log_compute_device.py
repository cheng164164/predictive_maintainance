"""
Print CPU/GPU diagnostics for local runs and Azure ML command jobs.

This script has no required third-party dependencies. It uses nvidia-smi when it
is available and optionally reports torch CUDA information when PyTorch is
installed in the environment.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import config as run_config
except ImportError:  # pragma: no cover
    run_config = None


def cfg(name: str, default):
    return getattr(run_config, name, default) if run_config is not None else default


def bool_from_env(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off", ""}:
        return False
    return None


def log(message: str) -> None:
    print(f"[device-check] {message}", flush=True)


def run_command(command: list[str], timeout_seconds: int = 30) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        combined = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, combined.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return False, f"{type(exc).__name__}: {exc}"


def detect_device() -> dict:
    info: dict = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID", ""),
        "azureml_compute_name": os.environ.get("AZUREML_COMPUTE_NAME", ""),
        "aml_compute_target": os.environ.get("AML_COMPUTE_TARGET", ""),
        "aml_selected_compute_name": os.environ.get("AML_SELECTED_COMPUTE_NAME", ""),
        "aml_selected_environment": os.environ.get("AML_SELECTED_ENVIRONMENT", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_path": shutil.which("nvidia-smi") or "",
        "nvidia_smi_available": False,
        "nvidia_smi_output": "",
        "torch_installed": False,
        "torch_version": "",
        "torch_cuda_version": "",
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_device_names": [],
        "effective_device": "cpu",
    }

    if info["nvidia_smi_path"]:
        ok, output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        info["nvidia_smi_available"] = ok
        info["nvidia_smi_output"] = output

    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["torch_version"] = str(getattr(torch, "__version__", ""))
        info["torch_cuda_version"] = str(getattr(torch.version, "cuda", ""))
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if info["torch_cuda_available"]:
            info["torch_cuda_device_count"] = int(torch.cuda.device_count())
            info["torch_cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover
        info["torch_installed"] = False
        info["torch_version"] = f"not available ({type(exc).__name__}: {exc})"

    if info["nvidia_smi_available"] or info["torch_cuda_available"]:
        info["effective_device"] = "gpu"

    return info


def main() -> None:
    info = detect_device()

    log(f"effective_device={info['effective_device']}")
    log(f"hostname={info['hostname']}")
    log(f"python={info['python_version']}")
    log(f"platform={info['platform']}")
    log(f"AZUREML_RUN_ID={info['azureml_run_id'] or '<not set>'}")
    log(f"AZUREML_COMPUTE_NAME={info['azureml_compute_name'] or '<not set>'}")
    log(f"AML_COMPUTE_TARGET={info['aml_compute_target'] or '<not set>'}")
    log(f"AML_SELECTED_COMPUTE_NAME={info['aml_selected_compute_name'] or '<not set>'}")
    log(f"AML_SELECTED_ENVIRONMENT={info['aml_selected_environment'] or '<not set>'}")
    log(f"CUDA_VISIBLE_DEVICES={info['cuda_visible_devices'] or '<not set>'}")
    log(f"nvidia-smi path={info['nvidia_smi_path'] or '<not found>'}")
    if info["nvidia_smi_output"]:
        log(f"nvidia-smi GPU summary={info['nvidia_smi_output']}")
    else:
        log("nvidia-smi GPU summary=<none>")
    log(
        "torch CUDA: "
        f"installed={info['torch_installed']}, "
        f"torch_version={info['torch_version']}, "
        f"torch_cuda_version={info['torch_cuda_version']}, "
        f"cuda_available={info['torch_cuda_available']}, "
        f"device_count={info['torch_cuda_device_count']}, "
        f"device_names={info['torch_cuda_device_names']}"
    )

    output_dir = Path(cfg("OUTPUT_DIR", Path("outputs") / "data_preparation" / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "device_report.json"
    report_path.write_text(json.dumps(info, indent=2))
    log(f"device report saved to {report_path}")

    compute_target = os.environ.get("AML_COMPUTE_TARGET", str(cfg("AML_COMPUTE_TARGET", "cpu"))).strip().lower()
    expect_gpu = compute_target == "gpu"

    log(f"compute_target={compute_target}")
    log(f"expected_device={'gpu' if expect_gpu else 'cpu'}")
    if expect_gpu and info["effective_device"] != "gpu":
        raise RuntimeError(
            "GPU is required by config, but no GPU is visible. Check Azure ML node allocation, "
            "quota/availability, CUDA driver visibility, and the selected curated environment."
        )


if __name__ == "__main__":
    main()
