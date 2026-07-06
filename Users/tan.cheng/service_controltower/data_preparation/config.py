"""
Configuration for the predictive-maintenance snapshot build.

Current project layout supported by this config:

    service_controltower/
        data_preparation/
            config.py
            build_snapshot_dataframe.py
            log_compute_device.py
            submit_snapshot_build_aml_job.py
            output/
        enriched_data/
            fault_codes.csv
            machine.csv
            maintenance.csv
            warranty.csv                 # optional
            xgb_feature_freeze.xlsx       # optional
        requirements.txt

Run locally from the repository root or from data_preparation/:

    python data_preparation/build_snapshot_dataframe.py

or, if your terminal is already inside data_preparation/:

    python build_snapshot_dataframe.py
"""

from __future__ import annotations

from pathlib import Path


# -----------------------------------------------------------------------------
# Project-folder detection
# -----------------------------------------------------------------------------
def _candidate_project_roots() -> list[Path]:
    """Return likely service_controltower root folders.

    Azure ML sometimes exposes the same folder through both a /home/azureuser path
    and a /mnt/batch/... path.  Using absolute() instead of resolve() avoids
    forcing symlink resolution too early, and the scripts also validate the final
    files before running.
    """
    config_dir = Path(__file__).absolute().parent
    cwd = Path.cwd().absolute()

    candidates: list[Path] = []

    # Expected layout: service_controltower/data_preparation/config.py
    candidates.append(config_dir.parent)

    # Useful when running from service_controltower/ or data_preparation/.
    candidates.append(cwd)
    candidates.append(cwd.parent)

    # Walk upward from both locations and keep any folder that looks like project root.
    for base in (config_dir, cwd):
        for parent in [base, *base.parents]:
            candidates.append(parent)

    # Preserve order and remove duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def _detect_project_root() -> Path:
    """Find the root folder that contains both data_preparation and enriched_data."""
    for root in _candidate_project_roots():
        if (root / "data_preparation").exists() and (root / "enriched_data").exists():
            return root

    # Fallback to the expected parent of data_preparation.  The build script will
    # raise a clear FileNotFoundError if the input files are still not found.
    return Path(__file__).absolute().parent.parent


PROJECT_ROOT = _detect_project_root()
DATA_PREPARATION_DIR = PROJECT_ROOT / "data_preparation"
INPUT_DIR = PROJECT_ROOT / "enriched_data"
OUTPUT_DIR = DATA_PREPARATION_DIR / "output"


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
FAULT_CODES_PATH = INPUT_DIR / "fault_codes.csv"
MACHINE_PATH = INPUT_DIR / "machine.csv"
MAINTENANCE_PATH = INPUT_DIR / "maintenance.csv"

# Optional files. If warranty.csv does not exist, claim_next_45d is left blank.
WARRANTY_PATH = INPUT_DIR / "warranty.csv"
FEATURE_FREEZE_PATH = INPUT_DIR / "xgb_feature_freeze.xlsx"


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------
OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe.parquet"
SOURCE_STANDARDIZATION_SUMMARY_PATH = OUTPUT_DIR / "source_standardization_summary.csv"


# -----------------------------------------------------------------------------
# Snapshot and target settings
# -----------------------------------------------------------------------------
SNAPSHOT_FREQ_DAYS = 14
HORIZON_DAYS = 45
MIN_SNAPSHOT_DATE = None
MAX_SNAPSHOT_DATE = None

# Optional development limiter for all runs. Leave None for all machines.
MAX_MACHINES = None


# -----------------------------------------------------------------------------
# Mini validation mode
# -----------------------------------------------------------------------------
# Turn this on to build output for only a few machines before spending time on
# the full snapshot dataframe.
MINI_RUN_ENABLED = True

# Used only when MINI_RUN_ENABLED=True and MINI_RUN_SERIALS is empty.
MINI_RUN_MACHINE_COUNT = 2

# Optional exact serials to validate. Example: ["70356", "B11204", "30948"]
MINI_RUN_SERIALS = []

# Separate mini output files so a mini run does not overwrite the full output.
MINI_OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe_mini.parquet"
MINI_VALIDATION_SAMPLE_ROWS_PATH = OUTPUT_DIR / "mini_snapshot_validation_sample_rows.csv"
MINI_VALIDATION_BY_MACHINE_PATH = OUTPUT_DIR / "mini_snapshot_validation_by_machine.csv"


# -----------------------------------------------------------------------------
# Data cleaning/report settings
# -----------------------------------------------------------------------------
WRITE_CLEANING_REPORTS = True


# -----------------------------------------------------------------------------
# Machine population and progress
# -----------------------------------------------------------------------------
TARGET_MODEL_FAMILIES = ("D51", "D61", "D71")
PROGRESS_EVERY_MACHINES = 100


# -----------------------------------------------------------------------------
# Azure ML job submission settings
# -----------------------------------------------------------------------------
# Fill these in before running:
#     python data_preparation/submit_snapshot_build_aml_job.py
AML_SUBSCRIPTION_ID = "7f07baf7-8bba-4b88-b300-74ba5b15f52d"
AML_RESOURCE_GROUP = "ai-servicecontroltower"
AML_WORKSPACE_NAME = "ai-controltower-aml"
AML_COMPUTE_NAME = "tan-dev-gpu-cluster"

# Curated GPU environment. This avoids manually pinning CUDA wheels.
# If you run on CPU compute, you may switch to an AzureML sklearn/pandas curated env.
AML_ENVIRONMENT = "AzureML-acpt-pytorch-2.8-cuda12.6@latest"
AML_EXPERIMENT_NAME = "snapshot-data-preparation"
AML_DISPLAY_NAME = "build-snapshot-dataframe"

# Upload the whole project folder as the AML code snapshot so data_preparation/
# scripts and enriched_data/ are both reachable inside the job.
AML_CODE_DIR = PROJECT_ROOT
BUILD_SNAPSHOT_SCRIPT_PATH = DATA_PREPARATION_DIR / "build_snapshot_dataframe.py"
LOG_DEVICE_SCRIPT_PATH = DATA_PREPARATION_DIR / "log_compute_device.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

AML_INSTALL_REQUIREMENTS = True
AML_STREAM_LOGS = True
AML_REQUIRE_GPU = False

# Local device requirement if you run log_compute_device.py outside Azure ML.
REQUIRE_GPU = False
