"""
Configuration for data_preparation/build_snapshot_dataframe.py.

Current folder structure expected:

    service_controltower/
    ├── data_preparation/
    │   ├── build_snapshot_dataframe.py
    │   ├── config.py
    │   ├── log_compute_device.py
    │   ├── submit_snapshot_build_aml_job.py
    │   └── output/
    ├── enriched_data/
    │   ├── machine.csv          # canonical model_id + snapshot_date backbone
    │   ├── fault_codes.csv
    │   └── maintenance.csv
    └── requirements.txt

Run from service_controltower/:

    python data_preparation/build_snapshot_dataframe.py

or from service_controltower/data_preparation/:

    python build_snapshot_dataframe.py
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Project folders
# -----------------------------------------------------------------------------
# config.py is inside service_controltower/data_preparation/.
# Therefore the project root is two levels up from this file's path:
#   config.py -> data_preparation -> service_controltower
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "enriched_data"
OUTPUT_DIR = PROJECT_ROOT / "data_preparation" / "output"
SOURCE_SNAPSHOT_DIR = OUTPUT_DIR / "source_snapshots"


# -----------------------------------------------------------------------------
# Input files currently available
# -----------------------------------------------------------------------------
# machine.csv is now the official snapshot backbone. It must contain:
#   - model_id or machine_id
#   - snapshot_date
#   - full_model, recommended for D51/D61/D71 filtering
MACHINE_PATH = INPUT_DIR / "machine.csv"
FAULT_CODES_PATH = INPUT_DIR / "fault_codes.csv"
MAINTENANCE_PATH = INPUT_DIR / "maintenance.csv"

# Optional future inputs. The current build script includes placeholders but does
# not yet build features from these files.
WARRANTY_PATH = INPUT_DIR / "warranty.csv"
OIL_SAMPLE_PATH = INPUT_DIR / "oil_samples.csv"
SERVICE_PATH = INPUT_DIR / "service.csv"

# Optional feature-name validation file.
FEATURE_FREEZE_PATH = INPUT_DIR / "xgb_feature_freeze.xlsx"


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------
OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe.csv"
MINI_OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe_mini.csv"

# Save intermediate source-level snapshot tables, useful for QA and monitoring.
# Saved under data_preparation/output/source_snapshots/:
#   - machine_backbone.csv
#   - fault_snapshot.csv
#   - maintenance_snapshot.csv
SAVE_SOURCE_SNAPSHOTS = True


# -----------------------------------------------------------------------------
# Machine-backbone settings
# -----------------------------------------------------------------------------
# All source snapshots are forced to use exactly the model_id + snapshot_date rows
# from machine.csv. They do not generate independent snapshot calendars.
MODEL_ID_CANDIDATE_COLUMNS = (
    "model_id",
    "machine_id",
    "MACHINE_ID",
    "Machine_ID",
)

MACHINE_SNAPSHOT_DATE_CANDIDATE_COLUMNS = (
    "snapshot_date",
    "SNAPSHOT_DATE",
    "as_of_date",
    "AS_OF_DATE",
    "snapshot_dt",
)

# Temporary fallback for older event extracts that do not yet contain model_id or
# machine_id. Production extracts should include a real model_id/machine_id.
ALLOW_MODEL_ID_FALLBACK = True

# The model-family column is not written to the output, but these prefixes are
# still used internally to restrict the machine backbone to target dozer families.
TARGET_MODEL_FAMILIES = ("D51", "D61", "D71")

# Optional date limits for development or time-sliced builds.
# These filters are applied to the machine.csv backbone after it is loaded.
MIN_SNAPSHOT_DATE = None
MAX_SNAPSHOT_DATE = None


# -----------------------------------------------------------------------------
# Mini validation mode
# -----------------------------------------------------------------------------
# Turn this on when you want to build snapshots for only 2 or 3 machines/model_ids
# to inspect the logic before running the full build.
MINI_RUN_ENABLED = True
MINI_RUN_MACHINE_COUNT = 2
MINI_RUN_MODEL_IDS = []

# Backward-compatible limiter. Prefer MINI_RUN_ENABLED for validation.
MAX_MACHINES = None


# -----------------------------------------------------------------------------
# Snapshot and target settings for future extensions
# -----------------------------------------------------------------------------
# The current snapshot dates come from machine.csv, not from this frequency.
# Keep this setting for future rebuilding of the machine backbone if needed.
SNAPSHOT_FREQ_DAYS = 14

# Future warranty target horizon.
HORIZON_DAYS = 45


# -----------------------------------------------------------------------------
# Progress/reporting settings
# -----------------------------------------------------------------------------
PROGRESS_EVERY_MACHINES = 100
WRITE_CLEANING_REPORTS = True


# -----------------------------------------------------------------------------
# Azure ML job submission settings
# -----------------------------------------------------------------------------
# These are used by data_preparation/submit_snapshot_build_aml_job.py.
# Fill them in before submitting to Azure ML.
AML_SUBSCRIPTION_ID = ""
AML_RESOURCE_GROUP = ""
AML_WORKSPACE_NAME = ""
AML_COMPUTE_NAME = ""
AML_ENVIRONMENT = "AzureML-acpt-pytorch-2.8-cuda12.6@latest"
AML_EXPERIMENT_NAME = "snapshot-build"
AML_DISPLAY_NAME = "snapshot-build-machine-backbone"

AML_CODE_DIR = PROJECT_ROOT
BUILD_SNAPSHOT_SCRIPT_PATH = PROJECT_ROOT / "data_preparation" / "build_snapshot_dataframe.py"
LOG_DEVICE_SCRIPT_PATH = PROJECT_ROOT / "data_preparation" / "log_compute_device.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

AML_INSTALL_REQUIREMENTS = True
AML_STREAM_LOGS = True
AML_REQUIRE_GPU = False
REQUIRE_GPU = False
