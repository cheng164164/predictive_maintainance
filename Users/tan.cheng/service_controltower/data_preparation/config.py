"""
Configuration for service_controltower/data_preparation/build_snapshot_dataframe.py
and data_preparation/submit_snapshot_build_aml_job.py.

Current project layout:

    service_controltower/
    ├── data_preparation/
    │   ├── build_snapshot_dataframe.py
    │   ├── config.py
    │   ├── log_compute_device.py
    │   ├── submit_snapshot_build_aml_job.py
    │   ├── download_aml_run_results.py
    │   └── output/
    ├── enriched_data/        # local only; not uploaded with AML job
    └── requirements.txt

For AML jobs:
    - code uses the project root as AML code folder
    - .amlignore excludes .venv, enriched_data, outputs, and old results
    - no .aml_job_source staging folder is created
    - input CSVs are read from Blob/Azure ML data input
    - run outputs are written to job-name folders in the AML datastore
"""

from __future__ import annotations

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Project folders
# -----------------------------------------------------------------------------
# config.py is expected to be inside service_controltower/data_preparation/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# In AML jobs, submit_snapshot_build_aml_job.py injects AML_INPUT_DIR and
# AML_OUTPUT_DIR as mounted/downloaded paths. Locally, these default to folders
# inside the repo.
INPUT_DIR = Path(os.environ.get("AML_INPUT_DIR", PROJECT_ROOT / "enriched_data"))
OUTPUT_DIR = Path(os.environ.get("AML_OUTPUT_DIR", PROJECT_ROOT / "data_preparation" / "output"))
SOURCE_SNAPSHOT_DIR = Path(
    os.environ.get("AML_SOURCE_SNAPSHOT_DIR", OUTPUT_DIR / "source_snapshots")
)

PROGRESS_LOG_PATH = Path(
    os.environ.get("AML_PROGRESS_LOG_PATH", OUTPUT_DIR / "snapshot_build_progress_log.csv")
)
ARTIFACT_MANIFEST_PATH = Path(
    os.environ.get("AML_ARTIFACT_MANIFEST_PATH", OUTPUT_DIR / "snapshot_build_artifact_manifest.csv")
)


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
# machine.csv supplies eligible model_ids, original date bounds, and metadata.
# The builder reconstructs an exact configurable calendar from those bounds.
MACHINE_PATH = INPUT_DIR / "machine.csv"
FAULT_CODES_PATH = INPUT_DIR / "fault_codes.csv"
MAINTENANCE_PATH = INPUT_DIR / "maintenance.csv"
OPERATION_PATH = INPUT_DIR / "operation.csv"
FLUID_SAMPLES_PATH = INPUT_DIR / "fluid_samples.csv"
WARRANTY_PATH = INPUT_DIR / "warranty.csv"

# Updated feature freeze file. Keep both names supported for local experiments.
FEATURE_FREEZE_PATH = INPUT_DIR / "xgb_feature_freeze(all).csv"
FEATURE_FREEZE_FALLBACK_PATH = INPUT_DIR / "xgb_feature_freeze.xlsx"


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------
OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe.csv"
MINI_OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe_mini.csv"
SAVE_SOURCE_SNAPSHOTS = True


# -----------------------------------------------------------------------------
# Machine-backbone and feature settings
# -----------------------------------------------------------------------------
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

ALLOW_MODEL_ID_FALLBACK = True
TARGET_MODEL_FAMILIES = ("D51", "D61", "D71")
MIN_SNAPSHOT_DATE = None
MAX_SNAPSHOT_DATE = None

# Snapshot modeling design.
# Features use [snapshot_date - LOOKBACK_DAYS, snapshot_date).
# Labels use [snapshot_date, snapshot_date + HORIZON_DAYS).
LOOKBACK_DAYS = 90
HORIZON_DAYS = 90
SNAPSHOT_FREQ_DAYS = 45
APPLY_SNAPSHOT_FREQUENCY = True

# How to convert the dates supplied by machine.csv into the modeling calendar.
#   reconstruct    -> generate an exact N-day calendar between the original
#                     backbone start and end dates; recommended when machine.csv
#                     was built with a different cadence such as every 14 days.
#   select_existing -> legacy behavior that keeps only dates already present in
#                      machine.csv. This cannot produce an exact 45-day cadence
#                      when the source dates occur every 14 days.
SNAPSHOT_FREQUENCY_STRATEGY = "reconstruct"
SNAPSHOT_FREQUENCY_SCOPE = "global"  # options: "global", "per_model"

# None anchors the reconstructed calendar at the first eligible machine.csv
# date. Set an explicit date only when all experiments must share a fixed anchor.
SNAPSHOT_ANCHOR_DATE = None

# Drop snapshots whose complete future label window is not observable.
# When LABEL_OBSERVATION_END_DATE is None, the builder infers the latest date
# available across the standardized input sources and machine backbone.
REQUIRE_COMPLETE_LABEL_HORIZON = True
LABEL_OBSERVATION_END_DATE = None

# Select which feature set is returned by the unified modeling dataframe.
#   basic  -> simple 90-day window aggregations below
#   frozen -> the existing engineered FROZEN_FEATURES list
FEATURE_MODE = "frozen"

BASE_NUMERIC_FEATURES = [
    # Prior warranty context before the observation window
    "prior_claim_count_before_window",
    "days_since_prior_claim_before_window",

    # Source availability / coverage flags
    "has_any_source_window",
    "source_record_count_window",

    # Fault-code signals inside the observation window
    "has_fault_window",
    "fault_count_window",
    "fault_unique_code_count_window",
    "fault_l03plus_count_window",
    "fault_l04plus_count_window",
    "fault_max_action_level_window",
    "fault_max_evidence_score_window",
    "fault_mean_evidence_score_window",
    "fault_max_log_occurrence_window",
    "fault_days_since_latest_in_window",
    "fault_mechanical_count_window",
    "fault_electrical_count_window",

    # Fluid-sample signals inside the observation window
    "has_fluid_window",
    "fluid_sample_count_window",
    "fluid_max_severity_window",
    "fluid_latest_severity_window",
    "fluid_days_since_latest_sample_window",
    "fluid_max_cu_ppm_window",
    "fluid_max_fe_ppm_window",
    "fluid_max_pb_ppm_window",
    "fluid_max_soot_percent_window",
    "fluid_max_water_percent_window",

    # Maintenance signals inside the observation window
    "has_maintenance_window",
    "maintenance_event_count_window",
    "maintenance_monitor_reset_count_window",
    "maintenance_overdue_count_window",
    "maintenance_due_now_count_window",
    "maintenance_min_remaining_hours_window",
    "maintenance_days_since_latest_event_window",

    # Operation / usage signals inside the observation window
    "has_operation_window",
    "operation_day_count_window",
    "operation_working_hours_sum_window",
    "operation_working_hours_mean_window",
    "operation_working_hours_max_window",
    "operation_engine_running_hours_sum_window",
    "operation_idle_hours_sum_window",
    "operation_idle_share_window",
    "operation_latest_smr_window",
    "operation_smr_delta_window",
    "operation_high_throttle_day_count_window",
]

BASE_CATEGORICAL_FEATURES = [
    "full_model",
    "fault_dominant_component_window",
    "maintenance_dominant_component_window",
]

# Frozen mode keeps the existing 365-day fluid feature behavior.
FLUID_SAMPLE_LOOKBACK_DAYS = 365
WRITE_CLEANING_REPORTS = True
PROGRESS_EVERY_MACHINES = 100

# When False, the unified output contains only identifiers, the dynamic target,
# and the selected basic/frozen modeling features. Source snapshot CSVs still
# retain their source-level QA columns.
INCLUDE_QA_HELPER_COLUMNS = False


# -----------------------------------------------------------------------------
# Local mini-run settings
# -----------------------------------------------------------------------------
MINI_RUN_ENABLED = False
MINI_RUN_MACHINE_COUNT = 2
MINI_RUN_MODEL_IDS = []
MAX_MACHINES = None


# -----------------------------------------------------------------------------
# Azure ML workspace/job settings
# -----------------------------------------------------------------------------
# Choose which AML workspace/compute pair to use.
#   cpu -> ai-controltower-aml / tan-dev-cpu-cluster
#   gpu -> ehs-safety-aml / tan-dev-gpu
AML_COMPUTE_TARGET = "cpu"  # options: "cpu", "gpu"

# Optional global/legacy overrides. Leave these blank when using separate
# CPU/GPU workspace settings below.
AML_SUBSCRIPTION_ID = ""
AML_RESOURCE_GROUP = ""
AML_WORKSPACE_NAME = ""
AML_COMPUTE_NAME = ""
AML_ENVIRONMENT = ""

# CPU AML workspace and compute.
AML_CPU_SUBSCRIPTION_ID = "7f07baf7-8bba-4b88-b300-74ba5b15f52d"
AML_CPU_RESOURCE_GROUP = "ai-servicecontroltower"
AML_CPU_WORKSPACE_NAME = "ai-controltower-aml"
AML_CPU_COMPUTE_NAME = "tan-dev-cpu-cluster"
AML_CPU_ENVIRONMENT = "AzureML-acpt-pytorch-2.8-cuda12.6@latest"

# GPU AML workspace and compute.
# This is the older GPU cluster shown in the ehs-safety-aml workspace.
AML_GPU_SUBSCRIPTION_ID = "7f07baf7-8bba-4b88-b300-74ba5b15f52d"
AML_GPU_RESOURCE_GROUP = "EHS-Safety"
AML_GPU_WORKSPACE_NAME = "ehs-safety-aml"
AML_GPU_COMPUTE_NAME = "tan-dev-gpu"
AML_GPU_ENVIRONMENT = "AzureML-acpt-pytorch-2.8-cuda12.6@latest"

AML_EXPERIMENT_NAME = "snapshot-build"
AML_DISPLAY_NAME = "snapshot-build-machine-backbone"

# AML code folder. We intentionally use the project root directly and rely on
# .amlignore to prevent large folders from being uploaded. The submit script does
# not create a .aml_job_source folder.
AML_CODE_DIR = PROJECT_ROOT
AML_INSTALL_REQUIREMENTS = True
AML_STREAM_LOGS = True

# Authentication used by submit/download helper scripts.
# Recommended from Azure ML remote terminal:
#   az login --use-device-code
#   az account set --subscription 7f07baf7-8bba-4b88-b300-74ba5b15f52d
# Options: "azure_cli", "managed", "interactive", "default".
AML_AUTH_MODE = "azure_cli"

# GPU validation is automatic. Use AML_COMPUTE_TARGET="gpu" for GPU validation
# and AML_COMPUTE_TARGET="cpu" for a normal CPU run.

# AML mini-run settings are separate from local mini-run settings. These values
# are sent to the remote job as environment variables.
AML_MINI_RUN_ENABLED = False
AML_MINI_RUN_MACHINE_COUNT = 2
AML_MINI_RUN_MODEL_IDS = []


# -----------------------------------------------------------------------------
# Azure ML data input/output settings - Option B
# -----------------------------------------------------------------------------
# Input data is already in Blob Storage. It is NOT uploaded as part of the code
# package. The container should contain:
#   machine.csv, fault_codes.csv, maintenance.csv, operation.csv,
#   fluid_samples.csv, warranty.csv, xgb_feature_freeze(all).csv
AML_INPUT_DATA_URI = "wasbs://enriched-data@aicontroltower7969986141.blob.core.windows.net/"
AML_INPUT_MODE = "download"  # options: download, ro_mount

# Outputs are written under a job-name folder in the selected AML workspace's
# workspaceblobstore. CPU and GPU can have separate workspaces, so these are
# target-specific even though the relative path is the same.
AML_CPU_OUTPUT_BASE_DATA_URI = "azureml://datastores/workspaceblobstore/paths/service_controltower/snapshot_build_outputs/"
AML_GPU_OUTPUT_BASE_DATA_URI = "azureml://datastores/workspaceblobstore/paths/service_controltower/snapshot_build_outputs/"

# Optional global/backward-compatible output base. Leave blank when using the
# target-specific output bases above.
AML_OUTPUT_BASE_DATA_URI = ""
AML_OUTPUT_DATA_URI = ""  # backward-compatible alias only
AML_OUTPUT_MODE = "rw_mount"

# Data access identity for private Blob Storage.
# Options: "user", "managed", "none".
AML_DATA_IDENTITY = "user"

# Historical output settings.
AML_JOB_NAME_PREFIX = "snapshot-build"
AML_INCLUDE_COMPUTE_TARGET_IN_JOB_NAME = True
AML_JOB_NAME = ""  # leave blank to auto-generate a timestamped job name

# submit_snapshot_build_aml_job.py writes this file locally after submission.
# download_aml_run_results.py uses it when AML_DOWNLOAD_JOB_NAME is blank.
AML_LAST_JOB_INFO_PATH = PROJECT_ROOT / "data_preparation" / "aml_last_submitted_job.json"

# Local folder used by download_aml_run_results.py.
AML_RUN_RESULTS_LOCAL_DIR = PROJECT_ROOT / "data_preparation" / "aml_run_results"

# Optional: set this to download a specific historical run.
# Leave blank to download the most recently submitted job recorded in AML_LAST_JOB_INFO_PATH.
AML_DOWNLOAD_JOB_NAME = ""

# Optional target override for downloading a specific historical run.
# Leave blank to infer from the job name, or use the current AML_COMPUTE_TARGET.
AML_DOWNLOAD_COMPUTE_TARGET = ""  # options: "", "cpu", "gpu"
