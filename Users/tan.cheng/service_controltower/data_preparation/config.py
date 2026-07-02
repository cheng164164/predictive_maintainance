"""
Configuration for build_snapshot_dataframe.py.

Edit this file instead of typing long command-line arguments.
Keep this file in the same folder as build_snapshot_dataframe.py, then run:

    python build_snapshot_dataframe.py

Folder assumptions requested for this project:
    - Input CSV files are in: enriched_data/
    - Generated outputs are saved in: data_preparation/output/
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Project folders
# -----------------------------------------------------------------------------
# Paths are relative to the folder that contains this config.py file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw/enriched input data folder.
INPUT_DIR = PROJECT_ROOT / "enriched_data"

# Output folder for the snapshot dataframe and missing-value reports.
OUTPUT_DIR = PROJECT_ROOT / "data_preparation" / "output"


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
# Required files.
FAULT_CODES_PATH = INPUT_DIR / "fault_codes.csv"
MACHINE_PATH = INPUT_DIR / "machine.csv"
MAINTENANCE_PATH = INPUT_DIR / "maintenance.csv"

# Optional files.
# If WARRANTY_PATH does not exist, the script still runs but claim_next_45d is blank.
WARRANTY_PATH = INPUT_DIR / "warranty.csv"

# Optional validation file. The script already contains the frozen feature list,
# but this file is used as an extra check when it exists.
FEATURE_FREEZE_PATH = INPUT_DIR / "xgb_feature_freeze.xlsx"


# -----------------------------------------------------------------------------
# Output file
# -----------------------------------------------------------------------------
# Use .parquet for the main pipeline. If your environment does not have a parquet
# engine installed, the script automatically falls back to .csv.
OUTPUT_PATH = OUTPUT_DIR / "snapshot_dataframe.parquet"


# -----------------------------------------------------------------------------
# Snapshot and target settings
# -----------------------------------------------------------------------------
# One snapshot every 14 days per machine.
SNAPSHOT_FREQ_DAYS = 14

# Label = 1 when a warranty failure happens within the next 45 days.
HORIZON_DAYS = 45

# Limit snapshots by date when you want a smaller development run.
# Use strings like "2025-01-01", or leave as None for no limit.
MIN_SNAPSHOT_DATE = None
MAX_SNAPSHOT_DATE = None

# Development limiter. Example: set MAX_MACHINES = 50 for a quick test.
# Leave as None for all machines.
MAX_MACHINES = None


# -----------------------------------------------------------------------------
# Data cleaning/report settings
# -----------------------------------------------------------------------------
# When True, the script writes:
#   - missing_profile_fault_codes.csv
#   - missing_profile_machine.csv
#   - missing_profile_maintenance.csv
#   - missing_profile_warranty.csv, if warranty data exists
#   - missing_profile_all_files.csv
#   - cleaning_summary.csv
WRITE_CLEANING_REPORTS = True


# -----------------------------------------------------------------------------
# Machine population
# -----------------------------------------------------------------------------
# The snapshot universe is restricted to these model families.
TARGET_MODEL_FAMILIES = ("D51", "D61", "D71")
