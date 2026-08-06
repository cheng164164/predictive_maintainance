"""Foundry dataset and model paths to replace during Palantir integration.

Only this file should normally require path edits when the integration code is
copied into a Palantir Code Repository. Replace every placeholder with the
corresponding Foundry dataset path or resource identifier.
"""

# Source-history datasets used to build one current feature snapshot.
FAULT_DATASET = "/path/to/fault-history-dataset"
FLUID_DATASET = "/path/to/fluid-history-dataset"
MAINTENANCE_DATASET = "/path/to/maintenance-history-dataset"
OPERATION_DATASET = "/path/to/operation-history-dataset"
TARGET_HISTORY_DATASET = "/path/to/target-history-dataset"
MACHINE_ROSTER_DATASET = "/path/to/complete-machine-roster-dataset"

# Intermediate and final tabular datasets.
PREPARED_FEATURES_DATASET = "/path/to/prepared-machine-feature-dataset"
MODEL_PREDICTIONS_DATASET = "/path/to/model-predictions-dataset"
UI_RISK_OUTPUT_DATASET = "/path/to/ui-machine-risk-output-dataset"

# Unstructured dataset containing the locally trained model artifacts and the
# Palantir model asset that receives published model versions.
MODEL_FILES_DATASET = "/path/to/uploaded-xgboost-model-files-dataset"
MODEL_ASSET = "/path/to/published-machine-risk-model"
MODEL_VERSION = None  # Set a model-version RID or semantic version to pin inference.

# Filenames expected inside MODEL_FILES_DATASET. Upload these three files from
# rolling_origin_model/models/<target_source>/ after final training/promotion.
MODEL_FILE_NAME = "xgboost_base27_plus_history.json"
MODEL_METADATA_FILE_NAME = "model_metadata_base27_plus_history.json"
TIER_POLICY_FILE_NAME = "tier_policy_base27_plus_history.json"

# None uses the latest operation date as the scoring cutoff. Set an ISO date for
# a controlled historical or scheduled batch scoring run.
PALANTIR_SCORE_DATE = None
