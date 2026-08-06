"""Run the published machine-risk model against prepared Foundry features."""
from __future__ import annotations

from transforms.api import Input, LightweightInput, LightweightOutput, Output, transform
from palantir_models import ModelAdapter
from palantir_models.transforms import ModelInput

from .foundry_paths import (
    MODEL_ASSET,
    MODEL_PREDICTIONS_DATASET,
    MODEL_VERSION,
    PREPARED_FEATURES_DATASET,
)


@transform.using(
    data_in=Input(PREPARED_FEATURES_DATASET),
    model_input=ModelInput(
        MODEL_ASSET,
        model_version=MODEL_VERSION,
        use_sidecar=True,
    ),
    out=Output(MODEL_PREDICTIONS_DATASET),
)
def compute(
    data_in: LightweightInput,
    model_input: ModelAdapter,
    out: LightweightOutput,
) -> None:
    """Invoke the serialized adapter and write its full scored dataframe."""
    inference_results = model_input.transform(data_in)
    out.write_pandas(inference_results.df_out)
