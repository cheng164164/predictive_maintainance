"""Palantir ModelAdapter for calibrated XGBoost machine-risk inference."""
from __future__ import annotations

from typing import Mapping

import pandas as pd
import palantir_models as pm
from palantir_models.serializers import JsonSerializer
from xgboost import XGBClassifier

from .scoring_core import score_feature_dataframe
from .xgboost_classifier_serializer import XGBClassifierSerializer


class MachineRiskXGBoostAdapter(pm.ModelAdapter):
    """Serialize an XGBoost model together with calibration and tier settings."""

    @pm.auto_serialize(
        model=XGBClassifierSerializer(),
        settings=JsonSerializer(),
    )
    def __init__(self, model: XGBClassifier, settings: Mapping[str, object]):
        """Store the fitted model and its immutable production-scoring settings."""
        self.model = model
        self.settings = dict(settings)

    @classmethod
    def api(cls):
        """Declare one pandas feature input and one pandas scored output."""
        inputs = {"df_in": pm.Pandas()}
        outputs = {"df_out": pm.Pandas()}
        return inputs, outputs

    def predict(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Return raw scores, calibrated probabilities, risk indexes, and tiers."""
        return score_feature_dataframe(self.model, self.settings, df_in)
