"""Classifier-specific XGBoost serializer for Palantir model adapters.

Palantir's serialization API supports custom serializers. This implementation
uses XGBoost's stable JSON ``save_model``/``load_model`` interface but restores
an ``XGBClassifier`` explicitly, preserving ``predict_proba`` after model
publication and deserialization.
"""
from __future__ import annotations

from palantir_models import ModelSerializer
from palantir_models.models._serialization import ModelStateReader, ModelStateWriter
from xgboost import XGBClassifier


class XGBClassifierSerializer(ModelSerializer[XGBClassifier]):
    """Serialize and restore an ``XGBClassifier`` as an XGBoost JSON model."""

    file_name = "xgboost_classifier.json"

    def serialize(
        self,
        writer: ModelStateWriter,
        model: XGBClassifier,
    ) -> None:
        """Save the fitted classifier to Palantir-managed model storage."""
        with writer.open(self.file_name, "w") as model_file:
            model.save_model(model_file.name)

    def deserialize(self, reader: ModelStateReader) -> XGBClassifier:
        """Load the stored JSON model into the correct sklearn classifier type."""
        model = XGBClassifier()
        with reader.open(self.file_name, "r") as model_file:
            model.load_model(model_file.name)
        return model
