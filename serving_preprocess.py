from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import sys
import types
from sklearn.base import BaseEstimator, TransformerMixin


FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


class BundlePredictor:
    def __init__(self, bundle):
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.class_names = list(bundle["class_names"])

    def predict(self, data):
        prediction_code = self.model.predict(data)
        prediction = self.label_encoder.inverse_transform(prediction_code)[0]
        response = {"prediction": prediction}
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(data)[0]
            response["probabilities"] = {
                class_name: float(probability)
                for class_name, probability in zip(self.class_names, probabilities)
            }
        return response


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.nanquantile(values, 0.25, axis=0)
        q3 = np.nanquantile(values, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def _register_pickle_compatibility_module():
    compatibility_module = types.ModuleType("maintenance_core")
    compatibility_module.IQRClipper = IQRClipper
    sys.modules.setdefault("maintenance_core", compatibility_module)


class Preprocess:
    def load(self, local_model_file):
        _register_pickle_compatibility_module()
        return BundlePredictor(joblib.load(local_model_file))

    def preprocess(self, request, state, collect_custom_statistics_fn=None):
        payload = request.get("data", request)
        row = {column: payload[column] for column in FEATURE_COLUMNS}
        if collect_custom_statistics_fn:
            collect_custom_statistics_fn({"request_type": row["Type"]})
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        return data
