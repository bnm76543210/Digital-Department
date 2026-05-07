import unittest

import pandas as pd

from maintenance_core import (
    DATA_PATH,
    build_failure_target,
    feature_frame,
    model_bundle,
    predict_failure_type,
    predict_with_model_bundle,
    read_dataset,
    train_models,
)


class MaintenanceCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_dataset(DATA_PATH)

    def test_failure_target_marks_single_and_multiple_failures(self):
        rows = pd.DataFrame(
            [
                {
                    "UDI": 1,
                    "Product ID": "L1",
                    "Type": "L",
                    "Air temperature [K]": 300.0,
                    "Process temperature [K]": 310.0,
                    "Rotational speed [rpm]": 1500,
                    "Torque [Nm]": 40.0,
                    "Tool wear [min]": 10,
                    "Machine failure": 0,
                    "TWF": 0,
                    "HDF": 0,
                    "PWF": 0,
                    "OSF": 0,
                    "RNF": 0,
                },
                {
                    "UDI": 2,
                    "Product ID": "M1",
                    "Type": "M",
                    "Air temperature [K]": 302.0,
                    "Process temperature [K]": 312.0,
                    "Rotational speed [rpm]": 1200,
                    "Torque [Nm]": 55.0,
                    "Tool wear [min]": 190,
                    "Machine failure": 1,
                    "TWF": 0,
                    "HDF": 1,
                    "PWF": 1,
                    "OSF": 0,
                    "RNF": 0,
                },
            ]
        )

        labels = build_failure_target(rows).tolist()

        self.assertEqual(labels, ["No failure", "Multiple failures"])

    def test_feature_frame_keeps_only_model_inputs(self):
        features = feature_frame(self.data)

        self.assertEqual(features.shape[1], 6)
        self.assertIn("Type", features.columns)
        self.assertNotIn("Machine failure", features.columns)

    def test_training_report_and_prediction_are_created(self):
        labels = build_failure_target(self.data)
        sample = (
            self.data.assign(_target=labels)
            .groupby("_target", group_keys=False)
            .head(8)
            .drop(columns="_target")
        )

        report = train_models(
            sample,
            model_names=["Logistic Regression", "Random Forest"],
            test_size=0.25,
            cv_splits=2,
        )
        prediction, probabilities = predict_failure_type(
            report,
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 120,
            },
        )

        self.assertIn(report.best_model_name, report.results)
        self.assertIn(prediction, report.class_names)
        self.assertIsNotNone(probabilities)

        bundle_prediction, bundle_probabilities = predict_with_model_bundle(
            model_bundle(report),
            {
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 120,
            },
        )

        self.assertIn(bundle_prediction, report.class_names)
        self.assertIsNotNone(bundle_probabilities)


if __name__ == "__main__":
    unittest.main()
