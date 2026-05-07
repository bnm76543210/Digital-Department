from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


DATA_PATH = Path("data") / "ai4i2020.csv"

ID_COLUMNS = ["UDI", "Product ID"]
FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COLUMN = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] + FAILURE_COLUMNS


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    accuracy: float
    weighted_f1: float
    macro_f1: float
    confusion: np.ndarray
    report: pd.DataFrame
    cv_weighted_f1_mean: float | None = None
    cv_weighted_f1_std: float | None = None
    best_params: dict | None = None


@dataclass
class TrainingReport:
    results: dict[str, ModelResult]
    best_model_name: str
    label_encoder: LabelEncoder
    class_names: list[str]
    test_size: float
    random_state: int

    @property
    def best_result(self) -> ModelResult:
        return self.results[self.best_model_name]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features by fitted IQR limits before scaling."""

    def __init__(self, factor: float = 1.5):
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


def read_dataset(source=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(source)


def validate_dataset(data: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"В датасете отсутствуют обязательные столбцы: {joined}")


def build_failure_target(data: pd.DataFrame) -> pd.Series:
    validate_dataset(data)

    def row_to_label(row: pd.Series) -> str:
        active_failures = [column for column in FAILURE_COLUMNS if int(row[column]) == 1]
        if not active_failures:
            return "No failure"
        if len(active_failures) == 1:
            return active_failures[0]
        return "Multiple failures"

    return data.apply(row_to_label, axis=1).rename("Failure type")


def build_binary_target(data: pd.DataFrame) -> pd.Series:
    validate_dataset(data)
    return data[TARGET_COLUMN].astype(int)


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    validate_dataset(data)
    return data[FEATURE_COLUMNS].copy()


def dataset_summary(data: pd.DataFrame) -> dict[str, object]:
    target = build_failure_target(data)
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_values": data.isna().sum(),
        "failure_type_counts": target.value_counts(),
        "product_type_counts": data["Type"].value_counts(),
        "numeric_description": data[NUMERIC_FEATURES].describe().T,
        "correlation": data[NUMERIC_FEATURES + [TARGET_COLUMN]].corr(numeric_only=True),
    }


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", IQRClipper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models(random_state: int = 42) -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=240,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _safe_cv_splits(y: np.ndarray, requested_splits: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    return max(0, min(requested_splits, int(counts.min())))


def _make_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def _evaluate_model(
    name: str,
    estimator: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    cv_scores: np.ndarray | None = None,
    best_params: dict | None = None,
) -> ModelResult:
    predictions = estimator.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result = ModelResult(
        name=name,
        estimator=estimator,
        accuracy=accuracy_score(y_test, predictions),
        weighted_f1=f1_score(y_test, predictions, average="weighted", zero_division=0),
        macro_f1=f1_score(y_test, predictions, average="macro", zero_division=0),
        confusion=confusion_matrix(y_test, predictions, labels=np.arange(len(class_names))),
        report=pd.DataFrame(report).T,
        best_params=best_params,
    )
    if cv_scores is not None and len(cv_scores):
        result.cv_weighted_f1_mean = float(np.mean(cv_scores))
        result.cv_weighted_f1_std = float(np.std(cv_scores))
    return result


def train_models(
    data: pd.DataFrame,
    *,
    model_names: Iterable[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 3,
    optimize_random_forest: bool = False,
) -> TrainingReport:
    X = feature_frame(data)
    labels = build_failure_target(data)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    class_names = encoder.classes_.tolist()
    if len(class_names) < 2:
        raise ValueError("Для обучения нужны хотя бы два класса целевой переменной.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    models = candidate_models(random_state=random_state)
    if model_names is not None:
        requested = list(model_names)
        models = {name: models[name] for name in requested if name in models}
    if not models:
        raise ValueError("Не выбрана ни одна поддерживаемая модель.")

    results: dict[str, ModelResult] = {}
    actual_cv_splits = _safe_cv_splits(y_train, cv_splits)

    for name, model in models.items():
        pipeline = _make_pipeline(model)
        cv_scores = None
        if actual_cv_splits >= 2:
            cv = StratifiedKFold(
                n_splits=actual_cv_splits,
                shuffle=True,
                random_state=random_state,
            )
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=None,
            )
        pipeline.fit(X_train, y_train)
        results[name] = _evaluate_model(name, pipeline, X_test, y_test, class_names, cv_scores)

    if optimize_random_forest and actual_cv_splits >= 2:
        tuned_pipeline = _make_pipeline(
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            )
        )
        param_grid = {
            "model__n_estimators": [180, 260],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_leaf": [1, 3],
        }
        cv_for_search = max(2, actual_cv_splits) if actual_cv_splits else 2
        search = GridSearchCV(
            tuned_pipeline,
            param_grid=param_grid,
            cv=cv_for_search,
            scoring="f1_weighted",
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuned_name = "Random Forest Optimized"
        results[tuned_name] = _evaluate_model(
            tuned_name,
            search.best_estimator_,
            X_test,
            y_test,
            class_names,
            np.array([search.best_score_]),
            best_params=search.best_params_,
        )

    best_model_name = max(results, key=lambda key: results[key].weighted_f1)
    return TrainingReport(
        results=results,
        best_model_name=best_model_name,
        label_encoder=encoder,
        class_names=class_names,
        test_size=test_size,
        random_state=random_state,
    )


def result_table(report: TrainingReport) -> pd.DataFrame:
    rows = []
    for result in report.results.values():
        rows.append(
            {
                "model": result.name,
                "accuracy": result.accuracy,
                "weighted_f1": result.weighted_f1,
                "macro_f1": result.macro_f1,
                "cv_weighted_f1": result.cv_weighted_f1_mean,
                "cv_std": result.cv_weighted_f1_std,
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)


def predict_failure_type(
    report: TrainingReport,
    values: dict[str, object],
) -> tuple[str, pd.Series | None]:
    bundle = model_bundle(report)
    return predict_with_model_bundle(bundle, values)


def model_bundle(report: TrainingReport) -> dict[str, object]:
    return {
        "model": report.best_result.estimator,
        "label_encoder": report.label_encoder,
        "class_names": report.class_names,
        "feature_columns": FEATURE_COLUMNS,
        "best_model_name": report.best_model_name,
    }


def save_model_bundle(report: TrainingReport, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle(report), path)
    return path


def load_model_bundle(path: Path | str) -> dict[str, object]:
    return joblib.load(path)


def predict_with_model_bundle(
    bundle: dict[str, object],
    values: dict[str, object],
) -> tuple[str, pd.Series | None]:
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    input_frame = pd.DataFrame([values], columns=feature_columns)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    class_names = list(bundle["class_names"])

    prediction_code = model.predict(input_frame)
    prediction = label_encoder.inverse_transform(prediction_code)[0]

    if not hasattr(model, "predict_proba"):
        return prediction, None

    probabilities = model.predict_proba(input_frame)[0]
    probability_series = pd.Series(probabilities, index=class_names, name="probability")
    return prediction, probability_series.sort_values(ascending=False)
