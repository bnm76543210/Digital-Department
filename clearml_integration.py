from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

from maintenance_core import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TrainingReport,
    model_bundle,
    result_table,
    save_model_bundle,
)


CLEARML_PROJECT = "Predictive Maintenance Advanced"
ARTIFACTS_DIR = Path("artifacts")
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "predictive_maintenance_model.pkl"


class ClearMLUnavailableError(RuntimeError):
    pass


def _import_clearml():
    try:
        import clearml
    except ImportError as exc:
        raise ClearMLUnavailableError(
            "Библиотека clearml не установлена. Выполните: pip install -r requirements.txt"
        ) from exc
    return clearml


def clearml_status() -> dict[str, Any]:
    try:
        _import_clearml()
        package_installed = True
    except ClearMLUnavailableError:
        package_installed = False

    home_config = Path.home() / "clearml.conf"
    env_keys = {
        "CLEARML_API_ACCESS_KEY": bool(os.getenv("CLEARML_API_ACCESS_KEY")),
        "CLEARML_API_SECRET_KEY": bool(os.getenv("CLEARML_API_SECRET_KEY")),
        "CLEARML_API_HOST": bool(os.getenv("CLEARML_API_HOST")),
        "CLEARML_WEB_HOST": bool(os.getenv("CLEARML_WEB_HOST")),
        "CLEARML_FILES_HOST": bool(os.getenv("CLEARML_FILES_HOST")),
    }
    return {
        "package_installed": package_installed,
        "clearml_conf_exists": home_config.exists(),
        "configured_by_env": all(env_keys.values()),
        "env_keys": env_keys,
        "config_path": str(home_config),
    }


def init_clearml_task(project_name: str, task_name: str):
    clearml = _import_clearml()
    return clearml.Task.init(project_name=project_name, task_name=task_name, output_uri=True)


def log_training_report(
    report: TrainingReport,
    *,
    data: pd.DataFrame,
    project_name: str = CLEARML_PROJECT,
    task_name: str = "Streamlit training run",
):
    clearml = _import_clearml()
    task = clearml.Task.init(project_name=project_name, task_name=task_name, output_uri=True)
    task.connect(
        {
            "best_model": report.best_model_name,
            "test_size": report.test_size,
            "random_state": report.random_state,
            "rows": int(data.shape[0]),
            "columns": int(data.shape[1]),
            "classes": report.class_names,
        }
    )

    logger = task.get_logger()
    for result in report.results.values():
        logger.report_scalar("Accuracy", result.name, result.accuracy, iteration=0)
        logger.report_scalar("Weighted F1", result.name, result.weighted_f1, iteration=0)
        logger.report_scalar("Macro F1", result.name, result.macro_f1, iteration=0)
        if result.cv_weighted_f1_mean is not None:
            logger.report_scalar("CV weighted F1", result.name, result.cv_weighted_f1_mean, iteration=0)

    metrics = result_table(report)
    task.upload_artifact("model_metrics", artifact_object=metrics)
    task.upload_artifact("class_names", artifact_object=report.class_names)

    artifact_path = save_model_bundle(report, MODEL_ARTIFACT_PATH)
    output_model = clearml.OutputModel(task=task, framework="Scikit-learn")
    output_model.update_weights(weights_filename=str(artifact_path))
    task.close()
    return {
        "task_id": task.id,
        "model_id": output_model.id,
        "artifact_path": str(artifact_path),
    }


def create_clearml_dataset(
    data_path: Path | str = DATA_PATH,
    *,
    dataset_name: str = "AI4I 2020 Predictive Maintenance",
    dataset_project: str = CLEARML_PROJECT,
) -> str:
    clearml = _import_clearml()
    dataset = clearml.Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
    )
    dataset.add_files(str(data_path))
    dataset.upload()
    dataset.finalize()
    return dataset.id


def _select_downloaded_model_file(downloaded_path: str | Path) -> Path:
    path = Path(downloaded_path)
    if path.is_file():
        return path
    candidates = list(path.rglob("*.pkl")) + list(path.rglob("*.joblib"))
    if not candidates:
        raise FileNotFoundError("В скачанной модели ClearML не найден .pkl или .joblib файл.")
    return candidates[0]


def load_model_bundle_from_clearml(model_id: str) -> dict[str, object]:
    clearml = _import_clearml()
    model = clearml.Model(model_id=model_id)

    if hasattr(model, "get_local_copy"):
        downloaded = model.get_local_copy()
    elif hasattr(model, "download_model_weights"):
        downloaded = model.download_model_weights()
    elif hasattr(model, "download"):
        downloaded = model.download()
    else:
        raise AttributeError("У установленной версии clearml не найден метод скачивания модели.")

    model_file = _select_downloaded_model_file(downloaded)
    return joblib.load(model_file)


def call_clearml_serving(endpoint_url: str, values: dict[str, object], timeout: int = 20) -> dict[str, Any]:
    payload = {column: values[column] for column in FEATURE_COLUMNS}
    response = requests.post(endpoint_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
