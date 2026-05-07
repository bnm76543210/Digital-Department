from __future__ import annotations

import pandas as pd
import streamlit as st

from clearml_integration import (
    CLEARML_PROJECT,
    call_clearml_serving,
    clearml_status,
    create_clearml_dataset,
    load_model_bundle_from_clearml,
    log_training_report,
)
from maintenance_core import DATA_PATH, FEATURE_COLUMNS, predict_with_model_bundle, read_dataset


def _input_values(prefix: str) -> dict[str, object]:
    product_type = st.selectbox("Тип продукта", ["L", "M", "H"], key=f"{prefix}_type")
    air_temp = st.number_input("Температура воздуха [K]", value=300.0, step=0.1, key=f"{prefix}_air")
    process_temp = st.number_input("Температура процесса [K]", value=310.0, step=0.1, key=f"{prefix}_process")
    rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500, step=10, key=f"{prefix}_speed")
    torque = st.number_input("Крутящий момент [Nm]", value=40.0, step=0.1, key=f"{prefix}_torque")
    tool_wear = st.number_input("Износ инструмента [min]", value=120, step=1, key=f"{prefix}_wear")
    return {
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }


def _show_status() -> None:
    status = clearml_status()
    col_package, col_conf, col_env = st.columns(3)
    col_package.metric("Пакет clearml", "есть" if status["package_installed"] else "нет")
    col_conf.metric("clearml.conf", "есть" if status["clearml_conf_exists"] else "нет")
    col_env.metric("ENV-настройка", "полная" if status["configured_by_env"] else "неполная")

    env_rows = pd.DataFrame(
        [{"variable": key, "configured": value} for key, value in status["env_keys"].items()]
    )
    st.dataframe(env_rows, width="stretch")


def _experiment_logging_block() -> None:
    st.subheader("Логирование эксперимента")
    report = st.session_state.get("training_report")
    if report is None:
        st.info("Сначала обучите модели на странице `Модель и прогноз`.")
        return

    project_name = st.text_input("ClearML project", value=CLEARML_PROJECT, key="clearml_project")
    task_name = st.text_input("Task name", value="Streamlit training run", key="clearml_task")
    if st.button("Отправить метрики и модель в ClearML", type="primary"):
        data = read_dataset(DATA_PATH)
        try:
            result = log_training_report(
                report,
                data=data,
                project_name=project_name,
                task_name=task_name,
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state.clearml_last_task_id = result.get("task_id", "")
        st.session_state.clearml_last_model_id = result.get("model_id", "")
        st.success("Эксперимент отправлен в ClearML. Model ID автоматически подставлен ниже.")
        st.json(result)


def _dataset_block() -> None:
    st.subheader("ClearML Dataset")
    dataset_name = st.text_input(
        "Dataset name",
        value="AI4I 2020 Predictive Maintenance",
        key="clearml_dataset_name",
    )
    dataset_project = st.text_input("Dataset project", value=CLEARML_PROJECT, key="clearml_dataset_project")
    if st.button("Создать версию датасета"):
        try:
            dataset_id = create_clearml_dataset(
                DATA_PATH,
                dataset_name=dataset_name,
                dataset_project=dataset_project,
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state.clearml_last_dataset_id = dataset_id
        st.success(f"Создан ClearML Dataset: {dataset_id}")
        st.caption("Это Dataset ID. Его не нужно вставлять в поле Model ID.")


def _model_download_block() -> None:
    st.subheader("Загрузка модели из ClearML")
    last_model_id = st.session_state.get("clearml_last_model_id", "")
    last_dataset_id = st.session_state.get("clearml_last_dataset_id", "")

    if last_model_id:
        st.info(f"Последний Model ID после отправки эксперимента: `{last_model_id}`")
    if last_dataset_id:
        st.caption(f"Последний Dataset ID: `{last_dataset_id}`. Это не Model ID.")

    model_id = st.text_input(
        "Model ID",
        value=last_model_id,
        key="clearml_model_id",
        help="Вставьте model_id из результата отправки эксперимента, а не dataset_id.",
    ).strip()
    if st.button("Загрузить модель по ID") and model_id:
        if last_dataset_id and model_id == last_dataset_id:
            st.error("В поле Model ID вставлен Dataset ID. Используйте model_id из блока логирования эксперимента.")
            return
        try:
            st.session_state.clearml_model_bundle = load_model_bundle_from_clearml(model_id)
        except Exception as exc:
            st.error(str(exc))
            return
        st.success("Модель загружена.")

    bundle = st.session_state.get("clearml_model_bundle")
    if bundle is None:
        return

    with st.form("clearml_local_prediction"):
        values = _input_values("clearml_local")
        submitted = st.form_submit_button("Предсказать локально загруженной моделью")
    if submitted:
        prediction, probabilities = predict_with_model_bundle(bundle, values)
        st.success(f"Прогноз: {prediction}")
        if probabilities is not None:
            st.dataframe(probabilities.rename("Вероятность").to_frame(), width="stretch")


def _serving_block() -> None:
    st.subheader("ClearML Serving")
    endpoint = st.text_input(
        "Serving endpoint URL",
        value="http://127.0.0.1:8080/serve/predictive_maintenance",
        key="clearml_serving_url",
    )
    with st.form("clearml_serving_prediction"):
        values = _input_values("clearml_serving")
        submitted = st.form_submit_button("Отправить REST-запрос")
    if submitted:
        try:
            response = call_clearml_serving(endpoint, values)
        except Exception as exc:
            st.error(str(exc))
            return
        st.json(response)


def mlops_page() -> None:
    st.title("MLOps и ClearML")
    _show_status()

    st.code(
        "clearml-serving --id <service_id> model add "
        "--engine sklearn --endpoint predictive_maintenance "
        "--model-id <model_id> --preprocess serving_preprocess.py "
        "--name \"Predictive Maintenance Model\" "
        "--project \"Predictive Maintenance Advanced\""
    )

    _experiment_logging_block()
    _dataset_block()
    _model_download_block()
    _serving_block()
