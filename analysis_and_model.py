from __future__ import annotations

import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from maintenance_core import (
    DATA_PATH,
    TrainingReport,
    model_bundle,
    predict_failure_type,
    read_dataset,
    result_table,
    train_models,
)


@st.cache_data(show_spinner=False)
def _load_local_dataset() -> pd.DataFrame:
    return read_dataset(DATA_PATH)


def _dataset_selector(page_key: str) -> pd.DataFrame | None:
    source = st.radio(
        "Источник данных",
        ["Встроенный AI4I 2020", "Загрузить CSV"],
        horizontal=True,
        key=f"{page_key}_source",
    )
    if source == "Встроенный AI4I 2020":
        return _load_local_dataset()

    uploaded_file = st.file_uploader("CSV-файл", type="csv", key=f"{page_key}_upload")
    if uploaded_file is None:
        return None
    return read_dataset(uploaded_file)


def _store_report(report: TrainingReport) -> None:
    st.session_state.training_report = report
    st.session_state.best_model_name = report.best_model_name
    st.session_state.results = report.results


def _plot_confusion_matrix(report: TrainingReport, selected_model: str) -> None:
    result = report.results[selected_model]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=report.class_names,
        yticklabels=report.class_names,
        ax=ax,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(f"Матрица ошибок: {selected_model}")
    st.pyplot(fig)


def _download_model_button(report: TrainingReport) -> None:
    buffer = io.BytesIO()
    joblib.dump(model_bundle(report), buffer)
    st.download_button(
        "Скачать лучшую модель",
        data=buffer.getvalue(),
        file_name="predictive_maintenance_model.pkl",
        mime="application/octet-stream",
    )


def _prediction_form(report: TrainingReport) -> None:
    st.subheader("Прогноз типа отказа")
    with st.form("prediction_form"):
        product_type = st.selectbox("Тип продукта", ["L", "M", "H"], index=0)
        air_temp = st.number_input("Температура воздуха [K]", value=300.0, step=0.1)
        process_temp = st.number_input("Температура процесса [K]", value=310.0, step=0.1)
        rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500, step=10)
        torque = st.number_input("Крутящий момент [Nm]", value=40.0, step=0.1)
        tool_wear = st.number_input("Износ инструмента [min]", value=120, step=1)

        submitted = st.form_submit_button("Рассчитать")
        if not submitted:
            return

    values = {
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    prediction, probabilities = predict_failure_type(report, values)
    st.success(f"Прогноз: {prediction}")
    if probabilities is not None:
        st.dataframe(
            probabilities.rename("Вероятность").to_frame().style.format("{:.3f}"),
            width="stretch",
        )


def analysis_and_model_page() -> None:
    st.title("Модель и прогнозирование отказов")

    data = _dataset_selector("model")
    if data is None:
        st.info("Загрузите CSV или выберите встроенный датасет.")
        return

    with st.expander("Параметры обучения", expanded=True):
        selected_models = st.multiselect(
            "Модели",
            ["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost"],
            default=["Logistic Regression", "Random Forest", "Extra Trees", "XGBoost"],
        )
        col_left, col_middle, col_right = st.columns(3)
        with col_left:
            test_size = st.slider("Доля тестовой выборки", 0.1, 0.4, 0.2, 0.05)
        with col_middle:
            cv_splits = st.slider("K-Fold", 2, 5, 3)
        with col_right:
            optimize_rf = st.checkbox("Подбор Random Forest", value=False)

    if st.button("Обучить и сравнить", type="primary"):
        with st.spinner("Обучение моделей..."):
            try:
                report = train_models(
                    data,
                    model_names=selected_models,
                    test_size=test_size,
                    cv_splits=cv_splits,
                    optimize_random_forest=optimize_rf,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        _store_report(report)

    report: TrainingReport | None = st.session_state.get("training_report")
    if report is None:
        st.warning("Модели еще не обучены.")
        return

    best = report.best_result
    st.subheader("Результаты сравнения")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Лучшая модель", report.best_model_name)
    metric_cols[1].metric("Accuracy", f"{best.accuracy:.3f}")
    metric_cols[2].metric("Weighted F1", f"{best.weighted_f1:.3f}")
    metric_cols[3].metric("Macro F1", f"{best.macro_f1:.3f}")

    table = result_table(report)
    st.dataframe(table.style.format(precision=3), width="stretch")

    selected_model = st.selectbox("Детализация модели", list(report.results), index=0)
    _plot_confusion_matrix(report, selected_model)
    st.dataframe(
        report.results[selected_model].report.style.format(precision=3),
        width="stretch",
    )

    if report.results[selected_model].best_params:
        st.json(report.results[selected_model].best_params)

    _download_model_button(report)
    _prediction_form(report)

