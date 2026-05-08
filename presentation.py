from __future__ import annotations

import streamlit as st

from maintenance_core import result_table


FAILURE_CLASSES = {
    "No failure": "отказ не ожидается",
    "TWF": "износ инструмента",
    "HDF": "недостаточный отвод тепла",
    "PWF": "некорректная мощность процесса",
    "OSF": "перегрузка оборудования",
    "RNF": "случайный отказ",
    "Multiple failures": "несколько отказов одновременно",
}


PROJECT_TASKS = [
    ("Мультиклассовая классификация", "Модель предсказывает конкретный тип отказа, а не только факт поломки."),
    ("Детальный анализ данных", "Добавлены распределения, статистика, корреляции, гистограммы и scatter plot."),
    ("Единая предобработка", "Pipeline одинаково работает при обучении, ручном прогнозе и ClearML Serving."),
    ("Сравнение моделей", "Проверяются Logistic Regression, Random Forest, Extra Trees и XGBoost."),
    ("Кросс-валидация", "Используется StratifiedKFold и подбор параметров Random Forest."),
    ("ClearML", "Метрики, датасет и модель отправляются в ClearML, Serving проверяется через REST API."),
    ("Тесты", "Unit-тесты проверяют целевую переменную, подготовку признаков и прогноз."),
]


WORKFLOW_STEPS = [
    "Загрузка и проверка датасета AI4I 2020",
    "Формирование целевой переменной Failure type",
    "Очистка, кодирование и масштабирование признаков",
    "Обучение нескольких моделей классификации",
    "Сравнение метрик и матриц ошибок",
    "Отправка эксперимента и модели в ClearML",
    "Проверка REST-прогноза через Docker и ClearML Serving",
]


def _model_data() -> dict[str, object]:
    report = st.session_state.get("training_report")
    if report is None:
        return {
            "trained": False,
            "best_model": "Не обучена",
            "accuracy": None,
            "weighted_f1": None,
            "macro_f1": None,
            "avg_accuracy": None,
            "avg_weighted_f1": None,
            "improvements": ["Запустить обучение на странице «Модель и прогноз» и сравнить модели."],
        }

    best = report.best_result
    avg_accuracy = sum(model.accuracy for model in report.results.values()) / len(report.results)
    avg_weighted_f1 = sum(model.weighted_f1 for model in report.results.values()) / len(report.results)

    improvements = []
    if best.weighted_f1 < 0.9:
        improvements.append("Усилить обработку дисбаланса классов и подобрать гиперпараметры моделей.")
    if best.macro_f1 < 0.7:
        improvements.append("Добавить больше примеров редких отказов, чтобы улучшить качество по малым классам.")
    if report.best_model_name == "Logistic Regression":
        improvements.append("Проверить нелинейные модели: Random Forest, Extra Trees и XGBoost.")
    if not improvements:
        improvements = [
            "Провести дополнительные эксперименты с Optuna для более тонкой настройки гиперпараметров.",
            "Добавить мониторинг качества модели после развертывания через ClearML Serving.",
            "Проверить модель на реальных производственных данных, если они появятся.",
        ]

    return {
        "trained": True,
        "best_model": report.best_model_name,
        "accuracy": best.accuracy,
        "weighted_f1": best.weighted_f1,
        "macro_f1": best.macro_f1,
        "avg_accuracy": avg_accuracy,
        "avg_weighted_f1": avg_weighted_f1,
        "improvements": improvements,
    }


def _metric_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _task_card(title: str, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)


def _render_overview() -> None:
    with st.container(border=True):
        st.subheader("Цель проекта")
        st.write(
            "Разработать приложение для предиктивного обслуживания оборудования, "
            "которое анализирует технологические параметры, обучает модели и "
            "предсказывает тип возможного отказа."
        )

    st.subheader("Классы отказов")
    class_rows = [list(FAILURE_CLASSES.items())[:4], list(FAILURE_CLASSES.items())[4:]]
    for row in class_rows:
        columns = st.columns(len(row))
        for column, (label, description) in zip(columns, row):
            with column:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.caption(description)


def _render_current_result(model_data: dict[str, object]) -> None:
    st.subheader("Текущий результат")
    if model_data["trained"]:
        st.success(f"Лучшая модель: {model_data['best_model']}")
    else:
        st.warning("Модели еще не обучены. Откройте страницу «Модель и прогноз» и запустите обучение.")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", _metric_value(model_data["accuracy"]))
    metric_cols[1].metric("Weighted F1", _metric_value(model_data["weighted_f1"]))
    metric_cols[2].metric("Macro F1", _metric_value(model_data["macro_f1"]))
    metric_cols[3].metric("Средняя F1", _metric_value(model_data["avg_weighted_f1"]))


def _render_workflow() -> None:
    st.subheader("Этапы работы")
    for index, step in enumerate(WORKFLOW_STEPS, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {step}**")


def _render_tasks() -> None:
    st.subheader("Что реализовано")
    first_row = PROJECT_TASKS[:3]
    second_row = PROJECT_TASKS[3:6]
    third_row = PROJECT_TASKS[6:]

    for row in [first_row, second_row, third_row]:
        columns = st.columns(len(row))
        for column, (title, text) in zip(columns, row):
            with column:
                _task_card(title, text)


def _render_improvements(items: list[str]) -> None:
    st.subheader("Возможные улучшения")
    for item in items:
        st.info(item)


def presentation_page() -> None:
    model_data = _model_data()
    report = st.session_state.get("training_report")

    st.title("Итоги проекта")
    st.write(
        "Краткая презентация выполненной работы: от анализа датасета и обучения моделей "
        "до интеграции ClearML и проверки REST-прогноза через Serving."
    )

    _render_overview()

    st.divider()
    _render_workflow()

    st.divider()
    _render_tasks()

    st.divider()
    st.subheader("Таблица сравнения моделей")
    if report is not None:
        st.dataframe(result_table(report).style.format(precision=3), width="stretch")
    else:
        st.info("Таблица появится после обучения моделей на странице «Модель и прогноз».")

    st.divider()
    _render_current_result(model_data)

    st.divider()
    _render_improvements(model_data["improvements"])
