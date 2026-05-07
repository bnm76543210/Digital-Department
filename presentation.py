from __future__ import annotations

import streamlit as st

from maintenance_core import result_table


def presentation_page() -> None:
    st.title("Итоги проекта")

    report = st.session_state.get("training_report")
    if report is None:
        best_model_text = "Модели еще не обучены"
        metrics_text = "Запустите обучение на странице модели, чтобы здесь появились итоговые метрики."
    else:
        best = report.best_result
        best_model_text = report.best_model_name
        metrics_text = (
            f"Accuracy: **{best.accuracy:.3f}**, "
            f"Weighted F1: **{best.weighted_f1:.3f}**, "
            f"Macro F1: **{best.macro_f1:.3f}**"
        )

    st.markdown(
        f"""
        ## Продвинутая классификация отказов оборудования

        Проект решает задачу предиктивного обслуживания на датасете AI4I 2020.
        В новой версии приложение не только определяет факт отказа, но и
        классифицирует его тип: TWF, HDF, PWF, OSF, RNF, Multiple failures или No failure.

        ## Выполненные задачи

        - Добавлена мультиклассовая целевая переменная по типам отказов.
        - Добавлена отдельная страница детального анализа датасета.
        - Предобработка перенесена в `Pipeline`: заполнение пропусков, IQR-клиппинг выбросов,
          масштабирование числовых признаков и One-Hot Encoding для `Type`.
        - Реализовано сравнение Logistic Regression, Random Forest, Extra Trees и XGBoost.
        - Добавлена K-Fold кросс-валидация и опциональный подбор Random Forest через GridSearchCV.
        - Добавлен интерфейс прогноза нового наблюдения и скачивание лучшей модели.
        - Добавлена MLOps-страница для ClearML: логирование экспериментов, версия датасета,
          загрузка модели из Model Registry и REST-запросы в ClearML Serving.

        ## Текущий результат

        **Лучшая модель:** {best_model_text}

        {metrics_text}
        """
    )

    if report is not None:
        st.dataframe(result_table(report).style.format(precision=3), width="stretch")

