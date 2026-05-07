from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from maintenance_core import (
    DATA_PATH,
    NUMERIC_FEATURES,
    build_failure_target,
    dataset_summary,
    read_dataset,
)


@st.cache_data(show_spinner=False)
def _load_local_dataset() -> pd.DataFrame:
    return read_dataset(DATA_PATH)


def _dataset_selector() -> pd.DataFrame | None:
    source = st.radio(
        "Источник данных",
        ["Встроенный AI4I 2020", "Загрузить CSV"],
        horizontal=True,
        key="overview_source",
    )
    if source == "Встроенный AI4I 2020":
        return _load_local_dataset()

    uploaded_file = st.file_uploader("CSV-файл", type="csv", key="overview_upload")
    if uploaded_file is None:
        return None
    return read_dataset(uploaded_file)


def _bar_chart(series: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind="bar", ax=ax, color="#4b7bec")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Количество")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def data_overview_page() -> None:
    st.title("Детальный анализ датасета")

    data = _dataset_selector()
    if data is None:
        st.info("Загрузите CSV или выберите встроенный датасет.")
        return

    try:
        summary = dataset_summary(data)
        failure_target = build_failure_target(data)
    except ValueError as exc:
        st.error(str(exc))
        return

    col_rows, col_cols, col_missing = st.columns(3)
    col_rows.metric("Строк", summary["rows"])
    col_cols.metric("Столбцов", summary["columns"])
    col_missing.metric("Пропусков", int(summary["missing_values"].sum()))

    st.subheader("Фрагмент данных")
    st.dataframe(data.head(20), width="stretch")

    st.subheader("Распределение классов")
    left, right = st.columns(2)
    with left:
        _bar_chart(summary["failure_type_counts"], "Типы отказов")
    with right:
        _bar_chart(summary["product_type_counts"], "Типы продукта")

    st.subheader("Числовые признаки")
    st.dataframe(summary["numeric_description"].style.format(precision=3), width="stretch")

    selected_features = st.multiselect(
        "Гистограммы",
        NUMERIC_FEATURES,
        default=NUMERIC_FEATURES[:3],
    )
    if selected_features:
        plot_data = data.copy()
        plot_data["Failure type"] = failure_target
        fig, axes = plt.subplots(len(selected_features), 1, figsize=(9, 3 * len(selected_features)))
        if len(selected_features) == 1:
            axes = [axes]
        for ax, feature in zip(axes, selected_features):
            sns.histplot(
                data=plot_data,
                x=feature,
                hue="Failure type",
                bins=35,
                ax=ax,
                element="step",
            )
            ax.set_title(feature)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Корреляции")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(summary["correlation"], annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    st.pyplot(fig)

    st.subheader("Связи между признаками")
    scatter_col_left, scatter_col_right = st.columns(2)
    with scatter_col_left:
        x_feature = st.selectbox("Ось X", NUMERIC_FEATURES, index=2)
    with scatter_col_right:
        y_feature = st.selectbox("Ось Y", NUMERIC_FEATURES, index=3)

    plot_data = data.copy()
    plot_data["Failure type"] = failure_target
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=plot_data,
        x=x_feature,
        y=y_feature,
        hue="Failure type",
        alpha=0.75,
        ax=ax,
    )
    ax.legend(loc="best", fontsize="small")
    st.pyplot(fig)

    st.subheader("Пропущенные значения")
    st.dataframe(summary["missing_values"].rename("missing").to_frame(), width="stretch")

