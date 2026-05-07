import streamlit as st

from analysis_and_model import analysis_and_model_page
from data_overview import data_overview_page
from mlops_page import mlops_page
from presentation import presentation_page


st.set_page_config(page_title="Предиктивное обслуживание", layout="wide")

PAGES = {
    "Обзор данных": data_overview_page,
    "Модель и прогноз": analysis_and_model_page,
    "MLOps ClearML": mlops_page,
    "Итоги проекта": presentation_page,
}

st.sidebar.title("Навигация")
selected_page = st.sidebar.radio("Раздел", list(PAGES))
PAGES[selected_page]()
