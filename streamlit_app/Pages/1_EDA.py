import streamlit as st
from src.data_loader import DataLoader
from src.eda import (
    plot_price_distribution,
    plot_room_type_distribution,
    plot_correlation_matrix
)

# Configuración inicial
st.set_page_config(
    page_title="Exploración de Datos (EDA)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)
# Código principal para Streamlit
st.set_page_config(page_title="Exploración de Datos", layout="wide", page_icon="📊")

st.title("📊 Exploración de Datos (EDA)")
st.markdown("Analiza las características principales del conjunto de datos.")

# Cargar datos
@st.cache_data
def get_data():
    loader = DataLoader()
    return loader.load_data()

train_df, test_df = get_data()

# Visualizaciones
st.title("📊 Exploración de Datos (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución del Precio")
    plot_price_distribution(train_df)

    st.subheader("Distribución por Tipo de Habitación")
    plot_room_type_distribution(train_df)

with col2:
    st.subheader("Mapa de Calor de Correlaciones")
    plot_correlation_matrix(train_df)
