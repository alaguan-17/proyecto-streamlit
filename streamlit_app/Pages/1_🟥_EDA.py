import streamlit as st
from src.eda import (
    plot_price_distribution, plot_room_type_distribution,
    plot_correlation_matrix
)
from src.data_loader import DataLoader

# Cargar datos
data_loader = DataLoader()
train_df, _ = data_loader.load_data()

# Título y descripción
st.title("Exploración de Datos (EDA)")
st.markdown("Aquí puedes explorar los datos de manera visual e interactiva.")

# Diseño de columnas para visualizaciones
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribución del Precio")
    plot_price_distribution(train_df)

    st.subheader("Distribución por Tipo de Habitación")
    plot_room_type_distribution(train_df)

with col2:
    st.subheader("Mapa de Calor de Correlaciones")
    plot_correlation_matrix(train_df)

st.markdown("---")
st.info("🔍 **Interpreta estas visualizaciones para entender mejor los datos iniciales.**")