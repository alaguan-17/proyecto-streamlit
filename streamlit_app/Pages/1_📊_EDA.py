import streamlit as st
from src.data_loader import DataLoader
from src.eda import (
    plot_price_distribution,
    plot_room_type_distribution,
    plot_price_boxplot,
    plot_price_vs_rating,
    plot_correlation_heatmap,
)

# Cargar datos
@st.cache_data
def get_data():
    loader = DataLoader()
    return loader.load_data()

train_df, test_df = get_data()

# Visualizaciones
st.title("📊 Exploración de Datos (EDA)")

# Mostrar datos básicos en una tabla
st.subheader("📄 Información General del Dataset")
st.write(train_df.describe().iloc[:2])  # Mostrar solo las primeras 2 filas descriptivas

# Dividir en dos columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribución del Precio")
    st.pyplot(plot_price_distribution(train_df))

    st.subheader("📈 Boxplot del Precio")
    st.pyplot(plot_price_boxplot(train_df))

    st.subheader("📊 Relación entre Precio y Calificación")
    st.pyplot(plot_price_vs_rating(train_df))

with col2:
    st.subheader("📋 Distribución por Tipo de Habitación")
    st.pyplot(plot_room_type_distribution(train_df))

    st.subheader("🗺️ Mapa de Calor de Correlaciones")
    st.pyplot(plot_correlation_heatmap(train_df))

# Mostrar valores nulos en formato resumido
st.subheader("❓ Valores Nulos")
st.markdown(
    "La tabla a continuación muestra la **proporción** de valores nulos en cada columna del conjunto de datos. "
    "Por ejemplo, un valor de `0.25` indica que el 25% de los datos en esa columna son nulos."
)

null_proportion_train = train_df.isnull().mean()
null_proportion_sorted = null_proportion_train.sort_values(ascending=False)
st.write(null_proportion_sorted.head(5))  # Mostrar solo las 5 principales columnas con nulos

