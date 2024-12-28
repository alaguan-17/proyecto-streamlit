import streamlit as st
import pandas as pd
from models import train_linear_regression_model, train_random_forest_model

# Configuración de la página
st.set_page_config(page_title="Modelos de Machine Learning", layout="wide")

# Título de la página
st.title("📊 Modelos de Machine Learning")
st.markdown("Selecciona un modelo para entrenar y evaluar su desempeño en el dataset de Airbnb.")

# Cargar el dataset
def load_data():
    return pd.read_csv("data/processed/processed_data.csv")

data = load_data()

# Selección de modelo
model_option = st.selectbox("Selecciona el modelo que deseas entrenar:", ["Regresión Lineal", "Random Forest"])

# Entrenar y mostrar resultados según el modelo seleccionado
if model_option == "Regresión Lineal":
    st.subheader("Resultados: Regresión Lineal")
    rmse, r2, plot_function = train_linear_regression_model(data)

    # Mostrar métricas
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R^2:** {r2:.2f}")

    # Mostrar visualizaciones
    st.write("### Gráficos del Modelo")
    plot_function()

elif model_option == "Random Forest":
    st.subheader("Resultados: Random Forest")
    rmse, r2, plot_function = train_random_forest_model(data)

    # Mostrar métricas
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R^2:** {r2:.2f}")

    # Mostrar visualizaciones
    st.write("### Gráficos del Modelo")
    plot_function()

# Pie de página
st.markdown("---")
st.markdown("Desarrollado por Grupo UCA OMDENA | Fuente: [Kaggle Airbnb Listings](https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml)")
