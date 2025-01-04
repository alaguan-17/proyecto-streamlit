import streamlit as st
from src.models import Models
from src.data_loader import DataLoader

# Configuración de la página
st.title("🤖 Comparativa de Modelos de Machine Learning")
st.markdown("Analizamos dos modelos: **Regresión Lineal** y **Random Forest**, evaluando su desempeño en la predicción de precios de Airbnb.")

# Cargar datos
@st.cache_data
def load_data():
    data_loader = DataLoader()
    train_df, _ = data_loader.load_data()
    return train_df

train_df = load_data()

# Inicializar el modelo con una muestra significativa
models = Models(train_df, sample_fraction=0.1)

# Mostrar resultados de Regresión Lineal
st.subheader("📈 Regresión Lineal")
metrics_lr = models.linear_regression()
st.write(f"**RMSE**: {metrics_lr['rmse']:.2f}")
st.write(f"**R²**: {metrics_lr['r2']:.2f}")

# Mostrar resultados de Random Forest
st.subheader("🌲 Random Forest")
metrics_rf = models.random_forest()
st.write(f"**RMSE**: {metrics_rf['rmse']:.2f}")
st.write(f"**R²**: {metrics_rf['r2']:.2f}")

# Comparación gráfica
st.subheader("📊 Comparación Gráfica")
models.plot_comparison()
