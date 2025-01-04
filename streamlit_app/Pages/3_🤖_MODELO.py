import streamlit as st
from src.models import Models
from src.data_loader import DataLoader
import pandas as pd

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

# Comparación gráfica en dos columnas
st.subheader("📊 Comparación Gráfica")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Regresión Lineal")
    for fig in metrics_lr["figures"][:2]:  # Mostrar solo los primeros dos gráficos
        st.pyplot(fig)

with col2:
    st.markdown("### Random Forest")
    for fig in metrics_rf["figures"][:2]:  # Mostrar solo los primeros dos gráficos
        st.pyplot(fig)

# Comparación de valores reales y proyectados
st.subheader("📋 Comparación de Valores Reales vs Predichos")
comparison_df = pd.DataFrame({
    "Valores Reales": models.y_test,
    "Predicciones Regresión Lineal": metrics_lr["predictions"],
    "Predicciones Random Forest": metrics_rf["predictions"],
}).reset_index(drop=True)

st.dataframe(comparison_df)
