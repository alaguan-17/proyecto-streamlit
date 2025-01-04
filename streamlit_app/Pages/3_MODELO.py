import streamlit as st
# Configuración inicial
st.set_page_config(
    page_title="Comparativa de Modelos",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)
from src.data_loader import DataLoader
from src.models import Models



# Cargar datos
@st.cache_data
def get_data():
    loader = DataLoader()
    return loader.load_data()

train_df, test_df = get_data()

# Modelos
st.title("🤖 Comparativa de Modelos de Machine Learning")
models = Models(train_df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regresión Lineal")
    metrics_lr = models.linear_regression()
    st.write(f"📏 **RMSE:** {metrics_lr['rmse']:.2f}")
    st.write(f"📈 **R²:** {metrics_lr['r2']:.2f}")

with col2:
    st.subheader("Random Forest")
    metrics_rf = models.random_forest()
    st.write(f"📏 **RMSE:** {metrics_rf['rmse']:.2f}")
    st.write(f"📈 **R²:** {metrics_rf['r2']:.2f}")

st.markdown("### 📊 Comparación Gráfica")
models.plot_comparison()
