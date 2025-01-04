import streamlit as st
from src.data_loader import DataLoader
from src.eda import EDA
from src.hypotheses import Hypothesis
from src.models import Models
from PIL import Image

# Configuración inicial
st.set_page_config(
    page_title="Airbnb Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

# Cabecera principal
st.title("🏡 Airbnb Analytics Dashboard")
st.markdown("Un análisis interactivo para entender las tendencias y factores clave en el mercado de alquileres a corto plazo.")

# Menú de navegación
menu_options = {
    "Exploración de Datos (EDA)": "📊",
    "Hipótesis": "💡",
    "Modelos": "🤖"
}
menu = st.sidebar.radio(
    "Navega por las secciones:",
    options=list(menu_options.keys()),
    format_func=lambda x: f"{menu_options[x]} {x}"
)

# Cargar datos
data_loader = DataLoader()
train_df, test_df = data_loader.load_data()

# Exploración de Datos (EDA)
if menu == "Exploración de Datos (EDA)":
    st.header("📊 Exploración de Datos (EDA)")
    eda = EDA(train_df)

    # Diseño de mosaico
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Distribución del Precio")
        eda.plot_price_distribution()

        st.subheader("🏠 Distribución por Tipo de Habitación")
        eda.plot_room_type_distribution()

    with col2:
        st.subheader("🔍 Relación Precio-Calificación")
        eda.plot_price_rating_relationship()

        st.subheader("🌐 Correlaciones entre Variables")
        eda.plot_correlation_heatmap()

    st.markdown("---")
    st.info("🔎 **Explora tendencias clave para comprender mejor los datos iniciales.**")

# Hipótesis
elif menu == "Hipótesis":
    st.header("💡 Análisis de Hipótesis")
    hypothesis = Hypothesis(train_df)

    st.markdown("🎯 **Hipótesis Presentadas:**")

    # Hipótesis 1
    with st.expander("1️⃣ El precio promedio es mayor para propiedades completas."):
        result_h1 = hypothesis.hypothesis_1()
        st.write(f"**Resultado:** {result_h1['conclusion']} (p-valor: {result_h1['p_value']:.4f})")

    # Hipótesis 2
    with st.expander("2️⃣ Las propiedades con calificaciones altas tienen más reservas."):
        result_h2 = hypothesis.hypothesis_2()
        st.write(f"**Resultado:** {result_h2['conclusion']} (p-valor: {result_h2['p_value']:.4f})")

    # Hipótesis 3
    with st.expander("3️⃣ Las propiedades céntricas tienen precios más altos."):
        result_h3 = hypothesis.hypothesis_3()
        st.write(f"**Resultado:** {result_h3['conclusion']} (p-valor: {result_h3['p_value']:.4f})")

    # Hipótesis 4
    with st.expander("4️⃣ Los anfitriones con múltiples propiedades tienen precios más bajos."):
        result_h4 = hypothesis.hypothesis_4()
        st.write(f"**Resultado:** {result_h4['conclusion']} (p-valor: {result_h4['p_value']:.4f})")

    # Hipótesis 5
    with st.expander("5️⃣ Más amenidades están asociadas con mejores calificaciones."):
        result_h5 = hypothesis.hypothesis_5()
        st.write(f"**Resultado:** {result_h5['conclusion']} (correlación: {result_h5['correlation']:.4f})")

    st.markdown("---")
    st.success("✨ **Cada hipótesis incluye visualizaciones interactivas y análisis detallados.**")

# Modelos
elif menu == "Modelos":
    st.header("🤖 Comparativa de Modelos de Machine Learning")
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

    st.markdown("---")
    st.warning("🔍 **Descubre cuál modelo es más adecuado para predecir precios en el mercado de alquileres.**")

# Footer
st.sidebar.markdown("👨‍💻 **GRUPO UCA** | 🌐 [PROYECTO INTEGRADOR]")