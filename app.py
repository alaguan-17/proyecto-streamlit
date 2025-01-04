import streamlit as st
from PIL import Image
from src.data_loader import DataLoader

# Configuración inicial de la página
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
    "Exploración de Datos (EDA)": "streamlit_app/Pages/1_EDA.py",
    "Hipótesis": "streamlit_app/Pages/2_HIPOTESIS.py",
    "Modelos": "streamlit_app/Pages/3_MODELO.py"
}

menu = st.sidebar.radio(
    "Navega por las secciones:",
    options=list(menu_options.keys()),
    format_func=lambda x: x
)

# Cargar y redirigir a la página seleccionada
if menu in menu_options:
    exec(open(menu_options[menu]).read(), globals())

# Footer
st.sidebar.markdown("\ud83d\udc68\u200d\ud83d\udcbb **GRUPO UCA** | \ud83c\udf10 [PROYECTO INTEGRADOR]")
