import streamlit as st
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
    "EDA": "streamlit_app/Pages/1_EDA.py",
    "Hipótesis": "streamlit_app/Pages/2_HIPOTESIS.py",
    "Modelos": "streamlit_app/Pages/3_MODELO.py"
}

menu = st.sidebar.radio(
    "Navega por las secciones:",
    options=list(menu_options.keys()),
    format_func=lambda x: f"{menu_options[x].split('/')[-1][0]} {x}"
)

# Cargar y redirigir a la página seleccionada
if menu in menu_options:
    exec(open(menu_options[menu]).read(), globals())

# Footer
st.sidebar.markdown("👨‍💻 **GRUPO UCA** | 🌐 [PROYECTO INTEGRADOR]")
