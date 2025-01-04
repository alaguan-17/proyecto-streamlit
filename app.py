import streamlit as st

# Configuración inicial de la página principal
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
    "EDA": "streamlit_app/Pages/1_📊_EDA.py",
    "Hipótesis": "streamlit_app/Pages/2_💡_HIPOTESIS.py",
    "Modelos": "streamlit_app/Pages/3_🤖_MODELO.py"
}

menu = st.sidebar.radio(
    "Navega por las secciones:",
    options=list(menu_options.keys()),
    format_func=lambda x: x
)

# Redirigir a la página seleccionada
if menu in menu_options:
    with open(menu_options[menu], "r", encoding="utf-8") as file:
        exec(file.read(), globals())
