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
    "Exploración de Datos (EDA)": "streamlit_app/Pages/1__EDA.py",
    "Hipótesis": "streamlit_app/Pages/2__HIPOTESIS.py",
    "Modelos": "streamlit_app/Pages/3__MODELO.py"
}

# Navegación con emojis y nombres de las páginas
menu = st.sidebar.radio(
    "Navega por las secciones:",
    options=list(menu_options.keys()),
    format_func=lambda x: f"{x.split()[1]}"  # Muestra los títulos de las secciones con emojis
)

# Cargar y ejecutar la página seleccionada
if menu in menu_options:
    try:
        # Leer y ejecutar el archivo de la página con codificación utf-8
        exec(open(menu_options[menu], 'r', encoding='utf-8').read(), globals())
    except FileNotFoundError:
        st.error(f"No se pudo encontrar la página: {menu_options[menu]}")
    except UnicodeDecodeError as e:
        st.error(f"Error de codificación al leer la página: {menu_options[menu]}\n{str(e)}")

# Footer
st.sidebar.markdown("👨‍💻 **GRUPO UCA** | 🌐 [PROYECTO INTEGRADOR]")
