import streamlit as st
from PIL import Image
import os
import sys
from src.eda_streamlit import show_eda
from src.hypotheses_streamlit import show_hypotheses
from models_streamlit import show_models

# Agregar el directorio src al PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Airbnb",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Función para verificar si un archivo existe
def check_file_exists(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: El archivo {filepath} no se encuentra.")
        st.stop()

# Cargar imágenes de portada
cover_image_path = "utils/exploration.png"
check_file_exists(cover_image_path)
cover_image = Image.open(cover_image_path)
st.image(cover_image, use_column_width=True)

# Título principal
st.title("🏡 Análisis de Datos de Airbnb")
st.markdown("---")

# Menú de navegación
menu = ["Inicio", "EDA", "Hipótesis", "Modelos"]
choice = st.sidebar.selectbox("Selecciona una sección", menu)

# Configurar cada sección
def show_home():
    st.header("Bienvenido a la Aplicación de Análisis de Airbnb")
    st.markdown(
        "Esta aplicación te permitirá explorar datos, validar hipótesis y analizar modelos de predicción utilizando un dataset de Airbnb.")
    idea_image_path = "utils/idea.png"
    check_file_exists(idea_image_path)
    st.image(idea_image_path, caption="Explora las ideas detrás del análisis")

# Lógica de navegación
if choice == "Inicio":
    show_home()
elif choice == "EDA":
    show_eda()
elif choice == "Hipótesis":
    show_hypotheses()
elif choice == "Modelos":
    show_models()

# Pie de página
st.markdown("---")
st.markdown(
    "Desarrollado por Grupo UCA OMDENA | Fuente: [Kaggle Airbnb Listings](https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml)")
