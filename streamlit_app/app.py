import streamlit as st
from PIL import Image
import os
import sys

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
cover_image = Image.open("utils/exploration.png")
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
    st.image("utils/idea.png", caption="Explora las ideas detrás del análisis")


def show_eda():
    st.header("Exploración de Datos (EDA)")
    st.markdown(
        "Aquí puedes visualizar los principales insights del dataset."
    )
    # Llamar al archivo eda.py para mostrar visualizaciones
    with st.spinner("Cargando EDA..."):
        import src.eda as eda
        eda.display()


def show_hypotheses():
    st.header("Validación de Hipótesis")
    st.markdown(
        "Evalúa las hipótesis relacionadas con el dataset."
    )
    # Llamar al archivo hypotheses.py para mostrar resultados
    with st.spinner("Cargando Hipótesis..."):
        import src.hypotheses as hypotheses
        hypotheses.display()


def show_models():
    st.header("Análisis de Modelos")
    st.markdown(
        "Explora los modelos de predicción entrenados y sus métricas."
    )
    # Llamar al archivo models.py para mostrar métricas y gráficos
    with st.spinner("Cargando Modelos..."):
        import src.models as models
        models.display()

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
    "Desarrollado por [TuNombre] | Fuente: [Kaggle Airbnb Listings](https://www.kaggle.com/datasets/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml)")
