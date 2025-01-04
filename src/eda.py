import streamlit as st
import pandas as pd
from src.data_loader import DataLoader

# Definición de las funciones directamente en este archivo
def get_data_info(df):
    """Devuelve información básica del DataFrame."""
    return df.info()

def check_null_values(df):
    """Devuelve la cantidad y proporción de valores nulos."""
    null_values = df.isnull().sum()
    null_proportion = null_values / len(df)
    return null_values, null_proportion

def check_duplicates(df):
    """Devuelve la cantidad de duplicados en el DataFrame."""
    return df.duplicated().sum()

def get_descriptive_stats(df):
    """Devuelve estadísticas descriptivas del DataFrame."""
    return df.describe()

def get_categorical_frequencies(df):
    """Devuelve frecuencias de variables categóricas."""
    return {col: df[col].value_counts() for col in df.select_dtypes(include=['object']).columns}

# Configuración de la página de EDA
st.title("Exploración de Datos (EDA)")
st.markdown("Aquí puedes visualizar los principales insights del dataset.")

# Cargar los datos desde Kaggle
@st.cache_data
def load_data():
    data_loader = DataLoader()
    train_df, test_df = data_loader.load_data()
    return train_df, test_df

# Llamar la función para cargar los datos
train_df, test_df = load_data()

# Sección de información básica
st.header("Información Básica del Conjunto de Datos")
st.subheader("Conjunto de Entrenamiento")
st.text(get_data_info(train_df))

st.subheader("Conjunto de Prueba")
st.text(get_data_info(test_df))

# Sección de valores nulos
st.header("Valores Nulos")
null_values_train, null_proportion_train = check_null_values(train_df)
st.write("Valores nulos en el conjunto de entrenamiento:")
st.write(null_values_train)
st.write("Proporción de valores nulos:")
st.write(null_proportion_train)

# Sección de duplicados
st.header("Duplicados")
duplicates_train = check_duplicates(train_df)
duplicates_test = check_duplicates(test_df)
st.write(f"Filas duplicadas en el conjunto de entrenamiento: {duplicates_train}")
st.write(f"Filas duplicadas en el conjunto de prueba: {duplicates_test}")

# Estadísticas descriptivas
st.header("Estadísticas Descriptivas")
st.subheader("Estadísticas del Conjunto de Entrenamiento")
st.dataframe(get_descriptive_stats(train_df))

# Frecuencia de variables categóricas
st.header("Frecuencia de Variables Categóricas")
categorical_frequencies = get_categorical_frequencies(train_df)
for col, freq in categorical_frequencies.items():
    st.subheader(f"Frecuencia de {col}")
    st.write(freq)

# Visualización de histogramas
st.header("Visualizaciones")
st.subheader("Histograma del Precio")
st.pyplot(plot_price_histogram(train_df))

st.subheader("Boxplot del Precio")
st.pyplot(plot_price_boxplot(train_df))

st.subheader("Distribución de Tipos de Habitación")
st.pyplot(plot_room_type_distribution(train_df))

st.subheader("Relación entre Precio y Calificación")
st.pyplot(plot_price_vs_rating(train_df))

st.subheader("Mapa de Calor de Correlación")
st.pyplot(plot_correlation_heatmap(train_df))
