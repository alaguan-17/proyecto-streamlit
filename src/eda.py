import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader

# Definición de las funciones de visualización
def plot_price_histogram(df):
    """Genera un histograma para la columna de precio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['log_price'], bins=30, kde=True, ax=ax, color='blue')
    ax.set_title('Histograma del Precio')
    ax.set_xlabel('Precio (log)')
    ax.set_ylabel('Frecuencia')
    return fig

def plot_price_boxplot(df):
    """Genera un boxplot para la columna de precio."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, y='log_price', ax=ax, color='green')
    ax.set_title('Boxplot del Precio')
    ax.set_ylabel('Precio (log)')
    return fig

def plot_room_type_distribution(df):
    """Genera un gráfico de barras para la distribución de tipos de habitación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='room_type', ax=ax, palette='pastel')
    ax.set_title('Distribución de Tipos de Habitación')
    ax.set_xlabel('Tipo de Habitación')
    ax.set_ylabel('Frecuencia')
    return fig

def plot_price_vs_rating(df):
    """Genera un scatterplot para precio vs calificación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='review_scores_rating', y='log_price', alpha=0.6, ax=ax, color='purple')
    ax.set_title('Relación entre Precio y Calificación')
    ax.set_xlabel('Calificación')
    ax.set_ylabel('Precio (log)')
    return fig

def plot_correlation_heatmap(df):
    """Genera un mapa de calor de correlación."""
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calcular la matriz de correlación solo para variables numéricas
    corr_matrix = df.select_dtypes(include=['number']).corr()
    
    # Generar el mapa de calor
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title('Mapa de Calor de Correlación')
    
    return fig  # Devolver la figura


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
st.text(train_df.info())

st.subheader("Conjunto de Prueba")
st.text(test_df.info())

# Sección de valores nulos
st.header("Valores Nulos")
null_values_train = train_df.isnull().sum()
null_proportion_train = null_values_train / len(train_df)
st.write("Valores nulos en el conjunto de entrenamiento:")
st.write(null_values_train)
st.write("Proporción de valores nulos:")
st.write(null_proportion_train)

# Sección de duplicados
st.header("Duplicados")
duplicates_train = train_df.duplicated().sum()
duplicates_test = test_df.duplicated().sum()
st.write(f"Filas duplicadas en el conjunto de entrenamiento: {duplicates_train}")
st.write(f"Filas duplicadas en el conjunto de prueba: {duplicates_test}")

# Estadísticas descriptivas
st.header("Estadísticas Descriptivas")
st.subheader("Estadísticas del Conjunto de Entrenamiento")
st.dataframe(train_df.describe())

# Frecuencia de variables categóricas
st.header("Frecuencia de Variables Categóricas")
categorical_frequencies = {col: train_df[col].value_counts() for col in train_df.select_dtypes(include=['object']).columns}
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
