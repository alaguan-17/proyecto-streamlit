import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Función para obtener información básica de los DataFrames
def get_data_info(df):
    buffer = []
    df.info(buf=buffer)
    return "\n".join(buffer)

# Función para verificar valores nulos
def check_null_values(df):
    null_values = df.isnull().sum()
    null_proportion = null_values / len(df)
    return null_values, null_proportion

# Función para comprobar duplicados
def check_duplicates(df):
    duplicates = df.duplicated().sum()
    return duplicates

# Función para obtener estadísticas descriptivas de variables numéricas
def get_descriptive_stats(df):
    return df.describe()

# Función para obtener frecuencias de variables categóricas
def get_categorical_frequencies(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    frequencies = {}
    for col in categorical_columns:
        frequencies[col] = df[col].value_counts()
    return frequencies

# Función para graficar histograma de precios
def plot_price_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['log_price'], kde=True)
    plt.title('Distribución del precio')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    st.pyplot(plt)

# Función para graficar boxplot del precio
def plot_price_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['log_price'])
    plt.title('Boxplot del precio')
    st.pyplot(plt)

# Función para graficar distribución de tipos de habitación
def plot_room_type_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=df)
    plt.title('Distribución de tipos de habitación')
    plt.xlabel('Tipo de habitación')
    plt.ylabel('Frecuencia')
    st.pyplot(plt)

# Función para graficar relación entre precio y calificaciones
def plot_price_vs_rating(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='log_price', y='review_scores_rating', data=df)
    plt.title('Relación entre Precio y Calificación de reseñas')
    plt.xlabel('Precio')
    plt.ylabel('Calificación de reseñas')
    st.pyplot(plt)

# Función para graficar mapa de calor de correlación
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_cols.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor de la correlación entre variables')
    st.pyplot(plt)

# Función principal para mostrar el análisis exploratorio de datos
def display():
    st.subheader("Análisis Exploratorio de Datos (EDA)")

    # Cargar datos
    train_df = pd.read_csv("path/to/train.csv")  # Cambia la ruta al dataset real
    test_df = pd.read_csv("path/to/test.csv")  # Cambia la ruta al dataset real

    # Mostrar información básica
    st.markdown("### Información del Dataset de Entrenamiento")
    st.text(get_data_info(train_df))

    # Verificar valores nulos
    st.markdown("### Valores Nulos")
    null_values, null_proportion = check_null_values(train_df)
    st.write("Valores nulos:", null_values)
    st.write("Proporción de valores nulos:", null_proportion)

    # Comprobar duplicados
    st.markdown("### Duplicados")
    duplicates = check_duplicates(train_df)
    st.write(f"Número de filas duplicadas: {duplicates}")

    # Estadísticas descriptivas
    st.markdown("### Estadísticas Descriptivas")
    st.write(get_descriptive_stats(train_df))

    # Visualizaciones
    st.markdown("### Visualizaciones")
    st.markdown("#### Histograma del Precio")
    plot_price_histogram(train_df)

    st.markdown("#### Boxplot del Precio")
    plot_price_boxplot(train_df)

    st.markdown("#### Distribución de Tipos de Habitación")
    plot_room_type_distribution(train_df)

    st.markdown("#### Relación entre Precio y Calificación")
    plot_price_vs_rating(train_df)

    st.markdown("#### Mapa de Calor de Correlación")
    plot_correlation_heatmap(train_df)
