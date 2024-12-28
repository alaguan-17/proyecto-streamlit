# Instalamos kagglehub e importamos pandas
!pip install kagglehub
import kagglehub
import pandas as pd
import os

# Descargamos el dataset usando kagglehub
path = kagglehub.dataset_download("rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml")

print("Path to dataset files:", path)

# Listamos todos los archivos descargados
files = os.listdir(path)
print("Archivos descargados:", files)

# Identificamos los archivos CSV correspondientes
train_file = os.path.join(path, "train.csv")
test_file = os.path.join(path, "test.csv")

# Cargamos los archivos en DataFrames
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Mostramos las primeras filas de cada archivo
print("Primeras filas del dataset de entrenamiento:")
print(train_df.head())

print("\nPrimeras filas del dataset de prueba:")
print(test_df.head())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Función para cargar datos

def load_data():
    """Carga los datos desde la carpeta data y verifica que los archivos existen."""
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"El archivo {train_path} no se encuentra. Verifica la ruta y el archivo.")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"El archivo {test_path} no se encuentra. Verifica la ruta y el archivo.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# Función para obtener información básica de los DataFrames
def get_data_info(train_df, test_df):
    train_info = train_df.info()
    test_info = test_df.info()
    return train_info, test_info

# Función para verificar valores nulos
def check_null_values(df):
    null_values = df.isnull().sum()
    null_proportion = null_values / len(df)
    return null_values, null_proportion

# Función para comprobar duplicados
def check_duplicates(train_df, test_df):
    duplicates_train = train_df.duplicated().sum()
    duplicates_test = test_df.duplicated().sum()
    return duplicates_train, duplicates_test

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
    plt.show()

# Función para graficar boxplot del precio
def plot_price_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['log_price'])
    plt.title('Boxplot del precio')
    plt.show()

# Función para graficar distribución de tipos de habitación
def plot_room_type_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=df)
    plt.title('Distribución de tipos de habitación')
    plt.xlabel('Tipo de habitación')
    plt.ylabel('Frecuencia')
    plt.show()

# Función para graficar relación entre precio y calificaciones
def plot_price_vs_rating(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='log_price', y='review_scores_rating', data=df)
    plt.title('Relación entre Precio y Calificación de reseñas')
    plt.xlabel('Precio')
    plt.ylabel('Calificación de reseñas')
    plt.show()

# Función para graficar mapa de calor de correlación
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_cols.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor de la correlación entre variables')
    plt.show()

# Llamar la función load_data para usar en Streamlit
try:
    train_df, test_df = load_data()
except FileNotFoundError as e:
    print(e)
