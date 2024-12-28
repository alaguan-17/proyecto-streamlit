import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Función para cargar datos
def load_data():
    return train_df, test_df

# Función de exploración inicial para Streamlit
def eda_display(df, section="EDA"):
    import streamlit as st

    st.title("Exploración de Datos (EDA)")

    if section == "EDA":
        st.write("Aquí puedes visualizar los principales insights del dataset.")
        
        # Información básica
        st.subheader("Información del DataFrame")
        buffer = []
        df.info(buf=buffer.append)
        st.text("\n".join(buffer))
        
        # Valores nulos
        st.subheader("Valores nulos")
        null_values, null_proportion = check_null_values(df)
        st.write(null_values)
        st.write(null_proportion)
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas descriptivas")
        st.write(get_descriptive_stats(df))
        
        # Gráficos
        st.subheader("Visualización de datos")
        plot_price_histogram(df)
        plot_price_boxplot(df)
        plot_room_type_distribution(df)

# Funciones auxiliares para EDA
def check_null_values(df):
    null_values = df.isnull().sum()
    null_proportion = null_values / len(df)
    return null_values, null_proportion

def get_descriptive_stats(df):
    return df.describe()

def plot_price_histogram(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['log_price'], kde=True)
    plt.title('Distribución del precio')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_price_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['log_price'])
    plt.title('Boxplot del precio')
    plt.show()

def plot_room_type_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=df)
    plt.title('Distribución de tipos de habitación')
    plt.xlabel('Tipo de habitación')
    plt.ylabel('Frecuencia')
    plt.show()

# Ejecución de funciones
if __name__ == "__main__":
    # Llamar la función de carga de datos
    try:
        train_df, test_df = load_data()
        eda_display(train_df)  # Llamar la función de EDA para visualizar datos
    except FileNotFoundError as e:
        print(e)
