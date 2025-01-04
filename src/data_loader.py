import kagglehub
import pandas as pd
import os

def download_and_load_data():
    """Descarga y carga los datos desde Kaggle usando kagglehub."""
    # Descargar el dataset
    path = kagglehub.dataset_download("rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml")

    # Identificar archivos descargados
    train_file = os.path.join(path, "train.csv")
    test_file = os.path.join(path, "test.csv")

    # Cargar los datasets en DataFrames
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return train_df, test_df

if __name__ == "__main__":
    train_data, test_data = download_and_load_data()
    print("Datos de entrenamiento cargados:")
    print(train_data.head())
    print("\nDatos de prueba cargados:")
    print(test_data.head())
