import kagglehub
import pandas as pd
import os

class DataLoader:
    @staticmethod
    def load_data():
        """Descarga y carga los datos desde Kaggle."""
        # Descargar el dataset
        path = kagglehub.dataset_download("rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml")

        # Identificar archivos descargados
        train_file = os.path.join(path, "train.csv")
        test_file = os.path.join(path, "test.csv")

        # Cargar los datasets en DataFrames
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        return train_df, test_df
