import kagglehub
import pandas as pd
import os

def load_kaggle_data():
    # Descargamos el dataset usando kagglehub
    path = kagglehub.dataset_download("rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml")

    # Identificamos los archivos CSV correspondientes
    train_file = os.path.join(path, "train.csv")
    test_file = os.path.join(path, "test.csv")

    # Cargamos los archivos en DataFrames
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    return train_df, test_df
