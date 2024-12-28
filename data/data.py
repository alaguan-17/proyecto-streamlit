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