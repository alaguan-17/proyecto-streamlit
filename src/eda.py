# Descripción de la estructura
train_df.info()  # Información sobre las columnas y tipos de datos del conjunto de entrenamiento
test_df.info()   # Información sobre las columnas y tipos de datos del conjunto de prueba

# Verificación de valores nulos
valores_nulos_train = train_df.isnull().sum()  # Sumar valores nulos en el conjunto de entrenamiento
proporcion_nulos_train = valores_nulos_train / len(train_df)
print("Valores nulos en el conjunto de entrenamiento:")
print(valores_nulos_train)
print("Proporción de valores nulos en el conjunto de entrenamiento:")
print(proporcion_nulos_train)

# Comprobación de duplicados
duplicados_train = train_df.duplicated().sum()
duplicados_test = test_df.duplicated().sum()

print(f"Número de filas duplicadas en el conjunto de entrenamiento: {duplicados_train}")
print(f"Número de filas duplicadas en el conjunto de prueba: {duplicados_test}")

#Descripción del contenido del dataset
# Estadísticas descriptivas para las variables numéricas
print("Estadísticas descriptivas del conjunto de entrenamiento:")
print(train_df.describe())

# Frecuencia de las variables categóricas
categorical_columns = train_df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\nFrecuencia de la variable categórica {col}:")
    print(train_df[col].value_counts())

#Visualización de los datos
import matplotlib.pyplot as plt
import seaborn as sns

# Histograma del precio
plt.figure(figsize=(10, 6))
sns.histplot(train_df['log_price'], kde=True)
plt.title('Distribución del precio')
plt.xlabel('Precio')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot del precio
plt.figure(figsize=(10, 6))
sns.boxplot(x=train_df['log_price'])
plt.title('Boxplot del precio')
plt.show()

# Gráfico de barras para el tipo de habitación
plt.figure(figsize=(10, 6))
sns.countplot(x='room_type', data=train_df)
plt.title('Distribución de tipos de habitación')
plt.xlabel('Tipo de habitación')
plt.ylabel('Frecuencia')
plt.show()

#Relaciones entre distintas columnas
# Gráfico de dispersión entre precio y calificaciones
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_price', y='review_scores_rating', data=train_df)
plt.title('Relación entre Precio y Calificación de reseñas')
plt.xlabel('Precio')
plt.ylabel('Calificación de reseñas')
plt.show()

# Matriz de correlación entre variables numéricas
# Seleccionar solo las columnas numéricas del DataFrame
numeric_cols = train_df.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación solo con columnas numéricas
corr_matrix = numeric_cols.corr()

# Graficar el mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de calor de la correlación entre variables')
plt.show()