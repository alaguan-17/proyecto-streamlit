#Modelo Regresion lineal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


data = train_df  # DataFrame cargado previamente

# Seleccionar características y objetivo
X = data[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
y = data["log_price"]

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento: Codificación de variables categóricas y escalado
categorical_features = ["room_type", "property_type"]
numerical_features = ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]

# Crear transformador para preprocesamiento con imputación de valores faltantes
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Imputar valores faltantes con la media
            ("scaler", StandardScaler())
        ]), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),  # Ignorar categorías desconocidas
    ]
)

# Crear pipeline para la regresión lineal
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Realizar predicciones
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)  # Calcular el MSE
rmse = mse ** 0.5  # Calcular la raíz cuadrada del MSE
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo de regresión lineal:")
print(f"- RMSE: {rmse:.2f}")
print(f"- R^2: {r2:.2f}")

# Conclusión: Analizar la importancia de las características
# Nota: Esto aplica solo si es factible obtener coeficientes de regresión tras preprocesar
if hasattr(pipeline.named_steps["model"], "coef_"):
    importances = pipeline.named_steps["model"].coef_
    print("Importancia de características:", importances)
else:
    print("El modelo no tiene coeficientes para analizar.")

# Gráfico de dispersión entre valores reales y predichos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Valores reales vs Predichos')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=2)  # Línea diagonal
plt.show()


# Histograma de errores (residuales)
errores = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errores, kde=True, bins=30)
plt.title('Distribución de los errores (residuales)')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()


#Modelo Random forest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Usar el DataFrame ya cargado en el notebook
# Asegúrate de que 'train_df' esté definido previamente en el notebook
data = train_df  # DataFrame cargado previamente

# Seleccionar características y objetivo
X = data[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
y = data["log_price"]

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento: Codificación de variables categóricas y escalado
categorical_features = ["room_type", "property_type"]
numerical_features = ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]

# Crear transformador para preprocesamiento con imputación de valores faltantes
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),  # Imputar valores faltantes con la media
            ("scaler", StandardScaler())
        ]), numerical_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),  # Ignorar categorías desconocidas
    ]
)

# Crear pipeline para el Random Forest
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Realizar predicciones
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)  # Calcular el MSE
rmse = mse ** 0.5  # Calcular la raíz cuadrada del MSE
r2 = r2_score(y_test, y_pred)

print("Evaluación del modelo Random Forest:")
print(f"- RMSE: {rmse:.2f}")
print(f"- R^2: {r2:.2f}")

# Gráfico de dispersión entre valores reales y predichos
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.title('Valores reales vs Predichos (Random Forest)')
plt.xlabel('Valores reales')
plt.ylabel('Valores predichos')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=2)  # Línea diagonal
plt.show()

# Histograma de errores (residuales)
errores = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errores, kde=True, bins=30)
plt.title('Distribución de los errores (residuales) - Random Forest')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()

# Importancia de características
importances = pipeline.named_steps["model"].feature_importances_
feature_names = pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
feature_names = list(numerical_features) + list(feature_names)

# Agrupar características menos importantes como "Otras"
threshold = 0.01  # Umbral de importancia
important_features = [(name, imp) for name, imp in zip(feature_names, importances) if imp >= threshold]
other_features_importance = sum(imp for name, imp in zip(feature_names, importances) if imp < threshold)
important_features.append(("Others", other_features_importance))

# Separar nombres y valores
names, values = zip(*important_features)

plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=names)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

