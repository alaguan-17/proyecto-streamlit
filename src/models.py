import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def train_linear_regression_model(train_df):
    """Entrena un modelo de regresión lineal y devuelve métricas y visualizaciones."""
    # Seleccionar características y objetivo
    X = train_df[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
    y = train_df["log_price"]

    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocesamiento
    categorical_features = ["room_type", "property_type"]
    numerical_features = ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = pipeline.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Visualizaciones
    def plot_linear_regression():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.title('Valores reales vs Predichos')
        plt.xlabel('Valores reales')
        plt.ylabel('Valores predichos')
        plt.axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=2)
        plt.show()

        errores = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errores, kde=True, bins=30)
        plt.title('Distribución de los errores (residuales)')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.show()

    return rmse, r2, plot_linear_regression

def train_random_forest_model(train_df):
    """Entrena un modelo Random Forest y devuelve métricas y visualizaciones."""
    # Seleccionar características y objetivo
    X = train_df[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
    y = train_df["log_price"]

    # Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocesamiento
    categorical_features = ["room_type", "property_type"]
    numerical_features = ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = pipeline.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Visualizaciones
    def plot_random_forest():
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.title('Valores reales vs Predichos (Random Forest)')
        plt.xlabel('Valores reales')
        plt.ylabel('Valores predichos')
        plt.axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=2)
        plt.show()

        errores = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errores, kde=True, bins=30)
        plt.title('Distribución de los errores (residuales) - Random Forest')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.show()

        importances = pipeline.named_steps["model"].feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
        feature_names = list(numerical_features) + list(feature_names)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title('Feature Importance - Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()

    return rmse, r2, plot_random_forest
