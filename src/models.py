import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Entrena y evalúa un modelo de regresión lineal."""
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]),
            ("cat", OneHotEncoder(drop="first"), ["room_type", "property_type"]),
        ])),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Regresión Lineal:")
    print(f"- RMSE: {mse ** 0.5:.2f}")
    print(f"- R^2: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title('Valores reales vs Predichos (Regresión Lineal)')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.axline([0, 0], [1, 1], color='red', linestyle='--')
    plt.show()

def train_random_forest(X_train, X_test, y_train, y_test):
    """Entrena y evalúa un modelo de Random Forest."""
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]),
            ("cat", OneHotEncoder(drop="first"), ["room_type", "property_type"]),
        ])),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest:")
    print(f"- RMSE: {mse ** 0.5:.2f}")
    print(f"- R^2: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title('Valores reales vs Predichos (Random Forest)')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.axline([0, 0], [1, 1], color='red', linestyle='--')
    plt.show()

if __name__ == "__main__":
    from data_loader import download_and_load_data
    train_df, _ = download_and_load_data()

    X = train_df[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
    y = train_df["log_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando modelos...")
    train_linear_regression(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
