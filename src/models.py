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


class Models:
    def __init__(self, data):
        """Inicializa los datos y divide en conjuntos de entrenamiento y prueba."""
        self.data = data
        self.X = data[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
        self.y = data["log_price"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def linear_regression(self):
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

        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Visualización
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred)
        plt.title('Valores reales vs Predichos (Regresión Lineal)')
        plt.xlabel('Valores reales')
        plt.ylabel('Valores predichos')
        plt.axline([0, 0], [1, 1], color='red', linestyle='--')
        plt.show()

        return {"rmse": mse ** 0.5, "r2": r2}

    def random_forest(self):
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

        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Visualización
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred)
        plt.title('Valores reales vs Predichos (Random Forest)')
        plt.xlabel('Valores reales')
        plt.ylabel('Valores predichos')
        plt.axline([0, 0], [1, 1], color='red', linestyle='--')
        plt.show()

        return {"rmse": mse ** 0.5, "r2": r2}

    def plot_comparison(self):
        """Grafica una comparación entre ambos modelos."""
        metrics_lr = self.linear_regression()
        metrics_rf = self.random_forest()

        # Comparación de métricas
        metrics = pd.DataFrame({
            "Modelo": ["Regresión Lineal", "Random Forest"],
            "RMSE": [metrics_lr["rmse"], metrics_rf["rmse"]],
            "R²": [metrics_lr["r2"], metrics_rf["r2"]],
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modelo", y="RMSE", data=metrics)
        plt.title('Comparación de RMSE entre Modelos')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modelo", y="R²", data=metrics)
        plt.title('Comparación de R² entre Modelos')
        plt.show()


if __name__ == "__main__":
    from data_loader import download_and_load_data
    train_df, _ = download_and_load_data()

    print("Entrenando modelos...")
    models = Models(train_df)
    models.linear_regression()
    models.random_forest()
    models.plot_comparison()
