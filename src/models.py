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
import streamlit as st


class Models:
    def __init__(self, data, sample_fraction=0.1):
        """Inicializa los datos, toma una muestra significativa y divide en conjuntos de entrenamiento y prueba."""
        self.data = data.sample(frac=sample_fraction, random_state=42)  # Tomar una muestra del dataset
        self.X = self.data[["accommodates", "bathrooms", "bedrooms", "number_of_reviews", "room_type", "property_type"]]
        self.y = self.data["log_price"]
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
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["room_type", "property_type"]),
            ])),
            ("model", LinearRegression()),
        ])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Visualización
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred, ax=ax)
        ax.set_title('Valores reales vs Predichos (Regresión Lineal)')
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Valores predichos')
        ax.axline([0, 0], [1, 1], color='red', linestyle='--')
        st.pyplot(fig)

        return {"rmse": mse ** 0.5, "r2": r2}

    def random_forest(self):
        """Entrena y evalúa un modelo de Random Forest."""
        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]), ["accommodates", "bathrooms", "bedrooms", "number_of_reviews"]),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["room_type", "property_type"]),
            ])),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Visualización
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred, ax=ax)
        ax.set_title('Valores reales vs Predichos (Random Forest)')
        ax.set_xlabel('Valores reales')
        ax.set_ylabel('Valores predichos')
        ax.axline([0, 0], [1, 1], color='red', linestyle='--')
        st.pyplot(fig)

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

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Modelo", y="RMSE", data=metrics, ax=ax)
        ax.set_title('Comparación de RMSE entre Modelos')
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Modelo", y="R²", data=metrics, ax=ax)
        ax.set_title('Comparación de R² entre Modelos')
        st.pyplot(fig)


# Ejecución
if __name__ == "__main__":
    from src.data_loader import DataLoader
    data_loader = DataLoader()
    train_df, _ = data_loader.load_data()

    st.title("Comparativa de Modelos de Machine Learning")
    st.markdown("Analizamos dos modelos: Regresión Lineal y Random Forest, evaluando su desempeño en la predicción de precios de Airbnb.")
    models = Models(train_df, sample_fraction=0.1)  # Tomar el 10% del dataset
    models.plot_comparison()
