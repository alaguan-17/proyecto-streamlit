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
        self.data = data.sample(frac=sample_fraction, random_state=42)
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

        # Gráfico 1: Valores reales vs predichos
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred, ax=ax1)
        ax1.set_title('Valores reales vs Predichos (Regresión Lineal)')
        ax1.set_xlabel('Valores reales')
        ax1.set_ylabel('Valores predichos')
        ax1.axline([0, 0], [1, 1], color='red', linestyle='--')

        # Gráfico 2: Distribución de errores
        residuals = self.y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax2, color="blue")
        ax2.set_title('Distribución de Errores (Regresión Lineal)')
        ax2.set_xlabel('Errores')
        ax2.set_ylabel('Frecuencia')

        return {"rmse": mse ** 0.5, "r2": r2, "figures": [fig1, fig2], "predictions": y_pred}

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

        # Gráfico 1: Valores reales vs predichos
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=y_pred, ax=ax1)
        ax1.set_title('Valores reales vs Predichos (Random Forest)')
        ax1.set_xlabel('Valores reales')
        ax1.set_ylabel('Valores predichos')
        ax1.axline([0, 0], [1, 1], color='red', linestyle='--')

        # Gráfico 2: Distribución de errores
        residuals = self.y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax2, color="green")
        ax2.set_title('Distribución de Errores (Random Forest)')
        ax2.set_xlabel('Errores')
        ax2.set_ylabel('Frecuencia')

        return {"rmse": mse ** 0.5, "r2": r2, "figures": [fig1, fig2], "predictions": y_pred}

    def plot_comparison(self):
        """Comparación de modelos."""
        metrics_lr = self.linear_regression()
        metrics_rf = self.random_forest()

        # Mostrar gráficos en dos columnas
        st.header("Comparación de Gráficos")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Regresión Lineal")
            for fig in metrics_lr["figures"]:
                st.pyplot(fig)
        with col2:
            st.subheader("Random Forest")
            for fig in metrics_rf["figures"]:
                st.pyplot(fig)

        # Tabla comparativa de predicciones
        st.header("Comparación de Valores Reales vs Predichos")
        comparison_df = pd.DataFrame({
            "Valores Reales": self.y_test,
            "Predicciones Regresión Lineal": metrics_lr["predictions"],
            "Predicciones Random Forest": metrics_rf["predictions"],
        }).reset_index(drop=True)
        st.dataframe(comparison_df)

        return {
            "metrics_lr": metrics_lr,
            "metrics_rf": metrics_rf
        }


# Ejecución principal
if __name__ == "__main__":
    from src.data_loader import DataLoader
    data_loader = DataLoader()
    train_df, _ = data_loader.load_data()

    st.title("Comparativa de Modelos de Machine Learning")
    st.markdown("Analizamos dos modelos: Regresión Lineal y Random Forest, evaluando su desempeño en la predicción de precios de Airbnb.")

    models = Models(train_df, sample_fraction=0.1)
    models.plot_comparison()
