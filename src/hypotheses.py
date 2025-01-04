import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
import json  # Para manejar la columna `amenities` de manera segura
from src.data_loader import DataLoader  # Cargar datos desde Kaggle


class Hypothesis:
    def __init__(self, data):
        """Inicializa los datos y los asigna al objeto."""
        self.data = data

    def hypothesis_1(self):
        """Hipótesis 1: El precio promedio por noche es mayor para propiedades completas."""
        price_by_room_type = self.data.groupby('room_type')['log_price'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_room_type, x='room_type', y='log_price')
        plt.title('Precio promedio por tipo de habitación')
        plt.xlabel('Tipo de habitación')
        plt.ylabel('Precio promedio (log)')
        plt.show()

        # ANOVA
        groups = [self.data[self.data['room_type'] == room]['log_price'] for room in self.data['room_type'].unique()]
        anova_result = f_oneway(*groups)
        conclusion = "Diferencias significativas" if anova_result.pvalue < 0.05 else "No hay diferencias significativas"
        return {"p_value": anova_result.pvalue, "conclusion": conclusion}

    def hypothesis_2(self):
        """Hipótesis 2: Las propiedades con calificaciones altas tienen más reservas."""
        self.data['rating_group'] = self.data['review_scores_rating'].apply(
            lambda x: 'Alta (>= 4.5)' if x >= 4.5 else 'Baja (< 4.5)'
        )

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.data, x='rating_group', y='number_of_reviews')
        plt.title('Distribución del número de reservas por grupo de calificación')
        plt.xlabel('Grupo de calificación')
        plt.ylabel('Número de reservas')
        plt.show()

        # t-test
        high_ratings = self.data[self.data['rating_group'] == 'Alta (>= 4.5)']['number_of_reviews']
        low_ratings = self.data[self.data['rating_group'] == 'Baja (< 4.5)']['number_of_reviews']
        t_stat, p_value = ttest_ind(high_ratings, low_ratings, equal_var=False)
        conclusion = "Diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        return {"p_value": p_value, "conclusion": conclusion}

    def hypothesis_3(self):
        """Hipótesis 3: Las propiedades cerca del centro tienen precios más altos."""
        central_areas = ['Downtown', 'Midtown', 'Central Business District']
        self.data['is_central'] = self.data['neighbourhood'].apply(
            lambda x: 'Central' if x in central_areas else 'Periférica'
        )

        price_by_location = self.data.groupby('is_central')['log_price'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_location, x='is_central', y='log_price')
        plt.title('Precio promedio por ubicación')
        plt.xlabel('Ubicación')
        plt.ylabel('Precio promedio (log)')
        plt.show()

        # t-test
        central_prices = self.data[self.data['is_central'] == 'Central']['log_price']
        peripheral_prices = self.data[self.data['is_central'] == 'Periférica']['log_price']
        t_stat, p_value = ttest_ind(central_prices, peripheral_prices, equal_var=False)
        conclusion = "Diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        return {"p_value": p_value, "conclusion": conclusion}

    def hypothesis_4(self):
        """Hipótesis 4: Los anfitriones con múltiples propiedades tienen precios más bajos."""
        host_property_count = self.data['id'].value_counts()
        self.data['num_properties'] = self.data['id'].map(host_property_count)
        self.data['host_type'] = self.data['num_properties'].apply(
            lambda x: 'Múltiples Propiedades' if x > 1 else 'Propiedad Única'
        )

        price_by_host = self.data.groupby('host_type')['log_price'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_host, x='host_type', y='log_price', palette='pastel')
        plt.title('Precio promedio por tipo de anfitrión')
        plt.xlabel('Tipo de anfitrión')
        plt.ylabel('Precio promedio (log)')
        plt.show()

        # t-test
        single_property = self.data[self.data['host_type'] == 'Propiedad Única']['log_price']
        multiple_properties = self.data[self.data['host_type'] == 'Múltiples Propiedades']['log_price']
        t_stat, p_value = ttest_ind(single_property, multiple_properties, equal_var=False)
        conclusion = "Diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        return {"p_value": p_value, "conclusion": conclusion}

    def hypothesis_5(self):
        """Hipótesis 5: Las propiedades con más amenidades tienen calificaciones más altas."""
        # Procesa la columna `amenities` como JSON
        self.data['num_amenities'] = self.data['amenities'].apply(lambda x: len(json.loads(x)))

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.data, x='num_amenities', y='review_scores_rating', alpha=0.7)
        plt.title('Relación entre número de amenidades y calificación promedio')
        plt.xlabel('Número de amenidades')
        plt.ylabel('Calificación promedio')
        plt.show()

        correlation = self.data[['num_amenities', 'review_scores_rating']].corr().iloc[0, 1]
        conclusion = "Relación significativa" if abs(correlation) > 0.1 else "Relación débil o no significativa"
        return {"correlation": correlation, "conclusion": conclusion}


if __name__ == "__main__":
    loader = DataLoader()
    train_df, _ = loader.load_data()

    hypotheses = Hypothesis(train_df)

    print("Resultados Hipótesis 1:", hypotheses.hypothesis_1())
    print("Resultados Hipótesis 2:", hypotheses.hypothesis_2())
    print("Resultados Hipótesis 3:", hypotheses.hypothesis_3())
    print("Resultados Hipótesis 4:", hypotheses.hypothesis_4())
    print("Resultados Hipótesis 5:", hypotheses.hypothesis_5())
