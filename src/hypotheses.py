import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_ind, f_oneway
import ast  # Para manejar la columna `amenities` de manera segura


class Hypothesis:
    def __init__(self, data):
        """Inicializa los datos y los asigna al objeto."""
        self.data = data

    def hypothesis_1(self):
        """Hipótesis 1: El precio promedio por noche es mayor para propiedades completas."""
        price_by_room_type = self.data.groupby('room_type')['log_price'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=price_by_room_type, x='room_type', y='log_price', ax=ax)
        ax.set_title('Precio promedio por tipo de habitación')
        ax.set_xlabel('Tipo de habitación')
        ax.set_ylabel('Precio promedio (log)')
        st.pyplot(fig)

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

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=self.data, x='rating_group', y='number_of_reviews', ax=ax)
        ax.set_title('Distribución del número de reservas por grupo de calificación')
        ax.set_xlabel('Grupo de calificación')
        ax.set_ylabel('Número de reservas')
        st.pyplot(fig)

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

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=price_by_location, x='is_central', y='log_price', ax=ax)
        ax.set_title('Precio promedio por ubicación')
        ax.set_xlabel('Ubicación')
        ax.set_ylabel('Precio promedio (log)')
        st.pyplot(fig)

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

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=price_by_host, x='host_type', y='log_price', palette='pastel', ax=ax)
        ax.set_title('Precio promedio por tipo de anfitrión')
        ax.set_xlabel('Tipo de anfitrión')
        ax.set_ylabel('Precio promedio (log)')
        st.pyplot(fig)

        # t-test
        single_property = self.data[self.data['host_type'] == 'Propiedad Única']['log_price']
        multiple_properties = self.data[self.data['host_type'] == 'Múltiples Propiedades']['log_price']
        t_stat, p_value = ttest_ind(single_property, multiple_properties, equal_var=False)
        conclusion = "Diferencias significativas" if p_value < 0.05 else "No hay diferencias significativas"
        return {"p_value": p_value, "conclusion": conclusion}

    def hypothesis_5(self):
        """Hipótesis 5: Las propiedades con más amenidades tienen calificaciones más altas."""
        def count_amenities(amenities_str):
            try:
                amenities_list = ast.literal_eval(amenities_str)
                return len(amenities_list)
            except (ValueError, SyntaxError):
                return 0

        if 'amenities' not in self.data.columns:
            return {
                "error": "La columna 'amenities' no está presente en el dataset.",
                "correlation": None,
                "conclusion": "No se puede realizar el análisis."
            }

        self.data['num_amenities'] = self.data['amenities'].apply(count_amenities)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=self.data, x='num_amenities', y='review_scores_rating', alpha=0.7, ax=ax)
        ax.set_title('Relación entre número de amenidades y calificación promedio')
        ax.set_xlabel('Número de amenidades')
        ax.set_ylabel('Calificación promedio')
        st.pyplot(fig)

        correlation = self.data[['num_amenities', 'review_scores_rating']].corr().iloc[0, 1]
        conclusion = "Relación significativa" if abs(correlation) > 0.1 else "Relación débil o no significativa"
        return {"correlation": correlation, "conclusion": conclusion}
