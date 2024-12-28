import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import ast

def analyze_hypothesis_1(train_df):
    """Analiza la Hipótesis 1: El precio promedio por noche es mayor para propiedades completas."""
    # Cálculo del precio promedio por tipo de propiedad
    price_by_room_type = train_df.groupby('room_type')['log_price'].mean().reset_index()

    # Visualización
    def plot_hypothesis_1():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_room_type, x='room_type', y='log_price')
        plt.title('Precio promedio por tipo de habitación')
        plt.xlabel('Tipo de habitación')
        plt.ylabel('Precio promedio (log)')
        plt.show()

    # Prueba ANOVA para comparar las medias
    groups = [train_df[train_df['room_type'] == room]['log_price'] for room in train_df['room_type'].unique()]
    anova_result = f_oneway(*groups)

    return price_by_room_type, anova_result, plot_hypothesis_1

def analyze_hypothesis_2(train_df):
    """Analiza la Hipótesis 2: Los listados con calificaciones altas tienen más reservas."""
    # Dividir en grupos por calificación
    high_ratings = train_df[train_df['review_scores_rating'] >= 4.5]['number_of_reviews']
    low_ratings = train_df[train_df['review_scores_rating'] < 4.5]['number_of_reviews']

    # Prueba t
    t_stat, p_value = ttest_ind(high_ratings, low_ratings, equal_var=False)

    # Agregar una columna para identificar calificaciones altas y bajas
    train_df['rating_group'] = train_df['review_scores_rating'].apply(
        lambda x: 'Alta (>= 4.5)' if x >= 4.5 else 'Baja (< 4.5)'
    )

    # Visualización
    def plot_hypothesis_2():
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=train_df, x='rating_group', y='number_of_reviews')
        plt.title('Distribución del número de reservas por grupo de calificación')
        plt.xlabel('Grupo de calificación')
        plt.ylabel('Número de reservas')
        plt.show()

        avg_reservations = train_df.groupby('rating_group')['number_of_reviews'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=avg_reservations, x='rating_group', y='number_of_reviews', palette='pastel')
        plt.title('Promedio de reservas por grupo de calificación')
        plt.xlabel('Grupo de calificación')
        plt.ylabel('Número promedio de reservas')
        plt.show()

    return t_stat, p_value, plot_hypothesis_2

def analyze_hypothesis_3(train_df):
    """Analiza la Hipótesis 3: Las propiedades cerca del centro tienen precios más altos."""
    # Clasificación manual: Vecindarios céntricos vs periféricos
    central_areas = ['Downtown', 'Midtown', 'Central Business District']  # Ajusta según tu dataset
    train_df['is_central'] = train_df['neighbourhood'].apply(
        lambda x: 'Central' if x in central_areas else 'Periférica'
    )

    # Precio promedio por ubicación
    price_by_location = train_df.groupby('is_central')['log_price'].mean().reset_index()

    # Prueba t para comparar precios entre ubicaciones
    central_prices = train_df[train_df['is_central'] == 'Central']['log_price']
    peripheral_prices = train_df[train_df['is_central'] == 'Periférica']['log_price']
    t_stat, p_value = ttest_ind(central_prices, peripheral_prices, equal_var=False)

    # Visualización
    def plot_hypothesis_3():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_location, x='is_central', y='log_price')
        plt.title('Precio promedio por ubicación')
        plt.xlabel('Ubicación')
        plt.ylabel('Precio promedio (log)')
        plt.show()

    return price_by_location, t_stat, p_value, plot_hypothesis_3

def analyze_hypothesis_4(train_df):
    """Analiza la Hipótesis 4: Los anfitriones con múltiples propiedades listadas tienen precios más bajos."""
    if 'id' in train_df.columns:
        # Contar el número de propiedades por anfitrión
        host_property_count = train_df['id'].value_counts()
        train_df['num_properties'] = train_df['id'].map(host_property_count)

        # Clasificar anfitriones
        train_df['host_type'] = train_df['num_properties'].apply(
            lambda x: 'Múltiples Propiedades' if x > 1 else 'Propiedad Única'
        )

        # Precio promedio por tipo de anfitrión
        price_by_host = train_df.groupby('host_type')['log_price'].mean().reset_index()

        # Prueba t
        single_property = train_df[train_df['host_type'] == 'Propiedad Única']['log_price']
        multiple_properties = train_df[train_df['host_type'] == 'Múltiples Propiedades']['log_price']
        t_stat, p_value = ttest_ind(single_property, multiple_properties, equal_var=False)

        # Visualización
        def plot_hypothesis_4():
            plt.figure(figsize=(8, 5))
            sns.barplot(data=price_by_host, x='host_type', y='log_price', palette='pastel')
            plt.title('Precio promedio por tipo de anfitrión')
            plt.xlabel('Tipo de anfitrión')
            plt.ylabel('Precio promedio (log)')
            plt.show()

        return price_by_host, t_stat, p_value, plot_hypothesis_4
    else:
        return None, None, None, None

def analyze_hypothesis_5(train_df):
    """Analiza la Hipótesis 5: Las propiedades con más amenidades tienen calificaciones más altas."""
    if 'amenities' in train_df.columns:
        # Contar el número de amenidades
        def count_amenities(amenities_str):
            try:
                amenities_list = ast.literal_eval(amenities_str)
                return len(amenities_list)
            except (ValueError, SyntaxError):
                return 0

        train_df['num_amenities'] = train_df['amenities'].apply(count_amenities)

        # Calcular la correlación
        correlation = train_df[['num_amenities', 'review_scores_rating']].corr()

        # Visualización
        def plot_hypothesis_5():
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=train_df, x='num_amenities', y='review_scores_rating', alpha=0.7)
            plt.title('Relación entre número de amenidades y calificación promedio')
            plt.xlabel('Número de amenidades')
            plt.ylabel('Calificación promedio')
            plt.show()

        return correlation, plot_hypothesis_5
    else:
        return None, None