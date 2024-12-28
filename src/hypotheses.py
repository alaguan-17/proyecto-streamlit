# Instalamos kagglehub e importamos pandas
!pip install kagglehub
import kagglehub
import pandas as pd
import os

# Descargamos el dataset usando kagglehub
path = kagglehub.dataset_download("rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml")

print("Path to dataset files:", path)

# Listamos todos los archivos descargados
files = os.listdir(path)
print("Archivos descargados:", files)

# Identificamos los archivos CSV correspondientes
train_file = os.path.join(path, "train.csv")
test_file = os.path.join(path, "test.csv")

# Cargamos los archivos en DataFrames
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Mostramos las primeras filas de cada archivo
print("Primeras filas del dataset de entrenamiento:")
print(train_df.head())

print("\nPrimeras filas del dataset de prueba:")
print(test_df.head())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import ast
import streamlit as st

def analyze_hypothesis_1(train_df):
    """Analiza la Hipótesis 1: El precio promedio por noche es mayor para propiedades completas."""
    price_by_room_type = train_df.groupby('room_type')['log_price'].mean().reset_index()
    groups = [train_df[train_df['room_type'] == room]['log_price'] for room in train_df['room_type'].unique()]
    anova_result = f_oneway(*groups)

    def plot_hypothesis_1():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_room_type, x='room_type', y='log_price')
        plt.title('Precio promedio por tipo de habitación')
        plt.xlabel('Tipo de habitación')
        plt.ylabel('Precio promedio (log)')
        st.pyplot(plt)

    return price_by_room_type, anova_result, plot_hypothesis_1

def analyze_hypothesis_2(train_df):
    high_ratings = train_df[train_df['review_scores_rating'] >= 4.5]['number_of_reviews']
    low_ratings = train_df[train_df['review_scores_rating'] < 4.5]['number_of_reviews']
    t_stat, p_value = ttest_ind(high_ratings, low_ratings, equal_var=False)
    train_df['rating_group'] = train_df['review_scores_rating'].apply(
        lambda x: 'Alta (>= 4.5)' if x >= 4.5 else 'Baja (< 4.5)'
    )

    def plot_hypothesis_2():
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=train_df, x='rating_group', y='number_of_reviews')
        plt.title('Distribución del número de reservas por grupo de calificación')
        st.pyplot(plt)

        avg_reservations = train_df.groupby('rating_group')['number_of_reviews'].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=avg_reservations, x='rating_group', y='number_of_reviews', palette='pastel')
        plt.title('Promedio de reservas por grupo de calificación')
        st.pyplot(plt)

    return t_stat, p_value, plot_hypothesis_2

def analyze_hypothesis_3(train_df):
    central_areas = ['Downtown', 'Midtown', 'Central Business District']
    train_df['is_central'] = train_df['neighbourhood'].apply(
        lambda x: 'Central' if x in central_areas else 'Periférica'
    )
    price_by_location = train_df.groupby('is_central')['log_price'].mean().reset_index()
    central_prices = train_df[train_df['is_central'] == 'Central']['log_price']
    peripheral_prices = train_df[train_df['is_central'] == 'Periférica']['log_price']
    t_stat, p_value = ttest_ind(central_prices, peripheral_prices, equal_var=False)

    def plot_hypothesis_3():
        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_location, x='is_central', y='log_price')
        plt.title('Precio promedio por ubicación')
        st.pyplot(plt)

    return price_by_location, t_stat, p_value, plot_hypothesis_3

def analyze_hypothesis_4(train_df):
    if 'id' in train_df.columns:
        host_property_count = train_df['id'].value_counts()
        train_df['num_properties'] = train_df['id'].map(host_property_count)
        train_df['host_type'] = train_df['num_properties'].apply(
            lambda x: 'Múltiples Propiedades' if x > 1 else 'Propiedad Única'
        )
        price_by_host = train_df.groupby('host_type')['log_price'].mean().reset_index()
        single_property = train_df[train_df['host_type'] == 'Propiedad Única']['log_price']
        multiple_properties = train_df[train_df['host_type'] == 'Múltiples Propiedades']['log_price']
        t_stat, p_value = ttest_ind(single_property, multiple_properties, equal_var=False)

        def plot_hypothesis_4():
            plt.figure(figsize=(8, 5))
            sns.barplot(data=price_by_host, x='host_type', y='log_price', palette='pastel')
            plt.title('Precio promedio por tipo de anfitrión')
            st.pyplot(plt)

        return price_by_host, t_stat, p_value, plot_hypothesis_4
    else:
        return None, None, None, None

def analyze_hypothesis_5(train_df):
    if 'amenities' in train_df.columns:
        def count_amenities(amenities_str):
            try:
                amenities_list = ast.literal_eval(amenities_str)
                return len(amenities_list)
            except (ValueError, SyntaxError):
                return 0

        train_df['num_amenities'] = train_df['amenities'].apply(count_amenities)
        correlation = train_df[['num_amenities', 'review_scores_rating']].corr()

        def plot_hypothesis_5():
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=train_df, x='num_amenities', y='review_scores_rating', alpha=0.7)
            plt.title('Relación entre número de amenidades y calificación promedio')
            st.pyplot(plt)

        return correlation, plot_hypothesis_5
    else:
        return None, None

def display(train_df):
    st.header("Análisis de Hipótesis")

    st.subheader("Hipótesis 1")
    _, anova_result, plot_hypothesis_1 = analyze_hypothesis_1(train_df)
    st.write(f"Resultado ANOVA: p-value = {anova_result.pvalue}")
    plot_hypothesis_1()

    st.subheader("Hipótesis 2")
    t_stat, p_value, plot_hypothesis_2 = analyze_hypothesis_2(train_df)
    st.write(f"Prueba t: p-value = {p_value}")
    plot_hypothesis_2()

    st.subheader("Hipótesis 3")
    _, t_stat, p_value, plot_hypothesis_3 = analyze_hypothesis_3(train_df)
    st.write(f"Prueba t: p-value = {p_value}")
    plot_hypothesis_3()

    st.subheader("Hipótesis 4")
    _, t_stat, p_value, plot_hypothesis_4 = analyze_hypothesis_4(train_df)
    if t_stat is not None:
        st.write(f"Prueba t: p-value = {p_value}")
        plot_hypothesis_4()
    else:
        st.write("No se encontraron datos suficientes para esta hipótesis.")

    st.subheader("Hipótesis 5")
    correlation, plot_hypothesis_5 = analyze_hypothesis_5(train_df)
    if correlation is not None:
        st.write("Correlación entre número de amenidades y calificación promedio:")
        st.write(correlation)
        plot_hypothesis_5()
    else:
        st.write("No se encontraron datos suficientes para esta hipótesis.")
