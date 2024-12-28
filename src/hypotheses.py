#Hipótesis 1: El precio promedio por noche es mayor para propiedades completas.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Cálculo del precio promedio por tipo de propiedad
price_by_room_type = train_df.groupby('room_type')['log_price'].mean().reset_index()

# Visualización
plt.figure(figsize=(8, 5))
sns.barplot(data=price_by_room_type, x='room_type', y='log_price')
plt.title('Precio promedio por tipo de habitación')
plt.xlabel('Tipo de habitación')
plt.ylabel('Precio promedio (log)')
plt.show()

# Prueba ANOVA para comparar las medias
groups = [train_df[train_df['room_type'] == room]['log_price'] for room in train_df['room_type'].unique()]
anova_result = f_oneway(*groups)
print(f"Resultado de ANOVA: p-value = {anova_result.pvalue}")
if anova_result.pvalue < 0.05:
    print("Hay diferencias significativas en los precios promedio entre los tipos de habitación.")
else:
    print("No hay diferencias significativas en los precios promedio entre los tipos de habitación.")

#Hipótesis 2: Los listados con calificaciones altas tienen más reservas.
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Dividir en grupos por calificación
high_ratings = train_df[train_df['review_scores_rating'] >= 4.5]['number_of_reviews']
low_ratings = train_df[train_df['review_scores_rating'] < 4.5]['number_of_reviews']

# Estadísticas descriptivas
print(f"Promedio de reservas (alta calificación): {high_ratings.mean()}")
print(f"Promedio de reservas (baja calificación): {low_ratings.mean()}")

# Prueba t
t_stat, p_value = ttest_ind(high_ratings, low_ratings, equal_var=False)
print(f"Prueba t: p-value = {p_value}")
if p_value < 0.05:
    print("Hay diferencias significativas en el número de reservas entre los dos grupos.")
else:
    print("No hay diferencias significativas en el número de reservas entre los dos grupos.")

# Agregar una columna para identificar calificaciones altas y bajas
train_df['rating_group'] = train_df['review_scores_rating'].apply(
    lambda x: 'Alta (>= 4.5)' if x >= 4.5 else 'Baja (< 4.5)'
)

# Gráfico de cajas (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=train_df, x='rating_group', y='number_of_reviews')
plt.title('Distribución del número de reservas por grupo de calificación')
plt.xlabel('Grupo de calificación')
plt.ylabel('Número de reservas')
plt.show()

# Gráfico de barras con promedio
avg_reservations = train_df.groupby('rating_group')['number_of_reviews'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=avg_reservations, x='rating_group', y='number_of_reviews', palette='pastel')
plt.title('Promedio de reservas por grupo de calificación')
plt.xlabel('Grupo de calificación')
plt.ylabel('Número promedio de reservas')
plt.show()


#Hipótesis 3: Las propiedades cerca del centro tienen precios más altos.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Clasificación manual: Vecindarios céntricos vs periféricos
central_areas = ['Downtown', 'Midtown', 'Central Business District']  # Ajusta según tu dataset
train_df['is_central'] = train_df['neighbourhood'].apply(
    lambda x: 'Central' if x in central_areas else 'Periférica'
)

# Verificar distribución de datos
print(train_df['is_central'].value_counts())

# Precio promedio por ubicación
price_by_location = train_df.groupby('is_central')['log_price'].mean().reset_index()

# Visualización
plt.figure(figsize=(8, 5))
sns.barplot(data=price_by_location, x='is_central', y='log_price')
plt.title('Precio promedio por ubicación')
plt.xlabel('Ubicación')
plt.ylabel('Precio promedio (log)')
plt.show()

# Prueba t para comparar precios entre ubicaciones
central_prices = train_df[train_df['is_central'] == 'Central']['log_price']
peripheral_prices = train_df[train_df['is_central'] == 'Periférica']['log_price']
t_stat, p_value = ttest_ind(central_prices, peripheral_prices, equal_var=False)

print(f"Prueba t: p-value = {p_value}")
if p_value < 0.05:
    print("Hay diferencias significativas en los precios entre propiedades céntricas y periféricas.")
else:
    print("No hay diferencias significativas en los precios entre propiedades céntricas y periféricas.")


#Hipótesis 4: Los anfitriones con múltiples propiedades listadas tienen precios más bajos.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Verificar si la columna 'id' existe y contar las propiedades por anfitrión
if 'id' in train_df.columns:
    # Contar el número de propiedades por anfitrión
    host_property_count = train_df['id'].value_counts()
    train_df['num_properties'] = train_df['id'].map(host_property_count)

    # Clasificar anfitriones
    train_df['host_type'] = train_df['num_properties'].apply(
        lambda x: 'Múltiples Propiedades' if x > 1 else 'Propiedad Única'
    )

    # Verificar la cantidad de registros en cada grupo
    group_counts = train_df['host_type'].value_counts()
    print("Distribución de registros por tipo de anfitrión:")
    print(group_counts)

    # Verificar si ambos grupos tienen suficientes datos
    if all(group_counts >= 30):  # Al menos 30 registros por grupo
        # Precio promedio por tipo de anfitrión
        price_by_host = train_df.groupby('host_type')['log_price'].mean().reset_index()

        # Visualización del precio promedio
        plt.figure(figsize=(8, 5))
        sns.barplot(data=price_by_host, x='host_type', y='log_price', palette='pastel')
        plt.title('Precio promedio por tipo de anfitrión')
        plt.xlabel('Tipo de anfitrión')
        plt.ylabel('Precio promedio (log)')
        plt.show()

        # Prueba t
        single_property = train_df[train_df['host_type'] == 'Propiedad Única']['log_price']
        multiple_properties = train_df[train_df['host_type'] == 'Múltiples Propiedades']['log_price']
        t_stat, p_value = ttest_ind(single_property, multiple_properties, equal_var=False)

        print(f"Prueba t: p-value = {p_value}")
        if p_value < 0.05:
            print("Hay diferencias significativas en los precios entre anfitriones con una y múltiples propiedades.")
        else:
            print("No hay diferencias significativas en los precios entre los grupos.")
    else:
        print("No hay suficientes registros en uno o ambos grupos para realizar la prueba t.")
else:
    print("La columna 'id' no existe en el dataset. Verifica los datos o proporciona una columna alternativa.")

# Inspeccionar la distribución de la columna 'num_properties'
if 'num_properties' in train_df.columns:
    # Verificar la cantidad de propiedades por anfitrión
    host_property_count = train_df['id'].value_counts()
    print("Distribución del número de propiedades por anfitrión:")
    print(host_property_count.describe())  # Resumen estadístico
    print(host_property_count.value_counts())  # Conteo por número de propiedades

    # Contar los registros clasificados como 'Múltiples Propiedades'
    multiple_properties_count = (host_property_count > 1).sum()
    print(f"Total de anfitriones con múltiples propiedades: {multiple_properties_count}")
else:
    print("La columna 'num_properties' no se encuentra en el dataset.")

#Hipótesis 5: Las propiedades con más amenidades tienen calificaciones más altas.
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# Asegurarnos de que 'amenities' está en el dataset
if 'amenities' in train_df.columns:
    # Contar el número de amenidades
    def count_amenities(amenities_str):
        try:
            # Convertir el texto en una lista utilizando ast.literal_eval
            amenities_list = ast.literal_eval(amenities_str)
            return len(amenities_list)
        except (ValueError, SyntaxError):
            return 0  # Si hay algún error, retornamos 0

    train_df['num_amenities'] = train_df['amenities'].apply(count_amenities)

    # Visualización de la relación entre número de amenidades y calificación
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=train_df, x='num_amenities', y='review_scores_rating', alpha=0.7)
    plt.title('Relación entre número de amenidades y calificación promedio')
    plt.xlabel('Número de amenidades')
    plt.ylabel('Calificación promedio')
    plt.show()

    # Calcular la correlación
    correlation = train_df[['num_amenities', 'review_scores_rating']].corr()
    print(f"Correlación entre número de amenidades y calificación promedio: {correlation.loc['num_amenities', 'review_scores_rating']}")
else:
    print("La columna 'amenities' no está presente en el dataset.")