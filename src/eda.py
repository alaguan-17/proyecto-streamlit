import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data):
        self.data = data
def plot_price_distribution(train_df):
    """Grafica la distribución del precio."""
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['log_price'], kde=True)
    plt.title('Distribución del precio')
    plt.xlabel('Precio')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_room_type_distribution(train_df):
    """Grafica la distribución por tipo de habitación."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='room_type', data=train_df)
    plt.title('Distribución de tipos de habitación')
    plt.xlabel('Tipo de habitación')
    plt.ylabel('Frecuencia')
    plt.show()

def plot_correlation_matrix(train_df):
    """Grafica la matriz de correlación entre variables numéricas."""
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de calor de la correlación entre variables')
    plt.show()

if __name__ == "__main__":
    from data_loader import download_and_load_data
    train_df, _ = download_and_load_data()

    print("Realizando EDA...")
    plot_price_distribution(train_df)
    plot_room_type_distribution(train_df)
    plot_correlation_matrix(train_df)