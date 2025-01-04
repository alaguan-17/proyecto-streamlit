import streamlit as st
from src.data_loader import DataLoader
from src.hypotheses import Hypothesis

# Título y descripción
st.title("💡 Análisis de Hipótesis")
st.markdown("Exploramos hipótesis clave relacionadas con los precios y características de propiedades en Airbnb.")

# Cargar datos
@st.cache_data
def get_data():
    loader = DataLoader()
    return loader.load_data()

train_df, test_df = get_data()

# Análisis de hipótesis
hypothesis = Hypothesis(train_df)

st.markdown("🎯 **Hipótesis Presentadas:**")

with st.expander("1️⃣ El precio promedio es mayor para propiedades completas."):
    result = hypothesis.hypothesis_1()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with st.expander("2️⃣ Las propiedades con calificaciones altas tienen más reservas."):
    result = hypothesis.hypothesis_2()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with st.expander("3️⃣ Las propiedades céntricas tienen precios más altos."):
    result = hypothesis.hypothesis_3()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with st.expander("4️⃣ Los anfitriones con múltiples propiedades tienen precios más bajos."):
    result = hypothesis.hypothesis_4()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with st.expander("5️⃣ Más amenidades están asociadas con mejores calificaciones."):
    result = hypothesis.hypothesis_5()
    st.write(f"**Resultado:** {result['conclusion']} (correlación: {result['correlation']:.4f})")
