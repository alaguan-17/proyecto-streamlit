import streamlit as st
from src.hypotheses import Hypothesis
from src.data_loader import DataLoader

# Cargar datos
data_loader = DataLoader()
train_df, _ = data_loader.load_data()
hypothesis = Hypothesis(train_df)

# Título y descripción
st.title("Análisis de Hipótesis")
st.markdown("Evalúa las hipótesis relacionadas con los datos de Airbnb.")

# Hipótesis 1
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Hipótesis 1",
    "Hipótesis 2",
    "Hipótesis 3",
    "Hipótesis 4",
    "Hipótesis 5"
])

with tab1:
    st.subheader("Hipótesis 1: El precio promedio es mayor para propiedades completas.")
    result = hypothesis.hypothesis_1()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with tab2:
    st.subheader("Hipótesis 2: Las propiedades con calificaciones altas tienen más reservas.")
    result = hypothesis.hypothesis_2()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with tab3:
    st.subheader("Hipótesis 3: Las propiedades céntricas tienen precios más altos.")
    result = hypothesis.hypothesis_3()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with tab4:
    st.subheader("Hipótesis 4: Los anfitriones con múltiples propiedades tienen precios más bajos.")
    result = hypothesis.hypothesis_4()
    st.write(f"**Resultado:** {result['conclusion']} (p-valor: {result['p_value']:.4f})")

with tab5:
    st.subheader("Hipótesis 5: Más amenidades están asociadas con mejores calificaciones.")
    result = hypothesis.hypothesis_5()
    st.write(f"**Resultado:** {result['conclusion']} (correlación: {result['correlation']:.4f})")

st.markdown("---")
st.success("✨ **Cada hipótesis incluye visualizaciones interactivas y análisis detallados.**")
