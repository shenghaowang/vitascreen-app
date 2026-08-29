import streamlit as st

st.set_page_config(page_title="VitaScreen", page_icon="🩺", layout="wide")

st.title("VitaScreen")
st.write(
    "Fill in the questionnaire to get predictions from pretrained ML models, "
    "and interpret the results with SHAP and partial dependence plots."
)
