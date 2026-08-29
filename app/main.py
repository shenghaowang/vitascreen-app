from pathlib import Path

import streamlit as st
from core.questionnaire import Questionnaire

st.set_page_config(page_title="VitaScreen", page_icon="🩺", layout="wide")

st.title("VitaScreen")
st.write(
    "Fill in the questionnaire to get predictions from pretrained ML models, "
    "and interpret the results with SHAP and partial dependence plots."
)

QUESTIONNAIRE_PATH = Path(__file__).parent / "schemas" / "questionnaire.yaml"
RESPONSES_PATH = Path(__file__).parent.parent / "data" / "responses.yaml"

questionnaire = Questionnaire.load(QUESTIONNAIRE_PATH)

for index, question in enumerate(questionnaire.questions, start=1):
    answer_key = st.radio(
        f"{index}. {question.text}",
        options=question.possible_answers.keys(),
        format_func=lambda key, possible_answers=question.possible_answers: (
            possible_answers[key].text
        ),
    )
    questionnaire.set_response(question.text, answer_key)

if st.button("Submit"):
    questionnaire.save(RESPONSES_PATH)
    st.success("Your responses have been saved.")
