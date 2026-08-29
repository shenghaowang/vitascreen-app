from pathlib import Path

import numpy as np
import streamlit as st
from core.model import Model
from core.pdp_utils import THEME, LocalPDPPlotter
from core.questionnaire import Questionnaire

st.set_page_config(page_title="VitaScreen", page_icon="🩺", layout="wide")

st.title("VitaScreen")
st.write(
    "Fill in the questionnaire to get predictions from pretrained ML models, "
    "and interpret the results with SHAP and partial dependence plots."
)

QUESTIONNAIRE_PATH = Path(__file__).parent / "schemas" / "questionnaire.yaml"
RESPONSES_PATH = Path(__file__).parent.parent / "data" / "responses.yaml"
MODEL_PATH = Path(__file__).parent.parent / "models" / "cb_f21_enn_fe.cbm"

questionnaire = Questionnaire.load(QUESTIONNAIRE_PATH)


@st.cache_resource
def load_model() -> Model:
    return Model.load(MODEL_PATH)


model = load_model()
dark = st.context.theme.type == "dark"
theme = THEME["dark"] if dark else THEME["light"]

# Snapshot every question's current answer (from the last rerun, defaulting to
# the first option) so we can compute each feature's local PDP curve against
# the *current* full set of answers before this run's widgets are (re)drawn.
snapshot = {
    question.feature_name: question.possible_answers[
        st.session_state.get(
            f"answer_{question.feature_name}", next(iter(question.possible_answers))
        )
    ].value
    for question in questionnaire.questions
}
model.validate_feature_names(snapshot.keys())
x_sample = np.array([[snapshot[name] for name in model.feature_names()]], dtype=float)

plotter = LocalPDPPlotter(model, questionnaire.questions, dark=dark)
pdp_list = plotter.compute(x_sample)

st.markdown(
    f'<span style="color:{theme["line"]}">●</span> Predicted risk as this answer changes'
    f'&nbsp;&nbsp;&nbsp;<span style="color:{theme["highlight"]}">●</span> Your answer',
    unsafe_allow_html=True,
)

for index, question in enumerate(questionnaire.questions, start=1):
    col_question, col_chart = st.columns([2, 1])

    with col_question:
        answer_key = st.radio(
            f"{index}. {question.text}",
            options=question.possible_answers.keys(),
            format_func=lambda key, possible_answers=question.possible_answers: (
                possible_answers[key].text
            ),
            key=f"answer_{question.feature_name}",
        )
        questionnaire.set_response(question.text, answer_key)

    with col_chart:
        fig = plotter.plot_one(pdp_list[index - 1])
        st.plotly_chart(fig, width="stretch", key=f"chart_{question.feature_name}")

    st.divider()

if st.button("Submit"):
    questionnaire.save(RESPONSES_PATH)
    st.success("Your responses have been saved.")
