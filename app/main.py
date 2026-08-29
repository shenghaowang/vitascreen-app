from pathlib import Path

import numpy as np
import streamlit as st
from core.model import Model
from core.pdp_utils import THEME, LocalPDPPlotter
from core.questionnaire import Questionnaire
from core.recommendations import top_recommendations

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

show_charts = st.session_state.get("submitted", False)

plotter = LocalPDPPlotter(model, questionnaire.questions, dark=dark)
pdp_list = plotter.compute(x_sample) if show_charts else None

if show_charts:
    risk = model.predict_proba(x_sample)[0][1]
    st.subheader("Your results")
    st.metric("Predicted diabetes risk", f"{risk:.0%}")
    st.caption(
        "This is an estimate from a machine learning model, not a medical "
        "diagnosis. Consult a healthcare professional for medical advice."
    )

    recommendations = top_recommendations(pdp_list)
    if recommendations:
        st.markdown("**What could lower your risk**")
        for rec in recommendations:
            st.markdown(
                f"- {rec['action']} could lower your predicted risk from "
                f"{rec['current_prob']:.0%} to {rec['best_prob']:.0%}."
            )
        st.caption(
            "Each estimate holds every other answer fixed and changes only "
            "that one factor — effects may not simply add up if you change "
            "several things at once."
        )
    else:
        st.markdown(
            "Your answers are already at the lower-risk option for every "
            "factor we can suggest changes for."
        )

    st.divider()

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
        if show_charts:
            fig = plotter.plot_one(pdp_list[index - 1])
            st.plotly_chart(fig, width="stretch", key=f"chart_{question.feature_name}")
        else:
            st.caption(
                "Submit your answers to see how this affects your predicted risk."
            )

    st.divider()

if st.session_state.pop("just_saved", False):
    st.success("Your responses have been saved.")

if st.button("Submit"):
    questionnaire.save(RESPONSES_PATH)
    st.session_state["submitted"] = True
    st.session_state["just_saved"] = True
    st.rerun()
