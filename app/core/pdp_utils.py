from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from core.model import Model
from core.questionnaire import Question

_FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Validated categorical slots 1 (line) and 2 (highlight) from the design system
# palette; see references/palette.md in the dataviz skill.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "ink": "#0b0b0b",
        "muted_ink": "#52514e",
        "line": "#2a78d6",
        "highlight": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "ink": "#ffffff",
        "muted_ink": "#c3c2b7",
        "line": "#3987e5",
        "highlight": "#d95926",
    },
}


class LocalPDPPlotter:
    """Local (single-sample) partial dependence for a trained model.

    Sweeps one feature at a time across its possible values, holding the
    rest of a single sample fixed. Each feature's curve shows how the
    predicted probability would change had that one answer been different,
    with the user's actual answer highlighted on the curve.
    """

    def __init__(
        self, model: Model, questions: list[Question], dark: bool = False
    ) -> None:
        self.model = model
        self.questions = questions
        self.theme = THEME["dark"] if dark else THEME["light"]

    def question_value_range(self, idx: int) -> list[int]:
        """Sorted, distinct answer values for question idx."""
        possible_answers = self.questions[idx].possible_answers
        values = sorted({answer.value for answer in possible_answers.values()})
        return values or [0, 1]

    def local_pdp(self, x_sample, feature_name, feature_index, feature_values):
        """
        Local PDP (one-sample PDP)

        Parameters
        ----------
        x_sample : np.ndarray with exactly one row
        feature_name : name of the feature to vary
        feature_index : index of the feature to vary
        feature_values : iterable of values for the feature

        Returns
        -------
        pd.DataFrame
        """
        selected_value = x_sample[0, feature_index]
        preds = []

        for value in feature_values:
            x = x_sample.copy()
            x[0, feature_index] = value

            proba = self.model.predict_proba(x)[0][1]  # probability of positive class
            preds.append(proba)

        return pd.DataFrame(
            {
                "feature_value": feature_values,
                "feature_name": feature_name,
                "Probability": preds,
                "is_selected": [value == selected_value for value in feature_values],
            }
        )

    def compute(self, x_sample):
        """Compute local PDPs for every feature given a single-row sample."""
        feature_names = self.model.feature_names()

        return [
            self.local_pdp(
                x_sample=x_sample,
                feature_index=idx,
                feature_name=(
                    feature_names[idx] if idx < len(feature_names) else f"Q{idx + 1}"
                ),
                feature_values=self.question_value_range(idx),
            )
            for idx in range(len(self.questions))
        ]

    def plot_one(self, pdp_df: pd.DataFrame) -> go.Figure:
        """Render a single feature's local PDP curve as a compact standalone figure."""
        theme = self.theme
        feature_name = pdp_df["feature_name"].iloc[0]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pdp_df["feature_value"],
                y=pdp_df["Probability"],
                mode="lines+markers",
                line={"color": theme["line"], "width": 2},
                marker={"size": 8, "color": theme["line"]},
                hovertemplate=f"{feature_name} = %{{x}}<br>Predicted risk: %{{y:.1%}}<extra></extra>",
            )
        )

        selected = pdp_df[pdp_df["is_selected"]]
        fig.add_trace(
            go.Scatter(
                x=selected["feature_value"],
                y=selected["Probability"],
                mode="markers",
                marker={
                    "size": 13,
                    "color": theme["highlight"],
                    "line": {"width": 2, "color": theme["surface"]},
                },
                hovertemplate=f"{feature_name} = %{{x}} (your answer)<br>Predicted risk: %{{y:.1%}}<extra></extra>",
            )
        )

        y_min, y_max = pdp_df["Probability"].min(), pdp_df["Probability"].max()
        pad = max((y_max - y_min) * 0.25, 0.03)
        y_range = [max(0.0, y_min - pad), min(1.0, y_max + pad)]

        fig.update_xaxes(
            tickvals=pdp_df["feature_value"].tolist(),
            showgrid=False,
            zeroline=False,
            linecolor=theme["axis"],
            tickfont={"size": 10, "color": theme["muted_ink"]},
        )
        fig.update_yaxes(
            range=y_range,
            nticks=4,
            tickformat=".0%",
            showgrid=True,
            gridcolor=theme["grid"],
            gridwidth=1,
            zeroline=False,
            tickfont={"size": 10, "color": theme["muted_ink"]},
        )
        fig.update_layout(
            height=180,
            margin={"t": 10, "b": 30, "l": 10, "r": 10},
            plot_bgcolor=theme["surface"],
            paper_bgcolor=theme["surface"],
            font={"family": _FONT_FAMILY, "color": theme["ink"]},
            showlegend=False,
        )

        return fig
