from __future__ import annotations

import pandas as pd

# Feature name -> short phrase describing the action that lowers risk when
# the answer moves toward that feature's most favorable option. Only
# behavioral/lifestyle features with a plausible causal effect are listed
# here; diagnosed conditions (e.g. HighBP, Stroke) and fixed traits (e.g.
# Age, Sex) are excluded since there's no self-directed action to suggest.
# Healthcare-utilization features (CholCheck, AnyHealthcare) are also
# excluded: in observational survey data, getting screened/insured tends to
# correlate with *already being* at elevated risk, so a model trained on
# this data can learn that association backwards — recommending "skip your
# checkup" would be unsafe, even if the model's local PDP shows it.
ACTIONABLE_FEATURES = {
    "Smoker": "Quitting smoking",
    "PhysActivity": "Adding regular physical activity",
    "Fruits": "Eating fruit daily",
    "Veggies": "Eating vegetables daily",
    "HvyAlcoholConsump": "Cutting back on heavy alcohol consumption",
    "BMI": "Moving toward a healthier BMI range",
}

MIN_DELTA = 0.005  # ignore effects smaller than half a percentage point


def top_recommendations(pdp_list: list[pd.DataFrame], max_items: int = 3) -> list[dict]:
    """Rank actionable features by how much predicted risk could drop if the
    user's answer moved to that feature's most favorable option, holding
    every other answer fixed.

    Returns up to `max_items` dicts with `action`, `current_prob`, and
    `best_prob`, most impactful first. Each effect is isolated (one feature
    at a time) and not guaranteed to add up if several answers changed together.
    """
    candidates = []

    for pdp_df in pdp_list:
        feature_name = pdp_df["feature_name"].iloc[0]
        action = ACTIONABLE_FEATURES.get(feature_name)
        if action is None:
            continue

        current_row = pdp_df[pdp_df["is_selected"]].iloc[0]
        best_row = pdp_df.loc[pdp_df["Probability"].idxmin()]
        delta = current_row["Probability"] - best_row["Probability"]

        # Also require the two probabilities to round to different whole
        # percentage points, since that's the precision they're displayed
        # at (`.0%`) — otherwise a real-but-small delta reads as a
        # recommendation showing no change, e.g. "7% to 7%".
        rounds_differently = round(current_row["Probability"] * 100) != round(
            best_row["Probability"] * 100
        )

        if delta > MIN_DELTA and rounds_differently:
            candidates.append(
                {
                    "action": action,
                    "current_prob": current_row["Probability"],
                    "best_prob": best_row["Probability"],
                    "delta": delta,
                }
            )

    candidates.sort(key=lambda c: c["delta"], reverse=True)
    return candidates[:max_items]
