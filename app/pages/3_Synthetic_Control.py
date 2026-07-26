import sys
from pathlib import Path

# Streamlit puts this script's own directory on sys.path, not the project
# root, so `import src` fails without this when the app is launched the
# documented way (`streamlit run app/Home.py`).
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_state_year_panel
from src.methods.synthetic_control import estimate_synthetic_control, placebo_test

st.title("Synthetic Control")
st.markdown(
    "Builds a counterfactual for one treated state from a weighted blend of "
    "never-treated donor states, with weights constrained to the simplex so "
    "the counterfactual cannot extrapolate outside the donors' observed "
    "range. The gap between the two lines after treatment is the estimated "
    "effect for that state."
)

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data — no real panel found in data/processed/.")

if "adoption_year" not in panel.columns:
    st.error(
        "This panel has no `adoption_year` column. Rebuild it with "
        "`python -m src.data.build_panel`."
    )
    st.stop()

adoption = (
    panel.dropna(subset=["adoption_year"])
    .drop_duplicates(subset=["state"])
    .set_index("state")["adoption_year"]
    .astype(int)
)
never_treated = sorted(panel[panel["adoption_year"].isna()]["state"].unique())
first_year, last_year = int(panel["year"].min()), int(panel["year"].max())

# Only states with a real pre-period can be a case study.
eligible = sorted(s for s, y in adoption.items() if y > first_year)
if not eligible or not never_treated:
    st.error(
        "Need at least one state adopting after the panel starts and at least "
        f"one never-treated donor. Found {len(eligible)} eligible treated "
        f"states and {len(never_treated)} donors."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    treated_state = st.selectbox("Treated state (case study)", eligible)
with col2:
    treatment_year = st.number_input(
        "Treatment year",
        min_value=first_year + 1,
        max_value=last_year,
        value=int(adoption[treated_state]),
        help="Defaults to the state's own adoption year from the panel.",
    )

st.caption(
    f"Donor pool: {len(never_treated)} never-treated states "
    f"({', '.join(never_treated)})."
)
run_placebo = st.checkbox(
    "Also run permutation inference",
    help="Refits pretending each donor was treated, then ranks the real "
         "state's post/pre RMSPE ratio against that distribution. Slow.",
)

if st.button("Run synthetic control"):
    try:
        with st.spinner(f"Fitting synthetic {treated_state}..."):
            result = estimate_synthetic_control(
                panel, treated_state, int(treatment_year), donors=never_treated
            )
    except ValueError as e:
        st.error(f"Synthetic control failed: {e}")
        st.stop()

    paths = result["paths"]
    fig = go.Figure()
    for series_type, label, dash in (
        ("actual", treated_state, "solid"),
        ("synthetic", f"Synthetic {treated_state}", "dash"),
    ):
        group = paths[paths["type"] == series_type]
        fig.add_trace(go.Scatter(
            x=group["year"], y=group["unemployment_rate"],
            mode="lines", name=label, line=dict(dash=dash),
        ))
    fig.add_vline(x=treatment_year, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"{treated_state}: actual vs. synthetic counterfactual",
        xaxis_title="Year", yaxis_title="Unemployment rate (%)",
    )
    st.plotly_chart(fig, width="stretch")

    post_gap = result["gaps"][result["gaps"].index >= treatment_year].mean()
    a, b, c = st.columns(3)
    a.metric("Mean post-treatment gap", f"{post_gap:+.2f} pp")
    b.metric("Pre-treatment RMSPE", f"{result['pre_rmspe']:.3f}")
    c.metric("Post/pre RMSPE ratio", f"{result['rmspe_ratio']:.2f}")
    st.caption(
        "Pre-treatment RMSPE is the fit quality before treatment — a large "
        "value means the synthetic control never matched the state to begin "
        "with, and the post-treatment gap should not be read as an effect."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Donor weights")
        weights = result["weights"]
        weights = weights[weights > 0.001]
        st.dataframe(
            weights.rename("weight").to_frame().style.format("{:.3f}"),
            width="stretch",
        )
    with right:
        st.subheader("Yearly gap (actual − synthetic)")
        gaps = result["gaps"].rename("gap").to_frame()
        gaps.index.name = "year"
        st.dataframe(gaps.style.format("{:+.3f}"), width="stretch")

    if run_placebo:
        with st.spinner(f"Refitting for each of {len(never_treated)} donors..."):
            placebos, p_value = placebo_test(
                panel, treated_state, int(treatment_year),
                donors=never_treated, pre_rmspe_cutoff=5.0,
            )
        st.subheader("Permutation inference")
        st.metric("p-value (rank of the treated state's RMSPE ratio)", f"{p_value:.3f}")
        st.caption(
            "There is only one treated unit, so classical standard errors do "
            "not apply. This is the Abadie, Diamond & Hainmueller (2010) "
            "permutation test: donors fitting worse than 5x the treated "
            "state's pre-RMSPE are excluded from the ranking."
        )
        st.dataframe(placebos, width="stretch")
