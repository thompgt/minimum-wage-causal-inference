import shutil

import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_state_year_panel
from src.methods.r_bridge import run_synthetic_control

st.title("Synthetic Control")
st.markdown(
    "Builds a synthetic counterfactual for one treated state from a "
    "weighted blend of untreated donor states. Requires R with the "
    "`Synth` package installed (`Rscript install.R`)."
)

if shutil.which("Rscript") is None:
    st.error(
        "Rscript not found on PATH. Install R (https://cran.r-project.org/) "
        "and run `Rscript install.R` from the project root to enable this page."
    )
    st.stop()

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data.")

treated_options = (
    sorted(panel.loc[panel.get("treated", False) == True, "state"].unique())
    if "treated" in panel.columns else sorted(panel["state"].unique())
)
treated_state = st.selectbox("Treated state (case study)", treated_options)
treatment_year = st.number_input(
    "Treatment year", min_value=int(panel["year"].min()) + 1,
    max_value=int(panel["year"].max()), value=int(panel["year"].median()),
)

if st.button("Run synthetic control"):
    with st.spinner("Fitting synthetic control (calls out to R)..."):
        try:
            result = run_synthetic_control(panel, treated_state, int(treatment_year))
            fig = go.Figure()
            for series_type, group in result.groupby("type"):
                fig.add_trace(go.Scatter(
                    x=group["year"], y=group["unemployment_rate"],
                    mode="lines", name=series_type,
                ))
            fig.add_vline(x=treatment_year, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"{treated_state}: actual vs. synthetic counterfactual",
                xaxis_title="Year", yaxis_title="Unemployment rate",
            )
            st.plotly_chart(fig, width='stretch')
            st.dataframe(result, width='stretch')
        except Exception as e:
            st.error(f"Synthetic control failed: {e}")
