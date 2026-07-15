import streamlit as st

from src.data.loader import load_state_year_panel

st.set_page_config(page_title="Minimum Wage vs. Unemployment", layout="wide")

st.title("Minimum Wage vs. Unemployment: Causal Inference Explorer")
st.markdown(
    """
This app explores the causal effect of state minimum wage increases on
unemployment using several estimators side by side, so the sensitivity of
the answer to method choice is visible rather than hidden behind one number.

Use the pages in the sidebar:

- **Data Explorer** — browse the underlying panel by state/year
- **DiD Estimator** — interactive two-way fixed effects + event-study
- **Synthetic Control** — case-study counterfactual for one treated state
- **Method Comparison** — all estimators' point estimates side by side
"""
)

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.warning(
        "Showing **synthetic demo data** (no real BLS/minimum-wage data found "
        "in data/processed/). Run `python -m src.data.build_panel` after "
        "setting BLS_API_KEY and adding a minimum wage CSV to use real data."
    )

col1, col2, col3 = st.columns(3)
col1.metric("States", panel["state"].nunique())
col2.metric("Years", f"{panel['year'].min()}–{panel['year'].max()}")
col3.metric("Rows", len(panel))

st.dataframe(panel.head(20), width='stretch')
