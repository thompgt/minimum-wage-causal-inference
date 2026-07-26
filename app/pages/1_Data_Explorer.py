import sys
from pathlib import Path

# Streamlit puts this script's own directory on sys.path, not the project
# root, so `import src` fails without this when the app is launched the
# documented way (`streamlit run app/Home.py`).
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.express as px
import streamlit as st

from src.data.loader import load_state_year_panel

st.title("Data Explorer")

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data.")

states = sorted(panel["state"].unique())
selected_states = st.multiselect("States", states, default=states[:5])
year_range = st.slider(
    "Year range",
    int(panel["year"].min()), int(panel["year"].max()),
    (int(panel["year"].min()), int(panel["year"].max())),
)

filtered = panel[
    panel["state"].isin(selected_states)
    & panel["year"].between(*year_range)
]

if filtered.empty:
    st.warning("No data for the current selection.")
else:
    fig_unemp = px.line(
        filtered, x="year", y="unemployment_rate", color="state",
        title="Unemployment rate over time",
    )
    st.plotly_chart(fig_unemp, width='stretch')

    fig_wage = px.line(
        filtered, x="year", y="minimum_wage", color="state",
        title="Minimum wage over time",
    )
    st.plotly_chart(fig_wage, width='stretch')

    st.dataframe(filtered.sort_values(["state", "year"]), width='stretch')
