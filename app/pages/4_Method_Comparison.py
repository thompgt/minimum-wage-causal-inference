import shutil

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_state_year_panel
from src.methods.border_discontinuity import border_pair_diffs, estimate_border_effect
from src.methods.twfe_did import estimate_twfe

st.title("Method Comparison")
st.markdown(
    "Point estimates and 95% confidence intervals across estimators, run "
    "on the same panel. Differences across methods are the headline "
    "finding in this literature — a single number understates how "
    "sensitive the answer is to identification strategy."
)

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data.")

rows = []

try:
    twfe_result = estimate_twfe(panel)
    coef = twfe_result.params["log_minimum_wage"]
    ci = twfe_result.conf_int()
    rows.append({
        "method": "TWFE DiD",
        "coef": coef,
        "ci_lower": ci.loc["log_minimum_wage", "lower"],
        "ci_upper": ci.loc["log_minimum_wage", "upper"],
    })
except Exception as e:
    st.warning(f"TWFE failed: {e}")

try:
    states = sorted(panel["state"].unique())
    demo_pairs = list(zip(states[::2], states[1::2]))
    diffs = border_pair_diffs(panel, demo_pairs, treatment="minimum_wage")
    border_model = estimate_border_effect(diffs)
    ci = border_model.conf_int()
    rows.append({
        "method": "Border discontinuity",
        "coef": border_model.params["treatment_diff"],
        "ci_lower": ci.loc["treatment_diff", 0],
        "ci_upper": ci.loc["treatment_diff", 1],
    })
except Exception as e:
    st.warning(f"Border discontinuity failed: {e}")

if shutil.which("Rscript") is None:
    st.info(
        "Callaway-Sant'Anna and synthetic control estimates need R "
        "installed (`Rscript install.R`) — omitted from this comparison."
    )

comparison = pd.DataFrame(rows)
if comparison.empty:
    st.error("No estimators produced a result.")
else:
    st.dataframe(comparison, width='stretch')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison["method"], y=comparison["coef"], mode="markers",
        error_y=dict(
            type="data",
            array=comparison["ci_upper"] - comparison["coef"],
            arrayminus=comparison["coef"] - comparison["ci_lower"],
        ),
        marker=dict(size=12),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Estimated effect of log(minimum wage) on unemployment rate, by method",
        yaxis_title="Coefficient",
    )
    st.plotly_chart(fig, width='stretch')
