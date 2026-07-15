import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_state_year_panel
from src.diagnostics.parallel_trends import pretrend_joint_test
from src.methods.event_study import estimate_event_study
from src.methods.twfe_did import estimate_twfe, summarize_twfe

st.title("DiD Estimator")

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data.")

st.subheader("Two-way fixed effects")
try:
    result = estimate_twfe(panel)
    summary = summarize_twfe(result)
    st.dataframe(summary, width='stretch')
    coef = summary.loc["log_minimum_wage", "coef"]
    pval = summary.loc["log_minimum_wage", "pvalue"]
    st.write(
        f"A 10% increase in the minimum wage is associated with a "
        f"**{coef * 0.1:+.3f} percentage-point** change in the unemployment "
        f"rate (p={pval:.3f})."
    )
except Exception as e:
    st.error(f"TWFE estimation failed: {e}")

st.subheader("Event study")
if "adoption_year" not in panel.columns:
    st.warning("Panel has no `adoption_year` column — event study needs staggered "
               "treatment timing, only available in the synthetic panel for now.")
else:
    min_lead = st.slider("Min lead (periods before treatment)", -8, -1, -5)
    max_lag = st.slider("Max lag (periods after treatment)", 1, 8, 5)
    try:
        _, es_summary = estimate_event_study(panel, min_lead=min_lead, max_lag=max_lag)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=es_summary["rel_time"], y=es_summary["coef"], mode="markers+lines",
            error_y=dict(
                type="data",
                array=es_summary["ci_upper"] - es_summary["coef"],
                arrayminus=es_summary["coef"] - es_summary["ci_lower"],
            ),
            name="Coefficient",
        ))
        fig.add_vline(x=-0.5, line_dash="dash", line_color="gray")
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Event study: unemployment relative to minimum wage adoption",
            xaxis_title="Years relative to adoption",
            yaxis_title="Coefficient",
        )
        st.plotly_chart(fig, width='stretch')

        pretrend = pretrend_joint_test(es_summary)
        if pretrend["passes"]:
            st.success("Pre-trend check: no significant pre-period coefficients.")
        else:
            st.warning(
                f"Pre-trend check: significant coefficients at periods "
                f"{pretrend['violating_periods']} — parallel trends assumption "
                f"may be violated."
            )
    except Exception as e:
        st.error(f"Event study failed: {e}")
