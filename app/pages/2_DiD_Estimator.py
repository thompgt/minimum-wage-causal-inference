import sys
from pathlib import Path

# Streamlit puts this script's own directory on sys.path, not the project
# root, so `import src` fails without this when the app is launched the
# documented way (`streamlit run app/Home.py`).
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_state_year_panel
from src.diagnostics.parallel_trends import pretrend_joint_test
from src.methods.event_study import average_post_effect, estimate_event_study
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
        es_result, es_summary = estimate_event_study(
            panel, min_lead=min_lead, max_lag=max_lag
        )
        st.caption(
            f"{es_summary.attrs['n_adopters']} staggered adopters against "
            f"{es_summary.attrs['n_never_treated']} never-treated control states; "
            f"{len(es_summary.attrs['always_treated_states'])} always-treated states "
            "dropped (no pre-period)."
        )
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

        post = average_post_effect(es_result, es_summary)
        st.metric(
            "Average post-adoption effect (pp)",
            f"{post['estimate']:+.3f}",
            help="Linear contrast over the post-adoption coefficients, with a "
                 "standard error that accounts for their covariance — not the "
                 "average of their individual CI endpoints.",
        )
        st.caption(
            f"95% CI [{post['ci_lower']:+.3f}, {post['ci_upper']:+.3f}], "
            f"se {post['std_error']:.3f}, over {post['n_coefficients']} coefficients."
        )

        pretrend = pretrend_joint_test(es_result, es_summary)
        screen = pretrend["individual_screen"]
        if pretrend["passes"]:
            st.success(
                f"Joint pre-trend Wald test over leads {pretrend['leads_tested']}: "
                f"chi2({pretrend['df']}) = {pretrend['statistic']:.2f}, "
                f"p = {pretrend['p_value']:.3f} — no detectable pre-trend. "
                "Failing to reject is not evidence that parallel trends holds."
            )
        else:
            st.warning(
                f"Joint pre-trend Wald test rejects: chi2({pretrend['df']}) = "
                f"{pretrend['statistic']:.2f}, p = {pretrend['p_value']:.3f}. "
                f"Individually significant leads: "
                f"{screen['violating_periods'] or 'none'} — the leads are jointly "
                "off even where no single one is."
            )
    except Exception as e:
        st.error(f"Event study failed: {e}")
