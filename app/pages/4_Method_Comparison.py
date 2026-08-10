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
from src.methods.border_discontinuity import US_STATE_BORDER_PAIRS
from src.methods.comparison import build_comparison

st.title("Method Comparison")
st.markdown(
    "Point estimates and 95% confidence intervals across estimators, run "
    "on the same panel. Differences across methods are the headline "
    "finding in this literature — a single number understates how "
    "sensitive the answer is to identification strategy."
)

panel, is_synthetic = load_state_year_panel()
if is_synthetic:
    st.info("Using synthetic demo data — no real panel found in data/processed/.")


@st.cache_data(show_spinner=False)
def _comparison(panel, use_border_pairs, n_boot):
    pairs = US_STATE_BORDER_PAIRS if use_border_pairs else None
    return build_comparison(panel, border_pairs=pairs, n_boot=n_boot)


n_boot = st.slider(
    "Bootstrap draws", min_value=100, max_value=1000, value=500, step=100,
    help="Used by Callaway-Sant'Anna and the synthetic control average. "
         "Higher is smoother but slower.",
)
use_border_pairs = st.checkbox(
    "Include the border-discontinuity design", value=True,
    help=f"Restricts to the {len(US_STATE_BORDER_PAIRS)} contiguous state "
         "pairs in US_STATE_BORDER_PAIRS.",
)

with st.spinner("Running every estimator on the panel..."):
    comparison = _comparison(panel, use_border_pairs, n_boot)

factor = comparison.attrs.get("log_wage_gap", float("nan"))
st.markdown(
    "Every estimate below is the **average treatment effect on the treated, "
    "in percentage points of unemployment** — the effect of the minimum wage "
    "policy treated states actually enacted. The estimators do not natively "
    "answer that same question: TWFE and the border design regress on log "
    f"minimum wage, so their semi-elasticities are multiplied by {factor:.3f}, "
    "the average log gap between the state and federal minimum among treated "
    "state-years. The `scale` column records which rows rest on that step — "
    "and flags the synthetic control's interval as `conditional`, because it "
    "is conditional on the fitted per-unit effects rather than on the "
    "permutation inference the design actually supports (see the note)."
)

failed = comparison[comparison["scale"] == "failed"]
for row in failed.itertuples():
    st.warning(f"{row.method.replace(chr(10), ' ')} failed: {row.note}")

results = comparison[comparison["scale"] != "failed"].copy()
if results.empty:
    st.error("No estimators produced a result.")
    st.stop()

results["method"] = results["method"].str.replace("\n", " ", regex=False)
st.dataframe(
    results[["method", "estimate", "ci_lower", "ci_upper", "scale", "note"]],
    width="stretch",
    hide_index=True,
)

# One trace per scale, and every scale `comparison.py` can emit needs an
# entry: a scale missing from here is a row silently absent from the plot
# while still present in the table above.
SCALE_TRACES = (
    ("native", "#3b6ea5", "Binary treatment (native scale)"),
    ("converted", "#c4703a", f"Semi-elasticity x {factor:.3f} (converted)"),
    ("conditional", "#7a4fbf", "CI conditional on fitted effects"),
)
plotted = {scale for scale, _, _ in SCALE_TRACES}
missing = sorted(set(results["scale"]) - plotted)
if missing:
    st.warning(
        f"Rows with scale {missing} are in the table but not the plot; "
        "add them to SCALE_TRACES."
    )

fig = go.Figure()
for scale, color, label in SCALE_TRACES:
    subset = results[results["scale"] == scale]
    if subset.empty:
        continue
    fig.add_trace(go.Scatter(
        x=subset["estimate"], y=subset["method"], mode="markers",
        name=label,
        error_x=dict(
            type="data",
            array=subset["ci_upper"] - subset["estimate"],
            arrayminus=subset["estimate"] - subset["ci_lower"],
        ),
        marker=dict(size=12, color=color),
    ))
fig.add_vline(x=0, line_dash="dot", line_color="gray")
fig.update_layout(
    title="Estimated ATT of a state minimum wage above the federal floor",
    xaxis_title="Percentage points of unemployment",
    # Traces are added per scale, which would otherwise group the methods by
    # colour and reorder them away from the table above.
    yaxis=dict(
        categoryorder="array",
        categoryarray=list(results["method"])[::-1],
    ),
    legend=dict(orientation="h", yanchor="bottom", y=-0.35),
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "Intervals that straddle zero mean the design cannot rule out no effect. "
    "Read the spread across methods, not any single point estimate."
)
