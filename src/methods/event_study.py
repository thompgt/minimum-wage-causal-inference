"""Event-study specification: leads/lags of unemployment around adoption.

Only meaningful for treated states (those with a finite adoption_year).
Regresses unemployment on a set of relative-time-to-treatment dummies plus
state and year fixed effects, letting pre-treatment coefficients serve as
a visual/statistical parallel-trends check.
"""
import pandas as pd
from linearmodels.panel import PanelOLS


def build_event_time(panel, min_lead=-5, max_lag=5):
    """Add a `rel_time` column (year - adoption_year), clipped to [min_lead, max_lag].

    Never-treated states get rel_time = NaN and are dropped from the
    event-study regression (they contribute no leads/lags).
    """
    df = panel.copy()
    df["rel_time"] = df["year"] - df["adoption_year"]
    df = df[df["rel_time"].notna()].copy()
    df["rel_time"] = df["rel_time"].clip(min_lead, max_lag).astype(int)
    return df


def estimate_event_study(panel, outcome="unemployment_rate", min_lead=-5, max_lag=5,
                          omit=-1):
    """Fit an event-study regression with relative-time dummies (omitting `omit`)."""
    df = build_event_time(panel, min_lead, max_lag)
    dummy_cols = []
    col_to_reltime = {}
    for t in range(min_lead, max_lag + 1):
        if t == omit:
            continue
        col = f"rel_m{abs(t)}" if t < 0 else f"rel_p{t}"
        df[col] = (df["rel_time"] == t).astype(int)
        dummy_cols.append(col)
        col_to_reltime[col] = t

    df = df.set_index(["state", "year"])
    formula = f"{outcome} ~ " + " + ".join(dummy_cols) + " + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=df)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    summary = pd.DataFrame({
        "rel_time": [col_to_reltime[c] for c in dummy_cols],
        "coef": result.params.values,
        "ci_lower": result.conf_int()["lower"].values,
        "ci_upper": result.conf_int()["upper"].values,
    }).sort_values("rel_time").reset_index(drop=True)
    # Add the omitted reference period back in at coef=0 for plotting.
    summary = pd.concat([
        summary,
        pd.DataFrame([{"rel_time": omit, "coef": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}]),
    ]).sort_values("rel_time").reset_index(drop=True)

    return result, summary


if __name__ == "__main__":
    from src.data.synthetic import generate_state_year_panel

    panel = generate_state_year_panel()
    _, summary = estimate_event_study(panel)
    print(summary)
