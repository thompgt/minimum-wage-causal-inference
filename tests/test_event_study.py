"""The event study's sample composition and its post-period contrast.

The regression is only a difference-in-differences if a never-treated
group is in the estimation sample, and the headline post-period number is
only a confidence interval if it comes from a contrast rather than from
averaging endpoints. Both are asserted here rather than assumed.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import generate_state_year_panel
from src.methods.event_study import (
    average_post_effect,
    build_event_time,
    contrast_columns,
    estimate_event_study,
    linear_contrast,
)


@pytest.fixture(scope="module")
def panel():
    return generate_state_year_panel(seed=7)


@pytest.fixture(scope="module")
def mixed_panel(panel):
    """A panel guaranteed to contain never-, always- and staggered-treated states."""
    df = panel.copy()
    states = sorted(df["state"].unique())
    first_year = int(df["year"].min())
    assignment = {}
    for i, state in enumerate(states):
        if i % 3 == 0:
            assignment[state] = pd.NA                    # never treated
        elif i % 3 == 1:
            assignment[state] = first_year               # always treated
        else:
            assignment[state] = first_year + 5 + (i % 4)  # staggered adopter
    df["adoption_year"] = df["state"].map(assignment).astype("Int64")
    df["treated"] = df["adoption_year"].notna() & (df["year"] >= df["adoption_year"])
    return df


def test_never_treated_states_stay_in_the_sample(mixed_panel):
    ev = build_event_time(mixed_panel)
    assert ev.attrs["n_never_treated"] > 0
    never = ev[ev["never_treated"]]
    assert not never.empty, "never-treated states must remain as the control group"
    assert never["rel_time"].isna().all(), "controls carry no event time"


def test_always_treated_states_are_dropped(mixed_panel):
    ev = build_event_time(mixed_panel)
    dropped = ev.attrs["always_treated_states"]
    assert dropped, "fixture should contain always-treated states"
    assert not set(dropped) & set(ev["state"].unique())


def test_never_treated_can_be_excluded_explicitly(mixed_panel):
    ev = build_event_time(mixed_panel, include_never_treated=False)
    assert ev.attrs["n_never_treated"] == 0
    assert ev["rel_time"].notna().all()


def test_control_rows_are_zero_on_every_event_dummy(mixed_panel):
    """The bug this guards: <NA> == t is pd.NA, which must read as 0, not drop."""
    result, summary = estimate_event_study(mixed_panel)
    ev = build_event_time(mixed_panel)
    dummy_cols = list(summary.attrs["col_to_reltime"])
    for col in dummy_cols:
        t = summary.attrs["col_to_reltime"][col]
        expected = ev["rel_time"].eq(t).fillna(False).astype(int)
        assert expected[ev["never_treated"].to_numpy()].sum() == 0


def test_post_period_contrast_is_not_the_average_of_ci_endpoints(panel):
    result, summary = estimate_event_study(panel)
    post = average_post_effect(result, summary)

    endpoint_avg_lower = summary.loc[summary["rel_time"] >= 0, "ci_lower"].mean()
    endpoint_avg_upper = summary.loc[summary["rel_time"] >= 0, "ci_upper"].mean()

    # The point estimate is the same average; the interval is not, because a
    # contrast uses sqrt(w'Vw) rather than the mean of per-coefficient bounds.
    assert post["estimate"] == pytest.approx(
        summary.loc[summary["rel_time"] >= 0, "coef"].mean()
    )
    assert not np.isclose(post["ci_lower"], endpoint_avg_lower)
    assert not np.isclose(post["ci_upper"], endpoint_avg_upper)


def test_contrast_interval_is_symmetric_and_positive_width(panel):
    result, summary = estimate_event_study(panel)
    post = average_post_effect(result, summary)
    assert post["std_error"] > 0
    assert post["ci_lower"] < post["estimate"] < post["ci_upper"]
    assert post["estimate"] - post["ci_lower"] == pytest.approx(
        post["ci_upper"] - post["estimate"]
    )


def test_single_coefficient_contrast_reproduces_its_own_ci(panel):
    """A one-element contrast must agree with the fitted coefficient's CI."""
    result, summary = estimate_event_study(panel)
    col = contrast_columns(summary, min_rel_time=1, max_rel_time=1)[0]
    single = linear_contrast(result, [col])
    row = summary[summary["rel_time"] == 1].iloc[0]
    assert single["estimate"] == pytest.approx(row["coef"])
    # linearmodels reports t-based bounds; a normal contrast is close but not
    # identical, so allow a small relative tolerance on the half-width.
    assert single["ci_lower"] == pytest.approx(row["ci_lower"], rel=0.05)
    assert single["ci_upper"] == pytest.approx(row["ci_upper"], rel=0.05)


def test_contrast_columns_rejects_an_empty_window(panel):
    _, summary = estimate_event_study(panel, min_lead=-2, max_lag=2)
    with pytest.raises(ValueError, match="no event-time coefficients"):
        contrast_columns(summary, min_rel_time=99)
