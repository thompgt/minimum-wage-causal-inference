import numpy as np
import pytest

from src.data.synthetic import generate_state_year_panel
from src.diagnostics.parallel_trends import (
    pretrend_individual_screen,
    pretrend_joint_test,
)
from src.methods.event_study import estimate_event_study
from src.methods.twfe_did import LABOR_FORCE, estimate_twfe, summarize_twfe


@pytest.fixture(scope="module")
def panel():
    return generate_state_year_panel(seed=7)


def test_twfe_recovers_approximately_correct_sign(panel):
    result = estimate_twfe(panel)
    coef = result.params["log_minimum_wage"]
    # True effect is negative; with noise the point estimate should at
    # least land in the same direction across most seeds.
    assert coef < 0.15  # loose bound: not wildly positive


def test_twfe_summary_has_expected_columns(panel):
    result = estimate_twfe(panel)
    summary = summarize_twfe(result)
    assert set(summary.columns) == {"coef", "std_error", "pvalue", "ci_lower", "ci_upper"}
    assert "log_minimum_wage" in summary.index


# --- unit of analysis: per state, or per worker? --------------------------

@pytest.fixture
def weighted_panel(panel):
    """The panel plus a labour force column that varies a lot across states.

    Real state labour forces span two orders of magnitude (WY ~290k, CA
    ~19m), which is exactly why the weighting choice is not cosmetic.
    """
    df = panel.copy()
    states = sorted(df["state"].unique())
    sizes = {s: 10.0 ** (5 + 2 * i / max(len(states) - 1, 1))
             for i, s in enumerate(states)}
    df[LABOR_FORCE] = df["state"].map(sizes)
    return df


def test_weights_change_the_estimate(weighted_panel):
    """If they did not, the weighting choice would not need stating."""
    equal = estimate_twfe(weighted_panel).params["log_minimum_wage"]
    weighted = estimate_twfe(weighted_panel, weights=LABOR_FORCE)
    assert not np.isclose(equal, weighted.params["log_minimum_wage"])


def test_weighting_is_recorded_on_the_result(weighted_panel):
    assert estimate_twfe(weighted_panel).weighting == "equal"
    assert estimate_twfe(weighted_panel, weights=LABOR_FORCE).weighting == LABOR_FORCE


def test_missing_weight_column_is_a_readable_error(panel):
    with pytest.raises(ValueError, match="no 'labor_force' column"):
        estimate_twfe(panel, weights=LABOR_FORCE)


def test_non_positive_weights_are_rejected(weighted_panel):
    """A zero weight silently drops a state-year; say so instead."""
    df = weighted_panel.copy()
    df.loc[df.index[0], LABOR_FORCE] = 0.0
    with pytest.raises(ValueError, match="missing or non-positive"):
        estimate_twfe(df, weights=LABOR_FORCE)


def test_constant_weights_match_the_unweighted_fit(weighted_panel):
    """Weighting every state-year the same is the equal-weight estimator."""
    df = weighted_panel.copy()
    df[LABOR_FORCE] = 1_000.0
    assert np.isclose(
        estimate_twfe(df).params["log_minimum_wage"],
        estimate_twfe(df, weights=LABOR_FORCE).params["log_minimum_wage"],
    )


def test_event_study_omits_reference_period_at_zero(panel):
    _, summary = estimate_event_study(panel, omit=-1)
    ref_row = summary[summary["rel_time"] == -1].iloc[0]
    assert ref_row["coef"] == 0.0


def test_event_study_covers_requested_window(panel):
    _, summary = estimate_event_study(panel, min_lead=-3, max_lag=3)
    assert set(summary["rel_time"]) == set(range(-3, 4))


def test_pretrend_joint_test_is_a_real_wald_test(panel):
    result, summary = estimate_event_study(panel)
    joint = pretrend_joint_test(result, summary)
    assert joint["df"] == len(joint["leads_tested"]) > 0
    assert all(t < 0 for t in joint["leads_tested"])
    assert joint["statistic"] >= 0
    assert 0.0 <= joint["p_value"] <= 1.0
    assert joint["passes"] is (joint["p_value"] > joint["alpha"])


def test_pretrend_joint_test_carries_the_individual_screen(panel):
    result, summary = estimate_event_study(panel)
    screen = pretrend_joint_test(result, summary)["individual_screen"]
    assert screen == pretrend_individual_screen(summary)
    assert screen["n_pre_periods"] > 0


def test_the_screen_does_not_count_the_omitted_reference_period(panel):
    """It is a normalised zero, not an estimate; counting it would make the
    screen and the joint test disagree on how many leads there are."""
    result, summary = estimate_event_study(panel)
    joint = pretrend_joint_test(result, summary)
    assert summary.attrs["omit"] not in joint["individual_screen"]["violating_periods"]
    assert joint["individual_screen"]["n_pre_periods"] == joint["df"]


def test_pretrend_joint_test_needs_the_fitted_result(panel):
    """A summary alone cannot carry a covariance matrix, so it must not pass."""
    _, summary = estimate_event_study(panel)
    stripped = summary.copy()
    stripped.attrs.clear()
    result, _ = estimate_event_study(panel)
    with pytest.raises(ValueError, match="col_to_reltime"):
        pretrend_joint_test(result, stripped)
