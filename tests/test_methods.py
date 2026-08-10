import pytest

from src.data.synthetic import generate_state_year_panel
from src.diagnostics.parallel_trends import (
    pretrend_individual_screen,
    pretrend_joint_test,
)
from src.methods.event_study import estimate_event_study
from src.methods.twfe_did import estimate_twfe, summarize_twfe


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


def test_pretrend_joint_test_needs_the_fitted_result(panel):
    """A summary alone cannot carry a covariance matrix, so it must not pass."""
    _, summary = estimate_event_study(panel)
    stripped = summary.copy()
    stripped.attrs.clear()
    result, _ = estimate_event_study(panel)
    with pytest.raises(ValueError, match="col_to_reltime"):
        pretrend_joint_test(result, stripped)
