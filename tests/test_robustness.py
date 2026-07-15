import pytest

from src.data.synthetic import generate_state_year_panel
from src.diagnostics.robustness import bootstrap_state_cluster, leave_one_state_out
from src.methods.twfe_did import estimate_twfe


@pytest.fixture(scope="module")
def panel():
    return generate_state_year_panel(seed=3)


def test_bootstrap_returns_requested_count(panel):
    boot = bootstrap_state_cluster(panel, estimate_twfe, n_boot=10, seed=1)
    assert len(boot) <= 10
    assert len(boot) > 0


def test_bootstrap_is_deterministic_with_seed(panel):
    boot1 = bootstrap_state_cluster(panel, estimate_twfe, n_boot=5, seed=1)
    boot2 = bootstrap_state_cluster(panel, estimate_twfe, n_boot=5, seed=1)
    assert (boot1.values == boot2.values).all()


def test_leave_one_state_out_covers_all_states(panel):
    loo = leave_one_state_out(panel, estimate_twfe)
    assert set(loo["dropped_state"]) == set(panel["state"].unique())


def test_leave_one_state_out_sorted_by_influence(panel):
    loo = leave_one_state_out(panel, estimate_twfe)
    deltas = loo["delta_from_full"].abs().values
    assert (deltas[:-1] >= deltas[1:]).all()
