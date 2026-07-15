import numpy as np

from src.data.synthetic import STATES, YEARS, generate_panel, generate_state_year_panel


def test_generate_panel_shape():
    df = generate_panel()
    assert len(df) == len(STATES) * len(YEARS) * 12
    assert not df.duplicated(subset=["state", "year", "month"]).any()


def test_generate_panel_deterministic_with_seed():
    df1 = generate_panel(seed=1)
    df2 = generate_panel(seed=1)
    assert df1.equals(df2)


def test_generate_panel_different_seeds_differ():
    df1 = generate_panel(seed=1)
    df2 = generate_panel(seed=2)
    assert not df1["unemployment_rate"].equals(df2["unemployment_rate"])


def test_some_states_never_treated():
    df = generate_state_year_panel()
    never_treated = df.groupby("state")["treated"].max()
    assert (~never_treated).any(), "expected some never-treated control states"
    assert never_treated.any(), "expected some treated states"


def test_no_missing_core_columns():
    df = generate_state_year_panel()
    for col in ["state", "year", "unemployment_rate", "minimum_wage", "log_minimum_wage"]:
        assert not df[col].isna().any()
