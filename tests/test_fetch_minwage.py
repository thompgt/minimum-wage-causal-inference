import pandas as pd
import pytest

from src.data.build_panel import build_state_month_panel
from src.data.fetch_minwage import (
    FEDERAL_MINIMUM_WAGE_SCHEDULE,
    federal_minimum_wage,
    load_minimum_wage,
)


def _fed(year, month):
    return federal_minimum_wage([year], [month]).iloc[0]


def test_federal_schedule_is_monotone():
    wages = [w for *_, w in FEDERAL_MINIMUM_WAGE_SCHEDULE]
    assert wages == sorted(wages)


def test_federal_wage_at_known_dates():
    assert _fed(2000, 6) == pytest.approx(5.15)
    assert _fed(2015, 1) == pytest.approx(7.25)
    assert _fed(1996, 9) == pytest.approx(4.25)  # month before the Oct 1996 rise
    assert _fed(1996, 10) == pytest.approx(4.75)


def test_mid_month_increase_binds_the_following_month():
    """The 2007-2009 amendments took effect July 24, so July keeps the old rate."""
    assert _fed(2007, 7) == pytest.approx(5.15)
    assert _fed(2007, 8) == pytest.approx(5.85)
    assert _fed(2009, 7) == pytest.approx(6.55)
    assert _fed(2009, 8) == pytest.approx(7.25)


def test_before_schedule_is_missing():
    assert pd.isna(_fed(1970, 1))


def _write_csv(tmp_path, rows):
    path = tmp_path / "state_minimum_wage.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_user_csv_without_federal_column_is_filled(tmp_path):
    path = _write_csv(tmp_path, {
        "state": ["CA", "TX"],
        "year": [2015, 2015],
        "month": [1, 1],
        "minimum_wage": [9.0, 7.25],
    })
    out = load_minimum_wage(path)
    assert out["federal_minimum_wage"].tolist() == [7.25, 7.25]


def test_user_csv_below_federal_is_lifted_to_the_binding_wage(tmp_path):
    """A state law below the federal floor doesn't bind; the federal one does."""
    path = _write_csv(tmp_path, {
        "state": ["GA"],
        "year": [2015],
        "month": [1],
        "minimum_wage": [5.15],  # Georgia's nominal state law
    })
    out = load_minimum_wage(path)
    assert out["minimum_wage"].iloc[0] == pytest.approx(7.25)


def test_user_csv_federal_column_is_respected_where_supplied(tmp_path):
    path = _write_csv(tmp_path, {
        "state": ["CA", "CA"],
        "year": [2015, 2015],
        "month": [1, 2],
        "minimum_wage": [9.0, 9.0],
        "federal_minimum_wage": [8.00, None],
    })
    out = load_minimum_wage(path)
    assert out["federal_minimum_wage"].tolist() == [8.00, 7.25]


def test_user_csv_predating_the_schedule_raises(tmp_path):
    path = _write_csv(tmp_path, {
        "state": ["CA"],
        "year": [1970],
        "month": [1],
        "minimum_wage": [1.65],
    })
    with pytest.raises(ValueError, match="FLSA"):
        load_minimum_wage(path)


def test_user_csv_missing_required_column_raises(tmp_path):
    path = _write_csv(tmp_path, {"state": ["CA"], "year": [2015], "month": [1]})
    with pytest.raises(ValueError, match="minimum_wage"):
        load_minimum_wage(path)


def test_panel_builds_from_a_user_csv_without_a_federal_column(tmp_path):
    """The substitution route the README documents, end to end."""
    unemployment = pd.DataFrame({
        "state": ["CA", "CA", "TX", "TX"],
        "year": [2015] * 4,
        "month": [1, 2, 1, 2],
        "unemployment_rate": [6.5, 6.4, 4.4, 4.3],
    })
    u_path = tmp_path / "unemployment.parquet"
    unemployment.to_parquet(u_path, index=False)

    mw_path = _write_csv(tmp_path, {
        "state": ["CA", "CA", "TX", "TX"],
        "year": [2015] * 4,
        "month": [1, 2, 1, 2],
        "minimum_wage": [9.0, 9.0, 7.25, 7.25],
    })

    panel = build_state_month_panel(u_path, mw_path)
    assert len(panel) == 4
    above = panel.set_index(["state", "month"])["above_federal"]
    assert bool(above[("CA", 1)]) is True
    assert bool(above[("TX", 1)]) is False
