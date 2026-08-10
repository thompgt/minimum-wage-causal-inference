import pandas as pd
import pytest

from src.data.build_panel import (
    TREATMENT_CONVENTIONS,
    build_state_month_panel,
    build_state_year_panel,
)


@pytest.fixture
def sample_unemployment(tmp_path):
    df = pd.DataFrame({
        "state": ["CA", "CA", "TX", "TX"],
        "year": [2020, 2020, 2020, 2020],
        "month": [1, 2, 1, 2],
        "unemployment_rate": [10.1, 9.8, 7.2, 7.0],
    })
    path = tmp_path / "unemployment.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def sample_minwage(tmp_path):
    df = pd.DataFrame({
        "state": ["CA", "CA", "TX", "TX"],
        "year": [2020, 2020, 2020, 2020],
        "month": [1, 2, 1, 2],
        "minimum_wage": [13.0, 13.0, 7.25, 7.25],
    })
    path = tmp_path / "minwage.csv"
    df.to_csv(path, index=False)
    return path


def test_state_month_panel_has_no_duplicates(sample_unemployment, sample_minwage):
    panel = build_state_month_panel(sample_unemployment, sample_minwage)
    assert not panel.duplicated(subset=["state", "year", "month"]).any()


def test_state_month_panel_has_no_missing_treatment(sample_unemployment, sample_minwage):
    panel = build_state_month_panel(sample_unemployment, sample_minwage)
    assert not panel["minimum_wage"].isna().any()


def test_state_month_panel_expected_rows(sample_unemployment, sample_minwage):
    panel = build_state_month_panel(sample_unemployment, sample_minwage)
    assert len(panel) == 4
    assert set(panel["state"]) == {"CA", "TX"}


def test_state_year_panel_aggregates_correctly(sample_unemployment, sample_minwage):
    month_panel = build_state_month_panel(sample_unemployment, sample_minwage)
    year_panel = build_state_year_panel(month_panel)
    ca_row = year_panel[year_panel["state"] == "CA"].iloc[0]
    assert ca_row["unemployment_rate"] == pytest.approx((10.1 + 9.8) / 2)
    assert ca_row["minimum_wage"] == pytest.approx(13.0)


# --- the annual treatment convention -------------------------------------
#
# The shape that used to break: a state above the federal floor for part of
# a year and back at it by December. `last` for the wage and `any` for the
# flag disagreed, and because treatment is absorbing the disagreement
# propagated forever. Kentucky 2007 is the real-panel instance.

@pytest.fixture
def brief_riser(tmp_path):
    """KY 2007 in miniature: above the floor mid-year, at it by year end.

    Three years, three states -- one brief riser, one permanent adopter in
    year two, one that never moves.
    """
    years, months = [2006, 2007, 2008], [6, 12]
    wages = {
        "BR": {(2006, 6): 5.15, (2006, 12): 5.15,   # brief riser
               (2007, 6): 5.85, (2007, 12): 5.15,   # up in June, back by Dec
               (2008, 6): 5.15, (2008, 12): 5.15},
        "AD": {(2006, 6): 5.15, (2006, 12): 5.15,   # permanent adopter
               (2007, 6): 5.15, (2007, 12): 7.00,
               (2008, 6): 7.00, (2008, 12): 7.00},
        "NV": {(y, m): 5.15 for y in years for m in months},  # never treated
    }
    rows = [
        {"state": s, "year": y, "month": m, "minimum_wage": w[(y, m)],
         "federal_minimum_wage": 5.15}
        for s, w in wages.items() for y in years for m in months
    ]
    mw = tmp_path / "brief_minwage.csv"
    pd.DataFrame(rows).to_csv(mw, index=False)

    unemp = pd.DataFrame([
        {"state": s, "year": y, "month": m, "unemployment_rate": 5.0}
        for s in wages for y in years for m in months
    ])
    ue = tmp_path / "brief_unemployment.parquet"
    unemp.to_parquet(ue, index=False)
    return build_state_month_panel(ue, mw)


def adoption_years(year_panel):
    return year_panel.groupby("state")["adoption_year"].first().to_dict()


def test_year_end_convention_ignores_a_wage_that_did_not_last(brief_riser):
    adoption = adoption_years(build_state_year_panel(brief_riser))
    assert pd.isna(adoption["BR"])
    assert adoption["AD"] == 2007
    assert pd.isna(adoption["NV"])


def test_any_month_convention_still_picks_the_brief_riser_up(brief_riser):
    adoption = adoption_years(
        build_state_year_panel(brief_riser, treatment_convention="any-month")
    )
    assert adoption["BR"] == 2007
    assert adoption["AD"] == 2007


def test_treated_rows_never_contradict_their_own_wage(brief_riser):
    """The bug in one assertion: no treated row may sit at the federal floor.

    Under `any-month` the brief riser is treated in 2008 with a recorded
    minimum wage equal to the federal one -- a row that says the state is a
    minimum-wage state and that its minimum wage is the federal minimum.
    """
    year_end = build_state_year_panel(brief_riser)
    above = year_end["minimum_wage"] > year_end["federal_minimum_wage"]
    assert (year_end["above_federal"] == above).all()

    any_month = build_state_year_panel(brief_riser, treatment_convention="any-month")
    contradictory = any_month[
        any_month["above_federal"]
        & (any_month["minimum_wage"] <= any_month["federal_minimum_wage"])
    ]
    assert not contradictory.empty  # the reason the default changed


def test_convention_is_recorded_on_the_panel(brief_riser):
    for convention in TREATMENT_CONVENTIONS:
        panel = build_state_year_panel(brief_riser, treatment_convention=convention)
        assert panel.attrs["treatment_convention"] == convention


def test_unknown_convention_is_rejected(brief_riser):
    with pytest.raises(ValueError, match="unknown treatment_convention"):
        build_state_year_panel(brief_riser, treatment_convention="mean")


def test_missing_unemployment_file_raises(tmp_path, sample_minwage):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        build_state_month_panel(missing_path, sample_minwage)
