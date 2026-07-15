import pandas as pd
import pytest

from src.data.build_panel import build_state_month_panel, build_state_year_panel


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


def test_missing_unemployment_file_raises(tmp_path, sample_minwage):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError):
        build_state_month_panel(missing_path, sample_minwage)
