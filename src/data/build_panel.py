"""Merge unemployment and minimum wage sources into a clean analysis panel.

Produces both a state-month panel and a state-year panel (annual average
unemployment rate, December minimum wage) in data/processed/.
"""
from pathlib import Path

import pandas as pd

from src.data.fetch_minwage import load_minimum_wage

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def build_state_month_panel(unemployment_path=None, minwage_path=None):
    unemployment_path = unemployment_path or RAW_DIR / "bls_unemployment.parquet"
    if not unemployment_path.exists():
        raise FileNotFoundError(
            f"{unemployment_path} not found. Run `python -m src.data.fetch_bls` first."
        )
    unemployment = pd.read_parquet(unemployment_path)
    minwage = load_minimum_wage(minwage_path)

    panel = unemployment.merge(minwage, on=["state", "year", "month"], how="inner")
    panel["log_minimum_wage"] = panel["minimum_wage"].apply(
        lambda x: pd.NA if x <= 0 else __import__("math").log(x)
    )
    panel = panel.sort_values(["state", "year", "month"]).reset_index(drop=True)

    if panel.duplicated(subset=["state", "year", "month"]).any():
        raise ValueError("build_state_month_panel produced duplicate state-year-month rows")
    if panel["minimum_wage"].isna().any():
        raise ValueError("build_state_month_panel has missing minimum_wage values")

    return panel


def build_state_year_panel(state_month_panel):
    yearly = (
        state_month_panel.groupby(["state", "year"])
        .agg(
            unemployment_rate=("unemployment_rate", "mean"),
            minimum_wage=("minimum_wage", "last"),  # year-end minimum wage
        )
        .reset_index()
    )
    yearly["log_minimum_wage"] = yearly["minimum_wage"].apply(
        lambda x: pd.NA if x <= 0 else __import__("math").log(x)
    )
    return yearly.sort_values(["state", "year"]).reset_index(drop=True)


def main():
    month_panel = build_state_month_panel()
    year_panel = build_state_year_panel(month_panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    month_panel.to_parquet(PROCESSED_DIR / "panel_state_month.parquet", index=False)
    year_panel.to_parquet(PROCESSED_DIR / "panel_state_year.parquet", index=False)
    print(f"Wrote {len(month_panel)} state-month rows and {len(year_panel)} state-year rows")


if __name__ == "__main__":
    main()
