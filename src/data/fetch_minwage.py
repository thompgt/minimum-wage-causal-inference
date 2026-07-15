"""Load the historical state minimum wage panel.

This does NOT auto-download from the source, since that dataset's hosting
location changes over time and should be verified by hand. Instead:

1. Download the state minimum wage history CSV yourself, e.g. the
   Vaghul & Zipperer series (Washington Center for Equitable Growth) or the
   UC Berkeley IRLE minimum wage tracker.
2. Save it to data/raw/state_minimum_wage.csv with at least these columns:
   state (2-letter abbr), year, month, minimum_wage (state, in nominal $).
3. Run this module to validate and normalize it into data/interim/.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"

REQUIRED_COLUMNS = {"state", "year", "month", "minimum_wage"}


def load_minimum_wage(raw_path=None):
    raw_path = Path(raw_path) if raw_path else RAW_DIR / "state_minimum_wage.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. See module docstring: download the state "
            "minimum wage history CSV manually and place it there."
        )
    df = pd.read_csv(raw_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {missing}")

    df["state"] = df["state"].str.upper().str.strip()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["minimum_wage"] = df["minimum_wage"].astype(float)
    df = df.sort_values(["state", "year", "month"]).reset_index(drop=True)
    return df[["state", "year", "month", "minimum_wage"]]


if __name__ == "__main__":
    out = load_minimum_wage()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "minimum_wage.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
