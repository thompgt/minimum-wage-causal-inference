"""Load the historical state minimum wage panel.

Default source is the Vaghul & Zipperer historical state minimum wage
series (Washington Center for Equitable Growth), which is versioned as
GitHub release assets and so is stable enough to download automatically:

    https://github.com/benzipperer/historicalminwage

`load_minimum_wage()` downloads and caches the monthly state file on first
use, then reads from `data/raw/` thereafter. The series runs 1974m5 through
2022m12, which is what caps this project's panel at 2022.

To use a different source instead, drop your own CSV at
`data/raw/state_minimum_wage.csv` with columns
`state, year, month, minimum_wage` and it takes precedence over the
download. `federal_minimum_wage` is optional there: any row missing it is
filled from `FEDERAL_MINIMUM_WAGE_SCHEDULE` below, so a substituted source
only has to carry state law.
"""
from pathlib import Path

import pandas as pd
import requests

from src.data.manifest import MANIFEST_PATH, write_manifest

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"

VZ_VERSION = "v1.4.0"
VZ_URL = (
    "https://github.com/benzipperer/historicalminwage/releases/download/"
    f"{VZ_VERSION}/mw_state_excel.zip"
)
VZ_MEMBER = "mw_state_monthly.xlsx"
VZ_LOCAL = RAW_DIR / f"vz_{VZ_VERSION}_{VZ_MEMBER}"

REQUIRED_COLUMNS = {"state", "year", "month", "minimum_wage"}

# Federal floor under the FLSA, as (year, month, day, wage) effective dates.
# A substituted CSV only has to supply state wages; treatment is defined
# against the federal floor, so we fill it from the statute rather than
# leaving it missing (a missing floor makes `above_federal` undefined and
# the panel builder raise).
FEDERAL_MINIMUM_WAGE_SCHEDULE = [
    (1974, 5, 1, 2.00),
    (1975, 1, 1, 2.10),
    (1976, 1, 1, 2.30),
    (1978, 1, 1, 2.65),
    (1979, 1, 1, 2.90),
    (1980, 1, 1, 3.10),
    (1981, 1, 1, 3.35),
    (1990, 4, 1, 3.80),
    (1991, 4, 1, 4.25),
    (1996, 10, 1, 4.75),
    (1997, 9, 1, 5.15),
    (2007, 7, 24, 5.85),
    (2008, 7, 24, 6.55),
    (2009, 7, 24, 7.25),
]


def federal_minimum_wage(year, month):
    """The federal floor in force on the first day of a given year-month.

    Vectorized over pandas Series; also accepts scalars. Anchoring on the
    first of the month means the three FLSA amendments that took effect on
    July 24 (2007-2009) count from August, which is how the Vaghul-Zipperer
    series dates them too — so a substituted CSV and the default download
    define treatment identically.

    Year-months before the schedule starts return NaN.
    """
    year = pd.Series(year).astype(int).reset_index(drop=True)
    month = pd.Series(month).astype(int).reset_index(drop=True)
    stamp = year * 12 + month

    out = pd.Series(float("nan"), index=stamp.index)
    for eff_year, eff_month, eff_day, wage in FEDERAL_MINIMUM_WAGE_SCHEDULE:
        # An increase mid-month first binds a whole month the month after.
        effective = eff_year * 12 + eff_month + (1 if eff_day > 1 else 0)
        out = out.mask(stamp >= effective, wage)
    return out


def download_vz_minimum_wage(dest=None, force=False):
    """Download and cache the Vaghul-Zipperer monthly state minimum wage file.

    Returns the local path to the extracted .xlsx.
    """
    import io
    import zipfile

    dest = Path(dest) if dest else VZ_LOCAL
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(VZ_URL, timeout=180)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        if VZ_MEMBER not in z.namelist():
            raise RuntimeError(
                f"{VZ_MEMBER} not found in {VZ_URL} (contains: {z.namelist()})"
            )
        dest.write_bytes(z.read(VZ_MEMBER))
    return dest


def _parse_vz_monthly(path):
    """Reshape the VZ monthly workbook into state, year, month, minimum_wage.

    `Monthly State Minimum` is the state's effective floor (already the
    higher of the state and federal rate where a state has no law of its
    own); we still take the max against the federal column defensively, so
    `minimum_wage` is always the wage that actually binds in that state.
    """
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    # "Monthly Date" looks like " 2020m1" — leading space, unpadded month.
    dates = df["Monthly Date"].astype(str).str.strip().str.split("m", expand=True)
    df["year"] = dates[0].astype(int)
    df["month"] = dates[1].astype(int)
    df["state"] = df["State Abbreviation"].str.upper().str.strip()
    # The workbook stores wages at float32 precision (5.15 round-trips as
    # 5.150000095367432), so round to cents before anything compares them.
    df["federal_minimum_wage"] = df["Monthly Federal Minimum"].astype(float).round(2)
    df["minimum_wage"] = df[
        ["Monthly State Minimum", "Monthly Federal Minimum"]
    ].max(axis=1).astype(float).round(2)

    out = df[
        ["state", "year", "month", "minimum_wage", "federal_minimum_wage"]
    ].copy()
    return out.sort_values(["state", "year", "month"]).reset_index(drop=True)


def _load_user_csv(raw_path):
    df = pd.read_csv(raw_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{raw_path} is missing required columns: {missing}")
    df["state"] = df["state"].str.upper().str.strip()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["minimum_wage"] = df["minimum_wage"].astype(float)
    df = df.sort_values(["state", "year", "month"]).reset_index(drop=True)
    if "federal_minimum_wage" in df.columns:
        df["federal_minimum_wage"] = pd.to_numeric(
            df["federal_minimum_wage"], errors="coerce"
        )
        supplied = df["federal_minimum_wage"]
        df["federal_minimum_wage"] = supplied.fillna(
            federal_minimum_wage(df["year"], df["month"])
        )
    else:
        df["federal_minimum_wage"] = federal_minimum_wage(df["year"], df["month"])
    if df["federal_minimum_wage"].isna().any():
        first = df[df["federal_minimum_wage"].isna()].iloc[0]
        raise ValueError(
            f"{raw_path} covers {int(first['year'])}m{int(first['month'])}, which "
            "predates the FLSA schedule in FEDERAL_MINIMUM_WAGE_SCHEDULE. Supply "
            "a federal_minimum_wage column for those rows."
        )
    # The wage that actually binds is the higher of the two.
    df["minimum_wage"] = df[["minimum_wage", "federal_minimum_wage"]].max(axis=1)
    cols = ["state", "year", "month", "minimum_wage", "federal_minimum_wage"]
    return df[cols].sort_values(["state", "year", "month"]).reset_index(drop=True)


def load_minimum_wage(raw_path=None):
    """Return a tidy state-month minimum wage panel.

    Uses `data/raw/state_minimum_wage.csv` if it exists, otherwise
    downloads (and caches) the Vaghul-Zipperer series.
    """
    if raw_path is not None:
        return _load_user_csv(Path(raw_path))

    user_csv = RAW_DIR / "state_minimum_wage.csv"
    if user_csv.exists():
        return _load_user_csv(user_csv)

    return _parse_vz_monthly(download_vz_minimum_wage())


def main():
    out = load_minimum_wage()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "minimum_wage.parquet"
    out.to_parquet(out_path, index=False)
    print(
        f"Wrote {len(out)} rows ({out['state'].nunique()} states, "
        f"{out['year'].min()}-{out['year'].max()}) to {out_path}"
    )

    # Which of the two sources this was is not recoverable from the output.
    user_csv = RAW_DIR / "state_minimum_wage.csv"
    write_manifest(extra={"minimum_wage": {
        "source": "user csv" if user_csv.exists() else f"Vaghul-Zipperer {VZ_URL}",
        "start_year": int(out["year"].min()),
        "end_year": int(out["year"].max()),
        "n_states": int(out["state"].nunique()),
    }})
    print(f"Recorded data vintage in {MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()
