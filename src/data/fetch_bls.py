"""Pull state-level LAUS series (unemployment rate, labour force) from BLS.

No API key is required. The public v1 endpoint is keyless and is enough to
build the full 51-state panel (see LIMITS below). If you do set BLS_API_KEY
(free, from https://www.bls.gov/developers/) this module automatically uses
the v2 endpoint instead, which has looser per-request limits and a much
higher daily quota.

Raw JSON responses are cached under data/raw/bls_laus_raw/, so re-running is
free and does not re-spend your daily request quota.

Two measures are pulled. The unemployment rate is the outcome. The labour
force level is not used by any estimator by default -- every design in this
repo weights states equally -- but it is what makes population weighting
*possible*, so that choice can be tested rather than merely inherited. See
`src/methods/twfe_did.estimate_twfe` on what the two weightings mean.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Per-request limits differ by API version. Exceeding them makes the API
# return a soft error rather than truncating, so we batch to fit.
LIMITS = {
    # version: (url, max series per request, max years per request)
    1: ("https://api.bls.gov/publicAPI/v1/timeseries/data/", 25, 10),
    2: ("https://api.bls.gov/publicAPI/v2/timeseries/data/", 50, 20),
}

# LAUS series ID format: LASST{state_fips}00000000000{measure}, seasonally
# adjusted, statewide. Measure 03 is the unemployment rate, 06 the civilian
# labour force level.
LAUS_MEASURES = {
    "unemployment_rate": "03",
    "labor_force": "06",
}

STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56", "DC": "11",
}

DEFAULT_START_YEAR = 2000
DEFAULT_END_YEAR = 2022  # minimum wage source (Vaghul-Zipperer) ends 2022m12


def _series_id(state_abbr, measure="03"):
    return f"LASST{STATE_FIPS[state_abbr]}00000000000{measure}"


def _spans(start_year, end_year, max_years):
    """Year ranges no longer than `max_years`, covering [start_year, end_year]."""
    return [
        (s, min(s + max_years - 1, end_year))
        for s in range(start_year, end_year + 1, max_years)
    ]


def fetch_laus(
    measure="unemployment_rate",
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    states=None,
    api_key=None,
    cache_dir=None,
):
    """Fetch one monthly LAUS measure for [start_year, end_year].

    `measure` is a key of `LAUS_MEASURES`. Returns a long DataFrame:
    state, year, month, <measure>.

    `api_key` defaults to BLS_API_KEY from the environment; if neither is
    set, the keyless v1 endpoint is used.
    """
    if measure not in LAUS_MEASURES:
        raise ValueError(
            f"unknown LAUS measure {measure!r}; expected one of "
            f"{sorted(LAUS_MEASURES)}"
        )
    measure_code = LAUS_MEASURES[measure]
    api_key = api_key or os.environ.get("BLS_API_KEY") or None
    version = 2 if api_key else 1
    url, max_series, max_years = LIMITS[version]

    states = states or list(STATE_FIPS.keys())
    cache_dir = Path(cache_dir) if cache_dir else RAW_DIR / "bls_laus_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    series_ids = [_series_id(s, measure_code) for s in states]
    id_to_state = {_series_id(s, measure_code): s for s in states}

    frames = []
    for batch_start in range(0, len(series_ids), max_series):
        batch = series_ids[batch_start:batch_start + max_series]
        for span_start, span_end in _spans(start_year, end_year, max_years):
            stem = f"v{version}_batch{batch_start}_{span_start}_{span_end}"
            cache_file = cache_dir / f"{stem}_m{measure_code}.json"
            # Caches written before this module handled more than one measure
            # have no measure suffix and are all unemployment rate.
            legacy = cache_dir / f"{stem}.json"
            if not cache_file.exists() and measure_code == "03" and legacy.exists():
                cache_file = legacy
            if cache_file.exists():
                payload = json.loads(cache_file.read_text())
            else:
                body = {
                    "seriesid": batch,
                    "startyear": str(span_start),
                    "endyear": str(span_end),
                }
                if api_key:
                    body["registrationkey"] = api_key
                resp = requests.post(url, json=body, timeout=60)
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") != "REQUEST_SUCCEEDED":
                    raise RuntimeError(
                        f"BLS API v{version} error: {payload.get('message')}"
                    )
                cache_file.write_text(json.dumps(payload))
                time.sleep(0.5)  # be polite to the API

            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API v{version} error: {payload.get('message')}")

            for series in payload["Results"]["series"]:
                state = id_to_state[series["seriesID"]]
                for obs in series["data"]:
                    frames.append({
                        "state": state,
                        "year": int(obs["year"]),
                        "period": obs["period"],  # M01..M12 (M13 = annual avg)
                        "value": float(obs["value"]),
                    })

    df = pd.DataFrame(frames)
    if df.empty:
        raise RuntimeError("BLS returned no observations")
    df = df[df["period"].str.startswith("M") & (df["period"] != "M13")]
    df["month"] = df["period"].str[1:].astype(int)
    df = df.rename(columns={"value": measure})
    df = df.drop_duplicates(subset=["state", "year", "month"])
    return df[["state", "year", "month", measure]].sort_values(
        ["state", "year", "month"]
    ).reset_index(drop=True)


def fetch_unemployment(**kwargs):
    """Backwards-compatible alias for the unemployment rate measure."""
    return fetch_laus("unemployment_rate", **kwargs)


def main():
    key = os.environ.get("BLS_API_KEY")
    print(
        f"Fetching LAUS {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR} for "
        f"{len(STATE_FIPS)} states via API "
        f"{'v2 (key found)' if key else 'v1 (keyless)'}..."
    )
    out = fetch_laus("unemployment_rate")
    # Labour force is the weight, not an outcome. If BLS declines it, the
    # panel is still complete -- every estimator defaults to equal weights.
    try:
        labor = fetch_laus("labor_force")
        out = out.merge(labor, on=["state", "year", "month"], how="left")
        print(f"Merged labour force levels ({out['labor_force'].notna().sum()} rows)")
    except (requests.RequestException, RuntimeError) as exc:
        print(f"Labour force fetch failed, continuing unweighted: {exc}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "bls_unemployment.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows ({out['state'].nunique()} states) to {out_path}")


if __name__ == "__main__":
    main()
