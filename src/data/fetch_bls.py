"""Pull state-level unemployment rate (LAUS) series from the BLS public API.

No API key is required. The public v1 endpoint is keyless and is enough to
build the full 51-state panel (see LIMITS below). If you do set BLS_API_KEY
(free, from https://www.bls.gov/developers/) this module automatically uses
the v2 endpoint instead, which has looser per-request limits and a much
higher daily quota.

Raw JSON responses are cached under data/raw/bls_laus_raw/, so re-running is
free and does not re-spend your daily request quota.
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

# LAUS series ID format: LASST{state_fips}0000000000003 = unemployment rate,
# seasonally adjusted, statewide.
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


def _series_id(state_abbr):
    return f"LASST{STATE_FIPS[state_abbr]}0000000000003"


def _spans(start_year, end_year, max_years):
    """Year ranges no longer than `max_years`, covering [start_year, end_year]."""
    return [
        (s, min(s + max_years - 1, end_year))
        for s in range(start_year, end_year + 1, max_years)
    ]


def fetch_unemployment(
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    states=None,
    api_key=None,
    cache_dir=None,
):
    """Fetch monthly state unemployment rates for [start_year, end_year].

    Returns a long DataFrame: state, year, month, unemployment_rate.

    `api_key` defaults to BLS_API_KEY from the environment; if neither is
    set, the keyless v1 endpoint is used.
    """
    api_key = api_key or os.environ.get("BLS_API_KEY") or None
    version = 2 if api_key else 1
    url, max_series, max_years = LIMITS[version]

    states = states or list(STATE_FIPS.keys())
    cache_dir = Path(cache_dir) if cache_dir else RAW_DIR / "bls_laus_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    series_ids = [_series_id(s) for s in states]
    id_to_state = {_series_id(s): s for s in states}

    frames = []
    for batch_start in range(0, len(series_ids), max_series):
        batch = series_ids[batch_start:batch_start + max_series]
        for span_start, span_end in _spans(start_year, end_year, max_years):
            cache_file = (
                cache_dir / f"v{version}_batch{batch_start}_{span_start}_{span_end}.json"
            )
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
    df = df.rename(columns={"value": "unemployment_rate"})
    df = df.drop_duplicates(subset=["state", "year", "month"])
    return df[["state", "year", "month", "unemployment_rate"]].sort_values(
        ["state", "year", "month"]
    ).reset_index(drop=True)


def main():
    key = os.environ.get("BLS_API_KEY")
    print(
        f"Fetching LAUS {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR} for "
        f"{len(STATE_FIPS)} states via API "
        f"{'v2 (key found)' if key else 'v1 (keyless)'}..."
    )
    out = fetch_unemployment()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "bls_unemployment.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows ({out['state'].nunique()} states) to {out_path}")


if __name__ == "__main__":
    main()
