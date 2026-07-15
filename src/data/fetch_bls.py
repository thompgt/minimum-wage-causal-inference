"""Pull state-level unemployment rate (LAUS) series from the BLS public API.

Requires a free API key from https://www.bls.gov/developers/, set as
BLS_API_KEY in a .env file at the project root.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# LAUS series ID format: LASST{state_fips}0000000000003 = unemployment rate
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


def _series_id(state_abbr):
    return f"LASST{STATE_FIPS[state_abbr]}0000000000003"


def fetch_unemployment(start_year, end_year, states=None, api_key=None):
    """Fetch monthly state unemployment rates for [start_year, end_year].

    Returns a long DataFrame: state, year, period, value.
    Caches raw JSON responses per batch to data/raw/bls_laus_raw/.
    """
    api_key = api_key or os.environ.get("BLS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BLS_API_KEY not set. Copy .env.example to .env and add your key "
            "from https://www.bls.gov/developers/."
        )
    states = states or list(STATE_FIPS.keys())
    cache_dir = RAW_DIR / "bls_laus_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    series_ids = [_series_id(s) for s in states]
    id_to_state = {_series_id(s): s for s in states}

    frames = []
    # BLS API allows up to 50 series and a 20-year span per request with a key.
    for batch_start in range(0, len(series_ids), 50):
        batch = series_ids[batch_start:batch_start + 50]
        for span_start in range(start_year, end_year + 1, 20):
            span_end = min(span_start + 19, end_year)
            cache_file = cache_dir / f"batch{batch_start}_{span_start}_{span_end}.json"
            if cache_file.exists():
                payload = json.loads(cache_file.read_text())
            else:
                resp = requests.post(
                    BLS_API_URL,
                    json={
                        "seriesid": batch,
                        "startyear": str(span_start),
                        "endyear": str(span_end),
                        "registrationkey": api_key,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()
                cache_file.write_text(json.dumps(payload))
                time.sleep(0.5)  # be polite to the API

            if payload.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS API error: {payload.get('message')}")

            for series in payload["Results"]["series"]:
                state = id_to_state[series["seriesID"]]
                for obs in series["data"]:
                    frames.append({
                        "state": state,
                        "year": int(obs["year"]),
                        "period": obs["period"],  # e.g. M01..M12
                        "value": float(obs["value"]),
                    })

    df = pd.DataFrame(frames)
    df = df[df["period"].str.startswith("M") & (df["period"] != "M13")]
    df["month"] = df["period"].str[1:].astype(int)
    df = df.rename(columns={"value": "unemployment_rate"})
    return df[["state", "year", "month", "unemployment_rate"]].sort_values(
        ["state", "year", "month"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    out = fetch_unemployment(start_year=2000, end_year=2024)
    out_path = RAW_DIR / "bls_unemployment.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
