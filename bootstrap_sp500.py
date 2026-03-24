from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = Path("data/sp500_constituents.csv")


def fetch_sp500_table() -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise ValueError("No tables found on the S&P 500 source page.")

    df = tables[0].copy()

    required_cols = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[required_cols].copy()

    # Normalize symbols for downstream API compatibility
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)

    # Remove duplicates just in case
    df = df.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    return df


def main() -> None:
    df = fetch_sp500_table()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
