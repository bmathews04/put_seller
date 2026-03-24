from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from models import StockMetadata


class SP500UniverseProvider:
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    LOCAL_CSV = Path("data/sp500_constituents.csv")

    def get_universe(self) -> list[StockMetadata]:
        df = self._load_dataframe()

        results: list[StockMetadata] = []
        for _, row in df.iterrows():
            symbol = str(row["Symbol"]).strip().replace(".", "-")
            results.append(
                StockMetadata(
                    symbol=symbol,
                    company_name=row.get("Security"),
                    sector=row.get("GICS Sector"),
                    industry=row.get("GICS Sub-Industry"),
                    is_sp500_member=True,
                )
            )
        return results

    def _load_dataframe(self) -> pd.DataFrame:
        if self.LOCAL_CSV.exists():
            return pd.read_csv(self.LOCAL_CSV)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(self.WIKI_URL, headers=headers, timeout=20)
        resp.raise_for_status()

        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            raise ValueError("No tables found on S&P 500 source page.")

        df = tables[0].copy()

        # Cache locally if possible
        self.LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.LOCAL_CSV, index=False)

        return df
