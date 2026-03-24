import pandas as pd
from models import StockMetadata


class SP500UniverseProvider:
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def get_universe(self) -> list[StockMetadata]:
        tables = pd.read_html(self.WIKI_URL)
        df = tables[0].copy()

        results = []
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
