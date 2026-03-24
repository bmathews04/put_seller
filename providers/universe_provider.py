from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import StockMetadata


class SP500UniverseProvider:
    LOCAL_CSV = Path("data/sp500_constituents.csv")

    def get_universe(self) -> list[StockMetadata]:
        if not self.LOCAL_CSV.exists():
            raise FileNotFoundError(
                f"{self.LOCAL_CSV} not found. Run bootstrap_sp500.py locally first "
                "and commit the generated CSV to the repo."
            )

        df = pd.read_csv(self.LOCAL_CSV)

        required_cols = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.LOCAL_CSV} is missing required columns: {missing}"
            )

        results: list[StockMetadata] = []
        for _, row in df.iterrows():
            symbol = str(row["Symbol"]).strip().replace(".", "-", regex=False)
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
