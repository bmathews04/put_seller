from datetime import date, datetime

import pandas as pd
import yfinance as yf

from models import OptionContract
from providers.yfinance_fundamentals import YFinanceFundamentalsProvider


class YFinanceMarketProvider:
    def __init__(self) -> None:
        self.fundamentals = YFinanceFundamentalsProvider()

    def get_stock_metrics(self, symbol: str):
        return self.fundamentals.get_stock_metrics(symbol)

    def get_option_contracts(self, symbol: str) -> list[OptionContract]:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options or []
        contracts: list[OptionContract] = []

        for exp_str in expirations:
            try:
                exp_date = date.fromisoformat(exp_str)
            except ValueError:
                continue

            dte = (exp_date - date.today()).days
            if dte <= 0:
                continue

            try:
                chain = ticker.option_chain(exp_str)
                puts: pd.DataFrame = chain.puts.copy()
            except Exception:
                continue

            if puts.empty:
                continue

            for _, row in puts.iterrows():
                implied_vol = row.get("impliedVolatility")
                delta_est = self._estimate_delta_from_chain_row(
                    strike=row.get("strike"),
                    underlying_price=None,
                    implied_vol=implied_vol,
                    dte=dte,
                )

                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiration_date=exp_date,
                        dte=dte,
                        strike=float(row.get("strike")),
                        option_type="PUT",
                        bid=self._safe_float(row.get("bid")),
                        ask=self._safe_float(row.get("ask")),
                        last=self._safe_float(row.get("lastPrice")),
                        mark=None,
                        delta=delta_est,
                        gamma=None,
                        theta=None,
                        vega=None,
                        implied_volatility=self._safe_float(implied_vol),
                        open_interest=self._safe_int(row.get("openInterest")),
                        volume=self._safe_int(row.get("volume")),
                        in_the_money=bool(row.get("inTheMoney", False)),
                        quote_timestamp=datetime.now(),
                    )
                )
        return contracts

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _safe_int(value):
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _estimate_delta_from_chain_row(strike, underlying_price, implied_vol, dte):
        # Placeholder.
        # yfinance option chains do not reliably provide greeks directly in a simple universal way.
        # For now we return None and rely on a later enhancement if needed.
        return None
