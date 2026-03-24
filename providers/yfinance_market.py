from datetime import date, datetime

import pandas as pd
import yfinance as yf

from greeks import black_scholes_put_delta
from models import OptionContract
from providers.yfinance_fundamentals import YFinanceFundamentalsProvider


class YFinanceMarketProvider:
    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.fundamentals = YFinanceFundamentalsProvider()
        self.risk_free_rate = risk_free_rate

    def get_stock_metrics(self, symbol: str):
        return self.fundamentals.get_stock_metrics(symbol)

    def get_option_contracts(self, symbol: str) -> list[OptionContract]:
        ticker = yf.Ticker(symbol)

        # Pull underlying price once so delta estimation is consistent across expirations
        stock_metrics = self.get_stock_metrics(symbol)
        underlying_price = stock_metrics.stock_price

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
                strike = self._safe_float(row.get("strike"))
                bid = self._safe_float(row.get("bid"))
                ask = self._safe_float(row.get("ask"))
                last = self._safe_float(row.get("lastPrice"))
                implied_vol = self._safe_float(row.get("impliedVolatility"))
                open_interest = self._safe_int(row.get("openInterest"))
                volume = self._safe_int(row.get("volume"))
                in_the_money = bool(row.get("inTheMoney", False))

                delta_est = None
                if (
                    underlying_price is not None
                    and strike is not None
                    and implied_vol is not None
                    and implied_vol > 0
                    and dte > 0
                ):
                    delta_est = black_scholes_put_delta(
                        spot=underlying_price,
                        strike=strike,
                        dte=dte,
                        implied_volatility=implied_vol,
                        risk_free_rate=self.risk_free_rate,
                    )

                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiration_date=exp_date,
                        dte=dte,
                        strike=strike if strike is not None else 0.0,
                        option_type="PUT",
                        bid=bid,
                        ask=ask,
                        last=last,
                        mark=None,
                        delta=delta_est,
                        gamma=None,
                        theta=None,
                        vega=None,
                        implied_volatility=implied_vol,
                        open_interest=open_interest,
                        volume=volume,
                        in_the_money=in_the_money,
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
