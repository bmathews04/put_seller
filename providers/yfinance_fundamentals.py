from datetime import date, datetime

import yfinance as yf

from models import StockMetrics


def _safe_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


class YFinanceFundamentalsProvider:
    def get_stock_metrics(self, symbol: str) -> StockMetrics:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        earnings_date = None

        # yfinance can return earnings dates in different shapes depending on symbol/data availability
        cal = getattr(ticker, "calendar", None)
        if cal is not None:
            try:
                if hasattr(cal, "loc") and "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"][0]
                    earnings_date = _safe_date(val)
            except Exception:
                earnings_date = None

        market_cap = info.get("marketCap")
        free_cashflow = info.get("freeCashflow")

        fcf_yield = None
        if free_cashflow and market_cap and market_cap > 0:
            fcf_yield = free_cashflow / market_cap

        days_to_earnings = None
        if earnings_date is not None:
            days_to_earnings = (earnings_date - date.today()).days

        return StockMetrics(
            symbol=symbol,
            stock_price=price,
            price_timestamp=datetime.now(),
            earnings_date=earnings_date,
            days_to_earnings=days_to_earnings,
            fcf_yield=fcf_yield,
            quality_data_source="yfinance",
            quality_data_complete=fcf_yield is not None,
        )
