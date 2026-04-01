from datetime import date, datetime

import pandas as pd

from models import StockMetrics

try:  # pragma: no cover - exercised indirectly in integration runs
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - allows unit tests without yfinance installed
    from types import SimpleNamespace

    yf = SimpleNamespace(Ticker=None)


def _require_yfinance() -> None:
    if getattr(yf, "Ticker", None) is None:
        raise ModuleNotFoundError("yfinance is required for live fundamentals fetches")


def _safe_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


class YFinanceFundamentalsProvider:
    def get_stock_metrics(self, symbol: str) -> StockMetrics:
        _require_yfinance()
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        earnings_date = None

        # First try calendar
        cal = getattr(ticker, "calendar", None)
        if cal is not None:
            try:
                if hasattr(cal, "loc") and "Earnings Date" in cal.index:
                    raw = cal.loc["Earnings Date"]
                    if hasattr(raw, "__iter__") and not isinstance(raw, str):
                        first = list(raw)[0]
                    else:
                        first = raw
                    earnings_date = _safe_date(first)
            except Exception:
                earnings_date = None

        # Fallback to earnings_dates dataframe if available
        if earnings_date is None:
            try:
                edf = getattr(ticker, "earnings_dates", None)
                if edf is not None and len(edf) > 0:
                    idx0 = edf.index[0]
                    earnings_date = _safe_date(idx0)
            except Exception:
                pass

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
