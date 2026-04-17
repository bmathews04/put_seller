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
    try:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_get(mapping, key, default=None):
    try:
        if mapping is None:
            return default
        if hasattr(mapping, "get"):
            return mapping.get(key, default)
        return default
    except Exception:
        return default


class YFinanceFundamentalsProvider:
    """
    Much lighter and more fault-tolerant than eager ticker.info usage.

    Priority:
    1. fast_info for price / market cap
    2. recent history fallback for price
    3. calendar / earnings_dates for earnings
    4. info only as a *best-effort* fallback for fields like freeCashflow,
       and never as a required step
    """

    def _get_price_from_fast_info(self, ticker):
        try:
            fi = getattr(ticker, "fast_info", None)
            if fi is None:
                return None

            for key in (
                "lastPrice",
                "regularMarketPrice",
                "previousClose",
                "open",
            ):
                value = _safe_get(fi, key)
                value = _safe_float(value)
                if value is not None and value > 0:
                    return value
        except Exception:
            pass
        return None

    def _get_price_from_history(self, ticker):
        history_attempts = [
            {"period": "5d", "interval": "1d"},
            {"period": "1mo", "interval": "1d"},
        ]

        for params in history_attempts:
            try:
                hist = ticker.history(
                    period=params["period"],
                    interval=params["interval"],
                    auto_adjust=False,
                )
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].dropna()
                    if not closes.empty:
                        price = _safe_float(closes.iloc[-1])
                        if price is not None and price > 0:
                            return price
            except Exception:
                continue

        return None

    def _get_earnings_date(self, ticker):
        earnings_date = None

        # First try calendar
        try:
            cal = getattr(ticker, "calendar", None)
            if cal is not None and hasattr(cal, "loc") and "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"]
                if hasattr(raw, "__iter__") and not isinstance(raw, str):
                    raw_list = list(raw)
                    if raw_list:
                        earnings_date = _safe_date(raw_list[0])
                else:
                    earnings_date = _safe_date(raw)
        except Exception:
            earnings_date = None

        # Fallback to earnings_dates dataframe
        if earnings_date is None:
            try:
                edf = getattr(ticker, "earnings_dates", None)
                if edf is not None and len(edf) > 0:
                    idx0 = edf.index[0]
                    earnings_date = _safe_date(idx0)
            except Exception:
                pass

        return earnings_date

    def _get_market_cap(self, ticker):
        # Prefer fast_info
        try:
            fi = getattr(ticker, "fast_info", None)
            market_cap = _safe_float(_safe_get(fi, "marketCap"))
            if market_cap is not None and market_cap > 0:
                return market_cap
        except Exception:
            pass

        # Best-effort fallback to info
        try:
            info = ticker.info or {}
            market_cap = _safe_float(info.get("marketCap"))
            if market_cap is not None and market_cap > 0:
                return market_cap
        except Exception:
            pass

        return None

    def _get_free_cashflow(self, ticker):
        # Only best-effort via info. Never required.
        try:
            info = ticker.info or {}
            free_cashflow = _safe_float(info.get("freeCashflow"))
            return free_cashflow
        except Exception:
            return None

    def get_stock_metrics(self, symbol: str) -> StockMetrics:
        _require_yfinance()
        ticker = yf.Ticker(symbol)

        price = self._get_price_from_fast_info(ticker)
        if price is None:
            price = self._get_price_from_history(ticker)

        earnings_date = self._get_earnings_date(ticker)
        market_cap = self._get_market_cap(ticker)
        free_cashflow = self._get_free_cashflow(ticker)

        fcf_yield = None
        if free_cashflow is not None and market_cap is not None and market_cap > 0:
            fcf_yield = free_cashflow / market_cap

        days_to_earnings = None
        if earnings_date is not None:
            try:
                days_to_earnings = (earnings_date - date.today()).days
            except Exception:
                days_to_earnings = None

        # Important: do not hard-fail if price/fundamentals are incomplete.
        # Return a usable StockMetrics object and let downstream logic decide.
        return StockMetrics(
            symbol=symbol,
            stock_price=price,
            price_timestamp=datetime.now() if price is not None else None,
            earnings_date=earnings_date,
            days_to_earnings=days_to_earnings,
            fcf_yield=fcf_yield,
            quality_data_source="yfinance_fast",
            quality_data_complete=fcf_yield is not None,
        )
