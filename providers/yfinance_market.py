from datetime import date, datetime
from functools import wraps

import pandas as pd

try:  # pragma: no cover - exercised indirectly in integration runs
    import yfinance as yf
except ModuleNotFoundError:  # pragma: no cover - allows unit tests without yfinance installed
    from types import SimpleNamespace

    yf = SimpleNamespace(Ticker=None)

from config import ScanConfig
from greeks import black_scholes_put_delta
from models import OptionContract, StockMetrics
from providers.yfinance_fundamentals import YFinanceFundamentalsProvider

try:  # pragma: no cover - exercised indirectly in app environments
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised in test/non-UI environments
    class _StreamlitFallback:
        @staticmethod
        def cache_data(ttl=None, show_spinner=False):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)

                return wrapper

            return decorator

    st = _StreamlitFallback()


# ----------------------------
# Pure fetch helpers (no Streamlit dependency in behavior)
# ----------------------------
def _require_yfinance() -> None:
    if getattr(yf, "Ticker", None) is None:
        raise ModuleNotFoundError("yfinance is required for live market data fetches")


def _fetch_price_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    _require_yfinance()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=False)
    return hist.copy()


def _fetch_stock_metrics(symbol: str):
    provider = YFinanceFundamentalsProvider()
    return provider.get_stock_metrics(symbol)


def _fetch_option_chain(symbol: str, exp_str: str) -> pd.DataFrame:
    _require_yfinance()
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(exp_str)
    return chain.puts.copy()


# ----------------------------
# Streamlit-cached wrappers
# ----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _cached_price_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    return _fetch_price_history(symbol, period, interval)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_stock_metrics(symbol: str):
    return _fetch_stock_metrics(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_option_chain(symbol: str, exp_str: str) -> pd.DataFrame:
    return _fetch_option_chain(symbol, exp_str)


class YFinanceMarketProvider:
    def __init__(self, risk_free_rate: float = 0.045, use_cache: bool = True) -> None:
        self.fundamentals = YFinanceFundamentalsProvider()
        self.risk_free_rate = risk_free_rate
        self.use_cache = use_cache
        self.last_errors: list[str] = []

    def get_price_history(self, symbol: str, period: str = "6mo", interval: str = "1d"):
        try:
            if self.use_cache:
                return _cached_price_history(symbol, period, interval)
            return _fetch_price_history(symbol, period, interval)
        except Exception as e:
            self.last_errors.append(
                f"price_history {symbol} period={period} interval={interval}: {type(e).__name__}: {e}"
            )
            raise

    def get_stock_metrics(self, symbol: str):
        try:
            if self.use_cache:
                return _cached_stock_metrics(symbol)
            return _fetch_stock_metrics(symbol)
        except Exception as e:
            self.last_errors.append(f"stock_metrics {symbol}: {type(e).__name__}: {e}")
            raise

    def get_option_contracts(
        self,
        symbol: str,
        cfg: ScanConfig | None = None,
        stock_metrics: StockMetrics | None = None,
    ) -> list[OptionContract]:
        self.last_errors = []

        _require_yfinance()
        ticker = yf.Ticker(symbol)

        # Reuse caller-supplied metrics when available to avoid duplicate fetches
        metrics = stock_metrics if stock_metrics is not None else self.get_stock_metrics(symbol)
        underlying_price = metrics.stock_price

        try:
            expirations = ticker.options or []
        except Exception as e:
            self.last_errors.append(f"{symbol} expirations: {type(e).__name__}: {e}")
            return []

        contracts: list[OptionContract] = []

        for exp_str in expirations:
            try:
                exp_date = date.fromisoformat(exp_str)
            except ValueError:
                self.last_errors.append(f"{symbol} {exp_str}: invalid expiration format")
                continue

            dte = (exp_date - date.today()).days
            if dte <= 0:
                continue

            # Early DTE filter to reduce workload
            if cfg is not None and (dte < cfg.min_dte or dte > cfg.max_dte):
                continue

            try:
                if self.use_cache:
                    puts: pd.DataFrame = _cached_option_chain(symbol, exp_str)
                else:
                    puts = _fetch_option_chain(symbol, exp_str)
            except Exception as e:
                self.last_errors.append(f"{symbol} {exp_str}: {type(e).__name__}: {e}")
                continue

            if puts.empty:
                self.last_errors.append(f"{symbol} {exp_str}: empty_put_chain")
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
                    try:
                        delta_est = black_scholes_put_delta(
                            spot=underlying_price,
                            strike=strike,
                            dte=dte,
                            implied_volatility=implied_vol,
                            risk_free_rate=self.risk_free_rate,
                        )
                    except Exception as e:
                        self.last_errors.append(
                            f"{symbol} {exp_str} strike={strike}: delta_calc_failed: {type(e).__name__}: {e}"
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
