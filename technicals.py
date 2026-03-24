import math
import numpy as np
import pandas as pd


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _to_numeric_series(series)

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = _to_numeric_series(df["High"])
    low = _to_numeric_series(df["Low"])
    close = _to_numeric_series(df["Close"])

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    close = _to_numeric_series(close)
    log_returns = np.log(close / close.shift(1))
    rv = log_returns.rolling(window, min_periods=window).std() * math.sqrt(252)
    return rv


def classify_trend(close: float, ma20: float | None, ma50: float | None) -> str:
    if close is None or ma20 is None or ma50 is None:
        return "Unknown"

    if close > ma20 and close > ma50 and ma20 > ma50:
        return "Bullish"
    if close < ma20 and close < ma50 and ma20 < ma50:
        return "Weak"
    return "Neutral"


def choose_support_zone(support_20d: float | None, support_50d: float | None) -> float | None:
    candidates = [x for x in [support_20d, support_50d] if x is not None]
    if not candidates:
        return None
    return max(candidates)


def build_technical_context(
    hist: pd.DataFrame,
    stock_price: float | None,
    breakeven_price: float | None,
    implied_volatility: float | None = None,
) -> dict:
    if hist is None or hist.empty or stock_price is None:
        return {
            "ma20": None,
            "ma50": None,
            "atr14": None,
            "rsi14": None,
            "support_20d": None,
            "support_50d": None,
            "support_zone": None,
            "resistance_20d": None,
            "distance_to_support_pct": None,
            "breakeven_vs_support": None,
            "cushion_atr_units": None,
            "realized_vol_20d": None,
            "implied_volatility": implied_volatility,
            "iv_rv_ratio": None,
            "trend_state": "Unknown",
        }

    hist = hist.copy()

    for col in ["Close", "High", "Low"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    hist = hist.dropna(subset=["Close", "High", "Low"])

    if hist.empty:
        return {
            "ma20": None,
            "ma50": None,
            "atr14": None,
            "rsi14": None,
            "support_20d": None,
            "support_50d": None,
            "support_zone": None,
            "resistance_20d": None,
            "distance_to_support_pct": None,
            "breakeven_vs_support": None,
            "cushion_atr_units": None,
            "realized_vol_20d": None,
            "implied_volatility": implied_volatility,
            "iv_rv_ratio": None,
            "trend_state": "Unknown",
        }

    close = hist["Close"]
    low = hist["Low"]
    high = hist["High"]

    ma20 = close.rolling(20, min_periods=20).mean().iloc[-1] if len(close) >= 20 else None
    ma50 = close.rolling(50, min_periods=50).mean().iloc[-1] if len(close) >= 50 else None
    atr14 = compute_atr(hist, 14).iloc[-1] if len(hist) >= 14 else None
    rsi14 = compute_rsi(close, 14).iloc[-1] if len(close) >= 14 else None
    rv20 = compute_realized_vol(close, 20).iloc[-1] if len(close) >= 20 else None

    support_20d = low.tail(20).min() if len(low) >= 20 else low.min()
    support_50d = low.tail(50).min() if len(low) >= 50 else low.min()
    support_zone = choose_support_zone(
        float(support_20d) if support_20d is not None and not pd.isna(support_20d) else None,
        float(support_50d) if support_50d is not None and not pd.isna(support_50d) else None,
    )
    resistance_20d = high.tail(20).max() if len(high) >= 20 else high.max()

    distance_to_support_pct = None
    if support_zone is not None and stock_price and stock_price > 0:
        distance_to_support_pct = (stock_price - support_zone) / stock_price

    breakeven_vs_support = None
    if breakeven_price is not None and support_zone is not None:
        breakeven_vs_support = breakeven_price - support_zone

    cushion_atr_units = None
    if breakeven_price is not None and atr14 is not None and atr14 > 0:
        cushion_atr_units = (stock_price - breakeven_price) / atr14

    iv_rv_ratio = None
    if implied_volatility is not None and rv20 is not None and rv20 > 0:
        iv_rv_ratio = implied_volatility / rv20

    trend_state = classify_trend(stock_price, ma20, ma50)

    return {
        "ma20": float(ma20) if ma20 is not None and not pd.isna(ma20) else None,
        "ma50": float(ma50) if ma50 is not None and not pd.isna(ma50) else None,
        "atr14": float(atr14) if atr14 is not None and not pd.isna(atr14) else None,
        "rsi14": float(rsi14) if rsi14 is not None and not pd.isna(rsi14) else None,
        "support_20d": float(support_20d) if support_20d is not None and not pd.isna(support_20d) else None,
        "support_50d": float(support_50d) if support_50d is not None and not pd.isna(support_50d) else None,
        "support_zone": support_zone,
        "resistance_20d": float(resistance_20d) if resistance_20d is not None and not pd.isna(resistance_20d) else None,
        "distance_to_support_pct": float(distance_to_support_pct) if distance_to_support_pct is not None else None,
        "breakeven_vs_support": float(breakeven_vs_support) if breakeven_vs_support is not None else None,
        "cushion_atr_units": float(cushion_atr_units) if cushion_atr_units is not None else None,
        "realized_vol_20d": float(rv20) if rv20 is not None and not pd.isna(rv20) else None,
        "implied_volatility": implied_volatility,
        "iv_rv_ratio": float(iv_rv_ratio) if iv_rv_ratio is not None and not pd.isna(iv_rv_ratio) else None,
        "trend_state": trend_state,
    }
