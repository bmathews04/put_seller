import math
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def classify_trend(close: float, ma20: float | None, ma50: float | None) -> str:
    if close is None or ma20 is None or ma50 is None:
        return "Unknown"

    if close > ma20 and close > ma50 and ma20 > ma50:
        return "Bullish"
    if close < ma20 and close < ma50 and ma20 < ma50:
        return "Weak"
    return "Neutral"


def build_technical_context(
    hist: pd.DataFrame,
    stock_price: float | None,
    breakeven_price: float | None,
) -> dict:
    if hist is None or hist.empty or stock_price is None:
        return {
            "ma20": None,
            "ma50": None,
            "atr14": None,
            "rsi14": None,
            "support_20d": None,
            "resistance_20d": None,
            "distance_to_support_pct": None,
            "breakeven_vs_support": None,
            "cushion_atr_units": None,
            "trend_state": "Unknown",
        }

    close = hist["Close"]
    low = hist["Low"]
    high = hist["High"]

    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    atr14 = compute_atr(hist, 14).iloc[-1] if len(hist) >= 14 else None
    rsi14 = compute_rsi(close, 14).iloc[-1] if len(close) >= 14 else None

    support_20d = low.tail(20).min() if len(low) >= 20 else low.min()
    resistance_20d = high.tail(20).max() if len(high) >= 20 else high.max()

    distance_to_support_pct = None
    if support_20d is not None and stock_price and stock_price > 0:
        distance_to_support_pct = (stock_price - support_20d) / stock_price

    breakeven_vs_support = None
    if breakeven_price is not None and support_20d is not None:
        breakeven_vs_support = breakeven_price - support_20d

    cushion_atr_units = None
    if breakeven_price is not None and atr14 is not None and atr14 > 0:
        cushion_atr_units = (stock_price - breakeven_price) / atr14

    trend_state = classify_trend(stock_price, ma20, ma50)

    return {
        "ma20": float(ma20) if ma20 is not None and not pd.isna(ma20) else None,
        "ma50": float(ma50) if ma50 is not None and not pd.isna(ma50) else None,
        "atr14": float(atr14) if atr14 is not None and not pd.isna(atr14) else None,
        "rsi14": float(rsi14) if rsi14 is not None and not pd.isna(rsi14) else None,
        "support_20d": float(support_20d) if support_20d is not None and not pd.isna(support_20d) else None,
        "resistance_20d": float(resistance_20d) if resistance_20d is not None and not pd.isna(resistance_20d) else None,
        "distance_to_support_pct": float(distance_to_support_pct) if distance_to_support_pct is not None else None,
        "breakeven_vs_support": float(breakeven_vs_support) if breakeven_vs_support is not None else None,
        "cushion_atr_units": float(cushion_atr_units) if cushion_atr_units is not None else None,
        "trend_state": trend_state,
    }
