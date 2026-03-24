import pandas as pd
import streamlit as st

from config import ScanConfig
from providers.universe_provider import SP500UniverseProvider
from providers.yfinance_market import YFinanceMarketProvider
from recommendation_engine import build_recommendations_for_stock

st.set_page_config(page_title="Passive Put Scanner", layout="wide")
st.title("Passive Put Scanner")

cfg = ScanConfig()

with st.sidebar:
    st.header("Scan Settings")
    max_symbols = st.number_input("Max symbols to scan", min_value=10, max_value=500, value=50, step=10)
    min_dte = st.number_input("Min DTE", min_value=1, max_value=365, value=cfg.min_dte)
    max_dte = st.number_input("Max DTE", min_value=1, max_value=365, value=cfg.max_dte)

run_scan = st.button("Run Scan")

if run_scan:
    universe_provider = SP500UniverseProvider()
    market_provider = YFinanceMarketProvider()

    universe = universe_provider.get_universe()
    symbols = [u.symbol for u in universe[:max_symbols]]

    all_recommendations = []

    progress = st.progress(0)
    status = st.empty()

    for idx, symbol in enumerate(symbols, start=1):
        status.write(f"Scanning {symbol} ({idx}/{len(symbols)})")
        progress.progress(idx / len(symbols))

        try:
            metrics = market_provider.get_stock_metrics(symbol)
            contracts = market_provider.get_option_contracts(symbol)
            recs = build_recommendations_for_stock(metrics, contracts, cfg)
            if recs:
                all_recommendations.append(recs[0])  # best contract per stock
        except Exception as e:
            st.write(f"Skipped {symbol}: {e}")

    if not all_recommendations:
        st.warning("No recommendations found.")
    else:
        all_recommendations.sort(key=lambda r: r.scores.final_score, reverse=True)

        rows = []
        for rec in all_recommendations:
            c = rec.selected_contract
            rows.append({
                "symbol": rec.symbol,
                "stock_price": rec.stock_price,
                "expiration": c.expiration_date,
                "dte": c.dte,
                "strike": c.strike,
                "delta": c.delta_abs,
                "premium": c.premium,
                "breakeven": c.breakeven_price,
                "breakeven_discount_pct": c.breakeven_discount_pct,
                "annualized_yield": c.annualized_secured_yield,
                "entry_limit": rec.suggested_entry_limit,
                "profit_take_debit": rec.profit_take_debit,
                "stock_score": rec.scores.stock_score_total,
                "contract_score": rec.scores.contract_score_total,
                "pres": rec.scores.pres_normalized,
                "final_score": rec.scores.final_score,
                "confidence": rec.confidence_level,
                "reasons": "; ".join(rec.top_reasons),
                "risks": "; ".join(rec.top_risks),
            })

        df = pd.DataFrame(rows)
        st.subheader("Top Ranked Setups")
        st.dataframe(df, use_container_width=True)
