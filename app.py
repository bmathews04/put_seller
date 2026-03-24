import pandas as pd
import streamlit as st

from config import ScanConfig
from providers.universe_provider import SP500UniverseProvider
from providers.yfinance_market import YFinanceMarketProvider
from recommendation_engine import build_recommendations_for_stock


st.set_page_config(page_title="Passive Put Scanner", layout="wide")


# ---------------------------
# Helpers
# ---------------------------
def fmt_pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def fmt_num(value, decimals=2):
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def recommendation_to_row(rec):
    c = rec.selected_contract
    return {
        "symbol": rec.symbol,
        "stock_price": rec.stock_price,
        "expiration": c.expiration_date,
        "dte": c.dte,
        "strike": c.strike,
        "delta_abs": c.delta_abs,
        "premium": c.premium,
        "breakeven": c.breakeven_price,
        "breakeven_discount_pct": c.breakeven_discount_pct,
        "annualized_secured_yield": c.annualized_secured_yield,
        "entry_limit": rec.suggested_entry_limit,
        "entry_low": rec.acceptable_entry_low,
        "entry_high": rec.acceptable_entry_high,
        "profit_take_debit": rec.profit_take_debit,
        "fast_profit_take_debit": rec.fast_profit_take_debit,
        "review_at_dte": rec.review_at_dte,
        "defensive_review_price": rec.defensive_review_price,
        "stock_score": rec.scores.stock_score_total,
        "contract_score": rec.scores.contract_score_total,
        "pres": rec.scores.pres_normalized,
        "final_score": rec.scores.final_score,
        "confidence": rec.confidence_level,
        "reasons": "; ".join(rec.top_reasons),
        "risks": "; ".join(rec.top_risks),
        "warning_flags": "; ".join(rec.warning_flags),
    }


def summarize_exclusions(results_by_symbol):
    stock_exclusions = {}
    contract_exclusions = {}

    for _, payload in results_by_symbol.items():
        stock_reasons = payload.get("stock_exclusion_reasons", [])
        contract_reason_lists = payload.get("contract_exclusion_reasons", [])

        for reason in stock_reasons:
            stock_exclusions[reason] = stock_exclusions.get(reason, 0) + 1

        for reason_list in contract_reason_lists:
            for reason in reason_list:
                contract_exclusions[reason] = contract_exclusions.get(reason, 0) + 1

    stock_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in stock_exclusions.items()]
    ).sort_values("count", ascending=False) if stock_exclusions else pd.DataFrame(columns=["reason", "count"])

    contract_df = pd.DataFrame(
        [{"reason": k, "count": v} for k, v in contract_exclusions.items()]
    ).sort_values("count", ascending=False) if contract_exclusions else pd.DataFrame(columns=["reason", "count"])

    return stock_df, contract_df


# ---------------------------
# Sidebar / settings
# ---------------------------
st.title("Passive Put Scanner")

with st.sidebar:
    st.header("Scan Settings")

    max_symbols = st.number_input(
        "Max symbols to scan",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Controls how many S&P 500 names to scan on this run."
    )

    min_dte = st.number_input("Min DTE", min_value=1, max_value=365, value=25)
    max_dte = st.number_input("Max DTE", min_value=1, max_value=365, value=40)
    target_dte = st.number_input("Target DTE", min_value=1, max_value=365, value=32)

    min_delta = st.number_input("Min abs delta", min_value=0.01, max_value=1.00, value=0.12, step=0.01, format="%.2f")
    max_delta = st.number_input("Max abs delta", min_value=0.01, max_value=1.00, value=0.22, step=0.01, format="%.2f")
    target_delta = st.number_input("Target abs delta", min_value=0.01, max_value=1.00, value=0.17, step=0.01, format="%.2f")

    min_oi = st.number_input("Min open interest", min_value=0, max_value=100000, value=500, step=50)
    min_volume = st.number_input("Min volume", min_value=0, max_value=100000, value=25, step=5)
    max_spread_pct = st.number_input("Max spread %", min_value=0.01, max_value=1.00, value=0.12, step=0.01, format="%.2f")
    min_premium = st.number_input("Min premium", min_value=0.01, max_value=100.0, value=0.35, step=0.05, format="%.2f")

    exclude_earnings = st.checkbox("Exclude earnings before expiry", value=True)
    require_quality_data = st.checkbox("Require quality data", value=False)
    strict_data_mode = st.checkbox("Strict data mode", value=False)

    run_scan = st.button("Run Scan", type="primary")


cfg = ScanConfig(
    max_symbols_to_scan=max_symbols,
    min_dte=int(min_dte),
    max_dte=int(max_dte),
    target_dte=int(target_dte),
    min_abs_delta=float(min_delta),
    max_abs_delta=float(max_delta),
    target_abs_delta=float(target_delta),
    min_open_interest=int(min_oi),
    min_volume=int(min_volume),
    max_spread_pct=float(max_spread_pct),
    min_premium=float(min_premium),
    exclude_earnings_before_expiry=exclude_earnings,
    require_quality_data=require_quality_data,
    strict_data_mode=strict_data_mode,
)


# ---------------------------
# Tabs
# ---------------------------
tab_dashboard, tab_ranked, tab_details, tab_diagnostics = st.tabs(
    ["Dashboard", "Ranked Setups", "Contract Details", "Diagnostics"]
)


if not run_scan:
    with tab_dashboard:
        st.info("Set your parameters in the sidebar and click **Run Scan**.")
    with tab_ranked:
        st.caption("No scan run yet.")
    with tab_details:
        st.caption("Run a scan to inspect contract details.")
    with tab_diagnostics:
        st.caption("Diagnostics will appear after a scan.")
    st.stop()


# ---------------------------
# Data fetch / scan
# ---------------------------
universe_provider = SP500UniverseProvider()
market_provider = YFinanceMarketProvider()

try:
    universe = universe_provider.get_universe()
except Exception as e:
    st.error(f"Failed to load S&P 500 universe: {e}")
    st.stop()

if cfg.max_symbols_to_scan is not None:
    universe = universe[: cfg.max_symbols_to_scan]

symbols = [u.symbol for u in universe]
metadata_by_symbol = {u.symbol: u for u in universe}

all_recommendations = []
results_by_symbol = {}

progress = st.progress(0)
status = st.empty()

for idx, symbol in enumerate(symbols, start=1):
    status.write(f"Scanning {symbol} ({idx}/{len(symbols)})")
    progress.progress(idx / len(symbols))

    results_by_symbol[symbol] = {
        "stock_exclusion_reasons": [],
        "contract_exclusion_reasons": [],
        "contract_count": 0,
        "recommendation_count": 0,
        "exception": None,
    }

    try:
        metrics = market_provider.get_stock_metrics(symbol)
        contracts = market_provider.get_option_contracts(symbol)

        results_by_symbol[symbol]["contract_count"] = len(contracts)

        if not contracts:
            results_by_symbol[symbol]["stock_exclusion_reasons"].append("no_option_chain")
            continue

        recs = build_recommendations_for_stock(metrics, contracts, cfg)

        # collect exclusion reasons from the raw contracts after validation path
        contract_reason_lists = []
        for c in contracts:
            reasons = getattr(c, "contract_exclusion_reasons", [])
            if reasons:
                contract_reason_lists.append(reasons)
        results_by_symbol[symbol]["contract_exclusion_reasons"] = contract_reason_lists

        stock_reasons = getattr(metrics, "stock_exclusion_reasons", [])
        if stock_reasons:
            results_by_symbol[symbol]["stock_exclusion_reasons"] = stock_reasons

        if recs:
            results_by_symbol[symbol]["recommendation_count"] = len(recs)
            # attach metadata for downstream display
            for rec in recs:
                rec._company_name = metadata_by_symbol.get(symbol).company_name if symbol in metadata_by_symbol else None
                rec._sector = metadata_by_symbol.get(symbol).sector if symbol in metadata_by_symbol else None
            all_recommendations.append(recs[0])
        else:
            if not results_by_symbol[symbol]["stock_exclusion_reasons"]:
                results_by_symbol[symbol]["stock_exclusion_reasons"].append("no_valid_contracts")

    except Exception as e:
        results_by_symbol[symbol]["exception"] = str(e)

progress.empty()
status.empty()

all_recommendations.sort(key=lambda r: r.scores.final_score, reverse=True)


# ---------------------------
# Build dataframes
# ---------------------------
ranked_rows = []
for rec in all_recommendations:
    row = recommendation_to_row(rec)
    row["company_name"] = getattr(rec, "_company_name", None)
    row["sector"] = getattr(rec, "_sector", None)
    ranked_rows.append(row)

ranked_df = pd.DataFrame(ranked_rows)

stock_excl_df, contract_excl_df = summarize_exclusions(results_by_symbol)

scan_summary = {
    "symbols_in_universe": len(symbols),
    "symbols_with_recommendations": len(all_recommendations),
    "symbols_without_recommendations": max(len(symbols) - len(all_recommendations), 0),
    "total_contracts_pulled": sum(v["contract_count"] for v in results_by_symbol.values()),
    "symbols_with_provider_errors": sum(1 for v in results_by_symbol.values() if v["exception"]),
}


# ---------------------------
# Dashboard
# ---------------------------
with tab_dashboard:
    st.subheader("Scan Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe scanned", scan_summary["symbols_in_universe"])
    c2.metric("Names with setups", scan_summary["symbols_with_recommendations"])
    c3.metric("Names without setups", scan_summary["symbols_without_recommendations"])
    c4.metric("Contracts pulled", scan_summary["total_contracts_pulled"])
    c5.metric("Provider errors", scan_summary["symbols_with_provider_errors"])

    if ranked_df.empty:
        st.warning("No recommendations found for this scan.")
    else:
        avg_yield = ranked_df["annualized_secured_yield"].dropna().mean() if "annualized_secured_yield" in ranked_df else None
        avg_cushion = ranked_df["breakeven_discount_pct"].dropna().mean() if "breakeven_discount_pct" in ranked_df else None
        avg_final = ranked_df["final_score"].dropna().mean() if "final_score" in ranked_df else None

        c6, c7, c8 = st.columns(3)
        c6.metric("Avg annualized yield", fmt_pct(avg_yield))
        c7.metric("Avg break-even cushion", fmt_pct(avg_cushion))
        c8.metric("Avg final score", fmt_num(avg_final))

        st.markdown("### Top 5 Setups")
        top5 = ranked_df.head(5).copy()

        display_cols = [
            "symbol",
            "company_name",
            "sector",
            "stock_price",
            "expiration",
            "dte",
            "strike",
            "delta_abs",
            "premium",
            "entry_limit",
            "breakeven",
            "breakeven_discount_pct",
            "annualized_secured_yield",
            "final_score",
            "confidence",
        ]
        st.dataframe(top5[display_cols], use_container_width=True)


# ---------------------------
# Ranked setups
# ---------------------------
with tab_ranked:
    st.subheader("Ranked Setups")

    if ranked_df.empty:
        st.warning("No ranked setups available.")
    else:
        sort_col = st.selectbox(
            "Sort ranked setups by",
            options=[
                "final_score",
                "pres",
                "annualized_secured_yield",
                "breakeven_discount_pct",
                "stock_score",
                "contract_score",
            ],
            index=0,
        )
        ascending = st.checkbox("Ascending sort", value=False)

        ranked_sorted = ranked_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        ranked_sorted.insert(0, "rank", range(1, len(ranked_sorted) + 1))

        display_cols = [
            "rank",
            "symbol",
            "company_name",
            "sector",
            "stock_price",
            "expiration",
            "dte",
            "strike",
            "delta_abs",
            "premium",
            "entry_limit",
            "entry_low",
            "entry_high",
            "breakeven",
            "breakeven_discount_pct",
            "annualized_secured_yield",
            "stock_score",
            "contract_score",
            "pres",
            "final_score",
            "confidence",
            "warning_flags",
            "reasons",
            "risks",
        ]

        st.dataframe(ranked_sorted[display_cols], use_container_width=True)


# ---------------------------
# Contract details
# ---------------------------
with tab_details:
    st.subheader("Contract Details")

    if not all_recommendations:
        st.warning("No contract details available.")
    else:
        symbol_options = [rec.symbol for rec in all_recommendations]
        selected_symbol = st.selectbox("Select symbol", symbol_options, index=0)

        selected_rec = next(rec for rec in all_recommendations if rec.symbol == selected_symbol)
        c = selected_rec.selected_contract

        left, mid, right = st.columns(3)
        left.metric("Selected setup", f"{selected_rec.symbol} {c.expiration_date} {fmt_num(c.strike)}P")
        mid.metric("Suggested entry", fmt_num(selected_rec.suggested_entry_limit))
        right.metric("Confidence", selected_rec.confidence_level)

        left2, mid2, right2 = st.columns(3)
        left2.metric("Break-even", fmt_num(c.breakeven_price))
        mid2.metric("Break-even cushion", fmt_pct(c.breakeven_discount_pct))
        right2.metric("Annualized secured yield", fmt_pct(c.annualized_secured_yield))

        left3, mid3, right3 = st.columns(3)
        left3.metric("50% take-profit debit", fmt_num(selected_rec.profit_take_debit))
        mid3.metric("Fast take-profit debit", fmt_num(selected_rec.fast_profit_take_debit))
        right3.metric("Defensive review price", fmt_num(selected_rec.defensive_review_price))

        st.markdown("### Why it ranked well")
        if selected_rec.top_reasons:
            for reason in selected_rec.top_reasons:
                st.write(f"- {reason}")
        else:
            st.write("—")

        st.markdown("### Risks / watchouts")
        if selected_rec.top_risks:
            for risk in selected_rec.top_risks:
                st.write(f"- {risk}")
        else:
            st.write("—")

        st.markdown("### Entry notes")
        st.write(selected_rec.entry_notes or "—")

        st.markdown("### Management plan")
        st.write(selected_rec.management_plan_text or "—")

        st.markdown("### Warning flags")
        if selected_rec.warning_flags:
            st.write(", ".join(selected_rec.warning_flags))
        else:
            st.write("None")

        st.markdown("### Score breakdown")
        score_breakdown = pd.DataFrame(
            [
                {"component": "quality_score", "value": selected_rec.scores.quality_score},
                {"component": "event_stability_score", "value": selected_rec.scores.event_stability_score},
                {"component": "options_market_quality_score", "value": selected_rec.scores.options_market_quality_score},
                {"component": "assignment_comfort_score", "value": selected_rec.scores.assignment_comfort_score},
                {"component": "stock_score_total", "value": selected_rec.scores.stock_score_total},
                {"component": "breakeven_score", "value": selected_rec.scores.breakeven_score},
                {"component": "secured_yield_score", "value": selected_rec.scores.secured_yield_score},
                {"component": "delta_fit_score", "value": selected_rec.scores.delta_fit_score},
                {"component": "liquidity_score", "value": selected_rec.scores.liquidity_score},
                {"component": "dte_fit_score", "value": selected_rec.scores.dte_fit_score},
                {"component": "contract_score_total", "value": selected_rec.scores.contract_score_total},
                {"component": "pres_normalized", "value": selected_rec.scores.pres_normalized},
                {"component": "final_score", "value": selected_rec.scores.final_score},
            ]
        )
        st.dataframe(score_breakdown, use_container_width=True)

        st.caption(
            "This tab currently shows the best contract per symbol from the main scan. "
            "Once we add a full per-symbol chain drill-down, this section can show all qualifying contracts for the selected stock."
        )


# ---------------------------
# Diagnostics
# ---------------------------
with tab_diagnostics:
    st.subheader("Diagnostics")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("### Stock-level exclusions")
        if stock_excl_df.empty:
            st.write("No stock-level exclusions recorded.")
        else:
            st.dataframe(stock_excl_df, use_container_width=True)

    with diag_col2:
        st.markdown("### Contract-level exclusions")
        if contract_excl_df.empty:
            st.write("No contract-level exclusions recorded.")
        else:
            st.dataframe(contract_excl_df, use_container_width=True)

    st.markdown("### Provider errors")
    error_rows = []
    for symbol, payload in results_by_symbol.items():
        if payload.get("exception"):
            error_rows.append({"symbol": symbol, "error": payload["exception"]})

    if error_rows:
        st.dataframe(pd.DataFrame(error_rows), use_container_width=True)
    else:
        st.write("No provider errors.")

    st.markdown("### Per-symbol scan summary")
    per_symbol_rows = []
    for symbol, payload in results_by_symbol.items():
        per_symbol_rows.append(
            {
                "symbol": symbol,
                "contracts_pulled": payload.get("contract_count", 0),
                "recommendation_count": payload.get("recommendation_count", 0),
                "stock_exclusion_reasons": "; ".join(payload.get("stock_exclusion_reasons", [])),
                "provider_error": payload.get("exception"),
            }
        )

    per_symbol_df = pd.DataFrame(per_symbol_rows)
    st.dataframe(per_symbol_df, use_container_width=True)
