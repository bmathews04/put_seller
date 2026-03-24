import json
import ast
import pandas as pd
import streamlit as st

from config import ScanConfig
from tickers import TICKERS, TICKER_METADATA
from providers.yfinance_market import YFinanceMarketProvider
from recommendation_engine import build_recommendations_for_stock


st.set_page_config(page_title="Passive Put Scanner", layout="wide")

# ---------------------------
# Session state initialization
# ---------------------------
if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False

if "all_recommendations" not in st.session_state:
    st.session_state.all_recommendations = []

if "results_by_symbol" not in st.session_state:
    st.session_state.results_by_symbol = {}

if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = pd.DataFrame()

if "stock_excl_df" not in st.session_state:
    st.session_state.stock_excl_df = pd.DataFrame()

if "contract_excl_df" not in st.session_state:
    st.session_state.contract_excl_df = pd.DataFrame()

if "scan_summary" not in st.session_state:
    st.session_state.scan_summary = {}

if "last_scan_cfg" not in st.session_state:
    st.session_state.last_scan_cfg = None

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

    stock_df = (
        pd.DataFrame([{"reason": k, "count": v} for k, v in stock_exclusions.items()])
        .sort_values("count", ascending=False)
        if stock_exclusions
        else pd.DataFrame(columns=["reason", "count"])
    )

    contract_df = (
        pd.DataFrame([{"reason": k, "count": v} for k, v in contract_exclusions.items()])
        .sort_values("count", ascending=False)
        if contract_exclusions
        else pd.DataFrame(columns=["reason", "count"])
    )

    return stock_df, contract_df


def try_parse_structured(text: str):
    """
    Try JSON first, then Python literal parsing.
    Returns: (parsed_obj, parser_used, error_message)
    """
    if not text.strip():
        return None, None, "No content pasted."

    try:
        return json.loads(text), "json", None
    except Exception:
        pass

    try:
        return ast.literal_eval(text), "python_literal", None
    except Exception as e:
        return None, None, str(e)


def validate_contract_like(obj):
    required = [
        "symbol",
        "expiration_date",
        "dte",
        "strike",
        "option_type",
        "bid",
        "ask",
    ]
    optional_checks = [
        "delta",
        "open_interest",
        "volume",
        "implied_volatility",
        "in_the_money",
    ]

    findings = []
    missing = [k for k in required if k not in obj]
    if missing:
        findings.append(("Missing required fields", ", ".join(missing)))
    else:
        findings.append(("Required fields", "All required contract fields present"))

    if "bid" in obj and "ask" in obj:
        try:
            bid = float(obj["bid"]) if obj["bid"] is not None else None
            ask = float(obj["ask"]) if obj["ask"] is not None else None
            if bid is None or ask is None:
                findings.append(("Bid/ask check", "Bid or ask is missing"))
            elif ask < bid:
                findings.append(("Bid/ask check", f"Invalid spread: ask ({ask}) < bid ({bid})"))
            else:
                findings.append(("Bid/ask check", f"Valid spread: {ask - bid:.4f}"))
        except Exception:
            findings.append(("Bid/ask check", "Could not parse bid/ask as numeric"))

    if "delta" in obj:
        try:
            delta = obj["delta"]
            if delta is None:
                findings.append(("Delta", "Delta is missing"))
            else:
                findings.append(("Delta", f"Delta present: {delta}"))
        except Exception:
            findings.append(("Delta", "Could not parse delta"))

    for key in optional_checks:
        if key in obj:
            findings.append((key, f"Present: {obj[key]}"))
        else:
            findings.append((key, "Not present"))

    return pd.DataFrame(findings, columns=["check", "result"])


def validate_recommendation_like(obj):
    required = [
        "symbol",
        "stock_price",
        "selected_contract",
        "scores",
    ]

    findings = []
    missing = [k for k in required if k not in obj]
    if missing:
        findings.append(("Missing required fields", ", ".join(missing)))
    else:
        findings.append(("Required fields", "All required recommendation fields present"))

    if "selected_contract" in obj:
        if isinstance(obj["selected_contract"], dict):
            findings.append(("selected_contract", "Looks like nested dict payload"))
        else:
            findings.append(("selected_contract", f"Present, type={type(obj['selected_contract']).__name__}"))

    if "scores" in obj:
        if isinstance(obj["scores"], dict):
            findings.append(("scores", "Looks like nested dict payload"))
        else:
            findings.append(("scores", f"Present, type={type(obj['scores']).__name__}"))

    for key in [
        "suggested_entry_limit",
        "acceptable_entry_low",
        "acceptable_entry_high",
        "profit_take_debit",
        "confidence_level",
    ]:
        if key in obj:
            findings.append((key, f"Present: {obj[key]}"))
        else:
            findings.append((key, "Not present"))

    return pd.DataFrame(findings, columns=["check", "result"])


# ---------------------------
# Sidebar / settings
# ---------------------------
st.title("Passive Put Scanner")

with st.sidebar:
    st.header("Scan Settings")

    max_symbols = st.number_input(
        "Max symbols to scan",
        min_value=10,
        max_value=len(TICKERS),
        value=min(50, len(TICKERS)),
        step=10,
        help="Controls how many hardcoded names to scan on this run.",
    )

    min_dte = st.number_input("Min DTE", min_value=1, max_value=365, value=25)
    max_dte = st.number_input("Max DTE", min_value=1, max_value=365, value=40)
    target_dte = st.number_input("Target DTE", min_value=1, max_value=365, value=32)

    min_delta = st.number_input(
        "Min abs delta",
        min_value=0.01,
        max_value=1.00,
        value=0.12,
        step=0.01,
        format="%.2f",
    )
    max_delta = st.number_input(
        "Max abs delta",
        min_value=0.01,
        max_value=1.00,
        value=0.22,
        step=0.01,
        format="%.2f",
    )
    target_delta = st.number_input(
        "Target abs delta",
        min_value=0.01,
        max_value=1.00,
        value=0.17,
        step=0.01,
        format="%.2f",
    )

    min_oi = st.number_input(
        "Min open interest",
        min_value=0,
        max_value=100000,
        value=500,
        step=50,
    )
    min_volume = st.number_input(
        "Min volume",
        min_value=0,
        max_value=100000,
        value=25,
        step=5,
    )
    max_spread_pct = st.number_input(
        "Max spread %",
        min_value=0.01,
        max_value=1.00,
        value=0.12,
        step=0.01,
        format="%.2f",
    )
    min_premium = st.number_input(
        "Min premium",
        min_value=0.01,
        max_value=100.0,
        value=0.35,
        step=0.05,
        format="%.2f",
    )

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
tab_dashboard, tab_ranked, tab_details, tab_diagnostics, tab_debug = st.tabs(
    ["Dashboard", "Ranked Setups", "Contract Details", "Diagnostics", "Debug / Validation"]
)


if not run_scan and not st.session_state.scan_completed:
    with tab_dashboard:
        st.info("Set your parameters in the sidebar and click **Run Scan**.")
    with tab_ranked:
        st.caption("No scan run yet.")
    with tab_details:
        st.caption("Run a scan to inspect contract details.")
    with tab_diagnostics:
        st.caption("Diagnostics will appear after a scan.")
    with tab_debug:
        st.caption("Paste logs, JSON, recommendation output, or raw contract data here for validation.")
    st.stop()

market_provider = YFinanceMarketProvider()

symbols = TICKERS[: cfg.max_symbols_to_scan] if cfg.max_symbols_to_scan else TICKERS
metadata_by_symbol = {
    symbol: TICKER_METADATA.get(symbol, {"company_name": None, "sector": None})
    for symbol in symbols
}

# ---------------------------
# Data fetch / scan
# ---------------------------
if run_scan:
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
            contracts = market_provider.get_option_contracts(symbol, cfg)

            results_by_symbol[symbol]["contract_count"] = len(contracts)

            if not contracts:
                results_by_symbol[symbol]["stock_exclusion_reasons"].append("no_option_chain")
                continue

            recs = build_recommendations_for_stock(metrics, contracts, cfg)

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
                for rec in recs:
                    rec._company_name = metadata_by_symbol.get(symbol, {}).get("company_name")
                    rec._sector = metadata_by_symbol.get(symbol, {}).get("sector")
                all_recommendations.append(recs[0])
            else:
                if not results_by_symbol[symbol]["stock_exclusion_reasons"]:
                    results_by_symbol[symbol]["stock_exclusion_reasons"].append("no_valid_contracts")

        except Exception as e:
            results_by_symbol[symbol]["exception"] = str(e)

    progress.empty()
    status.empty()

    all_recommendations.sort(key=lambda r: r.scores.final_score, reverse=True)

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

    st.session_state.all_recommendations = all_recommendations
    st.session_state.results_by_symbol = results_by_symbol
    st.session_state.ranked_df = ranked_df
    st.session_state.stock_excl_df = stock_excl_df
    st.session_state.contract_excl_df = contract_excl_df
    st.session_state.scan_summary = scan_summary
    st.session_state.scan_completed = True
    st.session_state.last_scan_cfg = cfg

else:
    all_recommendations = st.session_state.all_recommendations
    results_by_symbol = st.session_state.results_by_symbol
    ranked_df = st.session_state.ranked_df
    stock_excl_df = st.session_state.stock_excl_df
    contract_excl_df = st.session_state.contract_excl_df
    scan_summary = st.session_state.scan_summary


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
        avg_yield = (
            ranked_df["annualized_secured_yield"].dropna().mean()
            if "annualized_secured_yield" in ranked_df
            else None
        )
        avg_cushion = (
            ranked_df["breakeven_discount_pct"].dropna().mean()
            if "breakeven_discount_pct" in ranked_df
            else None
        )
        avg_final = (
            ranked_df["final_score"].dropna().mean()
            if "final_score" in ranked_df
            else None
        )

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


# ---------------------------
# Debug / Validation
# ---------------------------
with tab_debug:
    st.subheader("Debug / Validation")

    st.markdown(
        """
Use this tab to paste:
- traceback / error logs
- raw JSON or dict output
- recommendation rows
- contract payloads
- copied DataFrame rows

This is intended to make validation faster when something looks off.
"""
    )

    debug_mode = st.radio(
        "What are you pasting?",
        options=[
            "Raw text / traceback",
            "JSON / dict payload",
            "Recommendation row",
            "Contract payload",
            "DataFrame rows / CSV-like text",
        ],
        horizontal=False,
    )

    pasted_text = st.text_area(
        "Paste content here",
        height=250,
        placeholder="Paste logs, JSON, dicts, recommendation output, or rows here...",
    )

    validate_now = st.button("Validate pasted content")

    if validate_now:
        if not pasted_text.strip():
            st.warning("Paste something first.")
        else:
            if debug_mode == "Raw text / traceback":
                st.markdown("### Raw text preview")
                st.code(pasted_text)

                st.markdown("### Quick checks")
                quick_flags = []
                lower_text = pasted_text.lower()

                if "traceback" in lower_text:
                    quick_flags.append("Contains a Python traceback")
                if "keyerror" in lower_text:
                    quick_flags.append("Contains KeyError")
                if "modulenotfounderror" in lower_text:
                    quick_flags.append("Contains ModuleNotFoundError")
                if "403" in lower_text or "forbidden" in lower_text:
                    quick_flags.append("Contains HTTP 403 / Forbidden")
                if "yfinance" in lower_text:
                    quick_flags.append("Mentions yfinance")
                if "delta" in lower_text:
                    quick_flags.append("Mentions delta")

                if quick_flags:
                    for flag in quick_flags:
                        st.write(f"- {flag}")
                else:
                    st.write("No obvious patterns found.")

            elif debug_mode == "JSON / dict payload":
                parsed, parser_used, parse_error = try_parse_structured(pasted_text)

                if parsed is None:
                    st.error(f"Could not parse structured content: {parse_error}")
                    st.code(pasted_text)
                else:
                    st.success(f"Parsed successfully using: {parser_used}")
                    st.write("Parsed object type:", type(parsed).__name__)
                    st.json(parsed)

                    if isinstance(parsed, dict):
                        st.markdown("### Top-level keys")
                        st.write(list(parsed.keys()))

            elif debug_mode == "Recommendation row":
                parsed, parser_used, parse_error = try_parse_structured(pasted_text)

                if parsed is None or not isinstance(parsed, dict):
                    st.error("Recommendation row should parse into a dictionary-like object.")
                    if parse_error:
                        st.caption(parse_error)
                else:
                    st.success(f"Parsed recommendation row using: {parser_used}")
                    st.json(parsed)
                    st.markdown("### Validation results")
                    st.dataframe(validate_recommendation_like(parsed), use_container_width=True)

            elif debug_mode == "Contract payload":
                parsed, parser_used, parse_error = try_parse_structured(pasted_text)

                if parsed is None or not isinstance(parsed, dict):
                    st.error("Contract payload should parse into a dictionary-like object.")
                    if parse_error:
                        st.caption(parse_error)
                else:
                    st.success(f"Parsed contract payload using: {parser_used}")
                    st.json(parsed)
                    st.markdown("### Validation results")
                    st.dataframe(validate_contract_like(parsed), use_container_width=True)

            elif debug_mode == "DataFrame rows / CSV-like text":
                st.markdown("### Raw pasted rows")
                st.code(pasted_text)

                lines = [line for line in pasted_text.splitlines() if line.strip()]
                st.write(f"Detected {len(lines)} non-empty lines.")

                if len(lines) >= 2:
                    st.info(
                        "This looks like tabular text. If you want, the next step can be adding a parser for tab-separated or CSV rows."
                    )
                else:
                    st.warning("Not enough lines detected to infer a table.")
