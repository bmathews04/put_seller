import json
import ast
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import ScanConfig
from tickers import TICKERS, TICKER_METADATA
from providers.yfinance_market import YFinanceMarketProvider
from recommendation_engine import build_recommendations_for_stock
from technicals import (
    build_technical_context,
    score_technical_context,
    grade_trade_separation,
)
from decisioning import confidence_rank, technical_rank, derive_decision_status

st.set_page_config(page_title="Passive Put Scanner", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.0rem;
    max-width: 1500px;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTabs"]) {
    gap: 0.35rem;
}

h1, h2, h3 {
    margin-bottom: 0.35rem !important;
}
h1 {
    margin-top: 0.1rem !important;
}
h2, h3 {
    margin-top: 0.5rem !important;
}

div[data-testid="metric-container"] {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
section[data-testid="stSidebar"] hr {
    margin-top: 0.6rem;
    margin-bottom: 0.6rem;
}

div[data-testid="stExpander"] {
    margin-top: 0.35rem;
    margin-bottom: 0.35rem;
}

div[data-testid="stDataFrame"] {
    margin-top: 0.25rem;
    margin-bottom: 0.35rem;
}

.decision-badge {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.decision-ready {
    background: rgba(34, 197, 94, 0.14);
    color: rgb(21, 128, 61);
}
.decision-review {
    background: rgba(234, 179, 8, 0.16);
    color: rgb(161, 98, 7);
}
.decision-pass {
    background: rgba(239, 68, 68, 0.14);
    color: rgb(185, 28, 28);
}

.eligibility-badge {
    display: inline-block;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.eligibility-clean {
    background: rgba(34, 197, 94, 0.12);
    color: rgb(21, 128, 61);
}
.eligibility-warning {
    background: rgba(234, 179, 8, 0.16);
    color: rgb(161, 98, 7);
}
.eligibility-fail {
    background: rgba(239, 68, 68, 0.14);
    color: rgb(185, 28, 28);
}

.idea-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 0.75rem 0.85rem 0.55rem 0.85rem;
    margin-bottom: 0.65rem;
    background: rgba(255,255,255,0.02);
}
.idea-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.idea-sub {
    font-size: 0.9rem;
    opacity: 0.85;
    margin-bottom: 0.45rem;
}
.idea-watchout {
    font-size: 0.84rem;
    opacity: 0.9;
    margin-top: 0.25rem;
}

.caption-tight {
    font-size: 0.84rem;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

if "scan_completed" not in st.session_state:
    st.session_state.scan_completed = False
if "all_recommendations" not in st.session_state:
    st.session_state.all_recommendations = []
if "results_by_symbol" not in st.session_state:
    st.session_state.results_by_symbol = {}
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = pd.DataFrame()
if "previous_ranked_df" not in st.session_state:
    st.session_state.previous_ranked_df = pd.DataFrame()
if "stock_excl_df" not in st.session_state:
    st.session_state.stock_excl_df = pd.DataFrame()
if "contract_excl_df" not in st.session_state:
    st.session_state.contract_excl_df = pd.DataFrame()
if "stock_warn_df" not in st.session_state:
    st.session_state.stock_warn_df = pd.DataFrame()
if "contract_warn_df" not in st.session_state:
    st.session_state.contract_warn_df = pd.DataFrame()
if "scan_summary" not in st.session_state:
    st.session_state.scan_summary = {}
if "last_scan_cfg" not in st.session_state:
    st.session_state.last_scan_cfg = None
if "selected_detail_symbol" not in st.session_state:
    st.session_state.selected_detail_symbol = None


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def fmt_num(value, decimals=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{decimals}f}"


def decision_emoji(value):
    return {
        "Ready": "🟢",
        "Review": "🟡",
        "Pass": "🔴",
    }.get(value, "⚪")


def decision_badge_html(status: str) -> str:
    css_class = {
        "Ready": "decision-ready",
        "Review": "decision-review",
        "Pass": "decision-pass",
    }.get(status, "decision-review")
    return f'<span class="decision-badge {css_class}">{decision_emoji(status)} {status}</span>'


def eligibility_badge_html(status: str) -> str:
    label_map = {
        "eligible": "Eligible",
        "eligible_with_warnings": "Eligible with warnings",
        "ineligible": "Ineligible",
    }
    css_class = {
        "eligible": "eligibility-clean",
        "eligible_with_warnings": "eligibility-warning",
        "ineligible": "eligibility-fail",
    }.get(status, "eligibility-warning")
    label = label_map.get(status, status or "Unknown")
    return f'<span class="eligibility-badge {css_class}">{label}</span>'


def joined(items):
    return "; ".join(items) if items else ""


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
        "trend_state": getattr(rec, "_trend_state", None),
        "distance_to_support_pct": getattr(rec, "_distance_to_support_pct", None),
        "cushion_atr_units": getattr(rec, "_cushion_atr_units", None),
        "technical_score": getattr(rec, "_technical_score", None),
        "technical_label": getattr(rec, "_technical_label", None),
        "stock_eligibility_status": rec.stock_eligibility_status,
        "stock_warning_reasons": joined(rec.stock_warning_reasons),
        "contract_eligibility_status": rec.contract_eligibility_status,
        "contract_hard_fail_reasons": joined(getattr(c, "contract_hard_fail_reasons", [])),
        "contract_warning_reasons": joined(rec.contract_warning_reasons),
        "decision_status": rec.decision_status,
        "decision_rationale": rec.decision_rationale,
        "decision_blockers": joined(rec.decision_blockers),
        "decision_cautions": joined(rec.decision_cautions),
        "warning_severity_label": rec.warning_severity_label,
        "warning_severity_points": rec.warning_severity_points,
        "warning_count_total": rec.warning_count_total,
        "high_severity_warning_count": rec.high_severity_warning_count,
        "medium_severity_warning_count": rec.medium_severity_warning_count,
        "low_severity_warning_count": rec.low_severity_warning_count,
    }


def summarize_reason_lists(rows, key_name="reason"):
    counts = {}
    for row in rows:
        for reason in row:
            counts[reason] = counts.get(reason, 0) + 1

    if not counts:
        return pd.DataFrame(columns=[key_name, "count"])

    return (
        pd.DataFrame([{key_name: k, "count": v} for k, v in counts.items()])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def summarize_eligibility(results_by_symbol):
    stock_hard_fail_lists = []
    stock_warning_lists = []
    contract_hard_fail_lists = []
    contract_warning_lists = []

    for _, payload in results_by_symbol.items():
        stock_hard_fail_lists.append(payload.get("stock_hard_fail_reasons", []))
        stock_warning_lists.append(payload.get("stock_warning_reasons", []))

        for reason_list in payload.get("contract_hard_fail_reasons", []):
            contract_hard_fail_lists.append(reason_list)

        for reason_list in payload.get("contract_warning_reasons", []):
            contract_warning_lists.append(reason_list)

    stock_excl_df = summarize_reason_lists(stock_hard_fail_lists, "reason")
    stock_warn_df = summarize_reason_lists(stock_warning_lists, "reason")
    contract_excl_df = summarize_reason_lists(contract_hard_fail_lists, "reason")
    contract_warn_df = summarize_reason_lists(contract_warning_lists, "reason")

    return stock_excl_df, contract_excl_df, stock_warn_df, contract_warn_df


def try_parse_structured(text: str):
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
        "contract_eligibility_status",
        "contract_hard_fail_reasons",
        "contract_warning_reasons",
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

    for key in [
        "suggested_entry_limit",
        "acceptable_entry_low",
        "acceptable_entry_high",
        "profit_take_debit",
        "confidence_level",
        "decision_status",
        "decision_rationale",
        "warning_severity_label",
        "warning_severity_points",
    ]:
        if key in obj:
            findings.append((key, f"Present: {obj[key]}"))
        else:
            findings.append((key, "Not present"))

    return pd.DataFrame(findings, columns=["check", "result"])


def describe_trade_setup(rec):
    contract = rec.selected_contract
    reasons = rec.top_reasons[:3] if rec.top_reasons else []

    lines = []
    lines.append(
        f"Sell to open the {rec.symbol} {contract.expiration_date} {fmt_num(contract.strike)} put around {fmt_num(rec.suggested_entry_limit)}."
    )

    if reasons:
        joined_reasons = ", ".join(r.lower() for r in reasons)
        lines.append(f"This stands out because of {joined_reasons}.")

    lines.append(
        f"Your break-even is {fmt_num(contract.breakeven_price)}, and the standard profit-taking level is {fmt_num(rec.profit_take_debit)}."
    )

    return lines


def render_price_chart(hist: pd.DataFrame, tech: dict, breakeven_price: float | None):
    if hist is None or hist.empty:
        st.write("Chart unavailable.")
        return

    chart_df = hist.tail(90).copy()
    chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
    chart_df["MA50"] = chart_df["Close"].rolling(50).mean()

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(chart_df.index, chart_df["Close"], label="Close")
    ax.plot(chart_df.index, chart_df["MA20"], label="20D MA")
    ax.plot(chart_df.index, chart_df["MA50"], label="50D MA")

    support_zone = tech.get("support_zone")
    resistance_20d = tech.get("resistance_20d")

    if support_zone is not None:
        ax.axhline(support_zone, linestyle="--", linewidth=1, label="Support")
    if resistance_20d is not None:
        ax.axhline(resistance_20d, linestyle="--", linewidth=1, label="Resistance")
    if breakeven_price is not None:
        ax.axhline(breakeven_price, linestyle=":", linewidth=1.5, label="Break-even")

    ax.set_title("Price chart with break-even and trend overlays")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)


def filter_ranked_df(
    df: pd.DataFrame,
    min_confidence: str,
    min_yield: float,
    min_cushion: float,
    technical_filter: str,
    max_dte_filter: int,
    clean_only: bool,
):
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if min_confidence != "Any":
        min_conf_rank = confidence_rank(min_confidence)
        filtered = filtered[
            filtered["confidence"].apply(lambda x: confidence_rank(x) >= min_conf_rank)
        ]

    if "annualized_secured_yield" in filtered.columns:
        filtered = filtered[
            filtered["annualized_secured_yield"].fillna(-999) >= min_yield
        ]

    if "breakeven_discount_pct" in filtered.columns:
        filtered = filtered[
            filtered["breakeven_discount_pct"].fillna(-999) >= min_cushion
        ]

    if technical_filter != "Any":
        min_tech_rank = technical_rank(technical_filter)
        filtered = filtered[
            filtered["technical_label"].apply(lambda x: technical_rank(x) >= min_tech_rank)
        ]

    if "dte" in filtered.columns:
        filtered = filtered[filtered["dte"].fillna(9999) <= max_dte_filter]

    if clean_only:
        filtered = filtered[
            (filtered["warning_count_total"].fillna(0) == 0)
            & (filtered["contract_eligibility_status"] == "eligible")
            & (filtered["stock_eligibility_status"] == "eligible")
            & (filtered["decision_status"] == "Ready")
        ]

    return filtered


def build_scan_changes(current_df: pd.DataFrame, previous_df: pd.DataFrame):
    if current_df.empty and previous_df.empty:
        return [], [], pd.DataFrame(columns=["symbol", "previous_score", "current_score", "score_change"])

    current = current_df.copy() if not current_df.empty else pd.DataFrame(columns=["symbol", "final_score"])
    previous = previous_df.copy() if not previous_df.empty else pd.DataFrame(columns=["symbol", "final_score"])

    current_symbols = set(current["symbol"].tolist()) if "symbol" in current.columns else set()
    previous_symbols = set(previous["symbol"].tolist()) if "symbol" in previous.columns else set()

    new_symbols = sorted(current_symbols - previous_symbols)
    dropped_symbols = sorted(previous_symbols - current_symbols)

    movers_df = pd.DataFrame(columns=["symbol", "previous_score", "current_score", "score_change"])
    if "symbol" in current.columns and "symbol" in previous.columns:
        merged = previous[["symbol", "final_score"]].rename(columns={"final_score": "previous_score"}).merge(
            current[["symbol", "final_score"]].rename(columns={"final_score": "current_score"}),
            on="symbol",
            how="inner",
        )
        if not merged.empty:
            merged["score_change"] = merged["current_score"] - merged["previous_score"]
            movers_df = merged.reindex(
                merged["score_change"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)

    return new_symbols, dropped_symbols, movers_df


def render_action_idea_row(row, idx):
    status = row.get("decision_status", "Review")
    eligibility_status = row.get("contract_eligibility_status", "eligible_with_warnings")

    st.markdown(
        f"""
        <div class="idea-card">
            <div class="idea-title">#{idx} {row.get('symbol')} — {decision_badge_html(status)}</div>
            <div class="idea-sub">
                {fmt_num(row.get('strike'))}P / {row.get('expiration')} / {row.get('dte')} DTE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(eligibility_badge_html(eligibility_status), unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Entry", fmt_num(row.get("entry_limit")))
    c2.metric("Yield", fmt_pct(row.get("annualized_secured_yield")))
    c3.metric("Cushion", fmt_pct(row.get("breakeven_discount_pct")))
    c4.metric("Score", fmt_num(row.get("final_score")))
    c5.metric("Confidence", row.get("confidence"))
    c6.metric("Warning severity", row.get("warning_severity_label"))

    st.markdown(
        f'<div class="idea-watchout"><strong>Technical:</strong> {row.get("technical_label", "Unknown")} | '
        f'<strong>Watchout:</strong> {row.get("risks", "—")}</div>',
        unsafe_allow_html=True,
    )

    if row.get("decision_rationale"):
        st.caption(f"Decision rationale: {row.get('decision_rationale')}")
    if row.get("contract_warning_reasons"):
        st.caption(f"Contract warnings: {row.get('contract_warning_reasons')}")
    if row.get("stock_warning_reasons"):
        st.caption(f"Stock warnings: {row.get('stock_warning_reasons')}")


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

    run_scan = st.button("Run Scan", type="primary")

    if st.button("Clear cached results"):
        st.session_state.scan_completed = False
        st.session_state.all_recommendations = []
        st.session_state.results_by_symbol = {}
        st.session_state.ranked_df = pd.DataFrame()
        st.session_state.previous_ranked_df = pd.DataFrame()
        st.session_state.stock_excl_df = pd.DataFrame()
        st.session_state.contract_excl_df = pd.DataFrame()
        st.session_state.stock_warn_df = pd.DataFrame()
        st.session_state.contract_warn_df = pd.DataFrame()
        st.session_state.scan_summary = {}
        st.session_state.last_scan_cfg = None
        st.session_state.selected_detail_symbol = None
        st.rerun()

    st.markdown("---")

    with st.expander("Advanced scan rules", expanded=False):
        target_dte = st.number_input("Target DTE", min_value=1, max_value=365, value=32)
        target_delta = st.number_input(
            "Target abs delta",
            min_value=0.01,
            max_value=1.00,
            value=0.17,
            step=0.01,
            format="%.2f",
        )
        min_oi = st.number_input("Min open interest", min_value=0, max_value=100000, value=500, step=50)
        min_volume = st.number_input("Min volume", min_value=0, max_value=100000, value=25, step=5)
        require_quality_data = st.checkbox("Require quality data", value=False)
        strict_data_mode = st.checkbox("Strict data mode", value=False)

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

if run_scan:
    all_recommendations = []
    results_by_symbol = {}

    progress = st.progress(0)
    status = st.empty()

    for idx, symbol in enumerate(symbols, start=1):
        status.write(f"Scanning {symbol} ({idx}/{len(symbols)})")
        progress.progress(idx / len(symbols))

        results_by_symbol[symbol] = {
            "stock_eligibility_status": None,
            "stock_hard_fail_reasons": [],
            "stock_warning_reasons": [],
            "contract_hard_fail_reasons": [],
            "contract_warning_reasons": [],
            "contract_count": 0,
            "recommendation_count": 0,
            "recommendations": [],
            "technical_context": {},
            "technical_summary": {},
            "exception": None,
            "provider_detail_errors": [],
        }

        try:
            metrics = market_provider.get_stock_metrics(symbol)
            contracts = market_provider.get_option_contracts(symbol, cfg, stock_metrics=metrics)
            results_by_symbol[symbol]["provider_detail_errors"] = list(getattr(market_provider, "last_errors", []))
            results_by_symbol[symbol]["contract_count"] = len(contracts)

            if not contracts:
                results_by_symbol[symbol]["stock_eligibility_status"] = "ineligible"
                results_by_symbol[symbol]["stock_hard_fail_reasons"].append("no_option_chain")
                continue

            recs = build_recommendations_for_stock(metrics, contracts, cfg)

            results_by_symbol[symbol]["stock_eligibility_status"] = getattr(metrics, "stock_eligibility_status", None)
            results_by_symbol[symbol]["stock_hard_fail_reasons"] = list(getattr(metrics, "stock_hard_fail_reasons", []))
            results_by_symbol[symbol]["stock_warning_reasons"] = list(getattr(metrics, "stock_warning_reasons", []))

            contract_hard_fail_lists = []
            contract_warning_lists = []

            for c in contracts:
                hard_reasons = list(getattr(c, "contract_hard_fail_reasons", []))
                warning_reasons = list(getattr(c, "contract_warning_reasons", []))
                if hard_reasons:
                    contract_hard_fail_lists.append(hard_reasons)
                if warning_reasons:
                    contract_warning_lists.append(warning_reasons)

            results_by_symbol[symbol]["contract_hard_fail_reasons"] = contract_hard_fail_lists
            results_by_symbol[symbol]["contract_warning_reasons"] = contract_warning_lists

            if recs:
                hist = market_provider.get_price_history(symbol)
                top_contract = recs[0].selected_contract

                tech = build_technical_context(
                    hist=hist,
                    stock_price=recs[0].stock_price,
                    breakeven_price=top_contract.breakeven_price,
                    implied_volatility=top_contract.implied_volatility,
                )
                tech_summary = score_technical_context(tech)

                results_by_symbol[symbol]["recommendation_count"] = len(recs)
                results_by_symbol[symbol]["recommendations"] = recs
                results_by_symbol[symbol]["technical_context"] = tech
                results_by_symbol[symbol]["technical_summary"] = tech_summary

                for rec in recs:
                    rec._company_name = metadata_by_symbol.get(symbol, {}).get("company_name")
                    rec._sector = metadata_by_symbol.get(symbol, {}).get("sector")
                    rec._trend_state = tech.get("trend_state")
                    rec._distance_to_support_pct = tech.get("distance_to_support_pct")
                    rec._cushion_atr_units = tech.get("cushion_atr_units")
                    rec._technical_score = tech_summary.get("technical_score")
                    rec._technical_label = tech_summary.get("technical_label")

                    decision_status, decision_meta = derive_decision_status(
                        final_score=rec.scores.final_score,
                        confidence=rec.confidence_level,
                        technical_label=rec._technical_label,
                        contract_eligibility_status=rec.contract_eligibility_status,
                        stock_eligibility_status=rec.stock_eligibility_status,
                        stock_warning_reasons=rec.stock_warning_reasons,
                        contract_warning_reasons=rec.contract_warning_reasons,
                        warning_flags=rec.warning_flags,
                    )
                    rec.decision_status = decision_status
                    rec.decision_rationale = decision_meta.get("decision_rationale")
                    rec.decision_blockers = list(decision_meta.get("decision_blockers", []))
                    rec.decision_cautions = list(decision_meta.get("decision_cautions", []))
                    rec.warning_severity_label = decision_meta.get("severity_label", "None")
                    rec.warning_severity_points = int(decision_meta.get("severity_points", 0))
                    rec.warning_count_total = int(decision_meta.get("warning_count_total", 0))
                    rec.high_severity_warning_count = int(decision_meta.get("high_severity_count", 0))
                    rec.medium_severity_warning_count = int(decision_meta.get("medium_severity_count", 0))
                    rec.low_severity_warning_count = int(decision_meta.get("low_severity_count", 0))

                all_recommendations.append(recs[0])
            else:
                if not results_by_symbol[symbol]["stock_hard_fail_reasons"]:
                    results_by_symbol[symbol]["stock_eligibility_status"] = "ineligible"
                    results_by_symbol[symbol]["stock_hard_fail_reasons"].append("no_valid_contracts")

        except Exception as e:
            results_by_symbol[symbol]["exception"] = str(e)

    progress.empty()
    status.empty()

    all_recommendations.sort(
        key=lambda r: (
            {"Ready": 0, "Review": 1, "Pass": 2}.get(r.decision_status, 9),
            -(r.scores.final_score or 0),
        )
    )

    ranked_rows = []
    for rec in all_recommendations:
        row = recommendation_to_row(rec)
        row["company_name"] = getattr(rec, "_company_name", None)
        row["sector"] = getattr(rec, "_sector", None)
        ranked_rows.append(row)

    ranked_df = pd.DataFrame(ranked_rows)
    stock_excl_df, contract_excl_df, stock_warn_df, contract_warn_df = summarize_eligibility(results_by_symbol)

    scan_summary = {
        "symbols_in_universe": len(symbols),
        "symbols_with_recommendations": len(all_recommendations),
        "symbols_without_recommendations": max(len(symbols) - len(all_recommendations), 0),
        "total_contracts_pulled": sum(v["contract_count"] for v in results_by_symbol.values()),
        "symbols_with_provider_errors": sum(
            1 for v in results_by_symbol.values() if v.get("exception") or v.get("provider_detail_errors")
        ),
        "symbols_with_stock_warnings": sum(1 for v in results_by_symbol.values() if v.get("stock_warning_reasons")),
        "symbols_with_contract_warnings": sum(1 for v in results_by_symbol.values() if v.get("contract_warning_reasons")),
        "ready_count": int(ranked_df["decision_status"].eq("Ready").sum()) if not ranked_df.empty else 0,
        "review_count": int(ranked_df["decision_status"].eq("Review").sum()) if not ranked_df.empty else 0,
        "pass_count": int(ranked_df["decision_status"].eq("Pass").sum()) if not ranked_df.empty else 0,
    }

    st.session_state.previous_ranked_df = st.session_state.ranked_df.copy()
    st.session_state.all_recommendations = all_recommendations
    st.session_state.results_by_symbol = results_by_symbol
    st.session_state.ranked_df = ranked_df
    st.session_state.stock_excl_df = stock_excl_df
    st.session_state.contract_excl_df = contract_excl_df
    st.session_state.stock_warn_df = stock_warn_df
    st.session_state.contract_warn_df = contract_warn_df
    st.session_state.scan_summary = scan_summary
    st.session_state.scan_completed = True
    st.session_state.last_scan_cfg = cfg

    if not ranked_df.empty:
        st.session_state.selected_detail_symbol = ranked_df.iloc[0]["symbol"]

else:
    all_recommendations = st.session_state.all_recommendations
    results_by_symbol = st.session_state.results_by_symbol
    ranked_df = st.session_state.ranked_df
    stock_excl_df = st.session_state.stock_excl_df
    contract_excl_df = st.session_state.contract_excl_df
    stock_warn_df = st.session_state.stock_warn_df
    contract_warn_df = st.session_state.contract_warn_df
    scan_summary = st.session_state.scan_summary

previous_ranked_df = st.session_state.previous_ranked_df
new_symbols, dropped_symbols, movers_df = build_scan_changes(ranked_df, previous_ranked_df)

with tab_dashboard:
    st.subheader("Decision Board")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Universe scanned", scan_summary.get("symbols_in_universe", 0))
    c2.metric("Names with setups", scan_summary.get("symbols_with_recommendations", 0))
    c3.metric("Ready", scan_summary.get("ready_count", 0))
    c4.metric("Review", scan_summary.get("review_count", 0))
    c5.metric("Pass", scan_summary.get("pass_count", 0))
    c6.metric("Stock warnings", scan_summary.get("symbols_with_stock_warnings", 0))
    c7.metric("Contract warnings", scan_summary.get("symbols_with_contract_warnings", 0))

    if ranked_df.empty:
        st.warning("No recommendations found for this scan.")
    else:
        dashboard_df = ranked_df.copy()
        dashboard_df["decision_order"] = dashboard_df["decision_status"].map({"Ready": 0, "Review": 1, "Pass": 2}).fillna(9)
        dashboard_df = dashboard_df.sort_values(["decision_order", "final_score"], ascending=[True, False]).drop(columns=["decision_order"]).reset_index(drop=True)

        st.markdown("### Best trade right now")
        best = dashboard_df.iloc[0]

        next_gap = None
        if len(dashboard_df) > 1:
            next_gap = best["final_score"] - dashboard_df.iloc[1]["final_score"]

        hero1, hero2, hero3, hero4, hero5, hero6 = st.columns(6)
        hero1.metric("Symbol", best["symbol"])
        hero2.markdown(decision_badge_html(best.get("decision_status", "Review")), unsafe_allow_html=True)
        hero2.caption("Decision status")
        hero3.metric("Contract", f"{fmt_num(best['strike'])}P")
        hero4.metric("Expiration / DTE", f"{best['expiration']} / {int(best['dte'])}")
        hero5.metric("Suggested entry", fmt_num(best["entry_limit"]))
        hero6.metric("Confidence", best["confidence"])

        st.markdown(eligibility_badge_html(best.get("contract_eligibility_status", "eligible")), unsafe_allow_html=True)

        hero7, hero8, hero9, hero10, hero11, hero12 = st.columns(6)
        hero7.metric("Premium", fmt_num(best["premium"]))
        hero8.metric("Break-even", fmt_num(best["breakeven"]))
        hero9.metric("Break-even cushion", fmt_pct(best["breakeven_discount_pct"]))
        hero10.metric("Annualized yield", fmt_pct(best["annualized_secured_yield"]))
        hero11.metric("Final score", fmt_num(best["final_score"]))
        hero12.metric("Warning severity", best.get("warning_severity_label"))

        st.markdown(
            f'<div class="caption-tight"><strong>Technical:</strong> {best.get("technical_label", "Unknown")} | '
            f'<strong>Trend:</strong> {best.get("trend_state", "Unknown")} | '
            f'<strong>Watchout:</strong> {best.get("risks", "—")}</div>',
            unsafe_allow_html=True,
        )
        st.write(f"**Why it stands out:** {best.get('reasons', '—')}")
        st.write(f"**Decision rationale:** {best.get('decision_rationale', '—')}")

        if best.get("decision_cautions"):
            st.caption(f"Decision cautions: {best.get('decision_cautions')}")
        if best.get("contract_warning_reasons"):
            st.caption(f"Contract warnings: {best.get('contract_warning_reasons')}")
        if best.get("stock_warning_reasons"):
            st.caption(f"Stock warnings: {best.get('stock_warning_reasons')}")

        st.markdown("### Top actionable ideas")
        for idx, (_, row) in enumerate(dashboard_df.head(3).iterrows(), start=1):
            render_action_idea_row(row, idx)

        st.markdown("### What changed since last scan")
        change_col1, change_col2 = st.columns(2)

        with change_col1:
            st.markdown("**New symbols**")
            st.write(", ".join(new_symbols) if new_symbols else "No new symbols.")
            st.markdown("**Dropped symbols**")
            st.write(", ".join(dropped_symbols) if dropped_symbols else "No dropped symbols.")

        with change_col2:
            st.markdown("**Biggest score movers**")
            st.write("No overlapping symbols to compare." if movers_df.empty else "")
            if not movers_df.empty:
                st.dataframe(movers_df.head(5), use_container_width=True)

with tab_ranked:
    st.subheader("Ranked Setups")

    if ranked_df.empty:
        st.warning("No ranked setups available.")
    else:
        default_max_dte = int(ranked_df["dte"].max()) if "dte" in ranked_df.columns else int(cfg.max_dte)

        st.markdown("### Filter the ranked list")
        st.caption("Use these to narrow the candidate list before comparing contracts.")
        rank_filter_col1, rank_filter_col2, rank_filter_col3 = st.columns(3)

        with rank_filter_col1:
            min_confidence = st.selectbox("Minimum confidence", ["Any", "Medium", "High"], index=0, key="ranked_min_confidence")
            technical_filter = st.selectbox("Minimum technical label", ["Any", "Cautious", "Mixed", "Supportive", "Strongly supportive"], index=0, key="ranked_technical_filter")

        with rank_filter_col2:
            min_yield = st.slider("Minimum annualized yield", min_value=0.00, max_value=0.50, value=0.00, step=0.01, key="ranked_min_yield")
            min_cushion = st.slider("Minimum break-even cushion", min_value=0.00, max_value=0.30, value=0.00, step=0.01, key="ranked_min_cushion")

        with rank_filter_col3:
            max_dte_filter = st.slider("Maximum DTE", min_value=1, max_value=max(365, default_max_dte), value=default_max_dte, step=1, key="ranked_max_dte_filter")
            clean_only = st.checkbox("Show only clean Ready-style setups", value=False, key="ranked_clean_only")

        filtered_ranked_df = filter_ranked_df(
            ranked_df,
            min_confidence=min_confidence,
            min_yield=min_yield,
            min_cushion=min_cushion,
            technical_filter=technical_filter,
            max_dte_filter=max_dte_filter,
            clean_only=clean_only,
        )

        st.markdown("### Compare candidates")
        st.caption("Default view keeps only the most decision-useful columns visible.")
        sort_col = st.selectbox(
            "Sort ranked setups by",
            options=[
                "decision_status",
                "final_score",
                "warning_severity_points",
                "annualized_secured_yield",
                "breakeven_discount_pct",
                "pres",
                "stock_score",
                "contract_score",
                "distance_to_support_pct",
                "cushion_atr_units",
                "technical_score",
            ],
            index=0,
        )
        ascending = st.checkbox("Ascending sort", value=False)
        show_advanced_columns = st.checkbox("Show advanced columns", value=False)

        ranked_sorted = filtered_ranked_df.copy()
        if sort_col == "decision_status":
            ranked_sorted["decision_order"] = ranked_sorted["decision_status"].map({"Ready": 0, "Review": 1, "Pass": 2}).fillna(9)
            ranked_sorted = ranked_sorted.sort_values(["decision_order", "final_score"], ascending=[True, False]).drop(columns=["decision_order"])
        else:
            ranked_sorted = ranked_sorted.sort_values(sort_col, ascending=ascending)

        ranked_sorted = ranked_sorted.reset_index(drop=True)
        ranked_sorted.insert(0, "rank", range(1, len(ranked_sorted) + 1))

        default_cols = [
            "rank", "symbol", "expiration", "dte", "strike", "premium", "entry_limit",
            "breakeven_discount_pct", "annualized_secured_yield", "technical_label",
            "contract_eligibility_status", "warning_severity_label", "decision_status",
            "final_score", "confidence", "risks",
        ]
        advanced_cols = [
            "company_name", "sector", "stock_price", "delta_abs", "breakeven", "entry_low",
            "entry_high", "trend_state", "distance_to_support_pct", "cushion_atr_units",
            "technical_score", "stock_score", "contract_score", "pres", "warning_flags",
            "stock_warning_reasons", "contract_warning_reasons", "contract_hard_fail_reasons",
            "warning_count_total", "high_severity_warning_count", "medium_severity_warning_count",
            "low_severity_warning_count", "warning_severity_points", "decision_rationale",
            "decision_cautions", "reasons",
        ]

        display_cols = default_cols + (advanced_cols if show_advanced_columns else [])
        available_cols = [col for col in display_cols if col in ranked_sorted.columns]
        st.dataframe(ranked_sorted[available_cols], use_container_width=True)

with tab_details:
    st.subheader("Contract Details")

    if not all_recommendations:
        st.warning("No contract details available.")
    else:
        default_symbol = st.session_state.selected_detail_symbol or all_recommendations[0].symbol
        symbol_options = [rec.symbol for rec in all_recommendations]
        default_index = symbol_options.index(default_symbol) if default_symbol in symbol_options else 0
        selected_symbol = st.selectbox("Select symbol", symbol_options, index=default_index)
        st.session_state.selected_detail_symbol = selected_symbol

        selected_rec = next(rec for rec in all_recommendations if rec.symbol == selected_symbol)
        c = selected_rec.selected_contract

        symbol_payload = results_by_symbol.get(selected_symbol, {})
        all_symbol_recs = symbol_payload.get("recommendations", [])
        tech = symbol_payload.get("technical_context", {})
        tech_summary = symbol_payload.get("technical_summary", {})
        separation = grade_trade_separation(
            [rec.scores.final_score for rec in all_symbol_recs],
            selected_rec.scores.final_score,
        )

        with st.expander("Trade summary", expanded=True):
            summary_lines = describe_trade_setup(selected_rec)
            for line in summary_lines:
                st.write(line)

            a1, a2, a3, a4, a5, a6 = st.columns(6)
            a1.markdown(decision_badge_html(selected_rec.decision_status), unsafe_allow_html=True)
            a1.caption("Decision status")
            a2.metric("Contract", f"{selected_rec.symbol} {fmt_num(c.strike)}P")
            a3.metric("Suggested entry", fmt_num(selected_rec.suggested_entry_limit))
            a4.metric("50% take-profit", fmt_num(selected_rec.profit_take_debit))
            a5.metric("Break-even", fmt_num(c.breakeven_price))
            a6.metric("Confidence", selected_rec.confidence_level)

            st.markdown(eligibility_badge_html(selected_rec.contract_eligibility_status), unsafe_allow_html=True)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Annualized yield", fmt_pct(c.annualized_secured_yield))
            b2.metric("Break-even cushion", fmt_pct(c.breakeven_discount_pct))
            b3.metric("Defensive review", fmt_num(selected_rec.defensive_review_price))
            b4.metric("Score gap vs next best", fmt_num(separation.get("separation_gap")))

            b5, b6, b7 = st.columns(3)
            b5.metric("Warning severity", selected_rec.warning_severity_label)
            b6.metric("Warning points", selected_rec.warning_severity_points)
            b7.metric("Total warnings/flags", selected_rec.warning_count_total)

        with st.expander("Why this works", expanded=True):
            st.markdown("### Why this setup stands out")
            if selected_rec.top_reasons:
                for reason in selected_rec.top_reasons:
                    st.write(f"- {reason}")
            else:
                st.write("No key strengths available.")

            st.markdown("### Trade plan")
            st.write(f"- Try to enter around **{fmt_num(selected_rec.suggested_entry_limit)}**.")
            st.write(f"- Standard profit-taking level: **buy to close near {fmt_num(selected_rec.profit_take_debit)}**.")
            st.write(f"- Faster/aggressive take-profit level: **{fmt_num(selected_rec.fast_profit_take_debit)}**.")
            st.write(f"- Review the trade if the stock approaches **{fmt_num(selected_rec.defensive_review_price)}**.")
            if selected_rec.entry_notes:
                st.write(f"- Entry note: {selected_rec.entry_notes}")

            st.markdown("### Decisioning summary")
            st.write(f"- Decision status: **{selected_rec.decision_status}**")
            st.write(f"- Decision rationale: **{selected_rec.decision_rationale or '—'}**")
            st.write(f"- Warning severity: **{selected_rec.warning_severity_label}**")
            st.write(f"- Warning points: **{selected_rec.warning_severity_points}**")

            if selected_rec.decision_cautions:
                st.write(f"- Decision cautions: **{', '.join(selected_rec.decision_cautions)}**")
            if selected_rec.decision_blockers:
                st.write(f"- Decision blockers: **{', '.join(selected_rec.decision_blockers)}**")

            st.markdown("### Trade separation")
            s1, s2 = st.columns(2)
            s1.metric("Trade separation", separation.get("separation_label"))
            s2.metric("Score gap vs next best", fmt_num(separation.get("separation_gap")))
            st.write(separation.get("separation_text"))

            st.markdown("### Watchouts")
            if selected_rec.top_risks:
                for risk in selected_rec.top_risks:
                    st.write(f"- {risk}")
            else:
                st.write("No major watchouts identified.")

            st.markdown("### Validation / eligibility details")
            hard_fails = getattr(c, "contract_hard_fail_reasons", [])

            st.write(f"- Contract eligibility status: **{selected_rec.contract_eligibility_status}**")
            st.write(f"- Stock eligibility status: **{selected_rec.stock_eligibility_status}**")

            if hard_fails:
                st.write(f"- Hard fail reasons: **{', '.join(hard_fails)}**")
            else:
                st.write("- Hard fail reasons: none")

            if selected_rec.contract_warning_reasons:
                st.write(f"- Contract warning reasons: **{', '.join(selected_rec.contract_warning_reasons)}**")
            else:
                st.write("- Contract warning reasons: none")

            if selected_rec.stock_warning_reasons:
                st.write(f"- Stock warning reasons: **{', '.join(selected_rec.stock_warning_reasons)}**")
            else:
                st.write("- Stock warning reasons: none")

            if selected_rec.warning_flags:
                st.write(f"- Merged warning flags: **{', '.join(selected_rec.warning_flags)}**")

        with st.expander("Technicals", expanded=False):
            summary1, summary2, summary3 = st.columns(3)
            summary1.metric("Trend", tech.get("trend_state", "Unknown"))
            summary2.metric("Technical label", tech_summary.get("technical_label", "Unknown"))
            summary3.metric("Technical score", fmt_num(tech_summary.get("technical_score")))

            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("20D MA", fmt_num(tech.get("ma20")))
            tc2.metric("50D MA", fmt_num(tech.get("ma50")))
            tc3.metric("Support zone", fmt_num(tech.get("support_zone")))

            tc4, tc5, tc6 = st.columns(3)
            tc4.metric("ATR(14)", fmt_num(tech.get("atr14")))
            tc5.metric("RSI(14)", fmt_num(tech.get("rsi14")))
            tc6.metric("20D resistance", fmt_num(tech.get("resistance_20d")))

            tc7, tc8, tc9 = st.columns(3)
            tc7.metric("Distance to support", fmt_pct(tech.get("distance_to_support_pct")))
            tc8.metric("Cushion in ATRs", fmt_num(tech.get("cushion_atr_units")))
            tc9.metric("IV / RV ratio", fmt_num(tech.get("iv_rv_ratio")))

            st.markdown("### Price chart")
            hist = market_provider.get_price_history(selected_symbol)
            render_price_chart(hist, tech, c.breakeven_price)

        with st.expander("Alternatives on this stock", expanded=False):
            if not all_symbol_recs:
                st.write("No qualifying contracts stored for this stock.")
            else:
                contract_rows = []
                for rec in all_symbol_recs:
                    contract_row = recommendation_to_row(rec)
                    contract_rows.append(contract_row)

                contract_df = pd.DataFrame(contract_rows)

                contract_sort = st.selectbox(
                    "Sort contracts by",
                    options=["decision_status", "final_score", "warning_severity_points", "annualized_secured_yield", "breakeven_discount_pct", "contract_score", "delta_abs", "dte"],
                    index=0,
                    key="contract_sort_select",
                )

                if contract_sort == "decision_status":
                    contract_df["decision_order"] = contract_df["decision_status"].map({"Ready": 0, "Review": 1, "Pass": 2}).fillna(9)
                    contract_df = contract_df.sort_values(["decision_order", "final_score"], ascending=[True, False]).drop(columns=["decision_order"])
                else:
                    contract_df = contract_df.sort_values(contract_sort, ascending=False)

                contract_df = contract_df.reset_index(drop=True)
                contract_df.insert(0, "rank", range(1, len(contract_df) + 1))

                contract_display_cols = [
                    "rank", "expiration", "dte", "strike", "premium", "entry_limit",
                    "breakeven_discount_pct", "annualized_secured_yield",
                    "contract_eligibility_status", "warning_severity_label",
                    "decision_status", "final_score", "confidence",
                    "contract_warning_reasons", "risks",
                ]
                available_cols = [col for col in contract_display_cols if col in contract_df.columns]
                st.dataframe(contract_df[available_cols], use_container_width=True)

with tab_diagnostics:
    st.subheader("Diagnostics")

    diag_row1_col1, diag_row1_col2 = st.columns(2)
    with diag_row1_col1:
        st.markdown("### Stock-level hard fails")
        st.dataframe(stock_excl_df, use_container_width=True) if not stock_excl_df.empty else st.write("No stock-level hard fails recorded.")
    with diag_row1_col2:
        st.markdown("### Contract-level hard fails")
        st.dataframe(contract_excl_df, use_container_width=True) if not contract_excl_df.empty else st.write("No contract-level hard fails recorded.")

    diag_row2_col1, diag_row2_col2 = st.columns(2)
    with diag_row2_col1:
        st.markdown("### Stock-level warnings")
        st.dataframe(stock_warn_df, use_container_width=True) if not stock_warn_df.empty else st.write("No stock-level warnings recorded.")
    with diag_row2_col2:
        st.markdown("### Contract-level warnings")
        st.dataframe(contract_warn_df, use_container_width=True) if not contract_warn_df.empty else st.write("No contract-level warnings recorded.")

    if not ranked_df.empty:
        st.markdown("### Decision status distribution")
        decision_dist = (
            ranked_df.groupby(["decision_status", "warning_severity_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["decision_status", "count"], ascending=[True, False])
        )
        st.dataframe(decision_dist, use_container_width=True)

with tab_debug:
    st.subheader("Debug / Validation")

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

    pasted_text = st.text_area("Paste content here", height=250)

    if st.button("Validate pasted content"):
        if debug_mode == "Raw text / traceback":
            st.code(pasted_text)
        elif debug_mode == "JSON / dict payload":
            parsed, parser_used, parse_error = try_parse_structured(pasted_text)
            if parsed is None:
                st.error(f"Could not parse structured content: {parse_error}")
                st.code(pasted_text)
            else:
                st.success(f"Parsed successfully using: {parser_used}")
                st.json(parsed)
        elif debug_mode == "Recommendation row":
            parsed, parser_used, parse_error = try_parse_structured(pasted_text)
            if parsed is None or not isinstance(parsed, dict):
                st.error("Recommendation row should parse into a dictionary-like object.")
                if parse_error:
                    st.caption(parse_error)
            else:
                st.success(f"Parsed recommendation row using: {parser_used}")
                st.json(parsed)
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
                st.dataframe(validate_contract_like(parsed), use_container_width=True)
        else:
            st.code(pasted_text)
