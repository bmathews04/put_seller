from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class StockMetadata:
    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    is_sp500_member: bool = True
    market_cap: float | None = None


@dataclass
class StockMetrics:
    symbol: str
    stock_price: float | None = None
    price_timestamp: datetime | None = None
    earnings_date: date | None = None
    days_to_earnings: int | None = None
    fcf_yield: float | None = None
    quality_data_source: str | None = None
    quality_data_complete: bool = False
    avg_candidate_chain_spread_pct: float | None = None
    candidate_contract_count: int = 0

    stock_valid: bool = True
    stock_eligibility_status: str = "eligible"

    # Backward-compatible alias used by current UI/reporting paths
    stock_exclusion_reasons: list[str] = field(default_factory=list)

    # New split model
    stock_hard_fail_reasons: list[str] = field(default_factory=list)
    stock_warning_reasons: list[str] = field(default_factory=list)


@dataclass
class OptionContract:
    symbol: str
    expiration_date: date
    dte: int
    strike: float
    option_type: str

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mark: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    implied_volatility: float | None = None
    open_interest: int | None = None
    volume: int | None = None
    in_the_money: bool = False
    quote_timestamp: datetime | None = None

    mid_price: float | None = None
    premium: float | None = None
    spread_dollars: float | None = None
    spread_pct: float | None = None
    cash_secured_requirement: float | None = None
    credit_per_contract: float | None = None
    breakeven_price: float | None = None
    breakeven_discount_pct: float | None = None
    annualized_secured_yield: float | None = None
    delta_abs: float | None = None

    liquidity_valid: bool = True
    contract_valid: bool = True
    contract_eligibility_status: str = "eligible"

    # Backward-compatible alias used by current UI/reporting paths
    contract_exclusion_reasons: list[str] = field(default_factory=list)

    # New split model
    contract_hard_fail_reasons: list[str] = field(default_factory=list)
    contract_warning_reasons: list[str] = field(default_factory=list)


@dataclass
class ScoreCard:
    quality_score: float = 0.0
    event_stability_score: float = 0.0
    options_market_quality_score: float = 0.0
    assignment_comfort_score: float = 0.0
    stock_score_total: float = 0.0

    breakeven_score: float = 0.0
    secured_yield_score: float = 0.0
    delta_fit_score: float = 0.0
    liquidity_score: float = 0.0
    dte_fit_score: float = 0.0
    contract_score_total: float = 0.0

    pres_raw: float = 0.0
    pres_normalized: float = 0.0
    final_score: float = 0.0


@dataclass
class Recommendation:
    symbol: str
    stock_price: float
    selected_contract: OptionContract
    scores: ScoreCard

    suggested_entry_limit: float | None = None
    acceptable_entry_low: float | None = None
    acceptable_entry_high: float | None = None
    entry_style: str | None = None
    entry_notes: str | None = None

    profit_take_debit: float | None = None
    fast_profit_take_debit: float | None = None
    review_at_dte: int | None = None
    defensive_review_price: float | None = None
    roll_candidate_flag: bool = False
    management_plan_text: str | None = None

    top_reasons: list[str] = field(default_factory=list)
    top_risks: list[str] = field(default_factory=list)
    warning_flags: list[str] = field(default_factory=list)
    confidence_level: str = "Medium"
