from dataclasses import dataclass


@dataclass(frozen=True)
class ScanConfig:
    universe_name: str = "sp500"
    max_symbols_to_scan: int | None = 100

    min_stock_price: float = 20.0
    max_stock_price: float = 1000.0

    min_dte: int = 25
    max_dte: int = 40
    target_dte: int = 32

    min_abs_delta: float = 0.12
    max_abs_delta: float = 0.22
    target_abs_delta: float = 0.17

    min_open_interest: int = 500
    min_volume: int = 25
    max_spread_pct: float = 0.12

    min_bid: float = 0.35
    min_premium: float = 0.35
    otm_only: bool = True

    exclude_earnings_before_expiry: bool = True
    require_quality_data: bool = False
    quality_fallback_mode: bool = True
    strict_data_mode: bool = False

    profit_take_pct: float = 0.50
    fast_profit_take_pct: float = 0.70
    review_dte: int = 21
    allow_expiry_week_hold: bool = False

    weight_stock_score: float = 0.40
    weight_contract_score: float = 0.35
    weight_pres: float = 0.25

    weight_quality: float = 0.45
    weight_event_stability: float = 0.20
    weight_options_market_quality: float = 0.20
    weight_assignment_comfort: float = 0.15

    weight_breakeven: float = 0.30
    weight_secured_yield: float = 0.25
    weight_delta_fit: float = 0.20
    weight_liquidity: float = 0.15
    weight_dte_fit: float = 0.10

    fcf_half_sat: float = 0.05
    breakeven_half_sat: float = 0.04
    annualized_yield_half_sat: float = 0.10
    oi_half_sat: float = 1000.0
    volume_half_sat: float = 50.0

    default_quality_score_if_missing: float = 55.0

    unknown_event_score: float = 60.0
    unsafe_event_score: float = 20.0
    safe_event_score: float = 100.0
