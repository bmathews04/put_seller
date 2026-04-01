from dataclasses import dataclass


@dataclass
class ScanConfig:
    max_symbols_to_scan: int = 50

    min_stock_price: float = 5.0
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
    min_bid: float = 0.25
    min_premium: float = 0.35

    otm_only: bool = True
    exclude_earnings_before_expiry: bool = True

    # New: controls whether unknown earnings date is a hard fail when
    # exclude_earnings_before_expiry is enabled.
    strict_earnings_date_handling: bool = False

    require_quality_data: bool = False
    strict_data_mode: bool = False
    quality_fallback_mode: bool = True
    default_quality_score_if_missing: float = 55.0

    allow_expiry_week_hold: bool = False
    review_dte: int = 3
    profit_take_pct: float = 0.50
    fast_profit_take_pct: float = 0.35

    weight_stock_score: float = 0.40
    weight_contract_score: float = 0.45
    weight_pres: float = 0.15
