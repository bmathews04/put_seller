from config import ScanConfig
from models import OptionContract, StockMetrics


def _reset_stock_validation(metrics: StockMetrics) -> None:
    metrics.stock_valid = True
    metrics.stock_eligibility_status = "eligible"
    metrics.stock_exclusion_reasons = []
    metrics.stock_hard_fail_reasons = []
    metrics.stock_warning_reasons = []


def _reset_contract_validation(contract: OptionContract) -> None:
    contract.liquidity_valid = True
    contract.contract_valid = True
    contract.contract_eligibility_status = "eligible"
    contract.contract_exclusion_reasons = []
    contract.contract_hard_fail_reasons = []
    contract.contract_warning_reasons = []


def _add_stock_hard_fail(metrics: StockMetrics, reason: str) -> None:
    if reason not in metrics.stock_hard_fail_reasons:
        metrics.stock_hard_fail_reasons.append(reason)


def _add_stock_warning(metrics: StockMetrics, reason: str) -> None:
    if reason not in metrics.stock_warning_reasons:
        metrics.stock_warning_reasons.append(reason)


def _finalize_stock_validation(metrics: StockMetrics) -> StockMetrics:
    metrics.stock_valid = len(metrics.stock_hard_fail_reasons) == 0

    if metrics.stock_hard_fail_reasons:
        metrics.stock_eligibility_status = "ineligible"
    elif metrics.stock_warning_reasons:
        metrics.stock_eligibility_status = "eligible_with_warnings"
    else:
        metrics.stock_eligibility_status = "eligible"

    # Backward-compatible alias for existing UI/reporting code
    metrics.stock_exclusion_reasons = list(metrics.stock_hard_fail_reasons)
    return metrics


def _add_contract_hard_fail(contract: OptionContract, reason: str) -> None:
    if reason not in contract.contract_hard_fail_reasons:
        contract.contract_hard_fail_reasons.append(reason)


def _add_contract_warning(contract: OptionContract, reason: str) -> None:
    if reason not in contract.contract_warning_reasons:
        contract.contract_warning_reasons.append(reason)


def _finalize_contract_validation(contract: OptionContract) -> OptionContract:
    contract.contract_valid = len(contract.contract_hard_fail_reasons) == 0

    contract.liquidity_valid = not any(
        reason in {"missing_oi_and_volume", "oi_below_min", "volume_below_min", "spread_too_wide"}
        for reason in contract.contract_hard_fail_reasons
    )

    if contract.contract_hard_fail_reasons:
        contract.contract_eligibility_status = "ineligible"
    elif contract.contract_warning_reasons:
        contract.contract_eligibility_status = "eligible_with_warnings"
    else:
        contract.contract_eligibility_status = "eligible"

    # Backward-compatible alias for existing UI/reporting code
    contract.contract_exclusion_reasons = list(contract.contract_hard_fail_reasons)
    return contract


def validate_stock(metrics: StockMetrics, cfg: ScanConfig) -> StockMetrics:
    _reset_stock_validation(metrics)

    if metrics.stock_price is None:
        _add_stock_hard_fail(metrics, "missing_stock_price")
        return _finalize_stock_validation(metrics)

    if metrics.stock_price < cfg.min_stock_price:
        _add_stock_hard_fail(metrics, "stock_price_below_min")

    if metrics.stock_price > cfg.max_stock_price:
        _add_stock_hard_fail(metrics, "stock_price_above_max")

    if cfg.require_quality_data and not metrics.quality_data_complete:
        _add_stock_hard_fail(metrics, "missing_quality_data_required")
    elif not metrics.quality_data_complete:
        _add_stock_warning(metrics, "quality_data_incomplete")

    return _finalize_stock_validation(metrics)


def validate_contract(
    contract: OptionContract,
    cfg: ScanConfig,
    metrics: StockMetrics | None = None,
) -> OptionContract:
    _reset_contract_validation(contract)

    if contract.option_type.upper() != "PUT":
        _add_contract_hard_fail(contract, "not_put")

    if cfg.otm_only and contract.in_the_money:
        _add_contract_hard_fail(contract, "itm")

    if contract.dte < cfg.min_dte or contract.dte > cfg.max_dte:
        _add_contract_hard_fail(contract, "dte_out_of_range")

    if contract.delta is None:
        _add_contract_hard_fail(contract, "missing_delta")
    else:
        delta_abs = abs(contract.delta)
        if delta_abs < cfg.min_abs_delta or delta_abs > cfg.max_abs_delta:
            _add_contract_hard_fail(contract, "delta_out_of_range")

    if contract.bid is None or contract.ask is None:
        _add_contract_hard_fail(contract, "missing_bid_ask")
        return _finalize_contract_validation(contract)

    if contract.bid < 0 or contract.ask < 0 or contract.ask < contract.bid:
        _add_contract_hard_fail(contract, "invalid_bid_ask")
        return _finalize_contract_validation(contract)

    if contract.bid < cfg.min_bid:
        _add_contract_hard_fail(contract, "bid_below_min")

    if contract.premium is None:
        _add_contract_hard_fail(contract, "missing_premium")
    elif contract.premium < cfg.min_premium:
        _add_contract_hard_fail(contract, "premium_too_low")

    if contract.spread_pct is not None and contract.spread_pct > cfg.max_spread_pct:
        _add_contract_hard_fail(contract, "spread_too_wide")
    elif contract.spread_pct is not None and contract.spread_pct > min(0.10, cfg.max_spread_pct):
        _add_contract_warning(contract, "spread_moderately_wide")

    if cfg.exclude_earnings_before_expiry and metrics is not None:
        if metrics.days_to_earnings is None:
            _add_contract_hard_fail(contract, "earnings_date_unknown")
        elif metrics.days_to_earnings <= contract.dte:
            _add_contract_hard_fail(contract, "earnings_before_expiry")
    elif metrics is not None:
        if metrics.days_to_earnings is None:
            _add_contract_warning(contract, "earnings_date_unknown")
        elif metrics.days_to_earnings <= contract.dte:
            _add_contract_warning(contract, "earnings_before_expiry")

    if contract.open_interest is None and contract.volume is None:
        _add_contract_hard_fail(contract, "missing_oi_and_volume")
        return _finalize_contract_validation(contract)

    if cfg.strict_data_mode:
        if contract.open_interest is None or contract.open_interest < cfg.min_open_interest:
            _add_contract_hard_fail(contract, "oi_below_min")
        if contract.volume is None or contract.volume < cfg.min_volume:
            _add_contract_hard_fail(contract, "volume_below_min")
    else:
        if contract.open_interest is None:
            _add_contract_warning(contract, "oi_unknown")
        elif contract.open_interest < cfg.min_open_interest:
            _add_contract_warning(contract, "oi_below_min")

        if contract.volume is None:
            _add_contract_warning(contract, "volume_unknown")
        elif contract.volume < cfg.min_volume:
            _add_contract_warning(contract, "volume_below_min")

    return _finalize_contract_validation(contract)
