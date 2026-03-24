from config import ScanConfig
from models import OptionContract, StockMetrics


def validate_stock(metrics: StockMetrics, cfg: ScanConfig) -> StockMetrics:
    metrics.stock_valid = True
    metrics.stock_exclusion_reasons = []

    if metrics.stock_price is None:
        metrics.stock_valid = False
        metrics.stock_exclusion_reasons.append("missing_stock_price")
        return metrics

    if metrics.stock_price < cfg.min_stock_price:
        metrics.stock_valid = False
        metrics.stock_exclusion_reasons.append("stock_price_below_min")

    if metrics.stock_price > cfg.max_stock_price:
        metrics.stock_valid = False
        metrics.stock_exclusion_reasons.append("stock_price_above_max")

    if cfg.require_quality_data and not metrics.quality_data_complete:
        metrics.stock_valid = False
        metrics.stock_exclusion_reasons.append("missing_quality_data_required")

    return metrics


def validate_contract(contract: OptionContract, cfg: ScanConfig) -> OptionContract:
    contract.contract_valid = True
    contract.contract_exclusion_reasons = []

    if contract.option_type.upper() != "PUT":
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("not_put")

    if cfg.otm_only and contract.in_the_money:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("itm")

    if contract.dte < cfg.min_dte or contract.dte > cfg.max_dte:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("dte_out_of_range")

    if contract.delta is None:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("missing_delta")
    else:
        delta_abs = abs(contract.delta)
        if delta_abs < cfg.min_abs_delta or delta_abs > cfg.max_abs_delta:
            contract.contract_valid = False
            contract.contract_exclusion_reasons.append("delta_out_of_range")

    if contract.bid is None or contract.ask is None:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("missing_bid_ask")
        return contract

    if contract.bid < 0 or contract.ask < 0 or contract.ask < contract.bid:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("invalid_bid_ask")

    if contract.open_interest is None and contract.volume is None:
        contract.contract_valid = False
        contract.contract_exclusion_reasons.append("missing_oi_and_volume")
        return contract

    if contract.open_interest is not None and contract.open_interest < cfg.min_open_interest:
        contract.contract_exclusion_reasons.append("oi_below_min")

    if contract.volume is not None and contract.volume < cfg.min_volume:
        contract.contract_exclusion_reasons.append("volume_below_min")

    if cfg.strict_data_mode:
        if contract.open_interest is not None and contract.open_interest < cfg.min_open_interest:
            contract.contract_valid = False
        if contract.volume is not None and contract.volume < cfg.min_volume:
            contract.contract_valid = False

    return contract
