from models import OptionContract
from utils import safe_div


def derive_contract_metrics(contract: OptionContract, stock_price: float) -> OptionContract:
    if contract.bid is not None and contract.ask is not None:
        contract.mid_price = (contract.bid + contract.ask) / 2.0
        contract.spread_dollars = contract.ask - contract.bid

    if contract.mark is not None:
        contract.premium = contract.mark
    elif contract.mid_price is not None:
        contract.premium = contract.mid_price
    else:
        contract.premium = contract.bid

    if contract.mid_price is not None and contract.spread_dollars is not None and contract.mid_price > 0:
        contract.spread_pct = contract.spread_dollars / contract.mid_price

    if contract.premium is not None:
        contract.credit_per_contract = contract.premium * 100.0

    contract.cash_secured_requirement = contract.strike * 100.0

    if contract.premium is not None:
        contract.breakeven_price = contract.strike - contract.premium

    if contract.breakeven_price is not None and stock_price > 0:
        contract.breakeven_discount_pct = safe_div(
            stock_price - contract.breakeven_price,
            stock_price,
            default=0.0,
        )

    if (
        contract.credit_per_contract is not None
        and contract.cash_secured_requirement is not None
        and contract.cash_secured_requirement > 0
        and contract.dte > 0
    ):
        secured_yield = contract.credit_per_contract / contract.cash_secured_requirement
        contract.annualized_secured_yield = secured_yield * (365.0 / contract.dte)

    if contract.delta is not None:
        contract.delta_abs = abs(contract.delta)

    return contract
