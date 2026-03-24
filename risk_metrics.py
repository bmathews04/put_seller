from models import OptionContract
from utils import min_max_normalize


def quality_multiplier_from_score(quality_score: float) -> float:
    return 0.85 + (0.003 * quality_score)


def liquidity_penalty_from_score(liquidity_score: float) -> float:
    return max(0.0, (70.0 - liquidity_score) / 100.0)


def event_penalty_from_score(event_score: float) -> float:
    if event_score >= 95:
        return 0.0
    if event_score >= 55:
        return 0.15
    return 0.50


def compute_pres_raw(
    contract: OptionContract,
    quality_score: float,
    liquidity_score: float,
    event_stability_score: float,
) -> float:
    annualized_yield = contract.annualized_secured_yield or 0.0
    breakeven_discount = contract.breakeven_discount_pct or 0.0
    delta_abs = contract.delta_abs or 0.0
    spread_pct = contract.spread_pct or 0.0

    q_mult = quality_multiplier_from_score(quality_score)
    l_pen = liquidity_penalty_from_score(liquidity_score)
    e_pen = event_penalty_from_score(event_stability_score)

    numerator = annualized_yield * (1.0 + breakeven_discount) * q_mult
    denominator = (
        (1.0 + 2.0 * delta_abs)
        * (1.0 + spread_pct)
        * (1.0 + l_pen)
        * (1.0 + e_pen)
    )

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def normalize_pres_values(raw_values: list[float]) -> list[float]:
    return min_max_normalize(raw_values)
