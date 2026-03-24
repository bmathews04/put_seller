from config import ScanConfig
from models import OptionContract, ScoreCard
from utils import clamp, center_fit_score, half_saturation_score


def score_breakeven(contract: OptionContract, cfg: ScanConfig) -> float:
    b = contract.breakeven_discount_pct or 0.0
    return half_saturation_score(max(b, 0.0), cfg.breakeven_half_sat)


def score_secured_yield(contract: OptionContract, cfg: ScanConfig) -> float:
    y = contract.annualized_secured_yield or 0.0
    return half_saturation_score(max(y, 0.0), cfg.annualized_yield_half_sat)


def score_delta_fit(contract: OptionContract, cfg: ScanConfig) -> float:
    if contract.delta_abs is None:
        return 0.0
    return center_fit_score(
        value=contract.delta_abs,
        target=cfg.target_abs_delta,
        minimum=cfg.min_abs_delta,
        maximum=cfg.max_abs_delta,
    )


def score_liquidity(contract: OptionContract, cfg: ScanConfig) -> float:
    spread_score = 0.0
    oi_score = 0.0
    volume_score = 0.0

    if contract.spread_pct is not None:
        spread_score = clamp(100.0 * (1.0 - contract.spread_pct / cfg.max_spread_pct))

    if contract.open_interest is not None and contract.open_interest > 0:
        oi_score = half_saturation_score(contract.open_interest, cfg.oi_half_sat)

    if contract.volume is not None and contract.volume > 0:
        volume_score = half_saturation_score(contract.volume, cfg.volume_half_sat)

    return (
        0.55 * spread_score
        + 0.25 * oi_score
        + 0.20 * volume_score
    )


def score_dte_fit(contract: OptionContract, cfg: ScanConfig) -> float:
    return center_fit_score(
        value=contract.dte,
        target=cfg.target_dte,
        minimum=cfg.min_dte,
        maximum=cfg.max_dte,
    )


def build_contract_scorecard(contract: OptionContract, cfg: ScanConfig) -> ScoreCard:
    scorecard = ScoreCard()
    scorecard.breakeven_score = score_breakeven(contract, cfg)
    scorecard.secured_yield_score = score_secured_yield(contract, cfg)
    scorecard.delta_fit_score = score_delta_fit(contract, cfg)
    scorecard.liquidity_score = score_liquidity(contract, cfg)
    scorecard.dte_fit_score = score_dte_fit(contract, cfg)

    scorecard.contract_score_total = (
        cfg.weight_breakeven * scorecard.breakeven_score
        + cfg.weight_secured_yield * scorecard.secured_yield_score
        + cfg.weight_delta_fit * scorecard.delta_fit_score
        + cfg.weight_liquidity * scorecard.liquidity_score
        + cfg.weight_dte_fit * scorecard.dte_fit_score
    )
    return scorecard
