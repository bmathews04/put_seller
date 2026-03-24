from config import ScanConfig
from models import ScoreCard, StockMetrics
from utils import clamp, half_saturation_score


def score_quality(metrics: StockMetrics, cfg: ScanConfig) -> float:
    if metrics.fcf_yield is None:
        return cfg.default_quality_score_if_missing
    return half_saturation_score(metrics.fcf_yield, cfg.fcf_half_sat)


def score_event_stability(metrics: StockMetrics, expiry_dte: int | None, cfg: ScanConfig) -> float:
    if metrics.earnings_date is None or metrics.days_to_earnings is None:
        return cfg.unknown_event_score

    if expiry_dte is not None and metrics.days_to_earnings <= expiry_dte:
        return cfg.unsafe_event_score

    return cfg.safe_event_score


def score_options_market_quality(
    median_spread_pct: float | None,
    median_oi: float | None,
    median_volume: float | None,
    candidate_count: int,
    cfg: ScanConfig,
) -> float:
    spread_score = 50.0
    oi_score = 50.0
    volume_score = 50.0
    candidate_depth_score = 40.0

    if median_spread_pct is not None:
        spread_score = clamp(100.0 * (1.0 - (median_spread_pct / 0.20)))

    if median_oi is not None and median_oi > 0:
        oi_score = half_saturation_score(median_oi, cfg.oi_half_sat)

    if median_volume is not None and median_volume > 0:
        volume_score = half_saturation_score(median_volume, cfg.volume_half_sat)

    if candidate_count >= 5:
        candidate_depth_score = 100.0
    elif candidate_count >= 3:
        candidate_depth_score = 70.0
    elif candidate_count >= 1:
        candidate_depth_score = 40.0
    else:
        candidate_depth_score = 0.0

    return (
        0.50 * spread_score
        + 0.25 * oi_score
        + 0.20 * volume_score
        + 0.05 * candidate_depth_score
    )


def score_assignment_comfort(best_breakeven_discount_pct: float | None, cfg: ScanConfig) -> float:
    if best_breakeven_discount_pct is None or best_breakeven_discount_pct <= 0:
        return 0.0
    return half_saturation_score(best_breakeven_discount_pct, cfg.breakeven_half_sat)


def build_stock_scorecard(
    metrics: StockMetrics,
    best_contract_dte: int | None,
    median_spread_pct: float | None,
    median_oi: float | None,
    median_volume: float | None,
    best_breakeven_discount_pct: float | None,
    cfg: ScanConfig,
) -> ScoreCard:
    scorecard = ScoreCard()
    scorecard.quality_score = score_quality(metrics, cfg)
    scorecard.event_stability_score = score_event_stability(metrics, best_contract_dte, cfg)
    scorecard.options_market_quality_score = score_options_market_quality(
        median_spread_pct=median_spread_pct,
        median_oi=median_oi,
        median_volume=median_volume,
        candidate_count=metrics.candidate_contract_count,
        cfg=cfg,
    )
    scorecard.assignment_comfort_score = score_assignment_comfort(best_breakeven_discount_pct, cfg)

    scorecard.stock_score_total = (
        cfg.weight_quality * scorecard.quality_score
        + cfg.weight_event_stability * scorecard.event_stability_score
        + cfg.weight_options_market_quality * scorecard.options_market_quality_score
        + cfg.weight_assignment_comfort * scorecard.assignment_comfort_score
    )
    return scorecard
