from statistics import median

from config import ScanConfig
from contract_scoring import build_contract_scorecard
from derivations import derive_contract_metrics
from models import OptionContract, Recommendation, ScoreCard, StockMetrics
from risk_metrics import compute_pres_raw, normalize_pres_values
from stock_scoring import build_stock_scorecard
from validators import validate_contract, validate_stock


def suggest_entry(contract: OptionContract) -> tuple[float | None, float | None, float | None, str, str]:
    if contract.bid is None or contract.ask is None or contract.mid_price is None or contract.spread_dollars is None:
        return None, None, None, "Unavailable", "Entry suggestion unavailable due to incomplete quote data."

    spread_pct = contract.spread_pct or 0.0
    mid = contract.mid_price
    spread = contract.spread_dollars
    bid = contract.bid

    if spread_pct <= 0.05:
        suggested = mid + 0.10 * spread
        low = mid
        high = mid + 0.20 * spread
        style = "Mid-plus"
        note = "Tight spread supports working near or slightly above midpoint."
    elif spread_pct <= 0.12:
        suggested = mid
        low = bid + 0.40 * spread
        high = mid
        style = "Mid"
        note = "Moderate spread suggests working around midpoint."
    elif spread_pct <= 0.20:
        suggested = mid - 0.10 * spread
        low = bid + 0.25 * spread
        high = mid
        style = "Conservative"
        note = "Looser spread suggests being more conservative on entry."
    else:
        return None, None, None, "Avoid", "Spread is too wide for a high-confidence entry."

    return round(suggested, 2), round(low, 2), round(high, 2), style, note


def build_management_plan(stock_price: float, contract: OptionContract, cfg: ScanConfig):
    if contract.premium is None or contract.breakeven_price is None:
        return None, None, cfg.review_dte, None, False, "Management plan unavailable due to incomplete contract data."

    entry_credit = contract.premium
    profit_take_debit = round(entry_credit * (1.0 - cfg.profit_take_pct), 2)
    fast_profit_take_debit = round(entry_credit * (1.0 - cfg.fast_profit_take_pct), 2)

    cushion = max(stock_price - contract.breakeven_price, 0.0)
    defensive_review_price = round(contract.breakeven_price + 0.25 * cushion, 2)

    roll_candidate_flag = True

    # Make allow_expiry_week_hold a real management setting
    if cfg.allow_expiry_week_hold:
        review_at_dte = cfg.review_dte
        hold_guidance = "Holding into expiration week can be acceptable if price action, liquidity, and assignment comfort remain favorable."
    else:
        review_at_dte = max(cfg.review_dte, 5)
        hold_guidance = "Avoid holding into expiration week; plan to review, close, or roll before 5 DTE unless conditions materially improve."

    text = (
        f"Default plan: consider buying back at {profit_take_debit:.2f} "
        f"for the standard profit target, review the position around {review_at_dte} DTE, "
        f"and begin defensive review if the stock approaches {defensive_review_price:.2f}. "
        f"{hold_guidance}"
    )

    return (
        profit_take_debit,
        fast_profit_take_debit,
        review_at_dte,
        defensive_review_price,
        roll_candidate_flag,
        text,
    )


def assign_confidence(metrics: StockMetrics, contract: OptionContract, stock_score: ScoreCard, contract_score: ScoreCard) -> str:
    concerns = 0

    if not metrics.quality_data_complete:
        concerns += 1

    if metrics.earnings_date is None:
        concerns += 2

    if contract.spread_pct is not None and contract.spread_pct > 0.10:
        concerns += 1

    if contract_score.liquidity_score < 55:
        concerns += 1

    if stock_score.event_stability_score < 50:
        concerns += 1

    if concerns <= 1:
        return "High"
    if concerns <= 3:
        return "Medium"
    return "Low"


def build_reasons_and_risks(metrics: StockMetrics, contract: OptionContract, stock_scores: ScoreCard, contract_scores: ScoreCard):
    positives = []
    negatives = []
    flags = []

    positives.append((stock_scores.quality_score, "Strong underlying quality profile"))
    positives.append((contract_scores.breakeven_score, "Healthy break-even cushion"))
    positives.append((contract_scores.secured_yield_score, "Attractive secured yield"))
    positives.append((contract_scores.delta_fit_score, "Delta sits near the preferred range"))
    positives.append((contract_scores.liquidity_score, "Options market looks tradable"))
    positives.append((stock_scores.event_stability_score, "No obvious near-term event pressure"))

    if not metrics.quality_data_complete:
        negatives.append((80, "Quality score used fallback data"))
        flags.append("quality_data_incomplete")

    if metrics.earnings_date is None:
        negatives.append((75, "Earnings date is unknown"))
        flags.append("earnings_unknown")

    if contract.spread_pct is not None and contract.spread_pct > 0.10:
        negatives.append((70, "Spread is on the wider side"))
        flags.append("spread_moderately_wide")

    if contract.delta_abs is not None and contract.delta_abs > 0.20:
        negatives.append((60, "Delta is near the high end of the target band"))
        flags.append("delta_near_upper_band")

    if contract_scores.liquidity_score < 50:
        negatives.append((65, "Liquidity is only fair"))
        flags.append("liquidity_borderline")

    if contract.annualized_secured_yield is not None and contract.annualized_secured_yield < 0.06:
        negatives.append((50, "Premium is relatively thin for the capital required"))
        flags.append("thin_yield")

    top_reasons = [text for _, text in sorted(positives, key=lambda x: x[0], reverse=True)[:3]]
    top_risks = [text for _, text in sorted(negatives, key=lambda x: x[0], reverse=True)[:2]]

    return top_reasons, top_risks, flags


def combine_scorecards(stock_scores: ScoreCard, contract_scores: ScoreCard, pres_normalized: float, cfg: ScanConfig) -> ScoreCard:
    combined = ScoreCard(
        quality_score=stock_scores.quality_score,
        event_stability_score=stock_scores.event_stability_score,
        options_market_quality_score=stock_scores.options_market_quality_score,
        assignment_comfort_score=stock_scores.assignment_comfort_score,
        stock_score_total=stock_scores.stock_score_total,
        breakeven_score=contract_scores.breakeven_score,
        secured_yield_score=contract_scores.secured_yield_score,
        delta_fit_score=contract_scores.delta_fit_score,
        liquidity_score=contract_scores.liquidity_score,
        dte_fit_score=contract_scores.dte_fit_score,
        contract_score_total=contract_scores.contract_score_total,
        pres_normalized=pres_normalized,
    )

    combined.final_score = (
        cfg.weight_stock_score * combined.stock_score_total
        + cfg.weight_contract_score * combined.contract_score_total
        + cfg.weight_pres * combined.pres_normalized
    )
    return combined


def build_recommendations_for_stock(metrics: StockMetrics, raw_contracts: list[OptionContract], cfg: ScanConfig) -> list[Recommendation]:
    metrics = validate_stock(metrics, cfg)
    if not metrics.stock_valid or metrics.stock_price is None:
        return []

    valid_contracts: list[OptionContract] = []

    for raw_contract in raw_contracts:
        contract = derive_contract_metrics(raw_contract, metrics.stock_price)
        contract = validate_contract(contract, cfg, metrics)

        if contract.contract_valid:
            valid_contracts.append(contract)

    if not valid_contracts:
        metrics.stock_valid = False
        metrics.stock_exclusion_reasons.append("no_valid_contracts")
        return []

    metrics.candidate_contract_count = len(valid_contracts)

    median_spread_pct = median([c.spread_pct for c in valid_contracts if c.spread_pct is not None]) if valid_contracts else None
    median_oi = median([c.open_interest for c in valid_contracts if c.open_interest is not None]) if valid_contracts else None
    median_volume = median([c.volume for c in valid_contracts if c.volume is not None]) if valid_contracts else None

    seed_stock_scores = build_stock_scorecard(
        metrics=metrics,
        best_contract_dte=valid_contracts[0].dte,
        median_spread_pct=median_spread_pct,
        median_oi=median_oi,
        median_volume=median_volume,
        best_breakeven_discount_pct=max((c.breakeven_discount_pct or 0.0) for c in valid_contracts),
        cfg=cfg,
    )

    contract_scorecards = []
    pres_raw_values = []

    for contract in valid_contracts:
        c_score = build_contract_scorecard(contract, cfg)
        p_raw = compute_pres_raw(
            contract=contract,
            quality_score=seed_stock_scores.quality_score,
            liquidity_score=c_score.liquidity_score,
            event_stability_score=seed_stock_scores.event_stability_score,
        )
        c_score.pres_raw = p_raw
        contract_scorecards.append(c_score)
        pres_raw_values.append(p_raw)

    pres_norm_values = normalize_pres_values(pres_raw_values)

    recommendations = []

    for contract, c_score, pres_norm in zip(valid_contracts, contract_scorecards, pres_norm_values):
        stock_scores = build_stock_scorecard(
            metrics=metrics,
            best_contract_dte=contract.dte,
            median_spread_pct=median_spread_pct,
            median_oi=median_oi,
            median_volume=median_volume,
            best_breakeven_discount_pct=contract.breakeven_discount_pct,
            cfg=cfg,
        )

        combined_scores = combine_scorecards(stock_scores, c_score, pres_norm, cfg)

        entry_limit, entry_low, entry_high, entry_style, entry_notes = suggest_entry(contract)
        profit_take_debit, fast_profit_take_debit, review_at_dte, defensive_review_price, roll_candidate_flag, management_plan_text = build_management_plan(
            stock_price=metrics.stock_price,
            contract=contract,
            cfg=cfg,
        )

        top_reasons, top_risks, flags = build_reasons_and_risks(
            metrics=metrics,
            contract=contract,
            stock_scores=stock_scores,
            contract_scores=c_score,
        )

        recommendations.append(
            Recommendation(
                symbol=contract.symbol,
                stock_price=metrics.stock_price,
                selected_contract=contract,
                scores=combined_scores,
                suggested_entry_limit=entry_limit,
                acceptable_entry_low=entry_low,
                acceptable_entry_high=entry_high,
                entry_style=entry_style,
                entry_notes=entry_notes,
                profit_take_debit=profit_take_debit,
                fast_profit_take_debit=fast_profit_take_debit,
                review_at_dte=review_at_dte,
                defensive_review_price=defensive_review_price,
                roll_candidate_flag=roll_candidate_flag,
                management_plan_text=management_plan_text,
                top_reasons=top_reasons,
                top_risks=top_risks,
                warning_flags=flags,
                confidence_level=assign_confidence(metrics, contract, stock_scores, c_score),
            )
        )

    recommendations.sort(key=lambda r: r.scores.final_score, reverse=True)
    return recommendations
