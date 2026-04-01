from datetime import date, timedelta

from config import ScanConfig
from models import OptionContract, StockMetrics
from recommendation_engine import build_recommendations_for_stock


def make_metrics(**overrides) -> StockMetrics:
    metrics = StockMetrics(
        symbol="TEST",
        stock_price=100.0,
        earnings_date=date.today() + timedelta(days=45),
        days_to_earnings=45,
        fcf_yield=0.08,
        quality_data_source="test",
        quality_data_complete=True,
        candidate_contract_count=0,
    )
    for key, value in overrides.items():
        setattr(metrics, key, value)
    return metrics


def make_contract(**overrides) -> OptionContract:
    contract = OptionContract(
        symbol="TEST",
        expiration_date=date.today() + timedelta(days=32),
        dte=32,
        strike=95.0,
        option_type="PUT",
        bid=1.00,
        ask=1.10,
        last=1.05,
        mark=1.05,
        delta=-0.17,
        implied_volatility=0.25,
        open_interest=1200,
        volume=150,
        in_the_money=False,
    )
    for key, value in overrides.items():
        setattr(contract, key, value)
    return contract


def base_cfg(**overrides) -> ScanConfig:
    cfg = ScanConfig(
        min_dte=25,
        max_dte=40,
        target_dte=32,
        min_abs_delta=0.12,
        max_abs_delta=0.22,
        target_abs_delta=0.17,
        min_open_interest=500,
        min_volume=25,
        max_spread_pct=0.12,
        min_premium=0.35,
        min_bid=0.25,
        exclude_earnings_before_expiry=True,
        require_quality_data=False,
        strict_data_mode=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_recommendation_stores_decision_fields_for_clean_candidate():
    metrics = make_metrics()
    contracts = [make_contract()]
    cfg = base_cfg()

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert len(recs) == 1
    rec = recs[0]

    assert rec.stock_eligibility_status == "eligible"
    assert rec.contract_eligibility_status in {"eligible", "eligible_with_warnings"}
    assert isinstance(rec.stock_warning_reasons, list)
    assert isinstance(rec.contract_warning_reasons, list)

    assert rec.decision_status in {"Ready", "Review", "Pass"}
    assert rec.decision_rationale is not None
    assert isinstance(rec.decision_blockers, list)
    assert isinstance(rec.decision_cautions, list)

    assert rec.warning_severity_label in {"None", "Low", "Medium", "High"}
    assert isinstance(rec.warning_severity_points, int)
    assert isinstance(rec.warning_count_total, int)
    assert isinstance(rec.high_severity_warning_count, int)
    assert isinstance(rec.medium_severity_warning_count, int)
    assert isinstance(rec.low_severity_warning_count, int)


def test_recommendation_carries_stock_warning_reasons_when_quality_data_incomplete():
    metrics = make_metrics(
        fcf_yield=None,
        quality_data_complete=False,
    )
    contracts = [make_contract()]
    cfg = base_cfg(require_quality_data=False)

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert len(recs) == 1
    rec = recs[0]

    assert rec.stock_eligibility_status == "eligible_with_warnings"
    assert "quality_data_incomplete" in rec.stock_warning_reasons
    assert rec.warning_count_total >= 1
    assert rec.decision_status in {"Review", "Pass"}


def test_recommendation_carries_contract_warning_reasons_in_non_strict_mode():
    metrics = make_metrics()
    contracts = [
        make_contract(
            open_interest=100,   # below min_open_interest
            volume=10,           # below min_volume
        )
    ]
    cfg = base_cfg(strict_data_mode=False)

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert len(recs) == 1
    rec = recs[0]

    assert rec.contract_eligibility_status == "eligible_with_warnings"
    assert "oi_below_min" in rec.contract_warning_reasons
    assert "volume_below_min" in rec.contract_warning_reasons
    assert rec.warning_count_total >= 2
    assert rec.warning_severity_label in {"Medium", "High"}


def test_no_recommendation_when_contract_becomes_ineligible():
    metrics = make_metrics()
    contracts = [
        make_contract(
            bid=0.10,   # below min_bid
            ask=0.20,
            mark=0.15,
        )
    ]
    cfg = base_cfg()

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert recs == []


def test_no_recommendation_when_stock_is_ineligible():
    metrics = make_metrics(stock_price=3.0)
    contracts = [make_contract()]
    cfg = base_cfg()
    cfg.min_stock_price = 5.0

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert recs == []


def test_recommendation_warning_counts_match_clean_case():
    metrics = make_metrics()
    contracts = [make_contract()]
    cfg = base_cfg()

    recs = build_recommendations_for_stock(metrics, contracts, cfg)
    rec = recs[0]

    total_from_parts = (
        rec.high_severity_warning_count
        + rec.medium_severity_warning_count
        + rec.low_severity_warning_count
    )

    assert rec.warning_count_total == total_from_parts


def test_recommendation_warning_counts_match_warned_case():
    metrics = make_metrics(
        fcf_yield=None,
        quality_data_complete=False,
    )
    contracts = [
        make_contract(
            open_interest=100,
            volume=10,
        )
    ]
    cfg = base_cfg(strict_data_mode=False)

    recs = build_recommendations_for_stock(metrics, contracts, cfg)
    rec = recs[0]

    total_from_parts = (
        rec.high_severity_warning_count
        + rec.medium_severity_warning_count
        + rec.low_severity_warning_count
    )

    assert rec.warning_count_total == total_from_parts
    assert rec.warning_count_total >= 3


def test_recommendations_sorted_by_final_score_descending_within_stock():
    metrics = make_metrics()
    contracts = [
        make_contract(strike=95.0, bid=1.00, ask=1.10, mark=1.05, delta=-0.17),
        make_contract(strike=92.0, bid=0.55, ask=0.65, mark=0.60, delta=-0.13),
    ]
    cfg = base_cfg()

    recs = build_recommendations_for_stock(metrics, contracts, cfg)

    assert len(recs) == 2
    assert recs[0].scores.final_score >= recs[1].scores.final_score
