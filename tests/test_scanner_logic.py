from datetime import date, timedelta

from config import ScanConfig
from derivations import derive_contract_metrics
from models import OptionContract, StockMetrics
from recommendation_engine import build_management_plan
from stock_scoring import score_quality
from validators import validate_contract


def make_metrics(**overrides) -> StockMetrics:
    metrics = StockMetrics(
        symbol="TEST",
        stock_price=100.0,
        earnings_date=date.today() + timedelta(days=30),
        days_to_earnings=30,
        fcf_yield=0.08,
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
        open_interest=1000,
        volume=100,
        in_the_money=False,
    )
    for key, value in overrides.items():
        setattr(contract, key, value)
    return contract


def test_excludes_contract_when_earnings_before_expiry():
    cfg = ScanConfig(exclude_earnings_before_expiry=True)
    metrics = make_metrics(days_to_earnings=10, earnings_date=date.today() + timedelta(days=10))
    raw_contract = make_contract(dte=20, expiration_date=date.today() + timedelta(days=20))

    contract = derive_contract_metrics(raw_contract, metrics.stock_price)
    contract = validate_contract(contract, cfg, metrics)

    assert contract.contract_valid is False
    assert "earnings_before_expiry" in contract.contract_exclusion_reasons


def test_excludes_contract_when_earnings_date_unknown_and_exclusion_enabled():
    cfg = ScanConfig(exclude_earnings_before_expiry=True)
    metrics = make_metrics(days_to_earnings=None, earnings_date=None)
    raw_contract = make_contract()

    contract = derive_contract_metrics(raw_contract, metrics.stock_price)
    contract = validate_contract(contract, cfg, metrics)

    assert contract.contract_valid is False
    assert "earnings_date_unknown" in contract.contract_exclusion_reasons


def test_excludes_contract_when_bid_below_min():
    cfg = ScanConfig(min_bid=0.35)
    metrics = make_metrics()
    raw_contract = make_contract(bid=0.20, ask=0.40)

    contract = derive_contract_metrics(raw_contract, metrics.stock_price)
    contract = validate_contract(contract, cfg, metrics)

    assert contract.contract_valid is False
    assert "bid_below_min" in contract.contract_exclusion_reasons


def test_excludes_contract_when_premium_too_low():
    cfg = ScanConfig(min_premium=0.35)
    metrics = make_metrics()
    raw_contract = make_contract(bid=0.10, ask=0.20)

    contract = derive_contract_metrics(raw_contract, metrics.stock_price)
    contract = validate_contract(contract, cfg, metrics)

    assert contract.contract_valid is False
    assert "premium_too_low" in contract.contract_exclusion_reasons


def test_excludes_contract_when_spread_too_wide():
    cfg = ScanConfig(max_spread_pct=0.12)
    metrics = make_metrics()
    raw_contract = make_contract(bid=0.40, ask=0.70)

    contract = derive_contract_metrics(raw_contract, metrics.stock_price)
    contract = validate_contract(contract, cfg, metrics)

    assert contract.contract_valid is False
    assert "spread_too_wide" in contract.contract_exclusion_reasons


def test_quality_fallback_mode_on_uses_default_score():
    cfg = ScanConfig(
        quality_fallback_mode=True,
        default_quality_score_if_missing=55.0,
    )
    metrics = make_metrics(fcf_yield=None)

    score = score_quality(metrics, cfg)

    assert score == 55.0


def test_quality_fallback_mode_off_uses_zero_score():
    cfg = ScanConfig(
        quality_fallback_mode=False,
        default_quality_score_if_missing=55.0,
    )
    metrics = make_metrics(fcf_yield=None)

    score = score_quality(metrics, cfg)

    assert score == 0.0


def test_allow_expiry_week_hold_false_updates_management_plan():
    cfg = ScanConfig(allow_expiry_week_hold=False, review_dte=3)
    contract = derive_contract_metrics(make_contract(), 100.0)

    _, _, review_at_dte, _, _, management_text = build_management_plan(
        stock_price=100.0,
        contract=contract,
        cfg=cfg,
    )

    assert review_at_dte == 5
    assert "Avoid holding into expiration week" in management_text


def test_allow_expiry_week_hold_true_keeps_configured_review_dte():
    cfg = ScanConfig(allow_expiry_week_hold=True, review_dte=3)
    contract = derive_contract_metrics(make_contract(), 100.0)

    _, _, review_at_dte, _, _, management_text = build_management_plan(
        stock_price=100.0,
        contract=contract,
        cfg=cfg,
    )

    assert review_at_dte == 3
    assert "Holding into expiration week can be acceptable" in management_text
