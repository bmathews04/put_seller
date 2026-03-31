from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd

from models import StockMetrics
from providers.yfinance_market import YFinanceMarketProvider


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


def make_puts_df(
    strike=95.0,
    bid=1.00,
    ask=1.10,
    last_price=1.05,
    implied_volatility=0.25,
    open_interest=1000,
    volume=100,
    in_the_money=False,
):
    return pd.DataFrame(
        [
            {
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "lastPrice": last_price,
                "impliedVolatility": implied_volatility,
                "openInterest": open_interest,
                "volume": volume,
                "inTheMoney": in_the_money,
            }
        ]
    )


def test_get_option_contracts_reuses_supplied_stock_metrics(monkeypatch):
    provider = YFinanceMarketProvider(use_cache=False)

    supplied_metrics = make_metrics(stock_price=123.45)

    # Fail the test if provider tries to fetch stock metrics again internally
    def fail_get_stock_metrics(symbol):
        raise AssertionError("get_stock_metrics should not be called when stock_metrics is supplied")

    monkeypatch.setattr(provider, "get_stock_metrics", fail_get_stock_metrics)

    class FakeTicker:
        @property
        def options(self):
            return [(date.today() + timedelta(days=30)).isoformat()]

    monkeypatch.setattr(
        "providers.yfinance_market.yf.Ticker",
        lambda symbol: FakeTicker(),
    )

    monkeypatch.setattr(
        "providers.yfinance_market._fetch_option_chain",
        lambda symbol, exp_str: make_puts_df(),
    )

    contracts = provider.get_option_contracts("TEST", stock_metrics=supplied_metrics)

    assert len(contracts) == 1
    assert contracts[0].symbol == "TEST"
    assert contracts[0].delta is not None


def test_get_option_contracts_logs_partial_chain_failures_and_keeps_good_contracts(monkeypatch):
    provider = YFinanceMarketProvider(use_cache=False)

    good_expiry = (date.today() + timedelta(days=30)).isoformat()
    bad_expiry = (date.today() + timedelta(days=37)).isoformat()

    class FakeTicker:
        @property
        def options(self):
            return [good_expiry, bad_expiry]

    monkeypatch.setattr(
        "providers.yfinance_market.yf.Ticker",
        lambda symbol: FakeTicker(),
    )

    monkeypatch.setattr(
        provider,
        "get_stock_metrics",
        lambda symbol: make_metrics(),
    )

    def fake_fetch_option_chain(symbol, exp_str):
        if exp_str == bad_expiry:
            raise RuntimeError("simulated chain fetch failure")
        return make_puts_df()

    monkeypatch.setattr(
        "providers.yfinance_market._fetch_option_chain",
        fake_fetch_option_chain,
    )

    contracts = provider.get_option_contracts("TEST")

    assert len(contracts) == 1
    assert any("simulated chain fetch failure" in err for err in provider.last_errors)
    assert any(bad_expiry in err for err in provider.last_errors)


def test_get_option_contracts_logs_empty_chain(monkeypatch):
    provider = YFinanceMarketProvider(use_cache=False)

    empty_expiry = (date.today() + timedelta(days=30)).isoformat()

    class FakeTicker:
        @property
        def options(self):
            return [empty_expiry]

    monkeypatch.setattr(
        "providers.yfinance_market.yf.Ticker",
        lambda symbol: FakeTicker(),
    )

    monkeypatch.setattr(
        provider,
        "get_stock_metrics",
        lambda symbol: make_metrics(),
    )

    monkeypatch.setattr(
        "providers.yfinance_market._fetch_option_chain",
        lambda symbol, exp_str: pd.DataFrame(),
    )

    contracts = provider.get_option_contracts("TEST")

    assert contracts == []
    assert any("empty_put_chain" in err for err in provider.last_errors)


def test_delta_calc_failure_logs_error_and_returns_contract(monkeypatch):
    provider = YFinanceMarketProvider(use_cache=False)

    expiry = (date.today() + timedelta(days=30)).isoformat()

    class FakeTicker:
        @property
        def options(self):
            return [expiry]

    monkeypatch.setattr(
        "providers.yfinance_market.yf.Ticker",
        lambda symbol: FakeTicker(),
    )

    monkeypatch.setattr(
        provider,
        "get_stock_metrics",
        lambda symbol: make_metrics(stock_price=100.0),
    )

    monkeypatch.setattr(
        "providers.yfinance_market._fetch_option_chain",
        lambda symbol, exp_str: make_puts_df(implied_volatility=0.25),
    )

    def fail_delta_calc(**kwargs):
        raise ValueError("simulated delta failure")

    monkeypatch.setattr(
        "providers.yfinance_market.black_scholes_put_delta",
        fail_delta_calc,
    )

    contracts = provider.get_option_contracts("TEST")

    assert len(contracts) == 1
    assert contracts[0].delta is None
    assert any("delta_calc_failed" in err for err in provider.last_errors)
    assert any("simulated delta failure" in err for err in provider.last_errors)


def test_last_errors_resets_each_call(monkeypatch):
    provider = YFinanceMarketProvider(use_cache=False)

    first_expiry = (date.today() + timedelta(days=30)).isoformat()
    second_expiry = (date.today() + timedelta(days=37)).isoformat()

    class FakeTickerFirst:
        @property
        def options(self):
            return [first_expiry]

    class FakeTickerSecond:
        @property
        def options(self):
            return [second_expiry]

    ticker_calls = {"count": 0}

    def fake_ticker(symbol):
        ticker_calls["count"] += 1
        if ticker_calls["count"] == 1:
            return FakeTickerFirst()
        return FakeTickerSecond()

    monkeypatch.setattr(
        "providers.yfinance_market.yf.Ticker",
        fake_ticker,
    )

    monkeypatch.setattr(
        provider,
        "get_stock_metrics",
        lambda symbol: make_metrics(),
    )

    def fake_fetch_option_chain(symbol, exp_str):
        if exp_str == first_expiry:
            raise RuntimeError("first failure")
        return make_puts_df()

    monkeypatch.setattr(
        "providers.yfinance_market._fetch_option_chain",
        fake_fetch_option_chain,
    )

    provider.get_option_contracts("TEST")
    first_errors = list(provider.last_errors)

    provider.get_option_contracts("TEST")
    second_errors = list(provider.last_errors)

    assert any("first failure" in err for err in first_errors)
    assert not any("first failure" in err for err in second_errors)
