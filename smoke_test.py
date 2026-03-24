from config import ScanConfig
from providers.yfinance_market import YFinanceMarketProvider
from recommendation_engine import build_recommendations_for_stock

cfg = ScanConfig()
provider = YFinanceMarketProvider()

symbol = "AAPL"

metrics = provider.get_stock_metrics(symbol)
contracts = provider.get_option_contracts(symbol)
recs = build_recommendations_for_stock(metrics, contracts, cfg)

print(f"Symbol: {symbol}")
print(f"Stock price: {metrics.stock_price}")
print(f"Contracts pulled: {len(contracts)}")
print(f"Recommendations: {len(recs)}")

for rec in recs[:5]:
    c = rec.selected_contract
    print("-" * 60)
    print("Final score:", round(rec.scores.final_score, 2))
    print("Expiration:", c.expiration_date)
    print("DTE:", c.dte)
    print("Strike:", c.strike)
    print("Delta:", c.delta)
    print("Premium:", c.premium)
    print("Break-even:", c.breakeven_price)
    print("Annualized yield:", c.annualized_secured_yield)
    print("Suggested entry:", rec.suggested_entry_limit)
    print("Reasons:", rec.top_reasons)
    print("Risks:", rec.top_risks)
