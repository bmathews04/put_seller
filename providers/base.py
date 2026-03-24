from abc import ABC, abstractmethod
from models import OptionContract, StockMetadata, StockMetrics


class UniverseProvider(ABC):
    @abstractmethod
    def get_universe(self) -> list[StockMetadata]:
        raise NotImplementedError


class MarketDataProvider(ABC):
    @abstractmethod
    def get_stock_metrics(self, symbol: str) -> StockMetrics:
        raise NotImplementedError

    @abstractmethod
    def get_option_contracts(self, symbol: str) -> list[OptionContract]:
        raise NotImplementedError
