from typing import Generic, TypeVar

from source.applications.backtests import PORTFOLIO_INFO

REDISTRIBUTE_INPUT = TypeVar("REDISTRIBUTE_INPUT")


class Redistribution(Generic[REDISTRIBUTE_INPUT]):
    def get_delta(self, source_info: REDISTRIBUTE_INPUT) -> PORTFOLIO_INFO:
        raise NotImplementedError()
