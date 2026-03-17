from typing import Callable, Generic, TypeVar

N = TypeVar("N", float, int)


class CriterionComparators:
    @staticmethod
    def lower_is_better(a: N, b: N) -> bool:
        return a < b

    @staticmethod
    def higher_is_better(a: N, b: N) -> bool:
        return a > b


class CriterionCollector(Generic[N]):
    def __init__(self, comparator: Callable[[N, N], bool]):
        self._values: dict[int, N] = {}
        self._comparator = comparator

    def add_value(self, epoch: int, value: N) -> None:
        self._values[epoch] = value

    def get_value(self, epoch: int) -> N:
        assert epoch in self._values, f"Epoch {epoch} not found in criterion collector"
        return self._values[epoch]

    def get_best_epoch(self) -> int:
        assert len(self._values) > 0, "No values have been added to the criterion collector"
        best_epoch = next(iter(self._values.keys()))
        best_value = self._values[best_epoch]
        for epoch, value in self._values.items():
            if self._comparator(value, best_value):
                best_epoch = epoch
                best_value = value
        return best_epoch

    def get_max_epoch(self) -> int:
        return max(self._values.keys())

    def should_stop(self, patience_epochs: int) -> bool:
        return self.get_max_epoch() > self.get_best_epoch() + patience_epochs
