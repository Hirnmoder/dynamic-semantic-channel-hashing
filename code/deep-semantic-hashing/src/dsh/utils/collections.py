from typing import Any, Callable, Generic, Sequence, TypeVar, overload


T = TypeVar("T")


def find_duplicates(lst: list[T]) -> dict[T, list[int]]:
    """Find duplicates in a list and return them as a dictionary with the element as the key and a list containing its indices as the value."""
    result: dict[T, list[int]] = {}
    seen = set()
    for i, item in enumerate(lst):
        if item not in seen:
            seen.add(item)  # first occurrence
        else:
            if item not in result:
                result[item] = [lst.index(item)]  # re-find first occurrence index
            result[item].append(i)
    return result


def map_if_present(key: T, mapping: dict[T, T]) -> T:
    return mapping[key] if key in mapping else key


def first(lst: Sequence[T], predicate: Callable[[T], bool]) -> T:
    for item in lst:
        if predicate(item):
            return item
    raise ValueError("No item found that matches the predicate")


def first_not_none(*lst: T | None) -> T:
    for item in lst:
        if item is not None:
            return item
    raise ValueError("No item found that is not none")


@overload
def first_index_of(lst: Sequence[T], predicate: Callable[[T], bool], default: None = None) -> int | None: ...
@overload
def first_index_of(lst: Sequence[T], predicate: Callable[[T], bool], default: int) -> int: ...
def first_index_of(lst: Sequence[T], predicate: Callable[[T], bool], default: int | None = None) -> int | None:
    for idx, item in enumerate(lst):
        if predicate(item):
            return idx
    return default


N = TypeVar("N", int, float)


class NonFiniteRange(Generic[N]):
    def __init__(self, start: N | None, end: N | None):
        self.start = start
        self.end = end

    def __contains__(self, other: Any) -> bool:
        if not isinstance(other, (int, float)):
            return False
        return (self.start is None or other >= self.start) and (self.end is None or other < self.end)


# fmt: off
class NonFiniteIntRange(NonFiniteRange[int]): ...
class NonFiniteFloatRange(NonFiniteRange[float]): ...
# fmt: on
