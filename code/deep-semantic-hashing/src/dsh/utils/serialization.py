import json
from typing import Generic, TypeVar


T = TypeVar("T", bound=object)


class Serializer(Generic[T]):
    @staticmethod
    def load(filename: str, cls: type[T]) -> T:
        obj = cls()
        obj.__dict__ = load(filename)
        return obj

    @staticmethod
    def save(data: T, filename: str) -> None:
        save(data.__dict__, filename)


def load(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def save(data: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f)
