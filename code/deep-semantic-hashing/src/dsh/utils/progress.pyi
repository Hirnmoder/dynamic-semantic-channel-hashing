from typing import Any, Iterable, TypeVar
from tqdm.std import tqdm as _tqdm

_T = TypeVar("_T")

def tqdm(iterable: Iterable[_T], /, *args: Any, **kwargs: Any) -> _tqdm[_T]: ...
