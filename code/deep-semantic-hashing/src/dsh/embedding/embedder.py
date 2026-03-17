from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy.typing as npt

from dsh.utils.types import TextRawInput


class Embedder(ABC):
    @abstractmethod
    def __call__(self, text: str | list[str], /) -> TextRawInput: ...


@dataclass
class EmbedderInfo:
    name: str
    sequence_length: int
    embedding_dim: int
    dtype: npt.DTypeLike
    get_embedder: Callable[[], Embedder]
