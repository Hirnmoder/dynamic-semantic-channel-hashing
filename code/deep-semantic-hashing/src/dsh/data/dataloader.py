from typing import Generic, cast
import torch.utils.data
import torch.utils.data.dataloader

from dsh.data.dataset import D


class _DataLoaderIter(Generic[D], torch.utils.data.dataloader._BaseDataLoaderIter):
    def __init__(self, loader: torch.utils.data.DataLoader[D]):
        raise NotImplementedError("This method must not be called and is only here to satisfy the type checker.")

    def __next__(self) -> D:
        raise NotImplementedError("This method will not be called and is only here to satisfy the type checker.")


class DataLoader(Generic[D], torch.utils.data.DataLoader[D]):
    def __iter__(self) -> _DataLoaderIter[D]:
        return cast(_DataLoaderIter[D], super().__iter__())
