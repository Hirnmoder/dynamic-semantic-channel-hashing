from typing import Literal, Self
import torch

from dsh.utils.logger import Logger


class CleanCudaMemory:
    def __init__(self, device: torch.device):
        self.device = device

    def _emtpy_cache(self, operation: str = ""):
        before = torch.cuda.memory_allocated(self.device)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(self.device)
        Logger().info(f"[MEM] cuda.emtpy_cache freed {(before - after)/1024.0/1024.0:.2f} MiB on {self.device} ({operation})")

    def __enter__(self) -> Self:
        self._emtpy_cache("ENTER")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self._emtpy_cache("EXIT")
        return False  # Don't suppress exceptions
