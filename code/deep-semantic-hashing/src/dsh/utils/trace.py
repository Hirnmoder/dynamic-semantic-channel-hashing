from functools import partial
import pandas as pd
import PIL.Image
import torch
from typing import Any, Callable, Literal, Self


class VectorStatisticsTracker:
    def __init__(
        self, module: torch.nn.Module, get_current_epoch: Callable[[], int], get_current_batch: Callable[[], int]
    ) -> None:
        # list of tuple (epoch, batch, name, input_shape, output_shape, input_min, input_max, input_mean, input_var, output_min, output_max, output_mean, output_var)
        self._statistics = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.module = module
        self._get_current_epoch = get_current_epoch
        self._get_current_batch = get_current_batch

    def __enter__(self) -> Self:
        for name, layer in self.module.named_modules():
            self._hooks.append(layer.register_forward_hook(partial(self._trace_vector_statistics_hook, name)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        for hook in self._hooks:
            hook.remove()
        return False  # Don't suppress exceptions

    def _trace_vector_statistics_hook(self, name: str, _: torch.nn.Module, input: tuple[Any, ...], output: Any) -> None:
        assert isinstance(input, tuple), "Input must be a tuple"
        if isinstance(output, tuple):
            output = output[0]
        assert isinstance(output, torch.Tensor), "Output must be a tensor"
        i, o = input[0], output
        if isinstance(i, tuple):
            i = next(itm for itm in i if isinstance(itm, torch.Tensor))
        if isinstance(i, PIL.Image.Image):
            return  # Skip image inputs
        assert isinstance(i, torch.Tensor), "Input must be a tensor"
        self._statistics.append(
            (
                *[self._get_current_epoch(), self._get_current_batch(), name],
                *[i.shape, o.shape],
                *[i.min().item(), i.max().item(), i.mean().item(), i.var().item()],
                *[o.min().item(), o.max().item(), o.mean().item(), o.var().item()],
            )
        )

    def save(self, filename: str) -> None:
        df = pd.DataFrame(
            data=self._statistics,
            columns=[
                *["epoch", "batch", "name"],
                *["input_shape", "output_shape"],
                *["min_input", "max_input", "mean_input", "var_input"],
                *["min_output", "max_output", "mean_output", "var_output"],
            ],
        )
        df.to_csv(filename, index=False)
