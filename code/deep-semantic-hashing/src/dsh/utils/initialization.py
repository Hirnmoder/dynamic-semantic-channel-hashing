from enum import Enum
from functools import partial
import torch
from typing import Any, Callable, Optional


class InitializationType(Enum):
    HE_UNIFORM = "he-uniform"
    HE_NORMAL = "he-normal"
    XAVIER_UNIFORM = "xavier-uniform"
    XAVIER_NORMAL = "xavier-normal"


def init_weights(model: torch.nn.Module, initialization_type: InitializationType, **kwargs: Any) -> None:
    init_functions: dict[InitializationType, Callable[[torch.nn.Module], None]] = {
        InitializationType.HE_UNIFORM: init_weights_he_uniform,
        InitializationType.HE_NORMAL: init_weights_he_normal,
        InitializationType.XAVIER_UNIFORM: init_weights_xavier_uniform,
        InitializationType.XAVIER_NORMAL: init_weights_xavier_normal,
    }
    init_functions[initialization_type](model, **kwargs)


TensorInitFunc = Callable[[torch.Tensor], Optional[torch.Tensor]]


def init_weights_by_func(model: torch.nn.Module, weight_init_func: TensorInitFunc, bias_init_func: TensorInitFunc) -> None:
    for layer in model.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            weight_init_func(layer.weight)
            if layer.bias is not None:
                bias_init_func(layer.bias)


def init_weights_he_uniform(
    model: torch.nn.Module,
    mode: torch.nn.init._FanMode = "fan_in",
    nonlinearity: torch.nn.init._NonlinearityType = "relu",
) -> None:
    init_weights_by_func(
        model,
        partial(torch.nn.init.kaiming_uniform_, mode=mode, nonlinearity=nonlinearity),
        torch.nn.init.zeros_,
    )


def init_weights_he_normal(
    model: torch.nn.Module,
    mode: torch.nn.init._FanMode = "fan_in",
    nonlinearity: torch.nn.init._NonlinearityType = "relu",
) -> None:
    init_weights_by_func(
        model,
        partial(torch.nn.init.kaiming_normal_, mode=mode, nonlinearity=nonlinearity),
        torch.nn.init.zeros_,
    )


def init_weights_xavier_uniform(
    model: torch.nn.Module,
    nonlinearity: torch.nn.init._NonlinearityType = "relu",
) -> None:
    init_weights_by_func(
        model,
        partial(torch.nn.init.xavier_uniform_, gain=torch.nn.init.calculate_gain(nonlinearity)),
        torch.nn.init.zeros_,
    )


def init_weights_xavier_normal(
    model: torch.nn.Module,
    nonlinearity: torch.nn.init._NonlinearityType = "relu",
) -> None:
    init_weights_by_func(
        model,
        partial(torch.nn.init.xavier_normal_, gain=torch.nn.init.calculate_gain(nonlinearity)),
        torch.nn.init.zeros_,
    )
