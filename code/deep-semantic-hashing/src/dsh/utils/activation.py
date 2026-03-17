from dataclasses import dataclass
import torch.nn

from dsh.config.model import ActivationType
from dsh.utils.initialization import InitializationType


def get_activation(activation: ActivationType) -> torch.nn.Module:
    return get_activation_and_init_params(activation).activation


@dataclass
class ActivationAndInitParams:
    activation: torch.nn.Module
    name: str
    nonlinearity: torch.nn.init._NonlinearityType
    init_type: InitializationType


def get_activation_and_init_params(activation: ActivationType) -> ActivationAndInitParams:
    match activation:
        case ActivationType.RELU:
            return ActivationAndInitParams(torch.nn.ReLU(), "ReLU", "relu", InitializationType.HE_NORMAL)
        case ActivationType.SIGMOID:
            return ActivationAndInitParams(torch.nn.Sigmoid(), "Sigmoid", "sigmoid", InitializationType.XAVIER_NORMAL)
        case ActivationType.TANH:
            return ActivationAndInitParams(torch.nn.Tanh(), "TanH", "tanh", InitializationType.XAVIER_NORMAL)
        case ActivationType.IDENTITY:
            return ActivationAndInitParams(torch.nn.Identity(), "Identity", "linear", InitializationType.HE_NORMAL)
        case _:
            raise NotImplementedError(f"Unknown activation function {activation}")
