from abc import ABC, abstractmethod
import torch
from typing import Generic, TypeVar, cast

from dsh.config.env import ModelDataEnvironmentConfig
from dsh.config.model import HMC, MC
from dsh.utils.adapter import CrossModalDatasetToModelAdapter, DatasetToModelAdapter
from dsh.utils.types import FullModelInput, Output

MI = TypeVar("MI")  # Type of model input data.
MO = TypeVar("MO")  # Type of model output data.


class ModelBase(ABC, Generic[MC, MI, MO], torch.nn.Module):
    """Abstract base class for all models."""

    def __init__(self, config: MC, env: ModelDataEnvironmentConfig):
        """Initialize the model with given configuration.
        Args:
            config (C): Configuration of the model.
        """
        super().__init__()
        self.config = config
        self.env = env

    @abstractmethod
    def forward(self, input: MI) -> MO:
        """Forward pass through the network."""
        raise NotImplementedError()

    @abstractmethod
    def apply_freezing(self):
        """Apply freezing to the model parameters that should not be updated during training."""
        raise NotImplementedError()

    def __call__(self, i: MI) -> MO:
        return cast(MO, super().__call__(i))


class TopLevelModelBase(Generic[MC, MI, MO], ModelBase[MC, MI, MO]):
    """Abstract class for top-level models, i.e. bigger models consisting of smaller parts."""

    def __init__(self, config: MC, env: ModelDataEnvironmentConfig):
        super().__init__(config, env)

        self.env.add_model_resolver(lambda: self.model_name)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        raise NotImplementedError()

    @abstractmethod
    def get_adapter(self) -> DatasetToModelAdapter:
        """Get the adapter to make data preprocessing compatible with this model."""
        raise NotImplementedError()


class CrossModalTopLevelModelBase(Generic[HMC], TopLevelModelBase[HMC, FullModelInput, Output]):
    @abstractmethod
    def get_adapter(self) -> CrossModalDatasetToModelAdapter:
        """Get the adapter to make data preprocessing compatible with this model."""
        raise NotImplementedError()
