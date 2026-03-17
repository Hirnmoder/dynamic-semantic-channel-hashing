from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from dsh.config.env import MetricEnvironmentConfig
from dsh.config.metric import SMC
from dsh.utils.tensorboard import Writer
from dsh.utils.types import Constants, MetricEvalSet

MI = TypeVar("MI")  # Metric Input type
MO = TypeVar("MO", covariant=True)  # Metric Output type


class MetricBase(ABC, Generic[SMC, MI, MO]):
    """Abstract base class for all metrics."""

    def __init__(self, config: SMC, env: MetricEnvironmentConfig):
        self.config = config
        self.env = env

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def set(self) -> MetricEvalSet:
        return self.config.eval_set

    @property
    def namewithset(self) -> str:
        return Constants.Metrics.namewithset(self.name, self.set)

    @abstractmethod
    def calculate(self, data: MI, writer: Writer) -> MO:
        """Calculate the metric."""
        raise NotImplementedError("Subclasses must implement this method.")
