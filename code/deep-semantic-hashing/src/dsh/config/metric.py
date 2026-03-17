from abc import ABC
from enum import Enum
from typing import Optional, TypeVar

from dsh.config.configuration import ConfigurationBase, Field, register
from dsh.utils.types import MetricEvalSet

# Specific Metric Configuration
SMC = TypeVar("SMC", bound="MetricConfiguration", covariant=True)


@register("Metrics")
class MetricsConfiguration(ConfigurationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metrics = Field[list[MetricConfiguration]](lambda: [HammingMetricConfiguration()])()


class MetricConfiguration(ConfigurationBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.subselect_top_k_labels = Field[Optional[int]](None)()
        self.include_samples_without_labels = Field[bool](False)()
        self.eval_set = Field[MetricEvalSet](MetricEvalSet.VAL)()


class AveragePrecisionMode(Enum):
    LIKE_TDSRDH_PAPER = "tdsrdh"
    DEFAULT = "default"
    TIE_AWARE = "tie-aware"


@register("HammingMetric")
class HammingMetricConfiguration(MetricConfiguration):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.average_precision_mode = Field[AveragePrecisionMode](AveragePrecisionMode.TIE_AWARE)()
