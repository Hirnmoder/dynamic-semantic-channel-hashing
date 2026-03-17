from abc import ABC, abstractmethod
from enum import Enum
import torch
from typing import Any, Iterator, Literal, TypeGuard

from dsh.config.configuration import ConfigurationBase, ConfigurationCodecSettings, Field, register
from dsh.utils.criterioncollector import CriterionCollector, CriterionComparators
from dsh.utils.types import Constants, MetricEvalSet
from dsh.utils.unparsing import stringify


class TrainMode(Enum):
    IMG_TEXT_PER_EPOCH = "img-text-per-epoch"  # like TDSRDH paper
    IMG_TEXT_PER_BATCH = "img-text-per-batch"  # like TDSRDH paper, but switch back and forth more frequently
    SIMULTANEOUS = "simultaneous"  # like SCH paper

    @staticmethod
    def is_alternating(
        v: "TrainMode",
    ) -> TypeGuard["Literal[TrainMode.IMG_TEXT_PER_EPOCH]|Literal[TrainMode.IMG_TEXT_PER_BATCH]"]:
        return v == TrainMode.IMG_TEXT_PER_BATCH or v == TrainMode.IMG_TEXT_PER_EPOCH

    @staticmethod
    def is_simultaneous(v: "TrainMode") -> TypeGuard["Literal[TrainMode.SIMULTANEOUS]"]:
        return v == TrainMode.SIMULTANEOUS


@register("Train")
class TrainConfig(ConfigurationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epochs = Field[int](200)()  # TDSRDH paper: 200, but convergence should occur after 10 to 20
        self.batch_size = Field[int](128)()
        self.drop_last = Field[bool](True)()
        self.optimizer = Field[OptimizerConfig](AdamOptimizer)()
        self.train_mode = Field[TrainMode](TrainMode.IMG_TEXT_PER_EPOCH)()
        self.loss = Field[LossConfig](TDSRDHPaperLoss)()
        self.scheduler = Field[SchedulerConfig](ConstantScheduler)()
        self.save_frequency = Field[int](5)()
        self.early_stopping = Field[EarlyStoppingConfig | None](None)()


class OptimizerConfig(ConfigurationBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.learning_rate = Field[float](1e-5)()  # in TDSRDH paper they optimize lr ranging from 1e-5 to 1e-1

    @abstractmethod
    def construct(self, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer: ...

    @property
    def name(self):
        return ConfigurationCodecSettings.typename(type(self))


@register("Adam")
class AdamOptimizer(OptimizerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight_decay = Field[float](0.0)()
        self.eps = Field[float](1e-8)()
        self.betas = Field[tuple[float, float]]((0.9, 0.999))()

    def construct(self, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=params,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


@register("SGD")
class SGDOptimizer(OptimizerConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.momentum = Field[float](0.0)()
        self.dampening = Field[float](0.0)()
        self.weight_decay = Field[float](0.0)()

    def construct(self, params: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params=params,
            lr=self.learning_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
        )


class LossConfig(ConfigurationBase, ABC):
    @abstractmethod
    def stringify(self) -> str: ...


class PairwiseLossType(Enum):
    AS_PAPER = "as-paper"


class QuantizationLossType(Enum):
    CROSS_MODALITY = "cross-modality"
    SQUARED_FROBENIUS = "squared-frobenius"
    AVERAGE_HASHCODES = "average-hashcodes"
    AVERAGE_ACTIVATION_TO_HASHCODE = "average-activation-to-hashcode"
    ALWAYS_IMAGE = "always-image"
    ALWAYS_TEXT = "always-text"


class TripletLossType(Enum):
    AS_PAPER = "as-paper"


@register("TDSRDH-Loss")
class TDSRDHPaperLoss(LossConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Hyperparameters for weighting the three losses
        self.alpha = Field[float](0.05)()
        self.beta = Field[float](1.0)()
        self.epsilon = Field[float](0.3)()  # margin for triplet loss
        self.normalize_by_batch_size = Field[bool](True)()  # formulas in paper suggest this is false, but training may fail

        self.pairwise_loss = Field[PairwiseLossType](PairwiseLossType.AS_PAPER)()
        self.quantization_loss = Field[QuantizationLossType](QuantizationLossType.AVERAGE_ACTIVATION_TO_HASHCODE)()
        self.triplet_loss = Field[TripletLossType](TripletLossType.AS_PAPER)()

    def stringify(self) -> str:
        return stringify(self.__class__.__name__, **self.__dict__)


@register("SCH-Loss")
class SCHPaperLoss(LossConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Hyperparameters for Semantic Channel Hashing
        self.tau = Field[int](3)()  # target semantic channel width
        self.alpha = Field[float](1.0)()  # weight for L_fpos
        self.beta = Field[float](1.0)()  # weight for L_neg
        self.lneg_lambda_l = Field[float](0.5)()  # lambda^l as a fraction of hash_length
        self.normalize_by_batch_size = Field[bool](True)()
        self.use_k_approximation = Field[bool](False)()  # reference code approximates length of binary vector by sqrt(k)

    def stringify(self) -> str:
        return stringify(self.__class__.__name__, **self.__dict__)


@register("DSCH-Loss")
class DSCHLoss(LossConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # Hyperparameters for Dynamic Semantic Channel Hashing
        self.tau = Field[float](3.0)()  # minimum target semantic channel width
        self.alpha = Field[float](1.0)()  # weight for L_fpos
        self.beta = Field[float](1.0)()  # weight for L_neg
        self.gamma_w = Field[float](2.0)()  # channel width curve modifier
        self.gamma_l = Field[float](1.0)()  # loss curve modifier
        self.lambda_neg = Field[float](0.5)()  # lambda_neg as a fraction of hash_length
        self.anchor = Field[float](0.0)()  # anchor point for channel

    def stringify(self) -> str:
        return stringify(self.__class__.__name__, **self.__dict__)


@register("Our-Loss")
class OurLoss(LossConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.semanticchannel = Field[SCHPaperLoss | DSCHLoss](DSCHLoss)()
        self.q_weight = Field[float | None](0.01)()  # weight for quantization loss
        self.use_frobenius = Field[bool](False)()  # use frobenius norm instead of L1,1 norm
        self.normalize_by_batch_size = Field[bool](True)()  # normalize by batch size (overrides semanticchannel property)

    def stringify(self) -> str:
        return stringify(self.__class__.__name__, **self.__dict__, sc=self.semanticchannel.stringify())


class SchedulerConfig(ConfigurationBase, ABC):
    pass


@register("NoScheduler")
@register("ConstantScheduler")
class ConstantScheduler(SchedulerConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.factor = Field[float](1.0)()


@register("CosineScheduler")
class CosineScheduler(SchedulerConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.start_factor = Field[float](1.0)()
        self.end_factor = Field[float](0.1)()


@register("SequenceScheduler")
class SequenceScheduler(SchedulerConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.schedulers = Field[list[tuple[int, SchedulerConfig]]]([])()

    @staticmethod
    def preset_200_cos() -> "SequenceScheduler":
        """Constant, Cosine, Constant (200 epochs total)"""
        s = SequenceScheduler()
        c1 = ConstantScheduler()
        cos = CosineScheduler()
        c2 = ConstantScheduler()
        cos.start_factor = c1.factor = 1.0
        cos.end_factor = c2.factor = 0.1
        s.schedulers = [(0, c1), (75, cos), (150, c2)]
        return s


@register("EarlyStopping")
class EarlyStoppingConfig(ConfigurationBase):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.infer_frequency = Field[int](5)()
        self.patience_epochs = Field[int](10)()
        self.stopping_criterion = Field[StoppingCriterionConfig](ValidationLossStoppingCriterionConfig)()

    @staticmethod
    def preset_all_hamming_metrics() -> "EarlyStoppingConfig":
        sc = AllStoppingCriterionConfig()
        sc.criterions.append(MetricStoppingCriterionConfig.preset_hamming_ranking_map_i2t())
        sc.criterions.append(MetricStoppingCriterionConfig.preset_hamming_ranking_map_t2i())
        sc.criterions.append(MetricStoppingCriterionConfig.preset_hash_lookup_rocauc_i2t())
        sc.criterions.append(MetricStoppingCriterionConfig.preset_hash_lookup_rocauc_t2i())

        c = EarlyStoppingConfig()
        c.infer_frequency = 5
        c.patience_epochs = 10
        c.stopping_criterion = sc
        return c


class StoppingCriterionConfig(ConfigurationBase, ABC):
    @abstractmethod
    def update(self, metricwithset: str, epoch: int, result: Any) -> None: ...

    @abstractmethod
    def should_stop(self, patience_epochs: int) -> bool: ...

    @abstractmethod
    def get_best_epochs(self) -> dict[str, int]: ...


class StoppingCriterionComparators(Enum):
    LOWER_IS_BETTER = "-"
    HIGHER_IS_BETTER = "+"


@register("MetricCriterion")
class MetricStoppingCriterionConfig(StoppingCriterionConfig):
    def __init__(
        self,
        type: str = Constants.Metrics.HammingMetric,
        set: MetricEvalSet = MetricEvalSet.VAL,
        metric: str = Constants.Metrics.Hamming.MAP_I2T,
        better: StoppingCriterionComparators = StoppingCriterionComparators.HIGHER_IS_BETTER,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.type = Field[str](type)()
        self.set = Field[MetricEvalSet](MetricEvalSet.VAL)()
        self.metric = Field[str](metric)()
        self.better = Field[StoppingCriterionComparators](better)()
        self._criterion_collector: CriterionCollector[float] | None = None

    def update(self, metricwithset: str, epoch: int, result: Any) -> None:
        if self._criterion_collector == None:
            match self.better:
                case StoppingCriterionComparators.LOWER_IS_BETTER:
                    self._criterion_collector = CriterionCollector[float](CriterionComparators.lower_is_better)
                case StoppingCriterionComparators.HIGHER_IS_BETTER:
                    self._criterion_collector = CriterionCollector[float](CriterionComparators.higher_is_better)
                case _:
                    raise ValueError(f"Unknown comparator type {self.better}")
        if metricwithset == Constants.Metrics.namewithset(self.type, self.set):
            field_hierarchy = self.metric.split(".")
            current_result = result
            for field in field_hierarchy:
                current_result = getattr(current_result, field)
            assert isinstance(current_result, float), f"Expected a float value but got {type(current_result)}"
            self._criterion_collector.add_value(epoch, current_result)

    def should_stop(self, patience_epochs: int) -> bool:
        if self._criterion_collector == None:
            return False
        return self._criterion_collector.should_stop(patience_epochs)

    def get_best_epochs(self) -> dict[str, int]:
        if self._criterion_collector == None:
            return {}
        return {f"{Constants.Metrics.namewithset(self.type, self.set)}/{self.metric}": self._criterion_collector.get_best_epoch()}

    @staticmethod
    def preset_hamming_ranking_map_i2t() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.MAP_I2T,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hamming_ranking_map_i2i() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.MAP_I2I,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hamming_ranking_map_t2i() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.MAP_T2I,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hamming_ranking_map_t2t() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.MAP_T2T,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hash_lookup_rocauc_i2t() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.ROCAUC_I2T,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hash_lookup_rocauc_i2i() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.ROCAUC_I2I,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hash_lookup_rocauc_t2i() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.ROCAUC_T2I,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )

    @staticmethod
    def preset_hash_lookup_rocauc_t2t() -> "MetricStoppingCriterionConfig":
        return MetricStoppingCriterionConfig(
            Constants.Metrics.HammingMetric,
            MetricEvalSet.VAL,
            Constants.Metrics.Hamming.ROCAUC_T2T,
            StoppingCriterionComparators.HIGHER_IS_BETTER,
        )


@register("ValidationLossCriterion")
class ValidationLossStoppingCriterionConfig(StoppingCriterionConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._metric_criterion = MetricStoppingCriterionConfig(
            Constants.Metrics.Loss,
            MetricEvalSet.VAL,
            "",
            StoppingCriterionComparators.LOWER_IS_BETTER,
        )

    def update(self, metricwithset: str, epoch: int, result: Any) -> None:
        self._metric_criterion.update(metricwithset, epoch, result)

    def should_stop(self, patience_epochs: int) -> bool:
        return self._metric_criterion.should_stop(patience_epochs)

    def get_best_epochs(self) -> dict[str, int]:
        return self._metric_criterion.get_best_epochs()


class MultiStoppingCriterionConfig(StoppingCriterionConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.criterions = Field[list[StoppingCriterionConfig]]([])()

    def update(self, metricwithset: str, epoch: int, result: Any) -> None:
        for criterion in self.criterions:
            criterion.update(metricwithset, epoch, result)

    def get_best_epochs(self) -> dict[str, int]:
        best_epochs: dict[str, int] = {}
        for criterion in self.criterions:
            best_epochs.update(criterion.get_best_epochs())
        return best_epochs


@register("Criterions+Any")
class AnyStoppingCriterionConfig(MultiStoppingCriterionConfig):
    def should_stop(self, patience_epochs: int) -> bool:
        return any(criterion.should_stop(patience_epochs) for criterion in self.criterions)


@register("Criterions+All")
class AllStoppingCriterionConfig(MultiStoppingCriterionConfig):
    def should_stop(self, patience_epochs: int) -> bool:
        return all(criterion.should_stop(patience_epochs) for criterion in self.criterions)
