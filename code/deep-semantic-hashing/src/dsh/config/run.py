from abc import ABC, abstractmethod
from typing import Any, Generic, TypeGuard, overload

from dsh.config.configuration import ConfigurationBase, Field, register
from dsh.config.data import DC, CrossModalDatasetConfig, DatasetConfig, NUSWideConfig
from dsh.config.env import EC, MetricEnvironmentConfig, PipelineEnvironmentConfig, TrainEnvironmentConfig, InferEnvironmentConfig
from dsh.config.metric import MetricsConfiguration
from dsh.config.model import MC, ModelConfig, HashModelConfig, TDSRDHModelConfig
from dsh.config.train import TrainConfig
from dsh.utils.logger import Logger


class RunConfigBase(ConfigurationBase, Generic[EC], ABC):
    @abstractmethod
    def __init__(
        self,
        cls_default_env_config: type[EC],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.env = Field[EC](cls_default_env_config)()


class ModelRunConfigBase(RunConfigBase[EC], Generic[DC, EC], ABC):
    def __init__(
        self,
        cls_default_data_config: type[DC],
        cls_default_env_config: type[EC],
        **kwargs: Any,
    ):
        super().__init__(cls_default_env_config, **kwargs)

        self.data = Field[DC](cls_default_data_config)()


class TrainRunConfig(ModelRunConfigBase[DC, TrainEnvironmentConfig], Generic[MC, DC], ABC):
    @abstractmethod
    def __init__(
        self,
        cls_default_model_config: type[MC],
        cls_default_data_config: type[DC],
        cls_default_train_config: type[TrainConfig] = TrainConfig,
        cls_default_env_config: type[TrainEnvironmentConfig] = TrainEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_data_config,
            cls_default_env_config,
            **kwargs,
        )

        self.model = Field[MC](cls_default_model_config)()
        self.train = Field[TrainConfig](cls_default_train_config)()


@register("CrossModalTraining")
class CrossModalTrainRunConfig(TrainRunConfig[HashModelConfig, CrossModalDatasetConfig]):
    def __init__(
        self,
        cls_default_model_config: type[HashModelConfig] = TDSRDHModelConfig,
        cls_default_data_config: type[CrossModalDatasetConfig] = NUSWideConfig,
        cls_default_train_config: type[TrainConfig] = TrainConfig,
        cls_default_env_config: type[TrainEnvironmentConfig] = TrainEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_model_config,
            cls_default_data_config,
            cls_default_train_config,
            cls_default_env_config,
            **kwargs,
        )


class InferenceRunConfig(ModelRunConfigBase[DC, InferEnvironmentConfig], Generic[MC, DC], ABC):
    @abstractmethod
    def __init__(
        self,
        cls_default_data_config: type[DC],
        cls_default_env_config: type[InferEnvironmentConfig] = InferEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_data_config,
            cls_default_env_config,
            **kwargs,
        )

    def load_model_config(self):
        self.model: MC = self._load_model_config()

    def _load_model_config(self) -> MC:
        model_config_path = self.env.resolve(self.env.config_path)
        # Load train config from model_path
        train_config = TrainRunConfig[MC, DatasetConfig].loadf(model_config_path, allow_subclass=True)
        if train_config == None:
            Logger().error(f"[CFG] Unable to load train config from {model_config_path}.")
            exit(1)
        return train_config.model


@register("CrossModalInference")
class CrossModalInferenceRunConfig(InferenceRunConfig[HashModelConfig, CrossModalDatasetConfig]):
    def __init__(
        self,
        cls_default_data_config: type[CrossModalDatasetConfig] = NUSWideConfig,
        cls_default_env_config: type[InferEnvironmentConfig] = InferEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_data_config,
            cls_default_env_config,
            **kwargs,
        )


@register("MetricRun")
class MetricRunConfig(RunConfigBase[MetricEnvironmentConfig]):
    def __init__(
        self,
        cls_default_metric_config: type[MetricsConfiguration] = MetricsConfiguration,
        cls_default_env_config: type[MetricEnvironmentConfig] = MetricEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_env_config,
            **kwargs,
        )
        self.metric = Field[MetricsConfiguration](cls_default_metric_config)()


class PipelineRunConfig(ModelRunConfigBase[DC, PipelineEnvironmentConfig], Generic[MC, DC]):
    def __init__(
        self,
        cls_default_model_config: type[MC],
        cls_default_data_config: type[DC],
        cls_default_train_config: type[TrainConfig] = TrainConfig,
        cls_default_metric_config: type[MetricsConfiguration] = MetricsConfiguration,
        cls_default_env_config: type[PipelineEnvironmentConfig] = PipelineEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_data_config,
            cls_default_env_config,
            **kwargs,
        )

        self.model = Field[MC](cls_default_model_config)()
        self.train = Field[TrainConfig](cls_default_train_config)()
        self.metric = Field[MetricsConfiguration](cls_default_metric_config)()

    def to_train(self) -> TrainRunConfig[MC, DC]:
        tc = self._get_empty_train_run_config()
        tc.data = self.data
        tc.env = self.env.get_train_environment()
        tc.model = self.model
        tc.train = self.train
        return tc

    @abstractmethod
    def _get_empty_train_run_config(self) -> TrainRunConfig[MC, DC]:
        raise NotImplementedError()

    def to_infer(self, model_name: str, epoch: int) -> InferenceRunConfig[MC, DC]:
        ic = self._get_empty_infer_run_config()
        ic.data = self.data
        ic.env = self.env.get_infer_environment(model_name, epoch)
        ic.model = self.model
        return ic

    @abstractmethod
    def _get_empty_infer_run_config(self) -> InferenceRunConfig[MC, DC]:
        raise NotImplementedError()

    def to_metric(self, model_name: str, dataset: str, epoch: int) -> MetricRunConfig:
        mc = MetricRunConfig()
        mc.env = self.env.get_metric_environment(model_name, dataset, epoch)
        mc.metric = self.metric
        return mc


@register("CrossModalHashingPipeline")
class CrossModalHashingPipelineConfig(PipelineRunConfig[HashModelConfig, CrossModalDatasetConfig]):
    def __init__(
        self,
        cls_default_model_config: type[HashModelConfig] = TDSRDHModelConfig,
        cls_default_data_config: type[CrossModalDatasetConfig] = NUSWideConfig,
        cls_default_train_config: type[TrainConfig] = TrainConfig,
        cls_default_metric_config: type[MetricsConfiguration] = MetricsConfiguration,
        cls_default_env_config: type[PipelineEnvironmentConfig] = PipelineEnvironmentConfig,
        **kwargs: Any,
    ):
        super().__init__(
            cls_default_model_config,
            cls_default_data_config,
            cls_default_train_config,
            cls_default_metric_config,
            cls_default_env_config,
            **kwargs,
        )

    def _get_empty_train_run_config(self) -> TrainRunConfig[HashModelConfig, CrossModalDatasetConfig]:
        return CrossModalTrainRunConfig(HashModelConfig, CrossModalDatasetConfig)

    def _get_empty_infer_run_config(self) -> InferenceRunConfig[HashModelConfig, CrossModalDatasetConfig]:
        return CrossModalInferenceRunConfig(CrossModalDatasetConfig)


RunConfig = InferenceRunConfig[MC, DC] | TrainRunConfig[MC, DC] | PipelineRunConfig[MC, DC]


@overload
def is_model_instance(c: InferenceRunConfig[ModelConfig, DC], m: type[MC]) -> TypeGuard[InferenceRunConfig[MC, DC]]: ...
@overload
def is_model_instance(c: TrainRunConfig[ModelConfig, DC], m: type[MC]) -> TypeGuard[TrainRunConfig[MC, DC]]: ...
@overload
def is_model_instance(c: PipelineRunConfig[ModelConfig, DC], m: type[MC]) -> TypeGuard[PipelineRunConfig[MC, DC]]: ...
def is_model_instance(c: RunConfig[ModelConfig, DC], m: type[MC]) -> TypeGuard[RunConfig[MC, DC]]:
    return isinstance(c.model, m)


@overload
def is_data_instance(c: InferenceRunConfig[MC, DatasetConfig], d: type[DC]) -> TypeGuard[InferenceRunConfig[MC, DC]]: ...
@overload
def is_data_instance(c: TrainRunConfig[MC, DatasetConfig], d: type[DC]) -> TypeGuard[TrainRunConfig[MC, DC]]: ...
@overload
def is_data_instance(c: PipelineRunConfig[MC, DatasetConfig], d: type[DC]) -> TypeGuard[PipelineRunConfig[MC, DC]]: ...
def is_data_instance(c: RunConfig[MC, DatasetConfig], d: type[DC]) -> TypeGuard[RunConfig[MC, DC]]:
    return isinstance(c.data, d)
