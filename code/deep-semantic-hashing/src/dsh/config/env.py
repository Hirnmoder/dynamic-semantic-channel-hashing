import datetime
from enum import Enum
import os
import re
import torch
from typing import Callable, Iterator, Optional, TypeVar

from dsh.config.configuration import ConfigurationBase, register, Field
from dsh.utils.collections import map_if_present
from dsh.utils.logger import Logger
from dsh.utils.random import random_hex_string


class CUDNN_BENCHMARK(Enum):
    ENABLE = True
    DISABLE = False
    IGNORE = "ignore"


class InferEpochMode(Enum):
    LATEST = "latest"
    ALL = "all"
    ALREADY_SPECIFIED = None


VariableValue = str | int | float | bool | None
Resolver = Callable[[], VariableValue]
DynResolver = Callable[["EnvironmentConfig", re.Match], VariableValue]


class EnvironmentConfig(ConfigurationBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.variables = Field[dict[str, VariableValue]]({})()
        self.log_path = Field[str](r"/app/logs/${EXPERIMENT}")()

        self._initialized: bool = False
        self._replacements: dict[str, str] = {}
        self._resolvers: dict[str, Resolver] = {}
        self._dynresolvers: dict[str, DynResolver] = {}

        self.add_dynresolver(r"\${([^}]+)\.DIR}", lambda env, match: os.path.dirname(env.resolve(f"${{{match.group(1)}}}")))

    def initialize(
        self,
        time: datetime.datetime | None = None,
        random: str | None = None,
        raise_error: bool = True,
        **kwargs: VariableValue,
    ):
        if self._initialized:
            if raise_error:
                raise ValueError("This environment has already been initialized.")
            return

        # use current UTC time as default
        time = time if time else datetime.datetime.now(datetime.UTC)

        # generate a random hex string of length 8 as default
        random = random if random else random_hex_string(8)

        # use instance fields to generate replacements
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                name = k.upper()
                if isinstance(v, VariableValue):
                    self._replacements.update(**{name: str(v)})

        self._replacements.update(**{k.upper(): str(v) for k, v in self.variables.items()})
        self._replacements.update(**dict(TIME=time.strftime("%Y%m%d-%H%M%S"), RANDOM=random))
        self._replacements.update(**{k.upper(): str(v) for k, v in kwargs.items()})
        self._initialized = True

    def resolve(self, string: str) -> str:
        if not self._initialized:
            Logger().error(f"[ENV] {self.__class__.__name__} not initialized.")
        return self._resolve(string, self._replacements, self._resolvers, self._dynresolvers)

    def _resolve(
        self,
        string: str,
        variables: dict[str, str],
        resolvers: dict[str, Resolver],
        dynresolvers: dict[str, DynResolver],
    ) -> str:
        replaced = True  # initial value for looping at least once
        while replaced:
            replaced = False  # reset
            # first loop through resolvers in case they override a (static) variable replacement
            for name, resolver in resolvers.items():
                to_replace = f"${{{name}}}"
                if to_replace in string:
                    string = string.replace(to_replace, str(resolver()))
                    replaced = True
            # second loop through dynamic resolvers to do (dynamic) variable replacements
            for pattern, resolver in dynresolvers.items():
                for match in re.finditer(pattern, string):
                    replacement = resolver(self, match)
                    if replacement is not None:
                        string = string[: match.start()] + str(replacement) + string[match.end() :]
                        replaced = True
            # third loop through variables to do (static) variable replacement
            for name, value in variables.items():
                to_replace = f"${{{name}}}"
                if to_replace in string:
                    string = string.replace(to_replace, str(value))
                    replaced = True
        return string

    def add_resolver(self, name: str, resolver: Resolver):
        self._resolvers[name] = resolver

    def add_dynresolver(self, pattern: str, dynresolver: DynResolver):
        self._dynresolvers[pattern] = dynresolver

    def remove_resolver(self, name: str):
        del self._resolvers[name]

    def add_model_resolver(self, model: Resolver):
        self.add_resolver("MODEL", model)

    def remove_model_resolver(self):
        self.remove_resolver("MODEL")

    def add_epoch_resolver(self, epoch: Resolver):
        self.add_resolver("EPOCH", epoch)

    def remove_epoch_resolver(self):
        self.remove_resolver("EPOCH")

    def add_dataset_resolver(self, dataset: Resolver):
        self.add_resolver("DATASET", dataset)

    def remove_dataset_resolver(self):
        self.remove_resolver("DATASET")

    def add_metric_resolver(self, metric: Resolver):
        self.add_resolver("METRIC", metric)

    def remove_metric_resolver(self):
        self.remove_resolver("METRIC")

    def add_metricset_resolver(self, metric: Resolver):
        self.add_resolver("METRICSET", metric)

    def remove_metricset_resolver(self):
        self.remove_resolver("METRICSET")

    def add_resolvers(self, **kwargs: Resolver):
        for name, resolver in kwargs.items():
            self.add_resolver(name, resolver)


class ModelDataEnvironmentConfig(EnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = Field[str]("cuda")()
        self.cudnn_benchmark = Field[CUDNN_BENCHMARK](CUDNN_BENCHMARK.IGNORE)()

        self.model_path = Field[str](r"/app/models/${EXPERIMENT}/${EPOCH}-${MODEL}.pth")()
        self.config_path = Field[str](r"/app/models/${EXPERIMENT}/config.json5")()
        self.data_path = Field[str](r"/app/data/${DATASET}")()
        self.misc_path = Field[str](r"/app/data")()  # path for misc data, e.g. word embeddings

        self.trace_vector_statistics = Field[bool](False)()

    def initialize(
        self,
        time: datetime.datetime | None = None,
        random: str | None = None,
        raise_error: bool = True,
        **kwargs: VariableValue,
    ):
        if self._initialized:
            if raise_error:
                raise ValueError("This environment has already been initialized.")
            return
        # use best possible device
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                Logger().warning(f"[ENV] CUDA is not available. Fallback to CPU.")
                self.device = "cpu"
        elif self.device == "cpu":
            if torch.cuda.is_available():
                Logger().warning(f"[ENV] Current device is CPU, even though CUDA is available.")
        else:
            Logger().error(
                f"[ENV] Invalid device specified: {self.device}. Supported devices are 'cpu' and 'cuda'. Please specify a valid device."
            )
        return super().initialize(time, random, raise_error, **kwargs)


class MultiExperimentEnvironmentConfig(EnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.experiment = Field[str | list[str]](r"20250000-000000-MODEL")()

    def experiments(self) -> Iterator[str]:
        experiments = self.experiment if isinstance(self.experiment, list) else [self.experiment]
        for experiment in experiments:
            self.add_resolver("EXPERIMENT", lambda: experiment)
            yield experiment
            self.remove_resolver("EXPERIMENT")


@register("TrainEnv")
class TrainEnvironmentConfig(ModelDataEnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.experiment = Field[str](r"${TIME}-${MODEL}")()
        self.num_workers = Field[int](8)()
        self.resume_path = Field[str]("")()
        self.resume_epoch = Field[Optional[int]](None)()
        self.measure_time_precisely = Field[bool](False)()
        self.display_images_every_n_epochs = Field[int](0)()
        self.retain_checkpoints = Field[str | list[int]]("all")()


@register("InferEnv")
class InferEnvironmentConfig(ModelDataEnvironmentConfig, MultiExperimentEnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_workers = Field[int](8)()
        self.output_path = Field[str](r"${MODEL_PATH.DIR}/infer/${DATASET}/${EPOCH}.h5")()
        self.model_epoch = Field[int | list[int] | InferEpochMode](InferEpochMode.LATEST)()
        self.batch_size = Field[int](1024)()


@register("MetricEnv")
class MetricEnvironmentConfig(MultiExperimentEnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = Field[str](r"NUS-WIDE")()
        self.input_path = Field[str](r"/app/models/${EXPERIMENT}/infer/${DATASET}/${EPOCH}.h5")()
        self.output_path = Field[str](r"/app/models/${EXPERIMENT}/eval/${DATASET}/${EPOCH}-${METRIC}-${METRICSET}.json")()
        self.model_epoch = Field[int | list[int] | InferEpochMode](InferEpochMode.LATEST)()
        self.num_workers = Field[int](32)()
        self.samples_per_worker = Field[int](25)()


@register("PipelineEnv")
class PipelineEnvironmentConfig(ModelDataEnvironmentConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.experiment = Field[str](r"${TIME}-${MODEL}")()

        self.train_resume_path = Field[str]("")()
        self.train_resume_epoch = Field[Optional[int]](None)()
        self.train_measure_time_precisely = Field[bool](False)()
        self.train_num_workers = Field[int](8)()
        self.train_display_images_every_n_epochs = Field[int](0)()
        self.train_retain_checkpoints = Field[str | list[int]]("all")()

        self.infer_path = Field[str](r"${MODEL_PATH.DIR}/infer/${DATASET}/${EPOCH}.h5")()
        self.infer_batch_size = Field[int](1024)()
        self.infer_num_workers = Field[int](8)()

        self.metric_path = Field[str](r"/app/models/${EXPERIMENT}/eval/${DATASET}/${EPOCH}-${METRIC}-${METRICSET}.json")()
        self.metric_num_workers = Field[int](32)()
        self.metric_samples_per_worker = Field[int](50)()

    def get_train_environment(self) -> TrainEnvironmentConfig:
        tc = TrainEnvironmentConfig()
        for k, v in self.__dict__.items():
            if not k.startswith("_") and not k.startswith("infer_") and not k.startswith("metric_"):
                k = map_if_present(
                    k,
                    {
                        "train_resume_path": "resume_path",
                        "train_resume_epoch": "resume_epoch",
                        "train_measure_time_precisely": "measure_time_precisely",
                        "train_num_workers": "num_workers",
                        "train_display_images_every_n_epochs": "display_images_every_n_epochs",
                        "train_retain_checkpoints": "retain_checkpoints",
                    },
                )
                setattr(tc, k, v)
        tc.initialize()
        tc._replacements.update(self._replacements)
        return tc

    def get_infer_environment(self, model: str, epoch: int) -> InferEnvironmentConfig:
        ic = InferEnvironmentConfig()
        for k, v in self.__dict__.items():
            if not k.startswith("_") and not k.startswith("train_") and not k.startswith("metric_"):
                k = map_if_present(
                    k,
                    {
                        "infer_path": "output_path",
                        "infer_num_workers": "num_workers",
                        "infer_batch_size": "batch_size",
                    },
                )
                setattr(ic, k, v)
        ic.model_epoch = epoch
        ic.initialize()
        ic._replacements.update(self._replacements)
        ic.add_model_resolver(lambda: model)
        return ic

    def get_metric_environment(self, model: str, dataset: str, epoch: int) -> MetricEnvironmentConfig:
        mc = MetricEnvironmentConfig()
        for k, v in self.__dict__.items():
            if not k.startswith("_") and not k.startswith("train_") and not k.startswith("infer_"):
                k = map_if_present(
                    k,
                    {
                        "infer_path": "input_path",
                        "metric_path": "output_path",
                        "metric_num_workers": "num_workers",
                        "metric_samples_per_worker": "samples_per_worker",
                    },
                )
                setattr(mc, k, v)
        mc.dataset = dataset
        mc.model_epoch = epoch
        mc.initialize()
        mc._replacements.update(self._replacements)
        mc.add_model_resolver(lambda: model)
        return mc


EC = TypeVar("EC", bound=EnvironmentConfig, covariant=True)  # Type variable for environment config
