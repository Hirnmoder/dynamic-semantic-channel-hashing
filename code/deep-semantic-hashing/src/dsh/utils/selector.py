import argparse
from enum import Enum
from glob import glob
import numpy as np
import os
import torch
import torch.nn
from typing import Any, Iterator, Literal, overload

import dsh.config
import dsh.config.configuration
import dsh.config.data as cdata
import dsh.config.env as cenv
import dsh.config.metric as cmetric
import dsh.config.model as cmodel
import dsh.config.run as crun
import dsh.config.train as ctrain

import dsh.data.dataloader
import dsh.data.dataset
import dsh.data.mirflickr25k
import dsh.data.nuswide

import dsh.metric.hamming
import dsh.metric.metricbase

from dsh.model.cliphash import CLIPHash
from dsh.model.modelbase import CrossModalTopLevelModelBase, TopLevelModelBase
from dsh.model.tdsrdh import TDSRDH

import dsh.run.metric
import dsh.run.trainer
import dsh.run.inferrer

from dsh.utils.adapter import DatasetToModelAdapter, CrossModalDatasetToModelAdapter
from dsh.utils.collections import first_not_none
from dsh.utils.functions import sign_zero_is_negative, sign_zero_is_positive, sign_zero_is_zero
from dsh.utils.logger import Logger
import dsh.utils.logger
import dsh.utils.loss as loss
from dsh.utils.parsing import try_parse
import dsh.utils.tensorboard
from dsh.utils.types import CrossModalData, Data, DatasetForHashLearningInfo, DatasetInfo, Output, Quantize

__all__ = [
    "add_arguments_to_parser",
    "parse_args",
    "get_config",
    "get_model",
    "get_datasets",
    "get_dataloader",
    "get_device",
    "get_cpu_device",
    "get_optimizer",
    "get_scheduler",
    "get_trainer",
    "get_inferrer",
    "get_metric_calculator",
    "get_sign_function",
    "get_epoched_data_to_load",
    "get_metric",
    "get_loss",
    "ConfigMode",
    "init_logger",
]


def add_arguments_to_parser(parser: argparse.ArgumentParser) -> None:
    # add arguments to the parser
    parser.add_argument("--config", default="{main-dir}/config.{mode}.json5", help="Path to the configuration file")
    parser.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        help="Set the minimum logging level",
        choices=[
            ll.name
            for ll in dsh.utils.logger.LogLevel
            if ll not in (dsh.utils.logger.LogLevel.SPECIAL_ALL, dsh.utils.logger.LogLevel.SPECIAL_NONE)
        ]
        + ["ALL", "NONE"],
        type=str.upper,
    )
    parser.add_argument(
        "--enable-legacy-train-config",
        default=False,
        action="store_true",
        help="Enable loading legacy train config types (TDSRDHTrainConfig) for inference",
    )


def parse_args(parser: argparse.ArgumentParser, main_dir: str) -> argparse.Namespace:
    args, extra = parser.parse_known_args()
    # parse string to enum value
    args.log_level = (
        dsh.utils.logger.LogLevel.SPECIAL_ALL
        if args.log_level == "ALL"
        else (dsh.utils.logger.LogLevel.SPECIAL_NONE if args.log_level == "NONE" else dsh.utils.logger.LogLevel[args.log_level])
    )
    args.config = args.config.replace("{main-dir}", main_dir)

    # handle overrides
    overrides: dict[str, Any] = {}
    if len(extra) > 0:
        if extra[0] == "--":
            extra.pop(0)
        for arg in extra:
            k, v = arg.split("=", 1)
            # try casting v into a more specific type
            overrides[k] = try_parse(v)
    args.overrides = overrides  # store overrides directly in Namespace

    if args.enable_legacy_train_config:
        dsh.config.configuration.ConfigurationCodecSettings.register_type(crun.CrossModalTrainRunConfig, "TDSRDHTrainConfig")

    return args


def init_logger(args: argparse.Namespace | None) -> None:
    log_level = dsh.utils.logger.LogLevel.INFO if args is None else args.log_level
    Logger(
        dsh.utils.logger.ConsoleLoggerTarget(),
        minimum_log_level=log_level,
        exit_on_error=True,
    )
    Logger().info(f"Loaded arguments: {args}.")


class ConfigMode(Enum):
    TRAIN = "train"
    INFERENCE = "infer"
    METRICS = "metrics"
    ALL = "all"


# fmt: off
@overload
def get_config(args: argparse.Namespace, mode: Literal[ConfigMode.ALL], overrides: dict[str, Any] = {}, cmd_overrides_prefix: str = "") -> crun.PipelineRunConfig[cmodel.ModelConfig, cdata.DatasetConfig]: ...
@overload
def get_config(args: argparse.Namespace, mode: Literal[ConfigMode.TRAIN], overrides: dict[str, Any] = {}, cmd_overrides_prefix: str = "") -> crun.TrainRunConfig[cmodel.ModelConfig, cdata.DatasetConfig]: ...
@overload
def get_config(args: argparse.Namespace, mode: Literal[ConfigMode.INFERENCE], overrides: dict[str, Any] = {}, cmd_overrides_prefix: str = "") -> crun.InferenceRunConfig[cmodel.ModelConfig, cdata.DatasetConfig]: ...
@overload
def get_config(args: argparse.Namespace, mode: Literal[ConfigMode.METRICS], overrides: dict[str, Any] = {}, cmd_overrides_prefix: str = "") -> crun.MetricRunConfig: ...
# fmt: on
def get_config(
    args: argparse.Namespace,
    mode: ConfigMode,
    overrides: dict[str, Any] = {},
    cmd_overrides_prefix: str = "",
) -> crun.RunConfig[cmodel.ModelConfig, cdata.DatasetConfig] | crun.MetricRunConfig:
    # load configuration file or create new one if it does not exist
    config_filename = args.config.replace("{mode}", mode.value)
    if os.path.isfile(config_filename):
        Logger().info(f"Loading configuration from {config_filename}.")
        if mode == ConfigMode.METRICS:
            c = crun.MetricRunConfig.loadf(config_filename, allow_subclass=True)
        else:
            match mode:
                case ConfigMode.TRAIN:
                    EC = cenv.TrainEnvironmentConfig
                case ConfigMode.INFERENCE:
                    EC = cenv.InferEnvironmentConfig
                case ConfigMode.ALL:
                    EC = cenv.PipelineEnvironmentConfig
                case _:
                    raise ValueError("Invalid mode")
            c = crun.ModelRunConfigBase[cdata.DatasetConfig, EC].loadf(config_filename, allow_subclass=True)
        if c == None:
            Logger().error(f"Failed to load configuration.")
            exit(1)
    else:
        Logger().error("No configuration file found. Try creating a config using create-config command. Exiting...")
        exit(2)

    # handle programmatic overrides and commandline overrides
    for k, v in args.overrides.items():
        assert isinstance(k, str)
        # check if the current condition for cmd overrides is met
        if k.startswith((cmd_overrides_prefix, "*")):
            overrides[k] = v
    for k, v in overrides.items():
        element = c
        parts = k.split(".")
        for part in parts[:-1]:
            if not hasattr(element, part):
                Logger().warning(f"[CFG] Invalid override key {k}. Skipping this override.")
                break
            element = getattr(element, part)
        setattr(element, parts[-1], v)

    c.env.initialize()
    if isinstance(c, crun.InferenceRunConfig):
        pass
    elif isinstance(c, crun.TrainRunConfig):
        pass
    elif isinstance(c, crun.MetricRunConfig):
        pass
    elif isinstance(c, crun.PipelineRunConfig):
        pass
    else:
        raise ValueError(f"Expected configuration type {crun.RunConfig}, got {type(c)}")
    return c


# fmt: off
@overload
def get_model(cm: cmodel.HashModelConfig, ce: cenv.ModelDataEnvironmentConfig) -> CrossModalTopLevelModelBase[cmodel.HashModelConfig]: ...
@overload
def get_model(cm: cmodel.ModelConfig, ce: cenv.ModelDataEnvironmentConfig) -> TopLevelModelBase[cmodel.ModelConfig, Any, Output]: ...
# fmt: on
def get_model(
    cm: cmodel.ModelConfig,
    ce: cenv.ModelDataEnvironmentConfig,
) -> TopLevelModelBase[cmodel.ModelConfig, Any, Output]:
    if isinstance(cm, cmodel.TDSRDHModelConfig):
        model = TDSRDH(cm, ce)
    elif isinstance(cm, cmodel.CLIPHashModelConfig):
        model = CLIPHash(cm, ce)
    else:
        raise Exception(f"Unknown model: {cm}")

    return model


# fmt: off
@overload
def get_datasets(c: crun.ModelRunConfigBase[cdata.CDC, cenv.EC], adapter: CrossModalDatasetToModelAdapter, *modes: dsh.data.dataset.DatasetMode) -> tuple[DatasetForHashLearningInfo, tuple[dsh.data.dataset.CrossModalDataset[cdata.CrossModalDatasetConfig, CrossModalData], ...]]: ...
@overload
def get_datasets(c: crun.ModelRunConfigBase[cdata.DC, cenv.EC], adapter: DatasetToModelAdapter, *modes: dsh.data.dataset.DatasetMode) -> tuple[DatasetInfo, tuple[dsh.data.dataset.DatasetBase[cdata.DatasetConfig, Data], ...]]: ...
# fmt: on
def get_datasets(
    c: crun.ModelRunConfigBase,
    adapter: DatasetToModelAdapter,
    *modes: dsh.data.dataset.DatasetMode,
) -> tuple[DatasetInfo, tuple[dsh.data.dataset.DatasetBase, ...]]:
    if isinstance(c.data, cdata.NUSWideConfig):
        assert isinstance(adapter, CrossModalDatasetToModelAdapter)
        metadata, datasets = dsh.data.nuswide.create_datasets(c.data, c.env, adapter, set(modes))
    elif isinstance(c.data, cdata.MirFlickr25kConfig):
        assert isinstance(adapter, CrossModalDatasetToModelAdapter)
        metadata, datasets = dsh.data.mirflickr25k.create_datasets(c.data, c.env, adapter, set(modes))
    else:
        raise Exception(f"Unknown dataset type: {c.data}")

    for dataset in datasets.values():
        dataset.load_dataset()
    return metadata, tuple(datasets[mode] for mode in modes)


# fmt: off
DLCONFIG = (
    crun.TrainRunConfig[cmodel.ModelConfig, cdata.DatasetConfig]
    | crun.InferenceRunConfig[cmodel.ModelConfig, cdata.DatasetConfig]
)
# fmt: on
def get_dataloader(
    c: DLCONFIG,
    dataset: dsh.data.dataset.DatasetBase[cdata.DC, dsh.data.dataset.D],
    shuffle: bool,
    drop_last: bool,
) -> dsh.data.dataloader.DataLoader[dsh.data.dataset.D]:
    if isinstance(dataset, dsh.data.dataset.CrossModalDataset):
        collate_fn = CrossModalData.collate
    else:
        collate_fn = None
    if isinstance(c, crun.TrainRunConfig):
        batch_size = c.train.batch_size
    else:
        batch_size = c.env.batch_size
    return dsh.data.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=c.env.num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def get_device(c: cenv.ModelDataEnvironmentConfig) -> torch.device:
    if c.cudnn_benchmark == cenv.CUDNN_BENCHMARK.DISABLE:
        torch.backends.cudnn.benchmark = False
    elif c.cudnn_benchmark == cenv.CUDNN_BENCHMARK.ENABLE:
        torch.backends.cudnn.benchmark = True
    return torch.device(c.device)


def get_cpu_device() -> torch.device:
    return torch.device("cpu")


def get_optimizer(
    c: crun.TrainRunConfig[cmodel.ModelConfig, cdata.DatasetConfig],
    param: Iterator[torch.nn.parameter.Parameter],
) -> torch.optim.Optimizer:
    return c.train.optimizer.construct(param)


def get_scheduler(
    c: ctrain.SchedulerConfig,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if isinstance(c, ctrain.ConstantScheduler):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=c.factor, total_iters=epochs)
    elif isinstance(c, ctrain.CosineScheduler):
        # CosineAnnealingLR has a weird behavior where it is not independent of the optimizer's underlying learning rate
        # So we use a custom lambda scheduler instead
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: c.end_factor + (c.start_factor - c.end_factor) * (np.cos(epoch / epochs * np.pi) / 2.0 + 0.5),
        )
    elif isinstance(c, ctrain.SequenceScheduler):
        assert len(c.schedulers) > 0, "SequenceScheduler must have at least one scheduler"
        steps = [e for e, _ in c.schedulers]
        assert steps[0] == 0, "The first Scheduler in a SequenceScheduler must start at epoch 0"
        steps.append(epochs)  # Add the final epoch to ensure the last scheduler runs until the end
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        for start_epoch, end_epoch, (_, scheduler_config) in zip(steps[:-1], steps[1:], c.schedulers):
            schedulers.append(get_scheduler(scheduler_config, optimizer, end_epoch - start_epoch))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers,
            milestones=steps[1:-1],  # first and last epoch are implicit milestones
        )
    else:
        raise NotImplementedError(f"Unknown scheduler type {c}")


def get_trainer(
    c: crun.TrainRunConfig[cmodel.ModelConfig, cdata.DatasetConfig],
    model: TopLevelModelBase,
    writer: dsh.utils.tensorboard.Writer,
) -> dsh.run.trainer.Trainer:
    mode = c.train.train_mode
    if isinstance(model, TDSRDH):
        if not crun.is_model_instance(c, cmodel.TDSRDHModelConfig):
            raise ValueError("Model and config mismatch for TDSRDH")
        if not crun.is_data_instance(c, cdata.CrossModalDatasetConfig):
            raise ValueError("Data and config mismatch for TDSRDH")
        import dsh.run.tdsrdh  # lazy import to avoid circular imports # fmt: skip
        if ctrain.TrainMode.is_alternating(mode):
            return dsh.run.tdsrdh.AlternatingTDSRDHTrainer(c, model, writer)
        elif ctrain.TrainMode.is_simultaneous(mode):
            return dsh.run.tdsrdh.SimultaneousTDSRDHTrainer(c, model, writer)
        else:
            raise NotImplementedError(f"Unknown train mode {mode}")
    elif isinstance(model, CLIPHash):
        if not crun.is_model_instance(c, cmodel.CLIPHashModelConfig):
            raise ValueError("Model and config mismatch for CLIPHash")
        if not crun.is_data_instance(c, cdata.CrossModalDatasetConfig):
            raise ValueError("Data and config mismatch for CLIPHash")
        import dsh.run.cliphash  # lazy import to avoid circular imports # fmt: skip
        if ctrain.TrainMode.is_alternating(mode):
            return dsh.run.cliphash.AlternatingCLIPHashTrainer(c, model, writer)
        elif ctrain.TrainMode.is_simultaneous(mode):
            return dsh.run.cliphash.SimultaneousCLIPHashTrainer(c, model, writer)
        else:
            raise NotImplementedError(f"Unknown train mode {mode}")
    else:
        raise NotImplementedError(f"Unknown model type {type(model)}")


def get_inferrer(
    c: crun.InferenceRunConfig[cmodel.ModelConfig, cdata.DatasetConfig],
) -> dsh.run.inferrer.Inferrer:
    if crun.is_model_instance(c, cmodel.TDSRDHModelConfig):
        if not crun.is_data_instance(c, cdata.CrossModalDatasetConfig):
            raise ValueError("Data and config mismatch for TDSRDH")
        import dsh.run.tdsrdh  # lazy import to avoid circular imports # fmt: skip
        return dsh.run.tdsrdh.TDSRDHInferrer(c)
    elif crun.is_model_instance(c, cmodel.CLIPHashModelConfig):
        if not crun.is_data_instance(c, cdata.CrossModalDatasetConfig):
            raise ValueError("Data and config mismatch for CLIPHash")
        import dsh.run.cliphash  # lazy import to avoid circular imports # fmt: skip
        return dsh.run.cliphash.CLIPHashInferrer(c)
    else:
        raise NotImplementedError(f"Unknown inference config type {type(c)}")


def get_metric_calculator(
    c: crun.MetricRunConfig,
    writer: dsh.utils.tensorboard.Writer,
) -> dsh.run.metric.MetricRunner:
    return dsh.run.metric.H5MetricRunner(c, writer, get_metric)


def get_sign_function(s: cmodel.SignFunction) -> Quantize:
    match s:
        case cmodel.SignFunction.ZERO_IS_POSITIVE:
            return sign_zero_is_positive
        case cmodel.SignFunction.ZERO_IS_NEGATIVE:
            return sign_zero_is_negative
        case cmodel.SignFunction.ZERO_IS_ZERO:
            return sign_zero_is_zero
        case _:
            raise NotImplementedError(f"Unknown sign function {s}")


def get_epoched_data_to_load(e: cenv.InferEnvironmentConfig | cenv.MetricEnvironmentConfig) -> dict[int, str]:
    files_to_load: dict[int, str]
    file_path = e.model_path if isinstance(e, cenv.InferEnvironmentConfig) else e.input_path
    model_epoch = e.model_epoch
    if isinstance(model_epoch, int) or isinstance(model_epoch, list):
        if isinstance(model_epoch, int):
            model_epoch = [model_epoch]
        assert len(model_epoch) > 0, "At least one epoch must be specified"
        files_to_load = {}
        for epoch in model_epoch:
            e.add_epoch_resolver(lambda: epoch)
            files_to_load[epoch] = e.resolve(file_path)
            e.remove_epoch_resolver()
    elif model_epoch == cenv.InferEpochMode.ALREADY_SPECIFIED:
        files_to_load = {0: e.resolve(file_path)}
    elif model_epoch in (cenv.InferEpochMode.LATEST, cenv.InferEpochMode.ALL):
        # find all possible model files
        base_path = e.resolve(file_path)
        base_path_splitted = base_path.split(r"${EPOCH}")
        assert len(base_path_splitted) == 2, f"Path must contain EPOCH placeholder exactly once: {base_path}"
        base_path_wildcard = "*".join(base_path_splitted)
        files = glob(base_path_wildcard)
        file_epochs: dict[int, str] = {}
        for file in files:
            ep = file.replace(base_path_splitted[0], "").replace(base_path_splitted[1], "")
            assert str(int(ep)) == ep, f"Unable to parse epoch from file name: {ep}"
            file_epochs[int(ep)] = file
        assert len(file_epochs) > 0, f"No model files found for path: {base_path_wildcard}"
        if model_epoch == cenv.InferEpochMode.LATEST:
            ep = max(file_epochs.keys())
            files_to_load = {ep: file_epochs[ep]}
        elif model_epoch == cenv.InferEpochMode.ALL:
            files_to_load = {ep: file_epochs[ep] for ep in sorted(file_epochs.keys())}
        else:
            raise ValueError(f"Unknown model epoch mode: {model_epoch}")
    else:
        raise ValueError(f"Unknown model epoch mode: {model_epoch}")
    return files_to_load


def get_metric(e: cmetric.MetricConfiguration, env: cenv.MetricEnvironmentConfig) -> dsh.metric.metricbase.MetricBase:
    if isinstance(e, cmetric.HammingMetricConfiguration):
        return dsh.metric.hamming.HammingMetric(e, env)
    else:
        raise ValueError(f"Unknown metric configuration type: {type(e)}")


@overload
def get_loss(c: ctrain.TDSRDHPaperLoss, quantization: Quantize) -> loss.TDSRDHPaperLoss: ...
@overload
def get_loss(c: ctrain.SCHPaperLoss, quantization: Quantize) -> loss.SCHPaperLoss: ...
@overload
def get_loss(c: ctrain.OurLoss, quantization: Quantize) -> loss.OurLoss: ...
def get_loss(c: ctrain.LossConfig, quantization: Quantize) -> loss.LossBase:
    if isinstance(c, ctrain.TDSRDHPaperLoss):
        if c.pairwise_loss == ctrain.PairwiseLossType.AS_PAPER:
            pairwise_loss = loss.PairwiseNLLLoss()
        else:
            raise ValueError(f"Unsupported pairwise loss type {c.pairwise_loss}")
        if c.quantization_loss == ctrain.QuantizationLossType.CROSS_MODALITY:
            quantization_loss = loss.CrossModalityQuantizationLoss(quantization=quantization)
        elif c.quantization_loss == ctrain.QuantizationLossType.SQUARED_FROBENIUS:
            quantization_loss = loss.SameModalityQuantizationLoss(quantization=quantization)
        elif c.quantization_loss == ctrain.QuantizationLossType.AVERAGE_HASHCODES:
            quantization_loss = loss.AverageHashCodeQuantizationLoss(quantization=quantization)
        elif c.quantization_loss == ctrain.QuantizationLossType.AVERAGE_ACTIVATION_TO_HASHCODE:
            quantization_loss = loss.AverageActivationToHashCodeQuantizationLoss(quantization=quantization)
        elif c.quantization_loss == ctrain.QuantizationLossType.ALWAYS_IMAGE:
            quantization_loss = loss.AlwaysOneModalityQuantizationLoss(quantization=quantization, modality="image")
        elif c.quantization_loss == ctrain.QuantizationLossType.ALWAYS_TEXT:
            quantization_loss = loss.AlwaysOneModalityQuantizationLoss(quantization=quantization, modality="text")
        else:
            raise ValueError(f"Unsupported quantization loss type {c.quantization_loss}")
        if c.triplet_loss == ctrain.TripletLossType.AS_PAPER:
            triplet_loss = loss.CrossModalityTripletLoss(epsilon=c.epsilon, vectorized=True)
        else:
            raise ValueError(f"Unsupported triplet loss type {c.triplet_loss}")
        return loss.TDSRDHPaperLoss(
            pairwise_loss=pairwise_loss,
            quantization_loss=quantization_loss,
            triplet_loss=triplet_loss,
            alpha=c.alpha,
            beta=c.beta,
            normalize_losses_by_batch_size=c.normalize_by_batch_size,
        )
    elif isinstance(c, ctrain.SCHPaperLoss):
        return _get_schpaperloss(c, None, None)
    elif isinstance(c, ctrain.OurLoss):
        if isinstance(c.semanticchannel, ctrain.SCHPaperLoss):
            s = _get_schpaperloss(c.semanticchannel, c.normalize_by_batch_size, c.use_frobenius)
        elif isinstance(c.semanticchannel, ctrain.DSCHLoss):
            s = loss.DSCHLoss(
                tau=c.semanticchannel.tau,
                alpha=c.semanticchannel.alpha,
                beta=c.semanticchannel.beta,
                lambda_neg=c.semanticchannel.lambda_neg,
                gamma_w=c.semanticchannel.gamma_w,
                gamma_l=c.semanticchannel.gamma_l,
                anchor=c.semanticchannel.anchor,
                normalize_losses_by_batch_size=c.normalize_by_batch_size,
                use_frobenius=c.use_frobenius,
                use_k_approximation=False,
                vectorized=True,
            )
        else:
            raise ValueError(f"Unsupported semanticchannel type {type(c.semanticchannel)}")
        return loss.OurLoss(
            semanticchannel=s,
            quantization=loss.DefaultQuantizationLoss(
                quantization=quantization,
                use_frobenius=c.use_frobenius,
                normalize_by_batch_size=c.normalize_by_batch_size,
            ),
            q_weight=c.q_weight,
        )
    else:
        raise ValueError(f"Unsupported loss type {type(c)}")


def _get_schpaperloss(
    c: ctrain.SCHPaperLoss,
    normalize_by_batch_size: bool | None,
    use_frobenius: bool | None,
) -> loss.SCHPaperLoss:
    if use_frobenius is not None:
        return loss.ModSCHPaperLoss(
            tau=c.tau,
            alpha=c.alpha,
            beta=c.beta,
            lneg_lambda_l=c.lneg_lambda_l,
            normalize_losses_by_batch_size=first_not_none(normalize_by_batch_size, c.normalize_by_batch_size),
            use_frobenius=use_frobenius,
            use_k_approximation=c.use_k_approximation,
            vectorized=True,
        )
    else:
        return loss.SCHPaperLoss(
            tau=c.tau,
            alpha=c.alpha,
            beta=c.beta,
            lneg_lambda_l=c.lneg_lambda_l,
            normalize_losses_by_batch_size=first_not_none(normalize_by_batch_size, c.normalize_by_batch_size),
            use_k_approximation=c.use_k_approximation,
            vectorized=True,
        )
