import argparse
from dataclasses import dataclass
from functools import partial
import os
from typing import Any
from torch.utils.tensorboard.writer import SummaryWriter

import dsh.config.run
from dsh.model.modelbase import TopLevelModelBase

from dsh.utils.eventsystem import ActionEventHandler, EventDistributor, EventDistributorConnector, EventSystem, mpc
from dsh.utils.jobs import JobManager
from dsh.utils.memory import CleanCudaMemory
from dsh.utils.random import random_hex_string
from dsh.utils.selector import (
    ConfigMode,
    get_config,
    get_device,
    get_epoched_data_to_load,
    get_inferrer,
    get_metric_calculator,
    get_model,
    get_trainer,
)
import dsh.utils.tensorboard
from dsh.utils.types import Constants, EarlyStoppingInfo, MetricEvalSet

__all__ = [
    "train_model",
    "infer_model",
    "calculate_metrics",
    "interleaved_pipeline",
    "sequential_pipeline",
    "parallelized_pipeline",
]


def train_model(cfg: dsh.config.run.TrainRunConfig, model: TopLevelModelBase) -> None:
    tensorboard_folder = os.path.join(cfg.env.resolve(cfg.env.log_path))
    with SummaryWriter(tensorboard_folder) as w:
        writer = dsh.utils.tensorboard.Writer(w)
        trainer = get_trainer(cfg, model, writer)
        trainer.fit()
        w.close()


def infer_model(cfg: dsh.config.run.InferenceRunConfig) -> None:
    for _ in cfg.env.experiments():
        cfg.load_model_config()
        device = get_device(cfg.env)
        with CleanCudaMemory(device):
            inferrer = get_inferrer(cfg)
            model = get_model(cfg.model, cfg.env)
            models_to_load = get_epoched_data_to_load(cfg.env)
            inferrer.run(models_to_load, model)
            del model


def calculate_metrics(cfg: dsh.config.run.MetricRunConfig) -> None:
    for _ in cfg.env.experiments():
        tensorboard_folder = os.path.join(cfg.env.resolve(cfg.env.log_path))
        with SummaryWriter(tensorboard_folder) as w:
            writer = dsh.utils.tensorboard.Writer(w)
            metric_calculator = get_metric_calculator(cfg, writer)
            inferences_to_load = get_epoched_data_to_load(cfg.env)
            metric_calculator.run(inferences_to_load)
            w.close()


def sequential_pipeline(args: argparse.Namespace) -> None:
    cfg_train = get_config(args, ConfigMode.TRAIN)
    model = get_model(cfg_train.model, cfg_train.env)
    train_model(cfg_train, model)
    experiment = cfg_train.env.resolve(cfg_train.env.experiment)

    cfg_infer = get_config(
        args,
        ConfigMode.INFERENCE,
        overrides={
            "env.experiment": experiment,
            "env.device": cfg_train.env.device,
        },
    )
    infer_model(cfg_infer)

    cfg_metrics = get_config(
        args,
        ConfigMode.METRICS,
        overrides={
            "env.experiment": experiment,
            "env.model_epoch": cfg_infer.env.model_epoch,
        },
    )
    calculate_metrics(cfg_metrics)


@dataclass
class JobDescriptionBase:
    args: argparse.Namespace
    overrides: dict[str, Any]
    event_system_name: str
    event_system_connection: mpc.Connection


@dataclass
class InferenceJobDescription(JobDescriptionBase):
    pass


@dataclass
class MetricsJobDescription(JobDescriptionBase):
    pass


@dataclass
class JobResult:
    pass


@dataclass
class InferenceJobResult(JobResult):
    pass


@dataclass
class MetricsJobResult(JobResult):
    criterion: float


def parallelized_pipeline_inference_job(desc: InferenceJobDescription) -> InferenceJobResult:
    EventDistributorConnector(desc.event_system_name, desc.event_system_connection)
    cfg_infer = get_config(
        desc.args,
        ConfigMode.INFERENCE,
        overrides=desc.overrides,
        cmd_overrides_prefix=Constants.CMD.OverridePrefixes.INFERENCE,
    )
    infer_model(cfg_infer)
    EventDistributorConnector().stop()
    return InferenceJobResult()


def parallelized_pipeline_metrics_job(desc: MetricsJobDescription) -> MetricsJobResult:
    EventDistributorConnector(desc.event_system_name, desc.event_system_connection)
    cfg_metrics = get_config(
        desc.args,
        ConfigMode.METRICS,
        overrides=desc.overrides,
        cmd_overrides_prefix=Constants.CMD.OverridePrefixes.METRICS,
    )
    calculate_metrics(cfg_metrics)
    EventDistributorConnector().stop()
    return MetricsJobResult(0)


def parallelized_pipeline(args: argparse.Namespace) -> None:
    ijm = JobManager[InferenceJobDescription, InferenceJobResult](2)
    mjm = JobManager[MetricsJobDescription, MetricsJobResult](1)

    tconn = EventDistributor().add_event_system(Constants.EventSystem.Master)
    EventDistributorConnector(Constants.EventSystem.Master, tconn)
    ES = EventSystem()

    cfg_train = get_config(
        args,
        ConfigMode.TRAIN,
        cmd_overrides_prefix=Constants.CMD.OverridePrefixes.TRAIN,
    )
    model = get_model(cfg_train.model, cfg_train.env)
    experiment = cfg_train.env.resolve(cfg_train.env.experiment)
    overrides: dict[str, Any] = {"env.experiment": experiment}

    ES[Constants.Events.Train.TriggerInference] += ActionEventHandler(partial(_add_inference_job, args, overrides, ijm))
    ES[Constants.Events.Infer.EpochEnd] += ActionEventHandler(partial(_add_metrics_job, args, overrides, mjm))
    # ES[Constants.Events.Metrics.EpochEnd] +=  Not implemented yet

    train_model(cfg_train, model)

    EventDistributorConnector().stop()
    EventDistributor().stop()
    mjm.close()
    ijm.close()


def _add_inference_job(
    args: argparse.Namespace,
    overrides: dict[str, Any],
    jm: JobManager[InferenceJobDescription, InferenceJobResult],
    _: Any,
    epoch: int,
) -> None:
    name = f"{Constants.EventSystem.JobInfer}+{epoch:04d}+{random_hex_string(4)}"
    c = EventDistributor().add_event_system(name)
    overrides = {**overrides, "env.model_epoch": epoch}
    jm.submit(parallelized_pipeline_inference_job, InferenceJobDescription(args, overrides, name, c), print, name)


def _add_metrics_job(
    args: argparse.Namespace,
    overrides: dict[str, Any],
    jm: JobManager[MetricsJobDescription, MetricsJobResult],
    _: Any,
    dataset: str,
    epoch: int,
) -> None:
    name = f"{Constants.EventSystem.JobMetrics}+{epoch:04d}+{random_hex_string(4)}"
    c = EventDistributor().add_event_system(name)
    overrides = {**overrides, "env.model_epoch": epoch}
    jm.submit(parallelized_pipeline_metrics_job, MetricsJobDescription(args, overrides, name, c), print, name)


def interleaved_pipeline(args: argparse.Namespace):
    ES = EventSystem()
    cfg = get_config(args, ConfigMode.ALL)
    cfg_train = cfg.to_train()
    model = get_model(cfg_train.model, cfg_train.env)

    ES[Constants.Events.Train.TriggerInference] += partial(interleaved_pipeline_infer, cfg, model.model_name)
    ES[Constants.Events.Infer.EpochEnd] += partial(interleaved_pipeline_metrics, cfg, model.model_name)
    ES[Constants.Events.Train.EpochEvalLossUpdate] += lambda _, epoch, result: interleaved_pipeline_update_early_stopping(
        cfg, _, Constants.Metrics.namewithset(Constants.Metrics.Loss, MetricEvalSet.VAL), epoch, result
    )
    ES[Constants.Events.Metrics.MetricResult] += partial(interleaved_pipeline_update_early_stopping, cfg)
    ES[Constants.Events.Metrics.GlobalEnd] += partial(interleaved_pipeline_determine_early_stopping, cfg)

    train_model(cfg_train, model)


def interleaved_pipeline_infer(cfg: dsh.config.run.PipelineRunConfig, model_name: str, _: Any, epoch: int):
    cfg_infer = cfg.to_infer(model_name, epoch)
    infer_model(cfg_infer)


def interleaved_pipeline_metrics(cfg: dsh.config.run.PipelineRunConfig, model_name: str, _: Any, dataset: str, epoch: int):
    cfg_metric = cfg.to_metric(model_name, dataset, epoch)
    calculate_metrics(cfg_metric)


def interleaved_pipeline_update_early_stopping(
    cfg: dsh.config.run.PipelineRunConfig, _: Any, metricwithset: str, epoch: int, result: Any
):
    if cfg.train.early_stopping != None:
        es = cfg.train.early_stopping
        es.stopping_criterion.update(metricwithset, epoch, result)


def interleaved_pipeline_determine_early_stopping(cfg: dsh.config.run.PipelineRunConfig, _: Any):
    if cfg.train.early_stopping != None:
        es = cfg.train.early_stopping
        if es.stopping_criterion.should_stop(es.patience_epochs):
            best_epochs = es.stopping_criterion.get_best_epochs()
            EventSystem()[Constants.Events.Train.EarlyStop](None, True, EarlyStoppingInfo(best_epochs))
