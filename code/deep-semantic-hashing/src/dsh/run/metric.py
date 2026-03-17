from abc import ABC, abstractmethod
import json
from typing import Any, Callable
import h5py
import os

import dsh.config.env
import dsh.config.metric
import dsh.config.run
import dsh.metric.metricbase

from dsh.utils.eventsystem import EventSystem
from dsh.utils.logger import Logger
from dsh.utils.tensorboard import Writer
from dsh.utils.types import Constants

GET_METRIC = Callable[
    [dsh.config.metric.MetricConfiguration, dsh.config.env.MetricEnvironmentConfig], dsh.metric.metricbase.MetricBase
]


class MetricRunner(ABC):
    def __init__(
        self,
        cfg: dsh.config.run.MetricRunConfig,
        writer: Writer,
        get_metric: GET_METRIC,
    ):
        self.cfg = cfg
        self.writer = writer
        self.get_metric = get_metric

    def run(self, inferences_to_load: dict[int, str]) -> None:
        Logger().info(f"[MET] Start calculating metrics.")
        EventSystem()[Constants.Events.Metrics.GlobalStart](self)
        for metric_config in self.cfg.metric.metrics:
            metric = self.get_metric(metric_config, self.cfg.env)
            EventSystem()[Constants.Events.Metrics.MetricStart](self, metric.namewithset)
            Logger().info(f"[MET] Start calculating {metric.namewithset}.")
            self.cfg.env.add_metric_resolver(lambda: metric.name)
            self.cfg.env.add_metricset_resolver(lambda: metric.set)
            self._run_metric(inferences_to_load, metric)
            self.cfg.env.remove_metricset_resolver()
            self.cfg.env.remove_metric_resolver()
            EventSystem()[Constants.Events.Metrics.MetricEnd](self, metric.namewithset)
        EventSystem()[Constants.Events.Metrics.GlobalEnd](self)
        Logger().info("[MET] Finished calculating metrics.")

    @abstractmethod
    def _run_metric(self, inferences_to_load: dict[int, str], metric: dsh.metric.metricbase.MetricBase) -> None:
        raise NotImplementedError()


class H5MetricRunner(MetricRunner):
    def __init__(self, cfg: dsh.config.run.MetricRunConfig, writer: Writer, get_metric: GET_METRIC):
        super().__init__(cfg, writer, get_metric)

    def _run_metric(
        self,
        inferences_to_load: dict[int, str],
        metric: dsh.metric.metricbase.MetricBase[dsh.config.metric.MetricConfiguration, h5py.File, Any],
    ) -> None:
        for epoch, inference_path in inferences_to_load.items():
            Logger().info(f"[MET] Calculating {metric.namewithset} for epoch {epoch}.")
            EventSystem()[Constants.Events.Metrics.EpochStart](self, metric.namewithset, epoch)
            # Create output directory and output file
            self.cfg.env.add_epoch_resolver(lambda: epoch)
            output_path = self.cfg.env.resolve(self.cfg.env.output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Open the HDF5 file for reading
            with h5py.File(inference_path, "r") as infile:
                with self.writer.with_step_offset(epoch):
                    result = metric.calculate(infile, self.writer)
            EventSystem()[Constants.Events.Metrics.MetricResult, False](self, metric.namewithset, epoch, result)
            # Save the result to the output file
            with open(output_path, "w") as outfile:
                json.dump(result, outfile, indent=4, default=vars)
            self.cfg.env.remove_epoch_resolver()
            EventSystem()[Constants.Events.Metrics.EpochEnd](self, metric.namewithset, epoch)
