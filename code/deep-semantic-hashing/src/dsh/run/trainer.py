from abc import ABC, abstractmethod
import os
from typing import Any, Literal

from dsh.utils.eventsystem import EventSystem
from dsh.utils.logger import Logger
from dsh.utils.stopwatch import ProfilerStep, GlobalProfiler
from dsh.utils.types import Constants
from dsh.utils.types import EarlyStoppingInfo


class Trainer(ABC):
    def __init__(self, epochs: int, retain_checkpoints: str | list[int]):
        self.current_epoch: int = 0
        self.current_batch: int = 0
        self.epochs = epochs
        self.early_stop: bool = False
        self.early_stopping_info: EarlyStoppingInfo | None = None
        self.retain_checkpoints = retain_checkpoints
        EventSystem()[Constants.Events.Train.EarlyStop] += self._handle_early_stop_request

    def fit(self):
        success = False
        try:
            Logger().info("[FIT] Prepare training.")
            r = self._handle_resume()
            if isinstance(r, bool):
                self.resume_epoch = -1
            else:
                self.resume_epoch = r
            self.current_epoch = 0
            self.current_batch = 0
            self._prepare_fit()
            self.save_metadata()
            Logger().info("[FIT] Start training.")
            self._fit()
            success = True
        finally:
            if success:
                Logger().info("[FIT] Training completed successfully.")
            else:
                Logger().info("[FIT] Training failed. Trying to save model anyway.")
                self.save(True)

    @abstractmethod
    def save_metadata(self) -> None:
        raise NotImplementedError("Trainer must implement the save_metadata method")

    @abstractmethod
    def _handle_resume(self) -> int | Literal[False]:
        raise NotImplementedError("Trainer must implement the handle_resume method")

    @abstractmethod
    def _prepare_fit(self) -> None:
        raise NotImplementedError("Trainer must implement the prepare_fit method")

    def _fit(self) -> None:
        self._fit_impl()

    def _fit_impl(self) -> ProfilerStep:
        EventSystem()[Constants.Events.Train.Start](self)
        with GlobalProfiler().step("Fit") as step:
            for self.current_epoch in range(self.resume_epoch + 1, self.epochs):
                EventSystem()[Constants.Events.Train.EpochStart](self, self.current_epoch)
                with GlobalProfiler().step(f"Epoch {self.current_epoch}"):
                    Logger().info(f"[FIT] Fitting epoch {self.current_epoch}.")
                    self._prepare_fit_epoch()
                    self._model_to_device()
                    self._fit_epoch()
                    save_path = None
                    if self._check_trigger_inference_epoch():
                        save_path = self.save_model(False)
                        EventSystem()[Constants.Events.Train.Save](self, self.current_epoch)
                        self._model_to_cpu()
                        EventSystem()[Constants.Events.Train.TriggerInference](self, self.current_epoch)
                    elif self._check_save_epoch():
                        save_path = self.save_model(False)
                        EventSystem()[Constants.Events.Train.Save](self, self.current_epoch)
                    Logger().info(f"[FIT] Evaluating epoch {self.current_epoch}.")
                    self._model_to_device()
                    self._eval_epoch()
                EventSystem()[Constants.Events.Train.EpochEnd](self, self.current_epoch, step.duration)
                if self.early_stop:
                    Logger().info(f"[FIT] Early stopping triggered, stopping training after epoch {self.current_epoch}.")
                    if self.early_stopping_info != None:
                        Logger().info(f"[FIT] Best epochs:")
                        for name, epoch in self.early_stopping_info.best_epochs.items():
                            Logger().info(f"[FIT]     {epoch:>4d} {name}")
                    break
                if isinstance(self.retain_checkpoints, str):
                    if self.retain_checkpoints == "all":
                        pass  # retain all checkpoints
                    else:
                        raise ValueError("retain_checkpoints must be 'all' or a list of integer values.")
                elif isinstance(self.retain_checkpoints, list):
                    if self.current_epoch not in self.retain_checkpoints:
                        Logger().info(f"[FIT] Attempt deleting checkpoint of epoch {self.current_epoch}.")
                        if save_path != None and os.path.exists(save_path):
                            try:
                                os.remove(save_path)
                                Logger().info(f"[FIT] Deleted checkpoint {save_path}")
                            except Exception as e:
                                Logger().error(f"[FIT] Failed to delete checkpoint of epoch {self.current_epoch} due to {e}")
                else:
                    raise TypeError(f"Unknown type for retain_checkpoints: {type(self.retain_checkpoints)}")
        EventSystem()[Constants.Events.Train.End](self, step.duration)
        return step

    @abstractmethod
    def _prepare_fit_epoch(self) -> None:
        raise NotImplementedError("Trainer must implement the prepare_fit_epoch method")

    @abstractmethod
    def _fit_epoch(self) -> None:
        raise NotImplementedError("Trainer must implement the fit_epoch method")

    @abstractmethod
    def _check_save_epoch(self) -> bool:
        raise NotImplementedError("Trainer must implement the save_epoch method")

    def _check_trigger_inference_epoch(self) -> bool:
        return False

    @abstractmethod
    def _eval_epoch(self) -> None:
        raise NotImplementedError("Trainer must implement the eval_epoch method")

    @abstractmethod
    def save(self, mid_epoch: bool) -> None:
        raise NotImplementedError("Trainer must implement the save method")

    @abstractmethod
    def save_model(self, mid_epoch: bool) -> str:
        raise NotImplementedError("Trainer must implement the save_model method")

    @abstractmethod
    def _model_to_cpu(self) -> None:
        raise NotImplementedError("Trainer must implement the model_to_cpu method")

    @abstractmethod
    def _model_to_device(self) -> None:
        raise NotImplementedError("Trainer must implement the model_to_device method")

    def _handle_early_stop_request(
        self,
        sender: Any,
        should_stop_early: bool,
        additional_info: EarlyStoppingInfo | None = None,
    ) -> None:
        if should_stop_early:
            self.early_stop = True
            self.early_stopping_info = additional_info
            Logger().info("[FIT] Early stopping requested.")
