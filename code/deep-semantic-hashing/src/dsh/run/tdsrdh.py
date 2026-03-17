import torch
from torch.optim.optimizer import Optimizer as Optimizer

import dsh.config.data
import dsh.config.model
import dsh.config.run

from dsh.model.tdsrdh import TDSRDH

from dsh.utils.selector import get_optimizer, get_scheduler
import dsh.utils.tensorboard
from dsh.utils.types import HPARAM, T
from dsh.utils.unparsing import stringify_img_txt

from dsh.run.hashlearning import AlternatingHashLearningTrainer, HashLearningInferrer, SimultaneousHashLearningTrainer


def _tdsrdh_construct_optimizer_and_scheduler(self: "TDSRDHTrainer") -> None:
    self.optimizer_vision = get_optimizer(self.cfg, self.model.vision.parameters())
    self.optimizer_text = get_optimizer(self.cfg, self.model.text.parameters())
    self.scheduler_vision = get_scheduler(self.cfg.train.scheduler, self.optimizer_vision, self.cfg.train.epochs)
    self.scheduler_text = get_scheduler(self.cfg.train.scheduler, self.optimizer_text, self.cfg.train.epochs)


def _tdsrdh_step_scheduler_for_resume(self: "TDSRDHTrainer") -> None:
    self.scheduler_text.step()
    self.scheduler_vision.step()


def _tdsrdh_report_lr_and_step_scheduler(self: "TDSRDHTrainer"):
    self.writer.add_scalar(T.lr(T.TEXT), self.scheduler_text.get_last_lr()[0], self.current_epoch + 0.5)
    self.writer.add_scalar(T.lr(T.VISION), self.scheduler_vision.get_last_lr()[0], self.current_epoch + 0.5)
    self.scheduler_text.step()
    self.scheduler_vision.step()


def _tdsrdh_get_model_params_for_tensorboard(self: "TDSRDHTrainer") -> dict[str, HPARAM]:
    assert isinstance(self.cfg.model.vision.encoder, dsh.config.model.ResNetConfig)
    return dict(
        model_backbone=stringify_img_txt(self.cfg.model.vision.encoder.name, self.cfg.model.text.encoder.name),
        model_backbone_pretrained=stringify_img_txt(self.cfg.model.vision.encoder.pre_trained, False),
        model_hash=stringify_img_txt(self.cfg.model.vision.hash.stringify(), self.cfg.model.text.hash.stringify()),
        text_embedder=self.cfg.model.text_embedder.value,
        text_seqlen=self.cfg.model.text_sequence_length,
    )


class AlternatingTDSRDHTrainer(
    AlternatingHashLearningTrainer[TDSRDH, dsh.config.model.TDSRDHModelConfig, dsh.config.data.CrossModalDatasetConfig]
):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[dsh.config.model.TDSRDHModelConfig, dsh.config.data.CrossModalDatasetConfig],
        model: TDSRDH,
        writer: dsh.utils.tensorboard.Writer,
    ):
        super().__init__(cfg, model, writer)
        self.optimizer_vision: torch.optim.Optimizer
        self.optimizer_text: torch.optim.Optimizer
        self.scheduler_vision: torch.optim.lr_scheduler.LRScheduler
        self.scheduler_text: torch.optim.lr_scheduler.LRScheduler

    construct_optimizer_and_scheduler = _tdsrdh_construct_optimizer_and_scheduler
    _step_scheduler_for_resume = _tdsrdh_step_scheduler_for_resume
    _report_lr_and_step_scheduler = _tdsrdh_report_lr_and_step_scheduler
    _get_model_params_for_tensorboard = _tdsrdh_get_model_params_for_tensorboard

    def _set_model_train_vision(self) -> torch.optim.Optimizer:
        self.model.train()
        self.model.text.eval()
        return self.optimizer_vision

    def _set_model_train_text(self) -> torch.optim.Optimizer:
        self.model.train()
        self.model.vision.eval()
        return self.optimizer_text


class SimultaneousTDSRDHTrainer(
    SimultaneousHashLearningTrainer[TDSRDH, dsh.config.model.TDSRDHModelConfig, dsh.config.data.CrossModalDatasetConfig]
):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[dsh.config.model.TDSRDHModelConfig, dsh.config.data.CrossModalDatasetConfig],
        model: TDSRDH,
        writer: dsh.utils.tensorboard.Writer,
    ):
        super().__init__(cfg, model, writer)
        self.optimizer_vision: torch.optim.Optimizer
        self.optimizer_text: torch.optim.Optimizer
        self.scheduler_vision: torch.optim.lr_scheduler.LRScheduler
        self.scheduler_text: torch.optim.lr_scheduler.LRScheduler

    construct_optimizer_and_scheduler = _tdsrdh_construct_optimizer_and_scheduler
    _step_scheduler_for_resume = _tdsrdh_step_scheduler_for_resume
    _report_lr_and_step_scheduler = _tdsrdh_report_lr_and_step_scheduler
    _get_model_params_for_tensorboard = _tdsrdh_get_model_params_for_tensorboard

    def _set_model_train(self) -> tuple[torch.optim.Optimizer, ...]:
        self.model.train()
        return self.optimizer_text, self.optimizer_vision


TDSRDHTrainer = AlternatingTDSRDHTrainer | SimultaneousTDSRDHTrainer


class TDSRDHInferrer(HashLearningInferrer[TDSRDH, dsh.config.model.TDSRDHModelConfig, dsh.config.data.CrossModalDatasetConfig]):
    pass
