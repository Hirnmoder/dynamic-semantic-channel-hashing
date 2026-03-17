import torch

import dsh.config.data
import dsh.config.model
import dsh.config.run

from dsh.model.cliphash import CLIPHash

from dsh.utils.selector import get_optimizer, get_scheduler
from dsh.utils.types import HPARAM, T
import dsh.utils.tensorboard

from dsh.run.hashlearning import AlternatingHashLearningTrainer, HashLearningInferrer, SimultaneousHashLearningTrainer


def _cliphash_construct_optimizer_and_scheduler(self: "CLIPHashTrainer") -> None:
    self.optimizer = get_optimizer(self.cfg, self.model.parameters())
    self.scheduler = get_scheduler(self.cfg.train.scheduler, self.optimizer, self.cfg.train.epochs)


def _cliphash_step_scheduler_for_resume(self: "CLIPHashTrainer") -> None:
    self.scheduler.step()


def _cliphash_report_lr_and_step_scheduler(self: "CLIPHashTrainer") -> None:
    self.writer.add_scalar(T.lr(T.CLIP), self.scheduler.get_last_lr()[0], self.current_epoch + 0.5)
    self.scheduler.step()


def _cliphash_get_model_params_for_tensorboard(self: "CLIPHashTrainer") -> dict[str, HPARAM]:
    return dict(
        model_backbone=self.cfg.model.clip_model,
        model_backbone_pretrained=self.cfg.model.clip_model_pretrained,
        model_hash=self.cfg.model.hash.stringify(),
        text_embedder=self.model.text_embedder_name,
        text_seqlen=self.model.text_seqlen,
    )


class AlternatingCLIPHashTrainer(
    AlternatingHashLearningTrainer[CLIPHash, dsh.config.model.CLIPHashModelConfig, dsh.config.data.CrossModalDatasetConfig]
):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[dsh.config.model.CLIPHashModelConfig, dsh.config.data.CrossModalDatasetConfig],
        model: CLIPHash,
        writer: dsh.utils.tensorboard.Writer,
    ):
        super().__init__(cfg, model, writer)
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler

    construct_optimizer_and_scheduler = _cliphash_construct_optimizer_and_scheduler
    _step_scheduler_for_resume = _cliphash_step_scheduler_for_resume
    _report_lr_and_step_scheduler = _cliphash_report_lr_and_step_scheduler
    _get_model_params_for_tensorboard = _cliphash_get_model_params_for_tensorboard

    def _set_model_train_vision(self) -> torch.optim.Optimizer:
        self.model.train()
        return self.optimizer

    def _set_model_train_text(self) -> torch.optim.Optimizer:
        self.model.train()
        return self.optimizer


class SimultaneousCLIPHashTrainer(
    SimultaneousHashLearningTrainer[CLIPHash, dsh.config.model.CLIPHashModelConfig, dsh.config.data.CrossModalDatasetConfig]
):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[dsh.config.model.CLIPHashModelConfig, dsh.config.data.CrossModalDatasetConfig],
        model: CLIPHash,
        writer: dsh.utils.tensorboard.Writer,
    ):
        super().__init__(cfg, model, writer)
        self.optimizer: torch.optim.Optimizer
        self.scheduler: torch.optim.lr_scheduler.LRScheduler

    construct_optimizer_and_scheduler = _cliphash_construct_optimizer_and_scheduler
    _step_scheduler_for_resume = _cliphash_step_scheduler_for_resume
    _report_lr_and_step_scheduler = _cliphash_report_lr_and_step_scheduler
    _get_model_params_for_tensorboard = _cliphash_get_model_params_for_tensorboard

    def _set_model_train(self) -> tuple[torch.optim.Optimizer, ...]:
        self.model.train()
        return (self.optimizer,)


CLIPHashTrainer = AlternatingCLIPHashTrainer | SimultaneousCLIPHashTrainer


class CLIPHashInferrer(
    HashLearningInferrer[CLIPHash, dsh.config.model.CLIPHashModelConfig, dsh.config.data.CrossModalDatasetConfig]
):
    pass
