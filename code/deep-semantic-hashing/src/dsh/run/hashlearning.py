from abc import ABC, abstractmethod
import h5py
import numpy as np
import os
import torch
from typing import Generic, Literal, TypeVar


from dsh.config.data import CDC, CrossModalDatasetConfig
from dsh.config.model import HMC
import dsh.config.run
import dsh.config.train

from dsh.data.dataset import CrossModalDataset, DatasetMode

from dsh.model.modelbase import CrossModalTopLevelModelBase

from dsh.run.inferrer import Inferrer
from dsh.run.trainer import Trainer

from dsh.utils.adapter import CrossModalDatasetToModelAdapter, DatasetToModelAdapter
from dsh.utils.eventsystem import EventSystem
from dsh.utils.functions import create_cosine_similarity_cross_matrix, create_similarity_cross_matrix
from dsh.utils.logger import Logger, LogLevel
from dsh.utils.progress import tqdm
from dsh.utils.selector import get_datasets, get_dataloader, get_device, get_loss, get_sign_function
import dsh.utils.serialization
from dsh.utils.stopwatch import GlobalProfiler, StopWatch
import dsh.utils.tensorboard
from dsh.utils.trace import VectorStatisticsTracker
from dsh.utils.types import Constants, CrossModalData, DatasetForHashLearningInfo, HPARAM, T

M = TypeVar("M", bound=CrossModalTopLevelModelBase)


def _hashlearning_get_statedict(model: CrossModalTopLevelModelBase) -> object:
    return model.state_dict()


def _hashlearning_apply_statedict(model: CrossModalTopLevelModelBase, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint, strict=True)


class HashLearningTrainer(Trainer, ABC, Generic[M, HMC, CDC]):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[HMC, CDC],
        model: M,
        writer: dsh.utils.tensorboard.Writer,
    ):
        super().__init__(cfg.train.epochs, cfg.env.retain_checkpoints)
        self.model = model
        self.cfg = cfg
        self.writer = writer
        self.vector_statistics_tracker: VectorStatisticsTracker | None = None

        self.dataset_metadata, (self.train_dataset, self.val_dataset) = get_datasets(
            cfg,
            model.get_adapter(),
            DatasetMode.TRAIN,
            DatasetMode.VAL,
        )
        self.train_dataloader = get_dataloader(cfg, self.train_dataset, shuffle=True, drop_last=cfg.train.drop_last)
        self.val_dataloader = get_dataloader(cfg, self.val_dataset, shuffle=False, drop_last=cfg.train.drop_last)

        self.device = get_device(cfg.env)
        self.sign_function = get_sign_function(cfg.model.sign_function)

        self.construct_optimizer_and_scheduler()
        self.construct_criterion()

    @abstractmethod
    def construct_optimizer_and_scheduler(self) -> None: ...

    @abstractmethod
    def construct_criterion(self): ...

    def save(self, mid_epoch: bool):
        """Save model parameters and profiler data."""
        self.save_model(mid_epoch)
        self.save_profiler_data()
        self.save_statistics()

    def save_model(self, mid_epoch: bool) -> str:
        """Save model parameters."""
        with GlobalProfiler().step("Save") as step_save:
            # save model parameters
            path = self.cfg.env.resolve(self.cfg.env.model_path)
            if mid_epoch:
                path = path.replace(".pth", ".mid.pth")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self._get_statedict(), path)
            Logger().info(f"[FIT] Model saved to {path}.")
        self.writer.add_scalar(T.time(T.SAVE), step_save.duration, self.current_epoch + 0.5)
        return path

    def save_profiler_data(self):
        # save profiler data
        path = os.path.join(self.cfg.env.resolve(self.cfg.env.log_path), "profiler.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        GlobalProfiler().save(path, mode="overwrite")
        Logger().info(f"[FIT] Profiler data saved to {path}.")

    def save_statistics(self):
        if self.cfg.env.trace_vector_statistics and self.vector_statistics_tracker is not None:
            path = os.path.join(self.cfg.env.resolve(self.cfg.env.log_path), "statistics.csv")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.vector_statistics_tracker.save(path)
            Logger().info(f"[FIT] Statistics saved to {path}.")

    def save_metadata(self):
        config_path = self.cfg.env.resolve(self.cfg.env.config_path)
        datasetinfo_path = self.cfg.env.resolve(Constants.Files.Metadata.DatasetInformation)
        if self.cfg.env.resume_epoch != None:
            config_path = config_path.replace(".json", f".{self.cfg.env.resume_epoch}.json")
            datasetinfo_path = datasetinfo_path.replace(".json", f".{self.cfg.env.resume_epoch}.json")
        # save train config for future inspection
        self.cfg.dumpf(config_path)
        # save dataset subset information for future inspection
        dsh.utils.serialization.Serializer[DatasetForHashLearningInfo].save(self.dataset_metadata, datasetinfo_path)
        Logger().info(f"[FIT] Dataset Metadata saved to {datasetinfo_path}.")

    def _handle_resume(self) -> int | Literal[False]:
        if self.cfg.env.resume_epoch == None or len(self.cfg.env.resume_path) == 0:
            return False
        else:
            assert (
                self.cfg.env.resume_epoch != None and len(self.cfg.env.resume_path) > 0
            ), "resume_epoch and resume_path must be specified"
            # Load the model from the checkpoint
            self._apply_statedict(self.cfg.env.resolve(self.cfg.env.resume_path))
            for _ in range(self.cfg.env.resume_epoch):
                self._step_scheduler_for_resume()
            return self.cfg.env.resume_epoch

    def _get_statedict(self) -> object:
        return _hashlearning_get_statedict(self.model)

    def _apply_statedict(self, path: str) -> None:
        _hashlearning_apply_statedict(self.model, path, self.device)

    @abstractmethod
    def _step_scheduler_for_resume(self) -> None: ...

    def _prepare_fit(self) -> None:
        """Prepare training the model."""
        self.cfg.env.add_epoch_resolver(lambda: self.current_epoch)

        num_batches_train = len(self.train_dataloader)
        num_batches_val = len(self.val_dataloader)
        Logger().info(f"[FIT] Number of epochs: {self.cfg.train.epochs}.")
        Logger().info(f"[FIT] Training {num_batches_train} batches of size {self.cfg.train.batch_size} per epoch.")
        Logger().info(f"[FIT] Validating with {num_batches_val} batches per epoch.")
        Logger().info(f"[FIT] Device: {self.device}.")
        self._configure_tensorboard_custom_scalars()
        self._add_tensorboard_train_config()

        if self.cfg.env.measure_time_precisely:
            training_stopwatch = StopWatch(self.device, "Init", LogLevel.INFO)
        else:
            training_stopwatch = StopWatch(None, "Init", False)
        GlobalProfiler(training_stopwatch)

        self._select_fit_epoch()
        self.model.apply_freezing()

    @abstractmethod
    def _select_fit_epoch(self) -> None: ...

    def _fit_epoch(self) -> None:
        raise RuntimeError("You must call _prepare_fit before calling _fit_epoch")

    def _prepare_fit_epoch(self) -> None:
        for d in [self.train_dataset, self.val_dataset]:
            d.update_augmentation(self.current_epoch)

    def _model_to_device(self) -> None:
        self.model.to(self.device)

    def _model_to_cpu(self) -> None:
        self.model.cpu()

    def _fit(self) -> None:
        if self.cfg.env.trace_vector_statistics:
            with VectorStatisticsTracker(self.model, lambda: self.current_epoch, lambda: self.current_batch) as t:
                self.vector_statistics_tracker = t
                step = self._fit_impl()
        else:
            step = self._fit_impl()

        self.writer.add_scalar(T.time(T.TOTAL), step.duration, self.current_epoch + 1)
        self.save_profiler_data()

    @abstractmethod
    def _report_lr_and_step_scheduler(self) -> None: ...

    def _check_save_epoch(self) -> bool:
        if (self.current_epoch + 1) % self.cfg.train.save_frequency == 0:
            return True
        return False

    def _check_trigger_inference_epoch(self) -> bool:
        if self.cfg.train.early_stopping == None:
            return False
        if (self.current_epoch + 1) % self.cfg.train.early_stopping.infer_frequency == 0:
            return True
        return False

    def _configure_tensorboard_custom_scalars(self) -> None:
        Logger().info(f"[FIT] Configuring custom scalar plots for TensorBoard.")
        self.writer.writer.add_custom_scalars(
            {
                self.model.model_name: {
                    "Losses": [
                        "Multiline",
                        [
                            T.loss(T.TRAIN, T.VISION, T.BATCH),
                            T.loss(T.TRAIN, T.VISION, T.EPOCH),
                            T.loss(T.TRAIN, T.TEXT, T.BATCH),
                            T.loss(T.TRAIN, T.TEXT, T.EPOCH),
                            T.loss(T.TRAIN, T.EPOCH),
                            T.loss(T.EVAL, T.VISION, T.EPOCH),
                            T.loss(T.EVAL, T.TEXT, T.EPOCH),
                            T.loss(T.EVAL, T.EPOCH),
                        ],
                    ],
                    "Timing": [
                        "Multiline",
                        [
                            T.time(T.TRAIN, T.VISION),
                            T.time(T.TRAIN, T.TEXT),
                            T.time(T.TRAIN),
                            T.time(T.SAVE),
                            T.time(T.EVAL),
                        ],
                    ],
                }
            }
        )

    def _add_tensorboard_train_config(self) -> None:
        Logger().info(f"[FIT] Adding train configuration to TensorBoard.")

        optimizer_params: dict[str, HPARAM] = dict(
            optimizer_type=self.cfg.train.optimizer.name,
            optimizer_lr=self.cfg.train.optimizer.learning_rate,
        )
        if isinstance(self.cfg.train.optimizer, dsh.config.train.AdamOptimizer):
            optimizer_params.update(
                dict(
                    optimizer_weight_decay=self.cfg.train.optimizer.weight_decay,
                    optimizer_beta1=self.cfg.train.optimizer.betas[0],
                    optimizer_beta2=self.cfg.train.optimizer.betas[1],
                    optimizer_eps=self.cfg.train.optimizer.eps,
                )
            )
        elif isinstance(self.cfg.train.optimizer, dsh.config.train.SGDOptimizer):
            optimizer_params.update(
                dict(
                    optimizer_weight_decay=self.cfg.train.optimizer.weight_decay,
                    optimizer_momentum=self.cfg.train.optimizer.momentum,
                    optimizer_dampening=self.cfg.train.optimizer.dampening,
                )
            )
        else:
            raise NotImplementedError(f"Optimizer {type(self.cfg.train.optimizer)} not implemented.")

        data_params: dict[str, HPARAM] = dict(
            dataset=self.dataset_metadata.dataset_name,
            data_train_len=len(self.dataset_metadata.train_indices),
            data_val_len=len(self.dataset_metadata.validation_indices),
            data_num_labels=len(self.dataset_metadata.labels),
            data_train_aug=self.dataset_metadata.train_augmentations,
        )

        model_params = self._get_model_params_for_tensorboard()
        loss_params = self._get_loss_params_for_tensorboard()

        self.writer.add_hparams(
            hparam_dict=dict(
                batch_size=self.cfg.train.batch_size,
                train_mode=self.cfg.train.train_mode.value,
                **loss_params,
                **optimizer_params,
                sign_function=self.cfg.model.sign_function.value,
                hash_length=self.cfg.model.hash_length,
                **model_params,
                **data_params,
            ),
            metric_dict=dict(),
        )

    @abstractmethod
    def _get_model_params_for_tensorboard(self) -> dict[str, HPARAM]: ...
    @abstractmethod
    def _get_loss_params_for_tensorboard(self) -> dict[str, HPARAM]: ...


class AlternatingHashLearningTrainer(HashLearningTrainer[M, HMC, CDC], ABC, Generic[M, HMC, CDC]):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[HMC, CDC],
        model: M,
        writer: dsh.utils.tensorboard.Writer,
    ):
        assert dsh.config.train.TrainMode.is_alternating(
            cfg.train.train_mode
        ), f"Using {self.__class__.__name__} requires alternating training mode."
        super().__init__(cfg, model, writer)

    def construct_criterion(self):
        if isinstance(self.cfg.train.loss, dsh.config.train.TDSRDHPaperLoss):
            self.criterion = get_loss(self.cfg.train.loss, self.sign_function)
        else:
            raise ValueError(f"Unsupported loss configuration {self.cfg.train.loss}")

    def _select_fit_epoch(self) -> None:
        assert dsh.config.train.TrainMode.is_alternating(self.cfg.train.train_mode)
        match self.cfg.train.train_mode:
            case dsh.config.train.TrainMode.IMG_TEXT_PER_EPOCH:
                self._fit_epoch = self._fit_img_text_per_epoch
            case dsh.config.train.TrainMode.IMG_TEXT_PER_BATCH:
                self._fit_epoch = self._fit_img_text_per_batch
            case _:
                raise NotImplementedError(f"Train mode {self.cfg.train.train_mode} is not implemented.")

    def _fit_img_text_per_epoch(self) -> None:
        num_batches_train = len(self.train_dataloader)
        vision_loss = 0.0
        text_loss = 0.0

        with GlobalProfiler().step("Train") as step_train:
            with GlobalProfiler().step("Vision") as step_vision:
                optimizer_vision = self._set_model_train_vision()
                for self.current_batch, batch in enumerate(self.train_dataloader):
                    if self.current_batch == 0:  # first batch gets displayed at TensorBoard
                        if (
                            self.cfg.env.display_images_every_n_epochs > 0
                            and self.current_epoch % self.cfg.env.display_images_every_n_epochs == 0
                        ):
                            self.writer.add_images(T.img(T.TRAIN, T.EPOCH), batch.image, self.current_epoch, unnormalize=True)
                    batch_loss = self._fit_one_batch("image", optimizer_vision, batch)
                    vision_loss += batch_loss
                    self.writer.add_scalar(
                        T.loss(T.TRAIN, T.VISION, T.BATCH),
                        batch_loss,
                        self.current_epoch + self.current_batch / num_batches_train,
                    )
                self.writer.add_scalar(
                    T.loss(T.TRAIN, T.VISION, T.EPOCH), vision_loss / num_batches_train, self.current_epoch + 0.5
                )
            self.writer.add_scalar(T.time(T.TRAIN, T.VISION), step_vision.duration, self.current_epoch + 0.5)

            with GlobalProfiler().step("Text") as step_text:
                optimizer_text = self._set_model_train_text()
                for self.current_batch, batch in enumerate(self.train_dataloader):
                    batch_loss = self._fit_one_batch("text", optimizer_text, batch)
                    text_loss += batch_loss
                    self.writer.add_scalar(
                        T.loss(T.TRAIN, T.TEXT, T.BATCH),
                        batch_loss,
                        self.current_epoch + self.current_batch / num_batches_train,
                    )
                self.writer.add_scalar(T.loss(T.TRAIN, T.TEXT, T.EPOCH), text_loss / num_batches_train, self.current_epoch + 0.5)
            self.writer.add_scalar(T.time(T.TRAIN, T.TEXT), step_text.duration, self.current_epoch + 0.5)

            self._report_lr_and_step_scheduler()
        self.writer.add_scalar(T.loss(T.TRAIN, T.EPOCH), (vision_loss + text_loss) / num_batches_train, self.current_epoch + 0.5)
        self.writer.add_scalar(T.time(T.TRAIN), step_train.duration, self.current_epoch + 0.5)

    def _fit_img_text_per_batch(self) -> None:
        num_batches_train = len(self.train_dataloader)
        with GlobalProfiler().step(f"Epoch {self.current_epoch}"):
            vision_loss = 0.0
            text_loss = 0.0

            with GlobalProfiler().step("Train") as step_train:
                vision_time = 0.0
                text_time = 0.0
                for self.current_batch, batch in enumerate(self.train_dataloader):
                    if self.current_batch == 0:  # first batch gets displayed at TensorBoard
                        if (
                            self.cfg.env.display_images_every_n_epochs > 0
                            and self.current_epoch % self.cfg.env.display_images_every_n_epochs == 0
                        ):
                            self.writer.add_images(T.img(T.TRAIN, T.EPOCH), batch.image, self.current_epoch, unnormalize=True)
                    if self.current_batch % 2 == 0:
                        with GlobalProfiler().step("Vision") as step_vision:
                            optimizer_vision = self._set_model_train_vision()
                            batch_loss = self._fit_one_batch("image", optimizer_vision, batch)
                            vision_loss += batch_loss
                            self.writer.add_scalar(
                                T.loss(T.TRAIN, T.VISION, T.BATCH),
                                batch_loss,
                                self.current_epoch + self.current_batch / num_batches_train,
                            )
                        vision_time += step_vision.duration
                    else:
                        with GlobalProfiler().step("Text") as step_text:
                            optimizer_text = self._set_model_train_text()
                            batch_loss = self._fit_one_batch("text", optimizer_text, batch)
                            text_loss += batch_loss
                            self.writer.add_scalar(
                                T.loss(T.TRAIN, T.TEXT, T.BATCH),
                                batch_loss,
                                self.current_epoch + self.current_batch / num_batches_train,
                            )
                        text_time += step_text.duration

                self.writer.add_scalar(
                    T.loss(T.TRAIN, T.VISION, T.EPOCH), vision_loss * 2 / num_batches_train, self.current_epoch + 0.5
                )
                self.writer.add_scalar(
                    T.loss(T.TRAIN, T.TEXT, T.EPOCH), text_loss * 2 / num_batches_train, self.current_epoch + 0.5
                )
                self.writer.add_scalar(T.time(T.TRAIN, T.VISION), vision_time, self.current_epoch + 0.5)
                self.writer.add_scalar(T.time(T.TRAIN, T.TEXT), text_time, self.current_epoch + 0.5)

                self._report_lr_and_step_scheduler()
            self.writer.add_scalar(
                T.loss(T.TRAIN, T.EPOCH), (vision_loss + text_loss) / num_batches_train, self.current_epoch + 0.5
            )
            self.writer.add_scalar(T.time(T.TRAIN), step_train.duration, self.current_epoch + 0.5)

    @abstractmethod
    def _set_model_train_vision(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def _set_model_train_text(self) -> torch.optim.Optimizer: ...

    def _fit_one_batch(
        self,
        pred: Literal["image"] | Literal["text"],
        optimizer: torch.optim.Optimizer,
        batch: CrossModalData,
    ) -> float:
        with GlobalProfiler().step("Batch") as s:
            img = batch.image.to(self.device)
            txt = batch.text.to(self.device)
            lbl = batch.label.to(self.device)
            s.record("To_Device")

            sim = create_similarity_cross_matrix(lbl)
            s.record("Similarity_Matrix")

            optimizer.zero_grad()
            s.record("Zero_Grad")
            txt_hash = self.model(("text", txt))
            s.record("Forward_Text")
            img_hash = self.model(("image", img))
            s.record("Forward_Image")

            if pred == "image":
                loss = self.criterion(sim, img_hash, txt_hash.detach(), pred)
            else:
                loss = self.criterion(sim, txt_hash, img_hash.detach(), pred)
            s.record("Loss")

            loss.backward()
            s.record("Backward")

            optimizer.step()
            s.record("Step")
        return loss.cpu().item()

    def _eval_epoch(self) -> None:
        num_batches_val = len(self.val_dataloader)
        vision_loss = 0.0
        text_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            with GlobalProfiler().step("Eval") as step_eval:
                # evaluate the whole model on the validation set
                for self.current_batch, batch in enumerate(self.val_dataloader):
                    batch_vision_loss, batch_text_loss = self._eval_one_batch(batch)
                    vision_loss += batch_vision_loss
                    text_loss += batch_text_loss
            self.writer.add_scalar(T.loss(T.EVAL, T.VISION, T.EPOCH), vision_loss / num_batches_val, self.current_epoch + 0.5)
            self.writer.add_scalar(T.loss(T.EVAL, T.TEXT, T.EPOCH), text_loss / num_batches_val, self.current_epoch + 0.5)
            self.writer.add_scalar(T.loss(T.EVAL, T.EPOCH), (vision_loss + text_loss) / num_batches_val, self.current_epoch + 0.5)
            self.writer.add_scalar(T.time(T.EVAL), step_eval.duration, self.current_epoch + 0.5)
            EventSystem()[Constants.Events.Train.EpochEvalLossUpdate](
                self, self.current_epoch, (vision_loss + text_loss) / num_batches_val
            )

    def _eval_one_batch(self, batch: CrossModalData) -> tuple[float, float]:
        with GlobalProfiler().step("Batch") as s:
            sim = create_similarity_cross_matrix(batch.label)
            s.record("Similarity_Matrix")

            img = batch.image.to(self.device)
            txt = batch.text.to(self.device)
            sim = sim.to(self.device)
            s.record("To_Device")

            txt_hash = self.model(("text", txt))
            s.record("Forward_Text")
            img_hash = self.model(("image", img))
            s.record("Forward_Image")

            vision_loss = self.criterion(sim, img_hash, txt_hash, "image")
            text_loss = self.criterion(sim, txt_hash, img_hash, "text")
            s.record("Loss")
        return vision_loss.cpu().item(), text_loss.cpu().item()

    def _get_loss_params_for_tensorboard(self) -> dict[str, HPARAM]:
        return dict(loss_type=self.criterion.__class__.__name__, loss_config=self.cfg.train.loss.stringify())


class SimultaneousHashLearningTrainer(HashLearningTrainer[M, HMC, CDC], ABC, Generic[M, HMC, CDC]):
    def __init__(
        self,
        cfg: dsh.config.run.TrainRunConfig[HMC, CDC],
        model: M,
        writer: dsh.utils.tensorboard.Writer,
    ):
        assert (
            cfg.train.train_mode == dsh.config.train.TrainMode.SIMULTANEOUS
        ), f"Using {self.__class__.__name__} requires simultaneous training mode."
        super().__init__(cfg, model, writer)

    def construct_criterion(self):
        loss = self.cfg.train.loss
        if isinstance(loss, dsh.config.train.SCHPaperLoss) or isinstance(loss, dsh.config.train.OurLoss):
            self.criterion = get_loss(loss, self.sign_function)
        else:
            raise ValueError(f"Unsupported loss configuration {loss}")

    def _select_fit_epoch(self) -> None:
        assert dsh.config.train.TrainMode.is_simultaneous(self.cfg.train.train_mode)
        match self.cfg.train.train_mode:
            case dsh.config.train.TrainMode.SIMULTANEOUS:
                self._fit_epoch = self._fit_simultaneous
            case _:
                raise NotImplementedError(f"Train mode {self.cfg.train.train_mode} is not implemented.")

    def _fit_simultaneous(self) -> None:
        num_batches_train = len(self.train_dataloader)
        total_loss = 0.0

        with GlobalProfiler().step("Train") as step_train:
            optimizer = self._set_model_train()
            for self.current_batch, batch in enumerate(self.train_dataloader):
                if self.current_batch == 0:  # first batch gets displayed at TensorBoard
                    if (
                        self.cfg.env.display_images_every_n_epochs > 0
                        and self.current_epoch % self.cfg.env.display_images_every_n_epochs == 0
                    ):
                        self.writer.add_images(T.img(T.TRAIN, T.EPOCH), batch.image, self.current_epoch, unnormalize=True)
                batch_loss = self._fit_one_batch(optimizer, batch)
                total_loss += batch_loss
                self.writer.add_scalar(
                    T.loss(T.TRAIN, T.BATCH),
                    batch_loss,
                    self.current_epoch + self.current_batch / num_batches_train,
                )
            self._report_lr_and_step_scheduler()
        self.writer.add_scalar(T.loss(T.TRAIN, T.EPOCH), total_loss / num_batches_train, self.current_epoch + 0.5)
        self.writer.add_scalar(T.time(T.TRAIN), step_train.duration, self.current_epoch + 0.5)

    @abstractmethod
    def _set_model_train(self) -> tuple[torch.optim.Optimizer, ...]: ...

    def _fit_one_batch(
        self,
        optimizer: tuple[torch.optim.Optimizer, ...],
        batch: CrossModalData,
    ) -> float:
        with GlobalProfiler().step("Batch") as s:
            img = batch.image.to(self.device)
            txt = batch.text.to(self.device)
            lbl = batch.label.to(self.device)
            s.record("To_Device")

            sim = create_cosine_similarity_cross_matrix(lbl, 1.0)
            s.record("Similarity_Matrix")

            for o in optimizer:
                o.zero_grad()
            s.record("Zero_Grad")
            txt_hash = self.model(("text", txt))
            s.record("Forward_Text")
            img_hash = self.model(("image", img))
            s.record("Forward_Image")

            loss = self.criterion(sim, (img_hash, txt_hash))
            s.record("Loss")

            loss.backward()
            s.record("Backward")

            for o in optimizer:
                o.step()
            s.record("Step")
        return loss.cpu().item()

    def _eval_epoch(self) -> None:
        num_batches_val = len(self.val_dataloader)
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            with GlobalProfiler().step("Eval") as step_eval:
                # evaluate the whole model on the validation set
                for self.current_batch, batch in enumerate(self.val_dataloader):
                    batch_loss = self._eval_one_batch(batch)
                    total_loss += batch_loss
            self.writer.add_scalar(T.loss(T.EVAL, T.EPOCH), total_loss / num_batches_val, self.current_epoch + 0.5)
            self.writer.add_scalar(T.time(T.EVAL), step_eval.duration, self.current_epoch + 0.5)
            EventSystem()[Constants.Events.Train.EpochEvalLossUpdate](self, self.current_epoch, total_loss / num_batches_val)

    def _eval_one_batch(self, batch: CrossModalData) -> float:
        with GlobalProfiler().step("Batch") as s:
            sim = create_cosine_similarity_cross_matrix(batch.label, 1.0)
            s.record("Similarity_Matrix")

            img = batch.image.to(self.device)
            txt = batch.text.to(self.device)
            sim = sim.to(self.device)
            s.record("To_Device")

            txt_hash = self.model(("text", txt))
            s.record("Forward_Text")
            img_hash = self.model(("image", img))
            s.record("Forward_Image")

            loss = self.criterion(sim, (img_hash, txt_hash))
            s.record("Loss")
        return loss.cpu().item()

    def _get_loss_params_for_tensorboard(self) -> dict[str, HPARAM]:
        return dict(loss_type=self.criterion.__class__.__name__, loss_config=self.cfg.train.loss.stringify())


class HashLearningInferrer(
    Inferrer[M, HMC, CrossModalDataset[CrossModalDatasetConfig, CrossModalData], CDC, DatasetForHashLearningInfo],
    Generic[M, HMC, CDC],
):
    def __init__(self, cfg: dsh.config.run.InferenceRunConfig[HMC, CDC]):
        super().__init__(cfg)
        self.device = get_device(cfg.env)
        self.sign_function = get_sign_function(cfg.model.sign_function)
        self.current_batch: int = 0

    def _load_dataset(
        self, adapter: DatasetToModelAdapter
    ) -> tuple[DatasetForHashLearningInfo, CrossModalDataset[CrossModalDatasetConfig, CrossModalData]]:
        assert isinstance(adapter, CrossModalDatasetToModelAdapter)
        metadata, (dataset,) = get_datasets(self.cfg, adapter, DatasetMode.FULL)
        return metadata, dataset

    def _load_state_dict(self, model: M, path: str) -> None:
        _hashlearning_apply_statedict(model, path, self.device)

    def _infer(self, model: M, outfile: h5py.File) -> None:
        if self.cfg.env.trace_vector_statistics:
            path = self.cfg.env.resolve("${OUTPUT_PATH.DIR}/statistics.${EPOCH}.csv")
            with VectorStatisticsTracker(model, lambda: 0, lambda: self.current_batch) as t:
                self._infer_inner(model, outfile)
            t.save(path)
        else:
            self._infer_inner(model, outfile)

    def _infer_inner(self, model: M, outfile: h5py.File) -> None:
        # load dataset info
        train_di_path = self.cfg.env.resolve(Constants.Files.Metadata.DatasetInformation)
        train_dataset_info = dsh.utils.serialization.Serializer[DatasetForHashLearningInfo].load(
            train_di_path, DatasetForHashLearningInfo
        )
        if train_dataset_info.dataset_name == self.dataset_metadata.dataset_name:
            # we can use some of the train dataset info to determine subset membership
            train_indices = sorted(train_dataset_info.train_indices)
            val_indices = sorted(train_dataset_info.validation_indices)
            val_retrieval_indices = sorted(train_dataset_info.validation_retrieval_indices)
            test_indices = sorted(train_dataset_info.test_indices)
            test_retrieval_indices = sorted(train_dataset_info.test_retrieval_indices)
            if len(val_indices) == 0 or len(val_retrieval_indices) == 0:
                Logger().warning(
                    f"[INF] Validation query and/or validation retrieval indices are empty! #vq: {len(val_indices)}, #vr: {len(val_retrieval_indices)}"
                )
            if len(test_indices) == 0 or len(test_retrieval_indices) == 0:
                Logger().error(
                    f"[INF] Error: test retrieval and/or test query indices empty! #tq: {len(test_indices)}, #tr: {len(test_retrieval_indices)}."
                )

            def fill_subset_membership(dataset: h5py.Dataset) -> None:
                ssm = np.zeros(dataset.shape, dtype=np.bool)
                ssm[train_indices, Constants.H5.Sets.ColumnIndex.Train] = True
                ssm[test_indices, Constants.H5.Sets.ColumnIndex.Test] = True
                ssm[test_retrieval_indices, Constants.H5.Sets.ColumnIndex.TestRetrieval] = True
                ssm[val_indices, Constants.H5.Sets.ColumnIndex.Validation] = True
                ssm[val_retrieval_indices, Constants.H5.Sets.ColumnIndex.ValidationRetrieval] = True
                ssm[:, Constants.H5.Sets.ColumnIndex.Unknown] = ~ssm.any(axis=1)
                dataset[:, :] = ssm

        else:
            # subset membership cannot be determined therefore it is unknown
            Logger().warning(f"[INF] Subset membership is unknown as train dataset and inference dataset differ.")

            def fill_subset_membership(dataset: h5py.Dataset) -> None:
                ssm = np.zeros(dataset.shape, dtype=np.bool)
                ssm[:, Constants.H5.Sets.ColumnIndex.Unknown] = True
                dataset[:, :] = ssm

        dl = get_dataloader(self.cfg, self.dataset, shuffle=False, drop_last=False)
        n = self.dataset.max_number_of_samples  # independent of any dataset sampling strategy
        k = self.cfg.model.hash_length
        l = self.dataset.label_dim

        ihb = outfile.create_dataset(Constants.H5.Image.BoolHash, (n, k), dtype=np.bool)
        ihf = outfile.create_dataset(Constants.H5.Image.FloatHash, (n, k), dtype=np.float32)
        thb = outfile.create_dataset(Constants.H5.Text.BoolHash, (n, k), dtype=np.bool)
        thf = outfile.create_dataset(Constants.H5.Text.FloatHash, (n, k), dtype=np.float32)
        lab = outfile.create_dataset(Constants.H5.Labels, (n, l), dtype=np.bool)
        ssm = outfile.create_dataset(Constants.H5.Sets.SubsetMembership, (n, Constants.H5.Sets.NumberOfColumns), dtype=np.bool)

        fill_subset_membership(ssm)

        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for self.current_batch, batch in enumerate(tqdm(dl, desc=self.cfg.env.resolve("Epoch ${EPOCH}"))):
                img = batch.image.to(self.device)
                txt = batch.text.to(self.device)

                img_hash = model(("image", img)).cpu()
                txt_hash = model(("text", txt)).cpu()

                idx = batch.index.numpy()
                # h5py does not support arbitrary indexing; we need to sort the indices first to ensure the correct order is maintained
                sorted_indices = idx.argsort(kind="stable")  # fast for already sorted data, which should be the case here

                idx = idx[sorted_indices]
                img_hash_f = img_hash.numpy()[sorted_indices]
                img_hash_b = torch.where(self.sign_function(img_hash) > 0, True, False).numpy()[sorted_indices]
                txt_hash_f = txt_hash.numpy()[sorted_indices]
                txt_hash_b = torch.where(self.sign_function(txt_hash) > 0, True, False).numpy()[sorted_indices]
                lbl = batch.label.numpy().astype(bool)[sorted_indices]

                ihf[idx] = img_hash_f
                ihb[idx] = img_hash_b
                thf[idx] = txt_hash_f
                thb[idx] = txt_hash_b
                lab[idx] = lbl

            outfile.flush()
