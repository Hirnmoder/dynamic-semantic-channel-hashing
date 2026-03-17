from typing import Any, Literal, Self
import numpy
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from dsh.utils.augmentation import unnormalize_image_imagenet
from dsh.utils.types import HPARAM, SCALAR, CM, ImageAugmentation


class Writer:
    def __init__(self, underlying_writer: SummaryWriter):
        self._writer = underlying_writer
        self._step_offset = 0.0
        self.unnormalize_image = unnormalize_image_imagenet

    @property
    def writer(self):
        return self._writer

    @property
    def unnormalize_image(self):
        return self._unnormalize_image

    @unnormalize_image.setter
    def unnormalize_image(self, value: ImageAugmentation):
        self._unnormalize_image = value

    def with_step_offset(self, offset: float) -> "WriterStepOffset":
        return WriterStepOffset(self, offset)

    def add_hparams(
        self,
        hparam_dict: dict[str, HPARAM],
        metric_dict: dict[str, SCALAR],
        hparam_domain_discrete: None | dict[str, list[HPARAM]] = None,
        step: float | None = None,
    ):
        # custom implementation to allow tensorboard to display hparams of a run together with the metrics of that run
        step = int((self._step_offset + step) * 1000.0) if step is not None else None
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        fw = self._writer._get_file_writer()
        fw.add_summary(exp, step)
        fw.add_summary(ssi, step)
        fw.add_summary(sei, step)
        for k, v in metric_dict.items():
            self._writer.add_scalar(k, v, step)

    def add_scalar(self, tag: str, scalar_value: SCALAR, step: float, walltime=None, new_style=False, double_precision=False):
        self.writer.add_scalar(
            tag,
            scalar_value,
            int((self._step_offset + step) * 1000.0),
            walltime=walltime,
            new_style=new_style,
            double_precision=double_precision,
        )

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict[str, SCALAR], step: float, walltime=None):
        self.writer.add_scalars(
            main_tag,
            tag_scalar_dict,
            int((self._step_offset + step) * 1000.0),
            walltime=walltime,
        )

    def add_histogram(
        self,
        tag: str,
        values: torch.Tensor | numpy.ndarray,
        step: float | None = None,
        bins="tensorflow",
        walltime=None,
        max_bins=None,
    ):
        self.writer.add_histogram(
            tag,
            values,
            int((self._step_offset + step) * 1000.0) if step is not None else None,
            bins=bins,
            walltime=walltime,
            max_bins=max_bins,
        )

    def add_tensor(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: float,
        walltime=None,
    ):
        self.writer.add_tensor(
            tag,
            tensor,
            int((self._step_offset + step) * 1000.0),
            walltime=walltime,
        )

    def add_image(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: float,
        walltime=None,
        dataformats="CHW",
        unnormalize: bool = False,
    ):
        if unnormalize:
            img_tensor = self.unnormalize_image(img_tensor)
        self.writer.add_image(
            tag,
            img_tensor,
            int((self._step_offset + step) * 1000.0),
            walltime=walltime,
            dataformats=dataformats,
        )

    def add_images(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: float,
        walltime=None,
        dataformats="NCHW",
        unnormalize: bool = False,
    ):
        if unnormalize:
            img_tensor = self.unnormalize_image(img_tensor)
        self.writer.add_images(
            tag,
            img_tensor,
            int((self._step_offset + step) * 1000.0),
            walltime=walltime,
            dataformats=dataformats,
        )

    def add_video(self, tag: str, vid_tensor: torch.Tensor, step: float, fps=4, walltime=None):
        self.writer.add_video(
            tag,
            vid_tensor,
            int((self._step_offset + step) * 1000.0),
            fps=fps,
            walltime=walltime,
        )

    def add_audio(self, tag: str, snd_tensor: torch.Tensor, step: float, sample_rate=44100, walltime=None):
        self.writer.add_audio(
            tag,
            snd_tensor,
            int(step * 1000.0),
            sample_rate=sample_rate,
            walltime=walltime,
        )

    def add_text(self, tag: str, text_string: str, step: float, walltime=None):
        self.writer.add_text(tag, text_string, int((self._step_offset + step) * 1000.0), walltime=walltime)

    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Any | list[Any],
        verbose: bool = False,
        use_strict_trace: bool = True,
    ):
        self.writer.add_graph(model, input_to_model, verbose=verbose, use_strict_trace=use_strict_trace)

    def add_embedding(
        self,
        mat: torch.Tensor | numpy.ndarray,
        metadata: list | None = None,
        label_img: torch.Tensor | None = None,
        step: float | None = None,
        tag="default",
        metadata_header=None,
    ):
        self.writer.add_embedding(
            mat,
            metadata,
            label_img,
            int((self._step_offset + step) * 1000.0) if step is not None else None,
            tag,
            metadata_header=metadata_header,
        )

    def add_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor | numpy.ndarray,
        predictions: torch.Tensor | numpy.ndarray,
        step: float | None = None,
        num_thresholds: int = 127,
        label_weights: torch.Tensor | numpy.ndarray | None = None,
        walltime=None,
    ):
        self.writer.add_pr_curve(
            tag,
            labels,
            predictions,
            int((self._step_offset + step) * 1000.0) if step is not None else None,
            num_thresholds=num_thresholds,
            weights=label_weights,
            walltime=walltime,
        )

    def add_pr_curve_raw(
        self,
        tag: str,
        cfs: dict[int, CM] | dict[float, CM],
        max_number_of_entries: int,
        step: float | None = None,
        walltime=None,
    ):
        true_positive_counts = numpy.array([cf.true_positives for cf in cfs.values()])
        false_positive_counts = numpy.array([cf.false_positives for cf in cfs.values()])
        true_negative_counts = numpy.array([cf.true_negatives for cf in cfs.values()])
        false_negative_counts = numpy.array([cf.false_negatives for cf in cfs.values()])
        precision = numpy.array([cf.precision for cf in cfs.values()])
        recall = numpy.array([cf.recall for cf in cfs.values()])
        # TensorBoard does not support threshold values, but only the number of thresholds and spaces them linearly.
        # This is extremely stupid and not useful for our purposes.
        # We work around that limitation and try to use the maximum number of thresholds as the number of bits in the hash code.
        thresholds = max(max_number_of_entries, len(cfs))
        self.writer.add_pr_curve_raw(
            tag,
            true_positive_counts[::-1],  # TensorBoard needs it in reverse for whatever reason
            false_positive_counts[::-1],  # TensorBoard needs it in reverse for whatever reason
            true_negative_counts[::-1],  # TensorBoard needs it in reverse for whatever reason
            false_negative_counts[::-1],  # TensorBoard needs it in reverse for whatever reason
            precision[::-1],  # TensorBoard needs it in reverse for whatever reason
            recall[::-1],  # TensorBoard needs it in reverse for whatever reason
            int((self._step_offset + step) * 1000) if step is not None else None,
            thresholds,
            walltime=walltime,
        )


class WriterStepOffset:
    def __init__(self, writer: Writer, step_offset: float):
        self._writer = writer
        self._step_offset = step_offset
        self._previous_step_offset: float | None = None

    @property
    def writer(self) -> Writer:
        return self._writer

    @property
    def step_offset(self) -> float:
        return self._step_offset

    def __enter__(self) -> Self:
        self._previous_step_offset = self.writer._step_offset
        self.writer._step_offset = self.step_offset
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        if self._previous_step_offset is not None:
            self.writer._step_offset = self._previous_step_offset
        self._previous_step_offset = None
        return False
