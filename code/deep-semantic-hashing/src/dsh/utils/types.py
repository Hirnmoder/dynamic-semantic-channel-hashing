from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from PIL.Image import Image
from typing import Any, Callable, Final, Generic, Iterable, Literal, Protocol, TypeVar, Union, runtime_checkable
import torch

__all__ = [
    "Data",
    "CrossModalData",
    "ExtendedCrossModalData",
    "Output",
    "SCALAR",
    "HPARAM",
    "TextInput",
    "ImageInput",
    "FullModelInput",
    "TextRawInput",
    "ImageRawInput",
    "FullModelRawInput",
    "I",
    "FullModelToModalAugmentation",
    "GenericAugmentation",
    "DynamicAugmentation",
    "ImageAugmentation",
    "TextAugmentation",
    "TensorAugmentation",
    "NoopTensorAugmentation",
    "DynamicImageAugmentation",
    "DynamicTextAugmentation",
    "ModelDataPreprocessor",
    "RawImagePreprocessor",
    "RawTextPreprocessor",
    "EarlyStoppingInfo",
    "Quantize",
    "MetricEvalSet",
    "T",
    "Constants",
    "DatasetInfo",
    "DatasetForHashLearningInfo",
    "DSI",
    "CM",
]


class Data:
    pass


class CrossModalData(Data):
    def __init__(self, index: torch.Tensor, image: "ImageInput", text: "TextInput", label: torch.Tensor):
        self.index = index
        self.image = image
        self.text = text
        self.label = label

    @staticmethod
    def collate(batch: list["CrossModalData"]) -> "CrossModalData":
        return CrossModalData(
            index=torch.stack([x.index for x in batch]),
            image=torch.stack([x.image for x in batch]),
            text=torch.stack([x.text for x in batch]),
            label=torch.stack([x.label for x in batch]),
        )


class ExtendedCrossModalData(CrossModalData):
    def __init__(
        self, index: torch.Tensor, image: torch.Tensor, text: torch.Tensor, label: torch.Tensor, image_raw: Image, text_str: str
    ):
        super().__init__(index, image, text, label)
        self.image_raw = image_raw
        self.text_str = text_str


Output = torch.Tensor
SCALAR = float | str
HPARAM = SCALAR | bool | int | None


D = TypeVar("D")

TextInput = torch.Tensor
ImageInput = torch.Tensor
FullModelInput = Union[
    tuple[Literal["text"], TextInput],
    tuple[Literal["image"], ImageInput],
]
TextRawInput = torch.Tensor  # text should be embedded before training due to performance reasons
ImageRawInput = Image
FullModelRawInput = Union[
    tuple[Literal["text"], TextRawInput],
    tuple[Literal["image"], ImageRawInput],
]

I = TypeVar("I", ImageInput, TextInput)  # Input


class FullModelToModalAugmentation(Generic[I]):
    def __init__(self, mode: Literal["text", "image"], pipeline: Callable[[FullModelInput], FullModelInput]):
        self.mode = mode
        self.pipeline = pipeline

    def __call__(self, data: I) -> I:
        match self.mode:
            case "text":
                assert isinstance(data, TextInput)
                return_mode, return_data = self.pipeline((self.mode, data))
                assert isinstance(return_data, TextInput)
            case "image":
                assert isinstance(data, ImageInput)
                return_mode, return_data = self.pipeline((self.mode, data))
                assert isinstance(return_data, ImageInput)
            case _:
                raise NotImplementedError(f"Unknown mode {self.mode}")
        assert return_mode == self.mode
        return return_data


@runtime_checkable
class DynamicAugmentation(Protocol):
    def update_augmentation(self, epoch: int) -> None: ...


@runtime_checkable
class GenericAugmentation(Protocol[D]):
    def __call__(self, input: D, /) -> D: ...


@runtime_checkable
class ImageAugmentation(GenericAugmentation[ImageInput], Protocol):
    def __call__(self, image: ImageInput, /) -> ImageInput: ...


@runtime_checkable
class TextAugmentation(GenericAugmentation[TextInput], Protocol):
    def __call__(self, text: TextInput, /) -> TextInput: ...


@runtime_checkable
class TensorAugmentation(GenericAugmentation[torch.Tensor], Protocol):
    def __call__(self, input: torch.Tensor, /) -> torch.Tensor: ...


class NoopTensorAugmentation(TensorAugmentation):
    def __call__(self, input: torch.Tensor, /) -> torch.Tensor:
        return input


@runtime_checkable
class Quantize(Protocol):
    def __call__(self, input: torch.Tensor, /) -> torch.Tensor: ...


@runtime_checkable
class DynamicImageAugmentation(DynamicAugmentation, ImageAugmentation, Protocol): ...


@runtime_checkable
class DynamicTextAugmentation(DynamicAugmentation, TextAugmentation, Protocol): ...


DI = TypeVar("DI", contravariant=True)  # raw data input type
MI = TypeVar("MI")  # model input type


@runtime_checkable
class ModelDataPreprocessor(Generic[DI, MI], Protocol):
    def __call__(self, raw_data: DI, augmentation_pipeline: Callable[[MI], MI]) -> MI: ...


@runtime_checkable
class RawImagePreprocessor(Protocol):
    def __call__(self, raw_image: ImageRawInput, /) -> ImageInput: ...


@runtime_checkable
class RawTextPreprocessor(Protocol):
    def __call__(self, raw_text: TextRawInput, /) -> TextInput: ...


@dataclass
class EarlyStoppingInfo:
    best_epochs: dict[str, int]


def number_of_properties(cls: type) -> int:
    return len([*filter(lambda n: not n.startswith("_"), dir(cls))])


class MetricEvalSet(StrEnum):
    VAL = "val"
    TEST = "test"


class T(StrEnum):
    LOSS = "loss"
    TIME = "time"
    METRIC = "metric"
    LR = "lr"
    IMG = "img"

    TRAIN = "train"
    EVAL = "eval"
    SAVE = "save"
    INFER = "infer"

    VISION = "vision"
    TEXT = "text"
    CLIP = "clip"

    BATCH = "batch"
    EPOCH = "epoch"
    TOTAL = "total"

    HAMMING = "hamming"
    HAMMING_RANKING = "hamming/ranking"
    HASH_LOOKUP = "hamming/lookup"
    PR = "precision_recall"
    ROC_AUC = "roc_auc"
    IMG2TEXT = "img2text"
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    TEXT2TEXT = "text2text"

    def __add__(self, other: Any) -> str:
        if isinstance(other, T):
            return f"{self.value}/{other.value}"
        elif isinstance(other, str):
            return f"{self.value}/{other}"
        raise NotImplementedError()

    def __radd__(self, other: Any) -> str:
        if isinstance(other, T):
            return f"{other.value}/{self.value}"
        elif isinstance(other, str):
            return f"{other}/{self.value}"
        raise NotImplementedError()

    def __iadd__(self, other: Any) -> str:
        raise NotImplementedError()

    @staticmethod
    def loss(*args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.LOSS)

    @staticmethod
    def time(*args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.TIME)

    @staticmethod
    def times(set: MetricEvalSet, *args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.TIME) + set.value

    @staticmethod
    def metric(set: MetricEvalSet, *args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.METRIC + set.value)

    @staticmethod
    def lr(*args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.LR)

    @staticmethod
    def img(*args: "T") -> str:
        return reduce(lambda x, y: x + y, args, T.IMG)


class Constants:
    class H5:
        class Image:
            BoolHash: Final = "Image Hash Bool"
            FloatHash: Final = "Image Hash Float"

        class Text:
            BoolHash: Final = "Text Hash Bool"
            FloatHash: Final = "Text Hash Float"

        Labels: Final = "Labels"

        class Sets:
            SubsetMembership: Final = "Subset Membership"

            class ColumnIndex:
                Unknown: Final = 0
                Train: Final = 1
                Test: Final = 2
                TestRetrieval: Final = 3
                Validation: Final = 4
                ValidationRetrieval: Final = 5

            NumberOfColumns: Final = number_of_properties(ColumnIndex)

    class Files:
        class Metadata:
            DatasetInformation: Final = r"${CONFIG_PATH.DIR}/dataset_info.json"

    class Config:
        TYPE_FIELD_NAME: Final = "__type"

    class CMD:
        class OverridePrefixes:
            TRAIN: Final = "train."
            INFERENCE: Final = "infer."
            METRICS: Final = "metrics."

    class Events:
        class Train:
            Start: Final = "Train Start"
            End: Final = "Train End"
            EpochStart: Final = "Train Epoch Start"
            EpochEnd: Final = "Train Epoch End"
            Save: Final = "Train Save"
            TriggerInference: Final = "Train Trigger Inference"
            EpochEvalLossUpdate: Final = "Train Epoch Eval Loss Update"
            EarlyStop: Final = "Train Early Stop"

        class Infer:
            Start: Final = "Infer Start"
            End: Final = "Infer End"
            EpochStart: Final = "Infer Epoch Start"
            EpochEnd: Final = "Infer Epoch End"

        class Metrics:
            GlobalStart: Final = "Metrics Global Start"
            GlobalEnd: Final = "Metrics Global End"
            MetricStart: Final = "Metrics Metric Start"
            MetricResult: Final = "Metrics Metric Result"
            MetricEnd: Final = "Metrics Metric End"
            EpochStart: Final = "Metrics Epoch Start"
            EpochEnd: Final = "Metrics Epoch End"

    class EventSystem:
        Master: Final = "master"

        @staticmethod
        def job(jobname: str) -> str:
            return f"job/{jobname}"

        JobInfer: Final = job("infer")
        JobMetrics: Final = job("metrics")

    class Metrics:
        Loss: Final = "Loss"
        HammingMetric: Final = "HammingMetric"

        @staticmethod
        def namewithset(name: str, set: MetricEvalSet) -> str:
            return f"{name}@{set.value}"

        class Hamming:
            MAP_I2T: Final = "hamming_ranking.mean_average_precision_img2text"
            MAP_I2I: Final = "hamming_ranking.mean_average_precision_img2img"
            MAP_T2I: Final = "hamming_ranking.mean_average_precision_text2img"
            MAP_T2T: Final = "hamming_ranking.mean_average_precision_text2text"
            ROCAUC_I2T: Final = "hash_lookup.roc_auc_img2text"
            ROCAUC_I2I: Final = "hash_lookup.roc_auc_img2img"
            ROCAUC_T2I: Final = "hash_lookup.roc_auc_text2img"
            ROCAUC_T2T: Final = "hash_lookup.roc_auc_text2text"


class DatasetInfo:
    def __init__(self):
        self.dataset_name: str = ""


class DatasetForHashLearningInfo(DatasetInfo):
    def __init__(self):
        self.dataset_name: str = ""
        self.train_indices: list[int] = []
        self.test_indices: list[int] = []
        self.test_retrieval_indices: list[int] = []
        self.validation_indices: list[int] = []
        self.validation_retrieval_indices: list[int] = []
        self.labels: list[str] = []
        self.label_indices: list[int] = []
        self.train_augmentations: str = ""


DSI = TypeVar("DSI", bound=DatasetInfo, covariant=True)


@dataclass
class CM:
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        return (
            (self.true_positives + self.true_negatives)
            / (self.true_positives + self.false_positives + self.true_negatives + self.false_negatives)
            if (self.true_positives + self.false_positives + self.true_negatives + self.false_negatives) != 0
            else 0
        )

    @property
    def precision(self) -> float:
        return (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) != 0
            else 1
        )

    @property
    def recall(self) -> float:
        return (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) != 0
            else 0
        )

    @property
    def f1_score(self) -> float:
        precision = self.precision
        recall = self.recall
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    @property
    def true_positive_rate(self) -> float:
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        return (
            self.false_positives / (self.false_positives + self.true_negatives)
            if (self.false_positives + self.true_negatives) != 0
            else 0
        )

    @staticmethod
    def cumulate(confusion_matrices: Iterable["CM"]) -> "CM":
        cumulated = CM()
        for cm in confusion_matrices:
            cumulated.true_positives += cm.true_positives
            cumulated.false_positives += cm.false_positives
            cumulated.true_negatives += cm.true_negatives
            cumulated.false_negatives += cm.false_negatives
        return cumulated
