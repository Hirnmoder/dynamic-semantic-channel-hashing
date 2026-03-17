from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Callable, Final, Optional, TypeVar

from dsh.config.configuration import ConfigurationBase, Field, register
from dsh.utils.unparsing import stringify_img_txt


class DatasetConfig(ConfigurationBase):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class CrossModalDatasetConfig(DatasetConfig):
    pass


DC = TypeVar("DC", bound=DatasetConfig, covariant=True)  # Type variable for data config
CDC = TypeVar("CDC", bound=CrossModalDatasetConfig, covariant=True)  # Type variable for cross modal data config


class TrainTTSConfig(ConfigurationBase, ABC):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


@register("TTS-as-dataset")
class TrainTTSAsDataset(TrainTTSConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


class TTSRetrievalMode(StrEnum):
    ALL = "all"
    ALL_WITHOUT_TEST = "all-without-test"
    ALL_WITHOUT_TRAIN_AND_TEST = "all-without-train-and-test"


class TTSSamplingMode(StrEnum):
    RANDOM = "random"
    N_PER_CLASS = "n-per-class"
    ITERATIVE_STRATIFICATION = "iterative-stratification"


@register("TTS-by-numbers")
class TrainTTSByNumbers(TrainTTSConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.num_train = Field[int | float](0.8)()
        self.num_test = Field[int | float](0.1)()
        self.num_retrieval = Field[int | float | TTSRetrievalMode](TTSRetrievalMode.ALL_WITHOUT_TEST)()
        self.num_val = Field[int | float](0.1)()
        self.sampling_mode = Field[TTSSamplingMode](TTSSamplingMode.RANDOM)()
        self.seed = Field[Optional[int]](42)()

    @staticmethod
    def preset_mirflickr25k() -> "TrainTTSByNumbers":
        """MIR-Flickr 25K"""
        s = TrainTTSByNumbers()
        s.num_train = 10_000
        s.num_test = 2_000
        s.num_retrieval = TTSRetrievalMode.ALL_WITHOUT_TEST
        s.num_val = 2_000
        s.sampling_mode = TTSSamplingMode.RANDOM
        return s

    @staticmethod
    def preset_nuswide() -> "TrainTTSByNumbers":
        """NUS-WIDE"""
        s = TrainTTSByNumbers()
        s.num_train = 10_500
        s.num_test = 2_100
        s.num_retrieval = TTSRetrievalMode.ALL_WITHOUT_TEST
        s.num_val = 2_100
        s.sampling_mode = TTSSamplingMode.N_PER_CLASS
        return s


class AugmentationConfigBase(ConfigurationBase, ABC):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @abstractmethod
    def stringify(self) -> str: ...


class AugmentationType(StrEnum):
    NONE = "none"
    ALWAYS_HARD = "always-hard"
    ALWAYS_HARD_75_NONE_25 = "always-hard75-none25"

    def stringify(self) -> str:
        return self.value


class AugmentationHardnessLevel(StrEnum):
    NONE = "none"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


AHL = AugmentationHardnessLevel


@register("WeightedAHL")
class WAHL(ConfigurationBase):
    def __init__(self, level: AHL = AHL.NONE, weight: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.level = level
        self.weight = weight

    def stringify(self) -> str:
        return f"({self.weight}*{self.level.value})"


@register("Progressive")
class ProgressiveAugmentationConfig(ConfigurationBase):
    def __init__(self, default_steps: Callable[[], list["ProgressiveAugmentationStep"]] = list, **kwargs: Any):
        super().__init__(**kwargs)
        self.configuration = Field[list[ProgressiveAugmentationStep]](default_steps)()

    def stringify(self) -> str:
        return f"Progressive({', '.join([s.stringify() for s in self.configuration])})"

    @staticmethod
    def preset_200_progressive() -> "ProgressiveAugmentationConfig":
        """Progressive Augmentation over 200 epochs"""
        return ProgressiveAugmentationConfig(
            lambda: [
                # fmt: off
                ProgressiveAugmentationStep(None, 20, [WAHL(AHL.NONE, 1.0)]),
                ProgressiveAugmentationStep(20, 40, [WAHL(AHL.NONE, 0.25), WAHL(AHL.EASY, 0.75)]),
                ProgressiveAugmentationStep(40, 60, [WAHL(AHL.NONE, 0.25), WAHL(AHL.EASY, 0.50), WAHL(AHL.MEDIUM, 0.25)]),
                ProgressiveAugmentationStep(60, 80, [WAHL(AHL.NONE, 0.25), WAHL(AHL.EASY, 0.25), WAHL(AHL.MEDIUM, 0.50)]),
                ProgressiveAugmentationStep(80, 100, [WAHL(AHL.NONE, 0.25), WAHL(AHL.EASY, 0.25), WAHL(AHL.MEDIUM, 0.25), WAHL(AHL.HARD, 0.25)]),
                ProgressiveAugmentationStep(100, 150, [WAHL(AHL.NONE, 0.25), WAHL(AHL.MEDIUM, 0.25), WAHL(AHL.HARD, 0.50)]),
                ProgressiveAugmentationStep(150, None, [WAHL(AHL.NONE, 0.25), WAHL(AHL.HARD, 0.75)]),
                # fmt: on
            ]
        )


@register("Progressive-Step")
class ProgressiveAugmentationStep(ConfigurationBase):
    def __init__(
        self,
        start: int | None = None,
        end: int | None = None,
        augmentation_mixtures: list[WAHL] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.start = Field[int | None](start)()
        self.end = Field[int | None](end)()
        self.augmentation_mixtures = Field[list[WAHL]](augmentation_mixtures or [])()

    def stringify(self) -> str:
        return (
            f"{{{self.start or '-inf'}:{self.end or '+inf'} = {' + '.join([m.stringify() for m in self.augmentation_mixtures])}}}"
        )


AugmentationTypeConfig: Final = ProgressiveAugmentationConfig | AugmentationType


@register("Image-Text-Augmentation")
class CrossModalAugmentationConfig(AugmentationConfigBase):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.image_aug = Field[AugmentationTypeConfig](AugmentationType.NONE)()
        self.text_aug = Field[AugmentationTypeConfig](AugmentationType.NONE)()

    def stringify(self) -> str:
        return stringify_img_txt(self.image_aug.stringify(), self.text_aug.stringify())

    @staticmethod
    def preset_img_progressive() -> "CrossModalAugmentationConfig":
        """Progressive Image Augmentation, No Text Augmentation"""
        c = CrossModalAugmentationConfig()
        c.image_aug = ProgressiveAugmentationConfig.preset_200_progressive()
        c.text_aug = AugmentationType.NONE
        return c


@register("NUS-WIDE")
class NUSWideConfig(CrossModalDatasetConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.top_k_labels = Field[Optional[int]](21)()
        self.include_samples_without_labels = Field[bool](False)()
        self.ttr_split = Field[TrainTTSConfig](TrainTTSByNumbers.preset_nuswide)()
        self.augmentation = Field[CrossModalAugmentationConfig](CrossModalAugmentationConfig)()


@register("MIR-FLICKR25K")
class MirFlickr25kConfig(CrossModalDatasetConfig):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.min_textual_tags = Field[Optional[int]](1)()
        self.only_samples_with_tag_that_occurs_min_k_times = Field[Optional[int]](20)()
        self.remove_tags_that_occur_less_than_j_times = Field[bool | int](False)()
        self.include_samples_without_labels = Field[bool](False)()
        self.ttr_split = Field[TrainTTSConfig](TrainTTSByNumbers.preset_mirflickr25k)()
        self.augmentation = Field[CrossModalAugmentationConfig](CrossModalAugmentationConfig)()
