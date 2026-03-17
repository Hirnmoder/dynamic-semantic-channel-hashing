from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, Type, TypeVar
import numpy as np
import torch
import torchvision.transforms.v2 as tf

from dsh.config.data import (
    AugmentationHardnessLevel,
    CrossModalAugmentationConfig,
    ProgressiveAugmentationConfig,
    AugmentationType,
)
from dsh.utils.collections import NonFiniteIntRange, first
from dsh.utils.types import (
    DynamicImageAugmentation,
    DynamicTextAugmentation,
    GenericAugmentation,
    I,
    ImageAugmentation,
    ImageInput,
    NoopTensorAugmentation,
    RawImagePreprocessor,
    TextAugmentation,
    TextInput,
)


Sampler = Callable[[], torch.Tensor]
RA = TypeVar("RA", "RandomImageAugmentation", "RandomTextAugmentation")  # Random Augmentation

CLIP_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_image_augmentation(
    config: CrossModalAugmentationConfig,
    sampler: Sampler | None = None,
) -> DynamicImageAugmentation:
    match config.image_aug:
        case AugmentationType.NONE:
            return InertImageAugmentation(get_image_augmentation_by_hardness(AugmentationHardnessLevel.NONE, sampler))
        case AugmentationType.ALWAYS_HARD:
            return InertImageAugmentation(get_image_augmentation_by_hardness(AugmentationHardnessLevel.HARD, sampler))
        case AugmentationType.ALWAYS_HARD_75_NONE_25:
            return InertImageAugmentation(
                RandomImageAugmentation(
                    [
                        (0.75, get_image_augmentation_by_hardness(AugmentationHardnessLevel.HARD, sampler)),
                        (0.25, get_image_augmentation_by_hardness(AugmentationHardnessLevel.NONE, sampler)),
                    ]
                )
            )
        case ProgressiveAugmentationConfig() as pac:
            return ProgressiveImageAugmentation(pac, sampler, get_image_augmentation_by_hardness)


def get_text_augmentation(
    config: CrossModalAugmentationConfig,
    sampler: Sampler | None = None,
) -> DynamicTextAugmentation:
    match config.text_aug:
        case AugmentationType.NONE:
            return InertTextAugmentation(NoopTensorAugmentation())
        case _:
            raise NotImplementedError(f"Type {config.text_aug} not implemented.")


class InertImageAugmentation:
    def __init__(self, transforms: ImageAugmentation):
        self.transforms = transforms

    def __call__(self, image: ImageInput) -> ImageInput:
        return self.transforms(image)

    def update_augmentation(self, epoch: int) -> None:
        pass


class InertTextAugmentation:
    def __init__(self, transforms: TextAugmentation):
        self.transforms = transforms

    def __call__(self, text: TextInput) -> TextInput:
        return self.transforms(text)

    def update_augmentation(self, epoch: int) -> None:
        pass


class ProgressiveAugmentation(Generic[RA]):
    def __init__(
        self,
        config: ProgressiveAugmentationConfig,
        sampler: Sampler | None,
        get_augmentation_by_hardness: Callable[[AugmentationHardnessLevel, Sampler | None], Callable],
        random_augmentation_type: Type[RA],
    ):
        self._augs = list[tuple[NonFiniteIntRange, random_augmentation_type]]()
        for step in config.configuration:
            r = NonFiniteIntRange(step.start, step.end)
            aug = random_augmentation_type(
                [(m.weight, get_augmentation_by_hardness(m.level, sampler)) for m in step.augmentation_mixtures]
            )
            self._augs.append((r, aug))
        self._selected_aug = first(self._augs, lambda x: 0 in x[0])[1]

    def update_augmentation(self, epoch: int) -> None:
        self._selected_aug = first(self._augs, lambda x: epoch in x[0])[1]


class ProgressiveImageAugmentation(ProgressiveAugmentation["RandomImageAugmentation"]):
    def __init__(
        self,
        config: ProgressiveAugmentationConfig,
        sampler: Sampler | None,
        get_augmentation_by_hardness: Callable[[AugmentationHardnessLevel, Sampler | None], ImageAugmentation],
    ):
        super().__init__(config, sampler, get_augmentation_by_hardness, RandomImageAugmentation)

    def __call__(self, image: ImageInput) -> ImageInput:
        return self._selected_aug(image)


class ProgressiveTextAugmentation(ProgressiveAugmentation["RandomTextAugmentation"]):
    def __init__(
        self,
        config: ProgressiveAugmentationConfig,
        sampler: Sampler | None,
        get_augmentation_by_hardness: Callable[[AugmentationHardnessLevel, Sampler | None], TextAugmentation],
    ):
        super().__init__(config, sampler, get_augmentation_by_hardness, RandomTextAugmentation)

    def __call__(self, text: TextInput) -> TextInput:
        return self._selected_aug(text)


def _init_generator(generator_or_seed: np.random.Generator | int | None) -> np.random.Generator:
    if generator_or_seed == None or isinstance(generator_or_seed, int):
        return np.random.default_rng(generator_or_seed)
    else:
        return generator_or_seed


def _init_weights(
    transforms: Sequence[tuple[float, GenericAugmentation[I]]] | Sequence[GenericAugmentation[I]],
) -> tuple[Sequence[GenericAugmentation[I]], np.ndarray]:
    assert len(transforms) > 0, "No transforms provided"
    _transforms = list[GenericAugmentation[I]]()
    _weights = list[float]()
    for t in transforms:
        if isinstance(t, tuple):
            _transforms.append(t[1])
            _weights.append(t[0])
        else:
            _transforms.append(t)
            _weights.append(1.0)
    assert sum(_weights) > 0, "Weights must not sum to zero"
    return _transforms, np.array(_weights) / sum(_weights)


class RandomNChoice(Generic[I]):
    def __init__(
        self,
        transforms: Sequence[GenericAugmentation[I]] | Sequence[tuple[float, GenericAugmentation[I]]],
        min_n: int,
        max_n: int,
        generator_or_seed: np.random.Generator | int | None = None,
    ):
        assert 0 <= min_n <= max_n <= len(transforms), "Invalid range for min_n and max_n"
        self.transforms, self.weights = _init_weights(transforms)
        self.max_n = max_n
        self.min_n = min_n
        self.generator = _init_generator(generator_or_seed)

    def __call__(self, input: I) -> I:
        n = self.generator.integers(self.min_n, self.max_n + 1)
        chosen_indices = (
            self.generator.choice(
                len(self.transforms),
                n,
                replace=False,
                p=self.weights,
                shuffle=True,
            )
            .flatten()
            .tolist()
        )

        for index in chosen_indices:
            input = self.transforms[index](input)
        return input

    def __repr__(self):
        format_string = f"{self.__class__.__name__}(min_n={self.min_n}, max_n={self.max_n}, transforms=["
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n])"
        return format_string


class RandomImageAugmentation:
    def __init__(
        self,
        transforms: Sequence[tuple[float, ImageAugmentation]],
        generator_or_seed: np.random.Generator | int | None = None,
    ):
        self.transforms, self.weights = _init_weights(transforms)
        self.generator = _init_generator(generator_or_seed)

    def __call__(self, input: ImageInput) -> ImageInput:
        n = self.generator.choice(len(self.transforms), 1, p=self.weights).item()
        return self.transforms[n](input)


class RandomTextAugmentation:
    def __init__(
        self,
        transforms: Sequence[tuple[float, TextAugmentation]],
        generator_or_seed: np.random.Generator | int | None = None,
    ):
        self.transforms, self.weights = _init_weights(transforms)
        self.generator = _init_generator(generator_or_seed)

    def __call__(self, input: TextInput) -> TextInput:
        n = self.generator.choice(len(self.transforms), 1, p=self.weights).item()
        return self.transforms[n](input)


class RandomBlend(ABC):
    def __init__(self, blend_factor: float | tuple[float, float] = (0.1, 0.8)):
        if isinstance(blend_factor, tuple):
            min_blend, max_blend = blend_factor
        else:
            min_blend = float(blend_factor)
            max_blend = float(blend_factor)
        assert 0.0 <= min_blend <= max_blend <= 1.0
        self.min_blend = min_blend
        self.max_blend = max_blend

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        blend_factor = torch.rand(1).item() * (self.max_blend - self.min_blend) + self.min_blend
        return torch.lerp(img, self.get_blend_target(img.shape, img.dtype, img.device), blend_factor)

    @abstractmethod
    def get_blend_target(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor: ...


class RandomUniformNoise(RandomBlend):
    def get_blend_target(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.rand(shape, dtype=dtype, device=device)


class RandomNormalNoise(RandomBlend):
    def get_blend_target(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=device)


class AdversarialBlend(RandomBlend):
    def __init__(self, sampler: Sampler, blend_factor: float | tuple[float, float] = (0.1, 0.3)):
        super().__init__(blend_factor)
        self.sampler = sampler

    def get_blend_target(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        adversarial = self.sampler()
        assert adversarial.shape == shape, f"Expected shape {shape}, got {adversarial.shape}"
        assert adversarial.dtype == dtype, f"Expected dtype {dtype}, got {adversarial.dtype}"
        assert adversarial.device == device, f"Expected device {device}, got {adversarial.device}"
        return adversarial


class AdversarialPIP:
    def __init__(self, sampler: Sampler, adversarial_scale: tuple[float, float] = (0.1, 0.2)):
        self.sampler = sampler
        self.min_scale, self.max_scale = min(adversarial_scale), max(adversarial_scale)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        scale = torch.rand(1).item() * (self.max_scale - self.min_scale) + self.min_scale
        adversarial = self.sampler()
        _, ha, wa = adversarial.shape
        adversarial_scaled = tf.functional.resize_image(adversarial, [int(wa * scale), int(ha * scale)])
        _, hi, wi = img.shape
        _, hs, ws = adversarial_scaled.shape
        x = int(torch.rand(1).item() * (wi - ws))
        y = int(torch.rand(1).item() * (hi - hs))
        img_blended = img.clone()
        img_blended[:, x : x + ws, y : y + hs] = adversarial_scaled
        return img_blended


def get_image_preprocessing_default(resize: int = 232, crop: int = 224) -> RawImagePreprocessor:
    return tf.Compose(
        [
            tf.ToImage(),
            tf.ToDtype(torch.float32, scale=True),
            tf.Resize(resize, interpolation=tf.InterpolationMode.BILINEAR),
            tf.CenterCrop(crop),
        ]
    )


def get_image_normalization_imagenet() -> ImageAugmentation:
    return get_image_normalization(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_image_normalization_clip() -> ImageAugmentation:
    return get_image_normalization(mean=CLIP_DATASET_MEAN, std=CLIP_DATASET_STD)


def get_image_normalization(mean: tuple[float, float, float], std: tuple[float, float, float]) -> ImageAugmentation:
    return tf.Compose([tf.Normalize(mean=mean, std=std)])


def unnormalize_image_imagenet(img: ImageInput) -> ImageInput:
    return unnormalize_image(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def unnormalize_image_openai(img: ImageInput) -> ImageInput:
    return unnormalize_image(img, mean=CLIP_DATASET_MEAN, std=CLIP_DATASET_STD)


def unnormalize_image(img: ImageInput, mean: tuple[float, float, float], std: tuple[float, float, float]) -> ImageInput:
    inv_norm = tf.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1.0 / s for s in std], inplace=False)
    return torch.clamp(inv_norm(img), 0.0, 1.0)


def get_image_augmentation_by_hardness(
    hardness_level: AugmentationHardnessLevel,
    sampler: Sampler | None,
) -> ImageAugmentation:
    return tf.Compose(_get_image_augmentation_by_hardness(hardness_level, sampler))


def _get_image_augmentation_by_hardness(
    hardness_level: AugmentationHardnessLevel,
    sampler: Sampler | None,
) -> list[ImageAugmentation]:
    match hardness_level:
        case AugmentationHardnessLevel.NONE:
            return [NoopTensorAugmentation()]
        case AugmentationHardnessLevel.EASY:
            return [
                RandomNChoice(
                    [
                        # No "random" rotation, just fixed rotations for easy level
                        tf.RandomRotation((-90, -90)),
                        tf.RandomRotation((-45, -45)),
                        tf.RandomRotation((45, 45)),
                        tf.RandomRotation((90, 90)),
                        tf.RandomHorizontalFlip(p=1.0),
                        tf.Grayscale(3),
                        tf.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(1, 1)),  # Square crop
                    ],
                    min_n=1,
                    max_n=1,
                ),
            ]
        case AugmentationHardnessLevel.MEDIUM:
            return [
                tf.RandomHorizontalFlip(),
                RandomNChoice(
                    [
                        tf.RandomRotation((-45, 45)),
                        tf.Grayscale(3),
                        tf.RandomResizedCrop(224, scale=(0.35, 1.1), ratio=(0.9, 1.1)),  # Variable scale and ratio
                        tf.RandomErasing(p=1.0, scale=(0.02, 0.05), ratio=(1 / 3, 3)),  # Variable scale and ratio
                        tf.ColorJitter(0.2, 0.2, 0.2, 0.1),  # Brightness, contrast, saturation, hue
                        RandomUniformNoise((0.0, 0.5)),
                    ],
                    min_n=1,
                    max_n=2,
                ),
            ]
        case AugmentationHardnessLevel.HARD:
            assert sampler != None, "Sampler must be provided for hard augmentation"
            return [
                tf.RandomHorizontalFlip(),
                RandomNChoice(
                    [
                        (
                            0.8,
                            RandomNChoice(
                                [
                                    # This is generally the same as medium but with more aggressive transformation parameters
                                    tf.RandomRotation((-90, 90)),
                                    tf.Grayscale(3),
                                    tf.RandomResizedCrop(224, scale=(0.2, 1.1), ratio=(3 / 4, 4 / 3)),  # Variable scale and ratio
                                    tf.RandomErasing(p=1.0, scale=(0.05, 0.15), ratio=(1 / 4, 4)),  # Variable scale and ratio
                                    tf.ColorJitter(0.5, 0.5, 0.5, 0.25),  # Brightness, contrast, saturation, hue
                                    # Additional augmentations for hard
                                    tf.RandomVerticalFlip(p=1.0),
                                    tf.ElasticTransform(alpha=50.0, sigma=5.0),
                                    RandomUniformNoise((0.0, 0.9)),
                                ],
                                min_n=1,
                                max_n=3,
                            ),
                        ),
                        (
                            0.2,
                            RandomNChoice(
                                [
                                    AdversarialBlend(sampler, (0.0, 0.45)),
                                    AdversarialPIP(sampler, (0.05, 0.25)),
                                ],
                                min_n=1,
                                max_n=1,
                            ),
                        ),
                    ],
                    min_n=1,
                    max_n=1,
                ),
            ]
        case _:
            raise ValueError(f"Unknown augmentation hardness level {hardness_level}")
