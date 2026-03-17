from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Literal, TypeVar

from dsh.config.configuration import ConfigurationBase, Field, register
from dsh.utils.unparsing import stringify


class ModelConfig(ConfigurationBase, ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class SignFunction(Enum):
    ZERO_IS_POSITIVE = "zero-is-positive"
    ZERO_IS_NEGATIVE = "zero-is-negative"
    ZERO_IS_ZERO = "zero-is-zero"


class TextEmbedderType(Enum):
    W2V_GN300_FC = "w2v-gn300-fc"  # word2vec pre-trained Google News 300d vectors with a learnable fully connected layer to expand to needed token dimensions


class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    IDENTITY = "identity"


class PositionalEncodingType(Enum):
    SINCOS = "sincos"
    LEARNED = "learned"
    NONE = "none"


class EncoderConfig(ModelConfig, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@register("Transformer")
class TransformerConfig(EncoderConfig):
    def __init__(
        self,
        n_head: int = 4,  # taken from TDSRDH Paper
        n_layer: int = 1,  # taken from TDSRDH Paper
        dim_ff: int = 2048,  # not stated in paper!
        dropout: float = 0.1,  # not stated in paper!
        activation: ActivationType = ActivationType.RELU,  # not stated in paper!
        embedding_dim: int = 512,  # taken from TDSRDH Paper
        positional_encoding: PositionalEncodingType = PositionalEncodingType.SINCOS,  # not stated in paper!
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.n_head = Field[int](n_head)()
        self.n_layer = Field[int](n_layer)()
        self.dim_ff = Field[int](dim_ff)()
        self.dropout = Field[float](dropout)()
        self.activation = Field[ActivationType](activation)()
        self.embedding_dim = Field[int](embedding_dim)()
        self.positional_encoding = Field[PositionalEncodingType](positional_encoding)()

    @property
    def name(self) -> str:
        return stringify("Transformer", **self.__dict__)


RESNET_SIZES = Literal[18, 34, 50, 101, 152] # fmt: skip
@register("ResNet")
class ResNetConfig(EncoderConfig):

    def __init__(self, depth: RESNET_SIZES = 152, pre_trained: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depth = Field[RESNET_SIZES](depth)()
        self.pre_trained = Field[bool](pre_trained)()

    @property
    def name(self) -> str:
        return f"ResNet{self.depth}"


@register("MLP")
class MLPConfig(ModelConfig):
    def __init__(
        self,
        dims: list[int] = [4096, 4096],
        dropout: float = 0.5,
        activation: ActivationType = ActivationType.TANH,
        final_activation: ActivationType = ActivationType.IDENTITY,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = Field[list[int]](dims)()
        self.dropout = Field[float](dropout)()
        self.activation = Field[ActivationType](activation)()
        self.final_activation = Field[ActivationType](final_activation)()

    def stringify(self) -> str:
        return stringify("MLP", **self.__dict__)

    @staticmethod
    def preset_clip_hash() -> "MLPConfig":
        """MLP for use with CLIPHash (4096, 4096)"""
        return MLPConfig(dims=[4096, 4096], dropout=0.1, activation=ActivationType.TANH, final_activation=ActivationType.IDENTITY)

    @staticmethod
    def preset_vision_hash() -> "MLPConfig":
        """MLP for use with ResNet (4096, 4096)"""
        return MLPConfig(dims=[4096, 4096], dropout=0.1, activation=ActivationType.TANH, final_activation=ActivationType.IDENTITY)

    @staticmethod
    def preset_text_hash() -> "MLPConfig":
        """MLP for use with Transformer (1024, 8192)"""
        return MLPConfig(dims=[1024, 8192], dropout=0.1, activation=ActivationType.TANH, final_activation=ActivationType.IDENTITY)


@register("EncHash")
class SingleModalEncoderConfig(ModelConfig):
    def __init__(
        self,
        default_encoder: Callable[[], EncoderConfig] = EncoderConfig,
        default_mlp: Callable[[], MLPConfig] = MLPConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder = Field[EncoderConfig](default_encoder)()
        self.hash = Field[MLPConfig](default_mlp)()

    @staticmethod
    def preset_vision() -> "SingleModalEncoderConfig":
        """ResNet vision with MLP hash"""
        return SingleModalEncoderConfig(
            lambda: ResNetConfig(depth=152, pre_trained=True),
            MLPConfig.preset_vision_hash,
        )

    @staticmethod
    def preset_text() -> "SingleModalEncoderConfig":
        """Transformer text with MLP hash"""
        return SingleModalEncoderConfig(
            lambda: TransformerConfig(
                n_head=4,
                n_layer=1,
                dim_ff=2048,
                dropout=0.1,
                activation=ActivationType.RELU,
                embedding_dim=512,
                positional_encoding=PositionalEncodingType.SINCOS,
            ),
            MLPConfig.preset_text_hash,
        )


class HashModelConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hash_length: int = 32
        self.sign_function = Field[SignFunction](SignFunction.ZERO_IS_POSITIVE)()


@register("TDSRDH")
class TDSRDHModelConfig(HashModelConfig):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.text_embedder = Field[TextEmbedderType](TextEmbedderType.W2V_GN300_FC)()
        self.text_sequence_length: int = 128
        self.vision = Field[SingleModalEncoderConfig](SingleModalEncoderConfig.preset_vision)()
        self.text = Field[SingleModalEncoderConfig](SingleModalEncoderConfig.preset_text)()


@register("CLIPHash")
class CLIPHashModelConfig(HashModelConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.clip_model = Field[str]("ViT-H-14-quickgelu")()
        self.clip_model_pretrained = Field[str]("dfn5b")()

        self.hash = Field[MLPConfig](MLPConfig.preset_clip_hash)()


MC = TypeVar("MC", bound=ModelConfig, covariant=True)  # Type variable for model config
HMC = TypeVar("HMC", bound=HashModelConfig, covariant=True)  # Type variable for hash model config
