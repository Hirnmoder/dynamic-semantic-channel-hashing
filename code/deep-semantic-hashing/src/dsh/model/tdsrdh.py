import math
from typing import Callable
import torch
from torchvision.models import (
    resnet152,
    ResNet152_Weights,
    resnet101,
    ResNet101_Weights,
    resnet50,
    ResNet50_Weights,
    resnet34,
    ResNet34_Weights,
    resnet18,
    ResNet18_Weights,
)

from dsh.config.env import ModelDataEnvironmentConfig
from dsh.config.model import (
    PositionalEncodingType,
    ResNetConfig,
    SingleModalEncoderConfig,
    TDSRDHModelConfig,
    TextEmbedderType,
    TransformerConfig,
)
from dsh.model.common import get_hash_mlp, get_embedder_info_from_config
from dsh.model.modelbase import ModelBase, CrossModalTopLevelModelBase
from dsh.utils.activation import get_activation
from dsh.utils.adapter import CrossModalDatasetToModelAdapter
from dsh.utils.augmentation import get_image_preprocessing_default, get_image_normalization_imagenet
from dsh.utils.initialization import init_weights, InitializationType
from dsh.utils.types import (
    ImageAugmentation,
    ImageRawInput,
    NoopTensorAugmentation,
    Output,
    TextAugmentation,
    TextInput,
    ImageInput,
    FullModelInput,
    TextRawInput,
)


class TDSRDH(CrossModalTopLevelModelBase[TDSRDHModelConfig]):
    def __init__(self, config: TDSRDHModelConfig, env: ModelDataEnvironmentConfig):
        super().__init__(config, env)

        match config.text_embedder:
            case TextEmbedderType.W2V_GN300_FC:
                embedding_rescale = [300]
            case _:
                raise NotImplementedError(f"Unknown text embedder {config.text_embedder}")

        self.vision = TDSRDHVision(config.vision, env, config.hash_length)
        self.text = TDSRDHText(config.text, env, config.hash_length, embedding_rescale, config.text_sequence_length)

    def forward(self, input: FullModelInput) -> Output:
        t, data = input
        if t == "text":
            output = self.text(data)
        elif t == "image":
            output = self.vision(data)
        else:
            raise NotImplementedError(f"Unsupported input type {t}")
        return output

    def apply_freezing(self):
        self.vision.apply_freezing()
        self.text.apply_freezing()

    @property
    def model_name(self) -> str:
        return "TDSRDH"

    def get_adapter(self) -> CrossModalDatasetToModelAdapter:
        return CrossModalDatasetToModelAdapter(
            get_embedder_info_from_config(self.config, self.env),
            self.vision.preprocess_data,
            self.vision.image_preprocessor,
            self.text.preprocess_data,
            self.text.text_preprocessor,
        )


class TDSRDHVision(ModelBase[SingleModalEncoderConfig, ImageInput, Output]):
    def __init__(self, config: SingleModalEncoderConfig, env: ModelDataEnvironmentConfig, hash_length: int):
        super().__init__(config, env)

        # two parts: image encoder f and hash-code learning d
        if isinstance(config.encoder, ResNetConfig):
            match config.encoder.depth:
                case 152:
                    self.image_encoder = resnet152(
                        weights=ResNet152_Weights.IMAGENET1K_V2 if config.encoder.pre_trained else None
                    )
                case 101:
                    self.image_encoder = resnet101(
                        weights=ResNet101_Weights.IMAGENET1K_V2 if config.encoder.pre_trained else None
                    )
                case 50:
                    self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if config.encoder.pre_trained else None)
                case 34:
                    self.image_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if config.encoder.pre_trained else None)
                case 18:
                    self.image_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if config.encoder.pre_trained else None)
                case _:
                    raise ValueError(f"Unsupported ResNet depth: {config.encoder.depth}")

            # replace the last layer with a new layer that does nothing
            n_features = self.image_encoder.fc.in_features
            self.image_encoder.fc = torch.nn.Identity()  # type: ignore # ResNet's fc is not typed correctly
            self.image_preprocessor = get_image_preprocessing_default()
            self.image_postprocessor = get_image_normalization_imagenet()

            self.hash_network = get_hash_mlp(config.hash, in_dim=n_features, out_dim=hash_length)
        else:
            raise ValueError(f"Unsupported encoder type: {config.encoder}")

        assert self.image_encoder, "Internal implementation error"
        assert self.hash_network, "Internal implementation error"
        assert self.image_preprocessor, "Internal implementation error"
        assert self.image_postprocessor, "Internal implementation error"

    def forward(self, input: ImageInput) -> Output:
        x = self.image_encoder(input)
        x = self.hash_network(x)
        return x

    def apply_freezing(self):
        assert isinstance(self.config.encoder, ResNetConfig)
        if self.config.encoder.pre_trained:
            for p in self.image_encoder.parameters():
                # freeze all parameters in already pretrained backbone
                p.requires_grad = False

    def preprocess_data(self, raw_data: ImageRawInput, augmentation_pipeline: ImageAugmentation) -> ImageInput:
        prep = self.image_preprocessor(raw_data)
        aug = augmentation_pipeline(prep)
        post = self.image_postprocessor(aug)
        return post


class TDSRDHText(ModelBase[SingleModalEncoderConfig, TextInput, Output]):
    def __init__(
        self,
        config: SingleModalEncoderConfig,
        env: ModelDataEnvironmentConfig,
        hash_length: int,
        embedding_rescale: list[int],
        text_max_length: int,
    ):
        super().__init__(config, env)

        if isinstance(config.encoder, TransformerConfig):
            assert len(embedding_rescale) > 0, "embedding_rescale must have at least one element"
            text_emb_dim = config.encoder.embedding_dim
            embedding_rescale.append(text_emb_dim)

            self.resize_embedding = torch.nn.Sequential(
                *[
                    torch.nn.Linear(emb_r_in, emb_r_out, bias=False)
                    for emb_r_in, emb_r_out in zip(embedding_rescale[:-1], embedding_rescale[1:])
                ]
            )

            match config.encoder.positional_encoding:
                case PositionalEncodingType.SINCOS:
                    self.positional_encoding = PositionalEncodingSinCos(text_emb_dim, text_max_length)
                case PositionalEncodingType.LEARNED:
                    self.positional_encoding = PositionalEncodingLearned(text_emb_dim, text_max_length)
                case PositionalEncodingType.NONE:
                    self.positional_encoding = torch.nn.Identity()
                case _:
                    raise ValueError(f"Unknown positional encoding type: {config.encoder.positional_encoding}")

            enc_layer = torch.nn.TransformerEncoderLayer(
                d_model=text_emb_dim,
                nhead=config.encoder.n_head,
                dim_feedforward=config.encoder.dim_ff,
                dropout=config.encoder.dropout,
                activation=get_activation(config.encoder.activation),
                batch_first=True,  # data is: batch, seq, embedding
            )
            self.text_encoder = torch.nn.TransformerEncoder(
                enc_layer,
                num_layers=config.encoder.n_layer,
            )
            self.text_encoder_pooling: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = self._pool_mean_mask_aware

            init_weights(self.resize_embedding, InitializationType.HE_NORMAL, nonlinearity="linear")
            init_weights(self.text_encoder, InitializationType.XAVIER_NORMAL)

            self.text_preprocessor = NoopTensorAugmentation()
            self.text_postprocessor = NoopTensorAugmentation()

            self.hash_network = get_hash_mlp(config.hash, in_dim=text_emb_dim, out_dim=hash_length)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder}")

        assert self.text_encoder, "Internal implementation error"
        assert self.hash_network, "Internal implementation error"
        assert self.text_preprocessor, "Internal implementation error"
        assert self.text_postprocessor, "Internal implementation error"

    def forward(self, input: TextInput) -> Output:
        n, s, d = input.shape  # batch, seq, embedding
        # generate mask for padding tokens
        mask = (input != 0).any(dim=-1)

        x = input.view((-1, d))
        x = self.resize_embedding(x)
        x = x.view((n, s, -1))

        x_pe = self.positional_encoding(x)

        x_tr = self.text_encoder(x_pe)
        x_po = self.text_encoder_pooling(x_tr, mask)
        x_ha = self.hash_network(x_po)
        return x_ha

    def apply_freezing(self):
        pass

    def preprocess_data(self, raw_data: TextRawInput, augmentation_pipeline: TextAugmentation) -> TextInput:
        prep = self.text_preprocessor(raw_data)
        aug = augmentation_pipeline(prep)
        post = self.text_postprocessor(aug)
        return post

    def _pool_mean_mask_aware(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert mask.shape == x.shape[:-1], f"mask shape {mask.shape} does not match x shape {x.shape}"
        n, s, d = x.shape  # batch, seq, embedding
        mask = mask.unsqueeze(-1).expand(n, s, d).float()
        sum_x = torch.sum(x * mask, dim=1)  # sum along token dimension
        sum_m = torch.clamp(mask.sum(dim=1), min=1e-9)  # avoid division by zero
        return sum_x / sum_m


class PositionalEncodingSinCos(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, constant: float = 10000.0):
        super().__init__()

        pe = torch.zeros((max_len, d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(constant) / d_model))  # (d_model//2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices -> sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices -> cos

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: TextInput) -> TextInput:
        assert isinstance(self.pe, torch.Tensor)
        n, seq_len, _ = x.shape
        return x + self.pe[:, :seq_len, :].expand((n, -1, -1))


class PositionalEncodingLearned(torch.nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = torch.nn.Parameter(torch.zeros((max_len, d_model)))

    def forward(self, x: TextInput) -> TextInput:
        assert isinstance(self.pe, torch.Tensor)
        n, seq_len, _ = x.shape
        return x + self.pe[:seq_len, :].expand((n, -1, -1))
