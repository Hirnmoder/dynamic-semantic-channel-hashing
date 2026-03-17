from typing import Callable
import numpy as np
import open_clip as clip

from dsh.embedding.clipembedder import CLIPEmbedder
from dsh.embedding.embedder import Embedder, EmbedderInfo
from dsh.model.common import get_hash_mlp
from dsh.model.modelbase import CrossModalTopLevelModelBase
from dsh.config.model import CLIPHashModelConfig
from dsh.config.env import ModelDataEnvironmentConfig
from dsh.utils.adapter import CrossModalDatasetToModelAdapter
from dsh.utils.augmentation import get_image_normalization_clip, get_image_preprocessing_default
from dsh.utils.types import (
    ImageRawInput,
    NoopTensorAugmentation,
    Output,
    TextInput,
    ImageInput,
    FullModelInput,
    TextRawInput,
)


class CLIPHash(CrossModalTopLevelModelBase[CLIPHashModelConfig]):
    def __init__(self, config: CLIPHashModelConfig, env: ModelDataEnvironmentConfig):
        super().__init__(config, env)

        encoder_config = clip.get_model_config(config.clip_model)
        assert isinstance(encoder_config, dict), "Encoder config invalid"
        encoder_output_size = encoder_config.pop("embed_dim", -1)
        assert encoder_output_size > 0, "Encoder output size invalid"
        self.encoder_config = encoder_config

        encoder = clip.create_model(config.clip_model, pretrained=config.clip_model_pretrained)
        if not isinstance(encoder, clip.CLIP):
            raise ValueError("Model is not a CLIP model")
        self.encoder = encoder

        self.text_preprocessor = NoopTensorAugmentation()
        self.text_postprocessor = self._reshape_text

        assert (
            isinstance(encoder.visual.image_size, tuple)
            and len(encoder.visual.image_size) == 2
            and isinstance(encoder.visual.image_size[0], int)
        )
        image_size = encoder.visual.image_size[0]
        self.image_preprocessor = get_image_preprocessing_default(resize=image_size, crop=image_size)
        self.image_postprocessor = get_image_normalization_clip()

        self.hash_network = get_hash_mlp(config.hash, in_dim=encoder_output_size, out_dim=config.hash_length)

    def _reshape_text(self, text: TextInput) -> TextInput:
        _, *dims = text.shape
        if len(dims) == 0:
            return text
        elif len(dims) == 1:
            assert dims[0] == 1, "Text input must be of shape (seq_len, 1)"
            return text.squeeze(dim=-1)
        raise ValueError("Text input must be of shape (seq_len,) or (seq_len, 1)")

    def forward(self, input: FullModelInput) -> Output:
        t, data = input
        match t:
            case "text":
                encoded = self.encoder.encode_text(data)
            case "image":
                encoded = self.encoder.encode_image(data)
            case _:
                raise NotImplementedError(f"Unsupported input type {t}")
        hash = self.hash_network(encoded)
        return hash

    def apply_freezing(self):
        if self.config.clip_model_pretrained and len(self.config.clip_model_pretrained) > 0:
            for p in self.encoder.parameters():
                # freeze all parameters in already pretrained backbone
                p.requires_grad = False

    @property
    def model_name(self):
        return "CLIPHash"

    def text_preprocessing_pipeline(
        self,
        raw_data: TextRawInput,
        augmentation_pipeline: Callable[[TextInput], TextInput],
    ) -> TextInput:
        prep = self.text_preprocessor(raw_data)
        aug = augmentation_pipeline(prep)
        post = self.text_postprocessor(aug)
        return post

    def image_preprocessing_pipeline(
        self,
        raw_data: ImageRawInput,
        augmentation_pipeline: Callable[[ImageInput], ImageInput],
    ) -> ImageInput:
        prep = self.image_preprocessor(raw_data)
        aug = augmentation_pipeline(prep)
        post = self.image_postprocessor(aug)
        return post

    def get_embedder(self) -> Embedder:
        tokenizer = clip.get_tokenizer(model_name=self.config.clip_model)
        return CLIPEmbedder(tokenizer)

    def get_adapter(self) -> CrossModalDatasetToModelAdapter:
        text_config = self.encoder_config["text_cfg"]
        assert isinstance(text_config, dict)
        sequence_length = text_config.pop("context_length", 77)
        embedding_dim = 1  # we use tokenizer only, not an embedding -> dimension is 1
        name = self.config.clip_model

        self.text_embedder_name = name
        self.text_seqlen = sequence_length

        return CrossModalDatasetToModelAdapter(
            EmbedderInfo(
                name,
                sequence_length,
                embedding_dim,
                np.long,
                self.get_embedder,
            ),
            self.image_preprocessing_pipeline,
            self.image_preprocessor,
            self.text_preprocessing_pipeline,
            self.text_preprocessor,
        )
