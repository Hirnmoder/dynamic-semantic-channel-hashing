from abc import ABC

from dsh.embedding.embedder import EmbedderInfo
from dsh.utils.types import (
    ModelDataPreprocessor,
    ImageRawInput,
    ImageInput,
    RawImagePreprocessor,
    TextRawInput,
    TextInput,
    RawTextPreprocessor,
)


class DatasetToModelAdapter(ABC):
    pass


class CrossModalDatasetToModelAdapter(DatasetToModelAdapter):
    def __init__(
        self,
        embedder_info: EmbedderInfo,
        image_preprocessing_pipeline: ModelDataPreprocessor[ImageRawInput, ImageInput],
        image_preprocessing_first_step: RawImagePreprocessor,
        text_preprocessing_pipeline: ModelDataPreprocessor[TextRawInput, TextInput],
        text_preprocessing_first_step: RawTextPreprocessor,
    ):
        self.embedder_info = embedder_info
        self.image_preprocessing_pipeline = image_preprocessing_pipeline
        self.image_preprocessing_first_step = image_preprocessing_first_step
        self.text_preprocessing_pipeline = text_preprocessing_pipeline
        self.text_preprocessing_first_step = text_preprocessing_first_step
