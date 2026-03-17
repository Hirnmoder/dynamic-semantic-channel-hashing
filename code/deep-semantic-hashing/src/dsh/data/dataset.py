from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar
from torch.utils.data import Dataset

from dsh.config.data import DC, CDC
from dsh.config.env import ModelDataEnvironmentConfig
from dsh.utils.types import CrossModalData, Data

D = TypeVar("D", bound=Data, covariant=True)  # Type variable for dataset data
CD = TypeVar("CD", bound=CrossModalData, covariant=True)  # Type variable for dataset data


class DatasetMode(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    TEST_RETRIEVAL = "test-retrieval"
    VAL_RETRIEVAL = "val-retrieval"
    FULL = "full"


class DatasetBase(ABC, Generic[DC, D], Dataset[D]):
    """Abstract class for all datasets."""

    def __init__(self, config: DC, env: ModelDataEnvironmentConfig, mode: DatasetMode):
        """Initializes a dataset.
        Args:
            config (DatasetConfig): Configuration for the dataset.
            env (ModelDataEnvironmentConfig): Environment configuration.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.mode = mode

        self.env.add_dataset_resolver(lambda: self.dataset_name)

    @abstractmethod
    def load_dataset(self) -> None:
        """Prepare and load dataset, e.g., calculate target values if needed and persist."""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index) -> D:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError()


class CrossModalDataset(Generic[CDC, CD], DatasetBase[CDC, CD]):
    """Abstract class for cross-modal datasets."""

    def __init__(self, config: CDC, env: ModelDataEnvironmentConfig, mode: DatasetMode):
        super().__init__(config, env, mode)

    @property
    @abstractmethod
    def label_dim(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_number_of_samples(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def update_augmentation(self, epoch: int) -> None:
        raise NotImplementedError()


DS = TypeVar("DS", bound=DatasetBase, covariant=True)
