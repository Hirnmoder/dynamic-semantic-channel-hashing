from collections import Counter
from glob import glob
import PIL.Image
import h5py
import numpy as np
import os
import torch
from typing import Callable, Final, NamedTuple

from dsh.data.common import handle_tts_by_numbers, handle_embeddings
from dsh.data.dataset import CrossModalDataset, DatasetMode
from dsh.config.data import MirFlickr25kConfig, TrainTTSByNumbers, TrainTTSAsDataset
from dsh.config.env import ModelDataEnvironmentConfig
from dsh.utils.adapter import CrossModalDatasetToModelAdapter
from dsh.utils.augmentation import (
    get_image_augmentation,
    get_text_augmentation,
    InertImageAugmentation,
    InertTextAugmentation,
)
from dsh.utils.logger import Logger, LogLevel
from dsh.utils.parsing import try_parse_int
from dsh.utils.progress import tqdm
from dsh.utils.types import (
    CrossModalData,
    DatasetForHashLearningInfo,
    DynamicImageAugmentation,
    DynamicTextAugmentation,
    ExtendedCrossModalData,
    ImageRawInput,
    NoopTensorAugmentation,
    TextInput,
    TextRawInput,
)

H5_EMBEDDING_DATASET_NAME: Final = "mirflickr25k_embeddings"
DATASET_NAME: Final = "MIR-FLICKR25K"


class _MirFlickr25k_Metadata(NamedTuple):
    ids: list[int]
    id_to_path: dict[int, str]
    id_to_tags: dict[int, list[str]]
    id_to_labels: dict[int, np.ndarray]
    label_names: list[str]
    id_to_emb_index: dict[int, int]
    embedding_path: str
    max_number_of_samples: int


class MirFlickr25kDataset(CrossModalDataset[MirFlickr25kConfig, CrossModalData]):
    def __init__(
        self,
        config: MirFlickr25kConfig,
        env: ModelDataEnvironmentConfig,
        mode: DatasetMode,
        adapter: CrossModalDatasetToModelAdapter,
        metadata: _MirFlickr25k_Metadata,
    ):
        super().__init__(config, env, mode)
        self.adapter = adapter
        self._dataset_metadata = metadata
        self._return_extended_data = False

    @property
    def dataset_name(self) -> str:
        return DATASET_NAME

    @property
    def label_dim(self) -> int:
        return len(self._dataset_metadata.label_names)

    @property
    def max_number_of_samples(self) -> int:
        return self._dataset_metadata.max_number_of_samples

    @property
    def return_extended_data(self) -> bool:
        return self._return_extended_data

    @return_extended_data.setter
    def return_extended_data(self, value: bool):
        self._return_extended_data = value

    def load_dataset(self) -> None:
        if self._check_dataset_integrity():
            Logger().error(f"Dataset {self.dataset_name} is corrupted or missing files.")
            return

        self.image_preprocessing_pipeline = self.adapter.image_preprocessing_pipeline
        self.image_preprocessing_first_step = self.adapter.image_preprocessing_first_step
        self.text_preprocessing_pipeline = self.adapter.text_preprocessing_pipeline
        self.text_preprocessing_first_step = self.adapter.text_preprocessing_first_step

        self.image_augmentations: DynamicImageAugmentation
        self.text_augmentations: DynamicTextAugmentation
        if self.mode == DatasetMode.TRAIN:
            self.image_augmentations = get_image_augmentation(self.config.augmentation, self.__sample_image_min_preproc)
            self.text_augmentations = get_text_augmentation(self.config.augmentation, self.__sample_text_min_preproc)
        else:
            self.image_augmentations = InertImageAugmentation(NoopTensorAugmentation())
            self.text_augmentations = InertTextAugmentation(NoopTensorAugmentation())

    def __len__(self) -> int:
        return len(self._dataset_metadata.ids)

    def __sample_image(self, index: int) -> ImageRawInput:
        id = self._dataset_metadata.ids[index]
        path = self._dataset_metadata.id_to_path[id]
        image_raw = PIL.Image.open(path).convert("RGB")
        return image_raw

    def __sample_image_min_preproc(self, index: int | None = None) -> TextInput:
        if index is None:
            index = np.random.randint(0, len(self))
        img = self.__sample_image(index)
        return self.image_preprocessing_first_step(img)

    def __sample_text(self, index: int) -> TextRawInput:
        if not hasattr(self, "_embeddings_file"):
            self._embeddings_file = h5py.File(self._dataset_metadata.embedding_path, "r")
        if not hasattr(self, "_embeddings"):
            self._embeddings = self._embeddings_file[H5_EMBEDDING_DATASET_NAME]
        assert isinstance(self._embeddings, h5py.Dataset)
        id = self._dataset_metadata.ids[index]
        tag_embeddings = torch.tensor(self._embeddings[self._dataset_metadata.id_to_emb_index[id]])
        return tag_embeddings

    def __sample_text_min_preproc(self, index: int | None = None) -> TextInput:
        if index is None:
            index = np.random.randint(0, len(self))
        text = self.__sample_text(index)
        return self.text_preprocessing_first_step(text)

    def __getitem__(self, index: int) -> CrossModalData:
        image_raw = self.__sample_image(index)
        image = self.image_preprocessing_pipeline(image_raw, self.image_augmentations)

        text_raw = self.__sample_text(index)
        text = self.text_preprocessing_pipeline(text_raw, self.text_augmentations)

        id = self._dataset_metadata.ids[index]
        labels = self._dataset_metadata.id_to_labels[id]
        labels_tensor = torch.tensor(labels)
        index_tensor = torch.tensor(self._dataset_metadata.id_to_emb_index[id])

        if self.return_extended_data:
            return ExtendedCrossModalData(
                index=index_tensor,
                image=image,
                text=text,
                label=labels_tensor,
                image_raw=image_raw,
                text_str=" ".join(self._dataset_metadata.id_to_tags[id]),
            )

        return CrossModalData(
            index=index_tensor,
            image=image,
            text=text,
            label=labels_tensor,
        )

    def __getstate__(self) -> dict:
        # pickle cannot handle h5py.File objects
        state = self.__dict__.copy()
        if "_embeddings" in state:
            del state["_embeddings"]  # remove h5py.File object from state
        if "_embeddings_file" in state:
            del state["_embeddings_file"]  # remove h5py.Dataset object from state
        return state

    def _check_dataset_integrity(self) -> bool:
        any_error = False
        Logger().info("[DAT] Do some simple checks to ensure that all necessary files exist.")
        id_set = set(self._dataset_metadata.ids)
        id_set_from_emb = set(self._dataset_metadata.id_to_emb_index.keys())
        id_set_from_labels = set(self._dataset_metadata.id_to_labels.keys())
        id_set_from_path = set(self._dataset_metadata.id_to_path.keys())
        id_set_from_tags = set(self._dataset_metadata.id_to_tags.keys())

        if len(id_set.symmetric_difference(id_set_from_emb)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_emb))} samples without tag embeddings.")
            Logger().warning(f"[DAT] Found {len(id_set_from_emb.difference(id_set))} tag embeddings without reference.")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_labels)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_labels))} samples without labels.")
            Logger().warning(f"[DAT] Found {len(id_set_from_labels.difference(id_set))} labels without reference.")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_path)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_path))} samples without image path.")
            Logger().warning(f"[DAT] Found {len(id_set_from_path.difference(id_set))} image paths without reference.")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_tags)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_tags))} samples without tags.")
            Logger().warning(f"[DAT] Found {len(id_set_from_tags.difference(id_set))} tags without reference.")
            any_error = True

        # Do some sanity checks on the labels
        label_dim = len(self._dataset_metadata.label_names)
        for id, label in self._dataset_metadata.id_to_labels.items():
            if label.shape[0] != label_dim:
                Logger().warning(f"[DAT] Label for sample {id} has incorrect shape: {label.shape}.")
                any_error = True

        if not os.path.exists(self._dataset_metadata.embedding_path):
            Logger().warning("[DAT] Embeddings not found.")
            any_error = True

        return any_error

    def update_augmentation(self, epoch: int) -> None:
        self.image_augmentations.update_augmentation(epoch)
        self.text_augmentations.update_augmentation(epoch)


def create_datasets(
    config: MirFlickr25kConfig,
    env: ModelDataEnvironmentConfig,
    adapter: CrossModalDatasetToModelAdapter,
    modes: set[DatasetMode],
) -> tuple[DatasetForHashLearningInfo, dict[DatasetMode, MirFlickr25kDataset]]:
    env.add_dataset_resolver(lambda: DATASET_NAME)
    base_path = env.resolve(env.data_path)
    Logger().info(f"[DAT] Checking {DATASET_NAME} dataset folder to contain all needed files.")
    image_paths = glob(os.path.join(base_path, "data", "im*.jpg"))
    tag_paths = glob(os.path.join(base_path, "data", "meta", "tags", "tags*.txt"))

    ignore_unused_label_files: Callable[[str], bool] = lambda p: "README" not in p and "_r1." not in p
    label_paths = [*filter(ignore_unused_label_files, glob(os.path.join(base_path, "annotations_v080", "*.txt")))]

    if len(image_paths) != len(tag_paths):
        Logger().warning(f"[DAT] Found {len(image_paths)} images, but {len(tag_paths)} tag files.")

    # Write number of instances only if they differ from expected
    Logger().write(f"[DAT] Found {len(image_paths)} image files.", LogLevel.DEBUG if len(image_paths) == 25000 else LogLevel.INFO)
    Logger().write(f"[DAT] Found {len(tag_paths)} tag files.", LogLevel.DEBUG if len(tag_paths) == 25000 else LogLevel.INFO)
    Logger().write(f"[DAT] Found {len(label_paths)} label files.", LogLevel.DEBUG if len(label_paths) == 24 else LogLevel.INFO)

    max_number_of_samples = len(image_paths)
    ids_to_paths = {int(os.path.basename(p).removeprefix("im").removesuffix(".jpg")): p for p in image_paths}
    ids_to_tags: dict[int, list[str]] = {}
    Logger().info(f"[DAT] Loading tags...")
    for tag_path in tqdm(tag_paths):
        tag_id = int(os.path.basename(tag_path).removeprefix("tags").removesuffix(".txt"))
        with open(tag_path, "r") as tag_file:
            tags = tag_file.readlines()
        ids_to_tags[tag_id] = [t for t in map(_sanitize_tag, tags) if t != None]

    ids = sorted(list(ids_to_paths.keys()))
    ids_ndarray = np.array(ids)
    ids_to_index = {id: idx for idx, id in enumerate(ids)}

    tag_embeddings_path = handle_embeddings(
        adapter,
        base_path,
        H5_EMBEDDING_DATASET_NAME,
        ids,
        ids_to_index,
        ids_to_tags,
    )

    Logger().info(f"[DAT] Loading labels...")
    labels_raw: dict[str, set[int]] = {}
    for label_path in tqdm(label_paths):
        label_name = os.path.basename(label_path).removesuffix(".txt")
        if label_name in labels_raw:
            Logger().warning(f"[DAT] Duplicate label file {label_name}, skipping...")
            continue
        labels_raw[label_name] = set()
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue  # skip empty lines
            s, i = try_parse_int(line)
            if s:
                assert isinstance(i, int)
                labels_raw[label_name].add(i)
            else:
                Logger().warning(f"[DAT] Found invalid image id '{line}' for label {label_name}.")
    label_names = sorted(labels_raw.keys())
    combined_labels = np.zeros((len(ids), len(label_names)), dtype=np.bool)
    for l_idx, label_name in enumerate(label_names):
        idxs = np.where(np.isin(ids_ndarray, list(labels_raw[label_name])))[0]
        combined_labels[idxs, l_idx] = True

    ids_to_remove_set = set()
    if config.only_samples_with_tag_that_occurs_min_k_times is not None:
        # count the occurrences of each tag and filter the samples to contain at least one tag that occurs at least min_k_times times
        all_tags = Counter[str]()
        for tags in ids_to_tags.values():
            all_tags.update(tags)
        all_tags_strings: list[str] = []
        all_tags_counts: list[int] = []
        for tag, count in all_tags.items():
            all_tags_strings.append(tag)
            all_tags_counts.append(count)
        k = config.only_samples_with_tag_that_occurs_min_k_times
        set_of_min_k_occurring_tags = set(all_tags_strings[i] for i in np.argwhere(np.array(all_tags_counts) >= k).flatten())
        Logger().info(f"[DAT] Found {len(set_of_min_k_occurring_tags)} tags that occur at least k={k} times")
        for id, tags in list(ids_to_tags.items()):
            if not any(tag in set_of_min_k_occurring_tags for tag in tags):
                ids_to_remove_set.add(id)
        Logger().info(f"[DAT] Removed {len(ids_to_remove_set)} samples without tags that occur at least k={k} times")
        # Remove tags that occur less than j times if specified
        filter_tags = True
        j = k
        if isinstance(config.remove_tags_that_occur_less_than_j_times, bool):
            filter_tags = config.remove_tags_that_occur_less_than_j_times
        else:
            j = config.remove_tags_that_occur_less_than_j_times
        if filter_tags:
            set_of_min_j_occurring_tags = set(all_tags_strings[i] for i in np.argwhere(np.array(all_tags_counts) >= j).flatten())
            Logger().info(f"[DAT] Found {len(set_of_min_j_occurring_tags)} tags that occur at least j={j} times")
            ids_to_tags = {id: [tag for tag in tags if tag in set_of_min_j_occurring_tags] for id, tags in ids_to_tags.items()}

    if not config.include_samples_without_labels:
        no_labels_ids = ids_ndarray[np.where(combined_labels.sum(axis=1) == 0)[0]].flatten().tolist()
        ids_to_remove_set.update(no_labels_ids)
        Logger().info(f"[DAT] Ignoring {len(no_labels_ids)} samples without labels.")

    if config.min_textual_tags is not None:
        tag_lens = np.array([len(ids_to_tags[id]) for id in ids])
        not_sufficient_tags_ids = ids_ndarray[np.where(tag_lens < config.min_textual_tags)].flatten().tolist()
        Logger().info(
            f"[DAT] Ignoring {len(not_sufficient_tags_ids)} samples failing to meet minimum number of tags requirement of {config.min_textual_tags}."
        )
        ids_to_remove_set.update(not_sufficient_tags_ids)

    final_ids = sorted(set(ids).difference(ids_to_remove_set))
    Logger().info(f"[DAT] Using {len(final_ids)} out of {len(ids)} samples, ignoring {len(ids_to_remove_set)} samples.")

    labels: dict[int, np.ndarray] = {k: combined_labels[np.where(ids_ndarray == k)[0].item()] for k in ids}
    dataset_metadatas: dict[DatasetMode, _MirFlickr25k_Metadata] = {}
    dataset_metadata_factory = _create_subset_metadata_factory(
        paths=ids_to_paths,
        tags=ids_to_tags,
        labels=labels,
        label_names=label_names,
        embedding_indices=ids_to_index,
        embedding_path=tag_embeddings_path,
        max_number_of_samples=max_number_of_samples,
    )
    info = DatasetForHashLearningInfo()
    info.dataset_name = DATASET_NAME
    info.label_indices = list(range(len(label_names)))
    info.labels = label_names
    if isinstance(config.ttr_split, TrainTTSAsDataset):
        raise ValueError(f"{DATASET_NAME} contains no information about train-test-split.")
    elif isinstance(config.ttr_split, TrainTTSByNumbers):
        for mode, subset in handle_tts_by_numbers(config.ttr_split, final_ids, labels).items():
            dataset_metadatas[mode] = dataset_metadata_factory(subset)
    else:
        raise NotImplementedError(f"Unsupported ttr_split type {type(config.ttr_split)}")

    info.train_indices = sorted(list(dataset_metadatas[DatasetMode.TRAIN].id_to_emb_index.values()))
    info.test_indices = sorted(list(dataset_metadatas[DatasetMode.TEST].id_to_emb_index.values()))
    info.test_retrieval_indices = sorted(list(dataset_metadatas[DatasetMode.TEST_RETRIEVAL].id_to_emb_index.values()))
    info.validation_indices = sorted(list(dataset_metadatas[DatasetMode.VAL].id_to_emb_index.values()))
    info.validation_retrieval_indices = sorted(list(dataset_metadatas[DatasetMode.VAL_RETRIEVAL].id_to_emb_index.values()))

    assert len(info.train_indices) > 0, f"Implementation not completed"
    assert len(info.test_indices) > 0, f"Implementation not completed"
    assert len(info.test_retrieval_indices) > 0, f"Implementation not completed"
    assert len(info.validation_indices) > 0, f"Implementation not completed"
    assert len(info.validation_retrieval_indices) > 0, f"Implementation not completed"

    info.train_augmentations = config.augmentation.stringify()

    datasets: dict[DatasetMode, MirFlickr25kDataset] = {}
    for mode in modes:
        datasets[mode] = MirFlickr25kDataset(config, env, mode, adapter, dataset_metadatas[mode])

    return info, datasets


def _sanitize_tag(tag: str) -> str | None:
    tag = tag.strip()
    if "=" in tag:  # ignore e.g. "geo:lat=123456789"
        return None
    return tag


def _create_subset_metadata_factory(
    paths: dict[int, str],
    tags: dict[int, list[str]],
    labels: dict[int, np.ndarray],
    label_names: list[str],
    embedding_indices: dict[int, int],
    embedding_path: str,
    max_number_of_samples: int,
) -> Callable[[list[int]], _MirFlickr25k_Metadata]:
    def _create_subset_metadata(subset_id_list: list[int]) -> _MirFlickr25k_Metadata:
        ids = sorted(subset_id_list)
        return _MirFlickr25k_Metadata(
            ids=ids,
            id_to_path={k: paths[k] for k in ids},
            id_to_tags={k: tags[k] for k in ids},
            id_to_labels={k: labels[k] for k in ids},
            label_names=label_names,
            id_to_emb_index={k: embedding_indices[k] for k in ids},
            embedding_path=embedding_path,
            max_number_of_samples=max_number_of_samples,
        )

    return _create_subset_metadata
