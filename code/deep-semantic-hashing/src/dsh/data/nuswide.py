from glob import glob
import PIL.Image
import h5py
import numpy as np
import os
import torch
from typing import Callable, Final, NamedTuple

from dsh.data.common import handle_embeddings, handle_tts_by_numbers
from dsh.data.dataset import CrossModalDataset, DatasetMode
from dsh.config.data import NUSWideConfig, TrainTTSByNumbers, TrainTTSAsDataset
from dsh.config.env import ModelDataEnvironmentConfig
from dsh.utils.adapter import CrossModalDatasetToModelAdapter
from dsh.utils.augmentation import (
    get_image_augmentation,
    get_text_augmentation,
    InertImageAugmentation,
    InertTextAugmentation,
)
from dsh.utils.collections import find_duplicates
from dsh.utils.logger import Logger, LogLevel
from dsh.utils.progress import tqdm
from dsh.utils.types import (
    CrossModalData,
    DatasetForHashLearningInfo,
    DynamicImageAugmentation,
    DynamicTextAugmentation,
    ExtendedCrossModalData,
    ImageInput,
    ImageRawInput,
    NoopTensorAugmentation,
    TextRawInput,
)


KNOWN_DUPLICATE_IMAGE_IDS: Final = [
    "702409954",
    "2728487708",
    "815043568",
    "1100787682",
    "2729498990",
    "2230197395",
]

H5_EMBEDDING_DATASET_NAME: Final = "nuswide_embeddings"
DATASET_NAME: Final = "NUS-WIDE"


class _NUSWide_Metadata(NamedTuple):
    ids: list[str]
    id_to_name: dict[str, str]
    id_to_tags: dict[str, list[str]]
    id_to_emb_idx: dict[str, int]
    name_to_path: dict[str, str]
    id_to_label: dict[str, np.ndarray]
    label_names: list[str]
    label_indices: list[int]
    embedding_path: str
    max_number_of_samples: int


class NUSWideDataset(CrossModalDataset[NUSWideConfig, CrossModalData]):
    def __init__(
        self,
        config: NUSWideConfig,
        env: ModelDataEnvironmentConfig,
        mode: DatasetMode,
        adapter: CrossModalDatasetToModelAdapter,
        metadata: _NUSWide_Metadata,
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
        name = self._dataset_metadata.id_to_name[id]
        path = self._dataset_metadata.name_to_path[name]
        image_raw = PIL.Image.open(path).convert("RGB")
        return image_raw

    def __sample_image_min_preproc(self, index: int | None = None) -> ImageInput:
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
        tag_embeddings = torch.tensor(self._embeddings[self._dataset_metadata.id_to_emb_idx[id]])
        return tag_embeddings

    def __sample_text_min_preproc(self, index: int | None = None) -> TextRawInput:
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
        labels = self._dataset_metadata.id_to_label[id]
        labels_tensor = torch.tensor(labels)
        index_tensor = torch.tensor(self._dataset_metadata.id_to_emb_idx[id])

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
        if len(id_set) != len(self._dataset_metadata.ids):
            duplicate_images = find_duplicates(self._dataset_metadata.ids)
            if len(set(duplicate_images.keys()).difference(set(KNOWN_DUPLICATE_IMAGE_IDS))) == 0:
                # known duplicates
                Logger().info(f"[DAT] Found known duplicate images: {[*duplicate_images.keys()]}.")
            else:
                Logger().warning(
                    f"[DAT] Found {len(id_set)} unique image ids, but the original list contains {len(self._dataset_metadata.ids)} elements."
                    + " This discrepancy might indicate an error in the dataset configuration or data loading process."
                )
                for img in duplicate_images.keys():
                    Logger().info(f"[DAT] Duplicate image: {img}.")
                any_error = True

        id_set_from_tags = set(self._dataset_metadata.id_to_tags.keys())
        id_set_from_tag_embeddings = set(self._dataset_metadata.id_to_emb_idx.keys())
        id_set_from_names = set(self._dataset_metadata.id_to_name.keys())
        id_set_from_labels = set(self._dataset_metadata.id_to_label.keys())
        name_set_from_names = set(self._dataset_metadata.id_to_name.values())
        name_set_from_paths = set(self._dataset_metadata.name_to_path.keys())

        if len(id_set.symmetric_difference(id_set_from_tags)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_tags))} images without tags.")
            Logger().warning(f"[DAT] Found {len(id_set_from_tags.difference(id_set))} tags without valid image.")
            for id in id_set.symmetric_difference(id_set_from_tags):
                Logger().info(f"[DAT] {id}")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_tag_embeddings)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_tag_embeddings))} images without tag embeddings.")
            Logger().warning(
                f"[DAT] Found {len(id_set_from_tag_embeddings.difference(id_set))} tag embeddings without valid image."
            )
            for id in id_set.symmetric_difference(id_set_from_tag_embeddings):
                Logger().info(f"[DAT] {id}")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_names)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_names))} images without corresponding name.")
            Logger().warning(f"[DAT] Found {len(id_set_from_names.difference(id_set))} image names without valid id.")
            any_error = True

        if len(id_set.symmetric_difference(id_set_from_labels)) > 0:
            Logger().warning(f"[DAT] Found {len(id_set.difference(id_set_from_labels))} images without labels.")
            Logger().warning(f"[DAT] Found {len(id_set_from_labels.difference(id_set))} label entries without valid id.")
            any_error = True

        if len(name_set_from_names.symmetric_difference(name_set_from_paths)) > 0:
            # expected to be len(KNOWN_DUPLICATE_IMAGE_IDS)
            difference_names = name_set_from_paths.difference(name_set_from_names)
            if len(difference_names) == len(KNOWN_DUPLICATE_IMAGE_IDS) and all(
                any(k in dn for dn in difference_names) for k in KNOWN_DUPLICATE_IMAGE_IDS
            ):
                Logger().info(f"[DAT] Found known duplicate images: {difference_names}.")
            else:
                Logger().warning(f"[DAT] Found {len(name_set_from_names.difference(name_set_from_paths))} missing images.")
                Logger().warning(f"[DAT] Found {len(name_set_from_paths.difference(name_set_from_names))} unreferenced images.")
                any_error = True

        # Do some sanity checks on the labels
        label_dim = len(self._dataset_metadata.label_names)
        for id, label in self._dataset_metadata.id_to_label.items():
            if label.shape[0] != label_dim:
                Logger().warning(f"[DAT] Label for image {id} has incorrect shape: {label.shape}.")
                any_error = True

        if not os.path.exists(self._dataset_metadata.embedding_path):
            Logger().warning("[DAT] Embeddings not found.")
            any_error = True

        return any_error

    def update_augmentation(self, epoch: int) -> None:
        self.image_augmentations.update_augmentation(epoch)
        self.text_augmentations.update_augmentation(epoch)


def create_datasets(
    config: NUSWideConfig,
    env: ModelDataEnvironmentConfig,
    adapter: CrossModalDatasetToModelAdapter,
    modes: set[DatasetMode],
) -> tuple[DatasetForHashLearningInfo, dict[DatasetMode, NUSWideDataset]]:
    env.add_dataset_resolver(lambda: DATASET_NAME)
    base_path = env.resolve(env.data_path)
    Logger().info(f"[DAT] Checking NUS-WIDE dataset folder to contain all needed files.")
    images_paths = [
        *glob(os.path.join(base_path, "kaggle", "images", "*.jpg")),
        # *glob(os.path.join(base_path, "kaggle", "images", "*.png")), # ignore PNG files; they are duplicates anyways
    ]

    image_list_path = os.path.join(base_path, "ImageList", "Imagelist.txt")
    train_image_list_path = os.path.join(base_path, "ImageList", "TrainImagelist.txt")
    test_image_list_path = os.path.join(base_path, "ImageList", "TestImagelist.txt")
    tag_list_path = os.path.join(base_path, "NUS_WID_Tags", "All_Tags.txt")
    labels_paths = glob(os.path.join(base_path, "Groundtruth", "AllLabels", "*.txt"))

    if not os.path.exists(image_list_path):
        Logger().error(f"[DAT] Missing crucial file {image_list_path}.")
        exit(-2)
    if not os.path.exists(train_image_list_path):
        Logger().error(f"[DAT] Missing crucial file {train_image_list_path}.")
        exit(-2)
    if not os.path.exists(test_image_list_path):
        Logger().error(f"[DAT] Missing crucial file {test_image_list_path}.")
        exit(-2)
    if not os.path.exists(tag_list_path):
        Logger().error(f"[DAT] Missing crucial file {tag_list_path}.")
        exit(-2)

    with open(image_list_path, "r") as f:
        image_list_raw = f.readlines()

    image_id_list = [_extract_image_id_from_name(line) for line in image_list_raw]
    # todo image_list ? -> order in labels

    with open(tag_list_path, "r") as f:
        tag_list = f.readlines()

    # Write number of instances only if they differ from expected
    # fmt: off
    Logger().write(f"[DAT] Found {len(images_paths)} image files.", LogLevel.DEBUG if len(images_paths) == 269648 else LogLevel.INFO)
    Logger().write(f"[DAT] Found {len(image_id_list)} referenced images.", LogLevel.DEBUG if len(image_id_list) == 269648 else LogLevel.INFO)
    Logger().write(f"[DAT] Found {len(tag_list)} corresponding tags.", LogLevel.DEBUG if len(tag_list) == 269648 else LogLevel.INFO)
    Logger().write(f"[DAT] Found {len(labels_paths)} label files.", LogLevel.DEBUG if len(labels_paths) == 81 else LogLevel.INFO)
    # fmt: on
    max_number_of_samples = len(image_id_list)

    image_name_to_path = {os.path.basename(p): p for p in images_paths}
    image_id_to_name = {
        # image_filename has the following form: 0000_123..789.jpg; we want the variable-length 123...789 part
        image_filename[image_filename.index("_") + 1 : -4]: image_filename
        for image_filename in image_name_to_path.keys()
    }

    # each line in tag_list contains the image id first, then 6 spaces
    image_id_to_tags: dict[str, list[str]] = {}
    for tag_line in tag_list:
        primary_split = tag_line.split(" " * 6, maxsplit=1)  # six spaces
        id = primary_split[0].strip()
        if len(primary_split) > 1:
            tags = primary_split[1].strip().split(" ")  # single spaces
        else:
            tags = []
        image_id_to_tags[id] = tags

    image_id_to_index: dict[str, int] = {id: i for i, id in enumerate(image_id_list)}
    tag_embeddings_path = handle_embeddings(
        adapter,
        base_path,
        H5_EMBEDDING_DATASET_NAME,
        image_id_list,
        image_id_to_index,
        image_id_to_tags,
    )

    # load labels
    Logger().info(f"[DAT] Loading labels...")
    labels_raw: dict[str, list[bool]] = {}
    for p in tqdm(labels_paths):
        # path is constructed as follows: .../Labels_<label>.txt
        label_name = p[p.rfind("_") + 1 : -4]
        if label_name in labels_raw:
            Logger().warning(f"[DAT] Duplicate label file {label_name}, skipping...")
            continue
        labels_raw[label_name] = []
        with open(p, "r") as f:
            lines = f.readlines()
        if len(lines) != len(image_id_list):
            Logger().error(f"[DAT] Label file {p} has a different number of images than the dataset.")
        for line in lines:
            match line.strip():
                case "0":
                    labels_raw[label_name].append(False)
                case "1":
                    labels_raw[label_name].append(True)
                case _:
                    Logger().warning(f"[DAT] Invalid label value in {p}: {line.strip()}.")
                    labels_raw[label_name].append(False)

    label_names = sorted(labels_raw.keys())
    label_data: list[np.ndarray] = []
    for label_name in label_names:
        label_data.append(np.array(labels_raw[label_name]))
    label_indices = list(range(len(label_names)))
    # combine all labels into one array and then transpose it so each image has a row of labels
    combined_labels = np.array(label_data).T
    # select top k labels
    if config.top_k_labels != None:
        k = config.top_k_labels
        instances_per_label = combined_labels.sum(axis=0)
        top_k_label_indices = np.argsort(instances_per_label)[::-1][:k]
        combined_labels = combined_labels[:, top_k_label_indices]
        label_names = [label_names[idx] for idx in top_k_label_indices]
        label_indices = [int(i) for i in top_k_label_indices]

    # now slice it into an array per image
    image_id_to_labels: dict[str, np.ndarray] = {}
    unused_image_ids: set[str] = set()
    for i, id in enumerate(image_id_list):
        if id not in image_id_to_labels:
            if any(combined_labels[i]) or config.include_samples_without_labels:
                image_id_to_labels[id] = combined_labels[i]
            else:
                unused_image_ids.add(id)
        else:
            if id not in KNOWN_DUPLICATE_IMAGE_IDS:
                Logger().warning(f"[DAT] Unknown duplicate image ID found: {id}.")
            # check if the labels are the same, and if not, log a warning
            if not np.array_equal(image_id_to_labels[id], combined_labels[i]):
                # find the indices where the labels differ and log them
                differing_indices = np.where(image_id_to_labels[id] != combined_labels[i])[0]
                differing_label_names = [label_names[idx] for idx in differing_indices]
                Logger().warning(f"[DAT] Duplicate image ID found with different labels: {id}; {differing_label_names}.")

    Logger().info(
        f"[DAT] Using {len(image_id_to_labels)} out of {len(image_id_list)} images, ignoring {len(unused_image_ids)} images without labels."
    )
    assert not config.include_samples_without_labels | len(unused_image_ids) == 0, "Internal implementation error"
    # now remove all unused image ids from dataset
    image_id_list = [i for i in image_id_list if i not in unused_image_ids]
    for id in unused_image_ids:
        name = image_id_to_name[id]
        del image_id_to_name[id]
        del image_id_to_tags[id]
        del image_id_to_index[id]
        del image_name_to_path[name]

    # depending on the train-test-(retrieval) split, we need to create different subsets of the data
    dataset_metadatas: dict[DatasetMode, _NUSWide_Metadata] = {}
    dataset_metadata_factory = _create_subset_metadata_factory(
        image_id_list,
        image_id_to_name,
        image_id_to_tags,
        image_id_to_index,
        image_name_to_path,
        label_names,
        label_indices,
        image_id_to_labels,
        tag_embeddings_path,
        max_number_of_samples,
    )
    info = DatasetForHashLearningInfo()
    info.dataset_name = DATASET_NAME
    info.label_indices = label_indices
    info.labels = label_names
    if isinstance(config.ttr_split, TrainTTSAsDataset):
        for mode in DatasetMode:
            mode_id_list = _handle_tts_as_dataset(
                config.ttr_split,
                mode,
                image_id_list,
                train_image_list_path,
                test_image_list_path,
            )
            dataset_metadatas[mode] = dataset_metadata_factory(mode_id_list)
    elif isinstance(config.ttr_split, TrainTTSByNumbers):
        for mode, subset in handle_tts_by_numbers(config.ttr_split, image_id_list, image_id_to_labels).items():
            dataset_metadatas[mode] = dataset_metadata_factory(subset)
    else:
        raise NotImplementedError(f"Unsupported ttr_split type {type(config.ttr_split)}")

    info.train_indices = sorted(list(dataset_metadatas[DatasetMode.TRAIN].id_to_emb_idx.values()))
    info.test_indices = sorted(list(dataset_metadatas[DatasetMode.TEST].id_to_emb_idx.values()))
    info.test_retrieval_indices = sorted(list(dataset_metadatas[DatasetMode.TEST_RETRIEVAL].id_to_emb_idx.values()))
    info.validation_indices = sorted(list(dataset_metadatas[DatasetMode.VAL].id_to_emb_idx.values()))
    info.validation_retrieval_indices = sorted(list(dataset_metadatas[DatasetMode.VAL_RETRIEVAL].id_to_emb_idx.values()))

    assert len(info.train_indices) > 0, f"Implementation not completed"
    assert len(info.test_indices) > 0, f"Implementation not completed"
    assert len(info.test_retrieval_indices) > 0, f"Implementation not completed"
    assert len(info.validation_indices) > 0, f"Implementation not completed"
    assert len(info.validation_retrieval_indices) > 0, f"Implementation not completed"

    info.train_augmentations = config.augmentation.stringify()

    datasets: dict[DatasetMode, NUSWideDataset] = {}
    for mode in modes:
        datasets[mode] = NUSWideDataset(config, env, mode, adapter, dataset_metadatas[mode])

    return info, datasets


def _create_subset_metadata_factory(
    image_id_list: list[str],
    image_id_to_name: dict[str, str],
    image_id_to_tags: dict[str, list[str]],
    image_id_to_index: dict[str, int],
    image_name_to_path: dict[str, str],
    label_names: list[str],
    label_indices: list[int],
    image_id_to_labels: dict[str, np.ndarray],
    tag_embeddings_path: str,
    max_number_of_samples: int,
) -> Callable[[list[str]], _NUSWide_Metadata]:
    def _create_subset_metadata(subset_image_id_list: list[str]) -> _NUSWide_Metadata:
        id_set = set(subset_image_id_list)
        id_to_name = {k: v for k, v in image_id_to_name.items() if k in id_set}
        return _NUSWide_Metadata(
            ids=[i for i in image_id_list if i in id_set],
            id_to_name=id_to_name,
            id_to_tags={k: v for k, v in image_id_to_tags.items() if k in id_set},
            id_to_emb_idx={k: v for k, v in image_id_to_index.items() if k in id_set},
            name_to_path={name: image_name_to_path[name] for name in id_to_name.values()},
            label_names=label_names,
            label_indices=label_indices,
            id_to_label={k: v for k, v in image_id_to_labels.items() if k in id_set},
            embedding_path=tag_embeddings_path,
            max_number_of_samples=max_number_of_samples,
        )

    return _create_subset_metadata


def _handle_tts_as_dataset(
    config: TrainTTSAsDataset,
    mode: DatasetMode,
    image_id_list: list[str],
    train_image_list_path: str,
    test_image_list_path: str,
) -> list[str]:
    if mode == DatasetMode.FULL:
        # Return everything if we are doing full dataset
        return image_id_list

    if mode == DatasetMode.TRAIN or mode == DatasetMode.TEST_RETRIEVAL or mode == DatasetMode.VAL_RETRIEVAL:
        id_path = train_image_list_path
    elif mode == DatasetMode.TEST or mode == DatasetMode.VAL:
        id_path = test_image_list_path
    else:
        raise NotImplementedError(f"Unsupported dataset mode {mode}")
    with open(id_path, "r") as f:
        relevant_image_ids_raw = f.readlines()
    relevant_image_id_set = set([_extract_image_id_from_name(line) for line in relevant_image_ids_raw])
    # now remove all images not in the relevant set
    filtered_image_id_list = [i for i in image_id_list if i in relevant_image_id_set]
    return filtered_image_id_list


def _extract_image_id_from_name(line: str) -> str:
    # image_list_raw contains filenames in the format: <category>\0000_123...789.jpg; we want the variable-length 123...789 part
    return line[line.rfind("_") + 1 :].strip()[:-4]
