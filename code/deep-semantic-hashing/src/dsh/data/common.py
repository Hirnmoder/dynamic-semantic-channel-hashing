import os
from typing import Final, TypeVar

import h5py
import numpy as np

from dsh.config.data import TTSSamplingMode, TrainTTSByNumbers, TTSRetrievalMode
from dsh.data.dataset import DatasetMode
from dsh.utils.adapter import CrossModalDatasetToModelAdapter
from dsh.utils.logger import Logger
from dsh.utils.progress import tqdm
from dsh.utils.random import random_sample, random_sample_by_class, random_split

_T = TypeVar("_T", covariant=True)


def handle_tts_by_numbers(
    s: TrainTTSByNumbers,
    id_list: list[_T],
    classes_onehot: dict[_T, np.ndarray],
) -> dict[DatasetMode, list[_T]]:
    n_train = int(s.num_train * len(id_list)) if isinstance(s.num_train, float) else s.num_train
    n_test = int(s.num_test * len(id_list)) if isinstance(s.num_test, float) else s.num_test
    n_val = int(s.num_val * len(id_list)) if isinstance(s.num_val, float) else s.num_val
    assert n_train > 0 and n_train <= len(id_list), f"Number of training samples {n_train} is invalid."
    assert n_test > 0 and n_test <= len(id_list), f"Number of test samples {n_test} is invalid."
    assert n_val > 0 and n_val <= len(id_list), f"Number of validation samples {n_val} is invalid."

    if s.sampling_mode == TTSSamplingMode.RANDOM:
        # split the dataset into test/not-test based on the number of samples
        test_id_list, nottest_id_list = random_split(id_list, (n_test, len(id_list) - n_test), s.seed)
        # split not-test into train/not-train based on the number of samples
        train_id_list, nottrain_id_list = random_split(nottest_id_list, (n_train, len(nottest_id_list) - n_train), s.seed)
        # sample from not-train to get the validation set
        val_id_list, notval_id_list = random_split(nottrain_id_list, (n_val, len(nottrain_id_list) - n_val), s.seed)
    elif s.sampling_mode == TTSSamplingMode.N_PER_CLASS:
        num_class: int = next(iter(classes_onehot.values())).shape[0]
        num_test_per_class = int(np.round(s.num_test / num_class))
        num_train_per_class = int(np.round(s.num_train / num_class))
        num_val_per_class = int(np.round(s.num_val / num_class))
        # split the dataset into test/not-test based on the number of samples
        test_id_list, nottest_id_list = random_sample_by_class(
            id_list,
            np.stack([classes_onehot[i] for i in id_list]),
            num_test_per_class,
            s.seed,
        )
        # split not-test into train/not-train based on the number of samples
        train_id_list, nottrain_id_list = random_sample_by_class(
            nottest_id_list,
            np.stack([classes_onehot[i] for i in nottest_id_list]),
            num_train_per_class,
            s.seed,
        )
        # sample from not-train to get the validation set
        val_id_list, notval_id_list = random_sample_by_class(
            nottrain_id_list,
            np.stack([classes_onehot[i] for i in nottrain_id_list]),
            num_val_per_class,
            s.seed,
        )
    elif s.sampling_mode == TTSSamplingMode.ITERATIVE_STRATIFICATION:
        raise NotImplementedError()
    else:
        raise ValueError(f"Invalid sampling mode: {s.sampling_mode}")

    if isinstance(s.num_retrieval, TTSRetrievalMode):
        if s.num_retrieval == TTSRetrievalMode.ALL:
            test_retrieval_id_list = id_list[:]  # copy
            val_retrieval_id_list = id_list[:]  # copy
        elif s.num_retrieval == TTSRetrievalMode.ALL_WITHOUT_TEST:
            test_retrieval_id_list = nottest_id_list[:]  # copy
            val_retrieval_id_list = notval_id_list[:]  # copy
        elif s.num_retrieval == TTSRetrievalMode.ALL_WITHOUT_TRAIN_AND_TEST:
            test_retrieval_id_list = nottrain_id_list[:]  # copy
            val_retrieval_id_list = notval_id_list[:]  # copy
        else:
            raise NotImplementedError(f"Unsupported TTSRetrievalMode {s.num_retrieval}")
    else:
        n_retrieval = int(s.num_retrieval * len(id_list)) if isinstance(s.num_retrieval, float) else s.num_retrieval
        test_retrieval_id_list = random_sample(nottest_id_list, n_retrieval, s.seed)
        val_retrieval_id_list = random_sample(notval_id_list, n_retrieval, s.seed)
    assert len(test_retrieval_id_list) > 0 and len(test_retrieval_id_list) <= len(id_list), f"Number of test retrieval samples {len(test_retrieval_id_list)} is invalid." # fmt: skip
    assert len(val_retrieval_id_list) > 0 and len(val_retrieval_id_list) <= len(id_list), f"Number of val retrieval samples {len(val_retrieval_id_list)} is invalid." # fmt: skip

    dataset_subsets: dict[DatasetMode, list[_T]] = {
        DatasetMode.FULL: id_list,
        DatasetMode.TRAIN: train_id_list,
        DatasetMode.TEST: test_id_list,
        DatasetMode.TEST_RETRIEVAL: test_retrieval_id_list,
        DatasetMode.VAL: val_id_list,
        DatasetMode.VAL_RETRIEVAL: val_retrieval_id_list,
    }
    return dataset_subsets


def handle_embeddings(
    adapter: CrossModalDatasetToModelAdapter,
    base_path: str,
    h5_embedding_dataset_name: str,
    id_list: list[_T],
    id_to_index: dict[_T, int],
    id_to_tags: dict[_T, list[str]],
) -> str:
    sequence_length: Final = adapter.embedder_info.sequence_length
    embed_dim: Final = adapter.embedder_info.embedding_dim

    tag_embeddings_path = os.path.join(base_path, f"{adapter.embedder_info.name}.{sequence_length}.h5")
    Logger().info(f"[DAT] Checking for existing embeddings and computing them if necessary.")
    if not os.path.exists(tag_embeddings_path):
        with h5py.File(tag_embeddings_path, "w") as f:
            embeddings = f.create_dataset(
                h5_embedding_dataset_name,
                (len(id_list), sequence_length, embed_dim),
                dtype=adapter.embedder_info.dtype,
            )
            embedder = adapter.embedder_info.get_embedder()
            for id in tqdm(id_list):
                i = id_to_index[id]
                tags = id_to_tags[id]
                e = embedder(" ".join(tags))
                assert e.shape == (sequence_length, embed_dim)
                embeddings[i] = e
    # do some sanity check
    with h5py.File(tag_embeddings_path, "r") as f:
        embeddings = f[h5_embedding_dataset_name]
        assert isinstance(embeddings, h5py.Dataset)
        assert embeddings.shape == (len(id_list), sequence_length, embed_dim)

    return tag_embeddings_path
