import random as r
import numpy as np
import torch
from typing import Sequence, TypeVar, cast, overload


def get_generator(seed: int | None) -> torch.Generator:
    g = torch.Generator()
    if seed is None:
        g.seed()
    else:
        g = g.manual_seed(seed)
    return g


T = TypeVar("T")


# fmt:off
@overload
def random_split(data: Sequence[T], lens: tuple[int], seed: int | None = None) -> tuple[list[T]]: ...
@overload
def random_split(data: Sequence[T], lens: tuple[int, int], seed: int | None = None) -> tuple[list[T], list[T]]: ...
@overload
def random_split(data: Sequence[T], lens: tuple[int, int, int], seed: int | None = None) -> tuple[list[T], list[T], list[T]]: ...
# fmt: on
def random_split(data: Sequence[T], lens: tuple[int, ...], seed: int | None = None) -> tuple[list[T], ...]:
    g = get_generator(seed)
    splitted_data = tuple(
        [
            cast(list[T], [*subset])
            for subset in torch.utils.data.random_split(cast(torch.utils.data.Dataset, data), lens, generator=g)
        ]
    )
    return splitted_data


def random_sample(data: Sequence[T], num_samples: int, seed: int | None = None) -> list[T]:
    return random_split(data, (num_samples, len(data) - num_samples), seed)[0]


def random_sample_by_class(
    data: Sequence[T],
    classes_onehot: np.ndarray,
    samples_per_class: int,
    seed: int | None = None,
) -> tuple[list[T], list[T]]:
    n, c = classes_onehot.shape
    assert len(data) == n, f"Length of data ({len(data)}) does not match length of classes_onehot ({n})"
    num_per_class = classes_onehot.sum(axis=0)
    assert (num_per_class >= samples_per_class).all(), f"There are classes that contain fewer than {samples_per_class} samples"

    sampled_indices = set[int]()
    for i in np.argsort(num_per_class, stable=True):
        sample_indices_for_class = np.argwhere(classes_onehot[:, i] == True).flatten().tolist()
        possible_indices = list(set(sample_indices_for_class) - sampled_indices)
        assert len(possible_indices) >= samples_per_class, f"Class {i} has only {len(possible_indices)} samples left"
        sampled_indices_of_class = random_sample(possible_indices, samples_per_class, seed=seed)
        sampled_indices.update(set(sampled_indices_of_class))

    remaining_indices = list(set(range(len(data))) - sampled_indices)
    sampled_indices = list(sampled_indices)
    assert len(sampled_indices) == c * samples_per_class, f"Expected {c*samples_per_class} samples but got {len(sampled_indices)}"

    sampled_data: list[T] = []
    for index in sampled_indices:
        sampled_data.append(data[index])

    remaining_data: list[T] = []
    for index in remaining_indices:
        remaining_data.append(data[index])

    return sampled_data, remaining_data


def random_hex_string(length: int = 8) -> str:
    return "".join([f"{r.choice(range(256)):02X}" for _ in range(length)])
