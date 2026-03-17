import os
from typing import Callable, OrderedDict
import numpy as np
import torch

from dsh.embedding.embedder import Embedder, EmbedderInfo
from dsh.embedding.w2vembedder import W2VEmbedder
from dsh.config.env import ModelDataEnvironmentConfig
from dsh.config.model import MLPConfig, TDSRDHModelConfig, TextEmbedderType
from dsh.utils.activation import get_activation_and_init_params
from dsh.utils.initialization import init_weights


def get_hash_mlp(config: MLPConfig, in_dim: int, out_dim: int) -> torch.nn.Module:
    dims = [in_dim, *config.dims, out_dim]
    inner_activation = get_activation_and_init_params(config.activation)
    final_activation = get_activation_and_init_params(config.final_activation)

    layers = OrderedDict[str, torch.nn.Module]()
    # loop over dims to create layers
    i = -1
    for i, (n_in, n_out) in enumerate(zip(dims[:-2], dims[1:-1])):
        # create and add layer to layers dictionary
        layer = torch.nn.Linear(n_in, n_out)
        init_weights(layer, inner_activation.init_type, nonlinearity=inner_activation.nonlinearity)
        layers.update({f"linear{i}": layer})
        layers.update({f"{inner_activation.name}{i}": inner_activation.activation})
        if config.dropout > 0:  # add dropout only if configured with probability > 0
            layers.update({f"dropout{i}": torch.nn.Dropout(p=config.dropout)})

    # create and add final layer to layers dictionary
    fin_layer = torch.nn.Linear(dims[-2], dims[-1])
    init_weights(fin_layer, final_activation.init_type, nonlinearity=final_activation.nonlinearity)
    layers.update({f"linear{i+1}": fin_layer})
    layers.update({f"{final_activation.name}{i+1}": final_activation.activation})
    layers.update({f"{final_activation.name}{i+1}": final_activation.activation})

    return torch.nn.Sequential(layers)


def get_embedder_info_from_config(config: TDSRDHModelConfig, env: ModelDataEnvironmentConfig) -> EmbedderInfo:
    match config.text_embedder:
        case TextEmbedderType.W2V_GN300_FC:
            embedding_dim = 300
        case _:
            raise NotImplementedError(f"Unknown text embedder {self.config.text_embedder}")
    return EmbedderInfo(
        config.text_embedder.value,
        config.text_sequence_length,
        embedding_dim,
        np.float32,
        get_embedder(env, config.text_embedder, config.text_sequence_length),
    )


def get_embedder(
    env: ModelDataEnvironmentConfig,
    t: TextEmbedderType,
    sequence_length: int,
) -> Callable[[], Embedder]:

    if t == TextEmbedderType.W2V_GN300_FC:

        def _get_embedder() -> Embedder:
            path = os.path.join(env.resolve(env.misc_path) + "/GoogleNews-vectors-negative300.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Word2Vec model file not found at {path}")
            return W2VEmbedder.load(path, sequence_length)

        return _get_embedder
    else:
        raise NotImplementedError(f"Unknown text embedder type {t}")
