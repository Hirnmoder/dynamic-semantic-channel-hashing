from functools import partial
from tqdm.std import tqdm as _tqdm


__all__ = ["tqdm"]


tqdm = partial(_tqdm, ncols=100, dynamic_ncols=True)
