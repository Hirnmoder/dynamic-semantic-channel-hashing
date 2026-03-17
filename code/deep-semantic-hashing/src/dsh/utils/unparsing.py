from enum import Enum

from dsh.utils.types import HPARAM

__all__ = [
    "stringify",
    "stringify_img_txt",
]


def stringify_img_txt(img: HPARAM, txt: HPARAM) -> str:
    return f"(I={img}, T={txt})"


def stringify(name: str | None, arg: HPARAM | Enum | list[HPARAM] | None = None, **kwargs: HPARAM | Enum | list[HPARAM]) -> str:
    if arg is not None:
        assert name is None, "Cannot provide both positional arguments and name."
        assert len(kwargs) == 0, "Cannot provide both positional arguments and keyword arguments."
        if isinstance(arg, list):
            return f"[{', '.join(stringify(None, _) for _ in arg)}]"
        elif isinstance(arg, Enum):
            return arg.value
        else:
            return str(arg)
    else:
        element_str = ", ".join(
            [
                f"{k}={stringify(None, v)}"
                for k, v in kwargs.items()
                if not k.startswith("_") and isinstance(v, HPARAM | Enum | list)
            ]
        )
        if name is None:
            return element_str
        else:
            return f"{name}({element_str})"
