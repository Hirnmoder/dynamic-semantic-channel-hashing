from typing import Callable, Literal, TypeVar


def try_parse_int(s: str) -> tuple[Literal[True], int] | tuple[Literal[False], str]:
    try:
        return True, int(s)
    except ValueError:
        return False, s


def try_parse_float(s: str) -> tuple[Literal[True], float] | tuple[Literal[False], str]:
    try:
        return True, float(s)
    except ValueError:
        return False, s


def try_parse_bool(s: str) -> tuple[Literal[True], bool] | tuple[Literal[False], str]:
    if s.lower() in ["true", "false"]:
        return True, s.lower() == "true"
    else:
        return False, s


def try_parse(s: str) -> str | int | float | bool:
    for func in [
        try_parse_bool,
        try_parse_int,
        try_parse_float,
    ]:
        is_valid, parsed = func(s)
        if is_valid:
            return parsed
    return s


D = TypeVar("D")
R = TypeVar("R")


def parse_default(s: str, default: D, func: Callable[[str], tuple[Literal[True], R] | tuple[Literal[False], str]]) -> R | D:
    is_valid, parsed = func(s)
    if is_valid:
        return parsed  # type: ignore # func is guaranteed to return a valid value if it returns True
    else:
        return default


def parse_default_int(s: str, default: D) -> int | D:
    return parse_default(s, default, try_parse_int)


def parse_default_float(s: str, default: D) -> float | D:
    return parse_default(s, default, try_parse_float)


def parse_default_bool(s: str, default: D) -> bool | D:
    return parse_default(s, default, try_parse_bool)
