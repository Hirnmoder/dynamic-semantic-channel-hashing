from enum import Enum
from typing import Any, TypeVar

from dsh.utils.logger import Logger

T = TypeVar("T", bound=Enum)


def resolve_value_case_insensitive(cls: type[T], value: str | Any) -> T | None:
    Logger().warning(f'Trying to find an enum member with the given value "{value}".')
    if isinstance(value, str):
        # try finding a member by its name case insensitive
        for _, member in cls.__members__.items():
            if value.lower() == member.value.lower():
                Logger().info(f'Found an enum member with the given name case insensitive "{value}" -> "{repr(member)}".')
                return member
        Logger().error(f'Could not find an enum member with the given value "{value}".')


def stringify(*values: Enum, sep: str = ", ") -> str:
    """Converts a list of enums to a comma separated string."""
    return sep.join([str(v.value) for v in values])
