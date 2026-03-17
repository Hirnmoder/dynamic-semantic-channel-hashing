from abc import ABC, ABCMeta
import inspect
import os
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    cast,
    get_args,
    _UnionGenericAlias,  # type: ignore # it does exist
    _LiteralGenericAlias,  # type: ignore # it does exist
)
from typing_extensions import TypeAliasType
from types import UnionType, GenericAlias
import json5
from enum import Enum, IntFlag

from dsh.utils.logger import Logger, LogLevel
from dsh.utils.enumextensions import resolve_value_case_insensitive
from dsh.utils.types import Constants

T = TypeVar("T", bound="ConfigurationBase")


class ConfigurationBaseMeta(ABCMeta):
    _defer_post_init = False
    _deferred_calls: list["ConfigurationBase"] = []

    def __call__(self, *args, **kwargs):
        obj: ConfigurationBase = super(ConfigurationBaseMeta, self).__call__(*args, **kwargs)
        if not self._defer_post_init:
            obj._post_init()
        else:
            ConfigurationBaseMeta._deferred_calls.append(obj)
        return obj

    @staticmethod
    def defer_post_init(*, cancel: bool = False):
        class Stub:
            def __enter__(self, *args, **kwargs):
                Logger().debug("[CFG] Suppressing post init calls for configuration class instances.")
                if len(ConfigurationBaseMeta._deferred_calls) != 0:
                    Logger().warning(
                        f"[CFG] Unhandled deferred post_init calls found: {len(ConfigurationBaseMeta._deferred_calls)}"
                    )
                ConfigurationBaseMeta._defer_post_init = True

            def __exit__(self, *args, **kwargs):
                if cancel:
                    Logger().debug("[CFG] Cancelling post init calls for configuration class instances.")
                    while len(ConfigurationBaseMeta._deferred_calls) > 0:
                        ConfigurationBaseMeta._deferred_calls.pop()
                else:
                    Logger().debug("[CFG] Resuming post init calls for configuration class instances.")
                    while len(ConfigurationBaseMeta._deferred_calls) > 0:
                        obj = ConfigurationBaseMeta._deferred_calls.pop()
                        obj._post_init()
                ConfigurationBaseMeta._defer_post_init = False

        return Stub()


class ConfigurationBase(metaclass=ConfigurationBaseMeta):
    def __init__(self, **kwargs: Any):
        pass

    @classmethod
    def register_type(cls: type[T], typename: str | None = None) -> None:
        return ConfigurationCodecSettings.register_type(cls, typename)

    @classmethod
    def loadf(cls: type[T], filename: str, allow_subclass: bool = False) -> Optional[T]:
        with open(filename, "rt", encoding="utf-8") as f:
            config_str = f.read()
        return cast(Optional[T], cast(ConfigurationBase, cls).loads(config_str, allow_subclass))

    @classmethod
    def loads(cls: type[T], config_str: str, allow_subclass: bool = False) -> Optional[T]:
        with ConfigurationBaseMeta.defer_post_init():
            config_obj = cast(
                Any,
                json5.loads(
                    config_str,
                    allow_duplicate_keys=ConfigurationCodecSettings.allow_duplicate_keys,
                    object_hook=ConfigurationDecoder(),
                ),
            )
            if type(config_obj) == cls or (allow_subclass and isinstance(config_obj, cls)):
                return config_obj
            Logger().error(f"[CFG] Expected object of type {cls}, got {type(config_obj)}.")
            return None

    def dumpf(self, filename: str) -> None:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        config_str = self.dumps()
        with open(filename, "wt", encoding="utf-8") as f:
            f.write(config_str)

    def dumps(self, **kwargs: Any) -> str:
        return json5.dumps(  # type: ignore (bug in typing annotation)
            self,
            allow_duplicate_keys=ConfigurationCodecSettings.allow_duplicate_keys,
            indent=ConfigurationCodecSettings.indent,
            cls=ConfigurationEncoder,  # type: ignore (bug in typing annotation)
            **kwargs,
        )

    def _post_init(self) -> None:
        Logger().debug(f"[CFG] Post init called on configuration object of type {type(self)}.")
        for prop in self.__dict__.keys():
            value = self.__dict__[prop]
            if isinstance(value, Field):
                Logger().debug(f"[CFG] Setting default value for property '{prop}' to '{value.default}'.")
                self.__dict__[prop] = value.default


class ConfigurationCodecSettings:
    _registered_types: dict[str, type] = {}
    _registered_types_inv: dict[type, list[str]] = {}
    allow_duplicate_keys = False
    indent = 4

    def __init__(self):
        raise RuntimeError("You must not instantiate static class ConfigurationCodecSettings")

    @staticmethod
    def register_type(t: type[ConfigurationBase], typename: str | None = None) -> None:
        if inspect.isabstract(t):
            raise TypeError(f"Cannot register abstract class {t.__name__}.")
        typename = t.__name__ if typename is None else typename
        if typename in ConfigurationCodecSettings._registered_types:
            if ConfigurationCodecSettings._registered_types[typename] == t:
                Logger().write(f'[CFG] Re-registering type "{typename}"', LogLevel.INFO)
                return
            raise KeyError(f"Type name '{typename}' already registered.")
        ConfigurationCodecSettings._registered_types[typename] = t
        if t not in ConfigurationCodecSettings._registered_types_inv:
            ConfigurationCodecSettings._registered_types_inv[t] = []
        ConfigurationCodecSettings._registered_types_inv[t] += [typename]

    @staticmethod
    def register_types(*types: type[ConfigurationBase] | tuple[type[ConfigurationBase], str | None]) -> None:
        for t in types:
            Logger().debug(f"[CFG] Registering type {t}.")
            if isinstance(t, tuple):
                cls, typename = t
            else:
                cls = t
                typename = None
            ConfigurationCodecSettings.register_type(cls, typename)

    @staticmethod
    def typename(t: type[ConfigurationBase], index: int = 0) -> str:
        if t in ConfigurationCodecSettings._registered_types_inv:
            lst = ConfigurationCodecSettings._registered_types_inv[t]
            return lst[index % len(lst)]
        Logger().error(f"[CFG] Type {t} not registered.")
        return ""

    @staticmethod
    def typeof(typename: str) -> type[ConfigurationBase]:
        if typename in ConfigurationCodecSettings._registered_types:
            return ConfigurationCodecSettings._registered_types[typename]
        Logger().error(f"[CFG] Typename '{typename}' not registered.")
        return ConfigurationBase

    @staticmethod
    def get_registered_types() -> list[type]:
        return [*ConfigurationCodecSettings._registered_types.values()]


JSON5Encoder: type = json5.JSON5Encoder
if json5.VERSION <= "0.12.0":
    # see https://github.com/dpranke/pyjson5/issues/94
    class BugfixJSON5Encoder(json5.JSON5Encoder):
        def _encode_non_basic_type(self, obj, seen: Set, level: int) -> str:
            if self.check_circular:
                i = id(obj)
                if i in seen:
                    raise ValueError("Circular reference detected.")
                seen.add(i)
            if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
                s = self._encode_dict(obj, seen, level + 1)
            elif hasattr(obj, "__getitem__") and hasattr(obj, "__iter__"):
                s = self._encode_array(obj, seen, level + 1)
            else:
                # bugfix applied here: `level+1` -> `level`
                s = self.encode(self.default(obj), seen, level, as_key=False)
                assert s is not None

            if self.check_circular:
                seen.remove(i)  # type: ignore
            return s

    JSON5Encoder = BugfixJSON5Encoder


class ConfigurationEncoder(JSON5Encoder):
    def default(self, obj: Any) -> Any:
        Logger().debug(f"[CFG] Calling default for {obj}.")
        if isinstance(obj, ConfigurationBase):
            return {
                Constants.Config.TYPE_FIELD_NAME: ConfigurationCodecSettings.typename(type(obj)),
                **{k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
            }
        elif issubclass(type(obj), Enum):
            return obj.value
        elif isinstance(obj, Field):
            return obj.default
        return super().default(obj)


class ConfigurationDecoder:
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.decode(*args, **kwargs)

    def decode(self, obj: Mapping[str, Any]) -> Any:
        Logger().debug(f"[CFG] Decoding object with {len(obj)} entries.")
        if Constants.Config.TYPE_FIELD_NAME in obj:
            typename = obj[Constants.Config.TYPE_FIELD_NAME]
            cls = ConfigurationCodecSettings.typeof(typename)
            o = cls()
            Logger().debug(f'[CFG] Decoding "{typename}" as type {cls}.')
            set_properties = set()
            for k, v in obj.items():
                if k in o.__dict__:
                    t_of_k: type = type(o.__dict__[k])
                    if issubclass(t_of_k, Enum):
                        v = ConfigurationDecoder.decode_enum(v, t_of_k)
                    orig = o.__dict__[k]
                    if isinstance(orig, Field):
                        r, v = orig.check_instance(v, o)
                        if not r:
                            Logger().error(
                                f"[CFG] Type mismatch between expected type {orig.get_generics()} and actual type {type(v)}."
                            )
                    elif isinstance(orig, tuple) and isinstance(v, list):
                        if len(v) != len(orig):
                            Logger().error(
                                f"[CFG] Length mismatch between expected length {len(orig)} and actual length {len(v)}."
                            )
                        for _v, _o in zip(v, orig):
                            if not isinstance(_v, type(_o)):
                                Logger().warning(
                                    f"[CFG] Type mismatch in tuple between expected type {type(_o)} and actual type {type(_v)}."
                                )
                        v = tuple(v)
                    elif not isinstance(v, type(orig)):
                        Logger().info(
                            f"[CFG] Type mismatch between expected type {type(orig)} and actual type {type(v)}.",
                        )
                    o.__dict__[k] = v
                    set_properties.add(k)
                elif not k.startswith("_"):
                    # JSON contains a key that is not in the configuration object.
                    Logger().info(f"[CFG] '{k}' not applicable to {cls}, skipping.")
            # Check for configuration object fields that are not in the JSON.
            for k in o.__dict__.keys():
                if not k.startswith("_") and k not in set_properties:
                    Logger().info(f"[CFG] '{k}' not found in configuration for {cls} (using default).")
            return o
        else:
            Logger().debug(f"[CFG] Type {type(obj)} of object not recognized, returning as-is.")
            return obj

    @staticmethod
    def decode_enum(v: Any, t: type[Enum]) -> Enum:
        # Parse the enum value
        Logger().debug(f'[CFG] Decoding enum value "{v}" of type {t}.')
        try:
            v = t(v)
        except ValueError:
            v = resolve_value_case_insensitive(t, v)
        except Exception as e:
            Logger().error(f'[CFG] Failed to parse enum value "{v}" of type {t} with error {e}.')
            return v  # type: ignore
        return v


def register(
    typename: str | None = None,
) -> Callable[[type[T]], type[T]]:
    def _apply(cls: type[T]) -> type[T]:
        ConfigurationCodecSettings.register_type(cls, typename)
        return cls

    return _apply


F = TypeVar("F")

# Example TypeHierarchy is constructed as follows
# str | int | list[int|None] | dict[str, list[float|int]] | Literal["foo", "bar"]
# [
#    str,
#    int,
#    (list, [[int, None]]),
#    (dict, [str, [float, int]]),
#    Literal["foo", "bar"],
# ]
TypeHierarchy = type | TypeAliasType | list["TypeHierarchy"] | tuple[type | TypeAliasType, list["TypeHierarchy"]]


class Field(Generic[F]):
    def __init__(self, default: F | Callable[[], F], _override_is_callable: Optional[bool] = None):
        self._default_constructor: Optional[Callable[[], F]] = None
        self._default_value: Optional[F] = None

        is_callable = _override_is_callable if _override_is_callable != None else isinstance(default, Callable)
        if is_callable:
            self._default_constructor = cast(Callable[[], F], default)
            self._default_value_set = False
        else:
            self._default_value = cast(F, default)
            self._default_value_set = True

    @property
    def default(self) -> F:
        if not self._default_value_set:
            assert self._default_constructor != None
            self._default_value = self._default_constructor()
            self._default_value_set = True
        return cast(F, self._default_value)

    def check_instance(self, instance: Any, owner: Any) -> tuple[bool, Any]:
        Logger().debug(f"[CFG] Checking instance of type {type(instance)}.")
        field_generic_parameters = self.get_generics()
        assert len(field_generic_parameters) == 1  # why? -> Field is generic in one variable (F)

        # we have to handle the case when the generic parameter F is itself composite
        # e.g. list[int] or dict[str, list[int]] etc.
        # first, collect a hierarchical structure of the generic parameter F
        f_types = FieldHelper._construct_generic_hierarchy(field_generic_parameters[0], owner)
        # now, we can perform in-depth inspection of the instance and its type(s)
        Logger().debug(f"[CFG] Checking instance of type {type(instance)}, Field generics: {field_generic_parameters}.")
        r = self._check_instance_types(instance, f_types)
        if isinstance(r, tuple):
            assert r[0], "Internal implementation error"
            return r
        return r, instance

    def get_generic_types(self, owner: Any) -> TypeHierarchy:
        field_generic_parameters = self.get_generics()
        assert len(field_generic_parameters) == 1  # why? -> Field is generic in one variable (F)
        return FieldHelper._construct_generic_hierarchy(field_generic_parameters[0], owner)

    def _check_instance_types(self, instance: Any, t: TypeHierarchy) -> bool | Tuple[Literal[True], Any]:
        if isinstance(t, type):
            if issubclass(t, Enum):
                # special handling for enum types
                instance = ConfigurationDecoder.decode_enum(instance, t)
                if isinstance(instance, t):
                    return True, instance
                return False
            elif issubclass(t, tuple):
                # special handling for tuples as they are parsed into lists
                if isinstance(instance, list):
                    tuple_instance: tuple[Any] = (*instance,)  # re-construct
                    r = self._check_instance_types(tuple_instance, t)
                    assert not isinstance(r, tuple), "Internal implementation error"
                    if r:
                        return r, tuple_instance
                    return r
            return isinstance(instance, t)
        elif isinstance(t, list):
            for ti in t:
                r = self._check_instance_types(instance, ti)
                if isinstance(r, tuple):
                    assert r[0], "Internal implementation error"
                    return r
                elif r:
                    return True
            return False
        elif isinstance(t, tuple):
            assert len(t) == 2
            class_type, generics = t
            class_match = self._check_instance_types(instance, class_type)
            return_instance = False
            if isinstance(class_match, tuple):
                assert class_match[0], "Internal implementation error"
                instance = class_match[1]
                return_instance = True
            elif not class_match:
                return False

            instance_enumerator, enumerator_settings = FieldHelper._enumerator(instance)
            to_monkey_patch: list[tuple[int, int, Any, Any]] = []
            for iterator_index, item in enumerate(instance_enumerator):
                if InstanceEnumeratorOption.FORCE_DIMENSION in enumerator_settings:
                    if not len(item) == len(generics):
                        Logger().error(f"[CFG] Item consists of {len(item)} values, expected {len(generics)}: {item}.")
                        return False
                if InstanceEnumeratorOption.FORCE_TYPE in enumerator_settings:
                    for item_index, (ii, ti) in enumerate(zip(item, generics)):
                        r = self._check_instance_types(ii, ti)
                        if isinstance(r, tuple):
                            assert r[0], "Internal implementation error"
                            if InstanceEnumeratorOption.ALLOW_MONKEY_PATCHING in enumerator_settings:
                                to_monkey_patch.append((iterator_index, item_index, ii, r[1]))
                            else:
                                Logger().error(f"[CFG] Type {type(instance)} does not allow monkey patching its items.")
                        else:
                            if not r:
                                Logger().error(f"[CFG] Item type {type(ii)} does not match the expected generic type {ti}: {ii}.")
                                return False
            for monkey_patch_args in to_monkey_patch:
                FieldHelper._monkey_patch(instance, *monkey_patch_args)
            if return_instance:
                return True, instance
            return True
        elif isinstance(t, _LiteralGenericAlias):
            if len(t.__parameters__) != 0:
                Logger().error(f"[CFG] Type {t} has parameters which are not supported.")
                return False
            assert hasattr(t, "__args__") and t.__args__ is not None, "Internal implementation error"  # type: ignore
            if any(instance == a for a in t.__args__):  # type: ignore
                return True, instance
            else:
                return False
        elif isinstance(t, TypeAliasType):
            raise NotImplementedError(f"Not yet implemented for TypeAliasType")
        else:
            raise NotImplementedError(f"Unsupported generic parameter type {t}.")

    def get_generics(self) -> Tuple[Any, ...]:
        return get_args(self.__orig_class__)  # type: ignore  (it does work - trust me!)

    def __call__(self) -> F:  # helper function for casting
        return cast(F, self)


class InstanceEnumeratorOption(IntFlag):
    NOTHING = 0x00
    FORCE_DIMENSION = 0x01
    FORCE_TYPE = 0x02
    ALLOW_MONKEY_PATCHING = 0x04

    DEFAULT = FORCE_DIMENSION | FORCE_TYPE
    DEFAULT_WITH_MONKEY_PATCHING = DEFAULT | ALLOW_MONKEY_PATCHING


InstanceIterator = Iterator[Tuple[Any, ...]]


class FieldHelper(ABC):
    @staticmethod
    def _construct_generic_hierarchy(f: Any, owner: Any) -> TypeHierarchy:
        if isinstance(f, GenericAlias):
            return (
                f.__origin__,
                [FieldHelper._construct_generic_hierarchy(fi, owner) for fi in f.__args__],
            )
        elif isinstance(f, UnionType) or (isinstance(f, _UnionGenericAlias) and f._name == "Optional"):
            return [FieldHelper._construct_generic_hierarchy(fi, owner) for fi in f.__args__]
        elif isinstance(f, _LiteralGenericAlias):
            return f
        elif isinstance(f, type):
            return f
        elif isinstance(f, TypeVar):
            # infer from owner -> loop through owner and all base classes of owner to find the corresponding generic type
            to_check = [owner]
            while len(to_check) > 0:
                o = to_check.pop()
                if type(o) == type(object):
                    raise ValueError(f"Unable to resolve type for {f}")
                if hasattr(o, "__origin__"):
                    if hasattr(o.__origin__, "__parameters__") and hasattr(o, "__args__"):
                        for param, arg in zip(o.__origin__.__parameters__, get_args(o)):
                            if param == f:
                                return arg
                    if hasattr(o.__origin__, "__orig_bases__"):
                        to_check.extend(o.__origin__.__orig_bases__)
                if hasattr(o, "__orig_bases__"):
                    to_check.extend(o.__orig_bases__)

            raise NotImplementedError("TypeVar is not supported in this context.")
        else:
            raise NotImplementedError(f"Unsupported type {type(f)}")

    @staticmethod
    def _enumerator(instance: Any) -> tuple[InstanceIterator, InstanceEnumeratorOption]:
        t = type(instance)

        enumerator_method_name = f"_enumerate_{t.__name__}"
        if hasattr(FieldHelper, enumerator_method_name):
            enumerator = getattr(FieldHelper, enumerator_method_name)(instance)
        else:
            raise NotImplementedError(f"Unsupported type {type(instance)}")

        option_method_name = f"_option_{t.__name__}"
        if hasattr(FieldHelper, option_method_name):
            option = getattr(FieldHelper, option_method_name)()
        else:
            option = InstanceEnumeratorOption.DEFAULT
        return enumerator, option

    @staticmethod
    def _monkey_patch(instance: Any, iterator_index: int, item_index: int, value: Any, new_value: Any):
        monkey_patch_method_name = f"_monkey_patch_{type(instance).__name__}"
        if hasattr(FieldHelper, monkey_patch_method_name):
            getattr(FieldHelper, monkey_patch_method_name)(instance, iterator_index, item_index, value, new_value)
        else:
            raise NotImplementedError(f"Cannot find monkey patching method for type {type(instance)}")

    @staticmethod
    def _enumerate_tuple(instance: tuple) -> InstanceIterator:
        yield instance

    @staticmethod
    def _enumerate_list(instance: list) -> InstanceIterator:
        for item in instance:
            yield (item,)

    @staticmethod
    def _option_list() -> InstanceEnumeratorOption:
        return InstanceEnumeratorOption.DEFAULT_WITH_MONKEY_PATCHING

    @staticmethod
    def _monkey_patch_list(instance: list, iterator_index: int, item_index: int, value: Any, new_value: Any):
        assert 0 <= iterator_index < len(instance), f"Iterator index out of bounds: {iterator_index}"
        assert item_index == 0, f"Item index out of bounds: {item_index}"
        if instance[iterator_index] == new_value:
            Logger().debug(f"[CFG] Monkey patching list at index {iterator_index} with {new_value}, but it is already patched.")
        else:
            assert instance[iterator_index] == value, f"Item value does not match: {value} != {instance[iterator_index]}"
            instance[iterator_index] = new_value

    @staticmethod
    def _enumerate_dict(instance: dict) -> InstanceIterator:
        for key, value in instance.items():
            yield (key, value)

    @staticmethod
    def _option_dict() -> InstanceEnumeratorOption:
        return InstanceEnumeratorOption.DEFAULT_WITH_MONKEY_PATCHING

    @staticmethod
    def _monkey_patch_dict(instance: dict, iterator_index: int, item_index: int, value: Any, new_value: Any):
        assert 0 <= iterator_index < len(instance), f"Iterator index out of bounds: {iterator_index}"
        assert item_index == 1, f"Item index out of bounds: {item_index}"
        key = [*instance.keys()][iterator_index]
        assert instance[key] == value, f"Item value does not match: {value} != {instance[key]}"
        instance[key] = new_value
