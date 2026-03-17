import enum
from typing import Any, Callable, Generic, Iterator, TypeGuard, TypeVar

from dsh.utils.types import HPARAM
import pandas as pd


T = TypeVar("T")


class ExperimentBagBase(Generic[T]):

    @classmethod
    def member(cls, f: Callable[[T], bool], /) -> enum.member:
        """Helper function for making Tag subclasses entries out of lambdas"""
        return enum.member(f)

    class Tag(enum.Enum):
        NA = enum.nonmember("not applicable")

        def check(self, o: T) -> bool:
            if callable(self.value):
                r = self.value(o)
                assert isinstance(r, bool)
                return r
            raise NotImplementedError()

        def __call__(self, o: T) -> bool:
            return self.check(o)

        def __neg__(self) -> set["ExperimentBagBase[T].Tag"]:
            """Inverts the current Tag by providing the set consisting of its siblings."""
            return set(tag for tag in self.__class__ if tag != self) | set([ExperimentBagBase[T].Tag.NA])  # type: ignore

        def __invert__(self) -> set["ExperimentBagBase[T].Tag"]:
            """Fuzzy select of any given Tag."""
            return set((self, ExperimentBagBase[T].Tag.NA))  # type: ignore

        def __or__(self, other: "ExperimentBagBase[T].Tag") -> set["ExperimentBagBase[T].Tag"]:
            """Combine two Tags into a set."""
            assert other is not None and other.__class__ == self.__class__
            return set([self, other])

        @property
        def repr(self) -> HPARAM:
            return self.name

        @classmethod
        def all_tag_types(cls: type["ExperimentBagBase[T].Tag"]) -> list[type["ExperimentBagBase[T].Tag"]]:
            return cls.__subclasses__()

    def __init__(self, experiments: dict[str, T], precomputed_tags: dict[str, set[Tag]] | None = None):
        self.configs = experiments
        if precomputed_tags is not None and len(precomputed_tags) == len(experiments):
            self.tags = precomputed_tags
        else:
            self.tags: dict[str, set[ExperimentBagBase[T].Tag]] = {}
            for n, c in experiments.items():
                self.tags[n] = set()
                for tagtype in ExperimentBagBase[T].Tag.all_tag_types():
                    for tag in tagtype:
                        if tag(c):
                            self.tags[n].add(tag)
        assert len(self.tags) == len(self.configs)

    def __call__(self, *tags: Tag | set[Tag]) -> "ExperimentBagBase[T]":
        resulting_configs = dict[str, T]()
        tag_sets_to_search = [
            set([tags_entry]) if isinstance(tags_entry, ExperimentBagBase[T].Tag) else tags_entry for tags_entry in tags
        ]
        for name, t in self.tags.items():
            # in order for a experiment to be selected, all tags must match
            # example: tag_sets_to_search = [ [DS.One], [Loss.A, Loss.C], [Param.X, Tag.NA] ]
            #   so each experiment must match  ^exactly,  ^one of them^,   ^Param.X if any Param is given
            all_tags_match = True
            for tag_set_full in tag_sets_to_search:
                tag_set = set(tag for tag in tag_set_full if tag != ExperimentBagBase[T].Tag.NA)
                # now first assert that all Tags are of same type
                tag_set_types = set(tag.__class__ for tag in tag_set)
                if len(tag_set_types) != 1:
                    raise ValueError(f"Set of Tags contains multiple types: {tag_set_types}")
                tag_set_type = tag_set_types.pop()
                if ExperimentBagBase[T].Tag.NA in tag_set_full:
                    # case 3
                    if len(t.intersection(tag_set_type)) == 0:
                        continue
                # cases 1 and 2 (basically the same) or case 3 when there is a type match
                match len(t.intersection(tag_set)):
                    case 0:
                        all_tags_match = False
                        break
                    case 1:
                        continue
                    case _:
                        raise NotImplementedError("It should not be possible...")

            if all_tags_match:
                resulting_configs[name] = self.configs[name]
        return ExperimentBagBase[T](
            resulting_configs,
            {name: self.tags[name] for name in resulting_configs},
        )

    def __repr__(self) -> str:
        return f"Bag holding {len(self)} experiments"

    def __len__(self) -> int:
        assert len(self.tags) == len(self.configs)
        return len(self.configs)

    def __iter__(self) -> Iterator[str]:
        for n in self.configs.keys():
            yield n

    def __or__(self, other: "ExperimentBagBase[T]") -> "ExperimentBagBase[T]":
        assert isinstance(other, ExperimentBagBase), "Can only merge with other experiment bags"
        resulting_configs: dict[str, T] = {**self.configs, **other.configs}
        resulting_tags: dict[str, set[ExperimentBagBase[T].Tag]] = {**self.tags, **other.tags}
        return ExperimentBagBase[T](resulting_configs, resulting_tags)

    def __and__(self, other: "ExperimentBagBase[T]") -> "ExperimentBagBase[T]":
        assert isinstance(other, ExperimentBagBase), "Can only intersect with other experiment bags"
        resulting_names: set[str] = set(self.configs.keys()) & set(other.configs.keys())
        resulting_configs: dict[str, T] = {n: self.configs[n] for n in resulting_names}
        resulting_tags: dict[str, set[ExperimentBagBase[T].Tag]] = {n: self.tags[n] for n in resulting_names}
        return ExperimentBagBase[T](resulting_configs, resulting_tags)

    def __sub__(self, other: "ExperimentBagBase[T]") -> "ExperimentBagBase[T]":
        assert isinstance(other, ExperimentBagBase), "Can only subtract other experiment bags"
        resulting_names: set[str] = set(self.configs.keys()) - set(other.configs.keys())
        resulting_configs: dict[str, T] = {n: self.configs[n] for n in resulting_names}
        resulting_tags: dict[str, set[ExperimentBagBase[T].Tag]] = {n: self.tags[n] for n in resulting_names}
        return ExperimentBagBase[T](resulting_configs, resulting_tags)

    def items(self) -> Iterator[tuple[str, T]]:
        for n, c in self.configs.items():
            yield n, c

    @property
    def distinct_sets(self) -> list[set[Tag]]:
        result = list[set[ExperimentBagBase[T].Tag]]()
        for tags in self.tags.values():
            if tags not in result:
                result.append(tags)
        return sorted(result)

    @property
    def df(self) -> pd.DataFrame:
        df_data: dict[str, list[HPARAM]] = {n: [] for n in self.configs}
        df_cols = list[str]()
        for tagtype in ExperimentBagBase[T].Tag.all_tag_types():
            df_cols.append(tagtype.__doc__ or str(tagtype))
            for n in df_data:
                exp_tag = self.tags[n].intersection(tagtype)
                if len(exp_tag) == 1:
                    df_data[n].append(exp_tag.pop().repr)
                elif len(exp_tag) == 0:
                    df_data[n].append("")
                else:
                    raise ValueError(f"Experiment {n} contains {len(exp_tag)} tags of type {tagtype}: {exp_tag}")
        return pd.DataFrame(df_data, index=df_cols).T

    @property
    def grouped_df(self) -> pd.DataFrame:
        df = self.df
        df = df.reset_index(names=["count"]).groupby(df.columns.to_list()).count()
        return df.reset_index()


def t(a: Any, b: type[T]) -> TypeGuard[T]:
    """Short-named helper function to determine type"""
    return isinstance(a, b)
