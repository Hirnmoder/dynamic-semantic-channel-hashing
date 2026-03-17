from abc import ABC, abstractmethod
import argparse
from enum import Enum
import os
from types import FunctionType, NoneType
from typing import (
    Any,
    Callable,
    Generic,
    Sequence,
    TypeVar,
    _LiteralGenericAlias,  # type: ignore # it does exist
)


from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import has_focus
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.layout import (
    VSplit,
    Window,
    Layout,
    HSplit,
    AnyContainer,
    FloatContainer,
    Dimension,
)
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import RadioList, Frame, TextArea, Button, Dialog

from dsh.config import run
from dsh.config import configuration as c
from dsh.utils.collections import first_index_of
from dsh.utils.parsing import parse_default_float, parse_default_int
from dsh.utils.selector import ConfigMode

__all__ = ["interactive_config_setup"]
G = TypeVar("G")
I = TypeVar("I", str, int, float)


def gather_full_info() -> tuple[
    dict[type[c.ConfigurationBase], dict[str, c.TypeHierarchy]],
    dict[type[c.ConfigurationBase], set[type[c.ConfigurationBase]]],
    dict[type[c.ConfigurationBase], str],
]:
    type_infos = dict[type[c.ConfigurationBase], dict[str, c.TypeHierarchy]]()
    # we don't want to override attributes of type Field with their default values
    with c.ConfigurationBaseMeta.defer_post_init(cancel=True):
        visited = set[type]()
        to_visit = set[type[c.ConfigurationBase]](c.ConfigurationCodecSettings.get_registered_types())
        while len(to_visit) > 0:
            current = to_visit.pop()
            visited.add(current)
            type_infos[current] = _gather_type_infos_rec(current)

    inheritance = dict[type[c.ConfigurationBase], set[type[c.ConfigurationBase]]]()
    names = dict[type[c.ConfigurationBase], str]()
    for t in visited:
        _fill_inheritance_dict(t, inheritance)
        names[t] = c.ConfigurationCodecSettings.typename(t)

    return type_infos, inheritance, names


def _fill_inheritance_dict(t: type, d: dict[type[c.ConfigurationBase], set[type[c.ConfigurationBase]]]) -> None:
    for b in t.__bases__:
        if issubclass(b, c.ConfigurationBase):
            if b not in d:
                d[b] = set[type[c.ConfigurationBase]]()
            d[b].add(t)
            _fill_inheritance_dict(b, d)


def _gather_type_infos_rec(t: type) -> dict[str, c.TypeHierarchy]:
    result_dict = dict[str, c.TypeHierarchy]()
    if issubclass(t, c.ConfigurationBase):
        # iterate over all entries
        instance = t()
        for k, v in instance.__dict__.items():
            if k.startswith("_"):
                continue  # only consider public members
            fl = _collect_type_infos(v, f"{t.__name__}.{k}", instance)
            result_dict[k] = fl
    else:
        raise NotImplementedError(f"Type {t} not supported for configuration setup.")
    return result_dict


def _collect_type_infos(v: Any, name: str, owner: Any) -> c.TypeHierarchy:
    if isinstance(v, (str, int, float, bool, Enum)):
        return type(v)
    elif isinstance(v, c.Field):
        return v.get_generic_types(owner)
    elif v == None:
        raise ValueError(f"Attribute {name} is None.")
    else:
        raise NotImplementedError(f"Type {type(v)} of attribute {name} is unsupported for configuration setup.")


class _Wrapper:
    def __init__(
        self,
        obj: run.RunConfigBase,
        config_mode: ConfigMode,
        type_infos: dict[type[c.ConfigurationBase], dict[str, c.TypeHierarchy]],
        inheritance: dict[type[c.ConfigurationBase], set[type[c.ConfigurationBase]]],
        names: dict[type[c.ConfigurationBase], str],
    ) -> None:
        self.obj = obj
        self.obj_t = type(obj)
        self.config_mode = config_mode
        self.type_infos = type_infos
        self.inheritance = inheritance
        self.names = names

    def set(self, path: list[str], prop: str, e: Any):
        o = self.obj
        for part in path:
            o = getattr(o, part)
        setattr(o, prop, e)

    def get(self, path: list[str]) -> Any:
        o = self.obj
        for part in path:
            if hasattr(o, part):
                o = getattr(o, part)
        return o

    def type_infos_of(self, path: list[str]) -> dict[str, c.TypeHierarchy]:
        t = self.type_of_element(path)
        if issubclass(t, c.ConfigurationBase):
            return self.type_infos[t]
        return {}

    def type_infos_of_element(self, path: list[str], prop: str) -> c.TypeHierarchy:
        t = self.type_infos_of(path)
        return t[prop]

    def type_of_element(self, path: list[str]) -> type:
        o = self.get(path)
        return type(o)


def init() -> _Wrapper:
    type_infos, inheritance, names = gather_full_info()
    cti = choice(
        "Select Configuration Type",
        options=[
            ((t, m), names[t])
            for t, m in [
                (run.CrossModalHashingPipelineConfig, ConfigMode.ALL),
                (run.CrossModalTrainRunConfig, ConfigMode.TRAIN),
                (run.CrossModalInferenceRunConfig, ConfigMode.INFERENCE),
                (run.MetricRunConfig, ConfigMode.METRICS),
            ]
        ],
        mouse_support=True,
        show_frame=True,
    )
    w = _Wrapper(
        cti[0](),
        cti[1],
        type_infos,
        inheritance,
        names,
    )
    return w


def prompt_options(options: Sequence[tuple[G, str]], default: G, selected: G | None = None, title: str | None = None) -> G:
    buttons = list[Button]()
    to_select: Button | None = None
    for o, name in options:
        b = Button(name, width=60)
        b.handler = lambda o=o: get_app().exit(result=o)
        buttons.append(b)
        if selected == o:
            to_select = b

    kb = KeyBindings()
    kb.add(Keys.Left, filter=~has_focus(buttons[0]))(focus_previous)
    kb.add(Keys.Up, filter=~has_focus(buttons[0]))(focus_previous)
    kb.add(Keys.Right, filter=~has_focus(buttons[-1]))(focus_next)
    kb.add(Keys.Down, filter=~has_focus(buttons[-1]))(focus_next)

    d = Dialog(
        VSplit(
            [
                Window(width=Dimension(min=1, max=None)),
                HSplit(buttons, key_bindings=kb),
                Window(width=Dimension(min=1, max=None)),
            ]
        ),
        title=title or "Select value",
        modal=True,
    )
    a = to_application(d)
    if to_select != None:
        a.layout.focus(to_select)
    result = a.run(in_thread=True)
    get_app().reset()
    return default if result == None else result


def prompt_input(type: type, default: I) -> I:
    def accept(buf: Buffer) -> bool:
        get_app().layout.focus(ok_button)
        return True  # Keep text

    def ok_handler() -> None:
        get_app().exit(result=textfield.text)

    textfield = TextArea(
        text=str(default) if default != None else "",
        multiline=False,
        password=False,
        accept_handler=accept,
        width=60,
    )
    ok_button = Button(text="Confirm", handler=ok_handler)
    cancel_button = Button(text="Cancel", handler=lambda: get_app().exit(result=None))

    d = Dialog(
        VSplit(
            [
                Window(width=Dimension(min=1, max=None)),
                HSplit([textfield, VSplit([ok_button, cancel_button])]),
                Window(width=Dimension(min=1, max=None)),
            ]
        ),
        title="Enter value",
        modal=True,
    )
    a = to_application(d)
    result: str | None = a.run(in_thread=True)
    if result == None:
        out = default
    else:
        if type == int:
            out = parse_default_int(result, default)
        elif type == float:
            out = parse_default_float(result, default)
        elif type == str:
            out = str(result)
        else:
            out = default
    get_app().reset()
    return out  # type: ignore


def to_application(d: Dialog) -> Application:
    kb = KeyBindings()
    kb.add(Keys.Escape)(lambda e: get_app().exit(result=None))
    a = Application(
        layout=Layout(d),
        mouse_support=True,
        full_screen=True,
        erase_when_done=True,
        style=Style.from_dict(
            {
                "text-area": "bg:black fg:white",
            }
        ),
        key_bindings=kb,
    )
    return a


def interactive_config_setup(args: argparse.Namespace):
    w = init()

    kb = KeyBindings()

    @kb.add(Keys.ControlC)
    def _(e: KeyPressEvent):
        e.app.exit()

    @kb.add(Keys.ControlS)
    def _(e: KeyPressEvent):
        config_filename = args.config.replace("{mode}", w.config_mode.value)
        output_filename = prompt_input(str, config_filename)
        if os.path.exists(output_filename):
            overwrite = prompt_options([(True, "Yes"), (False, "No")], False, False, title="File exists. Overwrite?")
            if not overwrite:
                return
        w.obj.dumpf(output_filename)

    @kb.add(Keys.Tab)
    def _(e: KeyPressEvent):
        if e.app.layout.has_focus(panes_container):
            e.app.layout.focus(result_container)
        else:
            e.app.layout.focus(panes_container)

    def wrap_list(l: PropertyList | TypeList) -> AnyContainer:
        f = Frame(l, title=l.title, width=Dimension(weight=1))
        return f

    plist = PropertyList(w)
    tlist = TypeList(w, plist)

    panes_container = VSplit([wrap_list(plist), wrap_list(tlist)], width=Dimension(min=30, weight=2))
    result_text = TextArea(
        "",
        multiline=True,
        focusable=True,
        focus_on_click=True,
        wrap_lines=True,
        read_only=True,
        line_numbers=True,
        scrollbar=True,
    )
    result_container = Frame(result_text, title="Result", width=Dimension(min=30, weight=1))
    root_content = VSplit([panes_container, result_container])
    root_container = FloatContainer(root_content, [])
    layout = Layout(root_container)

    def update_output(*args, **kwargs):
        cursor_pos = result_text.buffer.cursor_position
        result_text.text = w.obj.dumps()
        result_text.buffer.cursor_position = cursor_pos  # it's not perfect, but better than resetting to 0

    plist.on_path_change(update_output)
    tlist.on_set_value(update_output)
    update_output()

    a = Application(
        full_screen=True,
        layout=layout,
        mouse_support=True,
        erase_when_done=True,
        key_bindings=kb,
        style=Style.from_dict(
            {
                "radio-checked": "bold nodim",
                "radio-selected": "nodim",
                "radio-list": "dim",
                "property": "ansiblue",
                "type": "ansigreen",
                "value": "ansired",
                "noentry": "dim",
            }
        ),
    )

    @plist.kb.add(Keys.Right)
    def _(e: KeyPressEvent) -> None:
        a.layout.focus(tlist)

    @tlist.kb.add(Keys.Left)
    def _(e: KeyPressEvent) -> None:
        a.layout.focus(plist)

    a.run()


class HookableRadioList(RadioList[G], Generic[G]):
    def __init__(
        self,
        values: Sequence[tuple[G, AnyFormattedText]],
        default: G | None = None,
        show_numbers: bool = True,
        select_on_focus: bool = False,
        open_character: str = "",
        select_character: str = ">",
        close_character: str = "",
        container_style: str = "class:radio-list",
        default_style: str = "class:radio",
        selected_style: str = "class:radio-selected",
        checked_style: str = "class:radio-checked",
        number_style: str = "class:radio-number",
        multiple_selection: bool = False,
        show_cursor: bool = True,
        show_scrollbar: bool = True,
    ) -> None:
        super().__init__(
            values,
            default,
            show_numbers,
            select_on_focus,
            open_character,
            select_character,
            close_character,
            container_style,
            default_style,
            selected_style,
            checked_style,
            number_style,
            multiple_selection,
            show_cursor,
            show_scrollbar,
        )
        self._enter_hooks = list[Callable[["HookableRadioList", G], None]]()

    def on_select(self, hook: Callable[["HookableRadioList", G], None]) -> None:
        self._enter_hooks.append(hook)

    def _handle_enter(self) -> None:
        r = super()._handle_enter()
        for hook in self._enter_hooks:
            hook(self, self.current_value)
        return r


class _WrapperListBase(HookableRadioList[G], ABC, Generic[G]):
    def __init__(
        self,
        wrapper: _Wrapper,
        property_style: str = "class:property",
        type_style: str = "class:type",
        value_style: str = "class:value",
        token_style: str = "",
        noentry_style: str = "class:noentry",
        show_numbers: bool = False,
    ) -> None:
        self.wrapper = wrapper
        self._selected_indices = list[int]()

        self.property_style = property_style
        self.type_style = type_style
        self.value_style = value_style
        self.token_style = token_style
        self.noentry_style = noentry_style

        super().__init__(
            self._values(),
            None,
            show_numbers=show_numbers,
        )

        assert isinstance(self.control.key_bindings, KeyBindings)
        self.kb = self.control.key_bindings

    @abstractmethod
    def _values(self) -> Sequence[tuple[G, AnyFormattedText]]: ...

    def _stringify_type_hierarchy(self, h: c.TypeHierarchy) -> str:
        if isinstance(h, type):
            return self.wrapper.names[h] if h in self.wrapper.names else h.__name__
        elif isinstance(h, list):
            return " | ".join([self._stringify_type_hierarchy(hi) for hi in h])
        elif isinstance(h, tuple):
            return f"{self._stringify_type_hierarchy(h[0])}[{', '.join(self._stringify_type_hierarchy(hi) for hi in h[1])}]"
        elif isinstance(h, _LiteralGenericAlias):
            return f"Literal[{', '.join(h.__args__)}]"  # type: ignore
        else:
            return str(h)

    def _stringify_type_of(self, path: list[str], prop: str) -> str:
        value = self.wrapper.get(path + [prop])
        t = type(value)
        return self.wrapper.names[t] if t in self.wrapper.names else t.__name__


class PropertyList(_WrapperListBase[str]):
    def __init__(
        self,
        wrapper: _Wrapper,
        root_string: str = "<root>",
    ) -> None:
        self.path = list[str]()
        self.root_string = root_string
        self._path_change_hooks = list[Callable[["PropertyList"], None]]()

        super().__init__(wrapper)

        @self.kb.add(Keys.Escape)
        def _(e: KeyPressEvent) -> None:
            self.move_upwards()

    def move_upwards(self) -> None:
        idx = 0
        if len(self.path) > 0:
            self.path.pop()
            idx = self._selected_indices.pop()
        self._reset(idx)

    def select_property(self, prop: str) -> None:
        if prop == "..":
            self.move_upwards()
        else:
            prop_type = self.wrapper.type_of_element(self.path + [prop])
            if issubclass(prop_type, c.ConfigurationBase):
                self.path.append(prop)
                self._selected_indices.append(self._selected_index)
                self._reset()
            else:
                # TODO: message to user
                pass

    def _handle_enter(self) -> None:
        previous_value = self.current_value
        _ = super()._handle_enter()
        new_value = self.current_value
        if previous_value == new_value or new_value == "..":  # user accepted twice or wants to move up
            self.select_property(new_value)
        return _

    def on_path_change(self, hook: Callable[["PropertyList"], None]) -> None:
        self._path_change_hooks.append(hook)

    def refresh(self, selected_index: int | None = None) -> None:
        self.values = self._values()
        if selected_index != None:
            self._selected_index = selected_index
        self.current_value = self.values[self._selected_index][0]

    def _reset(self, selected_index: int = 0) -> None:
        self.refresh(selected_index)
        for hook in self._path_change_hooks:
            hook(self)

    def _values(self) -> Sequence[tuple[str, AnyFormattedText]]:
        properties = self.wrapper.type_infos_of(self.path)
        elements: Sequence[tuple[str, AnyFormattedText]] = [
            (
                k,
                [
                    (self.property_style, k),
                    (self.token_style, ": "),
                    (self.type_style, self._stringify_type_hierarchy(v)),
                    (self.token_style, " = "),
                    (self.type_style, self._stringify_type_of(self.path, k)),
                ],
            )
            for k, v in properties.items()
        ]
        elements.insert(0, ("..", [(self.property_style, "..")]))
        return elements

    def title(self) -> AnyFormattedText:
        return [
            (self.token_style, self.root_string),
            (self.property_style, "" if len(self.path) == 0 else "." + ".".join(self.path)),
            (self.token_style, ": "),
            (self.type_style, self._stringify_type_hierarchy(self.wrapper.type_of_element(self.path))),
        ]


class TypeList(_WrapperListBase[type]):
    def __init__(
        self,
        wrapper: _Wrapper,
        property_selector: PropertyList,
    ) -> None:
        self.property_selector = property_selector
        self.prop: str | None = None
        self._set_value_hooks = list[Callable[["TypeList"], None]]()
        super().__init__(
            wrapper,
            show_numbers=False,
        )
        self.property_selector.on_select(self.property_selected)
        self.property_selector.on_path_change(self.path_changed)

    def property_selected(self, caller: HookableRadioList, value: str) -> None:
        if value == "..":
            self.prop = None
            self._reset()
        elif self.prop != value:
            self.prop = value
            self._reset()

    def path_changed(self, caller: PropertyList) -> None:
        self.prop = None
        if len(caller.current_value.strip()) > 0 and caller.current_value.strip() != "..":
            self.prop = caller.current_value
        self._reset()

    def _handle_enter(self) -> None:
        _ = super()._handle_enter()
        if self.current_value and self.prop:
            self._set_value()
        return _

    def on_set_value(self, hook: Callable[["TypeList"], None]) -> None:
        self._set_value_hooks.append(hook)

    def _set_value(self) -> None:
        assert self.current_value
        assert self.prop
        t = self.current_value
        obj_value = self.wrapper.get(self.property_selector.path + [self.prop])
        if issubclass(t, c.ConfigurationBase):
            # find all static methods with names starting "preset_"
            presets: list[tuple[Callable[[], c.ConfigurationBase], str]] = [(t, "Default")]
            PRESET_PREFIX = "preset_"
            for n in t.__dict__:
                if n.startswith(PRESET_PREFIX):
                    method = getattr(t, n)
                    assert isinstance(method, FunctionType)
                    if method.__doc__ and len(method.__doc__.strip()) > 0:
                        name = method.__doc__
                    else:
                        name = n.removeprefix(PRESET_PREFIX).replace("_", " ").capitalize()
                    presets.append((method, name))
            preset = prompt_options(presets, None) if len(presets) > 1 else presets[0][0]
            if preset == None:
                return
            obj = preset()
        else:
            if t == NoneType:
                obj = None
            elif t in [int, float, str]:
                obj = prompt_input(self.current_value, obj_value)
            elif t == bool:
                obj = prompt_options([(True, str(True)), (False, str(False))], None, obj_value)
            elif issubclass(t, Enum):
                obj = prompt_options([(v, str(v.value)) for v in t], None, obj_value)
            elif t == list:
                # Not implemented
                obj = []
            elif t == dict:
                # Not implemented
                obj = {}
            else:
                raise ValueError(f"Unknown type {self.current_value}")
        if obj == None and t != NoneType:
            return

        self.wrapper.set(self.property_selector.path, self.prop, obj)
        self.property_selector.refresh()
        for hook in self._set_value_hooks:
            hook(self)

    def _reset(self) -> None:
        self.values = self._values()
        self.current_values = []
        self._selected_index = 0
        if self.prop != None:
            t = self.wrapper.type_of_element(self.property_selector.path + [self.prop])
            self._selected_index = first_index_of(self.values, lambda v: v[0] == t, 0)
        self.current_value = self.values[self._selected_index][0]

    def _values(self) -> Sequence[tuple[type, AnyFormattedText]]:
        if self.prop == None:
            return [(type(None), [(self.noentry_style, "No entry")])]
        else:
            types = self.get_types()

            elements: Sequence[tuple[type, AnyFormattedText]] = [
                (
                    t,
                    [(self.type_style, name)],
                )
                for (name, t) in types
            ]
            return elements

    def get_types(self) -> list[tuple[str, type]]:
        assert self.prop
        ti = self.wrapper.type_infos_of_element(self.property_selector.path, self.prop)
        return self._get_types_rec(ti)

    def _get_types_rec(self, h: c.TypeHierarchy) -> list[tuple[str, type]]:
        if isinstance(h, type):
            if h in self.wrapper.inheritance:
                return [(self.wrapper.names[hi], hi) for hi in self.wrapper.inheritance[h]]
            return [(self.wrapper.names[h] if h in self.wrapper.names else h.__name__, h)]
        elif isinstance(h, list):
            return [hii for hi in h for hii in self._get_types_rec(hi)]
        else:
            raise NotImplementedError()
        # elif isinstance(h, tuple):
        #    pass
        # elif isinstance(h, _LiteralGenericAlias):
        #    pass
        # else:
        #    pass

    def title(self) -> AnyFormattedText:
        if self.prop == None:
            return [(self.noentry_style, "Nothing selected")]
        else:
            return [
                (
                    self.type_style,
                    self._stringify_type_hierarchy(self.wrapper.type_infos_of_element(self.property_selector.path, self.prop)),
                ),
            ]
