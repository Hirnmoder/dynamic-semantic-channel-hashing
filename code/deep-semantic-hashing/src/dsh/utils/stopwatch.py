import os
import time
import torch
from typing import Literal, Optional, Self

from dsh.utils.logger import Logger, LogLevel
from dsh.utils.singleton import Singleton

FileMode = Literal["overwrite", "append", "create"]


class StopWatch:
    def __init__(
        self,
        device_to_sync: Optional[torch.device] = None,
        initial_entry_description: Optional[str] = None,
        default_print: None | Literal[False] | LogLevel = False,
    ):
        self.device_to_sync = device_to_sync
        self._start = time.time()
        self.history: list[tuple[str, float]] = [(initial_entry_description or "Start", self.time)]
        self._default_print: None | Literal[False] | LogLevel = default_print

    def record(self, description: str, print: None | Literal[False] | LogLevel = None) -> Self:
        if torch.cuda.is_available() and self.device_to_sync is not None and self.device_to_sync.type == "cuda":
            torch.cuda.synchronize(self.device_to_sync)
        t = self.time
        self.history.append((description, t))
        if print == None and self._default_print is not None:
            print = self._default_print
        if print != None and print != False:
            Logger().write(f"[TIM] {t:09.3f} | {self.last_duration:+07.3f}s {description}", level=print)
        return self

    @property
    def time(self) -> float:
        return time.time() - self._start

    @property
    def total_duration(self) -> float:
        if len(self.history) == 1:
            return 0
        return self.history[-1][1] - self.history[0][1]

    @property
    def last_duration(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return self.history[-1][1] - self.history[-2][1]

    def duration_since(self, description: str) -> float:
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == description:
                return self.history[-1][1] - self.history[i][1]
        return 0.0

    def save(self, filename: str, mode: FileMode = "create"):
        match mode:
            case "overwrite":
                filemode = "w"
                write_header = True
            case "append":
                filemode = "a"
                write_header = False
            case "create":
                if os.path.exists(filename):
                    raise FileExistsError(f"File {filename} already exists.")
                filemode = "w"
                write_header = True
            case _:
                raise ValueError("Invalid mode specified for saving the stopwatch history.")

        with open(filename, filemode) as f:
            if write_header:
                f.write("     id,timestamp,description\n")
            for i, entry in enumerate(self.history):
                f.write(f"{i:7d},{entry[1]:9.3f},{entry[0]}\n")


class Profiler:
    def __init__(
        self,
        stopwatch: StopWatch | None = None,
        hierarchy_sep: str = ".",
        enter_pattern: str = "Enter",
        exit_pattern: str = "Exit",
    ):
        self._stopwatch = stopwatch or StopWatch()
        self._hierarchy_stack = list[str]()
        self._hierarchy_sep = hierarchy_sep
        self._enter_pattern = enter_pattern
        self._exit_pattern = exit_pattern

    def step(self, description: str, **kwargs: str) -> "ProfilerStep":
        return ProfilerStep(self, description, **kwargs)

    def record(self, description: str, print: None | Literal[False] | LogLevel = None):
        return self.stopwatch.record(self.get_record_name(description), print)

    def get_record_name(self, description: str) -> str:
        return self._hierarchy_sep.join(self._hierarchy_stack) + self._hierarchy_sep + description

    def save(self, filename: str, mode: FileMode = "create"):
        self.stopwatch.save(filename, mode)

    def _enter(self, description: str) -> Self:
        self._hierarchy_stack.append(description)
        return self

    def _exit(self) -> str:
        return self._hierarchy_stack.pop()

    @property
    def stopwatch(self) -> StopWatch:
        return self._stopwatch


class ProfilerStep:
    def __init__(self, profiler: Profiler, description: str, **kwargs: str):
        self.profiler = profiler
        self.description = description
        self.kwargs = kwargs
        self.duration: float = 0

    def record(self, description: str, print: None | Literal[False] | LogLevel = None):
        return self.profiler.record(description, print)

    def __enter__(self) -> Self:
        self.__enter_str = ProfilerStep._replace_params(self.profiler._enter_pattern, description=self.description, **self.kwargs)
        self.profiler._enter(self.description).record(self.__enter_str)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Literal[False]:
        self.__exit_str = ProfilerStep._replace_params(self.profiler._exit_pattern, description=self.description, **self.kwargs)
        self.duration = self.profiler.record(self.__exit_str).duration_since(self.profiler.get_record_name(self.__enter_str))
        _d = self.profiler._exit()
        assert _d == self.description, f"Exiting Profile Step. Expected {self.description} but got {_d}."
        return False  # Don't suppress exceptions

    @staticmethod
    def _replace_params(string: str, **kwargs: str):
        for key, value in kwargs.items():
            string = string.replace("${" + key.upper() + "}", str(value))
        return string


class GlobalProfiler(Profiler, metaclass=Singleton):
    def __init__(self, stopwatch: StopWatch | None = None):
        super().__init__(stopwatch)
