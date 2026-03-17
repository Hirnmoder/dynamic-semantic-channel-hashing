from abc import ABC, abstractmethod
import datetime
from enum import IntEnum
from io import TextIOBase


from dsh.utils.singleton import Singleton


class LogLevel(IntEnum):
    SPECIAL_NONE = 0
    ERROR = 10
    WARNING = 20
    INFO = 30
    DEBUG = 40
    SPECIAL_ALL = 100


class LoggerTarget(ABC):
    @abstractmethod
    def log(self, msg: str, level: LogLevel) -> None: ...


class ConsoleLoggerTarget(LoggerTarget):
    def __init__(self):
        super().__init__()
        self._colors = {
            LogLevel.ERROR: "97;41",  # white on red
            LogLevel.WARNING: "93;40",  # yellow on black
            LogLevel.INFO: "94;40",  # blue on black
            LogLevel.DEBUG: "90;40",  # grey on black
        }

    def log(self, msg: str, level: LogLevel):
        reset = "0"
        level_str = f"[{level.name}]"
        color = self._colors[level] if level in self._colors else reset
        msg = msg.replace(level_str, f"\033[{color}m{level_str}\033[{reset}m")
        print(msg)


class FileLoggerTarget(LoggerTarget):
    def __init__(self, file: TextIOBase):
        super().__init__()
        self._file = file

    def log(self, msg: str, level: LogLevel):
        self._file.write(msg + "\n")


class LoggerFormatter(ABC):
    @abstractmethod
    def format(self, msg: str, level: LogLevel, time: datetime.datetime) -> str: ...


class DefaultLoggerFormatter(LoggerFormatter):
    def format(self, msg: str, level: LogLevel, time: datetime.datetime):
        return f'{time.strftime("%Y-%m-%d %H:%M:%S.")}{int(time.microsecond/1000):03.0f} [{level.name}] {msg}'


class Logger(metaclass=Singleton):
    def __init__(
        self,
        *targets: LoggerTarget,
        minimum_log_level: LogLevel = LogLevel.WARNING,
        formatter: LoggerFormatter | None = None,
        exit_on_error: bool = False,
    ):
        self.minimum_log_level = minimum_log_level
        self.formatter = formatter
        self._targets: list[LoggerTarget] = []
        self.add_target(*targets)
        self.exit_on_error = exit_on_error

    @property
    def minimum_log_level(self) -> LogLevel:
        return self._minimum_log_level

    @minimum_log_level.setter
    def minimum_log_level(self, mml: LogLevel) -> None:
        self._minimum_log_level = mml

    @property
    def exit_on_error(self) -> bool:
        return self._exit_on_error

    @exit_on_error.setter
    def exit_on_error(self, exit_on_error: bool) -> None:
        self._exit_on_error = exit_on_error

    @property
    def targets(self) -> list[LoggerTarget]:
        return [t for t in self._targets]

    def add_target(self, *targets: LoggerTarget) -> None:
        for target in targets:
            self._targets.append(target)

    def clear_targets(self) -> None:
        self._targets = []

    @property
    def formatter(self) -> LoggerFormatter:
        return self._formatter

    @formatter.setter
    def formatter(self, formatter: LoggerFormatter | None):
        if formatter == None:
            self._formatter = DefaultLoggerFormatter()
        else:
            self._formatter = formatter

    def write(self, msg: str, level: LogLevel = LogLevel.WARNING, time: datetime.datetime | None = None) -> None:
        time = datetime.datetime.now(datetime.UTC)
        if level <= self._minimum_log_level:
            formatted_msg = self._formatter.format(msg, level, time)
            for target in self.targets:
                target.log(formatted_msg, level)
        if level <= LogLevel.ERROR and self.exit_on_error:
            exit(-1)

    def debug(self, msg: str, time: datetime.datetime | None = None) -> None:
        return self.write(msg, LogLevel.DEBUG, time)

    def info(self, msg: str, time: datetime.datetime | None = None) -> None:
        return self.write(msg, LogLevel.INFO, time)

    def warning(self, msg: str, time: datetime.datetime | None = None) -> None:
        return self.write(msg, LogLevel.WARNING, time)

    def error(self, msg: str, time: datetime.datetime | None = None) -> None:
        return self.write(msg, LogLevel.ERROR, time)
