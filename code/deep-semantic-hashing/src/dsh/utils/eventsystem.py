from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
import multiprocessing.connection as mpc
import threading as t
from typing import Any, Callable, Generic, Self, TypeVarTuple

from dsh.utils.logger import Logger
from dsh.utils.singleton import Singleton


class EventHandlerResultType(Enum):
    SUCCESS = "success"
    CANCEL = "cancel"
    ERROR_HANDLED = "error-handled"
    ERROR_UNHANDLED = "error-unhandled"
    IGNORED = "ignored"


class EventHandlerResult:
    def __init__(self, type: EventHandlerResultType, value: Any = None):
        self.type = type
        self.value = value


class EventHandler(ABC):
    @abstractmethod
    def __call__(self, sender: Any, *args: Any, **kwargs: Any) -> EventHandlerResult:
        raise NotImplementedError("Event handler must implement the '__call__' method.")


Ins = TypeVarTuple("Ins")


class ActionEventHandler(EventHandler, Generic[*Ins]):
    def __init__(self, action: Callable[[Any, *Ins], None | EventHandlerResultType | EventHandlerResult]):
        self.action = action

    def __call__(self, sender: Any, *args: *Ins, **kwargs: Any) -> EventHandlerResult:
        try:
            result = self.action(sender, *args)
            if isinstance(result, EventHandlerResult):
                return result
            elif isinstance(result, EventHandlerResultType):
                return EventHandlerResult(result)
            else:
                return EventHandlerResult(EventHandlerResultType.SUCCESS, value=result)
        except Exception as e:
            raise e


@dataclass
class Event:
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    origin: str | None = None


class EventProxy:
    def __init__(self, system: "EventSystem", event_name: str, priority: int = 0, propagate: bool = True):
        self._system = system
        self._event_name = event_name
        self._priority = priority
        self._propagate = propagate

    def __iadd__(self, other: EventHandler | Callable[[Any, *Ins], Any]) -> Self:
        assert isinstance(
            other, (Callable, EventHandler)
        ), f"Only EventHandlers or Callables can be added to an event; got {type(other)}"
        if isinstance(other, Callable):
            other = ActionEventHandler(other)
        if not self._system.register_event_handler(self._event_name, other, self._priority, True):
            raise ValueError("Failed to add event handler.")
        return self

    def __isub__(self, other: Any) -> Self:
        assert isinstance(other, EventHandler), f"Only EventHandlers can be removed from an event; got {type(other)}"
        if not self._system.unregister_event_handler(self._event_name, other):
            raise ValueError("Failed to remove event handler.")
        return self

    def __call__(self, sender: Any, *args: Any, **kwargs: Any) -> tuple[EventHandlerResultType, list[EventHandlerResult]]:
        return self._system.fire(self._event_name, sender, *args, propagate=self._propagate, **kwargs)


class EventSystem(metaclass=Singleton):
    def __init__(self):
        self._event_handlers: dict[str, list[tuple[int, EventHandler]]] = {}

    def register_event(self, name: str) -> bool:
        if name in self._event_handlers:
            return False
        self._event_handlers[name] = []
        return True

    def is_registered_event(self, name: str) -> bool:
        return name in self._event_handlers

    def register_event_handler(self, name: str, handler: EventHandler, priority: int = 0, register_event: bool = True) -> bool:
        if name not in self._event_handlers:
            if register_event:
                if not self.register_event(name):
                    return False
            else:
                return False
        assert self.is_registered_event(name)
        assert self._event_handlers[name] is not None
        self._event_handlers[name] = sorted(self._event_handlers[name] + [(priority, handler)], key=lambda x: x[0])
        return True

    def unregister_event_handler(self, name: str, handler: EventHandler) -> bool:
        if not self.is_registered_event(name):
            return False
        assert self._event_handlers[name] is not None
        self._event_handlers[name] = [h for h in self._event_handlers[name] if h[1] != handler]
        return True

    def __getitem__(self, args: str | tuple[str, int] | tuple[str, bool]) -> EventProxy:
        priority = 0
        propagate = True
        if isinstance(args, str):
            name = args
        else:
            name, priority_or_propagate = args
            if isinstance(priority_or_propagate, bool):
                propagate = priority_or_propagate
            elif isinstance(priority_or_propagate, int):
                priority = priority_or_propagate
        self.register_event(name)
        return EventProxy(self, name, priority, propagate)

    def __setitem__(self, args: str | tuple[str, int] | tuple[str, bool], value: Any) -> None:
        if isinstance(args, str):
            name, _ = args, 0
        else:
            name, _ = args
        if isinstance(value, EventProxy):
            if value._event_name == name:
                # do nothing here, because it is called by this construct:
                # EventSystem[name] += handler
                return
            else:
                # otherwise raise an error, e.g. when someone does this:
                # EventSystem[name] = EventSystem[othername]
                raise ValueError(f"EventProxy name mismatch: {value._event_name} != {name}")
        raise TypeError(f"Unsupported type for assignment to EventSystem[{name}]: {type(value)}")

    def fire(
        self,
        name: str,
        sender: Any,
        *args: Any,
        propagate: bool = True,
        **kwargs: Any,
    ) -> tuple[EventHandlerResultType, list[EventHandlerResult]]:
        Logger().debug(f"[EVT] Firing event {name} from {sender}.")
        e = Event(name, args, kwargs)
        r = self.handle(sender, e)
        if propagate:
            EventDistributorConnector().send(e)
        return r

    def handle(self, sender: Any, event: Event) -> tuple[EventHandlerResultType, list[EventHandlerResult]]:
        Logger().debug(f"[EVT] Handling event {event.name} from {sender} originating from {event.origin}.")
        if not self.is_registered_event(event.name):
            Logger().debug(f"[EVT] Event {event.name} not registered.")
            return EventHandlerResultType.IGNORED, []
        assert self._event_handlers[event.name] is not None
        results = []
        for _, handler in self._event_handlers[event.name]:
            result = handler(sender, *event.args, **event.kwargs)
            results.append(result)
            match result.type:
                case EventHandlerResultType.SUCCESS:
                    continue
                case EventHandlerResultType.ERROR_HANDLED:
                    continue
                case EventHandlerResultType.IGNORED:
                    continue
                case EventHandlerResultType.CANCEL:
                    return EventHandlerResultType.CANCEL, results
                case EventHandlerResultType.ERROR_UNHANDLED:
                    return EventHandlerResultType.ERROR_UNHANDLED, results
                case _:
                    raise ValueError(f"Unknown event result type: {result.type}")
        return EventHandlerResultType.SUCCESS, results


class EventDistributor(metaclass=Singleton):
    """This class handles the distribution of events to EventSystems in a multi-processing setup acting as master."""

    def __init__(self):
        Logger().debug(f"[EVT] Starting EventDistributor.")
        self.connections_lock = t.Lock()
        self.connections: dict[str, mpc.Connection[Event, Event]] = {}
        self.active: bool = True
        self.distributor: t.Thread | None = t.Thread(target=self._distribute)
        self.distributor.start()

    def stop(self):
        Logger().debug(f"[EVT] Stopping EventDistributor.")
        self.active = False
        if self.distributor is not None:
            self.distributor.join()
        with self.connections_lock:
            for conn in self.connections.values():
                conn.close()
        self.distributor = None
        self.connections.clear()

    def add_event_system(self, name: str) -> mpc.Connection:
        """Add an event system to the distributor."""
        Logger().debug(f"[EVT] Adding new event system to EventDistributor with name {name}.")
        with self.connections_lock:
            if name in self.connections:
                raise ValueError(f"Event system with name {name} already exists.")
            mine, other = mp.Pipe()
            self.connections[name] = mine
            return other

    def distribute(self):
        """Distribute events to all connected event systems."""
        # clone connection dict to avoid modifying it while iterating over it
        with self.connections_lock:
            connections = self.connections.copy()
        Logger().debug(f"[EVT] Distributing events to {len(connections)} connected systems.")
        # collect all incoming events
        incoming_events: list[tuple[str, Event]] = []
        mpc.wait([c for c in connections.values()], timeout=5)
        for name, c in connections.items():
            while not c.closed and c.readable and c.poll(0):
                incoming_events.append((name, c.recv()))

        # distribute collected events to all other event systems
        for target_name, connection in connections.items():
            for name, event in incoming_events:
                # skip sending the event back to its origin
                if name != target_name:
                    if not connection.closed and connection.writable:
                        connection.send(event)

    def _distribute(self):
        while self.active:
            try:
                self.distribute()
            except Exception as e:
                print(f"Error distributing events: {e}")

    def __del__(self):
        self.stop()


class EventDistributorConnector(metaclass=Singleton):
    """This class handles the communication to the main event distributor."""

    def __init__(self, name: str | None = None, connection: mpc.Connection | None = None):
        if name == None or connection == None:
            assert name == None and connection == None, "Name and connection must be provided or both must be None."
            self._name = None
            self._connection = None
            self._connection_lock = None
            self._listener = None
            self._listening = False
            Logger().debug(f"[EVT] EventDistributorConnector disabled.")
        else:
            self._name = name
            self._connection: mpc.Connection[Event, Event] | None = connection
            self._connection_lock = t.Lock()
            self._listening = True
            self._listener: t.Thread | None = t.Thread(target=self._listen)
            self._listener.start()
            Logger().debug(f"[EVT] EventDistributorConnector connected as {self._name}.")

    def _listen(self) -> None:
        assert self._connection != None
        while self._listening:
            events: list[Event] = []
            while not self._connection.closed and self._connection.readable and self._connection.poll(1):
                events.append(self._connection.recv())
            if len(events) > 0:
                Logger().debug(f"[EVT] EventDistributorConnector {self._name} received {len(events)} events.")
            for event in events:
                if event is not None:
                    EventSystem().handle(None, event)

    def send(self, event: Event) -> None:
        """Send an event to the main event distributor."""
        if self._connection != None and self._connection_lock != None:
            Logger().debug(f"[EVT] EventDistributorConnector {self._name} sending event {event.name}.")
            with self._connection_lock:
                if not self._connection.closed and self._connection.writable:
                    event.origin = self._name
                    self._connection.send(event)

    def stop(self) -> None:
        Logger().debug(f"[EVT] Stopping EventDistributorConnector {self._name}.")
        self._listening = False
        if self._listener is not None:
            self._listener.join()
        if self._connection_lock != None and self._connection != None:
            with self._connection_lock:
                if not self._connection.closed:
                    self._connection.close()
            self._lister = None
            self._connection = None

    def __del__(self) -> None:
        self.stop()
