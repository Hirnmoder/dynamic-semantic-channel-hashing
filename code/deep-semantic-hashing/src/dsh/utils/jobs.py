from dataclasses import dataclass
import multiprocessing as mp
import threading as t
import time
from typing import Callable, Final, Generic, TypeVar

from dsh.utils.random import random_hex_string


class SpecialJobNames:
    Shutdown: Final = "[special].shutdown"


In = TypeVar("In")
Out = TypeVar("Out")


@dataclass
class Job(Generic[In, Out]):
    name: str
    func: Callable[[In], Out] | None
    args: In | None


@dataclass
class Result(Generic[Out]):
    name: str
    success: bool
    data: Out | None
    time: float
    exception: Exception | None


class JobManager(Generic[In, Out]):
    def __init__(self, num_processes=1):
        self._job_queue: mp.Queue[Job[In, Out]] = mp.Queue()
        self._result_queue: mp.Queue[Result[Out]] = mp.Queue()
        self._then_handler: dict[str, Callable[[Result[Out]], None]] = {}
        self._processes: list[Worker] = []
        for _ in range(num_processes):
            w = Worker(self._job_queue, self._result_queue)
            self._processes.append(w)
            w.start()
        self._listener: t.Thread = t.Thread(target=self._listen)
        self._listener.start()

    def submit(
        self,
        func: Callable[[In], Out],
        args: In,
        then: Callable[[Result[Out]], None] | None = None,
        id: str | None = None,
    ) -> None:
        while id is None or id in self._then_handler:
            id = random_hex_string(8)
        if then != None:
            self._then_handler[id] = then
        self._job_queue.put_nowait(Job(id, func, args))

    def close(self, timeout: float | None = None):
        for w in self._processes:
            self._job_queue.put_nowait(Job(SpecialJobNames.Shutdown, None, None))
        self._result_queue.put_nowait(Result(SpecialJobNames.Shutdown, True, None, 0, None))

        if timeout == None:
            for w in self._processes:
                w.join()
        else:
            start = time.time()
            still_alive = False
            while time.time() - start < timeout:
                still_alive = False
                for w in self._processes:
                    still_alive |= w.is_alive()
                if not still_alive:
                    break
            if still_alive:
                # kill processes
                for w in self._processes:
                    if w.is_alive():
                        w.terminate()
                for w in self._processes:
                    w.join()

        self._listener.join()

    def _listen(self):
        while True:
            try:
                r = self._result_queue.get()
                if r.name == SpecialJobNames.Shutdown:
                    break
                else:
                    if r.name in self._then_handler:
                        h = self._then_handler[r.name]
                        h(r)
            finally:
                pass


class Worker(mp.Process, Generic[In, Out]):
    def __init__(self, job_queue: mp.Queue, result_queue: mp.Queue):
        super(Worker, self).__init__()
        self._job_queue: mp.Queue[Job[In, Out]] = job_queue
        self._result_queue: mp.Queue[Result[Out]] = result_queue

    def run(self):
        """Run the worker loop, constantly fetching and executing jobs from the queue."""
        while True:
            job = self._job_queue.get()
            if job.name == SpecialJobNames.Shutdown:
                break
            assert job.func != None
            assert job.args != None
            start = time.time()
            try:
                self.name = job.name
                data = job.func(job.args)
                success = True
                exception = None
            except Exception as e:
                data = None
                success = False
                exception = e
            except:
                data = None
                success = False
                exception = None
            finally:
                end = time.time()
                result = Result(name=job.name, success=success, data=data, time=end - start, exception=exception)
            self._result_queue.put(result)
