"""
This module contains different Solvers
"""
import multiprocessing
import sys
from abc import ABC, abstractmethod

from multiprocessing import Queue, JoinableQueue
from typing import Any, Sequence, Union, List, Iterator, Callable, Iterable, Generic, Optional
from ._typevars import Task, Result
import time
import logging

from .workers import Worker, MultiprocessingWorker, MPIWorker
from ._utils import tasks_sort
from .cache import DictCache, BaseCache

# TODO add info to return after .solve()
# TODO add additional threads to control workers' errors(?)

logger = logging.getLogger(__package__ + '.solver')


def task_iterator(iterable: Sequence[Task]) -> Iterator[Task]:
    """Returns Iterator from on iterable"""
    return iterable.__iter__()


class Solver(ABC, Generic[Task, Result]):
    """Base Solver class"""
    def __init__(
            self,
            worker: Worker[Task, Result],
            caching: bool = False,
            cache: Optional[BaseCache[Task, Result]] = None,
    ):
        self.worker = worker
        self.caching = caching

        if cache is None and caching:
            self.cache = DictCache()
        else:
            self.cache = cache

        self.total_workers = 0

    def solve(self,
              tasks: Sequence[Task],
              iterator: Callable[[Sequence[Task]], Union[Iterator[Task], Iterable[Task]]] = task_iterator
              ) -> List[Result]:
        """Solves tasks

        :param tasks: tasks to solve
        :param iterator: function which wraps iterable tasks list and return iterator
        :return: list of results
        """
        if self.caching:
            to_solve, cached, same = tasks_sort(tasks, self.cache)
        else:
            to_solve, cached, same = range(len(tasks)), [], []

        message = f'Starting to solve {len(tasks)} tasks with {self.total_workers} workers'
        if self.caching:
            message += f':\n\t{len(cached)} solutions will be reused\n' \
                       f'\t{len(same)} tasks are the same'
        else:
            message += ' (caching is off)'
        logger.info(message)
        start_time = time.time()

        results = [None for _ in tasks]
        task_to_solve = []
        for i, task in enumerate(tasks):
            if i in to_solve:
                task_to_solve.append(task)

        res = self._solve(tasks=task_to_solve, iterator=iterator)

        for i, task, result in zip(to_solve, task_to_solve, res):
            results[i] = result
            if self.caching:
                self.cache[task] = result

        for i in same:
            results[i] = self.cache[tasks[i]]
        for i in cached:
            results[i] = self.cache[tasks[i]]

        end_time = time.time()
        logger.info('All the tasks have been solved in %.2fs', end_time - start_time)
        return results

    @abstractmethod
    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> List[Result]:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MultiprocessingSolver(Solver):
    """Multiprocessing Solver implementation"""
    def __init__(self,
                 worker: MultiprocessingWorker,
                 workers_num: int = 1,
                 caching: bool = False,
                 cache: Optional[BaseCache[Task, Result]] = None,
                 ):

        if not isinstance(worker, MultiprocessingWorker):
            raise TypeError('Worker has to be the object of type MultiprocessingWorker')

        super().__init__(worker=worker, caching=caching, cache=cache)
        self.worker = worker

        if workers_num <= 0:
            raise RuntimeError('At least 1 worker is needed')
        self.total_workers = workers_num

        self.workers = []

        self._jobs = JoinableQueue()
        self._results = Queue()

        for _ in range(self.total_workers):
            process = multiprocessing.Process(
                target=self.worker.start_loop,
                args=(self._jobs, self._results),
                daemon=True
            )
            process.start()
            self.workers.append(process)

    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> List[Result]:
        results = [None for _ in tasks]

        for i, task in enumerate(tasks):
            self._jobs.put((i, task))

        for _ in iterator(tasks):
            (i, res) = self._results.get()
            if i == -1:
                raise RuntimeError(res)
            results[i] = res

        return results

    def _stop(self):
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
                worker.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        return False


class SimpleSolver(Solver):
    """Simplest Solver implementation"""
    def __init__(
            self,
            worker: Worker,
            caching: bool = False,
            cache: Optional[BaseCache[Task, Result]] = None
    ):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        super().__init__(worker=worker, caching=caching, cache=cache)
        self.worker.start()
        self.total_workers = 1

    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> List[Result]:
        results = [None for _ in tasks]
        for i, task in enumerate(iterator(tasks)):
            try:
                results[i] = self.worker.do_the_job(task)
            except Exception as err:
                raise RuntimeError from err
        return results


class MPISolver(Solver):
    """MPI Solver implementation"""
    def __init__(
            self,
            worker: MPIWorker,
            caching: bool = False,
            cache: Optional[BaseCache[Task, Result]] = None,
            buffer_size: int = 32768
    ):
        from mpi4py import MPI

        if not isinstance(worker, MPIWorker):
            raise TypeError('Worker has to be the object of type MPIWorker')

        super().__init__(worker=worker, caching=caching, cache=cache)
        self.worker = worker

        self.buffer_size = buffer_size
        self.comm = MPI.COMM_WORLD
        self._MPI = MPI
        self.rank = self.comm.Get_rank()
        self.total_workers = self.comm.Get_size() - 1

        if self.rank != 0:
            logger.debug('Starting loop in worker with rank %i', self.comm.Get_rank())
            self.worker.start_loop(self.comm)
            # self.start_listening()
            MPI.Finalize()
            sys.exit()

        if self.total_workers < 1:
            raise RuntimeError('Not enough processes! '
                               f'Need at least 2, detected {self.total_workers}.')

    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> List[Any]:
        results = [None for _ in tasks]
        requests = []
        for i, task in enumerate(tasks):
            dest = len(requests) % self.total_workers + 1
            req = self.comm.isend((i, task), dest=dest, tag=i)
            req.wait()
            requests.append(self.comm.irecv(self.buffer_size, source=dest))

        for _ in iterator(tasks):
            (k, (i, res)) = self._MPI.Request.waitany(requests)
            results[i] = res

        return results

    def _stop(self):
        for i in range(1, self.comm.Get_size()):
            req = self.comm.isend((None, None), dest=i)
            req.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()


def get_solver(
        worker: Union[Worker, MPIWorker, MultiprocessingWorker],
        caching: bool = False,
        cache: Optional[BaseCache[Task, Result]] = None,
        workers_num: int = 1,
        buffer_size: int = 2**15,
) -> Union[SimpleSolver, MPISolver, MultiprocessingSolver]:
    """Automatically choose solver based on worker type.

    :param worker: worker instance
    :param caching: is caching enabled
    :param cache: BaseCache object
    :param workers_num: number of workers to spawn if worker is instance of MultiprocessingWorker
    :param buffer_size: buffer size for messages with results used by MPI
    :return: solver
    """
    if isinstance(worker, MultiprocessingWorker):
        return MultiprocessingSolver(
            worker=worker,
            workers_num=workers_num,
            caching=caching,
            cache=cache
        )
    if isinstance(worker, MPIWorker):
        return MPISolver(
            worker=worker,
            caching=caching,
            cache=cache,
            buffer_size=buffer_size
        )
    if isinstance(worker, Worker):
        return SimpleSolver(
            worker=worker,
            caching=caching,
            cache=cache
        )
    raise NotImplementedError(f'can\'t get Solver for {type(worker)} worker type')
