"""
This module contains different Solvers
"""
import multiprocessing
import sys
from abc import ABC, abstractmethod

from multiprocessing import Queue, JoinableQueue
from typing import Any, Sequence, Union, List
import time
import logging

from .task import Task
from .workers import Worker, MultiprocessingWorker, MPIWorker
from ._utils import tasks_sort

# TODO add info to return after .solve()
# TODO add additional threads to control workers' errors

logger = logging.getLogger(__package__ + '.solver')


class Solver(ABC):
    """Base Solver class"""
    def __init__(self):
        self.caching = False
        self.cache = {}
        self.total_workers = 0

    def solve(self, tasks: Sequence[Task]) -> List[Any]:
        """Solves tasks

        :param tasks: tasks to solve
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

        res = self._solve(tasks, to_solve, cached, same)

        end_time = time.time()
        logger.info('All the tasks have been solved in %.2fs', end_time - start_time)
        return res

    @abstractmethod
    def _solve(
            self,
            tasks: Sequence[Task],
            to_solve: List[int],
            cached: List[int],
            same: List[int]
    ) -> List[Any]:
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
                 caching: bool = False
                 ):
        super().__init__()
        self.caching = caching
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
            to_solve: List[int],
            cached: List[int],
            same: List[int]
    ) -> List[Any]:
        results = [None for _ in tasks]
        for i, task in enumerate(tasks):
            if i in to_solve:
                self._jobs.put((i, task.args, task.kwargs))
            elif i in cached:
                results[i] = self.cache[task.tag]
            else:
                results[i] = None

        for _ in range(len(to_solve)):
            (i, res) = self._results.get()
            if i == -1:
                raise RuntimeError(res)
            if self.caching:
                self.cache[tasks[i].tag] = res
            results[i] = res

        for i in same:
            results[i] = self.cache[tasks[i].tag]

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
    def __init__(self, worker: Worker, caching: bool = False):
        if not isinstance(worker, Worker):
            raise TypeError('Worker has to be the object of type Worker')
        super().__init__()
        self.caching = caching
        self.worker = worker
        self.worker.start()
        self.total_workers = 1

    def _solve(
            self,
            tasks: Sequence[Task],
            to_solve: List[int],
            cached: List[int],
            same: List[int]
    ) -> List[Any]:
        results = [None for _ in tasks]
        for i, task in enumerate(tasks):
            if i in to_solve:
                try:
                    results[i] = self.worker.do_the_job(task.args, task.kwargs)
                except Exception as err:
                    raise RuntimeError from err
                if self.caching:
                    self.cache[task.tag] = results[i]
            elif i in cached or i in same:
                results[i] = self.cache[task.tag] if task.tag is not None else None
            else:
                results[i] = None
        return results


class MPISolver(Solver):
    """MPI Solver implementation"""
    def __init__(self, worker: MPIWorker, caching: bool = False, buffer_size: int = 32768):
        from mpi4py import MPI
        if not isinstance(worker, MPIWorker):
            raise TypeError('Worker has to be the object of type MPIWorker')
        super().__init__()
        self.worker = worker
        self.caching = caching
        self.buffer_size = buffer_size
        self.comm = MPI.COMM_WORLD
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
            to_solve: List[int],
            cached: List[int],
            same: List[int]
    ) -> List[Any]:
        results = [None for _ in tasks]
        requests = []
        for i, task in enumerate(tasks):
            if i in to_solve:
                dest = len(requests) % self.total_workers + 1
                req = self.comm.isend((i, task.args, task.kwargs), dest=dest, tag=i)
                req.wait()
                requests.append(self.comm.irecv(self.buffer_size, source=dest))
            elif i in cached:
                results[i] = self.cache[task.tag]
        for request in requests:
            i, res = request.wait()
            if self.caching:
                self.cache[tasks[i].tag] = res
            results[i] = res

        for i in same:
            results[i] = self.cache[tasks[i].tag]

        return results

    def _stop(self):
        for i in range(1, self.comm.Get_size()):
            req = self.comm.isend((None, None, None), dest=i)
            req.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()


def get_solver(
        worker: Union[Worker, MPIWorker, MultiprocessingWorker],
        caching: bool = False,
        workers_num: int = 1,
        buffer_size: int = 2**15,
) -> Union[SimpleSolver, MPISolver, MultiprocessingSolver]:
    """Automatically choose solver based on worker type.

    :param worker: worker instance
    :param caching: is caching enabled
    :param workers_num: number of workers to spawn if worker is instance of MultiprocessingWorker
    :param buffer_size: buffer size for messages with results used by MPI
    :return: solver
    """
    if isinstance(worker, MultiprocessingWorker):
        return MultiprocessingSolver(worker=worker, workers_num=workers_num, caching=caching)
    if isinstance(worker, MPIWorker):
        return MPISolver(worker, caching, buffer_size)
    if isinstance(worker, Worker):
        return SimpleSolver(worker, caching)
    raise NotImplementedError(f'can\'t get Solver for {type(worker)} worker type')
