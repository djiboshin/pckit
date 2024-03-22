"""
This module contains different Solvers
"""
import multiprocessing
import queue
import sys
from abc import ABC, abstractmethod

from multiprocessing import Queue, JoinableQueue
import threading
from typing import Sequence, Union, List, Iterator, Callable, Iterable, Generic, Optional, Generator, Tuple, Hashable
from ._typevars import Task, Result
from ._mpi_check import mpi_function
import time
import logging


from .workers import Worker, MultiprocessingWorker, MPIWorker
from .cache import DictCache, BaseCache

# TODO add info to return after .solve()
# TODO add additional threads to control workers' errors(?)
# TODO add zero rank worker for MPI solver

logger = logging.getLogger(__package__ + '.solver')


def tasks_sort(tasks: Sequence[Task], cache: BaseCache[Task, Result]) -> Tuple[List[int], List[int], List[int]]:
    """Sorts the tasks list into:
        * those that need to be solved
        * those that are already in the cache
        * those that are repeated in the tasks list
    and returns three lists of task indexes

    :param tasks: List of tasks
    :param cache: BaseCache
    :return: tasks indexes sorted into three lists
    :raises ValueError: raises if any task is not Hashable
    """
    to_solve, cached, same = [], [], []
    to_solve_hash = []
    for i, task in enumerate(tasks):
        if not isinstance(task, Hashable):
            raise ValueError('If caching is True all the tasks must be hashable')
        if task in to_solve_hash:
            same.append(i)
        elif task in cache:
            cached.append(i)
        else:
            to_solve.append(i)
            to_solve_hash.append(task)
    return to_solve, cached, same


def task_iterator(iterable: Sequence[Task]) -> Iterator[Task]:
    """Returns Iterator from on iterable"""
    return iterable.__iter__()


class WorkerSideError(RuntimeError):
    pass


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
        results = [None for _ in tasks]
        message = f'Starting to solve {len(tasks)} tasks with {self.total_workers} workers'
        if not self.caching:
            message += ' (caching is off)'
            logger.info(message)
            start_time = time.time()

            results = [None for _ in tasks]
            for i, result in self._solve(tasks=tasks, iterator=iterator):
                results[i] = result
        else:
            with self.cache:
                to_solve, cached, same = tasks_sort(tasks, self.cache)
                message += f':\n\t{len(cached)} solutions will be reused\n' \
                           f'\t{len(same)} tasks are the same'
                logger.info(message)
                start_time = time.time()

                task_to_solve = [tasks[i] for i in to_solve]
                for j, result in self._solve(tasks=task_to_solve, iterator=iterator):
                    i = to_solve[j]
                    task = tasks[i]
                    results[i] = result
                    self.cache[task] = result

                for i in same:
                    task = tasks[i]
                    results[i] = self.cache[task]
                for i in cached:
                    task = tasks[i]
                    results[i] = self.cache[task]

        end_time = time.time()
        logger.info('All the tasks have been solved in %.2fs', end_time - start_time)
        return results

    @abstractmethod
    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> Generator[tuple[int, Result], None, None]:
        """Inner solve function"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        self.__exit__(None, None, None)


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
    ) -> Generator[tuple[int, Result], None, None]:
        for i, task in enumerate(tasks):
            self._jobs.put((i, task))

        for _ in iterator(tasks):
            (i, res) = self._results.get()
            if i == -1:
                error = WorkerSideError('The error has been raised at some worker process')
                logger.error(error)
                raise error

            # enables print() in the iterator when running with mpiexec
            sys.stdout.flush()

            yield i, res

    def _stop_workers(self):
        logger.debug("Stopping the workers")
        for i in range(self.total_workers):
            self._jobs.put((None, None))
        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
                worker.close()

    def _stop(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_workers()
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
    ) -> Generator[tuple[int, Result], None, None]:
        for i, task in enumerate(iterator(tasks)):
            try:
                yield i, self.worker.do_the_job(task, task_id=i)
            except Exception as err:
                logger.exception(err)
                raise WorkerSideError()


class MPISolver(Solver):
    """MPI Solver implementation"""
    @mpi_function
    def __init__(
            self,
            worker: MPIWorker,
            caching: bool = False,
            cache: Optional[BaseCache[Task, Result]] = None,
            buffer_size: int = 32768,
            zero_rank_usage: bool = True
    ):
        if not isinstance(worker, MPIWorker):
            raise TypeError('Worker has to be the object of type MPIWorker')

        super().__init__(worker=worker, caching=caching, cache=cache)
        self.worker = worker

        self.buffer_size = buffer_size
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self._MPI = MPI
        self.rank = self.comm.Get_rank()

        self.zero_rank_usage = zero_rank_usage
        if zero_rank_usage:
            self.total_workers = self.comm.Get_size()
            self.workers_ranks = tuple(range(self.total_workers))
        else:
            self.total_workers = self.comm.Get_size() - 1
            self.workers_ranks = tuple(range(1, self.total_workers + 1))

        if self.rank != 0:
            logger.debug('Starting loop in worker with rank %i', self.rank)
            self.worker.start_loop(self.comm)
            self._stop()
        elif zero_rank_usage:
            logger.debug('Starting loop in worker with rank %i (zero_rank_usage is True)', self.rank)
            self.jobs = queue.Queue()
            self.thread = threading.Thread(
                target=self.worker.start_threading_loop,
                args=(self.comm, self.jobs),
                daemon=True
            )
            self.thread.start()

        if self.total_workers < 1:
            raise RuntimeError('Not enough workers! '
                               f'Need at least 1, detected {self.total_workers}.')

    def _send_job(self, dest, job: Tuple[int, Task]):
        if dest != 0:
            req = self.comm.isend(job, dest=dest)
            req.wait()
        else:
            # for zero-rank worker the jobs queue is used
            self.jobs.put(job)
        # create request for the job result
        request = self.comm.irecv(self.buffer_size, source=dest)
        return request

    def _solve(
            self,
            tasks: Sequence[Task],
            iterator: Callable[[Sequence[Task]], Iterator[Task]]
    ) -> Generator[tuple[int, Result], None, None]:
        requests = []
        tasks_generator = enumerate(tasks)

        def get_new_task_and_send(dest):
            """Get the next task from iterator and send it to the dest worker"""
            try:
                task_number, task = next(tasks_generator)
                request = self._send_job(dest, (task_number, task))
                requests.append(request)
            except StopIteration:
                pass

        for worker_rank in self.workers_ranks:
            # send one task for each worker
            get_new_task_and_send(worker_rank)

        for _ in iterator(tasks):
            # Status is used to get source rank of the result
            status = self._MPI.Status()
            # wait for any task to be done
            (k, (i, res)) = self._MPI.Request.waitany(requests, status=status)
            requests.pop(k)
            if i == -1:
                error = WorkerSideError(f'The error has been raised at the rank {status.source} worker')
                logger.error(error)
                raise error
            # send a new task for the worker
            get_new_task_and_send(status.source)

            # enables print() in the iterator when running with mpiexec
            sys.stdout.flush()

            yield i, res

    def _stop_workers(self):
        logger.debug("Stopping the workers")
        if self.rank == 0:
            for i in range(1, self.comm.Get_size()):
                req = self.comm.isend((None, None), dest=i)
                req.wait()
            if self.zero_rank_usage:
                self.jobs.put((None, None))
                if self.thread.is_alive():
                    self.thread.join()

    def _stop(self):
        # shutdown logging will cause `MPIFileHandler` to be closed
        logging.shutdown()
        self._MPI.Finalize()
        if self.rank != 0:
            sys.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_workers()
        self._stop()
        return False


def get_solver(
        worker: Union[Worker, MPIWorker, MultiprocessingWorker],
        caching: bool = False,
        cache: Optional[BaseCache[Task, Result]] = None,
        workers_num: int = 1,
        buffer_size: int = 2**15,
        zero_rank_usage: bool = True
) -> Union[SimpleSolver, MPISolver, MultiprocessingSolver]:
    """Automatically choose solver based on worker type.

    :param worker: worker instance
    :param caching: is caching enabled
    :param cache: BaseCache object
    :param workers_num: number of workers to spawn if worker is instance of MultiprocessingWorker
    :param buffer_size: buffer size for messages with results used by MPI
    :param zero_rank_usage: start MPIWorker in zero rank process
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
            buffer_size=buffer_size,
            zero_rank_usage=zero_rank_usage
        )
    if isinstance(worker, Worker):
        return SimpleSolver(
            worker=worker,
            caching=caching,
            cache=cache
        )
    raise NotImplementedError(f'can\'t get Solver for {type(worker)} worker type')
