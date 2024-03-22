"""
This module contains different Workers
"""
import multiprocessing
import queue
import logging
from typing import Generic, NamedTuple

from .models import Model
from .task import Task as OldTask
from ._typevars import Task, Result
from ._mpi_check import mpi_function

logger = logging.getLogger(__package__ + '.worker')


class NumberedTask(NamedTuple):
    id: int
    task: Task


class Worker(Generic[Task, Result]):
    """
    The Worker abstract base class.
    """
    def __init__(
            self,
            model: Model[Task, Result]
    ):
        self.model = model

    def start(self):
        """
        Function that stars the worker.
        Always called on each worker by solver when initializing.

        :return:
        """
        logger.debug('Starting the worker')

    def do_the_job(self, task: Task, task_id: int = None) -> Result:
        """
        calls Model.results() with specified params

        :param task: task for Model.results()
        :param task_id: id of task
        :return: pass Model.results()
        """
        if task_id is None:
            logger.debug('Starting doing the job')
        else:
            logger.debug('Starting doing the job %i', task_id)

        if isinstance(task, OldTask):
            return self.model.results(*task.args, **task.kwargs)
        return self.model.results(task)


class MultiprocessingWorker(Worker):
    """Class of basic multiprocessing worker."""
    def start_loop(
            self,
            jobs: multiprocessing.JoinableQueue,
            results: multiprocessing.Queue
    ):
        """
        Loop starting will be called by multiprocessing solver.

        :param JoinableQueue jobs: Queue for jobs
        :param Queue results: Queue for results
        """
        self.start()
        self._loop(jobs, results)

    def _loop(
            self,
            jobs: multiprocessing.JoinableQueue,
            results: multiprocessing.Queue
    ):
        logger.info('Entering the loop')
        while True:
            numbered_task = NumberedTask(*jobs.get())
            if numbered_task.id is None and numbered_task.task is None:
                jobs.task_done()
                break
            try:
                res = self.do_the_job(task=numbered_task.task, task_id=numbered_task.id)
                results.put((numbered_task.id, res))
                jobs.task_done()
            except Exception as err:
                logger.exception(err)
                results.put((-1, err))


class MPIWorker(Worker):
    """Class of basic MPI worker"""
    @mpi_function
    def __init__(
            self,
            model: Model
    ):
        super(MPIWorker, self).__init__(model=model)

    def start_loop(self, comm):
        """Loop start will be called by MPI solver.

        :param comm: Intracomm, COMM_WORLD commonly
        """
        from mpi4py import MPI
        if not isinstance(comm, MPI.Intracomm):
            raise TypeError('comm has to be instance of mpi4py.MPI.Intracomm')
        self.start()
        self._loop(comm)

    def start_threading_loop(self, comm, jobs: queue.Queue):
        from mpi4py import MPI
        if not isinstance(comm, MPI.Intracomm):
            raise TypeError('comm has to be instance of mpi4py.MPI.Intracomm')
        self.start()
        self._threading_loop(comm, jobs)

    def _do_the_job_and_send(self, comm, numbered_task: NumberedTask):
        try:
            res = self.do_the_job(task=numbered_task.task, task_id=numbered_task.id)
            req = comm.isend((numbered_task.id, res), dest=0)
            req.wait()
        except Exception as err:
            logger.exception(err)
            req = comm.isend((-1, err), dest=0)
            req.wait()

    def _threading_loop(self, comm, jobs: queue.Queue):
        """Threading loop makes it possible to run worker in zero rank"""
        logger.info('Entering the threading loop')
        while True:
            numbered_task = NumberedTask(*jobs.get())
            if numbered_task.id is None and numbered_task.task is None:
                break
            self._do_the_job_and_send(
                comm=comm,
                numbered_task=numbered_task
            )
            jobs.task_done()

    def _loop(self, comm):
        logger.info('Entering the loop')
        while True:
            req = comm.irecv(source=0)
            numbered_task = NumberedTask(*req.wait())
            if numbered_task.id is None and numbered_task.task is None:
                break
            self._do_the_job_and_send(
                comm=comm,
                numbered_task=numbered_task
            )
