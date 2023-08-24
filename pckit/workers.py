"""
This module contains different Workers
"""
from multiprocessing import JoinableQueue, Queue
import logging
from typing import Generic, NamedTuple

from .models import Model
from .task import Task as OldTask
from ._typevars import Task, Result

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

    def do_the_job(self, task: Task) -> Result:
        """
        calls Model.results() with specified params

        :param task: task for Model.results()
        :return: pass Model.results()
        """
        if isinstance(task, OldTask):
            return self.model.results(*task.args, **task.kwargs)
        return self.model.results(task)


class MultiprocessingWorker(Worker):
    """Class of basic multiprocessing worker."""
    def start_loop(self, jobs: JoinableQueue, results: Queue):
        """
        Loop starting will be called by multiprocessing solver.

        :param JoinableQueue jobs: Queue for jobs
        :param Queue results: Queue for results
        """
        self.start()
        self._loop(jobs, results)

    def do_the_job(self, numbered_task: NumberedTask) -> Result:
        logger.debug('Starting doing the job %i', numbered_task.id)
        return super(MultiprocessingWorker, self).do_the_job(numbered_task.task)

    def _loop(self, jobs: JoinableQueue, results: Queue):
        logger.info('Entering the loop')
        while True:
            numbered_task = NumberedTask(*jobs.get())
            try:
                res = self.do_the_job(numbered_task)
                results.put((numbered_task.id, res))
                jobs.task_done()
            except Exception as err:
                results.put((-1, err))
                raise err


class MPIWorker(Worker):
    """Class of basic MPI worker"""
    def start_loop(self, comm):
        """Loop start will be called by MPI solver.

        :param comm: Intracomm, COMM_WORLD commonly
        """
        import mpi4py
        if not isinstance(comm, mpi4py.MPI.Intracomm):
            raise TypeError('comm has to be instance of mpi4py.MPI.Intracomm')
        self.start()
        self._loop(comm)

    def do_the_job(self, numbered_task: NumberedTask) -> Result:
        logger.debug('Starting doing the job %i', numbered_task.id)
        return super(MPIWorker, self).do_the_job(numbered_task.task)

    def _loop(self, comm):
        logger.info('Entering the loop')
        while True:
            req = comm.irecv(source=0)
            numbered_task = NumberedTask(*req.wait())
            if numbered_task.id is None and numbered_task.task is None:
                return
            res = self.do_the_job(numbered_task)
            req = comm.isend((numbered_task.id, res), dest=0)
            req.wait()
