"""
This module contains different Workers
"""
import multiprocessing
from multiprocessing import JoinableQueue, Queue
import socket
import logging
from typing import Generic

from .models import Model
from .task import Task as OldTask
from ._typevars import Task, Result

logger = logging.getLogger(__package__ + '.worker')


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
        logger.info('%s %s Starting',
                    multiprocessing.current_process().name,
                    socket.gethostname())
        self.start()
        self._loop(jobs, results)

    def _loop(self, jobs: JoinableQueue, results: Queue):
        logger.info('%s %s Entering the loop',
                    multiprocessing.current_process().name,
                    socket.gethostname())
        while True:
            (i, task) = jobs.get()
            logger.debug('%s %s Starting doing the job %i',
                         multiprocessing.current_process().name,
                         socket.gethostname(),
                         i)
            try:
                res = self.do_the_job(task)
                results.put((i, res))
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
        logger.info('Rang %i %s Starting',
                    comm.Get_rank(),
                    socket.gethostname())
        self.start()
        self._loop(comm)

    def _loop(self, comm):
        logger.info('Rang %i %s Entering the loop',
                    comm.Get_rank(),
                    socket.gethostname())
        while True:
            req = comm.irecv(source=0)
            (i, task) = req.wait()
            if i is None and task is None:
                return
            logger.debug('Rang %i %s Starting doing the job %i',
                         comm.Get_rank(),
                         socket.gethostname(),
                         i)
            res = self.do_the_job(task)
            req = comm.isend((i, res), dest=0)
            req.wait()
