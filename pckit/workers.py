"""
This module contains different Workers
"""
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from multiprocessing import JoinableQueue, Queue
from pathlib import Path
import socket
import logging
import mph

from .models import ComsolModel, Model

logger = logging.getLogger(__package__ + '.worker')


class Worker(ABC):
    """
    The Worker abstract base class.
    """
    @abstractmethod
    def start(self):
        """Function that stars the worker. Always called on each worker by solver when initializing.

        :return:
        """

    @abstractmethod
    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        """calls Model.results() with specified params

        :param args: args for Model.results()
        :param kwargs: kwargs for Model.results()
        :return: pass Model.results()
        """


class MultiprocessingWorker(Worker, ABC):
    """Class of basic multiprocessing worker."""
    def start_loop(self, jobs: JoinableQueue, results: Queue):
        """Loop starting will be called by multiprocessing solver.

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
            (i, args, kwargs) = jobs.get()
            logger.debug('%s %s Starting doing the job %i',
                         multiprocessing.current_process().name,
                         socket.gethostname(),
                         i)
            try:
                res = self.do_the_job(args, kwargs)
                results.put((i, res))
                jobs.task_done()
            except Exception as err:
                results.put((-1, err))
                raise err


class MPIWorker(Worker, ABC):
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
            (i, args, kwargs) = req.wait()
            if i is None and args is None and kwargs is None:
                return
            logger.debug('Rang %i %s Starting doing the job %i',
                         comm.Get_rank(),
                         socket.gethostname(),
                         i)
            res = self.do_the_job(args, kwargs)
            req = comm.isend((i, res), dest=0)
            req.wait()


class SimpleWorker(Worker):
    """Simplest implementation of worker"""
    def __init__(self, model: Model):
        self.model = model

    def start(self):
        """Start of worker"""

    def do_the_job(self, args: Tuple[Any], kwargs: Dict[str, Any]) -> Any:
        """Get model results"""
        return self.model.results(*args, **kwargs)


class SimpleMultiprocessingWorker(SimpleWorker, MultiprocessingWorker):
    """Simplest multiprocessing worker"""


class SimpleMPIWorker(SimpleWorker, MPIWorker):
    """Simplest MPI worker"""


# noinspection PyMissingOrEmptyDocstring
class ComsolWorker(Worker):
    def __init__(self, model: ComsolModel, filepath: str or Path,
                 mph_options: dict = None,
                 client_args=None,
                 client_kwargs=None):
        if not isinstance(model, ComsolModel):
            raise TypeError('Model has to be an object of ComsolModel class')

        self.client = None
        self.model = model

        self._mph_options = {} if mph_options is None else mph_options
        self._client_args = [] if client_args is None else client_args
        self._client_kwargs = {} if client_kwargs is None else client_kwargs
        self._filepath = filepath

    def start(self):
        for option in self._mph_options:
            mph.option(option, self._mph_options[option])
        logger.debug(f'Opening COMSOL model {self._filepath}')
        self.client = mph.start(*self._client_args, **self._client_kwargs)  # type: mph.client
        self.model.java = self.client.load(self._filepath).java
        self.model.configure()

    def do_the_job(self, args, kwargs) -> Any:
        self.model.pre_build(*args, **kwargs)
        self.model.build()
        self.model.pre_solve(*args, **kwargs)
        self.model.mesh()
        self.model.solve()
        results = self.model.results(*args, **kwargs)
        self.model.pre_clear(*args, **kwargs)
        self.model.clear()
        return results


# noinspection PyMissingOrEmptyDocstring
class ComsolMultiprocessingWorker(ComsolWorker, MultiprocessingWorker):
    def start(self):
        super().start()


# noinspection PyMissingOrEmptyDocstring
class ComsolMPIWorker(ComsolWorker, MPIWorker):
    def start(self):
        super().start()
