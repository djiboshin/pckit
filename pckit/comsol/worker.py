from pathlib import Path
import mph
from model import ComsolModel

from ..workers import Worker, MultiprocessingWorker, MPIWorker
from .._typevars import Task, Result

import logging

logger = logging.getLogger(__package__ + '.worker')


# noinspection PyMissingOrEmptyDocstring
class ComsolWorker(Worker):
    def __init__(
            self,
            model: ComsolModel,
            filepath: str or Path,
            mph_options: dict = None,
            client_args=None,
            client_kwargs=None
    ):
        if not isinstance(model, ComsolModel):
            raise TypeError('Model has to be an object of ComsolModel class')

        super(ComsolWorker, self).__init__(model=model)

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

    def do_the_job(self, task: Task) -> Result:
        self.model.pre_build(task)
        self.model.build()
        self.model.pre_solve(task)
        self.model.mesh()
        self.model.solve()
        results = self.model.results(task)
        self.model.pre_clear(task)
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
