"""
This module contains different Models
"""
from abc import abstractmethod, ABCMeta
from typing import Generic

from ._typevars import Task, Result


class Model(Generic[Task, Result], metaclass=ABCMeta):
    """
    The Model abstract base class.
    """
    @abstractmethod
    def results(self, task: Task) -> Result:
        """
        Function to be solved
        """


# TODO SMUTHI Model
# TODO logging here?
