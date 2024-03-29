"""
This module contains cache classes
"""
from abc import ABCMeta, abstractmethod
from typing import Generic, Dict
from ._typevars import Task, Result


class BaseCache(Generic[Task, Result], metaclass=ABCMeta):
    @abstractmethod
    def __contains__(self, item: Task) -> bool:
        """
        Returns `True` if hashable item is in the cache, `False` otherwise.

        :param item:
        :return:
        """

    @abstractmethod
    def __getitem__(self, item: Task) -> Result:
        """
        Returns result by hashable item.
        If result is not in the cache raises `KeyError`.

        :param item:
        :return:
        """

    @abstractmethod
    def __setitem__(self, item: Task, value: Result):
        """
        Set result by hashable item.

        :param item:
        :param value:
        :return:
        """

    def get(self, item: Task) -> Result:
        """
        Returns result by hashable item.
        If result is not in the cache returns `None`.

        :param item:
        :return:
        """
        if item in self:
            return self[item]
        return None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DictCache(BaseCache):
    """Simple cache based on dict."""
    def __init__(self):
        self._cache: Dict[Task, Result] = dict()

    def __getitem__(self, item: Task) -> Result:
        return self._cache[item]

    def __setitem__(self, item: Task, value: Result):
        self._cache[item] = value

    def __contains__(self, item: Task) -> bool:
        return item in self._cache

