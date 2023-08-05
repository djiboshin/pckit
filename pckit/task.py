"""
This module contains Task class
"""
from typing import Hashable
from warnings import warn


class Task:
    """Task class"""
    def __init__(self, *args, tag=None, **kwargs):
        warn("Task class will be removed soon. Please use custom tasks.", DeprecationWarning, stacklevel=2)
        if not isinstance(tag, Hashable):
            raise TypeError('tag must be Hashable')
        self.tag = tag
        self.args = args
        self.kwargs = kwargs

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        return hash(self.tag) == hash(other.tag)
