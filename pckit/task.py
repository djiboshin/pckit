"""
This module contains Task class
"""
from typing import Hashable


class Task:
    """Task class"""
    def __init__(self, *args, tag=None, **kwargs):
        if not isinstance(tag, Hashable):
            raise TypeError('tag must be Hashable')
        self.tag = tag
        self.args = args
        self.kwargs = kwargs
