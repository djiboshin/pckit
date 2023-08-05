"""
This module is used by solvers and models mostly
"""
from typing import Tuple, List, Sequence, Hashable
from ._typevars import Task, Result
from .cache import BaseCache


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
        if task in cache:
            cached.append(i)
        elif task in to_solve_hash:
            same.append(i)
        else:
            to_solve.append(i)
            to_solve_hash.append(task)
    return to_solve, cached, same
