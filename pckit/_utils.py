"""
This module is used by solvers and models mostly
"""
from typing import Tuple, List, Sequence, Dict
import numpy as np
from .task import Task


def make_unique(labels: Sequence) -> list:
    """Gets labels list and makes all labels unique by adding '(number of inclusion)' postfix

    :param labels: List of labels
    :return: List of renamed labels
    """
    new_labels = []
    _, real_index, counts = np.unique(labels, return_counts=True, return_inverse=True)
    for index in range(len(labels)):
        for count in range(counts[real_index[index]]):
            new_label = labels[index] + (f'({count})' if count != 0 else '')
            if new_label not in new_labels:
                new_labels.append(new_label)
                break

    return new_labels


def tasks_sort(tasks: Sequence[Task], cache: Dict) -> Tuple[List[int], List[int], List[int]]:
    """Sorts the tasks list into:
        * those that need to be solved
        * those that are already in the cache
        * those that are repeated in the tasks list
    and returns three lists of task indexes

    :param tasks: List of tasks
    :param cache: Cache dict
    :return: tasks indexes sorted into three lists
    :raises ValueError: raises if any task does not contain
    tag property even if caching is used on solver
    """
    to_solve, cached, same = [], [], []
    to_solve_tags = []
    for i, task in enumerate(tasks):
        if task.tag is None:
            raise ValueError('If caching is True all the tasks must have a tag property')
        if task.tag in cache:
            cached.append(i)
        elif task.tag in to_solve_tags:
            same.append(i)
        else:
            to_solve.append(i)
            to_solve_tags.append(task.tag)
    return to_solve, cached, same
