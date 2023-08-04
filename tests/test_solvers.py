import multiprocessing
import os.path
from typing import Sequence

import pckit
import pytest
import subprocess
import sys

import pckit.task
from test_fixtures import model


def test_get_solver(model):
    """
    Tests if the get_solver() function works properly
    """
    worker = pckit.Worker(model)
    assert isinstance(pckit.get_solver(worker), pckit.SimpleSolver)

    worker = pckit.MultiprocessingWorker(model)
    assert isinstance(pckit.get_solver(worker), pckit.MultiprocessingSolver)

    # worker = pckit.SimpleMPIWorker(pckit.TestModel())
    # if importlib.util.find_spec('mpi4py') is not None:
    #     assert isinstance(pckit.get_solver(worker), pckit.MPISolver)


@pytest.mark.skip
def basic_test_solve(worker: pckit.Worker):
    # can solve a task
    with pckit.get_solver(worker=worker) as solver:
        tasks = [1, 0]
        res = solver.solve(tasks)
        # right results
        assert res[0] == 1 and res[1] == 0


@pytest.mark.skip
def basic_test_cache(worker: pckit.Worker):
    # can use cache
    with pckit.get_solver(worker=worker, caching=True) as solver:
        tasks = [
            1,
        ]
        solver.solve(tasks)
        assert (1 in solver.cache) and (3 not in solver.cache)


@pytest.mark.skip
def basic_test_error(worker: pckit.Worker):
    with pckit.get_solver(worker=worker) as solver:
        # can handle errors
        tasks = [-2]
        try:
            solver.solve(tasks)
        except Exception as e:
            assert isinstance(e, RuntimeError)


@pytest.mark.skip
def iterator(items: Sequence, iters):
    try:
        for item in items:
            iters.append(item)
            yield item
    finally:
        pass


@pytest.mark.skip
def basic_test_iterator(worker: pckit.Worker):
    with pckit.get_solver(worker=worker) as solver:
        # can use iterator
        tasks = [
            1,
            0,
            1,
            0
        ]
        iters = []
        solver.solve(tasks, iterator=lambda x: iterator(x, iters))
        assert len(iters) == len(tasks)
        # can use iterator with Iterable instead of Iterator in return
        solver.solve(tasks, iterator=lambda x: x)


def test_basic_functions(model):
    test_workers = [
        pckit.Worker(model=model),
        pckit.MultiprocessingWorker(model=model),
    ]

    for worker in test_workers:
        basic_test_solve(worker)
        basic_test_cache(worker)
        basic_test_error(worker)
        basic_test_iterator(worker)


def test_simple_solver(model):
    """
        Tests if SimpleSolver() works properly
    """

    worker = pckit.Worker(model=model)
    solver = pckit.SimpleSolver(worker=worker)


def test_multiprocessing_solver(model):
    """
        Tests if MultiprocessingSolver() works properly
    """
    worker = pckit.MultiprocessingWorker(model=model)

    # creates workers
    n = 4
    with pckit.MultiprocessingSolver(worker=worker, workers_num=n) as solver:
        # spawned Process
        assert not any(not isinstance(x, multiprocessing.context.Process) for x in solver.workers)
        # spawned alive
        assert not any(not x.is_alive() for x in solver.workers)
        # amount spawned
        assert len(solver.workers) == n == solver.total_workers

        tasks = [1, 0]
        res = solver.solve(tasks)
        # right results
        assert res[0] == 1 and res[1] == 0


@pytest.mark.skip
def test_mpi_solver():
    """
        Tests if MPISolver() works properly
    """
    # amount spawned, right results
    path = os.path.dirname(os.path.realpath(__file__))
    test_path = os.path.join(path, 'mpi_test.py')
    base_cmd = ['mpiexec', '-np', '3', sys.executable, '-m', 'mpi4py', test_path]
    subprocess.run(base_cmd, check=True)
    # can solve a task
    base_cmd.append('basic_test_solve')
    subprocess.run(base_cmd, check=True)
    # can use cache
    base_cmd[-1] = 'basic_test_cache'
    subprocess.run(base_cmd, check=True)
