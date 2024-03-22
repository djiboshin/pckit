import multiprocessing
from typing import Sequence, Type

import pckit
import pytest

import pckit.task
from test_fixtures import model

# TODO Turn on MPI tests


def test_get_solver(model):
    """
    Tests if the get_solver() function works properly with Worker and MultiprocessingWorker
    """
    worker = pckit.Worker(model)
    assert isinstance(pckit.get_solver(worker), pckit.SimpleSolver)

    worker = pckit.MultiprocessingWorker(model)
    assert isinstance(pckit.get_solver(worker), pckit.MultiprocessingSolver)


@pytest.fixture
def worker(request, model):
    worker_type: Type[pckit.Worker] = request.param
    return worker_type(model=model)


worker_types = [
    pckit.Worker,
    pckit.MultiprocessingWorker
]


@pytest.mark.parametrize('worker', worker_types, indirect=True)
def test_solve(worker: pckit.Worker):
    # can solve a task
    with pckit.get_solver(worker=worker) as solver:
        tasks = [1, 0]
        res = solver.solve(tasks)
        # right results
        assert res[0] == 1 and res[1] == 0


@pytest.mark.parametrize('worker', worker_types, indirect=True)
def test_cache(worker: pckit.Worker):
    # can use cache
    with pckit.get_solver(worker=worker, caching=True) as solver:
        tasks = [
            1,
        ]
        solver.solve(tasks)
        assert (1 in solver.cache) and (3 not in solver.cache)

    # cache is working
    with pckit.get_solver(worker=worker, caching=True) as solver:
        tasks = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0]
        results = solver.solve(tasks)
        for result, task in zip(results, tasks):
            assert result == task
        assert solver.cache[1] == 1
        assert solver.cache[0] == 0


@pytest.mark.parametrize('worker', worker_types, indirect=True)
def test_error(worker: pckit.Worker):
    with pckit.get_solver(worker=worker) as solver:
        # can handle errors
        tasks = [-2]
        try:
            solver.solve(tasks)
        except Exception as e:
            assert isinstance(e, RuntimeError)


def iterator(items: Sequence, iters):
    try:
        for item in items:
            iters.append(item)
            yield item
    finally:
        pass


@pytest.mark.parametrize('worker', worker_types, indirect=True)
def test_iterator(worker: pckit.Worker):
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


def test_simple_solver(model):
    """
        Tests if SimpleSolver() works properly
    """

    worker = pckit.Worker(model=model)
    with pckit.SimpleSolver(worker=worker) as solver:
        tasks = [1, 0]
        res = solver.solve(tasks)
        # right results
        assert res[0] == 1 and res[1] == 0


def test_multiprocessing_solver(model):
    """
        Tests if MultiprocessingSolver() works properly
    """
    worker = pckit.MultiprocessingWorker(model=model)

    # creates workers
    n = 4
    with pckit.MultiprocessingSolver(worker=worker, workers_num=n) as solver:
        # spawned Process
        for x in solver.workers:
            assert isinstance(x, multiprocessing.context.Process)
        # spawned alive
        for x in solver.workers:
            assert x.is_alive()
        # amount spawned
        assert len(solver.workers) == n == solver.total_workers

        tasks = [1, 0]
        res = solver.solve(tasks)
        # right results
        assert res[0] == 1 and res[1] == 0
