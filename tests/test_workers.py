import pytest

import pckit
import multiprocessing
from multiprocessing import Queue, JoinableQueue

import pckit.task
from test_fixtures import model


def test_simple_worker(model):
    worker = pckit.Worker(model=model)
    for task in [0, 1]:
        assert worker.do_the_job(task) == task
    try:
        task = 2
        worker.do_the_job(task)
        assert False
    except Exception as e:
        assert isinstance(e, ValueError)


def test_multiprocessing_worker(model):
    worker = pckit.MultiprocessingWorker(model=model)

    jobs = JoinableQueue()
    results = Queue()

#     process = multiprocessing.Process(
#         target=worker.start_loop,
#         args=(jobs, results),
#         daemon=True
#     )
#     process.start()
#     for task in [0, 1]:
#         jobs.put((1, task))
#         i, r = results.get(timeout=2)
#         assert r == task
#     task = 2
#     try:
#         jobs.put((1, task))
#         results.get(timeout=2)
#     except Exception as e:
#         assert isinstance(e, ValueError)
#
#     if process.is_alive():
#         process.terminate()
#         process.join()
#         process.close()
