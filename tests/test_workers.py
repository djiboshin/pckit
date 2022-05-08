import pckit
import multiprocessing
from multiprocessing import Queue, JoinableQueue

import pckit.task


def test_simple_worker():
    model = pckit.TestModel()
    worker = pckit.SimpleWorker(model=model)
    for t in [0, 1]:
        task = pckit.task.Task(t)
        assert worker.do_the_job(task.args, task.kwargs) == t
    try:
        task = pckit.task.Task(2)
        worker.do_the_job(task.args, task.kwargs)
        assert True
    except Exception as e:
        assert isinstance(e, ValueError)


def test_multiprocessing_worker():
    model = pckit.TestModel()
    worker = pckit.SimpleMultiprocessingWorker(model=model)

    jobs = JoinableQueue()
    results = Queue()

    process = multiprocessing.Process(
        target=worker.start_loop,
        args=(jobs, results),
        daemon=True
    )
    process.start()
    for t in [0, 1]:
        task = pckit.task.Task(t)
        jobs.put((1, task.args, task.kwargs))
        i, r = results.get(timeout=2)
        assert r == t
    task = pckit.task.Task(2)
    try:
        jobs.put((1, task.args, task.kwargs))
        results.get(timeout=2)
    except Exception as e:
        assert isinstance(e, ValueError)

    process.terminate()
    process.join()
    process.close()


def test_mpi_worker():
    pass
