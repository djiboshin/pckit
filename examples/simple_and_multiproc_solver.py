import sys
import pckit
import time

# adding logging for solver and workers
# Be careful with multiprocessing and file logging handler at the same time
import logging

import pckit.task

gs = logging.getLogger('pckit.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
gw = logging.getLogger('pckit.worker')
gw.addHandler(logging.StreamHandler(sys.stdout))
gw.setLevel(logging.DEBUG)


# adding own Model subclass with results method will be called for each task by solver
class MyModel(pckit.Model):
    def results(self, n, *args, **kwargs):
        time.sleep(n)
        return 0


if __name__ == '__main__':
    tasks = [pckit.task.Task(1) for _ in range(10)]
    # init the model
    model = MyModel()

    # -== Simple solver ==-
    # init the worker
    worker = pckit.SimpleWorker(model)
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)

    # -== Multiprocessing solver ==-
    worker = pckit.SimpleMultiprocessingWorker(model)
    # init the solver. Here we can choose how many workers will be spawned
    with pckit.get_solver(worker, workers_num=2) as solver:
        time.sleep(5)   # waiting for workers to start
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)
