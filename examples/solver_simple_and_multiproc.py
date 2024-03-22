import sys
import pckit
import time

# adding logging for solver and workers
# Be careful with multiprocessing and file logging handler at the same time
import logging
from pckit.logging_utils import MultiprocessingRecordFactory

# setting the formatter to the stdout
logging.setLogRecordFactory(MultiprocessingRecordFactory())
fmt = '%(asctime)s [%(proc_name)s %(hostname)s] [%(name)s] [%(levelname)s]: %(message)s'
formatter = logging.Formatter(fmt=fmt)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)

# adding stdout handler to loggers
gs = logging.getLogger('pckit.solver')
gs.addHandler(sh)
gs.setLevel(logging.DEBUG)

gw = logging.getLogger('pckit.worker')
gw.addHandler(sh)
gw.setLevel(logging.INFO)


# adding own Model subclass with results method will be called for each task by solver
class MyModel(pckit.Model):
    def results(self, n: int) -> int:
        time.sleep(n)
        return 0


if __name__ == '__main__':
    tasks = [1 for _ in range(10)]
    # init the model
    model = MyModel()

    # -== Simple solver ==-
    # init the worker
    worker = pckit.Worker(model)
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)

    # -== Multiprocessing solver ==-
    worker = pckit.MultiprocessingWorker(model)
    # init the solver. Here we can choose how many workers will be spawned
    with pckit.get_solver(worker, workers_num=2) as solver:
        time.sleep(5)   # waiting for workers to start
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)
