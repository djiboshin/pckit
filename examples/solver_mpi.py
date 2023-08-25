import sys
import pckit
import time

# adding logging for solver and workers
# Be careful with mpi and file logging handler at the same time
# You can use MPIFileHandler to avoid problems. See the logging_mpi.py example
import logging

gs = logging.getLogger('pckit.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.DEBUG)

gw = logging.getLogger('pckit.worker')
gw.addHandler(logging.StreamHandler(sys.stdout))
gw.setLevel(logging.INFO)


# adding own Model subclass with results method will be called for each task by solver
class MyModel(pckit.Model):
    def results(self, n: int) -> int:
        time.sleep(n)
        return 0


# To start with <n> workers, use
#   mpiexec -np <n> python -m mpi4py solver_mpi.py
# By default a single worker in zero rank process will be used
# This behavior can be controlled by `zero_rank_usage` argument in `get_solver` function

# The __name__ condition is not really needed since all
# spawned processes will be spawned as main.
if __name__ == '__main__':
    tasks = [1 for _ in range(10)]
    # init the model
    model = MyModel()

    # -== MPI solver ==-
    worker = pckit.MPIWorker(model)

    # init the solver
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)
