import sys
import pckit
import time

# adding logging for solver and workers
# Be careful with multiprocessing and file logging handler at the same time
import logging

gs = logging.getLogger('pckit.solver')
gs.addHandler(logging.StreamHandler(sys.stdout))
gs.setLevel(logging.INFO)
gw = logging.getLogger('pckit.worker')
gw.addHandler(logging.StreamHandler(sys.stdout))
gw.setLevel(logging.DEBUG)


# adding own Model subclass with results method will be called for each task by solver
class MyModel(pckit.Model):
    def results(self, n: int) -> int:
        time.sleep(n)
        return 0


# To start, use
#   mpiexec -np 2 python -m mpi4py mpi_solver.py

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
