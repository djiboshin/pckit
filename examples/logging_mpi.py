import sys
import pckit
import time
from pckit.logging_utils import MPIFileHandler, MPIRecordFactory
import logging

# add rank and hostname attrs for format parsing
logging.setLogRecordFactory(MPIRecordFactory())
fmt = '%(asctime)s [Rank %(rank)s %(hostname)s] [%(name)s] [%(levelname)s]: %(message)s'
formatter = logging.Formatter(fmt=fmt)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# add stdout handler
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)
# add file handler (MPI-safe)
fh = MPIFileHandler('log.txt')
fh.setFormatter(formatter)
logger.addHandler(fh)


# adding own Model subclass with results method will be called for each task by solver
class MyModel(pckit.Model):
    def results(self, n: int) -> int:
        time.sleep(n)
        return 0


# To start with <n> workers, use
#   mpiexec -np <n> python -m mpi4py mpi_solver.py
# See the solver_mpi.py example for more details

if __name__ == '__main__':
    tasks = [1 for _ in range(2)]
    # init the model
    model = MyModel()

    # -== MPI solver ==-
    worker = pckit.MPIWorker(model)

    # init the solver
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        logger.info(f'Solution is ready, results: {results}')

