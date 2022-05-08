import pckit
from mpi4py import MPI

import pckit.task
from test_solvers import basic_test_cache, basic_test_solve
import sys

worker = pckit.SimpleMPIWorker(model=pckit.TestModel())

if __name__ == '__main__':
    if len(sys.argv) == 2:
        test_type = sys.argv[1]
        if test_type == 'basic_test_cache':
            basic_test_cache(worker)
        elif test_type == 'basic_test_solve':
            basic_test_solve(worker)
    else:
        with pckit.get_solver(worker=worker) as solver:
            assert isinstance(solver, pckit.MPISolver)
            comm = MPI.COMM_WORLD
            # amount spawned
            assert comm.Get_size() - 1 == solver.total_workers

            tasks = [pckit.task.Task(1), pckit.task.Task(0)]
            res = solver.solve(tasks)
            # right results
            assert res[0] == 1 and res[1] == 0
