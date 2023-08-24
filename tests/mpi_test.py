import pckit.task
from test_fixtures import TestModel
import sys

worker = pckit.MPIWorker(model=TestModel())

if __name__ == '__main__':
    from mpi4py import MPI
    from _test_solvers import test_cache, test_solve

    if len(sys.argv) == 2:
        test_type = sys.argv[1]
        if test_type == 'basic_test_cache':
            test_cache(worker)
        elif test_type == 'basic_test_solve':
            test_solve(worker)
    else:
        with pckit.get_solver(worker=worker) as solver:
            assert isinstance(solver, pckit.MPISolver)
            comm = MPI.COMM_WORLD
            # amount spawned
            assert comm.Get_size() - 1 == solver.total_workers

            tasks = [1, 0]
            res = solver.solve(tasks)
            # right results
            assert res[0] == 1 and res[1] == 0
