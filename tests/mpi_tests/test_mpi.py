# import pytest
# import pckit
#
# from test_fixtures import model
#
#
# @pytest.mark.mpi
# def test_get_solver_mpi(model):
#     """
#         Tests if the get_solver() function works properly with MPI worker
#     """
#     pytest.importorskip('mpi4py')
#     worker = pckit.MPIWorker(model)
#     assert isinstance(pckit.get_solver(worker), pckit.MPISolver)
#
#
# @pytest.mark.mpi
# def test_solve(model):
#     # can solve a task
#     with pckit.get_solver(worker=pckit.MPIWorker(model=model)) as solver:
#         tasks = [1, 0]
#         res = solver.solve(tasks)
#         # right results
#         assert res[0] == 2 and res[1] == 0
#
#
# @pytest.mark.mpi
# def test_mpi_solver():
#     """
#         Tests if MPISolver() works properly
#     """
#     import mpi4py
#     print(mpi4py.MPI.COMM_WORLD.rank)
#     # # amount spawned, right results
#     # path = os.path.dirname(os.path.realpath(__file__))
#     # test_path = os.path.join(path, 'mpi_test.py')
#     # base_cmd = ['mpiexec', '-np', '3', sys.executable, '-m', 'mpi4py', test_path]
#     # subprocess.run(base_cmd, check=True)
#     # # can solve a task
#     # base_cmd.append('basic_test_solve')
#     # subprocess.run(base_cmd, check=True)
#     # # can use cache
#     # base_cmd[-1] = 'basic_test_cache'
#     # subprocess.run(base_cmd, check=True)



