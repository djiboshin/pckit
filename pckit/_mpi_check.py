try:
    from mpi4py import MPI
    is_mpi = True
except ModuleNotFoundError:
    is_mpi = False


def mpi_function(func):
    def _wrapper(*args, **kwargs):
        if not is_mpi:
            raise ModuleNotFoundError('install mpi4py to use MPI functionality')
        return func(*args, **kwargs)
    return _wrapper
