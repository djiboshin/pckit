"""
A simple package for parallel computing with Python
"""
from .models import (
    Model,
    ComsolModel
)
from .workers import (
    Worker,
    ComsolMultiprocessingWorker,
    ComsolWorker,
    Worker,
    MultiprocessingWorker,
    MPIWorker,
    ComsolMPIWorker,
    MPIWorker
)
from .solvers import (
    MultiprocessingSolver,
    SimpleSolver,
    MPISolver,
    Solver,
    get_solver
)

from .task import (
    DataclassTask
)
