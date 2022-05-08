"""
A simple package for parallel computing with Python
"""
from .models import (
    Model,
    TestModel,
    ComsolModel
)
from .workers import (
    Worker,
    ComsolMultiprocessingWorker,
    ComsolWorker,
    SimpleWorker,
    SimpleMultiprocessingWorker,
    MPIWorker,
    ComsolMPIWorker,
    SimpleMPIWorker
)
from .solvers import (
    MultiprocessingSolver,
    SimpleSolver,
    MPISolver,
    Solver,
    get_solver
)
from .task import Task
