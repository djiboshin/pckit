# pckit

This is a simple package for parallel computing with Python.

## Usage
### Multiprocessing
If you want to use any solver from the package you have to wrap your functions into a model. 
Here the example with square of 2 and 3 are evaluated by 2 workers.
`MyModel` is a subclass of the package `Model`. The method `results` is required.

```python
import pckit


class MyModel(pckit.Model):
    def results(self, x: int) -> int:
        # Solve here problem f(x) = x^2
        return x ** 2


if __name__ == '__main__':
    model = MyModel()
    worker = pckit.MultiprocessingWorker(model)
    with pckit.get_solver(worker, workers_num=2) as solver:
        # Create tasks to solve
        tasks = [2, 3]
        results = solver.solve(tasks)
        print(results)
        # >>> [4, 9]
```
Workers number can be controlled by `workers_num` argument.

### MPI
You can easily run scripts on the cluster with [mpi4py](https://github.com/mpi4py/mpi4py) implementation on MPI (See [mpi4py installation docs](https://mpi4py.readthedocs.io/en/stable/install.html)).
Simply change `MultiprocessingWorker` to `MPIWorker` in the previous example and start the script with MPI `mpiexec -np <n> python -m mpi4py your_script.py`, where `<n>` is a number of workers.

```python
worker = pckit.MPIWorker(model)
```
By default, zero rank process is also used as a worker.
It can be controlled by `zero_rank_usage` argument of `get_solver` function.

[//]: # (Moreover, a multiprocessing solver can be started inside an MPI solver.)

### Single thread
Single threaded execution is also available with `Worker`

```python
worker = pckit.Worker(model)
```

### Examples
[More examples](https://github.com/djiboshin/pckit/tree/main/examples)

## Features
### Cache
Dict based cache is available by `caching` argument in `get_solver()`.
Tasks are required to be hashable.

```python
with pckit.get_solver(worker, caching=True) as solver:
    tasks = [2, 2]
```

The second task's solution will be reused from the cache.

You can create your own cache by implementing `__contains__`, `__getitem__`, `__setitem__` of the `BaseCache` class.
See [example](https://github.com/djiboshin/pckit/blob/main/examples/cache_custom.py) for more details.

### Custom iterators
You can send emails or print anything during evaluation with custom iterator.
[tqdm](https://pypi.org/project/tqdm/) is also supported.
```python
import tqdm

results = solver.solve(tasks, iterator=tqdm.tqdm)
```
See [example](https://github.com/djiboshin/pckit/blob/main/examples/iterator_custom.py) to create your own iterator.

### Logging with MPI

See [example](https://github.com/djiboshin/pckit/blob/main/examples/logging_mpi.py)

### Comsol Models, Solvers, Workers
Based on [MPh](https://pypi.org/project/MPh/) package.

**TBD**ocumented
