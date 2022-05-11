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
    def results(self, x):
        # Solve here problem f(x) = x^2
        return x ** 2


if __name__ == '__main__':
    model = MyModel()
    worker = pckit.SimpleMultiprocessingWorker(model)
    with pckit.get_solver(worker, workers_num=2) as solver:
        # Create tasks to solve. You can put args or
        # kwargs for model.results() method in the Task
        tasks = [pckit.Task(2), pckit.Task(x=3)]
        results = solver.solve(tasks)
        print(results)
        # >>> [4, 9]
```

### MPI
You can easily run scripts on the cluster with [mpi4py](https://github.com/mpi4py/mpi4py) implementation on MPI (See [mpi4py installation docs](https://mpi4py.readthedocs.io/en/stable/install.html)).
Simply change `SimpleMultiprocessingWorker` to `SimpleMPIWorker` in the previous example and start the script with MPI `mpiexec -np 3 python -m mpi4py your_script.py`

```python
worker = pckit.SimpleMPIWorker(model)
```
Moreover, a multiprocessing solver can be started inside an MPI solver.

### Single thread
Single threaded execution is also available with `SimpleWorker`

```python
worker = pckit.SimpleWorker(model)
```

### Examples
[More examples](https://github.com/djiboshin/pckit/tree/main/examples)

## Features
### Cache
Dict based cache is available by `caching` argument in `get_solver()`.
`tag` property in `Task` is required and has to be hashable.

```python
with pckit.get_solver(worker, caching=True) as solver:
    tasks = [pckit.Task(2, tag='2'), pckit.Task(2, tag='2')]
```
The second task's solution will be reused from the cache.

### Custom iterators
You can send the email or print anything with custom iterator.
[tqdm](https://pypi.org/project/tqdm/) is also supported.
```python
import tqdm

results = solver.solve(tasks, iterator=tqdm.tqdm)
```
See [example](https://github.com/djiboshin/pckit/blob/main/examples/custom_iterator.py) to create your own iterator.

### Comsol Models, Solvers, Workers
Based on [MPh](https://pypi.org/project/MPh/) package.

**TBD**ocumented
