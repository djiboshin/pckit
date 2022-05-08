# pckit

This is a simple package for parallel computing with Python.

## Usage
### Multiprocessing
Simple multiprocess solver usage.
Here the square of 2 and 3 are evaluated by 2 workers.

```python
import pckit.task
import pckit


class MyModel(pckit.Model):
    def results(self, x):
        # solve here your problem f(x) = x^2
        return x ** 2


if __name__ == '__main__':
    model = MyModel()
    worker = pckit.SimpleMultiprocessingWorker(model)
    with pckit.get_solver(worker, workers_num=2) as solver:
        # create tasks to solve. You can put args or kwargs here
        tasks = [pckit.task.Task(2), pckit.task.Task(x=3)]
        results = solver.solve(tasks)
        print(results)
        # >>> [4, 9]
```

### MPI
You can simply run scripts on the cluster with [mpi4py](https://github.com/mpi4py/mpi4py) implementation on MPI. 
Simply by changing `SimpleMultiprocessingWorker` to `SimpleMPIWorker` in previous example and starting script with MPI `mpiexec -np 3 python -m mpi4py your_script.py`.

```python
worker = pckit.SimpleMPIWorker(model)
```

### Examples
[More examples](./examples)

## Features
### Cache
Dict based cache is available by `caching` argument in solver.
`tag` property in `Task` is required and has to be Hashable.

```python
with pckit.get_solver(worker, caching=True) as solver:
    tasks = [pckit.Task(2, tag='2'), pckit.Task(2, tag='2')]
```
Second result will be reused from cache.