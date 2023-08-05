import pckit
from dataclasses import dataclass


# Hashable task
@dataclass(eq=True, frozen=True)
class Task:
    n: int


class MyModel(pckit.Model):
    def results(self, task: Task) -> int:
        return task.n


if __name__ == '__main__':
    tasks = [Task(1), Task(n=2)]
    model = MyModel()

    worker = pckit.Worker(model)
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)
