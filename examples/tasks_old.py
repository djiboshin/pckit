"""
Warning! pckit.Task is depreciated!
"""
import pckit


class MyModel(pckit.Model):
    def results(self, n: int) -> int:
        return n


if __name__ == '__main__':
    tasks = [pckit.Task(1), pckit.Task(n=2)]
    model = MyModel()

    worker = pckit.Worker(model)
    with pckit.get_solver(worker) as solver:
        results = solver.solve(tasks)
        print(f'Solution is ready, results: ', results)
