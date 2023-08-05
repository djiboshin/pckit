import time
import tqdm
import pckit


class MyModel(pckit.Model):
    def results(self, n: int):
        time.sleep(n)
        return 0


if __name__ == '__main__':
    tasks = [1 for _ in range(10)]
    # init the model
    model = MyModel()

    # -== Simple solver ==-
    worker = pckit.Worker(model)
    # init the solver
    with pckit.get_solver(worker) as solver:
        # start solution with specified iterator
        results = solver.solve(tasks, iterator=tqdm.tqdm)
