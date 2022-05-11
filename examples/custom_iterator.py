import time
import pckit


class MyModel(pckit.Model):
    def results(self, n, *args, **kwargs):
        time.sleep(n)
        return 0


def iterator(lst):
    tasks_total = len(lst)
    tasks_done = 0
    try:
        for item in lst:
            yield item
            tasks_done += 1
            print(f'{tasks_done} / {tasks_total}')
    finally:
        pass


if __name__ == '__main__':
    tasks = [pckit.Task(1) for _ in range(10)]
    # init the model
    model = MyModel()

    # -== Simple solver ==-
    worker = pckit.SimpleWorker(model)
    # init the solver
    with pckit.get_solver(worker) as solver:
        # start solution with specified iterator
        results = solver.solve(tasks, iterator=iterator)
