import pckit


class MyCache(pckit.BaseCache):
    def __init__(self):
        self.data = {}

    def __contains__(self, item):
        print(f'__contains__: {item}')
        return item in self.data

    def __getitem__(self, item):
        print(f'__getitem__: {item}')
        return self.data[item]

    def __setitem__(self, key, value):
        print(f'__setitem__: {key}, {value}')
        self.data[key] = value

    def __enter__(self):
        print('__enter__')

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__')


class MyModel(pckit.Model):
    def results(self, task: int) -> int:
        print('results')
        return task**2


if __name__ == '__main__':
    tasks = [0, 0, 1, 2, 3, 3, 2]
    model = MyModel()
    cache = MyCache()

    worker = pckit.Worker(model)
    with pckit.get_solver(worker, caching=True, cache=cache) as solver:
        results = solver.solve(tasks)
        print(tasks, '->', results)
