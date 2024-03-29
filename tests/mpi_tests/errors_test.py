import pckit
from pckit import MPIWorker, Model

import time
import numpy as np


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


class TestModel(Model):
    """
    Simple subclass of Model fot tests
    """
    def results(self, x: int) -> int:
        """Returns squared number

        :param x: any number
        :return: squared number
        """
        time.sleep(2*np.random.random())
        if x == 1:
            return x
        elif x == 0:
            return x
        raise ValueError('Test value error')


if __name__ == '__main__':
    worker = MPIWorker(TestModel())
    with pckit.get_solver(worker, zero_rank_usage=False, workers_num=3) as solver:
        # try:
        print(solver.solve([0, 1, 0, 0, 0, 0, 2, 0, 0], iterator=iterator))
        # except ValueError:
        #     print(123)

