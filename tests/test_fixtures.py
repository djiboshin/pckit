import pytest
from pckit import Model


class TestModel(Model):
    """
    Simple subclass of Model fot tests
    """
    def results(self, x: int) -> int:
        """Returns squared number

        :param x: any number
        :return: squared number
        """
        if x == 1:
            return x
        elif x == 0:
            return x
        raise ValueError('Test value error')


@pytest.fixture
def model():
    return TestModel()
