import pytest


@pytest.fixture(scope='session', autouse=True)
def set_multiprocessing_start_method():
    # https://github.com/pytest-dev/pytest/issues/11174
    import multiprocessing as mp
    mp.set_start_method("forkserver", force=True)
