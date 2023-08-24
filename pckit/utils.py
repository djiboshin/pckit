"""
This module contains different useful classes
"""
import logging
import multiprocessing
import socket
from ._mpi_check import mpi_function

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass


class MPIFileStream:
    """File stream. See https://gist.github.com/sixy6e/ed35ea88ba0627e0f7dfdf115a3bf4d1"""
    @mpi_function
    def __init__(self, filename, amode: int, comm: 'MPI.Intracomm', encoding: str = 'utf-8'):
        self.file = MPI.File.Open(comm=comm, filename=filename, amode=amode)
        self.file.Set_atomicity(flag=True)
        self.encoding = encoding

    def write(self, msg: str):
        self.file.Write_shared(msg.encode(encoding=self.encoding))

    def flush(self):
        # Transfer all previous writes to the storage device
        self.file.Sync()

    def close(self):
        self.file.Close()


class MPIFileHandler(logging.FileHandler):
    """FileHandler implementation supports writing in single file"""
    @mpi_function
    def __init__(
            self,
            filename,
            amode: int = None,
            comm: 'MPI.Intracomm' = None,
            encoding: str = None,
            delay=False
    ):
        self.amode = amode if amode is not None else MPI.MODE_WRONLY | MPI.MODE_CREATE
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        super(MPIFileHandler, self).__init__(filename=filename, delay=delay, encoding=encoding)

    def _open(self):
        encoding = 'utf-8' if self.encoding is None else self.encoding
        return MPIFileStream(
            filename=self.baseFilename,
            amode=self.amode,
            comm=self.comm,
            encoding=encoding,
        )


class MPIRecordFactory:
    """Class for logging with rank and hostname in format"""
    @mpi_function
    def __init__(self, comm: 'MPI.Intracomm' = None) -> None:
        self.default_factory = logging.getLogRecordFactory()
        self.rank = MPI.COMM_WORLD.Get_rank() if comm is None else comm
        self.hostname = socket.gethostname()

    def __call__(self, *args, **kwargs) -> logging.LogRecord:
        record = self.default_factory(*args, **kwargs)
        record.rank = self.rank
        record.hostname = self.hostname
        return record


class MultiprocessingRecordFactory:
    """Class for logging with process name and hostname in format"""
    def __init__(self) -> None:
        self.default_factory = logging.getLogRecordFactory()
        self.proc_name = multiprocessing.current_process().name
        self.hostname = socket.gethostname()

    def __call__(self, *args, **kwargs) -> logging.LogRecord:
        record = self.default_factory(*args, **kwargs)
        record.proc_name = self.proc_name
        record.hostname = self.hostname
        return record
