"""
This module contains different useful classes
"""
import logging
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass


class MPIFileStream:
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
