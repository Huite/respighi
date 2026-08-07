import abc
from enum import Enum

from scipy import sparse

from respighi.constants import FloatArray


class MatrixType(Enum):
    NONSYMMETRIC = (11, 0)
    SYMMETRIC_INDEFINITE = (-2, 2)
    SYMMETRIC_POSITIVE_DEFINITE = (2, 1)

    def __init__(self, pardiso_mtype, mumps_symmetry):
        self.pardiso_mtype = pardiso_mtype
        self.mumps_sym = mumps_symmetry

    @property
    def triangle_only(self) -> bool:
        return self is not MatrixType.NONSYMMETRIC


class DirectSolver(abc.ABC):
    def __init__(self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray):
        self.A = A
        self.b = b
        self.x = x

    @abc.abstractmethod
    def analyze(self): ...

    @abc.abstractmethod
    def factorize(self): ...

    @abc.abstractmethod
    def solve(self): ...

    @abc.abstractmethod
    def solve_multi(self, B: FloatArray) -> FloatArray: ...

    @abc.abstractmethod
    def free_memory(self): ...
