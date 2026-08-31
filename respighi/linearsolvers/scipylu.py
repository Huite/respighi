import numpy as np
from scipy import sparse

from respighi.constants import FloatArray
from respighi.linearsolvers.solvertypes import DirectSolver


class ScipyLUWrapper(DirectSolver):
    """
    Wrapper around scipy.sparse.linalg.splu.
    Pure-Python fallback, no native dependencies.
    Slower than Pardiso/MUMPS but useful for testing or unsupported platforms.
    """

    def __init__(self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray):
        self.A = A
        self.b = b
        self.x = x
        self._lu = None

    def analyze(self):
        pass  # scipy combines analysis and factorization in splu

    def factorize(self):
        self._lu = sparse.linalg.splu(self.A.tocsc())

    def solve(self):
        self.x[:] = self._lu.solve(self.b)
        return True, 1

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        return self._lu.solve(B)

    def free_memory(self):
        self._lu = None
