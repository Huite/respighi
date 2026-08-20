import numpy as np
from scipy import sparse

from respighi.constants import FloatArray
from respighi.linearsolvers.pardiso import PardisoWrapper
from respighi.linearsolvers.solvertypes import DirectSolver, MatrixType


class ScipyWrapper(DirectSolver):
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

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        return self._lu.solve(B)

    def free_memory(self):
        self._lu = None


class MumpsWrapper(DirectSolver):
    """
    Wrapper around python-mumps.

    Unlike pypardiso, python-mumps allows separate analyze, factorize,
    and solve steps.

    This exists, therefore, mostly to provide a consistent interface with
    the PardisoWrapper.

    Note that we assume the shared references to A, b, x are maintained
    consistently!
    """

    def __init__(
        self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray, matrix_type=None
    ):
        import mumps

        if matrix_type is None:
            matrix_type = MatrixType.NONSYMMETRIC

        # Python-MUMPS currently only supports 0 (nonsymmetric) and 2 (symmetric, indefinite), not 1.
        symmetric = not (matrix_type == matrix_type.NONSYMMETRIC)

        self.A = A
        self.b = b
        self.x = x
        self.mumps = mumps.Context()
        self.mumps.set_matrix(A, symmetric=symmetric)

    def analyze(self):
        self.mumps.analyze()

    def factorize(self):
        self.mumps.factor(self.A, reuse_analysis=True)

    def solve(self):
        # mumps solves in-place; copy b into x so the result lands there
        self.x[:] = self.b[:]
        self.mumps.solve(b=self.x, overwrite_b=True)

    def free_memory(self):
        self.mumps.destroy()

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        if B.ndim != 2 or B.shape[0] != self.A.shape[0]:
            raise ValueError(
                "B must be 2D and the number of rows must match A. "
                f"B shape: {B.shape}, versus A shape: {self.A.shape}"
            )

        X = np.array(B, dtype=np.float64, order="F")
        self.mumps.solve(b=X, overwrite_b=True)
        return X

    @staticmethod
    def inverse_entries(mumps, pattern):
        """pattern: N x N sparse matrix; nonzero (i,j) requests A^{-1}[i,j]."""
        # See: https://gitlab.kwant-project.org/kwant/python-mumps/-/work_items/33
        from mumps import _mumps

        if not mumps.factored:
            raise RuntimeError("Run .factorize() first")

        b = sparse.csc_array(pattern)
        col_ptr = np.asfortranarray(b.indptr.astype(_mumps.int_dtype)) + 1
        row_ind = np.asfortranarray(b.indices.astype(_mumps.int_dtype)) + 1
        out = np.zeros(b.nnz, dtype=mumps.data.dtype, order="F")

        mumps.mumps_instance.set_sparse_rhs(col_ptr, row_ind, out)
        mumps.mumps_instance.icntl[20] = 1
        mumps.mumps_instance.icntl[30] = 1
        mumps.mumps_instance.job = 3
        try:
            mumps.call()
        finally:
            mumps.mumps_instance.icntl[30] = 0  # or every later solve breaks
        return out  # values in CSC order of `pattern`

    def inverse_diagonal(self, indices: np.ndarray):
        n = len(indices)
        N = self.A.shape[0]
        pattern = sparse.coo_array(
            (np.ones(n), (indices, indices)), shape=(N, N)
        ).tocsc()
        return self.inverse_entries(self.mumps, pattern)


def make_direct_solver(solver_backend: str, A, b, x, matrix_type=None):
    match solver_backend:
        case "pardiso":
            return PardisoWrapper(A, b, x, matrix_type)
        case "mumps":
            return MumpsWrapper(A, b, x, matrix_type)
        case "scipy":
            return ScipyWrapper(A, b, x)
        case _:
            raise ValueError(f"Unknown solver_backend: {solver_backend}")
