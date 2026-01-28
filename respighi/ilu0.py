from typing import NamedTuple, Tuple

import numba as nb
import numpy as np
from scipy import sparse

from respighi.constants import FloatArray, IntArray
from respighi.sparse import MatrixCSR, columns_and_values, nzrange, row_slice


@nb.njit(inline="always")
def lower_slice(ilu, row: int) -> slice:
    return slice(ilu.indptr[row], ilu.uptr[row])


@nb.njit(inline="always")
def upper_slice(ilu, row: int) -> slice:
    return slice(ilu.uptr[row], ilu.indptr[row + 1])


@nb.njit
def set_uptr(ilu: "ILU0Preconditioner") -> None:
    # i is row index, j is column index
    for i in range(ilu.n):
        for nzi in nzrange(ilu, i):
            j = ilu.indices[nzi]
            if j > i:
                ilu.uptr[i] = nzi
                break
    return


@nb.njit
def _update(ilu: "ILU0Preconditioner", A: MatrixCSR, delta: float, relax: float):
    """
    Perform zero fill-in incomplete lower-upper (ILU0) factorization
    using the values of A.
    """
    ilu.work[:] = 0.0
    visited = np.full(ilu.n, False)

    # i is row index, j is column index, v is value.
    for i in range(ilu.n):
        for j, v in columns_and_values(A, row_slice(A, i)):
            visited[j] = True
            ilu.work[j] += v

        rs = 0.0
        for j in ilu.indices[lower_slice(ilu, i)]:
            # Compute row multiplier
            multiplier = ilu.work[j] * ilu.diagonal[j]
            ilu.work[j] = multiplier
            # Perform linear combination
            for jj, vv in columns_and_values(ilu, upper_slice(ilu, j)):
                if visited[jj]:
                    ilu.work[jj] -= multiplier * vv
                else:
                    rs += multiplier * vv

        diag = ilu.work[i]
        multiplier = (1.0 + delta) * diag - (relax * rs)
        # Work around a zero-valued pivot and make sure the multiplier hasn't
        # changed sign.
        if multiplier == 0:
            multiplier = 1e-6
        elif np.sign(multiplier) != np.sign(diag):
            multiplier = np.sign(diag) * 1.0e-6
        ilu.diagonal[i] = 1.0 / multiplier

        # Reset work arrays, assign off-diagonal values
        visited[i] = False
        ilu.work[i] = 0.0
        for nzi in nzrange(ilu, i):
            j = ilu.indices[nzi]
            ilu.data[nzi] = ilu.work[j]
            ilu.work[j] = 0.0
            visited[j] = False

    return


@nb.njit
def _solve(ilu: "ILU0Preconditioner", r: np.ndarray):
    r"""
    LU \ r

    Stores the result in the pre-allocated work array.
    """
    ilu.work[:] = 0.0

    # forward
    for i in range(ilu.n):
        value = r[i]
        for j, v in columns_and_values(ilu, lower_slice(ilu, i)):
            value -= v * ilu.work[j]
        ilu.work[i] = value

    # backward
    for i in range(ilu.n - 1, -1, -1):
        value = ilu.work[i]
        for j, v in columns_and_values(ilu, upper_slice(ilu, i)):
            value -= v * ilu.work[j]
        ilu.work[i] = value * ilu.diagonal[i]

    return


class ILU0Preconditioner(NamedTuple):
    """
    Preconditioner based on zero fill-in lower-upper (ILU0) factorization.

    Data is stored in compressed sparse row (CSR) format. The diagonal
    values have been extracted for easier access. Upper and lower values
    are stored in CSR format. Next to the indptr array, which identifies
    the start and end of each row, the uptr array has been added to
    identify the start to the right of the diagonal. In case the row to the
    right of the diagonal is empty, it contains the end of the rows as
    indicated by the indptr array.

    Parameters
    ----------
    n: int
        Number of rows
    m: int
        Number of columns
    indptr: np.ndarray of int
        CSR format index pointer array of the matrix
    uptr: np.ndarray of int
        CSR format index pointer array of the upper elements (diagonal or higher)
    indices: np.ndarray of int
        CSR format index array of the matrix
    data: np.ndarray of float
        CSR format data array of the matrix
    diagonal: np.ndarray of float
        Diagonal values of LU factorization
    work: np.ndarray of float
        Work array. Used in factorization and solve.
    """

    n: int
    m: int
    indptr: IntArray
    uptr: IntArray
    indices: IntArray
    data: FloatArray
    diagonal: FloatArray
    work: FloatArray

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.n, self.m)

    @property
    def dtype(self):
        return self.data.dtype

    @staticmethod
    def from_csr_matrix(
        A: sparse.csr_matrix, delta: float = 0.0, relax: float = 0.0
    ) -> "ILU0Preconditioner":
        # Create a copy of the sparse matrix with the diagonals removed.
        n, m = A.shape
        coo = A.tocoo()
        i = coo.row
        j = coo.col
        offdiag = i != j
        ii = i[offdiag]
        indices = j[offdiag]
        indptr = sparse.csr_matrix((indices, (ii, indices)), shape=A.shape).indptr

        ilu = ILU0Preconditioner(
            n=n,
            m=m,
            indptr=indptr,
            uptr=indptr[1:].copy(),
            indices=indices,
            data=np.empty(indices.size),
            diagonal=np.empty(n),
            work=np.empty(n),
        )
        set_uptr(ilu)

        _update(ilu, MatrixCSR.from_csr_matrix(A), delta, relax)
        return ilu

    def update(self, A, delta=0.0, relax=0.0) -> None:
        _update(self, MatrixCSR.from_csr_matrix(A), delta, relax)
        return

    def matvec(self, r) -> FloatArray:
        _solve(self, r)
        return self.work

    def __repr__(self) -> str:
        return f"ILU0Preconditioner of type {self.dtype} and shape {self.shape}"


class DiagonalScaling(NamedTuple):
    diagA: FloatArray
    scale: FloatArray
    row_indices: IntArray
    col_indices: IntArray
    diag_indices: IntArray
    scale_rows: FloatArray
    scale_cols: FloatArray

    @classmethod
    def from_csr_matrix(cls, A):
        n, _ = A.shape
        row_indices = np.repeat(np.arange(n), np.diff(A.indptr))
        col_indices = A.indices
        diag_indices = np.where(row_indices == col_indices)[0]
        # Pre-allocate arrays
        diagA = np.empty(n)
        scale = np.empty(n)
        nnz = A.data.size
        scale_rows = np.empty(nnz)
        scale_cols = np.empty(nnz)
        return cls(
            diagA,
            scale,
            row_indices,
            col_indices,
            diag_indices,
            scale_rows,
            scale_cols,
        )

    def apply_scaling(self, A, rhs):
        """Apply diagonal scaling: D^{-1/2} A D^{-1/2} and D^{-1/2} rhs"""
        # Extract diagonal, compute scaling factors
        np.take(A.data, indices=self.diag_indices, out=self.diagA)
        np.sqrt(self.diagA, out=self.scale)
        np.reciprocal(self.scale, out=self.scale)
        # Scale matrix entries
        np.take(self.scale, self.row_indices, out=self.scale_rows)
        np.take(self.scale, self.col_indices, out=self.scale_cols)
        A.data *= self.scale_rows
        A.data *= self.scale_cols
        # Scale rhs
        rhs *= self.scale
        return
