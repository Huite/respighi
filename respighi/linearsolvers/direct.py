import abc
import ctypes

import numpy as np
from scipy import sparse

from respighi.constants import FloatArray


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
    def solve_multi(self, B: np.ndarray) -> np.ndarray: ...

    @abc.abstractmethod
    def free_memory(self): ...


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


class PardisoWrapper(DirectSolver):
    """
    Wrapper around the PyPardisoSolver for more fine-grained control.

    This does not re-allocate x, ia, ja every call and separates
    analyze, formulate, and solve steps more cleanly.

    Note that we assume the shared references to A, b, x are maintained
    consistently!
    """

    def __init__(self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray):
        import pypardiso

        self.A = A
        self.b = b
        self.x = x
        self.pardiso = pypardiso.PyPardisoSolver()
        self.args = self.pardiso_args(self.pardiso, self.A, self.b, self.x)

    @staticmethod
    def pardiso_args(pardiso, A, b, x):
        """
        Create a (mutable!) cache (i.e. a list) of arguments.

        When sparsity structure does not change, most data can be re-used for
        repeated solves (e.g. Picard iteration).

        Code here is taken almost verbatim from the
        PyPardisSolver._call_pardiso method, where x, ia, ja, and all ctypes
        objects are created on each call instead.
        """
        pardiso_error = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_float64_p = ctypes.POINTER(ctypes.c_double)

        # 1-based indexing
        ia = A.indptr.astype(np.int32) + 1
        ja = A.indices.astype(np.int32) + 1

        args = [
            pardiso.pt.ctypes.data_as(ctypes.POINTER(pardiso._pt_type[0])),  # pt
            ctypes.byref(ctypes.c_int32(1)),  # maxfct
            ctypes.byref(ctypes.c_int32(1)),  # mnum
            ctypes.byref(
                ctypes.c_int32(pardiso.mtype)
            ),  # mtype -> 11 for real-nonsymetric
            ctypes.byref(ctypes.c_int32(pardiso.phase)),  # phase -> 13
            ctypes.byref(
                ctypes.c_int32(A.shape[0])
            ),  # N -> number of equations/size of matrix
            A.data.ctypes.data_as(c_float64_p),  # A -> non-zero entries in matrix
            ia.ctypes.data_as(c_int32_p),  # ia -> csr-indptr
            ja.ctypes.data_as(c_int32_p),  # ja -> csr-indices
            pardiso.perm.ctypes.data_as(c_int32_p),  # perm -> empty
            ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
            pardiso.iparm.ctypes.data_as(c_int32_p),  # iparm-array
            ctypes.byref(
                ctypes.c_int32(pardiso.msglvl)
            ),  # msg-level -> 1: statistical info is printed
            b.ctypes.data_as(c_float64_p),  # b -> right-hand side vector/matrix
            x.ctypes.data_as(c_float64_p),  # x -> output
            ctypes.byref(pardiso_error),  # pardiso error
        ]
        return args

    def call_pardiso(self, args: list, phase: int):
        # Mutate the phase and include a fresh erro status, then call pardiso.
        # A and b are assumed to be shared references, shared here and by
        # whatever is updating coefficients.
        pardiso_error = ctypes.c_int32(0)
        args[4] = ctypes.byref(ctypes.c_int32(phase))
        args[-1] = ctypes.byref(pardiso_error)
        self.pardiso._mkl_pardiso(*args)
        if pardiso_error.value != 0:
            raise RuntimeError(pardiso_error.value)

    def analyze(self):
        """Phase 11: Symbolic factorization"""
        self.call_pardiso(self.args, 11)

    def factorize(self):
        """Phase 22: Numerical factorization"""
        self.call_pardiso(self.args, 22)

    def solve(self):
        """Phase 33: Solve"""
        self.call_pardiso(self.args, 33)

    def free_memory(self):
        self.pardiso.free_memory()

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        """
        Solve K X = B for multiple RHS columns simultaneously.
        B: shape (N, k), Fortran-contiguous for PARDISO's column-major layout.
        Returns X: shape (N, k).
        """
        assert B.ndim == 2 and B.shape[0] == self.A.shape[0]
        B = np.asfortranarray(B, dtype=np.float64)
        X = np.zeros_like(B, order="F")

        c_float64_p = ctypes.POINTER(ctypes.c_double)

        # Temporarily patch nrhs, b pointer, x pointer in args
        args = self.args.copy()
        args[10] = ctypes.byref(ctypes.c_int32(B.shape[1]))  # nrhs
        args[13] = B.ctypes.data_as(c_float64_p)  # b
        args[14] = X.ctypes.data_as(c_float64_p)  # x

        self.call_pardiso(args, 33)
        return X


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

    def __init__(self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray):
        import mumps

        self.A = A
        self.b = b
        self.x = x
        self.mumps = mumps.Context()

    def analyze(self):
        self.mumps.analyze(self.A)

    def factorize(self):
        self.mumps.factor(self.A)

    def solve(self):
        # mumps solves in-place; copy b into x so the result lands there
        self.x[:] = self.b[:]
        self.mumps.solve(b=self.x, overwrite_b=True)

    def free_memory(self):
        self.mumps.destroy()

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        assert B.ndim == 2 and B.shape[0] == self.A.shape[0]
        X = np.array(B, dtype=np.float64, order="F")
        self.mumps.solve(b=X, overwrite_b=True)
        return X


def make_direct_solver(solver_backend: str, A, b, x):
    match solver_backend:
        case "pardiso":
            return PardisoWrapper(A, b, x)
        case "mumps":
            return MumpsWrapper(A, b, x)
        case "scipy":
            return ScipyWrapper(A, b, x)
        case _:
            raise ValueError(f"Unknown solver_backend: {solver_backend}")
