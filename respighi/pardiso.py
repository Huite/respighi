import ctypes

from scipy import sparse
import numpy as np
import pypardiso

from respighi.constants import FloatArray


class PardisoWrapper:
    """
    Wrapper around the PyPardisoSolver for more fine-grained control.

    This does not re-allocate x, ia, ja every call and separates
    analyze, formulate, and solve steps more cleanly.

    Note that we assume the references to A, b, x are maintained
    consistently!
    """

    def __init__(self, A: sparse.csr_matrix, b: FloatArray, x: FloatArray):
        self.A = A
        self.b = b
        self.x = x
        self.pardiso = pypardiso.PyPardisoSolver()
        self.args = self.pardiso_args(self.pardiso, self.A, self.b, self.x)

    @staticmethod
    def pardiso_args(pardiso, A, b, x):
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
