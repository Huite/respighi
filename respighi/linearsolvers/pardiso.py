"""Explicit ctypes binding to Intel MKL PARDISO, real matrices only.

Phases are separated: analyze (11) / factorize (22) / solve (33) / release (-1).
A, b and x are borrowed by reference and must be mutated in place by the caller.

iparm indices below are 0-BASED numpy indices. The MKL docs number them
1-based, so iparm[10] here is iparm(11) in the documentation. Every value is
set explicitly (iparm[0] = 1), so pardisoinit is never called and mtype cannot
silently rewrite the array.

Mostly derived from the pypardiso wrapper.
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from ctypes.util import find_library

import numpy as np
from scipy import sparse

from respighi.linearsolvers.solvertypes import DirectSolver, MatrixType

_I = ctypes.c_int32  # LP64 interface. For ILP64 (pardiso_64) this is c_int64
# and the index arrays below must be int64 to match.
_IP = ctypes.POINTER(_I)
_DP = ctypes.POINTER(ctypes.c_double)


class Phase:
    ANALYZE = 11
    FACTORIZE = 22
    SOLVE = 33
    RELEASE_LU = 0
    RELEASE_ALL = -1


# Verify against your MKL version's docs before trusting these.
_ERRORS = {
    -1: "input inconsistent",
    -2: "not enough memory",
    -3: "reordering problem",
    -4: "zero pivot, numerical factorization or iterative refinement problem",
    -5: "unclassified internal error",
    -6: "reordering failed",
    -7: "diagonal matrix is singular",
    -8: "32-bit integer overflow",
    -9: "not enough memory for OOC",
    -10: "error opening OOC files",
    -11: "read/write error with OOC files",
    -12: "pardiso_64 called from the 32-bit library",
    -13: "interrupted by mkl_progress",
}


class PardisoError(RuntimeError):
    def __init__(self, code: int, phase: int):
        self.code = code
        self.phase = phase
        super().__init__(
            f"PARDISO phase {phase} failed with error {code}: "
            f"{_ERRORS.get(code, 'unknown error')}"
        )


def _load_mkl() -> ctypes.CDLL:
    """Locate libmkl_rt. Adapted from pypardiso's search."""
    path = (
        os.environ.get("MKL_RT_PATH")
        or find_library("mkl_rt")
        or find_library("mkl_rt.1")
    )
    if path is not None:
        return ctypes.CDLL(path)

    candidates = glob.glob(
        f"{sys.prefix}/[Ll]ib*/**/*mkl_rt*", recursive=True
    ) or glob.glob(f"{site.USER_BASE}/[Ll]ib*/**/*mkl_rt*", recursive=True)
    for candidate in sorted(candidates, key=len):
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise ImportError("mkl_rt not found; set MKL_RT_PATH.")


class PardisoWrapper(DirectSolver):
    def __init__(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        x: np.ndarray,
        matrix_type: None,
        msglvl: int = 0,
        check_matrix: bool = False,
    ):
        if matrix_type is None:
            matrix_type = MatrixType.NONSYMMETRIC

        self._validate(A, b, x, matrix_type)

        self.A, self.b, self.x = A, b, x
        self.matrix_type = matrix_type
        self._analyzed = False
        self._factorized = False
        self._released = False

        self._lib = _load_mkl()
        self._pardiso = self._lib.pardiso
        self._pardiso.restype = None
        self._pt_ctype = (
            ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32
        )
        self._pardiso.argtypes = [
            ctypes.POINTER(self._pt_ctype),  # pt
            _IP,
            _IP,
            _IP,
            _IP,
            _IP,  # maxfct, mnum, mtype, phase, n
            _DP,
            _IP,
            _IP,
            _IP,
            _IP,
            _IP,  # a, ia, ja, perm, nrhs, iparm
            _IP,  # msglvl
            _DP,
            _DP,  # b, x
            _IP,  # error
        ]

        # Opaque handle. Never touch after the first call.
        self._pt = np.zeros(
            64, dtype=np.int64 if self._pt_ctype is ctypes.c_int64 else np.int32
        )
        self._perm = np.zeros(A.shape[0], dtype=np.int32)
        self.iparm = np.zeros(64, dtype=np.int32)

        # Persistent scalars: mutate .value rather than rebuilding per call, so
        # mtype and n cannot drift between phases sharing the same pt handle.
        self._maxfct = _I(1)
        self._mnum = _I(1)
        self._mtype = _I(matrix_type.pardiso_mtype)
        self._phase = _I(0)
        self._n = _I(A.shape[0])
        self._nrhs = _I(1 if b.ndim == 1 else b.shape[1])
        self._msglvl = _I(int(msglvl))
        self._error = _I(0)

        self._set_iparm(check_matrix=check_matrix)

        # Zero-based indexing (iparm[34] = 1) lets the scipy arrays be passed
        # straight through with no +1 copy. Hold the references: these arrays
        # must outlive every call that uses their pointers.
        self._ia = np.ascontiguousarray(A.indptr, dtype=np.int32)
        self._ja = np.ascontiguousarray(A.indices, dtype=np.int32)

    @staticmethod
    def _validate(A, b, x, matrix_type):
        if not sparse.issparse(A) or A.format != "csr":
            raise TypeError(
                f"A must be CSR, got {type(A)} / {getattr(A, 'format', None)}"
            )
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got {A.shape}")
        if A.dtype != np.float64:
            raise TypeError(f"A must be float64, got {A.dtype}")
        if not A.has_sorted_indices:
            raise ValueError("A has unsorted indices.")
        if not np.diff(A.indptr).all():
            raise ValueError("A has empty rows and is structurally singular")
        if A.indptr[-1] > np.iinfo(np.int32).max:
            raise ValueError("nnz exceeds int32; the ILP64 interface is required")

        if matrix_type.triangle_only:
            # PARDISO reads only the upper triangle for symmetric mtypes and
            # will not warn if given the full matrix.
            if A.nnz != sparse.triu(A).nnz:
                raise ValueError(
                    f"{matrix_type.name} requires upper-triangle-only CSR; "
                    f"A has {A.nnz} nnz, upper triangle has {sparse.triu(A).nnz}"
                )

        for name, arr in (("b", b), ("x", x)):
            if arr.dtype != np.float64:
                raise TypeError(f"{name} must be float64, got {arr.dtype}")
            if arr.shape[0] != A.shape[0]:
                raise ValueError(f"{name}.shape[0] != A.shape[0]")
            if arr.ndim > 1 and not arr.flags.f_contiguous:
                raise ValueError(f"{name} must be Fortran-ordered for multiple RHS")

    def _set_iparm(self, check_matrix: bool):
        p = self.iparm
        p[0] = 1  # iparm(1)  do not fill defaults; every entry below is ours
        p[1] = 2  # iparm(2)  METIS nested dissection (3 = threaded METIS)
        p[3] = 0  # iparm(4)  no preconditioned CGS
        p[4] = 0  # iparm(5)  no user permutation
        p[5] = 0  # iparm(6)  write solution to x, leave b intact
        p[7] = 2  # iparm(8)  max iterative refinement steps
        p[8] = 0  # iparm(9)  backward error computed but not a stopping test
        p[9] = 13  # iparm(10) pivot perturbation 1e-13
        # Scaling and matching both require the numerical values of A to be
        # present at ANALYSIS time (phase 11), not just at factorization.
        p[10] = 1  # iparm(11) scaling
        p[11] = 0  # iparm(12) no transpose solve
        p[12] = 1  # iparm(13) matching
        p[17] = -1  # iparm(18) report nnz in factors
        p[18] = 0  # iparm(19) skip Mflop count (increases reordering time)
        p[20] = 1  # iparm(21) Bunch-Kaufman 1x1/2x2 pivoting; mtype -2/-4/6 only
        p[23] = 0  # iparm(24) classic factorization. The two-level algorithm (1)
        # is silently ignored unless scaling and matching are BOTH
        # off, so it is only reachable in the SPD branch below.
        p[24] = 0  # iparm(25) parallel forward/backward solve
        p[26] = int(check_matrix)  # iparm(27) matrix checker, debug only
        p[27] = 0  # iparm(28) float64
        p[34] = 1  # iparm(35) zero-based ia, ja and perm
        p[36] = 0  # iparm(37) CSR (>0 would be BSR with that block size)
        p[55] = 0  # iparm(56) no pivot callback / pardiso_getdiag
        p[59] = 0  # iparm(60) in-core

        if self.matrix_type is MatrixType.SYMMETRIC_INDEFINITE:
            p[9] = 9  # symmetric indefinite perturbation, 1e-9
            p[20] = 2  # 1x1 only
        elif self.matrix_type is MatrixType.SYMMETRIC_POSITIVE_DEFINITE:
            # mtype=2 uses Cholesky without pivoting: perturbation, scaling,
            # matching and Bunch-Kaufman all do not apply and must be zero.
            p[9] = 0
            p[10] = 0
            p[12] = 0
            p[20] = 0
            p[23] = 1  # reachable here: scaling and matching are off

    def _call(self, phase: int):
        if self._released:
            raise RuntimeError("solver has been released")
        self._phase.value = phase
        self._error.value = 0
        self._pardiso(
            self._pt.ctypes.data_as(ctypes.POINTER(self._pt_ctype)),
            ctypes.byref(self._maxfct),
            ctypes.byref(self._mnum),
            ctypes.byref(self._mtype),
            ctypes.byref(self._phase),
            ctypes.byref(self._n),
            self.A.data.ctypes.data_as(_DP),
            self._ia.ctypes.data_as(_IP),
            self._ja.ctypes.data_as(_IP),
            self._perm.ctypes.data_as(_IP),
            ctypes.byref(self._nrhs),
            self.iparm.ctypes.data_as(_IP),
            ctypes.byref(self._msglvl),
            self.b.ctypes.data_as(_DP),
            self.x.ctypes.data_as(_DP),
            ctypes.byref(self._error),
        )
        if self._error.value != 0:
            raise PardisoError(self._error.value, phase)

    def analyze(self):
        """Symbolic factorization. Depends on structure only; run once."""
        self._call(Phase.ANALYZE)
        self._analyzed = True
        self._factorized = False
        return self

    def factorize(self):
        """Numerical factorization. Rerun whenever A.data changes."""
        if not self._analyzed:
            self.analyze()
        self._call(Phase.FACTORIZE)
        self._factorized = True
        return self

    def solve(self):
        """Forward/backward substitution into x, using the current b."""
        if not self._factorized:
            raise RuntimeError("call factorize() before solve()")
        self._call(Phase.SOLVE)
        return self.x

    def solve_multi(self, B: np.ndarray) -> np.ndarray:
        """
        Solve K X = B for multiple RHS columns simultaneously.
        B: shape (N, k), Fortran-contiguous for PARDISO's column-major layout.
        Returns X: shape (N, k).
        """
        if B.ndim != 2 or B.shape[0] != self.A.shape[0]:
            raise ValueError(
                "B must be 2D and the number of rows must match A. "
                f"B shape: {B.shape}, versus A shape: {self.A.shape}"
            )

        B = np.asfortranarray(B, dtype=np.float64)
        X = np.zeros_like(B, order="F")

        nrhs_original = self._nrhs
        b_original = self.b
        x_original = self.x

        try:
            self._nrhs = _I(B.shape[1])
            self.b = B
            self.x = X
            self.solve()
        finally:
            # Move original values back
            self._nrhs = nrhs_original
            self.b = b_original
            self.x = x_original
        return X

    def free_memory(self):
        if not self._released and self._analyzed:
            self._call(Phase.RELEASE_ALL)
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    @property
    def perturbed_pivots(self) -> int:
        """iparm(14). Not reported for mtype=2, which does not perturb pivots.

        Climbing over iterations means the factorization is degrading.
        """
        if self.matrix_type is MatrixType.SYMMETRIC_POSITIVE_DEFINITE:
            raise AttributeError("mtype=2 does not perturb pivots")
        return int(self.iparm[13])

    @property
    def factor_nnz(self) -> int:
        """iparm(18). Fill-in produced by the ordering."""
        return int(self.iparm[17])

    @property
    def peak_memory_kb(self) -> int:
        """Max of iparm(15) and iparm(16) + iparm(17)."""
        return max(int(self.iparm[14]), int(self.iparm[15]) + int(self.iparm[16]))

    @property
    def inertia(self) -> tuple[int, int]:
        """iparm(22), iparm(23). Reported for symmetric indefinite matrices only."""
        if self.matrix_type is not MatrixType.SYMMETRIC_INDEFINITE:
            raise AttributeError(
                "inertia is only reported for symmetric indefinite matrices"
            )
        return int(self.iparm[21]), int(self.iparm[22])

    @property
    def first_bad_pivot(self) -> int:
        """iparm(30). Only populated for mtype=2 (and complex mtype=4).

        The EQUATION NUMBER at which a zero or negative pivot was found, not a
        count. Factorization stops there and returns error -4, so read this in
        the PardisoError handler rather than after a successful factorize().
        """
        if self.matrix_type is not MatrixType.SYMMETRIC_POSITIVE_DEFINITE:
            raise AttributeError("iparm(30) is only populated for mtype=2")
        return int(self.iparm[29])
