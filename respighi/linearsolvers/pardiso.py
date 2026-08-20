"""Explicit ctypes binding to Intel MKL PARDISO, real matrices only.

Phases are separated: analyze (11) / factorize (22) / solve (33) / release (-1).
A, b and x are borrowed by reference and must be mutated in place by the caller.

iparm is exposed through the Iparm class below, which is 1-BASED to match the
MKL documentation: self.iparm[11] here is iparm(11) in the docs. Every value is
set explicitly (iparm[1] = 1), so pardisoinit is never called and mtype cannot
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


# TODO: verify against current MKL version docs.
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


class Iparm:
    """1-based view of the PARDISO iparm array.

    Indices match the MKL documentation: iparm[11] here is iparm(11) in the
    docs. Index 0, negative indices and slices are rejected rather than
    silently aliased onto the wrong entry.

    Wraps the array MKL writes into, so output entries such as iparm(14) and
    iparm(18) read back through the same object after a call.
    """

    # 1-based. Might change across MKL versions: entries get un-reserved as
    # features are added, and a stale entry here is a false rejection.
    _RESERVED = frozenset(
        {3, 26, 29, 32, 33, 38, 40, 41, 42, 61, 62, 64}
        | set(range(44, 56))
        | set(range(57, 60))
    )

    __slots__ = ("_a",)

    def __init__(self):
        self._a = np.zeros(64, dtype=np.int32)

    @staticmethod
    def _ix(i) -> int:
        if not isinstance(i, (int, np.integer)):
            raise TypeError(
                f"iparm index must be a single int in 1..64, got {type(i).__name__}"
            )
        if not 1 <= i <= 64:
            raise IndexError(f"iparm index {i} out of range, valid indices are 1..64")
        return int(i) - 1

    def __getitem__(self, i) -> int:
        return int(self._a[self._ix(i)])

    def __setitem__(self, i, value) -> None:
        j = self._ix(i)
        if value != 0 and (j + 1) in self._RESERVED:
            raise ValueError(
                f"iparm({j + 1}) is reserved and must stay 0; writing {value} "
                "there is silently ignored by MKL. Check the index against the "
                "iparm table"
            )
        self._a[j] = value

    def __len__(self) -> int:
        return 64

    def __repr__(self) -> str:
        nz = ", ".join(f"iparm({j + 1})={v}" for j, v in enumerate(self._a) if v != 0)
        return f"<Iparm {nz or 'all zero'}>"

    @property
    def raw(self) -> np.ndarray:
        """The underlying 0-based int32 array. For the ctypes call only."""
        return self._a

    @property
    def ptr(self):
        """Pointer to the underlying buffer, for the pardiso() argument list."""
        return self._a.ctypes.data_as(_IP)


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
        self.iparm = Iparm()  # 1-based, see the class docstring

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
        # Zero-based indexing (iparm(35) = 1) lets the scipy arrays be passed
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
        # Indices below are 1-based and match the MKL documentation directly.
        p = self.iparm
        p[1] = 1  # do not fill defaults; every entry below is ours
        p[2] = 2  # METIS nested dissection (3 = threaded METIS)
        p[4] = 0  # no preconditioned CGS
        p[5] = 0  # no user permutation
        p[6] = 0  # write solution to x, leave b intact
        p[8] = 2  # max iterative refinement steps
        p[9] = 0  # backward error computed but not a stopping test
        p[10] = 13  # pivot perturbation 1e-13
        # Scaling and matching both require the numerical values of A to be
        # present at ANALYSIS time (phase 11), not just at factorization.
        p[11] = 1  # scaling
        p[12] = 0  # no transpose solve
        p[13] = 1  # matching
        p[18] = -1  # report nnz in factors
        p[19] = 0  # skip Mflop count (increases reordering time)
        p[21] = 1  # Bunch-Kaufman 1x1/2x2 pivoting; mtype -2/-4/6 only
        p[24] = 0  # classic factorization. The two-level algorithm (1) is
        # silently ignored unless scaling and matching are BOTH
        # off, so it is only reachable in the SPD branch below.
        p[25] = 0  # parallel forward/backward solve
        p[27] = int(check_matrix)  # matrix checker, debug only
        p[28] = 0  # float64
        p[35] = 1  # zero-based ia, ja and perm
        p[37] = 0  # CSR (>0 would be BSR with that block size)
        p[56] = 0  # no pivot callback / pardiso_getdiag
        p[60] = 0  # in-core

        if self.matrix_type is MatrixType.SYMMETRIC_INDEFINITE:
            p[10] = 9  # symmetric indefinite perturbation, 1e-9
            p[21] = 2  # 1x1 only
        elif self.matrix_type is MatrixType.SYMMETRIC_POSITIVE_DEFINITE:
            # mtype=2 uses Cholesky without pivoting: perturbation, scaling,
            # matching and Bunch-Kaufman all do not apply and must be zero.
            p[10] = 0
            p[11] = 0
            p[13] = 0
            p[21] = 0
            p[24] = 1  # reachable here: scaling and matching are off

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
            self.iparm.ptr,
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
        self.free_memory()
        return False

    @property
    def perturbed_pivots(self) -> int:
        """iparm(14). Not relevant for mtype=2, which does not perturb pivots.

        Climbing over iterations means the factorization is degrading.
        """
        return self.iparm[14]

    @property
    def factor_nnz(self) -> int:
        """iparm(18). Fill-in produced by the ordering."""
        return self.iparm[18]

    @property
    def peak_memory_kb(self) -> int:
        """Max of iparm(15) and iparm(16) + iparm(17)."""
        return max(self.iparm[15], self.iparm[16] + self.iparm[17])

    @property
    def inertia(self) -> tuple[int, int]:
        """iparm(22), iparm(23). Relevant for symmetric indefinite matrices only."""
        return self.iparm[22], self.iparm[23]

    @property
    def first_bad_pivot(self) -> int:
        """iparm(30). Only relevant for mtype=2 (and complex mtype=4).

        The EQUATION NUMBER at which a zero or negative pivot was found, not a
        count. Factorization stops there and returns error -4, so read this in
        the PardisoError handler rather than after a successful factorize().
        """
        return self.iparm[30]
