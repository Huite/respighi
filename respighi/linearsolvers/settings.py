"""Linear solver configuration, dispatching on settings type."""

import abc
import dataclasses

import numpy as np
from scipy import sparse

from respighi.linearsolvers.cg import PCGSolver
from respighi.linearsolvers.ilu0 import ILU0Preconditioner
from respighi.linearsolvers.mumps import MumpsWrapper
from respighi.linearsolvers.pardiso import PardisoWrapper
from respighi.linearsolvers.scipylu import ScipyLUWrapper
from respighi.linearsolvers.solvertypes import LinearSolver, MatrixType

# Matrix types a Krylov method with a symmetric preconditioner can handle.
_DEFINITE = frozenset({MatrixType.SYMMETRIC_POSITIVE_DEFINITE})


class LinearSettings(abc.ABC):
    """Configuration for a linear solver.

    Subclasses are dispatched on: each knows how to build its own solver, so
    callers never branch on a backend name. The capability flags let a caller
    check up front whether a backend can do what it needs, instead of catching
    ``NotImplementedError`` or inspecting the solver's concrete type.
    """

    @abc.abstractmethod
    def build(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        x: np.ndarray,
        matrix_type: MatrixType,
    ) -> LinearSolver:
        """Bind this configuration to a system and perform symbolic setup.

        Parameters
        ----------
        A
            System matrix. Retained by reference and expected to be mutated in
            place; rebinding it invalidates the solver.
        b
            Right hand side. Retained by reference.
        x
            Solution vector, written in place. Retained by reference.
        matrix_type
            Structural properties of ``A``, used to reject unusable
            combinations early and to select a backend-specific pivoting mode.
        """


@dataclasses.dataclass
class PCGSettings(LinearSettings):
    """Preconditioned conjugate gradients.

    Only valid for symmetric positive definite systems: CG assumes the inner
    product it minimises over is positive, and on an indefinite system it can
    break down rather than merely converge slowly. Saddle-point systems need a
    direct backend or a different Krylov method.

    Parameters
    ----------
    xclose
        Absolute tolerance on the infinity norm of the iterate change.
    rclose
        Absolute tolerance on the infinity norm of the residual. Both are
        absolute, so they need retuning if the units or the magnitude of the
        conductances change.
    maxiter
        Iteration budget.
    """

    xclose: float = 1e-5
    rclose: float = 1e-5
    maxiter: int = 100

    def __post_init__(self):
        if self.xclose <= 0.0 or self.rclose <= 0.0:
            raise ValueError("xclose and rclose must be positive")
        if self.maxiter < 1:
            raise ValueError(f"maxiter must be at least 1, got: {self.maxiter}")

    def build(self, A, b, x, _) -> LinearSolver:
        return PCGSolver(
            A,
            b,
            x,
            ILU0Preconditioner.from_csr_matrix(A),
            xclose=self.xclose,
            rclose=self.rclose,
            maxiter=self.maxiter,
        )


@dataclasses.dataclass
class PardisoSettings(LinearSettings):
    """Intel oneAPI PARDISO. Fast, but unavailable on Apple silicon."""

    def build(self, A, b, x, matrix_type) -> LinearSolver:
        solver = PardisoWrapper(A, b, x, matrix_type)
        solver.analyze()
        return solver


@dataclasses.dataclass
class MumpsSettings(LinearSettings):
    """MUMPS. The only backend here that can return entries of the inverse."""

    def build(self, A, b, x, matrix_type) -> LinearSolver:
        solver = MumpsWrapper(A, b, x, matrix_type)
        solver.analyze()
        return solver


@dataclasses.dataclass
class ScipyLUSettings(LinearSettings):
    """SciPy's SuperLU. Always available; the slowest of the three.

    Useful as a dependency-free fallback and for small problems in tests.
    """

    def build(self, A, b, x, matrix_type) -> LinearSolver:
        solver = ScipyLUWrapper(A, b, x)
        solver.analyze()
        return solver
