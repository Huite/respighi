"""Shared nonlinear iteration mixin for models that formulate their own linear system."""

import abc
import dataclasses
import warnings
from typing import NamedTuple

import numpy as np

from respighi.constants import FloatArray
from respighi.relaxation import Relaxation, ScalarRelaxation


@dataclasses.dataclass
class NonlinearSettings:
    """Tolerances and iteration budget for :class:`NonlinearIteration`.

    Parameters
    ----------
    relaxation
        Damping strategy. Its estimation block is primary-sized, so an
        :class:`AitkenRelaxation` should be constructed with the primary
        length, not the state length. Defaults to undamped.
    maxiter
        Iteration budget. Exhausting it warns and returns a non-converged result.
    xclose
        Tolerance on the infinity norm of the raw update, in primary space.
    rclose
        Tolerance on the residual, scaled per entry by the system diagonal so
        the criterion is dimensionally consistent across cells with very
        different conductances. Distinct from ``xclose``: the two have
        different units and there is no reason for them to share a value.
    """

    relaxation: Relaxation = dataclasses.field(default_factory=ScalarRelaxation)
    maxiter: int = 30
    xclose: float = 1e-4
    rclose: float = 1e-4

    def __post_init__(self):
        if self.maxiter < 1:
            raise ValueError(f"maxiter must be at least 1, got: {self.maxiter}")
        if self.xclose <= 0.0 or self.rclose <= 0.0:
            raise ValueError("xclose and rclose must be positive")


class SolverResult(NamedTuple):
    """Outcome of a single nonlinear solve.

    Carries the final norms as well as the flag, so a caller can log how close
    a failed solve came rather than only that it failed.
    """

    converged: bool
    iterations: int
    max_update: float
    max_residual: float


class NonlinearIteration(abc.ABC):
    """Damped Newton/Picard iteration, as a mixin.

    Each iteration evaluates the residual at the current state, tests for
    convergence, then solves the linearised system and applies a relaxed
    update. Testing before the first solve means an already-converged state
    costs zero linear solves.

    Convergence requires both the infinity norm of the raw update to fall below
    ``xclose`` and every scaled residual to fall below one. The update is
    measured undamped: with a small relaxation factor the applied step can be
    small while the underlying Newton step is not, so testing the damped step
    reports convergence that has not happened.

    Implementors provide :attr:`state`, :attr:`diagonal`, :meth:`primary`,
    :meth:`formulate`, :meth:`linear_solve` and :meth:`residual`, and a
    ``nonlinear_settings`` attribute. Work arrays are allocated lazily.
    """

    nonlinear_settings: NonlinearSettings

    @property
    @abc.abstractmethod
    def state(self) -> FloatArray:
        """Live solution vector, written in place by :meth:`linear_solve`."""

    @property
    @abc.abstractmethod
    def diagonal(self) -> FloatArray:
        """Diagonal used to scale residuals. Primary-sized."""

    @abc.abstractmethod
    def primary(self, vector) -> FloatArray:
        """Select the convergence subspace of a state vector."""

    @abc.abstractmethod
    def formulate(self, dt: float | None) -> None:
        """Assemble the system at the current state. ``dt`` of None is steady state."""

    @abc.abstractmethod
    def linear_solve(self) -> tuple[bool, int]:
        """Solve the assembled system, writing into :attr:`state`.

        Returns a convergence flag and an iteration count. Direct solvers can
        report ``(True, 1)``. Any factorization belongs in :meth:`formulate`,
        which is called exactly once per assembled system; factorizing here
        would repeat work on the first iteration.
        """

    @abc.abstractmethod
    def residual(self) -> FloatArray:
        """Residual at the current state and formulation. Primary-sized.

        May return a shared buffer; the caller does not retain it.
        """

    def _ensure_buffers(self) -> None:
        """Allocate work arrays on first use, or if the problem size changed."""
        n = self.state.size
        n_primary = self.diagonal.size
        update = getattr(self, "_nl_update", None)
        if update is None or update.size != n:
            self._nl_previous = np.empty(n, dtype=float)
            self._nl_update = np.empty(n, dtype=float)
        scale = getattr(self, "_nl_scale", None)
        if scale is None or scale.size != n_primary:
            self._nl_scale = np.empty(n_primary, dtype=float)

    def _set_residual_scale(self) -> None:
        """Freeze the per-entry residual scaling from the initial formulation.

        Held fixed for the whole solve so the target does not move as
        boundaries activate. The floor keeps entries with a zero diagonal
        (isolated, no boundary) from producing infinite scaled residuals.
        """
        scale = self._nl_scale
        np.abs(self.diagonal, out=scale)
        floor = 1e-12 * max(float(scale.max()), 1.0)
        np.maximum(scale, floor, out=scale)
        scale *= self.nonlinear_settings.rclose

    def _max_scaled_residual(self) -> float:
        """Largest residual relative to its tolerance. Below 1.0 passes."""
        return float(np.max(np.abs(self.residual()) / self._nl_scale))

    def nonlinear_solve(self, dt: float | None = None) -> SolverResult:
        """Iterate to a converged state, updating `state` in place.

        Parameters
        ----------
        dt
            Time step size, or None for steady state.
        """
        settings = self.nonlinear_settings
        self._ensure_buffers()
        settings.relaxation.reset()

        state = self.state
        previous = self._nl_previous
        update = self._nl_update

        self.formulate(dt=dt)
        self._set_residual_scale()

        max_update = np.inf
        max_residual = np.inf
        for i in range(settings.maxiter):
            max_residual = self._max_scaled_residual()
            if max_residual < 1.0 and max_update < settings.xclose:
                return SolverResult(True, i, max_update, max_residual)

            np.copyto(dst=previous, src=state)
            linear_converged, linear_iterations = self.linear_solve()
            if not linear_converged:
                warnings.warn(
                    f"Linear solver did not converge after {linear_iterations} "
                    f"iterations, in nonlinear iteration {i + 1}."
                )

            np.subtract(state, previous, out=update)
            primary_update = self.primary(update)
            # Both reads must precede `apply`, which scales `update` in place;
            # `primary_update` may be a view onto it.
            max_update = float(np.linalg.norm(primary_update, ord=np.inf))
            settings.relaxation.set_alpha(primary_update)
            settings.relaxation.apply(previous, state, update)

            self.formulate(dt=dt)

        warnings.warn(
            f"Nonlinear solver did not converge after {settings.maxiter} iterations. "
            f"Final update: {max_update:.2e}; "
            f"maximum scaled residual: {max_residual:.2e}"
        )
        return SolverResult(False, settings.maxiter, max_update, max_residual)
