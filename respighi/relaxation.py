"""Relaxation strategies for damping Newton/Picard updates.

A relaxation scheme takes the raw increment produced by a nonlinear solver
and scales it before it is applied to the current iterate::

    x_{k+1} = x_k + alpha_k * d_k

where ``d_k`` is the raw update (the Newton step, or ``g(x_k) - x_k`` for a
Picard/fixed-point iteration).
"""

import abc

import numpy as np

from respighi.constants import FloatArray


class Relaxation(abc.ABC):
    """Interface for relaxation strategies.

    Implementations are stateful across the iterations of a single nonlinear
    solve and must be reset before being reused for an independent solve
    (e.g. the next time step or the next continuation parameter).

    Subclasses set ``self.alpha``; applying it is handled by :meth:`apply`.
    """

    alpha: float

    @abc.abstractmethod
    def reset(self) -> None:
        """Discard any history accumulated during the previous solve.

        Must be called before starting an independent nonlinear solve.
        Failing to do so lets the relaxation factor and update history from an
        unrelated iteration leak into the new one.
        """

    @abc.abstractmethod
    def set_alpha(self, update: FloatArray) -> None:
        """Choose the factor for this iteration and advance any history.

        Parameters
        ----------
        update
            Raw, undamped update for the block the factor is estimated from.
            Not modified. Must be the same length on every call within a solve.

        Notes
        -----
        Call this before :meth:`apply`, which scales its argument in place. If
        the estimation block is a view into the full update vector, calling
        them in the other order feeds an already-damped update to the estimate.
        """

    def apply(
        self, state: FloatArray, newstate: FloatArray, update: FloatArray
    ) -> None:
        """Write ``state + alpha * update`` into ``newstate``.

        Parameters
        ----------
        state
            Current iterate. Not modified.
        newstate
            Output buffer. Overwritten. May not alias ``update``.
        update
            Raw update. **Scaled in place**, so on return it holds the damped
            step actually applied, not the step that was passed in.
            Take the norm before calling.
        """
        update *= self.alpha
        np.add(state, update, out=newstate)


class ScalarRelaxation(Relaxation):
    """Constant under-relaxation with a fixed factor.

    The cheapest and simplest scheme, useful for testing.

    Parameters
    ----------
    alpha
        Relaxation factor in ``(0, 1]``. A value of ``1.0`` applies the raw
        update, i.e. no damping.

    Raises
    ------
    ValueError
        If ``alpha`` lies outside ``(0, 1]``.
    """

    def __init__(self, alpha: float = 1.0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"Relaxation parameter must be in (0,1], got: {alpha}")
        self.alpha = alpha

    def reset(self) -> None:
        """No-op; this scheme carries no state between iterations."""
        return

    def set_alpha(self, update: FloatArray) -> None:
        """No-op; the factor is fixed at construction."""
        return


class AitkenRelaxation(Relaxation):
    """Aitken (Irons-Tuck) dynamic relaxation.

    Adapts the relaxation factor at each iteration from the change in the raw
    update, using the vector form of Aitken's delta-squared process::

        alpha_k = -alpha_{k-1} * <d_{k-1}, d_k - d_{k-1}> / <d_k - d_{k-1}, d_k - d_{k-1}>

    The estimate is clipped to ``[alpha_min, alpha_max]`` to bound the damage
    from a bad step. Raising ``alpha_max`` above ``1.0`` permits
    over-relaxation, the default ceiling of ``1.0`` restricts it to pure damping.

    Parameters
    ----------
    n: int
        Length of the vector passed to :meth:`set_alpha` — the estimation
        block, which for a blocked system is smaller than the full state.
    alpha_min: float
        Lower clip on the factor. Must be positive, so a bad estimate stalls
        the iteration rather than reversing it.
    alpha_max: float
        Upper clip on the factor, and the value used on the first iteration.

    Raises
    ------
    ValueError
        If the bounds do not satisfy ``0 < alpha_min <= alpha_max``.

    References
    ----------
    Irons, B. M. and Tuck, R. C. (1969), "A version of the Aitken accelerator
    for computer iteration", Int. J. Numer. Meth. Engng, 1: 275-277.
    """

    def __init__(self, n: int, alpha_min: float = 0.1, alpha_max: float = 1.0):
        if not 0.0 < alpha_min <= alpha_max:
            raise ValueError(
                f"Need 0 < alpha_min <= alpha_max, got: {alpha_min}, {alpha_max}"
            )
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha = alpha_max
        self.previous_update = np.zeros(n, dtype=float)
        self.delta = np.zeros(n, dtype=float)
        self._has_previous = False

    def reset(self) -> None:
        """Drop the stored update and restore ``alpha`` to ``alpha_max``."""
        self.alpha = self.alpha_max
        self._has_previous = False

    def _estimate(self, update: FloatArray) -> None:
        """Recompute ``self.alpha`` from the current and previous raw updates.

        Leaves ``self.alpha`` untouched when the two updates are too close to
        distinguish, since the quotient is then dominated by rounding error.
        Holding the previous factor is safer than resetting to one: the
        recursion is defined in terms of ``alpha_{k-1}``, so an unmotivated
        reset discards damping the iteration may still need.

        The guard compares against ``max(d2, dd)`` rather than ``d2`` alone so
        that it still triggers when the previous update was (near) zero, which
        happens as the iteration converges.
        """
        np.subtract(update, self.previous_update, out=self.delta)
        dd = np.dot(self.delta, self.delta)
        d2 = np.dot(self.previous_update, self.previous_update)
        if dd <= 1e-12 * max(d2, dd):
            return
        numerator = np.dot(self.previous_update, self.delta)
        alpha_new = -self.alpha * numerator / dd
        if np.isfinite(alpha_new):
            self.alpha = float(np.clip(alpha_new, self.alpha_min, self.alpha_max))

    def set_alpha(self, update: FloatArray) -> None:
        """Update the factor, then store the raw update for the next estimate.

        The update is stored undamped as the Aitken formula is defined on raw
        updates: feeding it damped ones distorts the factor.
        """
        if self._has_previous:
            self._estimate(update)
        self.previous_update[:] = update
        self._has_previous = True
