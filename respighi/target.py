import abc

import numpy as np
import pandas as pd
import xarray as xr
import xugrid as xu
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.floating]

# Placeholder coordinate for a static (non time-varying) source. Its value is
# never used for lookup: bind_time short-circuits a length-1 time axis, so no
# comparison against this timestamp is ever made.
STATIC_TIME = pd.Timestamp.min


def _promote(head, dims: tuple[str, ...]) -> xr.DataArray:
    """
    Promote an observation source to a DataArray with a leading time axis.

    A static input becomes a source with a single timestamp, so downstream code
    never has to branch on whether a target varies in time.

    Parameters
    ----------
    head:
        Array-like or DataArray of observation values.
    dims:
        Expected non-time dimensions, used both to label a bare array and to
        reject a DataArray carrying an unrecognised dimension.

    Raises
    ------
    ValueError
        If a labelled input has dimensions other than ``time`` and ``dims``.
        This catches the case where a time axis is named something else
        (``date``, ``period``) and would otherwise be silently flattened into
        the observation axis.
    """
    if not isinstance(head, xr.DataArray):
        head = xr.DataArray(np.asarray(head, dtype=float), dims=dims)

    unexpected = set(head.dims) - set(dims) - {"time"}
    if unexpected:
        raise ValueError(
            f"Unexpected dimension(s) {sorted(unexpected)}; expected "
            f"{list(dims)} with an optional 'time' dimension. A time axis "
            "under another name must be renamed to 'time'."
        )

    if "time" in head.dims:
        return head.transpose("time", ...)
    return head.expand_dims(time=[STATIC_TIME], axis=0)


def _multiplier_helper(
    a,
    nhead: int,
    name: str,
) -> FloatArray:
    """
    Convert a scalar or array-like multiplier to an observation-sized array.

    ``None`` gives an array of ones.
    """
    if a is None:
        return np.ones(nhead, dtype=float)

    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return np.full(nhead, a.item(), dtype=float)
    a = a.ravel()
    if a.size != nhead:
        raise ValueError(
            f"{name} size {a.size} does not match number of observations {nhead}"
        )
    return a


class FittingTarget(abc.ABC):
    """
    Least-squares fitting target.

    The solver-facing quantities ``P`` and ``d`` are scaled such that the
    objective is::

        ||P x - d||²

    For observation i, the row scaling is

        scale_i = sqrt(weight_i) / sigma_i

    so that the corresponding contribution to the objective is

        weight_i * ((prediction_i - head_i) / sigma_i)²

    Time-varying targets
    --------------------
    ``head`` is always stored as a DataArray with a leading ``time`` axis; a
    static source is simply one with a single timestamp. ``bind_time`` maps a
    simulation time axis onto the source's own timestamps, and ``advance``
    refreshes ``d`` when the mapped position changes. The observation operator
    ``P`` is fixed for the lifetime of the target: the set of observations may
    not vary between time steps.

    Parameters stored on each target
    --------------------------------
    P : scipy.sparse.csr_matrix
        Scaled observation operator. Fixed after construction.
    d : ndarray
        Scaled target values for the current time step.
    head : xr.DataArray
        Source values in physical head units, with a leading time axis. For
        the values corresponding to the current step, see ``.observed``.
    sigma : ndarray
        Observation standard deviations in physical head units.
        A value of 1.0 corresponds to unscaled least squares.
    weights : ndarray
        Dimensionless precision/importance weights, strictly positive.
    scale : ndarray
        Row scaling sqrt(weights) / sigma.
    """

    P: sparse.csr_matrix
    d: FloatArray
    head: xr.DataArray
    sigma: FloatArray
    weights: FloatArray
    scale: FloatArray

    # Flat positions of the observations within a raveled source slice. None
    # means every element of the slice is an observation. Set by subclasses
    # whose P covers only part of the source grid.
    _row_index: NDArray[np.integer] | None = None

    @property
    def n_targets(self) -> int:
        return self.P.shape[0]

    @property
    def observed(self) -> FloatArray:
        """
        Current-step values in physical head units, one per observation.

        Recovered exactly from ``d``, so it cannot go stale. Requires strictly
        positive weights, which ``_set_target`` enforces.
        """
        return self.d / self.scale

    def _extract(self, values: FloatArray) -> FloatArray:
        """Observation values from a single (raw) time slice of the source."""
        flat = values.reshape(-1)
        if self._row_index is None:
            return flat
        return flat[self._row_index]

    def _set_target(
        self,
        P: sparse.spmatrix,
        head: xr.DataArray,
        sigma=None,
        weights=None,
    ) -> None:
        """
        Store physical observations and construct the scaled least-squares target.

        The physical observation model is

            P0 x ~= head

        with objective

            sum_i weights_i * ((P0 x - head)_i / sigma_i)².

        Defining

            scale_i = sqrt(weights_i) / sigma_i

        gives the equivalent solver-facing problem

            ||P x - d||²

        with

            P = diag(scale) @ P0
            d = scale * head.
        """
        P = P.tocsr()
        nhead = P.shape[0]

        # Cross-check the operator against the source before anything else
        # depends on their agreement; a mismatch would otherwise surface as an
        # opaque shape error inside np.multiply.
        probe = self._extract(head.isel(time=0).to_numpy())
        if probe.size != nhead:
            raise ValueError(
                f"Observation operator has {nhead} rows, but a time slice of "
                f"the source yields {probe.size} observations."
            )

        sigma = _multiplier_helper(sigma, nhead, "sigma")
        weights = _multiplier_helper(weights, nhead, "weights")
        if np.any(~np.isfinite(sigma)):
            raise ValueError("sigma must contain only finite values")
        if np.any(sigma <= 0.0):
            raise ValueError("sigma must be strictly positive")
        if np.any(~np.isfinite(weights)):
            raise ValueError("weights must contain only finite values")
        if np.any(weights <= 0.0):
            raise ValueError(
                "weights must be strictly positive; omit an observation "
                "rather than assigning it zero weight"
            )

        self.head = head
        self.sigma = sigma
        self.weights = weights
        self.scale = np.sqrt(weights) / sigma
        self.P = (sparse.diags(self.scale) @ P).tocsr()

        # Default to the first slice so d is usable before any bind_time call.
        self.d = np.empty(nhead, dtype=float)
        self._index = np.zeros(1, dtype=int)
        self._current = -1
        self._set_d(0)
        return

    def _set_d(self, time_index: int) -> None:
        """Refresh ``d`` from time slice ``time_index`` of the source."""
        values = self._extract(self.head.isel(time=time_index).to_numpy())
        np.multiply(self.scale, values, out=self.d)
        self._current = time_index
        return

    def reset(self) -> None:
        """Forget which source slice ``d`` holds, forcing the next refresh."""
        self._current = -1
        return

    def bind_time(self, time: pd.DatetimeIndex) -> None:
        """
        Map a simulation time axis onto this target's own timestamps.

        Values hold until superseded, matching the stress-period convention.
        The index is computed once here rather than searched per step, so the
        coverage check below fails at setup rather than mid-run.
        """
        source = self.head.indexes["time"]
        if len(source) == 1:
            # Static source: constant for the whole run, and STATIC_TIME is
            # never compared against.
            self._index = np.zeros(len(time), dtype=int)
        else:
            index = source.get_indexer(time, method="pad")
            if (index < 0).any():
                first = time[index < 0][0]
                raise ValueError(
                    f"{type(self).__name__} has no data at or before {first}."
                )
            self._index = index
        self.reset()
        return

    def advance(self, time_index: int) -> None:
        """Refresh ``d`` for simulation step ``time_index``, if the source changed."""
        j = self._index[time_index]
        if j == self._current:
            return
        self._set_d(j)
        return


class GridSampling(FittingTarget):
    """
    Sample head values directly from corresponding grid cells.

    Only cells finite at every timestamp are included: the observation
    network is fixed for the run, so a cell that drops out partway would
    change the shape of ``P``.
    """

    def __init__(
        self,
        head: FloatArray | xr.DataArray,
        weights=None,
        sigma=None,
    ):
        head = _promote(head, dims=("y", "x"))
        notnull = head.notnull()
        mask = notnull.all("time")
        self._row_index = np.flatnonzero(mask)
        nhead = self._row_index.size
        if nhead == 0:
            raise ValueError("Full NoData at every timestamp.")
        if not np.array_equal(mask, notnull.any()):
            raise ValueError(
                "The set of NoData cells varies in time, but P is fixed. "
                "Drop intermittent cells or split the run into segments."
            )

        P = sparse.csr_matrix(
            (np.ones(nhead), (np.arange(nhead), self._row_index)),
            shape=(nhead, mask.size),
        )
        self._set_target(P=P, head=head, sigma=sigma, weights=weights)


class CellSampling(FittingTarget):
    """Sample at cell centers using nearest-neighbor lookup."""

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        head: FloatArray | xr.DataArray,
        grid: xu.Ugrid2d,
        weights=None,
        sigma=None,
    ):
        head = _promote(head, dims=("observation",))
        if head.ndim != 2:
            raise ValueError(
                f"Expected a source with dims (time, observation), got {head.dims}."
            )
        nhead = head.shape[1]

        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size != nhead or y.size != nhead:
            raise ValueError(
                f"x ({x.size}) and y ({y.size}) must both match the number of "
                f"observations ({nhead})."
            )

        j = grid.locate_points(np.column_stack((x, y)))
        # locate_points marks points outside the grid with -1, which would be a
        # valid (wrapping) column index in the matrix below.
        outside = np.flatnonzero(j < 0)
        if outside.size:
            raise ValueError(
                f"{outside.size} observation(s) fall outside the grid, first "
                f"at index {outside[0]} ({x[outside[0]]}, {y[outside[0]]})."
            )

        P = sparse.csr_matrix(
            (np.ones(nhead), (np.arange(nhead), j)),
            shape=(nhead, grid.n_face),
        )
        self._set_target(P=P, head=head, sigma=sigma, weights=weights)


class ModelTarget(FittingTarget):
    """Fit to cell-average heads from another model."""

    def __init__(
        self,
        head: FloatArray | xr.DataArray,
        grid: xu.Ugrid2d,
        weights=None,
        sigma=None,
    ):
        head = _promote(head, dims=("y", "x"))

        # Dummy source data for the regridder API; only the weights are used.
        source = xu.UgridDataArray.from_data(
            np.empty(grid.n_face),
            grid=grid,
            facet="face",
        )
        regridder = xu.OverlapRegridder(
            source=source,
            target=head.isel(time=0),
        )

        W = regridder._weights
        Wcsr = sparse.csr_matrix(
            (W.data, W.indices, W.indptr),
            shape=(W.n, W.m),
        )

        # Normalize overlap weights row-wise to produce cell averages. Rows
        # without overlap are dropped rather than zeroed: a zero row leaves the
        # solution unchanged but inflates the reported misfit by a constant the
        # model cannot influence.
        row_sums = np.asarray(Wcsr.sum(axis=1)).ravel()
        self._row_index = np.flatnonzero(row_sums > 0.0)
        if self._row_index.size == 0:
            raise ValueError("No target cell overlaps the source grid.")
        Wcsr = Wcsr[self._row_index]
        row_sums = row_sums[self._row_index]

        P = (sparse.diags(1.0 / row_sums) @ Wcsr).tocsr()
        self._set_target(P=P, head=head, sigma=sigma, weights=weights)


class CompositeTarget(FittingTarget):
    """
    Combine multiple fitting targets into a single P matrix and d vector.

    Since each component target is already scaled by its own ``sigma`` and
    ``weights``, composition is simply vertical stacking of the observation
    operators and concatenation of the target vectors.

    Only ``d`` is stacked. Each component's ``d`` is rebound as a view onto the
    corresponding row block, so a component's ``advance`` writes straight into
    the composite and the two cannot desync. Consequently a component belongs
    to exactly one composite, and its ``d`` must be written in place
    (``np.copyto``, ``out=``) rather than reassigned.
    """

    def __init__(
        self,
        targets: list[FittingTarget],
    ):
        if not targets:
            raise ValueError("At least one target required")

        # Flatten: rebinding a nested composite's d would detach its own
        # children, whose views point into the buffer being replaced. Their
        # writes would then land nowhere, silently.
        flat: list[FittingTarget] = []
        for target in targets:
            if isinstance(target, CompositeTarget):
                flat.extend(target.targets)
            else:
                flat.append(target)

        n_cols = flat[0].P.shape[1]
        for i, target in enumerate(flat):
            if target.P.shape[1] != n_cols:
                raise ValueError(
                    f"Incompatible grid sizes: target {i} "
                    f"({type(target).__name__}) has {target.P.shape[1]} "
                    f"columns, but target 0 has {n_cols}"
                )
            if getattr(target, "_composite", None) is not None:
                raise ValueError(
                    f"Target {i} ({type(target).__name__}) already belongs to "
                    "a composite. Construct a new component instead."
                )

        self.P = sparse.vstack([t.P for t in flat], format="csr")
        self.d = np.hstack([t.d for t in flat])

        # Relies on vstack preserving block order; the row count check is what
        # catches it if that ever stops holding.
        offsets = np.cumsum([0] + [t.n_targets for t in flat])
        if offsets[-1] != self.P.shape[0]:
            raise RuntimeError(
                f"Component rows sum to {offsets[-1]}, but stacked P has "
                f"{self.P.shape[0]} rows."
            )

        self.targets = flat
        self.slices = [slice(a, b) for a, b in zip(offsets[:-1], offsets[1:])]
        for target, target_slice in zip(flat, self.slices):
            target.d = self.d[target_slice]
            target._composite = self
        return

    def bind_time(self, time: pd.DatetimeIndex) -> None:
        for target in self.targets:
            target.bind_time(time)
        return

    def advance(self, time_index: int) -> None:
        # The bound views ensure the updates are materialized in the composite
        # d array.
        for target in self.targets:
            target.advance(time_index)
        return

    @property
    def scale(self) -> FloatArray:
        return np.hstack([t.scale for t in self.targets])

    @property
    def sigma(self) -> FloatArray:
        return np.hstack([t.sigma for t in self.targets])

    @property
    def weights(self) -> FloatArray:
        return np.hstack([t.weights for t in self.targets])

    @property
    def head(self):
        raise AttributeError(
            "A composite has no single source; its components' sources differ "
            "in shape and in time axis. Use .observed for current-step values "
            "in physical units, or inspect .targets individually."
        )

    def _extract(self, values):
        raise NotImplementedError("A composite has no single source.")

    def _set_d(self, time_index: int) -> None:
        raise NotImplementedError("Composite d is updated through components.")

    def reset(self) -> None:
        for target in self.targets:
            target.reset()
        return
