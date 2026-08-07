import abc
from copy import copy
from dataclasses import dataclass

import numpy as np
import xarray as xr
import xugrid as xu
from numpy.typing import NDArray
from scipy import sparse

FloatArray = NDArray[np.floating]


@dataclass
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

    Parameters stored on each target
    --------------------------------
    P : scipy.sparse.csr_matrix
        Scaled observation operator.
    d : ndarray
        Scaled target values.
    head : ndarray
        Original target values in physical head units.
    sigma : ndarray
        Observation standard deviations in physical head units.
        A value of 1.0 corresponds to unscaled least squares.
    weights : ndarray
        Dimensionless precision/importance weights.
    scale : ndarray
        Row scaling sqrt(weights) / sigma.
    """

    P: sparse.csr_matrix
    d: FloatArray
    head: FloatArray
    sigma: FloatArray
    weights: FloatArray
    scale: FloatArray

    @property
    def n_targets(self) -> int:
        return self.P.shape[0]


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


def _set_target(
    target: FittingTarget,
    P: sparse.spmatrix,
    head: FloatArray,
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
    head = np.asarray(head, dtype=float).ravel()
    nhead = head.size
    P = P.tocsr()

    if P.shape[0] != nhead:
        raise ValueError(
            f"Observation operator has {P.shape[0]} rows, "
            f"but there are {nhead} target values"
        )

    sigma = _multiplier_helper(sigma, nhead, "sigma")
    weights = _multiplier_helper(weights, nhead, "weights")
    if np.any(~np.isfinite(sigma)):
        raise ValueError("sigma must contain only finite values")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive")
    if np.any(~np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    scale = np.sqrt(weights) / sigma
    target.head = head
    target.sigma = sigma
    target.weights = weights
    target.scale = scale
    target.P = (sparse.diags(scale) @ P).tocsr()
    target.d = scale * head


class GridSampling(FittingTarget):
    """
    Sample head values directly from corresponding grid cells.
    Only finite values in ``head`` are included.
    Mostly useful for testing.
    """

    def __init__(
        self,
        head: FloatArray,
        weights=None,
        sigma=None,
    ):
        hflat = np.asarray(head, dtype=float).ravel()
        j = np.flatnonzero(np.isfinite(hflat))
        nhead = j.size
        i = np.arange(nhead)
        P = sparse.csr_matrix(
            (np.ones(nhead), (i, j)),
            shape=(nhead, hflat.size),
        )
        _set_target(
            self,
            P=P,
            head=hflat[j],
            sigma=sigma,
            weights=weights,
        )


class CellSampling(FittingTarget):
    """Sample at cell centers using nearest-neighbor lookup."""

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        head: FloatArray,
        grid: xu.Ugrid2d,
        weights=None,
        sigma=None,
    ):
        head = np.asarray(head, dtype=float).ravel()
        nhead = head.size
        xy = np.column_stack((x, y))
        i = np.arange(nhead)
        j = grid.locate_points(xy)
        P = sparse.csr_matrix(
            (np.ones(nhead), (i, j)),
            shape=(nhead, grid.n_face),
        )
        _set_target(
            self,
            P=P,
            head=head,
            sigma=sigma,
            weights=weights,
        )


class InterpolatedSampling(FittingTarget):
    """Sample with bilinear/barycentric interpolation at arbitrary x, y."""

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        head: FloatArray,
        grid: xu.Ugrid2d,
        weights=None,
        sigma=None,
    ):
        head = np.asarray(head, dtype=float).ravel()
        nhead = head.size
        xy = np.column_stack((x, y))
        # FIXME:
        # We need the Voronoi tessellation instead since this returns
        # vertex indices. The required logic can also be found in
        # xugrid/regrid/unstructured.py.
        j, barycentric_weights = grid.compute_barycentric_weights(xy)
        j = np.asarray(j)
        barycentric_weights = np.asarray(barycentric_weights)
        if j.shape != barycentric_weights.shape:
            raise ValueError(
                "Interpolation indices and barycentric weights "
                "must have matching shapes"
            )
        if j.ndim == 1:
            if j.size != nhead:
                raise ValueError(
                    "Interpolation result size does not match number of observations"
                )
            i = np.arange(nhead)
        else:
            if j.shape[0] != nhead:
                raise ValueError(
                    "Interpolation result size does not match number of observations"
                )
            i = np.broadcast_to(
                np.arange(nhead)[:, np.newaxis],
                j.shape,
            )

        P = sparse.csr_matrix(
            (
                barycentric_weights.ravel(),
                (i.ravel(), j.ravel()),
            ),
            shape=(nhead, grid.n_face),
        )
        _set_target(
            self,
            P=P,
            head=head,
            sigma=sigma,
            weights=weights,
        )


class ModelTarget(FittingTarget):
    """Fit to cell-average heads from another model."""

    def __init__(
        self,
        head: xr.DataArray,
        grid: xu.Ugrid2d,
        weights=None,
        sigma=None,
    ):
        # Create dummy source data for the regridder API.
        source = xu.UgridDataArray.from_data(
            np.empty(grid.n_face),
            grid=grid,
            facet="face",
        )

        regridder = xu.OverlapRegridder(
            source=source,
            target=head,
        )

        W = regridder._weights
        Wcsr = sparse.csr_matrix(
            (W.data, W.indices, W.indptr),
            shape=(W.n, W.m),
        )

        # Normalize overlap weights row-wise to produce cell averages.
        row_sums = np.asarray(Wcsr.sum(axis=1)).ravel()
        # Avoid division by zero for target cells without overlap.
        row_sums[row_sums == 0.0] = 1.0
        P = (sparse.diags(1.0 / row_sums) @ Wcsr).tocsr()
        head_flat = np.asarray(head.to_numpy(), dtype=float).ravel()
        if P.shape[0] != head_flat.size:
            raise ValueError(
                f"Regridding operator has {P.shape[0]} rows, "
                f"but target head contains {head_flat.size} values"
            )

        _set_target(
            self,
            P=P,
            head=head_flat,
            sigma=sigma,
            weights=weights,
        )

    def update_head(self, head: xr.DataArray):
        """
        Return a copy with new observations while reusing P.

        This assumes that the new head array corresponds to exactly the same
        target grid, so the observation operator and row scaling remain valid.
        """
        new = copy(self)
        new_head = np.asarray(head.to_numpy(), dtype=float).ravel()
        if new_head.size != self.n_targets:
            raise ValueError(
                f"New head contains {new_head.size} values, expected {self.n_targets}"
            )
        new.head = new_head
        new.d = new.scale * new_head
        return new


class CompositeTarget(FittingTarget):
    """
    Combine multiple fitting targets into a single P matrix and d vector.

    Since each component target is already scaled by its own ``sigma`` and
    ``weights``, composition is simply vertical stacking of the observation
    operators and concatenation of the target vectors.
    """

    def __init__(
        self,
        targets: list[FittingTarget],
    ):
        if not targets:
            raise ValueError("At least one target required")

        n_cols = targets[0].P.shape[1]
        for target in targets:
            if target.P.shape[1] != n_cols:
                raise ValueError(
                    f"Incompatible grid sizes: {target.P.shape[1]} vs {n_cols}"
                )

        self.P = sparse.vstack(
            [target.P for target in targets],
            format="csr",
        )
        self.d = np.hstack([target.d for target in targets])
        self.head = np.hstack([target.head for target in targets])
        self.sigma = np.hstack([target.sigma for target in targets])
        self.weights = np.hstack([target.weights for target in targets])
        self.scale = np.hstack([target.scale for target in targets])
