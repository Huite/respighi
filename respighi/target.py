import abc
from dataclasses import dataclass

import xarray as xr
import numpy as np
from scipy import sparse
from xugrid import Ugrid2d, OverlapRegridder, UgridDataArray
from respighi.constants import FloatArray


@dataclass
class FittingTarget(abc.ABC):
    """Operator P and target values d for a least-squares term ||P x - d||²."""

    P: sparse.csr_matrix
    d: FloatArray

    @property
    def n_targets(self):
        return self.P.shape[0]


class GridSampling(FittingTarget):
    """
    Provide a grid with notnull head values.
    Mostly useful for testing.
    """

    def __init__(self, head: FloatArray, weights=None):
        hflat = head.ravel()
        j = np.argwhere(np.isfinite(hflat))[0]
        nhead = len(j)
        i = np.arange(nhead)
        if weights is None:
            weights = np.ones(nhead)
        self.P = sparse.csr_matrix((weights, (i, j)), shape=(nhead, hflat.size))
        self.d = head[j]


class CellSampling(FittingTarget):
    """Sample at cell centers (nearest neighbor)."""

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        head: FloatArray,
        grid: Ugrid2d,
        weights=None,
    ):
        nhead = len(head)
        xy = np.column_stack((x, y))
        i = np.arange(nhead)
        j = grid.locate_points(xy)
        if weights is None:
            weights = np.ones(nhead)
        self.P = sparse.csr_matrix((weights, (i, j)), shape=(nhead, grid.n_face))
        self.d = head


class InterpolatedSampling(FittingTarget):
    """Sample with bilinear/barycentric interpolation at arbitrary x, y."""

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        head: FloatArray,
        grid: Ugrid2d,
        weights=None,
    ):
        nhead = len(head)
        xy = np.column_stack((x, y))
        i = np.arange(nhead)
        # FIXME:
        # We need the voronoi tesselation instead since this'll return vertex indices.
        # Also the required logic can be found in xugrid/regrid/unstructured.py
        j, barycentric_weights = grid.compute_barycentric_weights(xy)
        if weights is not None:
            barycentric_weights *= weights
        self.P = sparse.csr_matrix(
            (barycentric_weights, (i, j)), shape=(nhead, grid.n_face)
        )
        self.d = head


class ModelTarget(FittingTarget):
    """Fit to cell averages from a model."""

    def __init__(self, head: xr.DataArray, grid: Ugrid2d, weights=None):
        # Create dummy grid for regridder API
        source = UgridDataArray.from_data(
            np.empty(grid.n_face), grid=grid, facet="face"
        )
        regridder = OverlapRegridder(source=source, target=head)
        W = regridder._weights
        Wcsr = sparse.csr_matrix((W.data, W.indices, W.indptr))

        # Normalize rows to get averages
        row_sums = np.asarray(Wcsr.sum(axis=1)).ravel()
        if weights is not None:
            row_sums = row_sums / weights
        # Avoid division by zero for empty rows
        row_sums[row_sums == 0] = 1.0
        self.P = sparse.diags(1.0 / row_sums) @ Wcsr
        self.d = head.to_numpy().ravel()


class CompositeFittingTarget(FittingTarget):
    """Combine multiple fitting targets into a single P matrix and d vector."""

    def __init__(self, targets: list[FittingTarget]):
        if not targets:
            raise ValueError("At least one target required")
        n_cols = targets[0].P.shape[1]
        for t in targets:
            if t.P.shape[1] != n_cols:
                raise ValueError(f"Incompatible grid sizes: {t.P.shape[1]} vs {n_cols}")
        self.P = sparse.vstack([t.P for t in targets], format="csr")
        self.d = np.concatenate([t.d for t in targets])
