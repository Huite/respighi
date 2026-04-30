import warnings
from typing import Sequence

import geopandas as gpd
import numpy as np
import pypardiso
import xarray as xr
import xugrid as xu
from scipy import sparse

from respighi.cg import PCGSolver
from respighi.constants import BoolArray, FloatArray, IntArray
from respighi.ilu0 import ILU0Preconditioner


class Recharge:
    rate: FloatArray
    _rhs: FloatArray

    def __init__(self, rate):
        self.rate = rate.ravel()
        self._rhs = np.empty_like(self.rate)

    def formulate(self, rhs, area):
        np.multiply(self.rate, area, out=self._rhs)
        rhs += self._rhs
        return


class HeadBoundary:
    conductance: FloatArray
    head: FloatArray
    _rhs: FloatArray

    def __init__(self, conductance, head):
        self.conductance = conductance.ravel()
        self.head = head.ravel()
        self._rhs = np.empty_like(self.conductance)

    def formulate(self, hcof, rhs, head):
        hcof += self.conductance
        np.multiply(self.conductance, self.head, out=self._rhs)
        rhs += self._rhs
        return


class Drainage:
    conductance: FloatArray
    elevation: FloatArray
    _rhs: FloatArray
    _active: BoolArray

    def __init__(self, conductance, elevation):
        self.conductance = conductance.ravel()
        self.elevation = elevation.ravel()
        self._rhs = np.empty_like(self.conductance)
        self._active = np.empty(self.conductance.shape, dtype=bool)

    def formulate(self, hcof, rhs, head):
        # Only active if elevation < head
        np.less(self.elevation, head, out=self._active)
        np.add(hcof, self.conductance, out=hcof, where=self._active)
        np.multiply(self.conductance, self.elevation, out=self._rhs)
        np.add(rhs, self._rhs, out=rhs, where=self._active)
        return


class HorizontalFlowBarrier:
    layer: IntArray
    cell0: IntArray
    cell1: IntArray
    resistance: FloatArray

    def __init__(self, layer, cell0, cell1, resistance):
        self.layer = layer
        self.cell0 = cell0
        self.cell1 = cell1
        self.resistance = resistance

    @classmethod
    def from_geodataframe(
        cls,
        layer: int,
        barriers: gpd.GeoDataFrame,
        template: xr.DataArray,
        max_snap_distance: float,
    ):
        if "resistance" not in barriers.columns:
            raise ValueError("resistance must be present in barriers geodataframe")
        grid = xu.Ugrid2d.from_structured(template)
        uds, _ = xu.snap_to_grid(
            lines=barriers, grid=grid, max_snap_distance=max_snap_distance
        )
        edges = np.arange(grid.n_edge)

        is_hfb_edge = uds["resistance"].notnull().to_numpy()
        hfb_edges = edges[is_hfb_edge]
        hfb_faces = grid.edge_face_connectivity[hfb_edges]
        # Remove exterior edges
        is_interior = hfb_faces[:, 1] != -1
        cell0, cell1 = hfb_faces[is_interior].transpose()
        resistance = uds["resistance"].to_numpy()[is_hfb_edge][is_interior]
        return cls(
            layer=layer,
            cell0=cell0,
            cell1=cell1,
            resistance=resistance,
        )


class River:
    conductance: FloatArray
    stage: FloatArray
    bottom_elevation: FloatArray
    _fixed_rhs: FloatArray
    _rhs: FloatArray
    _fixed: BoolArray
    _linear: BoolArray

    def __init__(self, conductance, stage, bottom_elevation):
        self.conductance = conductance.ravel()
        self.stage = stage.ravel()
        self.bottom_elevation = bottom_elevation.ravel()
        self._fixed_rhs = self.conductance * (self.stage - self.bottom_elevation)
        self._rhs = np.empty_like(self.conductance)
        self._fixed = np.empty(self.conductance.shape, dtype=bool)
        self._linear = np.empty(self.conductance.shape, dtype=bool)

    def formulate(self, hcof, rhs, head):
        # Fixed rate if head < bottom_elevation, linear otherwise.
        np.less(head, self.bottom_elevation, out=self._fixed)
        np.logical_not(self._fixed, out=self._linear)
        # Fixed case: no hcof contribution, rhs += conductance * (stage - bottom_elevation)
        np.add(rhs, self._fixed_rhs, out=rhs, where=self._fixed)
        # Linear case: hcof += conductance, rhs += conductance * stage
        np.add(hcof, self.conductance, out=hcof, where=self._linear)
        np.multiply(self.conductance, self.stage, out=self._rhs)
        np.add(rhs, self._rhs, out=rhs, where=self._linear)
        return


def atleast_3d_front(a):
    a = np.asarray(a)
    while a.ndim < 3:
        a = a[np.newaxis]
    # Make sure it's owned by the groudwater model and that it's not a view.
    return a.copy()


class GroundwaterModel:
    def __init__(
        self,
        area,
        initial,
        recharge,
        head_boundaries,
        transmissivity: FloatArray,
        resistance: FloatArray | None = None,
        storativity: FloatArray | None = None,
        horizontal_flow_barriers: Sequence[HorizontalFlowBarrier] | tuple = (),
        xclose_linear: float = 1e-5,
        rclose_linear: float = 1e-5,
        maxiter_linear: int = 100,
        xclose: float = 1e-4,
        maxiter: int = 30,
    ):
        """
        Class for a confined groundwater flow model.

        Parameters
        ----------
        area
        initial
        recharge
        head_boundaries
        transmissivity
        resistance: optional
        storativity: optional
        horizontal_flow_barriers: optional
        xclose_linear: optional, float, default is 1e-5
            Linear convergence criterion
        rclose_linear: optional, float, default is 1e-5
            Linear convergence criterion
        maxiter_linear: int = 100,
            Maximum number of linear solver iterations.
        xclose: float = 1e-4,
            Non-linear convergence criterion.
        maxiter: int = 30,
            Maximum number of non-linear iterations.
        """
        transmissivity_3d = atleast_3d_front(transmissivity)
        initial_3d = atleast_3d_front(initial)
        if initial_3d.shape != transmissivity_3d.shape:
            raise ValueError("Shapes of transmissivity and initial head do not match.")
        nlayer, ny, nx = transmissivity_3d.shape
        if resistance is None:
            if nlayer != 1:
                raise ValueError(
                    "If resistance is not specified, transmissivity must be 2D or (1, ny, nx)."
                )
            resistance_3d = np.zeros((0, ny, nx))
        else:
            resistance_3d = atleast_3d_front(resistance)
            nlayer_c, ny_c, nx_c = resistance_3d.shape
            if nlayer_c != (nlayer - 1):
                raise ValueError(
                    "Resistance nlayer must equal transmissivity nlayer - 1"
                )
            if (ny_c != ny) or (nx_c != nx):
                raise ValueError(
                    "x, y sizes between transmissivity and resistance do not match."
                )

        if storativity is None:
            storativity_3d = np.zeros_like(transmissivity_3d)
        else:
            storativity_3d = atleast_3d_front(storativity)
            if storativity_3d.shape != transmissivity_3d.shape:
                raise ValueError(
                    "Shapes of storativity and transmissivity do not match."
                )

        self.initial = initial_3d.ravel()
        self.recharge = recharge
        self.head_boundaries = head_boundaries

        n = self.initial.size
        self.layer_n = ny * nx
        self.transmissivity = transmissivity_3d
        self.resistance = resistance_3d
        self.storativity = storativity_3d.ravel()
        self.horizontal_flow_barriers = horizontal_flow_barriers
        self.area = area
        self.n = n
        self.rhs = np.zeros(n)
        self.head = self.initial.copy()
        # Work arrays
        self._head_old = self.head.copy()
        self._head_iter = np.empty_like(self.head)
        self._storage_work = np.empty_like(self.storativity)
        self._update = np.empty_like(self.head)

        # Matrix assembly
        self.W = self._build_conductance(
            transmissivity_3d,
            resistance_3d,
            area,
            horizontal_flow_barriers,
        )

        # Compute the (weighted) degree matrix
        self.D = np.asarray(self.W.sum(axis=1)).ravel()
        self.hcof = self.D.copy()
        # Compute the Laplacian
        self.Abase = sparse.diags(self.D) - self.W
        self.A = self.Abase.copy()

        self.linearsolver = PCGSolver(
            self.A,
            self.rhs,
            self.head,
            ILU0Preconditioner.from_csr_matrix(self.A),
            xclose=xclose_linear,
            rclose=rclose_linear,
            maxiter=maxiter_linear,
        )
        self.maxiter = maxiter
        self.xclose = xclose

    @classmethod
    def _build_connectivity(cls, shape):
        size = np.prod(shape)
        index = np.arange(size).reshape(shape)
        # Build nD connectivity
        ii = []
        jj = []
        for d in range(len(shape)):
            slices = [slice(None)] * len(shape)

            slices[d] = slice(None, -1)
            left = index[tuple(slices)].ravel()
            slices[d] = slice(1, None)
            right = index[tuple(slices)].ravel()
            ii.extend([left, right])
            jj.extend([right, left])

        i = np.concatenate(ii)
        j = np.concatenate(jj)
        return i, j

    @classmethod
    def _build_conductance(
        cls, transmissivity, resistance, area, horizontal_flow_barriers
    ):
        # Get the Cartesian neighbors for a finite difference approximation.
        # TODO: check order of dimensions with DataArray
        _, ny, nx = transmissivity.shape
        size = transmissivity.size
        layer_size = ny * nx
        i, j = cls._build_connectivity(transmissivity.shape)
        kD = transmissivity.ravel()
        c = resistance.ravel()

        delta = abs(i - j)
        horizontal = delta < layer_size
        conductance = np.empty_like(i, dtype=float)
        kDi = kD[i[horizontal]]
        kDj = kD[j[horizontal]]
        C_ij = (2 * kDi * kDj) / (kDi + kDj)
        conductance[horizontal] = C_ij

        if not horizontal.all():
            vertical = ~horizontal
            i_upper = np.minimum(i[vertical], j[vertical])
            conductance[vertical] = area / c[i_upper]

        if horizontal_flow_barriers:
            i_all = [i]
            j_all = [j]
            data_all = [conductance]
            for hfb in horizontal_flow_barriers:
                # Utilize a negative correction: duplicate summing of the COO matrix does the
                # necessary work.
                cell0 = hfb.cell0 + hfb.layer * layer_size
                cell1 = hfb.cell1 + hfb.layer * layer_size
                kDi = kD[cell0]
                kDj = kD[cell1]
                C_ij = (2 * kDi * kDj) / (kDi + kDj)
                C_modified = C_ij / (1.0 + hfb.resistance * C_ij)
                C_delta = C_modified - C_ij
                i_all.extend([cell0, cell1])
                j_all.extend([cell1, cell0])
                data_all.extend([C_delta, C_delta])

            rows = np.concatenate(i_all)
            columns = np.concatenate(j_all)
            data = np.concatenate(data_all)
            return sparse.coo_matrix(
                (data, (rows, columns)), shape=(size, size)
            ).tocsr()
        else:
            return sparse.coo_matrix((conductance, (i, j)), shape=(size, size)).tocsr()

    def formulate(self, recharge=True, dt=0.0):
        # Reset
        self.rhs[:] = 0.0
        self.hcof[:] = self.D[:]

        # Formulate storage
        # dt = 0.0 encodes steady state behavior, i.e. no storage.
        if dt > 0.0:
            inv_dt = self.area / dt
            np.multiply(self.storativity, inv_dt, out=self._storage_work)
            self.hcof += self._storage_work
            self._storage_work *= self._head_old
            self.rhs[:] += self._storage_work

        # Touch only the first layer for boundary conditions
        rhs = self.rhs[: self.layer_n]
        hcof = self.hcof[: self.layer_n]
        head = self.head[: self.layer_n]

        # Accumulate boundary conditions
        if recharge:
            self.recharge.formulate(rhs, self.area)
        for boundary in self.head_boundaries:
            boundary.formulate(hcof, rhs, head)
        return

    def direct_linear_solve(self):
        self.A.setdiag(self.hcof)
        self.head[:] = pypardiso.spsolve(self.A, self.rhs)
        return

    def linear_solve(self, warn=True):
        self.A.setdiag(self.hcof)
        converged, iterations = self.linearsolver.solve()
        if warn and not converged:
            warnings.warn(
                f"Groundwater linear solver did not converge after {iterations} iterations."
            )
        return converged, iterations

    def nonlinear_solve(self, dt=0.0):
        """Solve nonlinear system using Picard iteration"""
        for i in range(self.maxiter):
            np.copyto(self._head_iter, self.head)
            self.formulate(dt=dt)
            converged_linear, _ = self.linear_solve(warn=False)
            np.subtract(self.head, self._head_iter, out=self._update)
            maxdx = np.linalg.norm(self._update, ord=np.inf)
            print(maxdx)
            if maxdx < self.xclose:
                return True, i + 1

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} iterations. "
            f"Final update: {maxdx:.2e}"
        )
        return False, self.maxiter

    def run(self, dts):
        np.copyto(dst=self.head, src=self.initial)
        out = []
        for dt in dts:
            np.copyto(dst=self._head_old, src=self.head)
            converged, iters = self.nonlinear_solve(dt=dt)
            out.append(self.head.copy())
        return out
