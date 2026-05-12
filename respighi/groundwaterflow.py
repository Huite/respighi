import warnings
from typing import Sequence

import geopandas as gpd
import numpy as np
import pypardiso
import xarray as xr
import xugrid as xu
from scipy import sparse

from respighi.constants import BoolArray, FloatArray, IntArray
from respighi.linearsolvers.cg import PCGSolver
from respighi.linearsolvers.ilu0 import ILU0Preconditioner


def constant_helper(dataset, template_var, constant, name):
    if constant is not None:
        template = dataset[template_var]
        return xr.full_like(template, constant).where(template.notnull())
    else:
        return dataset.get(name)


class Recharge:
    """
    Spatially distributed recharge boundary condition.

    Adds a source term to the RHS: ``rhs += rate * area``.
    """

    rate: FloatArray
    _rhs: FloatArray

    def __init__(self, rate):
        self.rate = rate.ravel()
        self._rhs = np.empty_like(self.rate)

    def formulate(self, rhs, area):
        np.multiply(self.rate, area, out=self._rhs)
        rhs += self._rhs
        return

    @classmethod
    def from_dataset(cls, dataset):
        return cls(
            rate=dataset["rate"].fillna(0.0).to_numpy(),
        )


class HeadBoundary:
    """
    Fixed-head  boundary condition.

    Adds a conductance term to the diagonal and a corresponding RHS contribution,
    driving the head toward the specified boundary head value.
    """

    conductance: FloatArray
    head: FloatArray
    _rhs: FloatArray
    sigma: FloatArray

    def __init__(self, conductance, head, sigma=None):
        self.conductance = conductance.ravel()
        self.head = head.ravel()
        self._rhs = np.empty_like(self.conductance)
        self.sigma = sigma

    def formulate(self, hcof, rhs, head):
        hcof += self.conductance
        np.multiply(self.conductance, self.head, out=self._rhs)
        rhs += self._rhs
        return

    @classmethod
    def from_dataset(cls, dataset, constant_sigma=None):
        sigma = constant_helper(dataset, "conductance", constant_sigma, "sigma")
        return cls(
            conductance=dataset["conductance"].fillna(0.0).to_numpy(),
            head=dataset["head"].fillna(0.0).to_numpy(),
            sigma=None if sigma is None else sigma.fillna(0.0).to_numpy(),
        )


class Drainage:
    """
    Drainage boundary condition.

    Active only where the computed head exceeds the drain elevation. When active,
    behaves like a head boundary at the drain elevation; otherwise contributes
    nothing to the system.
    """

    conductance: FloatArray
    elevation: FloatArray
    _rhs: FloatArray
    _active: BoolArray
    sigma: FloatArray

    def __init__(self, conductance, elevation, sigma=None):
        self.conductance = conductance.ravel()
        self.elevation = elevation.ravel()
        self._rhs = np.empty_like(self.conductance)
        self._active = np.empty(self.conductance.shape, dtype=bool)
        self.sigma = sigma

    def formulate(self, hcof, rhs, head):
        # Only active if elevation < head
        np.less(self.elevation, head, out=self._active)
        np.add(hcof, self.conductance, out=hcof, where=self._active)
        np.multiply(self.conductance, self.elevation, out=self._rhs)
        np.add(rhs, self._rhs, out=rhs, where=self._active)
        return

    @classmethod
    def from_dataset(cls, dataset, constant_conductance=None, constant_sigma=None):
        conductance = constant_helper(
            dataset, "elevation", constant_conductance, "conductance"
        )
        sigma = constant_helper(dataset, "elevation", constant_sigma, "sigma")
        return cls(
            conductance=conductance.fillna(0.0).to_numpy(),
            elevation=dataset["elevation"].fillna(0.0).to_numpy(),
            sigma=None if sigma is None else sigma.fillna(0.0).to_numpy(),
        )


class River:
    """
    River boundary condition.

    Linear (head-dependent) when head is above the river bottom elevation:
    flux = conductance * (stage - head). Fixed rate when head drops below the
    bottom elevation: flux = conductance * (stage - bottom_elevation).
    """

    conductance: FloatArray
    stage: FloatArray
    bottom_elevation: FloatArray
    _fixed_rhs: FloatArray
    _rhs: FloatArray
    _fixed: BoolArray
    _linear: BoolArray
    sigma: FloatArray

    def __init__(self, conductance, stage, bottom_elevation, sigma):
        self.conductance = conductance.ravel()
        self.stage = stage.ravel()
        self.bottom_elevation = bottom_elevation.ravel()
        self._fixed_rhs = self.conductance * (self.stage - self.bottom_elevation)
        self._rhs = np.empty_like(self.conductance)
        self._fixed = np.empty(self.conductance.shape, dtype=bool)
        self._linear = np.empty(self.conductance.shape, dtype=bool)
        self.sigma = sigma

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

    @classmethod
    def from_dataset(cls, dataset, constant_sigma=None):
        sigma = constant_helper(dataset, "conductance", constant_sigma, "sigma")
        return cls(
            conductance=dataset["conductance"].fillna(0.0).to_numpy(),
            stage=dataset["stage"].fillna(0.0).to_numpy(),
            bottom_elevation=dataset["bottom_elevation"].fillna(0.0).to_numpy(),
            sigma=None if sigma is None else sigma.fillna(0.0).to_numpy(),
        )


class HorizontalFlowBarrier:
    layer: IntArray  # TODO: 0-based indexing or 1-based? Currently 0-based.
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

    def modify_conductance(self, transmissivity: FloatArray):
        _, ny, nx = transmissivity.shape
        layer_size = ny * nx
        if self.cell0.max() > layer_size:
            raise ValueError("HFB cell0 index exceeds number of cells in a layer")
        if self.cell1.max() > layer_size:
            raise ValueError("HFB cell1 index exceeds number of cells in a layer")
        # Utilize a negative correction: duplicate summing of the COO matrix does the
        # necessary work.
        kD = transmissivity.ravel()
        i = self.cell0 + self.layer * layer_size
        j = self.cell1 + self.layer * layer_size
        kDi = kD[i]
        kDj = kD[j]
        C_ij = (2 * kDi * kDj) / (kDi + kDj)
        C_modified = C_ij / (1.0 + self.resistance * C_ij)
        C_delta = C_modified - C_ij
        return [i, j], [j, i], [C_delta, C_delta]


def atleast_3d_front(a):
    a = np.array(a)
    while a.ndim < 3:
        a = a[np.newaxis]
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
        horizontal_flow_barriers: Sequence[HorizontalFlowBarrier] = (),
        xclose_linear: float = 1e-5,
        rclose_linear: float = 1e-5,
        maxiter_linear: int = 100,
        xclose: float = 1e-4,
        maxiter: int = 30,
    ):
        """
        Confined groundwater flow model.

        Parameters
        ----------
        area: float
            Cell size area.
        initial: np.ndarray of floats
            Initial head.
        recharge: Recharge
            Recharge boundary condition.
        head_boundaries: sequence
            Boundaries such as drainage, river, (linear) head boundary.
        transmissivity: np.ndarray of floats
            May be shaped ``(ny, nx)`` for a single layer model;
            or shaped ``(nlayer, ny, nx``) for multi-layer models.
        resistance: np.ndarray of floats, optional
            May be shaped ``(ny, nx)`` for a two layer model;
            or shaped ``(nlayer - 1, ny, nx``) for more layers.
        storativity: np.ndarray of floats, optional
            May be shaped ``(ny, nx)`` for a single layer model;
            or shaped ``(nlayer, ny, nx``) for multi-layer models.
        horizontal_flow_barriers: sequence of HorizontalFlowBarrier, optional
            Horizontal flow barriers. These are static in time: the modification
            to intercell conductances is made at model initialization.
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

        if isinstance(transmissivity, xr.DataArray):
            coords = dict(transmissivity.coords)
            if "layer" not in coords:
                coords["layer"] = [0]  # TODO(?): 0 or 1-based indexing
            self._coords = coords
        else:
            dx = np.sqrt(area)
            self._coords = {
                "layer": np.arange(nlayer),
                "y": np.flip(np.arange(0.0, ny * dx, dx)),
                "x": np.arange(0.0, nx * dx, dx),
            }

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
        self._head = self.initial.copy()
        # Work arrays
        self._head_old = self._head.copy()
        self._head_iter = np.empty_like(self._head)
        self._storage_work = np.empty_like(self.storativity)
        self._update = np.empty_like(self._head)

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
            self._head,
            ILU0Preconditioner.from_csr_matrix(self.A),
            xclose=xclose_linear,
            rclose=rclose_linear,
            maxiter=maxiter_linear,
        )
        self.maxiter = maxiter
        self.xclose = xclose

    @classmethod
    def build_connectivity(cls, shape):
        """
        Return the row and column indices of all nearest-neighbour pairs for a
        grid of the given shape.

        Connections are built along every axis, so for a ``(nlayer, ny, nx)``
        grid this covers horizontal (x, y) and vertical (layer) neighbours.
        Each pair appears twice — once in each direction — yielding a symmetric
        sparsity pattern suitable for the conductance matrix.
        """

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
        """
        Assemble the weighted adjacency matrix of internodal conductances.

        Horizontal conductances between cells i and j use the harmonic mean of
        transmissivities: ``C_ij = 2*kDi*kDj / (kDi + kDj)``. Vertical
        conductances between layers use ``C = area / resistance``. Horizontal
        flow barriers apply a negative correction via duplicate COO entries.

        Returns a CSR sparse matrix of shape ``(n, n)`` where ``n`` is the total
        number of cells.
        """

        # Get the Cartesian neighbors for a finite difference approximation.
        # TODO: check order of dimensions with DataArray
        _, ny, nx = transmissivity.shape
        size = transmissivity.size
        layer_size = ny * nx
        i, j = cls.build_connectivity(transmissivity.shape)
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

        rows = [i]
        columns = [j]
        data = [conductance]
        for hfb in horizontal_flow_barriers:
            # Utilize a negative correction: duplicate summing of the COO matrix does the
            # necessary work.
            ij, ji, C_delta = hfb.modify_conductance(transmissivity)
            rows.extend(ij)
            columns.extend(ji)
            data.extend(C_delta)

        return sparse.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsr()

    def formulate(self, recharge=True, dt=0.0):
        """
        Assemble the RHS vector and diagonal (hcof) for the current iteration.

        Resets RHS and diagonal to their base values, then accumulates
        contributions from storage (if ``dt > 0``), recharge, and all head
        boundaries. A ``dt`` of 0.0 encodes steady-state behaviour — no storage
        term is added.

        Parameters
        ----------
        recharge:
            Whether to include the recharge boundary condition.
        dt:
            Time step size. Set to 0.0 for steady state.
        """
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
        head = self._head[: self.layer_n]

        # Accumulate boundary conditions
        if recharge:
            self.recharge.formulate(rhs, self.area)
        for boundary in self.head_boundaries:
            boundary.formulate(hcof, rhs, head)
        return

    def direct_linear_solve(self):
        """
        Solve the linear system directly using PARDISO.

        Updates ``_head`` in-place. Prefer this over ``linear_solve`` for
        problems where the iterative PCG solver struggles to converge, at the
        cost of higher memory usage.
        """
        self.A.setdiag(self.hcof)
        self._head[:] = pypardiso.spsolve(self.A, self.rhs)
        return

    def linear_solve(self, warn=True):
        """
        Solve the linear system iteratively using preconditioned conjugate gradients.

        Updates ``_head`` in-place via the PCG solver with ILU0 preconditioning.

        Parameters
        ----------
        warn:
            Whether to emit a warning if the solver does not converge.

        Returns
        -------
        converged: bool
            Whether the solver met the convergence criterion.
        iterations: int
            Number of iterations taken.
        """
        self.A.setdiag(self.hcof)
        converged, iterations = self.linearsolver.solve()
        if warn and not converged:
            warnings.warn(
                f"Groundwater linear solver did not converge after {iterations} iterations."
            )
        return converged, iterations

    def nonlinear_solve(self, dt=0.0):
        """
        Solve the nonlinear system using Picard iteration.

        Repeatedly calls ``formulate`` and ``linear_solve`` until the
        infinity-norm of the head update falls below ``xclose``, or
        ``maxiter`` iterations are reached.

        Parameters
        ----------
        dt:
            Time step size. Set to 0.0 for steady state.

        Returns
        -------
        converged: bool
            Whether the solver met the convergence criterion.
        iterations: int
            Number of iterations taken.
        """
        for i in range(self.maxiter):
            np.copyto(self._head_iter, self._head)
            self.formulate(dt=dt)
            converged_linear, _ = self.linear_solve(warn=False)
            np.subtract(self._head, self._head_iter, out=self._update)
            maxdx = np.linalg.norm(self._update, ord=np.inf)
            print(maxdx)
            if maxdx < self.xclose:
                return True, i + 1

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} iterations. "
            f"Final update: {maxdx:.2e}"
        )
        return False, self.maxiter

    def run(self, dts, callback=None):
        """
        Run a transient or batched simulation over a sequence of time steps.

        Resets the head to the initial condition, then advances the solution
        through each time step in ``dts`` using nonlinear Picard iteration.
        Before each step, the optional ``callback`` is invoked, allowing
        time-varying model state — recharge rates, boundary conditions, etc.,
        to be updated in-place.

        Parameters
        ----------
        dts:
            Sequence of time step sizes (same units as storativity).
        callback:
            Optional callable with signature ``callback(model, i, dt)``,
            where ``model`` is the ``GroundwaterModel`` instance, ``i`` is
            the zero-based step index, and ``dt`` is the current time step
            size. Called at the start of each step, before formulation.
        """

        np.copyto(dst=self._head, src=self.initial)
        out = []
        for i, dt in enumerate(dts):
            if callback is not None:
                callback(self, i, dt)
            np.copyto(dst=self._head_old, src=self._head)
            converged, iters = self.nonlinear_solve(dt=dt)
            out.append(self._head.copy())
        return out

    @property
    def head(self) -> xr.DataArray:
        """
        Current head as a labelled DataArray of shape ``(layer, y, x)``.

        Coordinates are taken from the transmissivity DataArray if one was
        provided at construction, otherwise synthesised from cell size and
        grid dimensions.
        """
        head_3d = self._head.reshape(self.transmissivity.shape)
        return xr.DataArray(
            head_3d, dims=("layer", "y", "x"), coords=self._coords, name="head"
        )
