import abc
import tempfile
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import xarray as xr
import xugrid as xu
from scipy import sparse
from scipy.sparse._sparsetools import csr_matvec

from respighi.constants import BoolArray, FloatArray, IntArray
from respighi.linearsolvers.settings import LinearSettings, PCGSettings
from respighi.linearsolvers.solvertypes import MatrixType
from respighi.nonlinear import NonlinearIteration, NonlinearSettings
from respighi.output import zarr_writer


def constant_helper(dataset, template_var, constant, name):
    """
    Resolve a field that may be given as a constant or as a dataset variable.

    Returns None when neither is available, so callers must handle a missing
    optional field rather than assuming a DataArray comes back.
    """
    if constant is not None:
        template = dataset[template_var]
        return xr.full_like(template, constant).where(template.notnull())
    return dataset.get(name)


def _to_array(dataarray):
    """Fill missing values with zero and return a flat float array, or None."""
    if dataarray is None:
        return None
    return dataarray.fillna(0.0).to_numpy().ravel()


class Boundary(abc.ABC):
    """
    Interface for boundary conditions contributing to the linear system.

    All boundaries share one ``formulate`` signature so that the model can
    accumulate them in a single loop. Boundaries that do not need every
    argument simply ignore it.
    """

    @abc.abstractmethod
    def formulate(self, hcof: FloatArray, rhs: FloatArray, head: FloatArray) -> None:
        """
        Accumulate this boundary's contribution into the diagonal and RHS.

        Parameters
        ----------
        hcof:
            Diagonal of the system matrix. Added to in place.
        rhs:
            Right hand side vector. Added to in place.
        head:
            Current head, for head-dependent boundaries.
        """

    @classmethod
    @abc.abstractmethod
    def from_dataset(cls, dataset):
        """Construct from an xarray Dataset of boundary parameters."""

    def bind_grid(self, area: float) -> None:
        """
        Receive grid geometry from the model at construction.

        Default is a no-op; boundaries whose contribution scales with cell
        area override this.
        """
        return

    def advance(self, time_index: int) -> None:
        """Move to the given time step. Default is a no-op."""
        return

    def bind_time(self, time) -> None:
        """Receive the simulation time axis. Default is a no-op."""
        return


class Recharge(Boundary):
    """
    Spatially distributed recharge boundary condition.

    Adds a source term to the RHS: ``rhs += rate * area``.
    """

    rate: FloatArray
    area: float | None
    _rhs: FloatArray

    def __init__(self, rate, area=None):
        self.rate = rate.ravel()
        self.area = area
        self._rhs = np.empty_like(self.rate)

    def bind_grid(self, area: float) -> None:
        self.area = area

    def formulate(self, hcof, rhs, head):
        if self.area is None:
            raise RuntimeError("bind_grid must be called before formulate")
        np.multiply(self.rate, self.area, out=self._rhs)
        rhs += self._rhs
        return

    @classmethod
    def from_dataset(cls, dataset):
        return cls(rate=dataset["rate"].fillna(0.0).to_numpy())


class HeadBoundary(Boundary):
    """
    Fixed-head boundary condition.

    Adds a conductance term to the diagonal and a corresponding RHS contribution,
    driving the head toward the specified boundary head value.
    """

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

    @classmethod
    def from_dataset(cls, dataset):
        return cls(
            conductance=dataset["conductance"].fillna(0.0).to_numpy(),
            head=dataset["head"].fillna(0.0).to_numpy(),
        )


def _smoothstep_inplace(x, width, work):
    """
    Transform x in place from a signed distance to a threshold
    (e.g. head - elevation) into an activation weight in [0, 1], and write the
    accompanying offset term into work.

    The weight is a clipped linear ramp across an interval of ``width``. The
    offset ``0.5 * width * t * (1 - t)`` is what makes the resulting flux a
    C1-continuous piecewise quadratic: combining them gives ``0.5 * C * w * t**2``,
    whose derivative ``C * t`` matches the fully active branch at t = 1 and
    vanishes at t = 0.

    x and work are preallocated arrays of the same shape; both are overwritten.
    """
    np.add(x, 0.5 * width, out=x)
    np.multiply(x, 1.0 / width, out=x)
    np.clip(x, 0.0, 1.0, out=x)
    np.subtract(1.0, x, out=work)  # 1 - t
    np.multiply(work, x, out=work)  # t * (1 - t)
    np.multiply(work, 0.5 * width, out=work)


class Drainage(Boundary):
    """
    Drainage boundary condition.

    Smoothly activates as head rises above the drain elevation, transitioning
    from inactive to a head boundary at the drain elevation over an interval
    `smoothing_width`, instead of switching discontinuously like the classical
    Drainage package.

    The diagonal contribution ``C * t`` is the exact derivative of the smoothed
    flux, so this term is linearised by its tangent rather than by a lagged
    coefficient.
    """

    conductance: FloatArray
    elevation: FloatArray
    _celev: FloatArray
    _weight: FloatArray
    _rhs: FloatArray
    smoothing_width: float

    def __init__(self, conductance, elevation, smoothing_width=0.01):
        if smoothing_width <= 0.0:
            raise ValueError(
                f"smoothing_width must be positive, got: {smoothing_width}"
            )
        self.conductance = conductance.ravel()
        self.elevation = elevation.ravel()
        self._celev = self.conductance * self.elevation  # constant: C * elevation
        self._weight = np.empty_like(self.conductance)
        self._rhs = np.empty_like(self.conductance)
        self.smoothing_width = smoothing_width

    def formulate(self, hcof, rhs, head):
        np.subtract(head, self.elevation, out=self._weight)
        _smoothstep_inplace(self._weight, self.smoothing_width, self._rhs)
        # rhs -= conductance * offset, while _rhs still holds the offset
        np.multiply(self._rhs, self.conductance, out=self._rhs)
        np.subtract(rhs, self._rhs, out=rhs)
        # hcof += conductance * t
        np.multiply(self.conductance, self._weight, out=self._rhs)
        np.add(hcof, self._rhs, out=hcof)
        # rhs += conductance * elevation * t
        np.multiply(self._celev, self._weight, out=self._rhs)
        np.add(rhs, self._rhs, out=rhs)
        return

    @classmethod
    def from_dataset(
        cls,
        dataset,
        constant_conductance=None,
        smoothing_width=0.01,
    ):
        conductance = constant_helper(
            dataset, "elevation", constant_conductance, "conductance"
        )
        if conductance is None:
            raise ValueError(
                "Drainage requires a 'conductance' variable or constant_conductance."
            )
        return cls(
            conductance=conductance.fillna(0.0).to_numpy(),
            elevation=dataset["elevation"].fillna(0.0).to_numpy(),
            smoothing_width=smoothing_width,
        )


class River(Boundary):
    """
    River boundary condition.

    Smoothly blends between fixed-rate flux (head at/below bottom_elevation)
    and linear head-dependent flux (head above bottom_elevation), transitioning
    over an interval `smoothing_width` instead of switching discontinuously like
    the classical River package.
    """

    conductance: FloatArray
    stage: FloatArray
    bottom_elevation: FloatArray
    _fixed_rhs: FloatArray
    _cbot: FloatArray
    _weight: FloatArray
    _rhs: FloatArray
    smoothing_width: float

    def __init__(self, conductance, stage, bottom_elevation, smoothing_width=0.01):
        if smoothing_width <= 0.0:
            raise ValueError(
                f"smoothing_width must be positive, got: {smoothing_width}"
            )
        self.conductance = conductance.ravel()
        self.stage = stage.ravel()
        self.bottom_elevation = bottom_elevation.ravel()
        self._fixed_rhs = self.conductance * (self.stage - self.bottom_elevation)
        self._cbot = self.conductance * self.bottom_elevation
        self._weight = np.empty_like(self.conductance)
        self._rhs = np.empty_like(self.conductance)
        self.smoothing_width = smoothing_width

    def formulate(self, hcof, rhs, head):
        np.subtract(head, self.bottom_elevation, out=self._weight)
        _smoothstep_inplace(self._weight, self.smoothing_width, self._rhs)
        np.multiply(self._rhs, self.conductance, out=self._rhs)
        np.subtract(rhs, self._rhs, out=rhs)
        np.multiply(self.conductance, self._weight, out=self._rhs)
        np.add(hcof, self._rhs, out=hcof)
        np.add(rhs, self._fixed_rhs, out=rhs)
        np.multiply(self._cbot, self._weight, out=self._rhs)
        np.add(rhs, self._rhs, out=rhs)
        return

    @classmethod
    def from_dataset(cls, dataset, smoothing_width=0.01):
        return cls(
            conductance=dataset["conductance"].fillna(0.0).to_numpy(),
            stage=dataset["stage"].fillna(0.0).to_numpy(),
            bottom_elevation=dataset["bottom_elevation"].fillna(0.0).to_numpy(),
            smoothing_width=smoothing_width,
        )


class HorizontalFlowBarrier:
    """
    Resistance applied to the connection between two laterally adjacent cells.

    Static in time: the modification to intercell conductances is applied once,
    at model construction.
    """

    layer: int  # 0-based
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
        if self.cell0.size == 0:
            return [], [], []
        if self.cell0.max() >= layer_size or self.cell1.max() >= layer_size:
            raise ValueError("HFB cell index exceeds number of cells in a layer")
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


def _csr_diagonal_indices(A: sparse.csr_matrix) -> IntArray:
    """
    Locate the diagonal entries of a CSR matrix within its data array.

    Assigning through these indices replaces the diagonal in O(n) without the
    structural checks ``setdiag`` performs on every call.

    Raises
    ------
    ValueError
        If any diagonal entry is not stored explicitly.
    """
    n = A.shape[0]
    rows = np.repeat(np.arange(n), np.diff(A.indptr))
    indices = np.flatnonzero(A.indices == rows)
    if indices.size != n:
        raise ValueError(
            f"Matrix is missing {n - indices.size} explicit diagonal entries."
        )
    return indices


class GroundwaterModel(NonlinearIteration):
    """
    Confined groundwater flow model.

    The nonlinear iteration itself lives in :class:`NonlinearIteration`; this
    class supplies the formulation, the linear solve, and the residual. State
    and convergence space coincide here, so :meth:`primary` is the identity.
    """

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
        linear_settings: LinearSettings | None = None,
        nonlinear_settings: NonlinearSettings | None = None,
        symmetric: bool | None = None,
    ):
        """
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
            or shaped ``(nlayer, ny, nx)`` for multi-layer models.
        resistance: np.ndarray of floats, optional
            May be shaped ``(ny, nx)`` for a two layer model;
            or shaped ``(nlayer - 1, ny, nx)`` for more layers.
        storativity: np.ndarray of floats, optional
            May be shaped ``(ny, nx)`` for a single layer model;
            or shaped ``(nlayer, ny, nx)`` for multi-layer models.
        horizontal_flow_barriers: sequence of HorizontalFlowBarrier, optional
            Horizontal flow barriers. These are static in time: the modification
            to intercell conductances is made at model initialization.
        linear_settings: LinearSettings, optional
            Which linear solver to use and how to configure it. Defaults to
            :class:`PCGSettings`.
        nonlinear_settings: NonlinearSettings, optional
            Tolerances, iteration budget and relaxation strategy for the
            nonlinear iteration. Defaults to undamped with the
            :class:`NonlinearSettings` defaults.
        symmetric : bool, optional
            Whether to store only the upper triangle or materialize both halves for
            a general solver. The system is symmetric, but the general treatment may
            be more robust in some cases.
        """
        transmissivity_3d = atleast_3d_front(transmissivity)
        initial_3d = atleast_3d_front(initial)
        if initial_3d.shape != transmissivity_3d.shape:
            raise ValueError("Shapes of transmissivity and initial head do not match.")
        nlayer, ny, nx = transmissivity_3d.shape

        if isinstance(transmissivity, xr.DataArray):
            coords = dict(transmissivity.coords)
            if "layer" not in coords:
                coords["layer"] = np.array([0])  # TODO(?): 0 or 1-based indexing
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
        self._head = self.initial.astype(float, copy=True)
        # Work arrays
        self._head_old = self._head.astype(float, copy=True)
        self._storage_work = np.empty_like(self.storativity, dtype=float)
        self._residual = np.empty_like(self._head, dtype=float)

        # Boundaries that scale with cell area need it before first formulation.
        self.recharge.bind_grid(area)
        for boundary in self.head_boundaries:
            boundary.bind_grid(area)

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
        # Compute the Laplacian. Force explicit storage of the diagonal, since
        # cells with a zero degree would otherwise leave a structural hole that
        # the per-iteration diagonal assignment cannot fill.
        A = sparse.diags(self.D) - self.W
        if symmetric:
            self.A = sparse.triu(A).tocsr()
            self.matrix_type = MatrixType.SYMMETRIC_POSITIVE_DEFINITE
        else:
            self.A = A.tocsr()
            self.matrix_type = MatrixType.NONSYMMETRIC

        self._diag_indices = _csr_diagonal_indices(self.A)
        # The linear solver holds A, rhs and head by reference, so formulate
        # must mutate them in place rather than rebind them.
        self.A.data[self._diag_indices] = self.hcof
        self.linear_settings = (
            linear_settings if linear_settings is not None else PCGSettings()
        )
        self.linearsolver = self.linear_settings.build(
            self.A, self.rhs, self._head, self.matrix_type
        )
        self.nonlinear_settings = (
            nonlinear_settings
            if nonlinear_settings is not None
            else NonlinearSettings()
        )

    @property
    def state(self) -> FloatArray:
        """Live head vector, written in place by the linear solver."""
        return self._head

    @property
    def diagonal(self) -> FloatArray:
        """Diagonal of the assembled system, used to scale residuals."""
        return self.hcof

    def primary(self, vector) -> FloatArray:
        """Head is the whole state here."""
        return vector

    def residual(self) -> FloatArray:
        """
        Residual ``b - A @ h`` for the current head and formulation.

        Computed into a work array that is reused on every call, so the result
        is invalidated by the next one. Assumes the matrix diagonal is already
        synchronised with ``hcof``, which :meth:`formulate` guarantees.
        """
        A = self.A
        # csr_matvec accumulates (y += A @ x), so negate around it rather
        # than allocating a temporary for A @ x.
        np.negative(self.rhs, out=self._residual)
        csr_matvec(*A.shape, A.indptr, A.indices, A.data, self._head, self._residual)
        np.negative(self._residual, out=self._residual)
        return self._residual

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

    def formulate(self, dt=None, recharge=True):
        """
        Assemble the RHS vector and system matrix for the current iteration.

        Resets RHS and diagonal to their base values, then accumulates
        contributions from storage (if ``dt is not None``), recharge, and all
        head boundaries. A ``dt`` of None encodes steady-state behaviour: no
        storage term is added. Finally, writes the diagonal into the matrix, so
        that ``residual`` and ``linear_solve`` both see a consistent system.
        The matrix is mutated in place, which is what lets the linear solver
        hold a reference to it across calls.

        Parameters
        ----------
        dt:
            Time step size. Set to None for steady state.
        recharge:
            Whether to include the recharge boundary condition. The inverse
            problem excludes it, since recharge is a free variable there.
        """
        if dt is not None and dt <= 0.0:
            raise ValueError(f"dt must be positive, got: {dt}")

        # Reset
        self.rhs[:] = 0.0
        self.hcof[:] = self.D[:]

        # Formulate storage
        if dt is not None:
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
            self.recharge.formulate(hcof, rhs, head)
        for boundary in self.head_boundaries:
            boundary.formulate(hcof, rhs, head)

        self.A.data[self._diag_indices] = self.hcof
        return

    def linear_solve(self) -> tuple[bool, int]:
        """
        Solve the currently assembled linear system.

        Updates ``_head`` in-place. The matrix diagonal is set by
        :meth:`formulate`, which the nonlinear iteration calls beforehand.
        Factorization happens here rather than in ``formulate`` because the
        nonlinear loop assembles one system more than it solves: the final
        formulation is only used to evaluate the residual.

        Returns
        -------
        converged: bool
            Whether the solver met the convergence criterion. Direct backends
            always report True.
        iterations: int
            Number of iterations taken.
        """
        self.linearsolver.factorize()
        return self.linearsolver.solve()

    def bind_time(self, time) -> None:
        self.recharge.bind_time(time)
        for boundary in self.head_boundaries:
            boundary.bind_time(time)
        return

    def advance(self, time_index: int) -> None:
        """Roll the head forward and move every boundary to the next step."""
        np.copyto(dst=self._head_old, src=self._head)
        self.recharge.advance(time_index)
        for boundary in self.head_boundaries:
            boundary.advance(time_index)
        return

    @staticmethod
    def _timestep_sizes(time) -> FloatArray:
        """
        Convert a time axis into positive step sizes in days.

        Fractional days are preserved: truncating to whole days silently turns
        sub-daily steps into zeros, which then divide by zero in the storage
        term.
        """
        stamps = np.asarray(time, dtype="datetime64[s]")
        dts = np.diff(stamps).astype("timedelta64[s]").astype(float) / 86400.0
        if dts.size == 0:
            raise ValueError("time must contain at least two stamps.")
        if not (dts > 0.0).all():
            raise ValueError("time must be strictly increasing.")
        return dts

    def run(
        self,
        time,
        path=None,
        steady_state: bool | BoolArray = True,
    ) -> xr.Dataset:
        """
        Run a simulation over a sequence of time steps.

        Resets the head to the initial condition, then advances the solution
        through each interval of ``time`` using the nonlinear iteration.

        Parameters
        ----------
        time:
            Time axis. ``n`` stamps define ``n - 1`` steps; results are written
            against ``time[:-1]``.
        path:
            Destination for the zarr store. A temporary directory is used, and
            cleaned up when the returned dataset is closed, if omitted.
        steady_state:
            Whether each step is steady state. A scalar applies to every step;
            an array must be broadcastable to the number of steps. Steady-state
            steps drop the storage term.

        Returns
        -------
        xr.Dataset
            Head, convergence flag and iteration count per time step.
        """
        tmp = None
        if path is None:
            tmp = tempfile.TemporaryDirectory(prefix="respighi-")
            path = Path(tmp.name) / "gwf-results.zarr"

        nlayer, ny, nx = self.transmissivity.shape
        dts = self._timestep_sizes(time)
        steady = np.broadcast_to(steady_state, len(dts))
        self.bind_time(time)
        np.copyto(dst=self._head, src=self.initial)
        np.copyto(dst=self._head_old, src=self._head)

        with zarr_writer(
            path=path, time=time[:-1], dims=("layer", "y", "x"), coords=self._coords
        ) as group:
            zarr_head = group["head"]
            zarr_converged = group["converged"]
            zarr_iters = group["iterations"]
            for i, dt in enumerate(dts):
                self.advance(i)
                result = self.nonlinear_solve(dt=None if steady[i] else dt)
                zarr_converged[i] = result.converged
                zarr_iters[i] = result.iterations
                zarr_head[i] = self._head.reshape((nlayer, ny, nx))

        ds = xr.open_zarr(path)
        if tmp is not None:
            ds.set_close(tmp.cleanup)
        return ds

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
