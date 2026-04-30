import warnings

import numpy as np
import pypardiso
from scipy import sparse

from respighi.cg import PCGSolver
from respighi.constants import BoolArray, FloatArray
from respighi.ilu0 import ILU0Preconditioner


class Recharge:
    rate: FloatArray
    _rhs: FloatArray

    def __init__(self, rate):
        self.rate = rate.ravel()
        self._rhs = np.empty_like(self.rate)

    def formulate(self, rhs, area):
        np.multiply(area, self.rate, out=self._rhs)
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


class River:
    conductance: FloatArray
    stage: FloatArray
    elevation: FloatArray
    _fixed_rhs: FloatArray
    _rhs: FloatArray
    _fixed: BoolArray
    _linear: BoolArray

    def __init__(self, conductance, stage, elevation):
        self.conductance = conductance.ravel()
        self.stage = stage.ravel()
        self.elevation = elevation.ravel()
        self._fixed_rhs = self.conductance * (self.stage - self.elevation)
        self._rhs = np.empty_like(self.conductance)
        self._fixed = np.empty(self.conductance.shape, dtype=bool)
        self._linear = np.empty(self.conductance.shape, dtype=bool)

    def formulate(self, hcof, rhs, head):
        # Fixed rate if head < bottom elevation, linear otherwise.
        np.less(head, self.elevation, out=self._fixed)
        np.logical_not(self._fixed, out=self._linear)
        # Fixed case: no hcof contribution, rhs += conductance * (stage - elevation)
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
    return a


class GroundwaterModel:
    def __init__(
        self,
        area,
        initial,
        recharge,
        head_boundaries,
        transmissivity: FloatArray,
        resistance: FloatArray | None = None,
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
        resistance
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

        self.initial = initial.ravel()
        self.recharge = recharge
        self.head_boundaries = head_boundaries

        n = self.initial.size
        self.layer_n = ny * nx
        self.transmissivity = transmissivity_3d
        self.resistance = resistance_3d
        self.area = np.full(self.layer_n, area)
        self.n = n
        self.rhs = np.zeros(n)
        self.head = np.zeros(n)
        self._head_old = np.empty_like(self.head)
        self._update = np.empty_like(self.head)

        # Matrix assembly
        self.W = self._build_conductance(transmissivity_3d, resistance_3d, area)
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
    def _build_conductance(cls, transmissivity, resistance, area):
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
        conductance[horizontal] = (2 * kDi * kDj) / (kDi + kDj)

        if not horizontal.all():
            vertical = ~horizontal
            i_upper = np.minimum(i[vertical], j[vertical])
            conductance[vertical] = area / c[i_upper]

        return sparse.coo_matrix((conductance, (i, j)), shape=(size, size)).tocsr()

    def formulate(self, recharge=True):
        # Reset
        self.rhs[:] = 0.0
        self.hcof[:] = self.D[:]

        # Touch only the first layer
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

    def nonlinear_solve(self):
        """Solve nonlinear system using Picard iteration"""
        # Initialize with current solution or initial guess
        np.copyto(self.head, self.initial)

        for i in range(self.maxiter):
            np.copyto(self._head_old, self.head)
            self.formulate()
            converged_linear, iterations_linear = self.linear_solve(warn=False)
            # self.direct_linear_solve()
            np.subtract(self.head, self._head_old, out=self._update)
            maxdx = np.linalg.norm(self._update, ord=np.inf)
            print(maxdx)
            if maxdx < self.xclose:
                return True, i + 1

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} iterations. "
            f"Final update: {maxdx:.2e}"
        )
        return False, self.maxiter
