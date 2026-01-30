import warnings

import numpy as np
from scipy import sparse

from respighi.cg import PCGSolver
from respighi.constants import FloatArray, BoolArray
from respighi.ilu0 import ILU0Preconditioner
import pypardiso

import xugrid as xu


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


class GroundwaterModel:
    def __init__(
        self,
        area,
        initial,
        recharge,
        head_boundaries,
        transmissivity: FloatArray | None = None,
        xclose_linear: float = 1e-5,
        rclose_linear: float = 1e-5,
        maxiter_linear: int = 100,
        xclose: float = 1e-4,
        maxiter: int = 30,
    ):
        self.initial = initial.ravel()
        self.recharge = recharge
        self.head_boundaries = head_boundaries

        n = self.initial.size
        if transmissivity is None:
            self.transmissivity = np.ones(n)
        else:
            self.transmissivity = transmissivity

        self.area = np.full(n, area)
        self.n = n
        self.rhs = np.zeros(n)
        self.head = np.zeros(n)
        self._head_old = np.empty_like(self.head)
        self._update = np.empty_like(self.head)

        # Matrix assembly
        W = self._build_connectivity(transmissivity)
        # Compute the (weighted) degree matrix
        self.D = np.asarray(W.sum(axis=1)).ravel()
        self.hcof = self.D.copy()
        # Compute the Laplacian
        self.Abase = sparse.diags(self.D) - W
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

    @staticmethod
    def _build_connectivity(transmissivity):
        # Get the Cartesian neighbors for a finite difference approximation.
        # TODO: check order of dimensions with DataArray
        shape = transmissivity.shape
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
        kD = transmissivity.ravel()
        # NOTE:
        # Treat as serial resistances -> harmonic mean
        # We assume dy == dx such that width and lengths cancel.
        kDi = kD[i]
        kDj = kD[j]
        conductance = (2 * kDi * kDj) / (kDi + kDj)
        return sparse.coo_matrix((conductance, (i, j)), shape=(size, size)).tocsr()

    def formulate(self, recharge=True):
        # Reset
        self.rhs[:] = 0.0
        self.hcof[:] = self.D[:]
        # Accumulate boundary conditions
        if recharge:
            self.recharge.formulate(self.rhs, self.area)
        head = self.head
        for boundary in self.head_boundaries:
            boundary.formulate(self.hcof, self.rhs, head)
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
                f"Linear solver did not converge after {iterations} iterations."
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
