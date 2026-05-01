import warnings

import numpy as np
import xarray as xr
from scipy import sparse

from respighi.groundwaterflow import GroundwaterModel
from respighi.pardiso import PardisoWrapper
from respighi.target import FittingTarget


class InverseProblem:
    """
    Inverse problem solver for groundwater model to fit a target head.

    Solves the constrained optimization problem of estimating recharge rates by
    minimizing the misfit between model predictions and observations, subject
    to regularization and the groundwater flow equations.

    The optimization problem minimizes:
        J(h, r) = ½||P·h - d||² + ½·α||L·r||²

    Subject to constraints:
        - A·h - Q·r = b_bc  (groundwater flow equation)
        - P·h = d + e       (observation equation)
        - L·r = s           (regularization equation)

    Where:
        - h: hydraulic head
        - r: recharge rates (parameters to estimate)
        - d: observed head values
        - P: observation operator
        - A: groundwater flow matrix (head-dependent)
        - Q: recharge-to-flux operator
        - L: regularization operator (Laplacian)
        - α: regularization weight

    The problem is solved using the Lagrangian approach with KKT conditions,
    forming a saddle-point system. Nonlinearity from head-dependent conductances
    is handled via Picard iteration.

    Parameters
    ----------
    groundwatermodel : GroundwaterModel
        The groundwater flow model providing system matrices and parameters
    target : FittingTarget
        Observation data and operator (P matrix and d vector)
    regularization_weight : float
        Weight for spatial smoothness regularization (α)
    maxiter : int, optional
        Maximum number of Picard iterations (default: 30)
    maxdh : float, optional
        Convergence tolerance for non-linear head updates (default: 1e-4)
    relax : float, optional
        Relaxation factor for Picard iteration (default: 0.0)

    Attributes
    ----------
    head : ndarray
        Current estimate of hydraulic head
    recharge : ndarray
        Current estimate of recharge rates
    lagrangian : ndarray
        Current estimate of Lagrange multipliers
    """

    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization_weight: float,
        maxiter: int = 30,
        maxdh=1e-4,
        relax=0.0,
    ):
        # Store core attributes
        self.gwf = groundwatermodel
        self.target = target
        self.n = self.gwf.n
        self.layer_n = self.gwf.layer_n
        self.regularization_weight = regularization_weight
        self.maxiter = maxiter
        self.maxdh = maxdh
        self.relax = relax
        self.K = self._build_matrix(regularization_weight)
        self.rhs = self._build_rhs_vector()
        self.x = np.zeros_like(self.rhs)
        self._x_old = np.zeros_like(self.rhs)
        self._x_update = np.zeros_like(self.rhs)
        self._head_old = np.zeros(self.n)
        self._head_update = np.zeros(self.n)
        self.linearsolver = None
        # Extract diagonal indices for efficient Picard updates
        self.At_diag_indices, self.A_diag_indices = self._extract_diagonal_indices()
        self.K.data[self.At_diag_indices] = self.gwf.hcof
        self.K.data[self.A_diag_indices] = self.gwf.hcof
        self.rhs_flow_slice = slice(self.n + self.layer_n, 2 * self.n + self.layer_n)
        self.rhs_obs_slice = slice(
            2 * self.n + self.layer_n, 2 * self.n + self.layer_n + len(target.d)
        )

    def _build_matrix(self, regularization_weight: float) -> sparse.csr_matrix:
        """Build optimality system matrix.

        Optimality conditions:
        ∂L/∂h = P^T μ_e + A^T λ = 0        → P^T (w_obs e) + A^T λ = 0
        ∂L/∂r = L^T μ_s - Q^T λ = 0        → L^T (w_reg s) - Q^T λ = 0
        ∂L/∂e = w_obs e - μ_e = 0          (used to eliminate μ_e)
        ∂L/∂s = w_reg s - μ_s = 0          (used to eliminate μ_s)

        Constraints:
        - A h - Q r = b_bc
        - P h - e = d
        - L r - s = 0

        Block structure: [h, r, e, s, λ]^T
        """
        # Mark diagonals with sentinel for later extraction
        A = self.gwf.A.copy()
        A.setdiag(np.inf)
        At = A.T

        # Single layer is easier ...
        # P = self.target.P
        # Pt = P.T

        P = self.target.P
        if P.shape[1] < self.n:
            padding = sparse.csr_matrix((P.shape[0], self.n - P.shape[1]))
            P = sparse.hstack([P, padding])
        Pt = P.T

        # NOTE:
        # also assumes constant cell sizes, and dx == dy.
        layer_n = self.gwf.layer_n
        ny, nx = self.gwf.transmissivity.shape[1:]
        i, j = GroundwaterModel._build_connectivity((ny, nx))
        W_2d = sparse.coo_matrix(
            (np.ones(len(i)), (i, j)), shape=(layer_n, layer_n)
        ).tocsr()
        D_2d = np.asarray(W_2d.sum(axis=1)).ravel()  # Degree matrix
        L = regularization_weight * (sparse.diags(D_2d) - W_2d)
        Lt = L.T

        # Q = sparse.diags(self.gwf.area)  # Single layer is easier...
        rows = np.arange(self.layer_n)
        area = np.full(self.layer_n, self.gwf.area)
        Q = sparse.coo_matrix(
            (area, (rows, rows)), shape=(self.n, self.layer_n)
        ).tocsr()
        Qt = Q.T

        n_obs = P.shape[0]
        I_e = sparse.eye(n_obs, format="csr")
        I_s = sparse.eye(self.layer_n, format="csr")

        return sparse.block_array(
            [
                # h,     r,      e,      s,      λ
                [None, None, Pt, None, At],
                [None, None, None, Lt, -Qt],
                [A, -Q, None, None, None],
                [P, None, -I_e, None, None],
                [None, L, None, -I_s, None],
            ],
            format="csr",
        )

    def _build_rhs_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(self.n),  # h
                np.zeros(self.layer_n),  # r
                self.gwf.rhs,  # flow equation
                self.target.d,  # observations
                np.zeros(self.layer_n),  # s
            ]
        )

    def _extract_diagonal_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract diagonal indices for efficient Picard iteration updates.
        Returns indices of A and At diagonals within the CSR data array.
        """
        inf_indices = np.where(np.isinf(self.K.data))[0]
        return inf_indices[: self.n], inf_indices[self.n :]

    def _formulate_gwf(self, dt):
        self.gwf.formulate(recharge=False, dt=dt)
        self.K.data[self.At_diag_indices] = self.gwf.hcof
        self.K.data[self.A_diag_indices] = self.gwf.hcof
        self.rhs[self.rhs_flow_slice] = self.gwf.rhs
        return

    def formulate(self, dt=0.0):
        """
        Formulate the system of equations, call PARDISO's analysis (phase 11)
        and numerical factorization (phase 22).
        """
        self._formulate_gwf(dt=dt)
        self.linearsolver = PardisoWrapper(self.K, self.rhs, self.x)
        # Analysis is the most costly phase.
        self.linearsolver.analyze()
        self.linearsolver.factorize()

    def reformulate(self, dt=0.0):
        """
        Formulate the system of equations, call PARDISO's numerical
        factorization; unlike ``.formulate``, this does not call the expensive
        analysis phase.
        """
        # Structure is static, reuse results of analysis.
        self._formulate_gwf(dt=dt)
        self.linearsolver.factorize()

    def update_observations(self, d):
        if d.shape != self.target.d.shape:
            raise ValueError("Observation size changed: rebuild instead.")
        self.rhs[self.rhs_obs_slice] = d

    def linear_solve(self):
        """Solve the linear system for ``[h, r, λ]^T``."""
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")
        self.linearsolver.solve()
        return

    def nonlinear_solve(self):
        """
        Solve the nonlinear system for ``[h, r, λ]^T`` using Picard iteration.

        Call .formulate() first.
        """
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")

        for i in range(self.maxiter):
            np.copyto(dst=self._x_old, src=self.x)
            np.copyto(dst=self._head_old, src=self._head)
            self.linear_solve()
            np.subtract(self._head, self._head_old, out=self._head_update)
            np.subtract(self.x, self._x_old, out=self._x_update)
            maxdh = np.linalg.norm(self._head_update, ord=np.inf)
            print(maxdh)
            if maxdh < self.maxdh:
                return True, i + 1
            self.x -= self.relax * self._x_update
            self.reformulate()

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} iterations. "
            f"Final update: {maxdh:.2e}"
        )
        return False, self.maxiter

    def run(self, dts, targets):
        self.formulate()
        out = []
        for dt, target in zip(dts, targets):
            self.update_observations(target.d)
            self.reformulate(dt=dt)
            self.nonlinear_solve()
            out.append(self._head.copy())
        return out

    @property
    def _head(self):
        """Current estimate of head."""
        return self.x[: self.n]

    @property
    def _recharge(self):
        return self.x[self.n : self.n + self.layer_n]

    @property
    def _lagrangian(self):
        return self.x[-self.layer_n :]

    def _datarray_helper(self, data: np.ndarray):
        return xr.DataArray(
            data, dims=("layer", "y", "x"), coords=self._coords, name="head"
        )

    @property
    def head(self):
        return self._datarray_helper(self._head.reshape(self.gwf.transmissivity.shape))

    @property
    def recharge(self):
        return self._datarray_helper(
            self._recharge.reshape(self.gwf.transmissivity.shape[1:])
        )

    @property
    def lagrangian(self):
        return self._datarray_helper(
            self._lagrangian.reshape(self.gwf.transmissivity.shape[1:])
        )
