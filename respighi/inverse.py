import platform
import warnings
from typing import Literal

import numpy as np
import xarray as xr
from scipy import sparse

from respighi.groundwaterflow import GroundwaterModel
from respighi.linearsolvers.direct import make_direct_solver
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

    The problem is solved using the Lagrangian approach,
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
    """

    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization_weight: float,
        maxiter: int = 30,
        maxdh=1e-4,
        relax=0.0,
        solver_backend: Literal["pardiso", "mumps", "scipy"] | None = None,
    ):
        # On macOS: Default to MUMPS instead of Intel Pardiso
        if solver_backend is None:
            solver_backend = "mumps" if platform.system() == "Darwin" else "pardiso"
        self.solver_backend = solver_backend

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
        self._head_iter = np.zeros(self.n)
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

        P = self.target.P
        if P.shape[1] < self.n:
            padding = sparse.csr_matrix((P.shape[0], self.n - P.shape[1]))
            P = sparse.hstack([P, padding])
        Pt = P.T

        # NOTE:
        # Assumes constant cell sizes, and dx == dy.
        layer_n = self.gwf.layer_n
        ny, nx = self.gwf.transmissivity.shape[1:]
        i, j = GroundwaterModel.build_connectivity((ny, nx))
        W_2d = sparse.coo_matrix(
            (np.ones(len(i)), (i, j)), shape=(layer_n, layer_n)
        ).tocsr()
        D_2d = np.asarray(W_2d.sum(axis=1)).ravel()  # Degree matrix
        L = regularization_weight * (sparse.diags(D_2d) - W_2d)
        Lt = L.T

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
        """
        Build the RHS vector for the full optimality system.

        Concatenates zero vectors for the adjoint equations, the groundwater
        flow RHS (boundary conditions), the observation vector, and the
        regularization RHS.
        """
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
        """
        Extract the CSR data indices of the A^T and A diagonals for efficient
        Picard updates.

        During ``_build_matrix``, the diagonals of both A and A^T are set to
        ``inf`` as sentinels. This method locates those entries in the CSR data
        array so that ``_formulate_gwf`` can patch them in-place without
        rebuilding the matrix. The first ``n`` inf entries correspond to A^T
        (upper block) and the second ``n`` to A (lower block), reflecting their
        order in the block structure.

        Returns
        -------
        at_indices: np.ndarray
            CSR data indices of the A^T diagonal entries.
        a_indices: np.ndarray
            CSR data indices of the A diagonal entries.
        """
        inf_indices = np.where(np.isinf(self.K.data))[0]
        return inf_indices[: self.n], inf_indices[self.n :]

    def _formulate_gwf(self, dt):
        """
        Update the groundwater flow contributions in the optimality system.

        Calls ``GroundwaterModel.formulate`` (without recharge, as recharge is
        a free variable here), then patches the diagonal entries of ``A`` and
        ``A^T`` in the block matrix and updates the flow equation RHS slice.
        """
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
        self.linearsolver = make_direct_solver(
            self.solver_backend, self.K, self.rhs, self.x
        )
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
        """
        Replace the observation vector in the RHS.

        Useful for transient runs where observations change between time steps
        without requiring a full rebuild of the system.

        Parameters
        ----------
        d:
            New observation vector. Must have the same shape as the original.
        """
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

        At each iteration, the linear system is solved and the head update is
        checked for convergence. Convergence is assessed on head only —
        specifically the infinity norm of the head change between iterations —
        rather than on the full solution vector, since head is the physically
        meaningful quantity and recharge and Lagrange multipliers are derived
        from it. A relaxation factor (``relax``) can be applied to damp
        oscillations if the iteration is slow to converge.

        Call ``.formulate()`` before calling this method.

        Parameters
        ----------
        (none)

        Returns
        -------
        converged: bool
            Whether the head update fell below ``maxdh``.
        iterations: int
            Number of iterations taken.
        """

        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")

        for i in range(self.maxiter):
            np.copyto(dst=self._x_old, src=self.x)
            np.copyto(dst=self._head_iter, src=self._head)
            self.linear_solve()
            np.subtract(self._head, self._head_iter, out=self._head_update)
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

    def run(self, dts, targets, callback=None):
        """
        Run a transient or batched inverse solve over a sequence of time steps.

        Performs the expensive PARDISO analysis once, then iterates over time
        steps, updating observations and refactorizing at each step.

        The optional ``callback`` is invoked before each step, allowing,
        boundary conditions, or other model state to be updated in-place.

        Parameters
        ----------
        dts:
            Sequence of time step sizes.
        targets:
            Sequence of FittingTarget objects, one per time step.
        callback:
            Optional callable with signature ``callback(problem, i, dt)``,
            where ``problem`` is the ``InverseProblem`` instance, ``i`` is
            the zero-based step index, and ``dt`` is the current time step
            size. Called at the start of each step, before refactorization.

        Returns
        -------
        list of np.ndarray
            Flat head arrays (length ``n``) after each time step.
        """
        np.copyto(dst=self._head, src=self.gwf.initial)
        self.formulate()
        out = []
        for i, (dt, target) in enumerate(zip(dts, targets)):
            if callback is not None:
                callback(self, i, dt)
            self.update_observations(target.d)
            # Copy head to gwf._head_old for storage term formulation.
            np.copyto(dst=self.gwf._head_old, src=self._head)
            self.reformulate(dt=dt)
            self.nonlinear_solve()
            out.append(self._head.copy())
        return out

    @property
    def _head(self):
        """Current head estimate; the first ``n`` entries of the solution vector."""
        return self.x[: self.n]

    @property
    def _recharge(self):
        """Current recharge estimate; entries ``n`` to ``n + layer_n`` of the solution vector."""
        return self.x[self.n : self.n + self.layer_n]

    @property
    def _lagrangian(self):
        """Current Lagrange multipliers; the final ``layer_n`` entries of the solution vector."""
        return self.x[-self.layer_n :]

    @property
    def head(self):
        """Head estimate as a labelled DataArray of shape ``(layer, y, x)``."""
        return xr.DataArray(
            data=self._head.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="head",
        )

    @property
    def recharge(self):
        """Recharge estimate as a labelled DataArray of shape ``(y, x)``."""
        return xr.DataArray(
            data=self._recharge.reshape(self.gwf.transmissivity.shape[1:]),
            dims=("y", "x"),
            coords={"y": self.gwf._coords["y"], "x": self.gwf._coords["x"]},
            name="recharge",
        )

    @property
    def lagrangian(self):
        """Lagrange multipliers as a labelled DataArray of shape ``(y, x)``."""
        return xr.DataArray(
            self._lagrangian.reshape(self.gwf.transmissivity.shape[1:]),
            dims=("y", "x"),
            coords={"y": self.gwf._coords["y"], "x": self.gwf._coords["x"]},
            name="lagrangian",
        )

    def observation_influence_functions(
        self,
        batch_size: int | None = None,
    ):
        """
        Estimate head variance contribution from observation uncertainty.

        For each observation i, computes the influence function

            phi_i = d h / d d_i

        by solving the already-factorized system with a unit perturbation in
        the observation RHS row. The variance contribution is

            v_h = sum_i sigma_i^2 * phi_i^2

        """
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before influence estimation")

        n_obs = len(self.target.d)
        N = len(self.rhs)
        obs_rows = np.arange(self.rhs_obs_slice.start, self.rhs_obs_slice.stop)
        if batch_size is None:
            batch_size = n_obs

        Phi = np.zeros((self.n, n_obs))
        for start in range(0, n_obs, batch_size):
            stop = min(start + batch_size, n_obs)
            m = stop - start
            B = np.zeros((N, m))
            B[obs_rows[start:stop], np.arange(m)] = 1.0
            X = self.linearsolver.solve_multi(B)
            Phi[:, start:stop] = X[: self.n, :]

        return Phi

    def boundary_influence_functions(self):
        """
        Compute head influence functions for all head boundaries.

        Column k gives psi_k = dh / d delta_k, the sensitivity of the head
        field to a conductance-and-sigma-weighted coherent shift of boundary k.
        """
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before influence estimation")

        N = len(self.rhs)
        n_boundaries = len(self.gwf.head_boundaries)
        flow_start = self.rhs_flow_slice.start

        B = np.zeros((N, n_boundaries))
        for k, boundary in enumerate(self.gwf.head_boundaries):
            B[flow_start : flow_start + self.n, k] = (
                boundary.conductance.ravel() * boundary.sigma.ravel()
            )

        X = self.linearsolver.solve_multi(B)
        return X[: self.n, :]

    def estimate_variance(self, batch_size: int | None = None):
        r"""
        Estimate head variance from observation and boundary uncertainty.

        Combines two sources of uncertainty via first-order linear error
        propagation through the factorized system:

        - Observation uncertainty: each piezometer contributes a variance
        proportional to ``target.sigma[i]**2``, weighted by its influence
        function ``phi_i = dh / dd_i``.
        - Boundary uncertainty: each head boundary contributes a variance
        from a spatially coherent shift, weighted by the conductance and
        ``boundary.sigma`` fields, expressed as ``psi_k = dh / d delta_k``.

        The total variance is:

        .. math::

            \text{Var}(\\mathbf{h}) =
            \sum_i \sigma_i^2 \, \\boldsymbol{\phi}_i^2
            + \sum_k \boldsymbol{\psi}_k^2

        where :math:`\sigma_i` is already absorbed into :math:`\boldsymbol{\psi}_k`
        via the conductance-sigma weighting in
        :meth:`boundary_influence_functions`.

        Observation and boundary errors are assumed mutually independent, so
        their variance contributions add. All influence functions are computed
        via multi-RHS solves reusing the existing PARDISO or MUMPS factorization; no
        additional factorization is required.

        .. note::

            This is a first-order estimate, linearized around the converged
            head solution. It captures uncertainty due to observation noise
            and boundary condition uncertainty, but not structural model error
            (e.g. transmissivity uncertainty, incorrect boundary placement).
            The spatial pattern is therefore more reliable than the absolute
            magnitudes, which depend on the physical calibration of
            ``target.sigma`` and ``boundary.sigma``.

        Parameters
        ----------
        batch_size : int, optional
            Number of observation influence functions to solve simultaneously.
            If None, all observations are solved in a single multi-RHS call.
            Reduce this if memory is a concern for large observation sets.

        Returns
        -------
        variance : xr.DataArray of shape (layer, y, x)
            Pointwise head variance in units of head squared (m²), with the
            same grid coordinates as the groundwater model.
        """
        sigma_obs = self.target.sigma
        Phi_obs = self.observation_influence_functions(batch_size=batch_size)
        Phi_bc = self.boundary_influence_functions()
        var = np.sum((Phi_obs * sigma_obs) ** 2, axis=1)
        var += np.sum(Phi_bc**2, axis=1)
        return xr.DataArray(
            data=var.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="variance",
        )
