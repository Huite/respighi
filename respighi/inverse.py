import platform
import tempfile
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from scipy import sparse

from respighi.constants import BoolArray
from respighi.groundwaterflow import GroundwaterModel
from respighi.linearsolvers.direct import MumpsWrapper, make_direct_solver
from respighi.linearsolvers.solvertypes import MatrixType
from respighi.output import zarr_writer
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
        Relaxation factor (0-1]. A value of one indicates no relaxation.
    solver_backend: str, optional: "pardiso", "mumps", "scipy".
        Which linear solver to use.
    explicit_residuals: bool, optional
        Represent observation and regularization residuals as explicit unknowns.
        This avoids forming ``P.T @ P`` and ``L.T @ L`` and can preserve sparsity when
        observations represent averages or coarse-model values.
    symmetric: bool, optional
        Whether to store only the upper triangle or materialize both halves for a general solver.
        the system is symmetric, but the general treatment may be more robust in some cases.
    """

    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization: float,
        maxiter: int = 30,
        maxdh=1e-4,
        relax=1.0,
        solver_backend: Literal["pardiso", "mumps", "scipy"] | None = None,
        explicit_residuals: bool = False,
        symmetric: bool = True,
    ):
        # On macOS: Default to MUMPS instead of Intel Pardiso
        if solver_backend is None:
            solver_backend = "mumps" if platform.system() == "Darwin" else "pardiso"
        self.solver_backend = solver_backend
        self.explicit_residuals = explicit_residuals
        self.symmetric = symmetric

        # Store core attributes
        self.gwf = groundwatermodel
        self.target = target
        self.n = self.gwf.n
        self.layer_n = self.gwf.layer_n
        self.regularization_weight = regularization
        self.maxiter = maxiter
        self.maxdh = maxdh
        self.relax = relax
        self.K, self.Pt, self.matrix_type = self._build_matrix(regularization)
        self.rhs = self._build_rhs_vector()
        self.x = np.zeros_like(self.rhs)
        self._x_old = np.zeros_like(self.rhs)
        self._x_update = np.zeros_like(self.rhs)
        self._head_iter = np.zeros(self.n)
        self._head_update = np.zeros(self.n)
        self.linearsolver = None

        # Extract diagonal indices for efficient Picard updates
        self._A_diag_indices = self._extract_diagonal_indices()
        self.K.data[self._A_diag_indices] = self.gwf.hcof

        if explicit_residuals:
            obs_start = self.n + self.layer_n
            n_obs_rhs = len(self.target.d)
            flow_start = obs_start + n_obs_rhs + self.layer_n
        else:
            obs_start = 0
            n_obs_rhs = self.n
            flow_start = self.n + self.layer_n

        self.rhs_obs_slice = slice(obs_start, obs_start + n_obs_rhs)
        self.rhs_flow_slice = slice(flow_start, flow_start + self.n)

    def _build_matrix(self, regularization) -> sparse.csr_matrix:
        """Build optimality system matrix.

        Optimality conditions:
        ∂L/∂h = P^T μ_e + A^T λ = 0        → P^T (w_obs e) + A^T λ = 0
        ∂L/∂r = L^T μ_s - Q^T λ = 0        → L^T (w_reg s) - Q^T λ = 0
        ∂L/∂e = e - μ_e = 0          (used to eliminate μ_e)
        ∂L/∂s = s - μ_s = 0          (used to eliminate μ_s)

        Constraints:
        - A h - Q r = b_bc
        - P h - e = d
        - L r - s = 0
        """
        # Mark diagonals with sentinel for later extraction
        A = self.gwf.A.copy()
        A.setdiag(np.inf)

        P = self.target.P
        Pt = P.T
        if P.shape[1] < self.n:
            padding = sparse.csr_matrix((P.shape[0], self.n - P.shape[1]))
            P = sparse.hstack([P, padding])

        # NOTE:
        # Assumes constant cell sizes, and dx == dy.
        ny, nx = self.gwf.transmissivity.shape[1:]
        L = regularization.build_tikhonov_operator(
            ny=ny, nx=nx, dx=np.sqrt(self.gwf.area)
        )

        rows = np.arange(self.layer_n)
        area = np.full(self.layer_n, self.gwf.area)
        Q = sparse.coo_matrix(
            (area, (rows, rows)), shape=(self.n, self.layer_n)
        ).tocsr()

        Z_n = sparse.csr_array((self.n, self.n))
        if self.explicit_residuals:
            n_obs = P.shape[0]
            Z_layer = sparse.csr_array((self.layer_n, self.layer_n))
            I_e = sparse.eye_array(n_obs, format="csr")
            I_s = sparse.eye_array(self.layer_n, format="csr")
            blocks = [
                # Zero diagonal blocks are needed: without them block_array
                # cannot infer the h and r block-column widths.
                [Z_n, None, Pt, None, A.T],
                [None, Z_layer, None, L.T, -Q.T],
                [None, None, -I_e, None, None],
                [None, None, None, -I_s, None],
                [None, None, None, None, Z_n],
            ]
        else:
            blocks = [
                [Pt @ P, None, A.T],
                [None, L.T @ L, -Q.T],
                [None, None, Z_n],
            ]

        Kupper = sparse.triu(sparse.block_array(blocks))
        if self.symmetric:
            # Symmetric solvers need an explicitly stored diagonal,
            # including structurally zero diagonal entries.
            K = Kupper
            K.setdiag(Kupper.diagonal())
            matrix_type = MatrixType.SYMMETRIC_INDEFINITE
        else:
            # Reconstruct the complete symmetric matrix without
            # duplicating the diagonal.
            K = Kupper + sparse.triu(Kupper, k=1).T
            matrix_type = MatrixType.NONSYMMETRIC

        K = K.tocsr()
        return K, Pt, matrix_type

    def _build_rhs_vector(self) -> np.ndarray:
        """
        Build the RHS vector for the full optimality system.

        Concatenates zero vectors for the adjoint equations, the groundwater
        flow RHS (boundary conditions), the observation vector, and the
        regularization RHS.
        """
        if self.explicit_residuals:
            rhs = np.concatenate(
                [
                    np.zeros(self.n),  # stationarity h
                    np.zeros(self.layer_n),  # stationarity r
                    self.target.d,  # observation constraint
                    np.zeros(self.layer_n),  # regularization constraint
                    self.gwf.rhs,  # flow constraint
                ]
            )
        else:
            rhs = np.concatenate(
                [
                    self.Pt @ self.target.d,  # h equation
                    np.zeros(self.layer_n),  # r equation
                    self.gwf.rhs,  # flow constraint
                ]
            )
        return rhs

    def _extract_diagonal_indices(self) -> np.ndarray:
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
        indices = np.flatnonzero(np.isinf(self.K.data))
        if indices.size not in (self.n, 2 * self.n):
            raise RuntimeError(
                f"Expected {self.n} or {2 * self.n} groundwater diagonal "
                f"sentinels, found {indices.size}."
            )
        return indices.reshape(-1, self.n)

    def _formulate_gwf(self, dt):
        """
        Update the groundwater flow contributions in the optimality system.

        Calls ``GroundwaterModel.formulate`` (without recharge, as recharge is
        a free variable here), then patches the diagonal entries of ``A`` and
        ``A^T`` in the block matrix and updates the flow equation RHS slice.
        """
        np.copyto(self.gwf._head, self._head)
        self.gwf.formulate(recharge=False, dt=dt)
        self.K.data[self._A_diag_indices] = self.gwf.hcof
        self.rhs[self.rhs_flow_slice] = self.gwf.rhs
        return

    def formulate(self, dt=None):
        """
        Formulate the system of equations, call PARDISO's analysis (phase 11)
        and numerical factorization (phase 22).
        """
        self._formulate_gwf(dt=dt)
        self.linearsolver = make_direct_solver(
            self.solver_backend, self.K, self.rhs, self.x, matrix_type=self.matrix_type
        )
        # Analysis is the most costly phase.
        self.linearsolver.analyze()
        self.linearsolver.factorize()

    def reformulate(self, dt=None):
        """
        Formulate the system of equations, call PARDISO's numerical
        factorization; unlike ``.formulate``, this does not call the expensive
        analysis phase.
        """
        # Structure is static, reuse results of analysis.
        self._formulate_gwf(dt=dt)
        self.linearsolver.factorize()

    def linear_solve(self):
        """Solve the linear system for ``[h, r, λ]^T``."""
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")
        self.linearsolver.solve()
        return

    def nonlinear_solve(self, dt=None):
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

        maxdh = np.inf
        for i in range(self.maxiter):
            self.reformulate(dt=dt)
            np.copyto(dst=self._x_old, src=self.x)
            np.copyto(dst=self._head_iter, src=self._head)
            self.linear_solve()
            np.subtract(self._head, self._head_iter, out=self._head_update)
            np.subtract(self.x, self._x_old, out=self._x_update)
            maxdh = np.linalg.norm(self._head_update, ord=np.inf)
            print(maxdh)
            if (i > 0) and (maxdh < self.maxdh):
                return True, i + 1
            if self.relax != 1.0:
                self._x_update *= self.relax
                np.add(self._x_old, self._x_update, out=self.x)

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} iterations. "
            f"Final update: {maxdh:.2e}"
        )
        return False, self.maxiter

    def advance(self, time_index: int):
        self.gwf.advance(time_index)
        self.target.advance(time_index)
        # Now copy over the observations from the target to the rhs.
        if self.explicit_residuals:
            self.rhs[self.rhs_obs_slice] = self.target.d
        else:
            self.rhs[self.rhs_obs_slice] = self.Pt @ self.target.d

    def run(
        self,
        time,
        path=None,
        steady_state: bool | BoolArray = True,
    ) -> xr.Dataset:
        """
        Run a transient or batched inverse solve over a sequence of time steps.

        Re-uses the analysis phase of the linear solver, and iterates over time
        steps, updating observations and refactorizing at each step.

        Parameters
        ----------
        time:
        path:
        steady_state: bool or array of bools

        Returns
        -------
        xarray.Dataset
        """
        tmp = None
        if path is None:
            tmp = tempfile.TemporaryDirectory(prefix="respighi-")
            path = Path(tmp.name) / "inverse-results.zarr"

        nlayer, ny, nx = self.gwf.transmissivity.shape
        dts = time.diff().days[1:]
        steady = np.broadcast_to(steady_state, len(dts))
        self.gwf.bind_time(time)
        self.target.bind_time(time)
        np.copyto(dst=self._head, src=self.gwf.initial)
        self.formulate(dt=None)

        with zarr_writer(
            path=path, time=time[:-1], dims=("layer", "y", "x"), coords=self.gwf._coords
        ) as group:
            zarr_head = group["head"]
            zarr_recharge = group["recharge"]
            zarr_converged = group["converged"]
            zarr_iterations = group["iterations"]
            for i, dt in enumerate(dts):
                self.advance(i)
                zarr_converged[i], zarr_iterations[i] = self.nonlinear_solve(
                    dt=None if steady[i] else dt
                )
                zarr_head[i] = self._head.reshape((nlayer, ny, nx))
                zarr_recharge[i] = self._recharge.reshape((ny, nx))

        ds = xr.open_zarr(path)
        if tmp is not None:
            ds.set_close(tmp.cleanup)
        return ds

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
        return self.x[-self.n :]

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

    def observation_mapping_matrix(self) -> np.ndarray:
        r"""
        Compute the local linear mapping from observations to reconstructed heads.

        For the non-explicit residual formulation,

        .. math::

            K
            \begin{bmatrix}
                h \\ r \\ \lambda
            \end{bmatrix}
            =
            \begin{bmatrix}
                P^T d \\ 0 \\ b_{\mathrm{bc}}
            \end{bmatrix}.

        Holding the KKT matrix fixed at the current, typically converged, Picard
        state gives

        .. math::

            \delta h = W \, \delta d.

        Column ``i`` of ``W`` is therefore the reconstructed head response to a
        unit perturbation of observation ``i``.

        Returns
        -------
        W : np.ndarray of shape (n_head, n_obs)
            Local linear mapping from observation perturbations to head
            perturbations.
        """
        if self.linearsolver is None:
            raise RuntimeError(
                "Must call formulate() before computing the observation mapping"
            )

        if self.explicit_residuals:
            raise RuntimeError(
                "observation_mapping_matrix() assumes the non-explicit "
                "residual formulation"
            )

        n_obs = self.Pt.shape[1]
        N = len(self.rhs)
        B = np.zeros((N, n_obs), dtype=float)
        # The observation-dependent part of the KKT RHS is P.T @ d.
        B[: self.n, :] = self.Pt.toarray()
        X = self.linearsolver.solve_multi(B)
        # The first block of the KKT solution is h.
        return X[: self.n, :]

    def observation_surrogate(self) -> xr.Dataset:
        r"""
        Build the local linear observation-to-head surrogate.

        The surrogate is linearized around the current head estimate and
        observation vector:

        .. math::

            h(d) \approx h_{ref} + W (d - d_{ref}).

        Returns
        -------
        xr.Dataset
            Dataset containing:

            - ``head_reference``: reference head field, with dimensions
                ``(layer, y, x)``.
            - ``observation_reference``: reference observation values, with
                dimension ``(observation,)``.
            - ``W``: observation-to-head mapping, with dimensions
                ``(layer, y, x, observation)``.
        """
        W = self.observation_mapping_matrix()
        n_obs = len(self.target.d)
        head_shape = self.gwf.transmissivity.shape
        return xr.Dataset(
            data_vars={
                "head_reference": self.head,
                "observation_reference": (
                    "observation",
                    np.asarray(self.target.d).copy(),
                ),
                "weights": (
                    ("layer", "y", "x", "observation"),
                    W.reshape(*head_shape, n_obs),
                ),
            },
            coords={
                **self.gwf._coords,
                "observation": np.arange(n_obs),
            },
        )

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

    def estimate_variance(self):
        # Only mumps supports inverted entries properly.
        if not isinstance(self.linearsolver, MumpsWrapper):
            linearsolver = make_direct_solver(
                "mumps", self.K, self.rhs, self.x, matrix_type=self.matrix_type
            )
            linearsolver.analyze()
            linearsolver.factorize()
        else:
            linearsolver = self.linearsolver

        variance = linearsolver.inverse_diagonal(indices=np.arange(self.n))
        return xr.DataArray(
            data=variance.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="variance",
        )
