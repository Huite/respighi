import platform
import tempfile
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse

from respighi.constants import BoolArray, FloatArray
from respighi.groundwaterflow import GroundwaterModel
from respighi.linearsolvers.mumps import MumpsWrapper
from respighi.linearsolvers.settings import (
    LinearSettings,
    MumpsSettings,
    PardisoSettings,
)
from respighi.linearsolvers.solvertypes import MatrixType
from respighi.nonlinear import NonlinearIteration, NonlinearSettings
from respighi.output import zarr_writer
from respighi.relaxation import AitkenRelaxation
from respighi.target import FittingTarget


class InverseProblem(NonlinearIteration):
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

    The problem is solved using the Lagrangian approach, forming a saddle-point
    system. Nonlinearity from head-dependent conductances is handled by the
    iteration in :class:`NonlinearIteration`.

    Parameters
    ----------
    groundwatermodel : GroundwaterModel
        The groundwater flow model providing system matrices and parameters.
    target : FittingTarget
        Observation data and operator (P matrix and d vector).
    regularization :
        Object providing ``build_tikhonov_operator``, which supplies the
        spatial smoothness operator L.
    linear_settings : LinearSettings, optional
        Which linear solver to use. The KKT system is symmetric indefinite, so
        this must be a direct backend; PCG will refuse it. Defaults to the
        platform's preferred direct backend.
    nonlinear_settings : NonlinearSettings, optional
        Tolerances, iteration budget and relaxation strategy. An
        :class:`AitkenRelaxation` here must be sized to the head block
        (``groundwatermodel.n``), not to the full state.
    explicit_residuals : bool, optional
        Represent observation and regularization residuals as explicit unknowns.
        This avoids forming ``P.T @ P`` and ``L.T @ L`` and can preserve sparsity
        when observations represent averages or coarse-model values.
    symmetric : bool, optional
        Whether to store only the upper triangle or materialize both halves for
        a general solver. The system is symmetric, but the general treatment may
        be more robust in some cases.
    """

    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization,
        linear_settings: LinearSettings | None = None,
        nonlinear_settings: NonlinearSettings | None = None,
        explicit_residuals: bool = False,
        symmetric: bool | None = None,
    ):
        self.explicit_residuals = explicit_residuals

        # The sparsity structure is static, so symbolic analysis happens once,
        # inside build. Numeric factorization is deferred to linear_solve.
        if linear_settings is None:
            if platform.system == "Darwin":
                self.linear_settings = MumpsSettings()
            else:
                self.linear_settings = PardisoSettings()
        else:
            self.linear_settings = linear_settings

        if symmetric is None:
            # PARDISO is not very reliable with the symmetric form.
            if isinstance(self.linear_settings, PardisoSettings):
                self.symmetric = False
            else:
                self.symmetric = True
        else:
            self.symmetric = symmetric

        self.gwf = groundwatermodel
        self.target = target
        self.n = self.gwf.n
        self.layer_n = self.gwf.layer_n
        self.regularization = regularization

        self.K, self.Pt, self.matrix_type = self._build_matrix(regularization)
        self.rhs = self._build_rhs_vector()
        self.x = np.zeros_like(self.rhs, dtype=float)
        self._flow_residual = np.empty(self.n, dtype=float)

        # Extract diagonal indices for efficient updates
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

        self.linearsolver = self.linear_settings.build(
            self.K, self.rhs, self.x, self.matrix_type
        )
        if nonlinear_settings is None:
            # Default to AitkenRelaxation for robustness
            self.nonlinear_settings = NonlinearSettings(
                relaxation=AitkenRelaxation(n=self.gwf.n)
            )
        else:
            self.nonlinear_settings = nonlinear_settings

    def _build_matrix(
        self, regularization
    ) -> tuple[sparse.csr_matrix, sparse.spmatrix, MatrixType]:
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

        Returns the matrix, the observation operator transpose, and the matrix
        type for the linear solver.
        """
        # Mark diagonals with sentinel for later extraction
        A = self.gwf.A.copy()
        A.setdiag(np.inf)

        P = self.target.P
        if P.shape[1] < self.n:
            padding = sparse.csr_matrix((P.shape[0], self.n - P.shape[1]))
            P = sparse.hstack([P, padding]).tocsr()
        # Transposed after padding: taking it beforehand leaves an operator with
        # too few rows, which silently misshapes both P^T P and the RHS.
        Pt = P.T.tocsr()

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
        updates.

        During ``_build_matrix``, the diagonals of both A and A^T are set to
        ``inf`` as sentinels. This method locates those entries in the CSR data
        array so that :meth:`formulate` can patch them in-place without
        rebuilding the matrix. The first ``n`` inf entries correspond to A^T
        (upper block) and the second ``n`` to A (lower block), reflecting their
        order in the block structure. Only the upper triangle is stored in the
        symmetric case, hence the two admissible counts.

        Returns
        -------
        indices: np.ndarray of shape (1, n) or (2, n)
            CSR data indices of the groundwater diagonal entries.
        """
        indices = np.flatnonzero(np.isinf(self.K.data))
        if indices.size not in (self.n, 2 * self.n):
            raise RuntimeError(
                f"Expected {self.n} or {2 * self.n} groundwater diagonal "
                f"sentinels, found {indices.size}."
            )
        return indices.reshape(-1, self.n)

    @property
    def state(self) -> FloatArray:
        """Full solution vector, written in place by the linear solver."""
        return self.x

    @property
    def diagonal(self) -> FloatArray:
        """Groundwater diagonal, used to scale the flow residual."""
        return self.gwf.hcof

    def primary(self, vector: FloatArray) -> FloatArray:
        """Select the head block, on which convergence is judged."""
        return vector[: self.n]

    def formulate(self, dt=None) -> None:
        """
        Update the groundwater flow contributions in the optimality system.

        Calls ``GroundwaterModel.formulate`` without recharge, since recharge
        is a free variable here, then patches the diagonal entries of ``A`` and
        ``A^T`` in the block matrix and updates the flow equation RHS slice.

        Parameters
        ----------
        dt:
            Time step size. Set to None for steady state.
        """
        np.copyto(dst=self.gwf._head, src=self._head)
        self.gwf.formulate(dt=dt, recharge=False)
        self.K.data[self._A_diag_indices] = self.gwf.hcof
        self.rhs[self.rhs_flow_slice] = self.gwf.rhs
        return

    def linear_solve(self) -> tuple[bool, int]:
        """Solve the KKT system for ``[h, r, lambda]``, in place."""
        self.linearsolver.factorize()
        return self.linearsolver.solve()

    def residual(self) -> FloatArray:
        """
        Residual of ``A h - Q r = b_bc``, including the recharge coupling.

        Head-sized, matching the convergence subspace. Computed into a work
        array reused on every call, so the result is invalidated by the next.
        """
        np.multiply(
            self.gwf.area, self._recharge, out=self._flow_residual[: self.layer_n]
        )
        self._flow_residual[self.layer_n :] = 0.0
        self._flow_residual += self.gwf.residual()
        return self._flow_residual

    def advance(self, time_index: int) -> None:
        """Move the model and the observations to the next time step."""
        # Sync head into the flow model first: gwf.advance rolls its own head
        # into head_old, and its copy is only refreshed during formulate.
        np.copyto(dst=self.gwf._head, src=self._head)
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
        progress: bool = False,
    ) -> xr.Dataset:
        """
        Run an inverse solve over a sequence of time steps.

        Reuses the solver's symbolic analysis across steps, updating
        observations and refactorizing at each one.

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
            an array must be broadcastable to the number of steps.
        progress:
            Whether to print the current time step and number of non-linear
            iterations.

        Returns
        -------
        xr.Dataset
            Head, recharge, convergence flag and iteration count per step.
        """
        tmp = None
        if path is None:
            tmp = tempfile.TemporaryDirectory(prefix="respighi-")
            path = Path(tmp.name) / "inverse-results.zarr"

        nlayer, ny, nx = self.gwf.transmissivity.shape
        dts = self.gwf._timestep_sizes(time)
        steady = np.broadcast_to(steady_state, len(dts))
        self.gwf.bind_time(time)
        self.target.bind_time(time)
        np.copyto(dst=self._head, src=self.gwf.initial)

        with zarr_writer(
            path=path, time=time[:-1], dims=("layer", "y", "x"), coords=self.gwf._coords
        ) as group:
            zarr_head = group["head"]
            zarr_recharge = group["recharge"]
            zarr_converged = group["converged"]
            zarr_iterations = group["iterations"]

            if progress:
                print("Time stepping:")

            for i, dt in enumerate(dts):
                self.advance(i)
                result = self.nonlinear_solve(dt=None if steady[i] else dt)
                zarr_converged[i] = result.converged
                zarr_iterations[i] = result.iterations
                zarr_head[i] = self._head.reshape((nlayer, ny, nx))
                zarr_recharge[i] = self._recharge.reshape((ny, nx))

                if progress:
                    print(
                        f"   - Finished timestep {i + 1}/{dts.size} in {result.iterations} non-linear iterations"
                    )

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
        """Current Lagrange multipliers; the final ``n`` entries of the solution vector."""
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
        """Lagrange multipliers as a labelled DataArray of shape ``(layer, y, x)``."""
        return xr.DataArray(
            self._lagrangian.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
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
            - ``weights``: observation-to-head mapping, with dimensions
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

    def estimate_variance(self) -> xr.DataArray:
        """
        Posterior variance of the head estimate, from the inverse diagonal.

        Requires a backend like MUMPS that can return entries of the inverse;
        constructs a MUMPS linear solver if not available.
        """
        # Only mumps supports inverted entries properly.
        if isinstance(self.linearsolver, MumpsWrapper):
            linearsolver = self.linearsolver
        else:
            # TODO: maybe bind result in a separate MUMPS instance as to support
            # "batched" variance estimates?
            warnings.warn(
                "Current linear solver is not MUMPS. Only MUMPS supports "
                "selected inversion, refactorizing and solving with MUMPS now."
            )
            linearsolver = MumpsWrapper(
                self.K, self.rhs, self.x, matrix_type=self.matrix_type
            )
            linearsolver.analyze()
            linearsolver.factorize()

        variance = linearsolver.inverse_diagonal(indices=np.arange(self.n))
        return xr.DataArray(
            data=variance.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="variance",
        )
