import warnings
from typing import Literal

import numpy as np
import xarray as xr
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres as _scipy_minres

from respighi.groundwaterflow import GroundwaterModel
from respighi.target import FittingTarget


def _minres_compat(A, b, x0, M, rtol, maxiter, callback):
    """Call scipy minres across the tol -> rtol API change (scipy >= 1.12)."""
    try:
        return _scipy_minres(
            A, b, x0=x0, M=M, rtol=rtol, maxiter=maxiter, callback=callback
        )
    except TypeError:
        return _scipy_minres(
            A, b, x0=x0, M=M, tol=rtol, maxiter=maxiter, callback=callback
        )


class InverseProblemMINRES:
    """
    Inverse recharge estimation via MINRES on the KKT system with the
    block diagonal augmented Lagrangian ("BDAL") preconditioner of
    Alger, Villa, Bui-Thanh & Ghattas (2017), SIAM J. Sci. Comput. 39(5),
    A2365-A2393.

    Notation mapping (paper -> here):

        parameter q          -> recharge r          (size layer_n, first layer)
        state u              -> head h              (size n)
        adjoint eta          -> eta                 (size n)
        forward operator A   -> flow matrix A       (symmetric; diagonal is
                                                     head-dependent via hcof)
        T                    -> -Q                  (Q = area * I injected into
                                                     layer 1, shape (n, layer_n))
        observation op B     -> target.P            (padded to n columns)
        alpha R*R            -> alpha * L2d         (unweighted 5-point graph
                                                     Laplacian on the 2D grid,
                                                     optionally + t*I)
        f                    -> gwf.rhs             (boundary condition RHS)

    following the paper's problem (3):

        min_{r,h}  1/2 ||P h - d||^2 + alpha/2 ||R r||^2
        s.t.       A h - Q r = b_bc

    The KKT system (paper eq. (4)/(36)), ordered [r, h, eta]:

        [ alpha R*R     0       -Q^T ] [ r ]     [ 0     ]
        [    0         P^T P     A   ] [ h ]  =  [ P^T d ]
        [   -Q          A        0   ] [ eta]    [ b_bc  ]

    is symmetric indefinite and is solved with MINRES, preconditioned by
    (paper eq. (5)/(37), with identity mass matrices):

        Pc = diag( alpha R*R + rho Q^T Q ,  P^T P + rho A^T A ,  (1/rho) I )

    with rho = sqrt(alpha) by default (paper's recommendation, Corollary
    12). Note that Q^T Q = area^2 * I, so the first block equals
    alpha*L2d + rho*area^2*I, which is SPD even though the pure Laplacian
    is singular. Consequently ``tikhonov_shift`` (the paper's ``t`` in
    R*R = Lap + t*I, section 6) may be left at 0.0; it only alters the
    regularization itself, not the solvability of the preconditioner
    blocks.

    Differences from the paper worth knowing about:

    - The paper uses FEM mass matrices W; on this uniform finite-
      difference grid all inner products are taken Euclidean (W = I).
      Any area/thickness weighting is effectively absorbed into the
      sweep over ``regularization_weight`` and ``rho``.
    - The paper's spectral theory assumes invertible T; here T = -Q is
      a tall rectangular injection into layer 1. Applying the
      preconditioner does not require invertibility of T (only the two
      SPD subsystem solves), matching the paper's remark to that effect.
    - Nonlinearity of A (smoothed drain/river switching) is handled by
      Picard iteration around the linear KKT solve, exactly as in
      ``InverseProblem``. MINRES is warm-started from the previous
      Picard iterate.

    Parameters
    ----------
    groundwatermodel : GroundwaterModel
        Provides A (Abase + hcof diagonal), the boundary condition RHS,
        cell area, and grid layout.
    target : FittingTarget
        Observation operator P and data d.
    regularization_weight : float
        The paper's alpha.
    rho : float, optional
        Augmentation parameter. Default sqrt(regularization_weight).
    tikhonov_shift : float, optional
        t in R*R = Lap2d + t*I (the paper uses t = 0.1 in section 6).
        Default 0.0.
    subsolver : {"amg", "exact"}
        How the two SPD preconditioner blocks are solved. "amg" uses
        PyAMG root-node smoothed aggregation with a fixed number of
        V-cycles (paper: 1 cycle for the r-block, 3 for the h-block).
        "exact" uses a sparse LU factorization per block.
    amg_cycles : tuple of int, optional
        (cycles for r-block, cycles for h-block). Default (1, 3).
    minres_rtol : float, optional
        Relative tolerance for MINRES (preconditioned residual).
    minres_maxiter : int, optional
        Maximum MINRES iterations per linear solve.
    maxiter : int, optional
        Maximum Picard iterations.
    maxdh : float, optional
        Nonlinear convergence criterion on the infinity norm of the
        head update.
    relax : float, optional
        Picard relaxation factor (0.0 = no relaxation).
    """

    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization_weight: float,
        rho: float | None = None,
        tikhonov_shift: float = 0.0,
        subsolver: Literal["amg", "exact"] = "amg",
        amg_cycles: tuple[int, int] = (1, 3),
        minres_rtol: float = 1e-8,
        minres_maxiter: int = 1000,
        maxiter: int = 30,
        maxdh: float = 1e-4,
        relax: float = 0.0,
    ):
        self.gwf = groundwatermodel
        self.target = target
        self.n = self.gwf.n
        self.layer_n = self.gwf.layer_n
        self.alpha = regularization_weight
        self.rho = np.sqrt(regularization_weight) if rho is None else rho
        self.subsolver = subsolver
        self.amg_cycles = amg_cycles
        self.minres_rtol = minres_rtol
        self.minres_maxiter = minres_maxiter
        self.maxiter = maxiter
        self.maxdh = maxdh
        self.relax = relax
        self.area = self.gwf.area

        n = self.n
        ln = self.layer_n

        # --- Regularization operator R*R: unweighted 2D graph Laplacian ---
        ny, nx = self.gwf.transmissivity.shape[1:]
        i, j = GroundwaterModel.build_connectivity((ny, nx))
        W2d = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(ln, ln)).tocsr()
        D2d = np.asarray(W2d.sum(axis=1)).ravel()
        RtR = sparse.diags(D2d) - W2d
        if tikhonov_shift != 0.0:
            RtR = RtR + tikhonov_shift * sparse.eye(ln)
        self.RtR = RtR.tocsr()

        # --- Observation operator, padded to n columns ---
        P = target.P
        if P.shape[1] < n:
            padding = sparse.csr_matrix((P.shape[0], n - P.shape[1]))
            P = sparse.hstack([P, padding])
        self.P = P.tocsr()
        self.Pt = self.P.T.tocsr()
        self.PtP = (self.Pt @ self.P).tocsr()

        # --- Flow matrix with in-place-updatable diagonal ---
        A = self.gwf.Abase.copy().tocsr()
        A.setdiag(np.inf)
        self._A_diag_idx = np.where(np.isinf(A.data))[0]
        A.data[self._A_diag_idx] = self.gwf.hcof
        self.A = A

        # --- Solution and RHS ---
        # x = [r (ln), h (n), eta (n)]
        self.N = ln + 2 * n
        self.x = np.zeros(self.N)
        self._x_old = np.zeros(self.N)
        self._x_update = np.zeros(self.N)
        self._head_iter = np.zeros(n)
        self._head_update = np.zeros(n)

        self.rhs = np.zeros(self.N)
        self.rhs[ln : ln + n] = self.Pt @ target.d
        self.rhs[ln + n :] = self.gwf.rhs

        # --- Preconditioner state ---
        self._solve_r_block = None
        self._solve_h_block = None
        self._formulated = False

        # --- Operators for MINRES ---
        self._K_op = LinearOperator(
            (self.N, self.N), matvec=self._K_matvec, dtype=float
        )
        self._M_op = LinearOperator(
            (self.N, self.N), matvec=self._M_matvec, dtype=float
        )
        self.last_minres_info: int | None = None
        self.last_minres_iterations: int = 0

    # ------------------------------------------------------------------
    # Matrix-vector products
    # ------------------------------------------------------------------
    def _K_matvec(self, z):
        """KKT operator applied to z = [r, h, eta]."""
        ln, n = self.layer_n, self.n
        r = z[:ln]
        h = z[ln : ln + n]
        eta = z[ln + n :]
        out = np.empty_like(z)
        # alpha R*R r - Q^T eta
        out[:ln] = self.alpha * (self.RtR @ r) - self.area * eta[:ln]
        # P^T P h + A eta
        out[ln : ln + n] = self.Pt @ (self.P @ h) + self.A @ eta
        # -Q r + A h
        out[ln + n :] = self.A @ h
        out[ln + n : ln + n + ln] -= self.area * r
        return out

    def _M_matvec(self, z):
        """Inverse of the BDAL preconditioner applied to z."""
        ln, n = self.layer_n, self.n
        out = np.empty_like(z)
        out[:ln] = self._solve_r_block(z[:ln])
        out[ln : ln + n] = self._solve_h_block(z[ln : ln + n])
        # Third block is (1/rho) I, so its inverse is rho * I.
        out[ln + n :] = self.rho * z[ln + n :]
        return out

    # ------------------------------------------------------------------
    # Preconditioner assembly
    # ------------------------------------------------------------------
    def _make_spd_solver(self, mat: sparse.csr_matrix, cycles: int):
        if self.subsolver == "exact":
            return sparse.linalg.factorized(mat.tocsc())
        elif self.subsolver == "amg":
            import pyamg

            ml = pyamg.rootnode_solver(mat.tocsr())

            def solve(b, _ml=ml, _cycles=cycles):
                # Fixed number of V-cycles: tol is unreachably small so the
                # cycle count, not the residual, terminates the solve.
                return _ml.solve(b, tol=1e-16, maxiter=_cycles, cycle="V", accel=None)

            return solve
        else:
            raise ValueError(f"Unknown subsolver: {self.subsolver}")

    def _build_r_block_solver(self):
        """Alpha R*R + rho Q^T Q with Q^T Q = area^2 I. Static across Picard."""
        mat = (
            self.alpha * self.RtR + self.rho * self.area**2 * sparse.eye(self.layer_n)
        ).tocsr()
        self._solve_r_block = self._make_spd_solver(mat, self.amg_cycles[0])

    def _build_h_block_solver(self):
        """P^T P + rho A^T A. A is symmetric, so A^T A = A @ A.

        Rebuilt every Picard iteration because the diagonal of A changes.
        For large grids the sparse product A @ A and (for "amg") the
        hierarchy setup dominate the cost; if Picard iterations are slow,
        consider freezing this block for a few iterations.
        """
        AtA = (self.A @ self.A).tocsr()
        mat = (self.PtP + self.rho * AtA).tocsr()
        self._solve_h_block = self._make_spd_solver(mat, self.amg_cycles[1])

    # ------------------------------------------------------------------
    # Formulation (Picard linearization)
    # ------------------------------------------------------------------
    def _formulate_gwf(self, dt):
        """Refresh A's diagonal and the flow RHS from the groundwater model."""
        np.copyto(self.gwf._head, self._head)
        self.gwf.formulate(recharge=False, dt=dt)
        self.A.data[self._A_diag_idx] = self.gwf.hcof
        self.rhs[self.layer_n + self.n :] = self.gwf.rhs

    def formulate(self, dt=0.0):
        """
        Formulate the KKT system and build both preconditioner blocks.

        Call once before solving; use ``reformulate`` inside Picard or
        time-stepping loops (the r-block is static and is not rebuilt).
        """
        self._formulate_gwf(dt=dt)
        self._build_r_block_solver()
        self._build_h_block_solver()
        self._formulated = True

    def reformulate(self, dt=0.0):
        """Refresh A and the flow RHS; rebuild only the h-block solver."""
        self._formulate_gwf(dt=dt)
        self._build_h_block_solver()

    def update_observations(self, d):
        """
        Replace the observation vector in the RHS.

        Parameters
        ----------
        d:
            New observation vector. Must have the same shape as the
            original ``target.d``.
        """
        if d.shape != self.target.d.shape:
            raise ValueError("Observation size changed: rebuild instead.")
        self.rhs[self.layer_n : self.layer_n + self.n] = self.Pt @ d

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def linear_solve(self):
        """
        Solve the KKT system with preconditioned MINRES, warm-started
        from the current solution vector.

        Returns
        -------
        info: int
            scipy MINRES exit code (0 = converged).
        iterations: int
            Number of MINRES iterations taken.
        """
        if not self._formulated:
            raise RuntimeError("Must call formulate() before solve")

        counter = {"it": 0}

        def callback(xk):
            counter["it"] += 1

        x, info = _minres_compat(
            self._K_op,
            self.rhs,
            x0=self.x,
            M=self._M_op,
            rtol=self.minres_rtol,
            maxiter=self.minres_maxiter,
            callback=callback,
        )
        self.x[:] = x
        self.last_minres_info = info
        self.last_minres_iterations = counter["it"]
        if info != 0:
            warnings.warn(
                f"MINRES did not converge (info={info}) after "
                f"{counter['it']} iterations."
            )
        return info, counter["it"]

    def nonlinear_solve(self, dt=0.0):
        """
        Solve the nonlinear KKT system using Picard iteration.

        Convergence is assessed on the infinity norm of the head update,
        as in ``InverseProblem``. Call ``formulate`` first.

        Parameters
        ----------
        dt:
            Time step size. 0.0 encodes steady state.

        Returns
        -------
        converged: bool
        iterations: int
        """
        if not self._formulated:
            raise RuntimeError("Must call formulate() before solve")

        maxdh = np.inf
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
            self.reformulate(dt=dt)

        warnings.warn(
            f"Nonlinear solver did not converge after {self.maxiter} "
            f"iterations. Final update: {maxdh:.2e}"
        )
        return False, self.maxiter

    def run(self, dts, targets, callback=None):
        """
        Run a transient or batched inverse solve over a sequence of
        time steps.

        Unlike the PARDISO version, there is no expensive analysis phase
        to amortize; each step refreshes the linearization and rebuilds
        the h-block preconditioner.

        Parameters
        ----------
        dts:
            Sequence of time step sizes.
        targets:
            Sequence of FittingTarget objects, one per time step.
        callback:
            Optional callable with signature ``callback(problem, i, dt)``.

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
            np.copyto(dst=self.gwf._head_old, src=self._head)
            self.reformulate(dt=dt)
            self.nonlinear_solve(dt=dt)
            out.append(self._head.copy())
        return out

    # ------------------------------------------------------------------
    # Views on the solution vector, x = [r, h, eta]
    # ------------------------------------------------------------------
    @property
    def _recharge(self):
        return self.x[: self.layer_n]

    @property
    def _head(self):
        return self.x[self.layer_n : self.layer_n + self.n]

    @property
    def _lagrangian(self):
        return self.x[self.layer_n + self.n :]

    @property
    def head(self):
        """Head estimate as a labelled DataArray of shape (layer, y, x)."""
        return xr.DataArray(
            data=self._head.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="head",
        )

    @property
    def recharge(self):
        """Recharge estimate as a labelled DataArray of shape (y, x)."""
        return xr.DataArray(
            data=self._recharge.reshape(self.gwf.transmissivity.shape[1:]),
            dims=("y", "x"),
            coords={"y": self.gwf._coords["y"], "x": self.gwf._coords["x"]},
            name="recharge",
        )

    @property
    def lagrangian(self):
        """Adjoint variable as a labelled DataArray of shape (layer, y, x).

        Note: unlike ``InverseProblem``, the adjoint here is the
        multiplier of the flow equation and lives on all cells (size n),
        not only the first layer.
        """
        return xr.DataArray(
            data=self._lagrangian.reshape(self.gwf.transmissivity.shape),
            dims=("layer", "y", "x"),
            coords=self.gwf._coords,
            name="lagrangian",
        )
