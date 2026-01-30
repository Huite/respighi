import warnings

from scipy import sparse
import numpy as np

from respighi.pardiso import PardisoWrapper
from respighi.groundwaterflow import GroundwaterModel
from respighi.target import FittingTarget


class InverseProblem:
    def __init__(
        self,
        groundwatermodel: GroundwaterModel,
        target: FittingTarget,
        regularization_weight: float,
        maxiter: int = 30,
        maxdh=1e-4,
        relax=0.5,
    ):
        # Store core attributes
        self.gwf = groundwatermodel
        self.target = target
        self.n = self.gwf.n
        self.regularization_weight = regularization_weight
        self.maxiter = maxiter
        self.maxdh = maxdh
        self.relax = relax
        # Build KKT system
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
        Pt = P.T

        # NOTE:
        # also assumes constant cell sizes, and dx == dy.
        W = self.gwf.W.copy()
        W.data[:] = 1.0
        D = np.asarray(W.sum(axis=1)).ravel()  # Degree matrix
        L = regularization_weight * (sparse.diags(D) - W)
        Lt = L.T

        Q = sparse.diags(self.gwf.area)
        Qt = Q.T

        n_obs = P.shape[0]
        I_e = sparse.eye(n_obs, format="csr")
        I_s = sparse.eye(self.n, format="csr")

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
        """Build right-hand side vector for KKT system."""
        return np.concatenate(
            [
                np.zeros(self.n),
                np.zeros(self.n),
                self.gwf.rhs,
                self.target.d,
                np.zeros(self.n),
            ]
        )

    def _extract_diagonal_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract diagonal indices for efficient Picard iteration updates.
        Returns indices of A and At diagonals within the CSR data array.
        """
        inf_indices = np.where(np.isinf(self.K.data))[0]
        return inf_indices[: self.n], inf_indices[self.n :]

    def _formulate_gwf(self):
        self.gwf.formulate(recharge=False)
        self.K.data[self.At_diag_indices] = self.gwf.hcof
        self.K.data[self.A_diag_indices] = self.gwf.hcof
        self.rhs[2 * self.n : 3 * self.n] = self.gwf.rhs
        return

    def formulate(self):
        self._formulate_gwf()
        self.linearsolver = PardisoWrapper(self.K, self.rhs, self.x)
        # Analysis is the most costly phase.
        self.linearsolver.analyze()
        self.linearsolver.factorize()

    def reformulate(self):
        # Structure is static, reuse results of analysis.
        self._formulate_gwf()
        self.linearsolver.factorize()

    def linear_solve(self):
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")
        self.linearsolver.solve()
        return

    def nonlinear_solve(self):
        """
        Solve nonlinear system using Picard iteration.

        Call .formulate() first.
        """
        if self.linearsolver is None:
            raise RuntimeError("Must call formulate() before solve")

        for i in range(self.maxiter):
            np.copyto(dst=self._x_old, src=self.x)
            np.copyto(dst=self._head_old, src=self.head)
            self.linear_solve()
            np.subtract(self.head, self._head_old, out=self._head_update)
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

    @property
    def head(self):
        return self.x[: self.n]

    @property
    def recharge(self):
        return self.x[self.n : 2 * self.n]

    @property
    def lagrangian(self):
        return self.x[-self.n :]
