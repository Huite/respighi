from typing import Optional

import numpy as np
from scipy.sparse.linalg import LinearOperator


class CGIterable:
    """
    Minimally allocating iterable CG solver for Ax = b.

    Pre-allocates all workspace arrays and performs in-place operations where possible.
    Allows custom convergence checks via iteration.

    Parameters
    ----------
    A : sparse.csr_matrix
        The N-by-N matrix (must be symmetric positive definite)
    b : ndarray
        Right-hand side vector
    x : ndarray
        Initial guess (modified in-place)
    M : LinearOperator or array-like, optional
        Preconditioner

    Attributes
    ----------
    x : ndarray
        Current solution (updated in-place each iteration)
    iteration : int
        Current iteration number
    """

    def __init__(
        self,
        A: LinearOperator,
        b: np.ndarray,
        x: np.ndarray,
        M: LinearOperator,
    ):
        self.A = A
        self.b = b
        self.x = x
        self.M = M

        n = len(b)
        self.n = n

        # Pre-allocate workspace arrays
        dtype = x.dtype
        self.r = np.empty(n, dtype=dtype)  # residual
        self.z = np.empty(n, dtype=dtype)  # preconditioned residual
        self.p = np.empty(n, dtype=dtype)  # search direction
        self.q = np.empty(n, dtype=dtype)  # A @ p
        self.temp = np.empty(n, dtype=dtype)  # temporary for scaled vectors

        self.matvec = A.dot
        self.psolve = M.matvec

        # State variables
        self.iteration = 0
        self.rho_prev = None
        self.first_iteration = True

    def _initialize_residual(self):
        """Initialize residual vector in workspace"""
        np.copyto(self.r, self.b)
        self.r -= self.matvec(self.x)

    def reset(self, x: Optional[np.ndarray] = None):
        """
        Reset the iterator to start over with a new initial guess.

        Parameters
        ----------
        x : ndarray, optional
            New initial guess. If None, keeps current x.
        """
        if x is not None:
            np.copyto(self.x, x)

        self.M.update(self.A)
        self.iteration = 0
        self.rho_prev = None
        self.first_iteration = True
        self._initialize_residual()

    def __iter__(self):
        """Return self as iterator"""
        return self

    def __next__(self) -> int:
        return self.do_iter()

    def do_iter(self) -> int:
        """
        Perform one CG iteration.

        Returns
        -------
        iteration : int
            Current iteration number (before increment)
        """
        # Apply preconditioner: z = M^{-1} @ r
        self.z[:] = self.psolve(self.r)

        # Compute rho = <r, z> (scalar, no out parameter needed)
        rho_cur = np.dot(self.r, self.z)

        if self.first_iteration:
            # p = z (first iteration)
            np.copyto(self.p, self.z)
            self.first_iteration = False
        else:
            # p = z + beta * p (in-place)
            beta = rho_cur / self.rho_prev
            self.p *= beta
            self.p += self.z

        # q = A @ p
        self.q[:] = self.matvec(self.p)

        # Compute alpha = rho / <p, q> (scalars)
        alpha = rho_cur / np.dot(self.p, self.q)
        # Avoid allocations by using temp
        np.multiply(alpha, self.p, out=self.temp)
        np.add(self.x, self.temp, out=self.x)
        np.multiply(alpha, self.q, out=self.temp)
        np.subtract(self.r, self.temp, out=self.r)

        # Store rho for next iteration
        self.rho_prev = rho_cur

        # Increment iteration counter
        current_iteration = self.iteration
        self.iteration += 1

        return current_iteration


class PCGSolver:
    def __init__(
        self,
        A: LinearOperator,
        b: np.ndarray,
        x: np.ndarray,
        M: LinearOperator,
        xclose: float,
        rclose: float,
        maxiter: int,
    ):
        self.cg_iterable = CGIterable(A, b, x, M)
        self.xold = np.empty_like(x)
        self.dx = np.empty_like(x)
        self.xclose = xclose
        self.rclose = rclose
        self.maxiter = maxiter

    def solve(self):
        cg = self.cg_iterable
        cg.reset()

        for i in range(self.maxiter):
            # Store old solution before iteration
            np.copyto(self.xold, cg.x)

            # Perform one CG iteration
            cg.do_iter()

            # Check convergence
            rmax = np.linalg.norm(cg.r, ord=np.inf)
            np.subtract(cg.x, self.xold, out=self.dx)
            dxmax = np.linalg.norm(self.dx, ord=np.inf)

            if (rmax < self.rclose) and (dxmax < self.xclose):
                return True, i + 1

        return False, self.maxiter
