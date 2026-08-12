from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.special import k1

from respighi.groundwaterflow import GroundwaterModel


def graph_laplacian(ny: int, nx: int) -> sparse.csr_matrix:
    layer_n = ny * nx
    i, j = GroundwaterModel.build_connectivity((ny, nx))
    W_2d = sparse.coo_matrix(
        (np.ones(len(i)), (i, j)), shape=(layer_n, layer_n)
    ).tocsr()
    D_2d = np.asarray(W_2d.sum(axis=1)).ravel()  # Degree matrix
    return sparse.diags(D_2d) - W_2d


class MaternSemivariogram(NamedTuple):
    """
    Matérn (nu=1) semivariogram.

    Parameters
    ----------
    standard_deviation : float
        Marginal standard deviation of the field (sill = standard_deviation**2).
    effective_range : float
        Effective range: the distance at which the semivariogram reaches
        ~86% of the sill. For nu=1, kappa = sqrt(8) / effective_range.
    """

    standard_deviation: float
    effective_range: float

    @property
    def sill(self):
        return self.standard_deviation**2

    @property
    def kappa(self):
        return np.sqrt(8) / self.effective_range

    @property
    def tau(self):
        return 1.0 / np.sqrt(4 * np.pi * self.kappa**2 * self.sill)

    def plot(self, xmax=None, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        if xmax is None:
            xmax = 1.5 * self.effective_range
        x = np.linspace(0.1, xmax, 1000)
        kappa_x = self.kappa * x
        semivariance = self.sill * (1 - kappa_x * k1(kappa_x))
        ax.plot(x, semivariance)
        ax.axhline(self.sill, linestyle="dotted", color="gray", label="sill")
        ax.axvline(
            self.effective_range, linestyle="dashed", color="black", label="range"
        )
        ax.set_ylabel("Variance")
        ax.set_xlabel("Distance")
        ax.legend()
        return ax

    def build_tikhonov_operator(self, ny: int, nx: int, dx: float) -> sparse.csr_matrix:
        L = graph_laplacian(ny, nx)
        _I = sparse.eye(L.shape[0], format="csr")
        kappa_grid = self.kappa * dx
        return (self.tau / dx) * (kappa_grid**2 * _I + L)


class UnscaledMinimumCurvature(NamedTuple):
    """Backwards compatibility."""

    weight: float

    def build_tikhonov_operator(self, ny: int, nx: int, dx: float) -> sparse.csr_matrix:
        L = graph_laplacian(ny, nx)
        return self.weight * L


class MinimumCurvature(NamedTuple):
    roughness_scale: float

    def build_tikhonov_operator(
        self,
        ny: int,
        nx: int,
        dx: float,
    ) -> sparse.csr_matrix:
        L = graph_laplacian(ny, nx)
        return L / (self.roughness_scale * dx)

    @classmethod
    def from_sinusoid(
        cls,
        amplitude: float,
        half_wavelength: float,
    ) -> "MinimumCurvature":
        """
        Scale minimum-curvature regularization using a conceptual
        sinusoidal recharge field.

        The reference field is

            r(x) = A sin(pi x / R)

        where A is the peak recharge anomaly and R is the
        half-wavelength.

        Parameters
        ----------
        amplitude : float
            Peak recharge anomaly in m/d. The reference field varies
            between -amplitude and +amplitude around its mean.

        half_wavelength : float
            Distance in model length units between a positive peak and
            the next negative peak. The full sinusoidal wavelength is
            twice this value.

        Notes
        -----
        The maximum curvature of the reference field is

            (pi / R)**2 * A,

        and the resulting minimum-curvature roughness scale is

            pi**2 * A / R,

        and its standard devation is

            sqrt(2) * A

        or approximately 0.7 * A.
        """
        return cls(roughness_scale=(np.pi**2 * amplitude / half_wavelength))
