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
        ax.set_ylabel("Semivariance")
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
    """
    Parameters
    ----------
    curvature_scale : float
        A characteristic curvature scale for the field, in units of
        recharge / length^2 (e.g. (m/d)/m^2 = m^-1 d^-1). Roughly:
        "how much does recharge bend over one length unit, relative
        to its own rate of change" -- smaller values penalize
        curvature more strongly (stiffer, smoother fields).
    length_scale: float
    """

    curvature_scale: float
    length_scale: float

    def build_tikhonov_operator(self, ny: int, nx: int, dx: float) -> sparse.csr_matrix:
        L = graph_laplacian(ny, nx)
        return L / (self.curvature_scale * self.length_scale * dx)

    @classmethod
    def from_sinusoid(
        cls,
        standard_deviation: float,
        effective_range: float,
    ) -> "MinimumCurvature":
        """
        Estimate a curvature_scale based on sinusoidal recharge pattern.

        Assumes a field r(x) = A sin(2 pi x / lambda) with amplitude
        A = standard_deviation * sqrt(2) and wavelength
        lambda = 2 * effective_range (i.e. effective_range is the
        crest-to-trough distance). Its peak curvature_scale,

            reference_curvature = (pi / effective_range)^2 * A

        is used as the curvature scale.

        Parameters
        ----------
        standard_deviation : float
            Typical recharge variation, same units as recharge (e.g. m/d).
        effective_range : float
            Distance over which that variation occurs, in model length
            units (e.g. m).
        """
        return cls(
            curvature_scale=(np.pi / effective_range) ** 2
            * standard_deviation
            * np.sqrt(2),
            length_scale=effective_range,
        )
