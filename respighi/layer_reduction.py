from typing import NamedTuple

import numba as nb
import numpy as np

FloatArray = np.ndarray


class ThomasWorkspace(NamedTuple):
    dl: FloatArray  # lower diagonal, length n-1
    d: FloatArray  # main diagonal,  length n
    du: FloatArray  # upper diagonal, length n-1
    rhs: FloatArray  # right-hand side, length n
    x: FloatArray  # solution output, length n
    dc: FloatArray  # scratch: copy of d for forward sweep
    bc: FloatArray  # scratch: copy of rhs for forward sweep


@nb.njit(cache=True)
def _thomas_solve(ws):
    """Thomas algorithm using pre-allocated workspace. Reads dl/d/du/rhs, writes x."""
    n = ws.d.shape[0]

    # Copy to scratch so inputs are not clobbered
    for i in range(n):
        ws.dc[i] = ws.d[i]
        ws.bc[i] = ws.rhs[i]

    # Forward sweep
    for i in range(1, n):
        w = ws.dl[i - 1] / ws.dc[i - 1]
        ws.dc[i] -= w * ws.du[i - 1]
        ws.bc[i] -= w * ws.bc[i - 1]

    # Back substitution
    ws.x[n - 1] = ws.bc[n - 1] / ws.dc[n - 1]
    for i in range(n - 2, -1, -1):
        ws.x[i] = (ws.bc[i] - ws.du[i] * ws.x[i + 1]) / ws.dc[i]


@nb.njit(cache=True)
def _fill_workspace(ws, T, c, k_sq):
    """
    Fill tridiagonal system into workspace for layers 1..nlay-1,
    with layer 0 as reference (alpha[0] = 1).

    T    : (nlay,)
    c    : (nlay-1,)
    k_sq : scalar
    """
    nlay = T.shape[0]
    n = nlay - 1

    for i in range(n):
        layer = i + 1
        leak_above = 1.0 / c[layer - 1]
        leak_below = 1.0 / c[layer] if layer < nlay - 1 else 0.0

        ws.d[i] = k_sq * T[layer] + leak_above + leak_below
        ws.rhs[i] = leak_above if i == 0 else 0.0

        if i > 0:
            ws.dl[i - 1] = -1.0 / c[layer - 1]
        if i < n - 1:
            ws.du[i] = -1.0 / c[layer]

    return


@nb.njit(cache=True)
def _teff_grid(T_flat, c_flat, k_sq_flat):
    """
    Compute effective transmissivity for each cell using a shared workspace.

    T_flat    : (ncells, nlay)
    c_flat    : (ncells, nlay-1)
    k_sq_flat : (ncells,)
    """
    ncells = T_flat.shape[0]
    nlay = T_flat.shape[1]
    n = nlay - 1  # unknowns: alpha[1] .. alpha[nlay-1]
    out = np.empty(ncells)
    ws = ThomasWorkspace(
        dl=np.empty(n - 1, dtype=np.float64),
        d=np.empty(n, dtype=np.float64),
        du=np.empty(n - 1, dtype=np.float64),
        rhs=np.empty(n, dtype=np.float64),
        x=np.empty(n, dtype=np.float64),
        dc=np.empty(n, dtype=np.float64),
        bc=np.empty(n, dtype=np.float64),
    )
    for cell in range(ncells):
        _fill_workspace(ws, T_flat[cell], c_flat[cell], k_sq_flat[cell])
        _thomas_solve(ws)
        teff = T_flat[cell, 0]  # reference layer contributes with alpha=1
        for i in range(n):
            teff += T_flat[cell, i + 1] * ws.x[i]
        out[cell] = teff
    return out


def effective_transmissivity(
    T: np.ndarray,
    c: np.ndarray,
    L: np.ndarray | float,
) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    if np.any(T <= 0):
        raise ValueError("All transmissivities must be positive.")
    if np.any(c <= 0):
        raise ValueError("All resistances must be positive.")

    nlay = T.shape[-1]
    if c.shape[-1] != nlay - 1:
        raise ValueError("Last axis of c must have length nlay - 1.")

    grid_shape = np.broadcast_shapes(T.shape[:-1], c.shape[:-1], L.shape)

    T = np.broadcast_to(T, grid_shape + (nlay,))
    c = np.broadcast_to(c, grid_shape + (nlay - 1,))
    L = np.broadcast_to(L, grid_shape)

    ncells = int(np.prod(grid_shape)) if grid_shape else 1

    T_flat = np.ascontiguousarray(T.reshape(ncells, nlay))
    c_flat = np.ascontiguousarray(c.reshape(ncells, nlay - 1))
    k_sq_flat = (1.0 / L.ravel()) ** 2
    out = _teff_grid(T_flat, c_flat, k_sq_flat)

    return out.reshape(grid_shape) if grid_shape else float(out[0])


def two_layer_effective_transmisivity(
    T, c, L_local=100.0, L_mid=1_000.0, L_large=10_000.0
):
    T_a = effective_transmissivity(T, c, L_local)
    T_mid = effective_transmissivity(T, c, L_mid)
    T_large = effective_transmissivity(T, c, L_large)
    T_b = T_large - T_a
    delta_mid = T_mid - T_a
    k_mid_sq = 1.0 / L_mid**2
    c_ab = (T_b / delta_mid - 1.0) / (k_mid_sq * T_b)
    return T_a, T_b, c_ab
