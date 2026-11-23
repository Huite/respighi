"""
Horizontal flow on structured grids, isotropic or anisotropic, assembled from
a table of connections.

Each horizontal connection k joins cells i < j and is described by the unit
normal from i to j and the accumulated resistance of any barriers on the
shared face. The operators follow by sparse algebra on that table:

    D = incidence      (m, n)  +1 at i, -1 at j
    F = face flux      (m, n)  Q = F @ h is the flow from i to j
    L = D.T @ F        (n, n)  net outflow per cell

Isotropic: ``F = diag(C) @ D``, with C the harmonic-mean conductance: the
weighted graph Laplacian of the original code.

Anisotropic (XT3D-style). With n the unit normal from i to j and t the unit
tangent, each cell sees ``q.n = -(Tnn dh/dn + Tnt dh/dt)``. The normal
derivative runs to an unknown face head half a cell away; the tangential
derivative is the cell's own gradient estimate projected on t. Requiring
q.n to be equal on both sides eliminates the face head::

    a_i = 2 Tnn_i / d
    K0  = a_i a_j / (a_i + a_j)
    K   = K0 / (1 + r w K0)            barrier of resistance r on the face
    Q   = w [K (h_i - h_j) - K/a_i Tnt_i (t.grad h)_i - K/a_j Tnt_j (t.grad h)_j]

Cells are square, so the face width w equals the cell distance d, and the
geometry cancels: w K0 is the harmonic mean of Tnn (the familiar
conductance), and w (t.grad h) is a head difference. Everything below is
therefore written for unit cells; only the orientation of each connection
matters. The convention is that rows run towards decreasing y (raster
layout), so the angle of the tensor is counter-clockwise in map view.

The cross terms reach the tangential neighbours of both cells: a 9-point
stencil in each layer. With Tnt = 0 they vanish and only the first term is
left. A barrier scales the whole face flux, cross terms included.

The tensor is horizontal only, so vertical connections stay two-point: with
layers the stencil has at most 9 + 2 = 11 points.
"""

from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


def _atleast_3d_front(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    while a.ndim < 3:
        a = a[np.newaxis]
    return a


@dataclass(frozen=True)
class AnisotropicTransmissivity:
    """
    Horizontally anisotropic transmissivity, as principal values and rotation.

    Mirrors MODFLOW 6 NPF's K / K22 / ANGLE1.

    Parameters
    ----------
    t1:
        Transmissivity along the principal axis. Shaped ``(ny, nx)`` or
        ``(nlayer, ny, nx)``; may be an xarray DataArray, in which case its
        coordinates are used by the model.
    t2:
        Transmissivity perpendicular to the principal axis. Broadcastable to t1.
    angle:
        Rotation of the t1 axis, in degrees, counter-clockwise from +x as seen
        from above. Broadcastable to t1.
    """

    t1: ArrayLike
    t2: ArrayLike
    angle: ArrayLike = 0.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return _atleast_3d_front(self.t1).shape

    def components(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return (Txx, Tyy, Txy) in grid axes, each shaped (nlayer, ny, nx)."""
        t1 = _atleast_3d_front(self.t1)
        t2 = np.broadcast_to(_atleast_3d_front(self.t2), t1.shape)
        theta = np.deg2rad(np.broadcast_to(_atleast_3d_front(self.angle), t1.shape))
        if np.any(t1 < 0.0) or np.any(t2 < 0.0):
            raise ValueError("Principal transmissivities must be non-negative.")
        c = np.cos(theta)
        s = np.sin(theta)
        txx = t1 * c * c + t2 * s * s
        tyy = t1 * s * s + t2 * c * c
        txy = (t1 - t2) * s * c
        # Snap roundoff to exact zero (sin * cos at 90 degrees is ~6e-17), so
        # axis-aligned cells contribute no cross terms to the sparsity pattern.
        txy = np.where(np.abs(txy) < 1e-12 * (txx + tyy), 0.0, txy)
        return txx, tyy, txy


def incidence(i: IntArray, j: IntArray, n: int) -> sparse.csr_matrix:
    """(m, n) matrix with +1 at (k, i[k]) and -1 at (k, j[k])."""
    m = i.size
    k = np.arange(m)
    return sparse.coo_matrix(
        (np.repeat([1.0, -1.0], m), (np.concatenate([k, k]), np.concatenate([i, j]))),
        shape=(m, n),
    ).tocsr()


def barrier_resistance(shape, barriers: Sequence) -> tuple[FloatArray, FloatArray]:
    """
    Accumulate horizontal flow barriers onto faces.

    Returns
    -------
    rx: (nlayer, ny, nx - 1)
        Resistance on the face between columns c and c + 1.
    ry: (nlayer, ny - 1, nx)
        Resistance on the face between rows r and r + 1.

    Zero means no barrier. Several barriers on one face act in series, so
    their resistances add.
    """
    nlayer, ny, nx = shape
    rx = np.zeros((nlayer, ny, max(nx - 1, 0)))
    ry = np.zeros((nlayer, max(ny - 1, 0), nx))
    for hfb in barriers:
        a = np.minimum(hfb.cell0, hfb.cell1)
        b = np.maximum(hfb.cell0, hfb.cell1)
        if a.size == 0:
            continue
        if b.max() >= ny * nx:
            raise ValueError("HFB cell index exceeds number of cells in a layer.")
        row, col = np.divmod(a, nx)
        is_x = ((b - a) == 1) & (col < nx - 1)  # b in the same row as a
        is_y = (b - a) == nx
        if not (is_x | is_y).all():
            raise ValueError("HFB cell pairs must be lateral neighbours.")
        resistance = np.broadcast_to(np.asarray(hfb.resistance, dtype=float), a.shape)
        np.add.at(rx, (hfb.layer, row[is_x], col[is_x]), resistance[is_x])
        np.add.at(ry, (hfb.layer, row[is_y], col[is_y]), resistance[is_y])
    return rx, ry


@dataclass
class Connections:
    """
    Table of horizontal connections, one row per face, with i < j.

    Attributes
    ----------
    i, j:
        Global cell indices.
    normal_x, normal_y:
        Unit normal from cell i to cell j.
    resistance:
        Barrier resistance on the face, 0 for none.
    """

    i: IntArray
    j: IntArray
    normal_x: FloatArray
    normal_y: FloatArray
    resistance: FloatArray

    @classmethod
    def structured(cls, shape, barriers: Sequence = ()) -> "Connections":
        """
        All lateral connections of a (nlayer, ny, nx) grid of square cells.

        Columns run towards increasing x, rows towards decreasing y. Rows of
        the table are the x-connections in the order of ``rx.ravel()``, then
        the y-connections in the order of ``ry.ravel()`` (see
        :func:`barrier_resistance`), so barrier resistances drop straight
        into the table.
        """
        index = np.arange(np.prod(shape)).reshape(shape)
        along_x = (index[:, :, :-1].ravel(), index[:, :, 1:].ravel())
        along_y = (index[:, :-1, :].ravel(), index[:, 1:, :].ravel())
        counts = [along_x[0].size, along_y[0].size]
        rx, ry = barrier_resistance(shape, barriers)
        return cls(
            i=np.concatenate([along_x[0], along_y[0]]),
            j=np.concatenate([along_x[1], along_y[1]]),
            normal_x=np.repeat([1.0, 0.0], counts),
            normal_y=np.repeat([0.0, -1.0], counts),
            resistance=np.concatenate([rx.ravel(), ry.ravel()]),
        )

    def __len__(self) -> int:
        return self.i.size

    def select(self, mask: BoolArray) -> "Connections":
        return Connections(
            **{f.name: getattr(self, f.name)[mask] for f in fields(self)}
        )

    def incidence(self, n: int) -> sparse.csr_matrix:
        return incidence(self.i, self.j, n)


def mean_slope(i: IntArray, j: IntArray, d: FloatArray, n: int) -> sparse.csr_matrix:
    """
    (n, n) operator giving, per cell, the mean slope of its links.

    Link k contributes the slope ``(h_j - h_i) / d[k]`` to both of its cells,
    with d the signed coordinate step from i to j; cells without links get
    zero.
    """
    D = incidence(i, j, n)
    slope = sparse.diags(-1.0 / d) @ D
    member = abs(D).T
    count = np.asarray(member.sum(axis=1)).ravel()
    mean = sparse.diags(np.divide(1.0, count, out=np.zeros(n), where=count > 0))
    return (mean @ member @ slope).tocsr()


def cell_gradients(
    conn: Connections, n: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    (n, n) operators estimating dh/dx and dh/dy in every cell.

    A cell's estimate is the mean slope of its own connections along each
    axis: on a uniform grid the central difference, one-sided next to edges
    and inactive cells (which have no connections).

    Barriers are deliberately ignored here, as in MODFLOW 6 (``xt3d_fhfb``
    only rescales the barrier face itself). Consequence: next to a strong
    barrier, the head jump enters the tangential gradient of the adjacent
    faces, which can produce spurious flow along the barrier.
    """
    x = conn.select(conn.normal_x != 0.0)
    y = conn.select(conn.normal_y != 0.0)
    gx = mean_slope(x.i, x.j, x.normal_x, n)
    gy = mean_slope(y.i, y.j, y.normal_y, n)
    return gx, gy


def project(
    txx: FloatArray,
    tyy: FloatArray,
    txy: FloatArray,
    normal: FloatArray,
    tangent: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Return ``n.T.n`` and ``t.T.n`` per connection.

    txx, tyy, txy are the tensor components of one side of each connection;
    normal and tangent are (2, m) unit vectors.
    """
    Tn_x = txx * normal[0] + txy * normal[1]
    Tn_y = txy * normal[0] + tyy * normal[1]
    tnn = normal[0] * Tn_x + normal[1] * Tn_y
    tnt = tangent[0] * Tn_x + tangent[1] * Tn_y
    return tnn, tnt


def tangential_derivative(
    gx: sparse.csr_matrix,
    gy: sparse.csr_matrix,
    tangent: FloatArray,
    cells: IntArray,
) -> sparse.csr_matrix:
    """(m, n) operator: row k estimates ``t.grad(h)`` in cell ``cells[k]``."""
    return sparse.diags(tangent[0]) @ gx[cells] + sparse.diags(tangent[1]) @ gy[cells]


def horizontal_operator(
    conn: Connections,
    txx: FloatArray,
    tyy: FloatArray,
    txy: FloatArray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Assemble the horizontal flow operators from a connection table.

    Parameters
    ----------
    conn:
        Connections between active cells, barriers accumulated.
    txx, tyy, txy:
        Flat per-cell tensor components in grid axes. txy=None for
        isotropic input: no cross terms, 5-point stencil.

    Returns
    -------
    L: (n, n)
        Net outflow per cell, ``(L @ h)[c]``.
    F: (m, n)
        Face flows, ``(F @ h)[k]`` from i[k] to j[k]; handy for budgets.
    """
    n = txx.size
    i, j = conn.i, conn.j
    normal = np.stack([conn.normal_x, conn.normal_y])
    tangent = np.stack([-normal[1], normal[0]])
    cross = np.zeros(n) if txy is None else txy

    tnn_i, tnt_i = project(txx[i], tyy[i], cross[i], normal, tangent)
    tnn_j, tnt_j = project(txx[j], tyy[j], cross[j], normal, tangent)
    # Unit cells: a = 2 Tnn, and k0 is the harmonic-mean conductance.
    a_i = 2.0 * tnn_i
    a_j = 2.0 * tnn_j
    a_sum = a_i + a_j
    k0 = np.divide(a_i * a_j, a_sum, out=np.zeros_like(a_sum), where=a_sum > 0.0)
    # A barrier scales the whole face flux, cross terms included; this is
    # MODFLOW 6's xt3d_fhfb with condhfb = 1 / r.
    k = k0 / (1.0 + conn.resistance * k0)

    D = conn.incidence(n)
    F = sparse.diags(k) @ D
    if txy is not None:
        gx, gy = cell_gradients(conn, n)
        # Weight of each side's cross term: K / a * Tnt (zero where a = 0).
        w_i = np.divide(k * tnt_i, a_i, out=np.zeros_like(a_i), where=a_i > 0.0)
        w_j = np.divide(k * tnt_j, a_j, out=np.zeros_like(a_j), where=a_j > 0.0)
        F = (
            F
            - sparse.diags(w_i) @ tangential_derivative(gx, gy, tangent, i)
            - sparse.diags(w_j) @ tangential_derivative(gx, gy, tangent, j)
        )
    F = F.tocsr()
    return (D.T @ F).tocsr(), F
