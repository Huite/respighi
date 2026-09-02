r"""
Transmissivity reduction theory
================================

In some geohydrological applications, a full multi-aquifer system is more
detailed than is needed for the intended calculation, or too cumbersome.  It is
then useful to replace the multi-aquifer system by a single equivalent aquifer
with a representative transmissivity.  The goal of the reduction is not to
reproduce every possible multi-layer response, but to define an effective
transmissivity for head variations occurring over a specified horizontal length
scale.

Multiple-aquifer analytical solution theory provides a natural way to think
about such a reduction.  One may choose a particular analytical response of the
multi-layer system, compare it with the corresponding single-layer response,
and select the single-layer transmissivity that reproduces the multi-layer head
or discharge response in some representative sense.  For example, one could
base the reduction on radial flow to a well, on the exponential response to a
straight head boundary, or on the response associated with a strip.

Those choices are physically meaningful, but they introduce geometry into the
definition of the reduced transmissivity.  A well response is radial and tied to
a point sink.  A straight-boundary response and a strip response are tied to
particular boundary geometries.  In addition, responses that decay away from a
boundary or well require the reduction to be evaluated at some chosen distance
from the source.  At sufficiently large distance the remaining signal becomes
small, so the fitted transmissivity may become sensitive to the choice of
distance.

The approach used here asks a simpler and more local question: for a head
variation with horizontal length scale :math:`L`, how much of the transmissivity
in the connected aquifers participates in the flow?  To answer this, we use a
horizontal eigenfunction of the Laplacian as a length-scale probe.  In one
horizontal dimension this may be written as the sinusoid

.. math::

    \psi(x) = \cos(kx), \qquad k = \frac{1}{L}.

The essential property of this choice is

.. math::

    \frac{d^2 \psi}{dx^2} = -k^2 \psi.

With the convention k = 1/L, the parameter L is the inverse
wavenumber, or horizontal curvature length scale.  The corresponding wavelength
is

.. math::

    \lambda = \frac{2\pi}{k} = 2\pi L.

The sinusoid should be interpreted first as a representative horizontal mode,
not as a claim that the physical boundary condition is literally sinusoidal.
It nevertheless has a useful physical interpretation: one half-period, or one
"hump", resembles the head mound that develops between drainage boundaries.
It is therefore a convenient way of probing the multi-aquifer system at a
prescribed horizontal scale.  Short-scale head variations have large k,
so vertical leakage has less horizontal distance over which to equilibrate heads
between aquifers.  Long-scale head variations have small k, so connected
aquifers can follow the reference aquifer more closely.
"""
# %%

import matplotlib.pyplot as plt
import numpy as np

from respighi.layer_reduction import (
    effective_transmissivity,
    two_layer_effective_transmisivity,
)

# %%


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}

# %%
# Schema
# ------


def compute_alphas(T, c, L):
    """
    Solve the amplitude system for h_i(x) = alpha_i cos(x/L),
    with alpha_0 = 1.
    """
    T = np.asarray(T, dtype=float)
    c = np.asarray(c, dtype=float)

    nlay = T.size
    n = nlay - 1
    k_sq = 1.0 / L**2

    M = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i in range(n):
        layer = i + 1  # unknowns correspond to layers 1..nlay-1

        leak_above = 1.0 / c[layer - 1]
        leak_below = 1.0 / c[layer] if layer < nlay - 1 else 0.0

        M[i, i] = k_sq * T[layer] + leak_above + leak_below
        b[i] = leak_above if i == 0 else 0.0

        if i > 0:
            M[i, i - 1] = -1.0 / c[layer - 1]
        if i < n - 1:
            M[i, i + 1] = -1.0 / c[layer]

    alpha_rest = np.linalg.solve(M, b)
    return np.concatenate(([1.0], alpha_rest))


T = np.array([100.0, 200.0, 300.0, 400.0])
c_top = 500.0
c_inter = np.array([600.0, 700.0, 800.0])
L = 700.0  # chosen horizontal scale [m]

# Two full periods of cos(x / L):
# period = 2*pi*L, so two periods = 4*pi*L
xmax = 4.0 * np.pi * L
x = np.linspace(0.0, xmax, 1200)
alphas = compute_alphas(T, c_inter, L)
period = 2.0 * np.pi * L


fig = plt.figure(figsize=(11, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.50)

axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])

# Schematic vertical geometry
aq_h = 0.95
at_h = 0.26

# Build from bottom upward
y_aq3_bot = 0.00
y_aq3_top = y_aq3_bot + aq_h
y_at3_bot = y_aq3_top
y_at3_top = y_at3_bot + at_h
y_aq2_bot = y_at3_top
y_aq2_top = y_aq2_bot + aq_h
y_at2_bot = y_aq2_top
y_at2_top = y_at2_bot + at_h
y_aq1_bot = y_at2_top
y_aq1_top = y_aq1_bot + aq_h
y_at1_bot = y_aq1_top
y_at1_top = y_at1_bot + at_h
y_aq0_bot = y_at1_top
y_top_mean = y_aq0_bot + aq_h
top_amp = 0.18

# Top sinusoidal boundary (blue)
y_top = y_top_mean + top_amp * np.cos(x / L)

# Lower aquifers / aquitards
axL.fill_between(
    x, y_aq3_bot, y_aq3_top, color="0.86", edgecolor="black", linewidth=1.0
)
axL.fill_between(
    x, y_at3_bot, y_at3_top, color="0.35", edgecolor="black", linewidth=1.0
)
axL.fill_between(
    x, y_aq2_bot, y_aq2_top, color="0.86", edgecolor="black", linewidth=1.0
)
axL.fill_between(
    x, y_at2_bot, y_at2_top, color="0.35", edgecolor="black", linewidth=1.0
)
axL.fill_between(
    x, y_aq1_bot, y_aq1_top, color="0.86", edgecolor="black", linewidth=1.0
)
axL.fill_between(
    x, y_at1_bot, y_at1_top, color="0.35", edgecolor="black", linewidth=1.0
)

# Top aquifer with sinusoidal top boundary
axL.fill_between(x, y_aq0_bot, y_top, color="0.86", edgecolor="black", linewidth=1.0)
axL.plot(x, y_top, color="tab:blue", linewidth=2.2)

# Labels inside aquifers
xmid = 0.50 * xmax
axL.text(
    xmid,
    0.5 * (y_aq0_bot + y_top_mean),
    r"Aquifer 0" "\n" r"$T_0 = 100$",
    ha="center",
    va="center",
    fontsize=10,
)
axL.text(
    xmid,
    0.5 * (y_aq1_bot + y_aq1_top),
    r"Aquifer 1" "\n" r"$T_1 = 200$",
    ha="center",
    va="center",
    fontsize=10,
)
axL.text(
    xmid,
    0.5 * (y_aq2_bot + y_aq2_top),
    r"Aquifer 2" "\n" r"$T_2 = 300$",
    ha="center",
    va="center",
    fontsize=10,
)
axL.text(
    xmid,
    0.5 * (y_aq3_bot + y_aq3_top),
    r"Aquifer 3" "\n" r"$T_3 = 400$",
    ha="center",
    va="center",
    fontsize=10,
)

# Labels for aquitards on the right
xlab = xmax * 1.01
axL.text(
    xlab,
    0.5 * (y_at1_bot + y_at1_top),
    r"$c_1 = 600$",
    ha="left",
    va="center",
    fontsize=10,
)
axL.text(
    xlab,
    0.5 * (y_at2_bot + y_at2_top),
    r"$c_2 = 700$",
    ha="left",
    va="center",
    fontsize=10,
)
axL.text(
    xlab,
    0.5 * (y_at3_bot + y_at3_top),
    r"$c_3 = 800$",
    ha="left",
    va="center",
    fontsize=10,
)

axL.set_xlim(0.0, xmax)
axL.set_ylim(-0.10, y_top.max() + 0.40)
axL.set_xlabel("horizontal distance x [m]")
axL.set_yticks([])
axL.set_title("a) Four-aquifer system")
axL.spines["left"].set_visible(False)
axL.spines["right"].set_visible(False)
axL.spines["top"].set_visible(False)

offsets = np.array([3.45, 2.35, 1.25, 0.15])
amp_scale = 0.40

COLORS = (
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
)

for i, (alpha, y0) in enumerate(zip(alphas, offsets)):
    ycurve = y0 + amp_scale * alpha * np.cos(x / L)

    axR.plot(x, ycurve, linewidth=2.0, color=COLORS[i])
    axR.plot([0.0, xmax], [y0, y0], color="0.82", linewidth=0.8)

    axR.text(
        -0.02 * xmax,
        y0,
        rf"Aquifer {i}",
        ha="right",
        va="center",
        fontsize=10,
    )

axR.set_xlim(0.0, xmax)
axR.set_ylim(-0.55, 4.20)
axR.set_xlabel("horizontal distance x [m]")
axR.set_yticks([])
axR.set_title("b) Head per aquifer")
axR.spines["left"].set_visible(False)
axR.spines["right"].set_visible(False)
axR.spines["top"].set_visible(False)
plt.show()


# %%
# Example: Limiting behavior
# --------------------------

fig, ax = plt.subplots(figsize=(10, 7))
c_all = 10 ** np.arange(-2, 6.1, 0.1)
T_test = np.array([100.0, 200.0])

for L, color in zip([50.0, 100.0, 200.0, 500.0, 1000.0], OKABE_ITO.values()):
    out = []
    for c_test in c_all:
        out.append(effective_transmissivity(T_test, np.array([c_test]), L))

    ax.plot(c_all, out, label=f"L={L}", color=color)

ax.set_xscale("log")
ax.axhline(y=T_test[0], color="r", label="$T_0$", ls="dashed")
ax.axhline(y=sum(T_test), color="k", label="$ \\sum T$", ls="dashed")
ax.legend()
ax.set_xlabel("c (d)")
ax.set_ylabel("Teff (m²/d)")
plt.show()

# %%
# Example: Comparison with recharge strip
# ---------------------------------------


def system_matrix(T, c_top, c_inter):
    """
    Multi-aquifer system matrix A for

        d2h/dx2 = A h - p(x)

    with a fixed-head top boundary through resistance c_top.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    c_inter = np.atleast_1d(np.asarray(c_inter, dtype=float))

    nlay = T.size
    if c_inter.size != nlay - 1:
        raise ValueError(
            f"len(c_inter) must equal len(T) - 1; got {c_inter.size} and {nlay}"
        )
    if np.any(T <= 0):
        raise ValueError("All T must be positive")
    if c_top <= 0:
        raise ValueError("c_top must be positive")
    if np.any(c_inter <= 0):
        raise ValueError("All c_inter must be positive")

    A = np.zeros((nlay, nlay), dtype=float)

    for i in range(nlay):
        if i == 0:
            leak_up = 1.0 / c_top
        else:
            leak_up = 1.0 / c_inter[i - 1]

        if i < nlay - 1:
            leak_down = 1.0 / c_inter[i]
        else:
            leak_down = 0.0

        A[i, i] = (leak_up + leak_down) / T[i]

        if i > 0:
            A[i, i - 1] = -1.0 / (T[i] * c_inter[i - 1])
        if i < nlay - 1:
            A[i, i + 1] = -1.0 / (T[i] * c_inter[i])

    return A


def recharge_strip_multilayer(T, c_top, c_inter, N, B, x):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    c_inter = np.atleast_1d(np.asarray(c_inter, dtype=float))
    x = np.asarray(x, dtype=float)

    nlay = T.size
    if c_inter.size != nlay - 1:
        raise ValueError(
            f"len(c_inter) must equal len(T) - 1; got {c_inter.size} and {nlay}"
        )

    A = system_matrix(T, c_top, c_inter)

    lam, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)

    lam = np.real_if_close(lam)
    if np.any(np.real(lam) <= 0):
        raise ValueError("System matrix has non-positive eigenvalues")

    s = np.sqrt(lam)

    f = np.zeros(nlay, dtype=float)
    f[0] = N / T[0]

    phi_p = np.linalg.solve(A, f)
    p = Vinv @ phi_p

    eB = np.exp(-B * s)
    shB = np.sinh(B * s)

    x_flat = x.ravel()
    out = np.empty((x_flat.size, nlay), dtype=np.result_type(V, p))

    inside = x_flat <= B
    outside = ~inside

    if np.any(inside):
        xi = x_flat[inside]
        coeff_in = p[None, :] - np.cosh(xi[:, None] * s[None, :]) * (eB * p)[None, :]
        out[inside] = coeff_in @ V.T

    if np.any(outside):
        xo = x_flat[outside]
        coeff_out = np.exp(-(xo[:, None] - B) * s[None, :]) * (shB * eB * p)[None, :]
        out[outside] = coeff_out @ V.T

    return np.real_if_close(out).reshape(x.shape + (nlay,)).transpose()


def recharge_strip_single_layer(T, c_top, N, B, x):
    """
    Single-layer semi-confined recharge-strip solution.

    Equation:
        T h'' - h/c + N(x) = 0
    with recharge N on |x|<B, zero outside, symmetry at x=0, h->0 at infinity.
    """
    lam = np.sqrt(T * c_top)
    x = np.asarray(x, dtype=float)
    h = np.empty_like(x)
    inside = x <= B
    outside = ~inside
    h[inside] = N * c_top * (1.0 - np.exp(-B / lam) * np.cosh(x[inside] / lam))
    h[outside] = N * c_top * np.exp(-x[outside] / lam) * np.sinh(B / lam)
    return h


N = 0.001
B = 500.0
x = np.linspace(0.0, 2500.0, 501)

h_multi = recharge_strip_multilayer(
    T=T,
    c_top=c_top,
    c_inter=c_inter,
    x=x,
    B=B,
    N=N,
)

L = B
T_eff = effective_transmissivity(T=T, c=c_inter, L=L)

h_eff = recharge_strip_single_layer(
    T=T_eff,
    c_top=c_top,
    x=x,
    B=B,
    N=N,
)
h_T0 = recharge_strip_single_layer(
    T=T[0],
    c_top=c_top,
    x=x,
    B=B,
    N=N,
)
h_Tsum = recharge_strip_single_layer(
    T=sum(T),
    c_top=c_top,
    x=x,
    B=B,
    N=N,
)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(x, h_multi[0], label="Multi-aquifer: aquifer 0", color=OKABE_ITO["black"])
ax.plot(
    x, h_eff, label=f"Reduced aquifer: $T_{{eff}}={T_eff:.1f}$", color=OKABE_ITO["blue"]
)
ax.plot(
    x,
    h_T0,
    "--",
    label=f"Reduced aquifer: $T_{0}={T[0]:.1f}$",
    color=OKABE_ITO["vermillion"],
)
ax.plot(
    x,
    h_Tsum,
    "--",
    label=f"Reduced aquifer: $ \\sum T={sum(T):.1f}$",
    color=OKABE_ITO["bluish_green"],
)

ax.axvspan(0.0, B, alpha=0.15, color="gray", label="Recharge strip")
ax.axvline(B, color="0.5", lw=1.0, ls=":")
ax.set_xlabel("Distance from strip center, x")
ax.set_ylabel("Head response")
ax.set_title(f"Recharge strip comparison: B={B:g}, L={L:g}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %%
# Spatial scales
# --------------

A_local = 0.20
A_regional = 1.00

L_local = 50.0  # local spatial scale [m]
L_regional = 500.0  # regional spatial scale [m]

# With h(x) = cos(x / L), the period is 2*pi*L.
# Two regional periods:
xmax = 4.0 * np.pi * L_regional
x = np.linspace(0.0, xmax, 4000)

# Components and superposition
h_local = A_local * np.cos(x / L_local)
h_regional = A_regional * np.cos(x / L_regional)
h_total = h_local + h_regional

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(x, h_total, color="black", lw=2.0)
ax.axhline(0.0, color="0.8", lw=0.9)

y_Lr = 1.5
y_Ll = 1.0

# Arrow for L_regional
ax.annotate(
    "",
    xy=(np.pi * L_regional, y_Lr),
    xytext=(3.0 * np.pi * L_regional, y_Lr),
    arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": "tab:blue"},
)
ax.text(
    2.0 * np.pi * L_regional,
    y_Lr + 0.05,
    "$L_r$",
    color=OKABE_ITO["blue"],
    ha="center",
    va="bottom",
    fontsize=14,
)

# Arrow for L_local
ax.annotate(
    "",
    xy=(12 * np.pi * L_local, y_Ll),
    xytext=(15 * np.pi * L_local, y_Ll),
    arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": "tab:orange"},
)
ax.text(
    13.5 * np.pi * L_local,
    y_Ll + 0.05,
    "$L_\\ell$",
    color=OKABE_ITO["vermillion"],
    ha="center",
    va="bottom",
    fontsize=14,
)

# Optional guide lines down to the axis
ax.plot(
    [12 * np.pi * L_local, 12 * np.pi * L_local],
    [0.0, y_Ll],
    color=OKABE_ITO["vermillion"],
    lw=0.9,
    ls=":",
)
ax.plot(
    [15 * np.pi * L_local, 15 * np.pi * L_local],
    [0.0, y_Ll],
    color=OKABE_ITO["vermillion"],
    lw=0.9,
    ls=":",
)

ax.plot(
    [np.pi * L_regional, np.pi * L_regional],
    [0.0, y_Lr],
    color=OKABE_ITO["blue"],
    lw=0.9,
    ls=":",
)
ax.plot(
    [3 * np.pi * L_regional, 3 * np.pi * L_regional],
    [0.0, y_Lr],
    color=OKABE_ITO["blue"],
    lw=0.9,
    ls=":",
)

ax.set_xlim(0.0, xmax)
ax.set_ylim(-1.45, 1.55)
ax.set_xlabel("horizontal distance x [m]")
ax.set_ylabel("Head[m]")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# %%
# Two effective layers
# --------------------

T = np.array([100.0, 200.0, 300.0, 400.0])
c_inter = np.array([600.0, 700.0, 800.0])
c_all = 10 ** np.arange(-2, 6.1, 0.1)
T_test = np.array([100.0, 200.0])

Tas = []
Tbs = []
cabs = []
for c_test in c_all:
    c = c_inter.copy()
    c[0] = c_test
    Ta, Tb, cab = two_layer_effective_transmisivity(T, c, 100.0, 1_000.0, 10_000.0)
    Tas.append(Ta)
    Tbs.append(Tb)
    cabs.append(cab)

Tas = np.asarray(Tas)
Tbs = np.asarray(Tbs)
cabs = np.asarray(cabs)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

Tas = np.asarray(Tas)
Tbs = np.asarray(Tbs)
cabs = np.asarray(cabs)

ax1.set_xscale("log")
ax1.plot(
    c_all,
    Tas,
    label=r"$T_a = T_\mathrm{eff}(L_\mathrm{local})$",
    color=OKABE_ITO["blue"],
)
ax1.plot(
    c_all,
    Tas + Tbs,
    label=r"$T_a + T_b = T_\mathrm{eff}(L_\mathrm{large})$",
    color=OKABE_ITO["bluish_green"],
)
ax1.plot(c_all, Tbs, label=r"$T_b$", color=OKABE_ITO["orange"])

ax1.axhline(y=T[0], color="r", label=r"$T_0$", ls="dashed")
ax1.axhline(y=np.sum(T), color="k", label=r"$\sum T$", ls="dashed")

ax1.set_ylabel(r"Transmissivity [m$^2$/d]")
ax1.legend()
ax1.grid(True, which="both", alpha=0.3)

# --- equivalent resistance ---
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.plot(c_all, cabs, label=r"$c_{ab}$", color=OKABE_ITO["blue"])

ax2.set_xlabel(r"Resistance $c_0$ [d]")
ax2.set_ylabel(r"$c_{ab}$ [d]")
ax2.legend()
ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()
