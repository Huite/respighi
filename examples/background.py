"""
Theoretical background & 1D Example
===================================

This example demonstrates the inverse problem on a simple one-dimensional
domain. It focuses on the mathematical formulation rather than a real-world
application.

We seek to estimate the steady-state phreatic surface h and spatially 
distributed recharge r by combining sparse piezometer observations with 
surface water boundary conditions, while enforcing the governing groundwater 
flow equation as a constraint.

The discretized steady-state groundwater flow equation on a structured grid is:

.. math::

    A h = Q r + b_{bc}

where:

- A is the conductance matrix (graph Laplacian plus head boundary contributions)
- Q = diag(Δx · Δy) maps recharge rates to volumetric fluxes
- b_bc contains contributions from head boundaries (e.g., rivers)

We pose the inverse problem as constrained optimization:

.. math::

    \\min_{h, r} \\; J(h, r) \\quad \\text{subject to} \\quad A h - Q r = b_{bc}

where the objective function is:

.. math::

    J(h, r) = \\frac{w_{obs}}{2} \\|P h - d\\|_{W}^{2} 
    + \\frac{w_{reg}}{2} \\|L_r r\\|_2^{2}

The sampling operator P extracts head values at the M observation locations, 
and d contains the observed piezometer heads.

The weighted norm is defined as:

.. math::

    \\|P h - d\\|_{W}^{2} = (P h - d)^T W (P h - d)

where W = diag(w_1, w_2, ..., w_M) is a diagonal matrix of observation weights. 
Typical choices include:

- w_i = 1/σ_i², where σ_i is the measurement uncertainty at location i
- w_i = 1 for uniform weighting (unweighted least squares)

The Laplacian operator L_r penalizes non-smooth recharge fields:

.. math::

    L_r = D_r - W_r

where D_r is the degree matrix and W_r the adjacency matrix of the grid graph.

Introducing Lagrange multipliers λ for the constraint yields the Lagrangian:

.. math::

    \\mathcal{L} = J(h, r) + \\lambda^T (A h - Q r - b_{bc})

Taking derivatives of the Lagrangian with respect to each variable:

.. math::

    \\frac{\\partial \\mathcal{L}}{\\partial h} &= w_{obs} P^T W (P h - d) + A^T \\lambda = 0

    \\frac{\\partial \\mathcal{L}}{\\partial r} &= w_{reg} L_r^T L_r r - Q^T \\lambda = 0

    \\frac{\\partial \\mathcal{L}}{\\partial \\lambda} &= A h - Q r - b_{bc} = 0

Rearranging the first two equations:

.. math::

    w_{obs} P^T W P h + A^T \\lambda &= w_{obs} P^T W d

    w_{reg} L_r^T L_r r - Q^T \\lambda &= 0

These give the symmetric KKT system shown below, where the (1,1) and (2,2) 
blocks are P^T W P and L_r^T L_r respectively.

.. math::

    \\begin{pmatrix}
        w_{obs} P^T W P & 0 & A^T \\\\
        0 & w_{reg} L_r^T L_r & -Q^T \\\\
        A & -Q & 0
    \\end{pmatrix}
    \\begin{pmatrix}
        h \\\\ r \\\\ \\lambda
    \\end{pmatrix}
    =
    \\begin{pmatrix}
        w_{obs} P^T W d \\\\
        0 \\\\
        b_{bc}
    \\end{pmatrix}

However, the blocks P^T W P and L_r^T L_r can become dense when P represents 
coarse-to-fine observation mappings. When the observation operator P represents a coarse model on a fine grid, each 
row of P may contain many nonzero entries. For example, if 100 fine cells map 
to 1 coarse observation cell, each row of P has 100 nonzeros. The product 
P^T W P densifies the (1,1) block, destroying sparsity and making direct 
solvers intractable for large problems.

To preserve sparsity, we introduce auxiliary variables that split the quadratic 
terms:

- e ∈ ℝ^M: observation residuals, e = P h - d
- s ∈ ℝ^N: regularization residuals, s = L_r r

The objective function is rewritten as:

.. math::

    J(h, r, e, s) = \\frac{w_{obs}}{2} \\|e\\|_2^2 + \\frac{w_{reg}}{2} \\|s\\|_2^2

subject to the constraints:

.. math::

    A h - Q r &= b_{bc} \\quad \\text{(PDE)}

    P h - e &= d \\quad \\text{(observation)}

    L_r r - s &= 0 \\quad \\text{(regularization)}

The Lagrangian introduces multipliers for each constraint:

.. math::

    \\mathcal{L} = \\frac{w_{obs}}{2} \\|e\\|_2^2 + \\frac{w_{reg}}{2} \\|s\\|_2^2
    + \\lambda^T (A h - Q r - b_{bc})
    + \\mu_e^T (P h - e - d)
    + \\mu_s^T (L_r r - s)

Taking derivatives with respect to each variable:

.. math::

    \\frac{\\partial \\mathcal{L}}{\\partial h} &= P^T \\mu_e + A^T \\lambda = 0

    \\frac{\\partial \\mathcal{L}}{\\partial r} &= L_r^T \\mu_s - Q^T \\lambda = 0

    \\frac{\\partial \\mathcal{L}}{\\partial e} &= w_{obs} e - \\mu_e = 0 
    \\quad \\Rightarrow \\quad \\mu_e = w_{obs} e

    \\frac{\\partial \\mathcal{L}}{\\partial s} &= w_{reg} s - \\mu_s = 0 
    \\quad \\Rightarrow \\quad \\mu_s = w_{reg} s

Substituting the expressions for μ_e and μ_s into the first two equations:

.. math::

    P^T (w_{obs} e) + A^T \\lambda &= 0

    L_r^T (w_{reg} s) - Q^T \\lambda &= 0

Combining all optimality conditions and constraints yields the block-structured 
system:

.. math::

    \\begin{pmatrix}
        0 & 0 & w_{obs} P^T & 0 & A^T \\\\
        0 & 0 & 0 & w_{reg} L_r^T & -Q^T \\\\
        A & -Q & 0 & 0 & 0 \\\\
        P & 0 & -I & 0 & 0 \\\\
        0 & L_r & 0 & -I & 0
    \\end{pmatrix}
    \\begin{pmatrix}
        h \\\\ r \\\\ e \\\\ s \\\\ \\lambda
    \\end{pmatrix}
    =
    \\begin{pmatrix}
        0 \\\\ 0 \\\\ b_{bc} \\\\ d \\\\ 0
    \\end{pmatrix}

The key advantage is that we never form the products:

- P^T W P (potentially dense)
- L_r^T L_r (potentially dense)

Instead, all matrix blocks remain sparse:

- P: sparse observation operator (e.g., aggregation from fine to coarse)
- L_r: sparse graph Laplacian
- A, Q: sparse PDE discretization matrices
- w_obs P^T, w_reg L_r^T: weighted transposes, still sparse

This maintains the sparse structure of the overall saddle-point system, enabling 
efficient direct solvers (e.g., sparse LU factorization) even for larger 
problems.

In code with auxiliary variables, construct the system as::

    # Auxiliary identity matrices
    I_e = sparse.eye(n_obs)  # shape (M, M)
    I_s = sparse.eye(n)      # shape (N, N)

    # Block matrix structure
    KKT = sparse.block_array([
        #    h,     r,     e,              s,               lambda
        [None,  None,  w_obs * P.T,    None,            A.T  ],  # dL/dh
        [None,  None,  None,           w_reg * L_r.T,  -Q.T  ],  # dL/dr
        [A,    -Q,     None,           None,            None ],  # PDE
        [P,     None, -I_e,            None,            None ],  # obs
        [None,  L_r,   None,          -I_s,             None ],  # reg
    ])

    rhs = np.concatenate([
        np.zeros(n),      # dL/dh
        np.zeros(n),      # dL/dr
        b_bc,             # PDE
        d,                # obs
        np.zeros(n),      # reg
    ])
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import respighi as rsp

# %%
# Forward problem: generating the "truth"
# ----------------------------------------
#
# We set up a 1D domain of 50 cells with unit transmissivity and unit cell
# area. Fixed head boundaries are applied at both ends (h=0 at left, h=0 at
# right). A spatially varying recharge drives the head distribution.

ncell = 50
transmissivity = np.full(ncell, 1.0)
area = np.full(ncell, 1.0)

# Spatially varying recharge: a smooth bump.
x = np.linspace(0, 1, ncell)
true_recharge = 0.5 * np.sin(np.pi * x)

recharge = rsp.Recharge(rate=true_recharge)

# Fixed head at both ends (conductance >> 1 pins the head value).
conductance = np.zeros(ncell)
conductance[0] = 1e6
conductance[-1] = 1e6
head_bc = np.zeros(ncell)
headboundary = rsp.HeadBoundary(conductance=conductance, head=head_bc)

gwf = rsp.GroundwaterModel(
    area=area,
    initial=np.zeros(ncell),
    recharge=recharge,
    head_boundaries=[headboundary],
    transmissivity=transmissivity,
)
gwf.formulate()
gwf.linear_solve()

# %%
# The forward solution produces the "true" head.

fig, axes = plt.subplots(nrows=2, sharex=True)
axes[0].plot(x, gwf.head, "k-", label="True head")
axes[0].set_ylabel("Head")
axes[0].legend()
axes[1].plot(x, true_recharge, "k-", label="True recharge")
axes[1].set_ylabel("Recharge")
axes[1].set_xlabel("x")
axes[1].legend()
fig.suptitle("Forward solution ('truth')")

# %%
# Inverse problem: recovering head from sparse observations
# ----------------------------------------------------------
#
# Suppose we only observe the head at 5 locations. The inverse problem must
# reconstruct the full head field *and* estimate the recharge, while
# satisfying the governing PDE as a hard constraint.
#
# The observation operator :math:`P` is a sparse matrix that picks out the
# observed cells from the full head vector.

observed_indices = np.array([5, 15, 25, 35, 45])
target_head = np.full(ncell, np.nan)
target_head[observed_indices] = gwf.head[observed_indices]
target = rsp.GridSampling(target_head)

# %%
# Effect of observation density
# -----------------------------
#
# More observations constrain the problem better, reducing the reliance on
# regularization to fill in the gaps.

observation_sets = {
    "1 point": np.array([25]),
    "3 points": np.array([10, 25, 40]),
    "5 points": np.array([5, 15, 25, 35, 45]),
    "10 points": np.linspace(2, 47, 10, dtype=int),
    "25 points": np.linspace(1, 48, 25, dtype=int),
}

alpha = 1.0
fig, axes = plt.subplots(nrows=2, ncols=len(observation_sets), figsize=(14, 5), sharex=True)

for col, (label, obs_idx) in enumerate(observation_sets.items()):
    target_head = np.full(ncell, np.nan)
    target_head[obs_idx] = gwf.head[obs_idx]
    target = rsp.GridSampling(target_head)

    inverse = rsp.InverseProblem(
        groundwatermodel=gwf,
        target=target,
        regularization_weight=alpha,
    )
    inverse.formulate()
    inverse.linear_solve()

    ax_h = axes[0, col]
    ax_r = axes[1, col]

    ax_h.plot(x, gwf.head, "k-", alpha=0.3, label="Truth")
    ax_h.plot(x, inverse.head, "C0-", label="Inverse")
    ax_h.plot(x[obs_idx], gwf.head[obs_idx], "ko", ms=4, label="Observed")
    ax_h.set_title(label)

    ax_r.plot(x, true_recharge, "k-", alpha=0.3, label="Truth")
    ax_r.plot(x, inverse.recharge, "C1-", label="Inverse")

axes[0, 0].set_ylabel("Head")
axes[1, 0].set_ylabel("Recharge")
for ax in axes[1]:
    ax.set_xlabel("x")
axes[0, -1].legend(fontsize="small")
axes[1, -1].legend(fontsize="small")
fig.suptitle(rf"Effect of observation density ($\alpha$ = {alpha})")
fig.tight_layout()


# %%
# Lagrange multipliers
# --------------------
#
# The Lagrange multipliers :math:`\lambda` enforce the PDE constraint
# :math:`A h - Q r = b_{bc}`. They can be interpreted as the sensitivity of
# the objective function to perturbations in the constraint: where
# :math:`|\lambda|` is large, the PDE constraint is "actively" pulling the
# solution away from a pure data fit.

obs_idx = np.array([5, 15, 25, 35, 45])
target_head = np.full(ncell, np.nan)
target_head[obs_idx] = gwf.head[obs_idx]
target = rsp.GridSampling(target_head)

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization_weight=1.0,
)
inverse.formulate()
inverse.linear_solve()

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))
axes[0].plot(x, inverse.head, "C0-")
axes[0].plot(x[obs_idx], gwf.head[obs_idx], "ko", ms=5)
axes[0].set_ylabel("Head")
axes[1].plot(x, inverse.recharge, "C1-")
axes[1].set_ylabel("Recharge")
axes[2].plot(x, inverse.lagrangian, "C2-")
axes[2].set_ylabel(r"Lagrange multiplier $\lambda$")
axes[2].set_xlabel("x")
fig.suptitle("Inverse solution with Lagrange multipliers")
fig.tight_layout()

# %%