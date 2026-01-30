# %%

import numpy as np
import matplotlib.pyplot as plt

import respighi as rsp

# %%

ncell = 5
transmissivity = np.full(ncell, 1.0)
area = np.full(ncell, 1.0)

recharge = rsp.Recharge(rate=np.full(ncell, 0.0))

conductance = np.zeros(ncell)
conductance[0] = 1.0
conductance[-1] = 1.0
head = np.zeros(ncell)
head[0] = -1.0
head[-1] = 1.0
headboundary = rsp.HeadBoundary(
    conductance=conductance,
    head=head,
)
initial = np.zeros(ncell)

gwf = rsp.GroundwaterModel(
    area=area,
    initial=initial,
    recharge=recharge,
    head_boundaries=[headboundary],
    transmissivity=transmissivity,
)
gwf.formulate()
gwf.linear_solve()

# %%

plt.plot(gwf.head)

# %%

dh = gwf.head[-1] - gwf.head[0]

# %%

ncell = 5
transmissivity = np.full(ncell, 1.0)
area = np.full(ncell, 1.0)

recharge = rsp.Recharge(rate=np.full(ncell, 0.1))

conductance = np.zeros(ncell)
conductance[0] = 1.0
conductance[-1] = 1.0
head = np.zeros(ncell)
head[0] = 1.0
head[-1] = 1.0
headboundary = rsp.HeadBoundary(
    conductance=conductance,
    head=head,
)
initial = np.zeros(ncell)

gwf = rsp.GroundwaterModel(
    area=area,
    initial=initial,
    recharge=recharge,
    head_boundaries=[headboundary],
    transmissivity=transmissivity,
)
gwf.formulate()
gwf.linear_solve()

# %%

plt.plot(gwf.head)


# %%

ncell = 10
transmissivity = np.full(ncell, 1.0)
transmissivity[5:] = 2.0
area = np.full(ncell, 1.0)

recharge = rsp.Recharge(rate=np.full(ncell, 0.0))

conductance = np.zeros(ncell)
conductance[0] = 1.0
conductance[-1] = 1.0
head = np.zeros(ncell)
head[0] = -1.0
head[-1] = 1.0
headboundary = rsp.HeadBoundary(
    conductance=conductance,
    head=head,
)
initial = np.zeros(ncell)

gwf = rsp.GroundwaterModel(
    area=area,
    initial=initial,
    recharge=recharge,
    head_boundaries=[headboundary],
    transmissivity=transmissivity,
)
gwf.formulate()
gwf.linear_solve()
# %%

plt.plot(gwf.head)
# %%

ncell = 10
transmissivity = np.full(ncell, 1.0)
transmissivity[5:] = 2.0
area = np.full(ncell, 1.0)

recharge = rsp.Recharge(rate=np.full(ncell, 0.1))

conductance = np.zeros(ncell)
conductance[0] = 1.0
conductance[-1] = 1.0
head = np.zeros(ncell)
head[0] = -1.0
head[-1] = 1.0
drainage = rsp.Drainage(
    conductance=conductance,
    elevation=head,
)
initial = np.zeros(ncell)

gwf = rsp.GroundwaterModel(
    area=area,
    initial=initial,
    recharge=recharge,
    head_boundaries=[drainage],
    transmissivity=transmissivity,
)
gwf.formulate()
gwf.nonlinear_solve()
# %%

plt.plot(gwf.head)
# %%

targethead = gwf.head.copy()
targethead[:-1] = np.nan
target = rsp.GridSampling(targethead)
inverse = rsp.InverseProblem(gwf, target, 1.0)
# %%

inverse.formulate()
# %%
inverse.nonlinear_solve()

# %%
plt.plot(inverse.head)

# %%
# Conflicting heads

targethead = gwf.head.copy()
midhead = targethead[5]
h1 = targethead.copy()
h1[:] = np.nan
h1[5] = midhead + 1.0
h2 = targethead.copy()
h2[:] = np.nan
h2[5] = midhead - 1.0
target1 = rsp.GridSampling(h1)
target2 = rsp.GridSampling(h2)
inverse = rsp.InverseProblem(gwf, rsp.CompositeFittingTarget([target1, target2]), 1.0)
# %%

inverse.formulate()

# %%

inverse.linear_solve()
plt.plot(inverse.head)

# %%
inverse.nonlinear_solve()

# %%
plt.plot(inverse.head)
# %%
