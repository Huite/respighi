"""
Model head scaling
==================

This examples show a synthetic example:

* We do a regular run with the groundwater model to create
  a steady-state hydraulic head.
* Next, we regrid the hydraulic head to a ten times coarser spatial resolution.
* Then, we pose an inverse problem, and attempt to downscale (or in this case:
  reconstruct) the head.

"""
# %%

import numpy as np
import xarray as xr
import xugrid as xu
import matplotlib.pyplot as plt

import respighi as rsp

import os

os.chdir("Z:/src/respighi/examples")

# %%
# We load a number of boundary conditions, prepared as netCDF.

riverds = xr.open_dataset("testdata/river.nc").astype(np.float64)
tubeds = xr.open_dataset("testdata/tube.nc").astype(np.float64)
ditchds = xr.open_dataset("testdata/ditch.nc").astype(np.float64)
olf = xr.open_dataarray("testdata/overlandflow.nc").astype(np.float64)
transmissivity = xr.open_dataarray("testdata/transmissivity.nc").astype(np.float64)
# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.

river = rsp.River(
    conductance=riverds["conductance"].fillna(0.0).to_numpy(),
    stage=riverds["stage"].fillna(0.0).to_numpy(),
    elevation=riverds["bottom"].fillna(0.0).to_numpy(),
)
ditch = rsp.Drainage(
    conductance=ditchds["conductance"].fillna(0.0).to_numpy(),
    elevation=ditchds["elevation"].fillna(0.0).to_numpy(),
)
overlandflow = rsp.Drainage(
    conductance=xr.full_like(olf, 500.0).to_numpy(),
    elevation=olf.to_numpy(),
)
tube = rsp.Drainage(
    conductance=tubeds["conductance"].fillna(0.0).to_numpy(),
    elevation=tubeds["elevation"].fillna(0.0).to_numpy(),
)


stage = riverds["stage"]
headboundary = rsp.HeadBoundary(
    conductance=xr.full_like(stage, 10.0)
    .where(stage.notnull(), 0.0)
    .to_numpy()
    .ravel(),
    head=stage.fillna(0.0).to_numpy().ravel(),
)

# %%
# To make the pattern slightly more interesting, we will create
# a spatially (strongly) varying recharge rate.

rate = xr.full_like(transmissivity, 0.001)
rate = rate * np.sin(rate["x"] / 1000.0) + 0.001

recharge = rsp.Recharge(
    rate=rate.to_numpy().ravel(),
)

recharge = rsp.Recharge(
    rate=rate.to_numpy(),
)
gwf = rsp.GroundwaterModel(
    area=25.0 * 25.0,
    initial=xr.full_like(transmissivity, 0.0).to_numpy(),
    recharge=recharge,
    head_boundaries=[river, ditch, tube, overlandflow],
    transmissivity=transmissivity.to_numpy(),
    xclose=1e-6,
    maxiter=50,
)
gwf.formulate()
gwf.nonlinear_solve()

# %%
# Let's check the result.

fig, ax = plt.subplots()
head = transmissivity.copy(data=gwf.head.reshape(transmissivity.shape))
head.name = "Head"
head.plot(levels=30, ax=ax)
ax.set_aspect(1.0)

# %%
# Coarsening
# ----------
#
# We will create a coarse head grid with cells of 250 m by 250 m.
# Xarray stores midpoint values, so we add or subtract half of the cellsize.

xmin = riverds["x"].min().item() - 12.5
ymin = riverds["y"].min().item() - 12.5
xmax = riverds["x"].max().item() + 12.5
ymax = riverds["y"].max().item() + 12.5

# %%
# Now let's make sure 250.0 is a whole divisor of the new grid's coordinates.

dy = dx = 250.0
xnew = np.arange(np.ceil(xmin / dx) * dx + 0.5 * dx, np.floor(xmax / dx) * dx, dx)
ynew = np.arange(np.ceil(ymin / dx) * dx + 0.5 * dx, np.floor(ymax / dy) * dy, dy)[::-1]
template = xr.DataArray(
    data=np.zeros((ynew.size, xnew.size)),
    coords={"y": ynew, "x": xnew},
    dims=("y", "x"),
)
regridder = xu.OverlapRegridder(source=head, target=template)
coarsehead = regridder.regrid(head)

fig, ax = plt.subplots(figsize=(10, 5))
coarsehead.plot(ax=ax)
ax.set_aspect(1.0)

# %%
# Target
# ------
#
# We use this coarse grid to create a fitting target. For now, respighi
# requires an xugrid.Ugrid2d topology as the grid definition.

grid = xu.Ugrid2d.from_structured(transmissivity)
target = rsp.ModelTarget(coarsehead, grid)

# %%
# Inverse Problem
# ---------------
#
# With the groundwater model and the target, we can pose an inverse problem to solve.

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization_weight=1.0,
    maxiter=100,
    relax=0.0,
)

# %%
# Formulate separately, so we get an impression of the time (about 4 seconds).

inverse.formulate()

# %%
# Solve.

inverse.nonlinear_solve()
# %%

# Now let's check the reconstructed head and compare with the original.

rehead = head.copy(data=inverse.head.reshape(head.shape))

fig, axes = plt.subplots(nrows=4, figsize=(10, 17))
head.plot(ax=axes[0], levels=30)
coarsehead.plot(ax=axes[1], levels=30)
rehead.plot(ax=axes[2], levels=30)
(rehead - head).plot(ax=axes[3])
for ax in axes:
    ax.set_aspect(1.0)

# %%
# Let's also check the recharge rates.
#
rerate = rate.copy(data=inverse.recharge.reshape(rate.shape))

fig, axes = plt.subplots(nrows=3, figsize=(10, 13))
rate.plot(ax=axes[0], levels=30)
rerate.plot(ax=axes[1], levels=30)
(rerate - rate).plot(ax=axes[2])
for ax in axes:
    ax.set_aspect(1.0)
