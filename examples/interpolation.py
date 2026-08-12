"""
Piezometer Interpolation
========================

This examples show a synthetic example:

* We do a regular run with the groundwater model to create
  a steady-state hydraulic head.
* Next, we take the head at 100 random sites; these represent
  "piezometers".
* Then, we pose an inverse problem, and attempt to interpolate
  (or in this case: reconstruct) the head.
"""
# %%

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu

import respighi as rsp

# %%
# We load a number of boundary conditions, prepared as netCDF.

riverds = xr.open_dataset("testdata/river.nc").astype(np.float64)
riverds = riverds.rename({"bottom": "bottom_elevation"})
tubeds = xr.open_dataset("testdata/tube.nc").astype(np.float64)
ditchds = xr.open_dataset("testdata/ditch.nc").astype(np.float64)
olfds = xr.open_dataset("testdata/overlandflow.nc").astype(np.float64)
transmissivity = xr.open_dataarray("testdata/transmissivity.nc").astype(np.float64)

# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.


WIDTH = 1e-2
river = rsp.River.from_dataset(riverds, smoothing_width=WIDTH)
ditch = rsp.Drainage.from_dataset(ditchds, smoothing_width=WIDTH)
tube = rsp.Drainage.from_dataset(tubeds, smoothing_width=WIDTH)
overlandflow = rsp.Drainage.from_dataset(
    olfds, constant_conductance=500.0, smoothing_width=WIDTH
)

# %%
# To make the pattern slightly more interesting, we will create
# a spatially (strongly) varying recharge rate.

rate = xr.full_like(transmissivity, 0.001)
rate = rate * np.sin(rate["x"] / 1000.0) + 0.001

recharge = rsp.Recharge(
    rate=rate.to_numpy().ravel(),
)

# %%
# We will now initialize and run the model.

gwf = rsp.GroundwaterModel(
    area=25.0 * 25.0,
    initial=xr.full_like(transmissivity, 0.0),
    recharge=recharge,
    head_boundaries=[river, ditch, tube, overlandflow],
    transmissivity=transmissivity,
    xclose=1e-6,
    maxiter=30,
    relax=0.0,
)
gwf.formulate()
gwf.nonlinear_solve()

# %%
# Let's check the result.

fig, ax = plt.subplots()
head = gwf.head.isel(layer=0).copy()
head.plot(levels=30, ax=ax)
ax.set_aspect(1.0)

# %%
# Piezometers
# -----------
#
# We will do random sampling, and select 100 sites.

xmin = riverds["x"].min().item()
ymin = riverds["y"].min().item()
xmax = riverds["x"].max().item()
ymax = riverds["y"].max().item()

rng = np.random.default_rng()
nsites = 100
x = xmin + (xmax - xmin) * rng.random(nsites)
y = ymin + (ymax - ymin) * rng.random(nsites)
headvalues = head.sel(x=xr.DataArray(x), y=xr.DataArray(y), method="nearest").to_numpy()

# Target
# ------
#
# We use these sites to create a fitting target.
# For now, respighi requires an xugrid.Ugrid2d topology as the grid definition.

grid = xu.Ugrid2d.from_structured(transmissivity)
target = rsp.CellSampling(x, y, headvalues, grid)

# Inverse Problem
# ---------------
#
# With the groundwater model and the target, we can pose an inverse problem to solve.

gwf._head[:] = 0.0
inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=rsp.UnscaledMinimumCurvature(100.0),
    maxiter=30,
)

# %%
# Formulate separately, so we get an impression of the time (about 4 seconds).

inverse.formulate()

# %%
# Solve.

inverse.nonlinear_solve()

# %%
# Now let's check the reconstructed head and compare with the original.

rehead = inverse.head.isel(layer=0)

fig, axes = plt.subplots(nrows=3, figsize=(10, 13))
head.plot(ax=axes[0], levels=30)
rehead.plot(ax=axes[1], levels=30)
axes[1].scatter(x=x, y=y, alpha=0.50, color="k")
(rehead - head).plot(ax=axes[2])
for ax in axes:
    ax.set_aspect(1.0)
# %%


# %%
# Let's also check the recharge rates.
#
fig, axes = plt.subplots(nrows=3, figsize=(10, 13))
rate.plot(ax=axes[0], levels=30)
inverse.recharge.plot(ax=axes[1], levels=30)
(inverse.recharge - rate).plot(ax=axes[2])
for ax in axes:
    ax.set_aspect(1.0)

# %%
# Finally, let's have a look at the Langrangian

fig, ax = plt.subplots(figsize=(10, 4))
inverse.lagrangian.plot(ax=ax)
ax.set_aspect(1.0)

# %%
