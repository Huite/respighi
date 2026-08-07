"""
IBRAHYM-Droogteportaal interpolation
====================================

The following example shows a use case with real world data:

* Piezometer data downloaded from the Droogteportaal.
* Boundary conditions taken from IBRAHYM.

We interpolate a mean head for the window, and estimate some
first-order uncertainty.
"""
# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu

import respighi as rsp

# %%

XMIN = 185_000.0
XMAX = 205_000.0
YMIN = 350_000.0
YMAX = 370_000.0
PIEZOMETER_SIGMA = 0.1
BOUNDARY_SIGMA = 0.2


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


# %%
# IBRAHYM data
# ------------
#
# We load some IBRAHYM data from a number of prepared netCDF files, and slice to the given window.

head = xr.open_dataset("../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
modelhead = slice_dataset(head.isel(time=-1))
drain_ds = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-drains-100m.nc"))
overlandflow_ds = slice_dataset(
    xr.open_dataset("../case/ibrahym/ibrahym-overlandflow-100m.nc")
)
river_ds = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-rivers-100m.nc"))
large_river_ds = slice_dataset(
    xr.open_dataset("../case/ibrahym/ibrahym-largerivers-100m.nc")
)
tiledrain_ds = slice_dataset(
    xr.open_dataset("../case/ibrahym/ibrahym-tiledrainage-100m.nc")
)
subsoil = slice_dataset(xr.open_dataset("../case/ibrahym/ibrahym-subsoil-100m.nc"))
hfb_gdf = gpd.read_file("../case/ibrahym/hfb-12.gpkg")

# Select the winter data
river_ds = river_ds.isel(time=0)

# Create a transmissivity
transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), 10_000.0)

# %%
# Model test run
# --------------
#
# Do a basic test run to check the model data.
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.


WIDTH = 0.01

transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), 3000.0)

river = rsp.River.from_dataset(
    river_ds, constant_sigma=BOUNDARY_SIGMA, smoothing_width=WIDTH
)
large_river = rsp.River.from_dataset(
    large_river_ds, constant_sigma=BOUNDARY_SIGMA, smoothing_width=WIDTH
)
drain = rsp.Drainage.from_dataset(
    drain_ds, constant_sigma=BOUNDARY_SIGMA, smoothing_width=WIDTH
)
tiledrain = rsp.Drainage.from_dataset(
    tiledrain_ds, constant_sigma=BOUNDARY_SIGMA, smoothing_width=WIDTH
)
overlandflow = rsp.Drainage.from_dataset(
    overlandflow_ds,
    constant_conductance=500.0,
    constant_sigma=BOUNDARY_SIGMA,
    smoothing_width=WIDTH,
)
recharge = rsp.Recharge(
    rate=xr.full_like(transmissivity, 0.001).to_numpy(),
)

hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
    layer=0,
    barriers=hfb_gdf,
    template=transmissivity,
    max_snap_distance=10.0,
)

gwf = rsp.GroundwaterModel(
    area=100.0 * 100.0,
    initial=modelhead,
    recharge=recharge,
    head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
    transmissivity=transmissivity,
    horizontal_flow_barriers=[hfb],
    xclose=1e-6,
    maxiter=50,
)
gwf.formulate()
gwf.nonlinear_solve()
gwf.head.isel(layer=0).plot.contour(levels=30)

# %%
# Inverse Problem
# ---------------
# We will start by making a respighi Target using the mean head values from the piezometers.

piezometers = gpd.read_file("../tmp-scripts/ibrahym-patch-piezometers.gpkg")
piezometers = piezometers.loc[piezometers["mean_head"].notnull()]
x = piezometers.geometry.x.to_numpy()
y = piezometers.geometry.y.to_numpy()
grid = xu.Ugrid2d.from_structured(modelhead)
head = piezometers["mean_head"].to_numpy()
sigma = np.full_like(head, PIEZOMETER_SIGMA)
target = rsp.CellSampling(x, y, piezometers["mean_head"], grid, sigma=sigma)

# %%
# With the groundwater model and the target, we can pose an inverse problem to solve.

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=rsp.UnscaledMinimumCurvature(1000.0),
    maxiter=30,
    maxdh=0.001,
)
inverse.formulate()
inverse.nonlinear_solve()

# %%
# Let's check the resulting head.

inversehead = inverse.head.isel(layer=0)
fig, ax = plt.subplots(figsize=(10, 10))
cs = inversehead.plot.contour(ax=ax, levels=np.arange(15.0, 60.0, 2.0))
ax.clabel(cs, inline=True, fontsize=8)
ax.scatter(x, y, color="k", alpha=0.5)

for xi, yi, zi in zip(x, y, target.d):
    ax.annotate(
        f"{zi:.2f}", xy=(xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=7
    )

ax.set_aspect(1.0)

# %%
# Uncertainty estimate
# --------------------
#
# We will also make an attempt to estimate the uncertainty given
# the a priori provided estimates (0.1 for piezometers, 0.2 for boundary conditions):

variance = inverse.estimate_variance(
    batch_size=64,
)
sigma = np.sqrt(variance).isel(layer=0)

# %%
# And make a plot.

fig, ax = plt.subplots(figsize=(7, 7))
sigma.plot.contourf(ax=ax, levels=np.arange(0.1, 1.05, 0.05), cmap="turbo")
ax.set_aspect(1.0)
ax.scatter(x, y, color="white", alpha=1.0, marker="^")
ax.set_aspect(1.0)
