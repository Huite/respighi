"""
Transient IBRAHYM-Droogteportaal interpolation
==============================================

The following example shows a use case with real world data:

* Piezometer data downloaded from the Droogteportaal.
* Boundary conditions taken from IBRAHYM.

We interpolate a series of timestamps for the window.
"""
# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    initial=np.full_like(transmissivity, 20.0),
    recharge=recharge,
    head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
    transmissivity=transmissivity,
    storativity=np.full_like(transmissivity, 0.15),
    horizontal_flow_barriers=[hfb],
    xclose=1e-5,
    maxiter=50,
)
gwf.formulate()
gwf.nonlinear_solve()
gwf.head.isel(layer=0).plot.contour(levels=30)

# %%
# Inverse Problem
# ---------------
# We will start by making a respighi Target using the pre-processed piezometers
# data.

head = xr.open_dataarray("../tmp-scripts/transient-observations.nc")
grid = xu.Ugrid2d.from_structured(modelhead)
sigma = np.full(head.shape[1], PIEZOMETER_SIGMA)
target = rsp.CellSampling(
    x=head["x"],
    y=head["y"],
    head=head,
    grid=grid,
    sigma=sigma,
)
# %%

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=rsp.UnscaledMinimumCurvature(100.0),
    maxiter=10,
    maxdh=0.001,
)
time = pd.date_range("2025-10-01", "2026-04-01")
steady = np.full(time.size - 1, False)
steady[0] = True
result = inverse.run(
    time=time,
    steady_state=steady,
)

# %%
# Show the match with the observations for nine piezometers:

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axes.ravel()):
    obs = head.isel(observation=i * 10).sel(time=slice(None, "2026-04-01"))
    obs.plot(ax=ax)
    result["head"].isel(layer=0).sel(x=obs["x"], y=obs["y"], method="nearest").plot(
        ax=ax
    )
    ax.set_title(f"Observation {i}")
    fig.savefig("transient.png", dpi=200)

# %%
