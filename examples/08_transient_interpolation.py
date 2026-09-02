"""
Transient IBRAHYM interpolation
===============================

The following example shows a test case:

* We load transient synthetic "piezometer data" sampled from IBRAHYM output from a netCDF file.
* Boundary conditions taken from IBRAHYM.

We interpolate a sequence of heads for the window.
"""
# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xugrid as xu

import respighi as rsp

# %%
# Set window for area of interest

XMIN = 185_000.0
XMAX = 205_000.0
YMIN = 350_000.0
YMAX = 370_000.0
START_DATE = "2018-01-01"
END_DATE = "2019-01-01"

# %%


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


ibrahym_head = slice_dataset(
    xr.open_dataset("../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
).sel(time=slice(START_DATE, END_DATE))
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

# %%
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.

transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), 2000.0)
storativity = xr.full_like(transmissivity, 0.15)

river = rsp.River.from_dataset(river_ds)
large_river = rsp.River.from_dataset(large_river_ds)
drain = rsp.Drainage.from_dataset(drain_ds)
tiledrain = rsp.Drainage.from_dataset(tiledrain_ds)
overlandflow = rsp.Drainage.from_dataset(overlandflow_ds, constant_conductance=500.0)
recharge = rsp.Recharge(
    rate=xr.full_like(transmissivity, 0.001).to_numpy(),
)

hfb = rsp.HorizontalFlowBarrier.from_geodataframe(
    layer=0,
    barriers=hfb_gdf,
    template=transmissivity,
    max_snap_distance=10.0,
)

# %%
# Trial run of the groundwater model.

gwf = rsp.GroundwaterModel(
    area=100.0 * 100.0,
    initial=ibrahym_head.isel(time=0),
    recharge=recharge,
    head_boundaries=[river, large_river, drain, tiledrain, overlandflow],
    transmissivity=transmissivity,
    storativity=storativity,
    horizontal_flow_barriers=[hfb],
)
gwf.formulate()
gwf.nonlinear_solve()
gwf.head.isel(layer=0).plot.contour(levels=30)

# %%
# Read transient results. Let's plot the locations on the map, and look at a few
# time series.

head_timeseries = xr.open_dataarray("synthetic_piezometers-100.nc")

# %%
# On the map:

fig, ax = plt.subplots()
gwf.head.isel(layer=0).plot.contour(levels=30, ax=ax)
ax.scatter(x=head_timeseries["x"], y=head_timeseries["y"], marker="o")

# %%
# Let's plot a sampling of the time series.

head_timeseries.isel(observation=slice(0, None, 10)).plot(hue="observation")

# %%
# Let's select a single year of data.

target_head = head_timeseries.sel(time=slice(START_DATE, END_DATE))
# %%
# We will now construct a transient target and estimate several interpolations
# over time.

PIEZOMETER_SIGMA = 0.1
grid = xu.Ugrid2d.from_structured(ibrahym_head)
target = rsp.CellSampling(
    x=target_head["x"],
    y=target_head["y"],
    head=target_head,
    grid=grid,
    sigma=PIEZOMETER_SIGMA,
)

# %%
inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=rsp.UnscaledMinimumCurvature(4000.0),
)

# %%
# We will interpolate values on the dates provided by the observation data set.
# The steady_state arrays determines whether a time step is computed with a steady-state
# or transient formulation.

time = target_head["time"]
steady = np.full(time.size - 1, True)
result = inverse.run(
    time=time,
    steady_state=steady,
    progress=True,
)

# %%
# Let's compute the difference with the actual model results and plot a moment in time:

difference = result["head"] - ibrahym_head
fig, ax = plt.subplots()
difference.isel(time=-1).plot(ax=ax, levels=np.arange(-2.0, 2.1, 0.2))
ax.scatter(x=head_timeseries["x"], y=head_timeseries["y"], marker="o")

# %%
# Also plot the interpolated head versus the measured head at ten sites.

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
for i, ax in enumerate(axes.ravel()):
    obs = target_head.isel(observation=i * 10)
    obs.plot(ax=ax)
    result["head"].isel(layer=0).sel(x=obs["x"], y=obs["y"], method="nearest").plot(
        ax=ax
    )
    ax.set_title(f"Observation {i * 10}")

# %%
# Show the mean absolute error per time step:

df = abs(difference).mean(("y", "x"))
df.plot()
