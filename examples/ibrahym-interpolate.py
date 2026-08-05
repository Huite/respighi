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
# XMIN = 170_000.0
# XMAX = XMIN + 15_000.0
# YMIN = 360_000.0
# YMAX = YMIN + 15_000.0
N_PIEZOMETERS = 500

# %%


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


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
# Initialize the relevant boundary condition classes, initialize the
# groundwater model, formulate, then solve.


transmissivity = xr.full_like(subsoil["kh"].isel(layer=0, drop=True), 1000.0)

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
# Make synthetic piezometers


def make_piezometers(n_piezometers, xmin, xmax, ymin, ymax):
    rng = np.random.default_rng(seed=12345)
    x = xmin + (xmax - xmin) * rng.random(n_piezometers)
    y = ymin + (ymax - ymin) * rng.random(n_piezometers)
    return x, y


x, y = make_piezometers(
    n_piezometers=200,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX,
)
# %%
# Check where they are located
fig, ax = plt.subplots()
modelhead.plot(ax=ax)
ax.scatter(x, y, color="k", alpha=0.5)

# %%
# Make a respighi Target

grid = xu.Ugrid2d.from_structured(modelhead)
headvalues = modelhead.sel(
    x=xr.DataArray(x), y=xr.DataArray(y), method="nearest"
).to_numpy()
target = rsp.CellSampling(x, y, headvalues, grid)

# %%
# Inverse Problem
# ---------------
#
# With the groundwater model and the target, we can pose an inverse problem to solve.

inverse = rsp.InverseProblem(
    groundwatermodel=gwf,
    target=target,
    regularization=rsp.UnscaledMinimumCurvature(1000.0),
    maxiter=100,
    relax=0.0,
)

# %%

inverse.formulate()

# %%
# Solve.

inverse.nonlinear_solve()
# %%

inversehead = inverse.head.isel(layer=0)
inversehead.plot.imshow()
# %%

error = inversehead - modelhead
print(abs(error).mean())

# %%

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(15, 7))
modelhead.plot.contour(ax=ax0, levels=30)
inversehead.plot.contour(ax=ax1, levels=30)
ax0.set_aspect(1.0)
ax1.set_aspect(1.0)

# %%

error.plot.imshow(levels=np.arange(-1.0, 1.0, 0.1))
print(abs(error).mean())

# %%
