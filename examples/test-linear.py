# %%

import numpy as np
import xarray as xr
import respighi as rsp
import matplotlib.pyplot as plt
import xugrid as xu
import scipy

# %%

import os

os.chdir("Z:/src/respighi/examples")

# %%

stage = (
    xr.open_dataarray("../riv-stage.nc")
    .isel(y=slice(0, 280), x=slice(0, 600))
    .drop_vars(("dx", "dy"))
)
stage.name = None

xmin = stage.x.min().item()
xmax = stage.x.max().item()
ymin = stage.y.min().item()
ymax = stage.y.max().item()

# %%

river = rsp.HeadBoundary(
    conductance=xr.full_like(stage, 10.0)
    .where(stage.notnull(), 0.0)
    .to_numpy()
    .ravel(),
    head=stage.fillna(0.0).to_numpy().ravel(),
)
initial = xr.full_like(
    stage, stage.mean()
)  # .sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
rate = xr.full_like(stage, 0.00005) * np.sin(stage["x"] / 1000.0)
# %%

recharge = rsp.Recharge(
    # rate=xr.full_like(stage, 0.00005).to_numpy().ravel(),
    rate=rate.to_numpy().ravel(),
)

model = rsp.GroundwaterModel(
    area=25.0 * 25.0,
    recharge=recharge,
    head_boundaries=[river],
    initial=initial.to_numpy(),
)

model.formulate()
# %%
model.linear_solve()  # 0.5 s, 170_000 cells in 78 iterations
# model.nonlinear_solve()
# %%

head = stage.copy(data=model.head.reshape(stage.shape))
fig, axes = plt.subplots(2, 2, figsize=(25, 12))
(ax0, ax1, ax2, ax3) = axes.ravel()
rate.plot.imshow(ax=ax0)
head.plot.imshow(ax=ax1, levels=30)
ax0.set_aspect(1.0)
ax1.set_aspect(1.0)

rate.isel(y=80).plot(ax=ax2)
head.isel(y=80).plot(ax=ax3)


ax0.set_title("R*")
ax1.set_title("Head")
ax2.set_title("R* (y=562 000)")
ax3.set_title("Head (y=562 000")

# %%

rng = np.random.default_rng()
nsites = 100
x = xmin + (xmax - xmin) * rng.random(nsites)
y = ymin + (ymax - ymin) * rng.random(nsites)
headvalues = head.sel(x=xr.DataArray(x), y=xr.DataArray(y), method="nearest").to_numpy()
# %%
grid = xu.Ugrid2d.from_structured(head)

# %%
sampling_target = rsp.CellSampling(x, y, headvalues, grid)

# %%

inverse = rsp.InverseProblem(
    groundwatermodel=model,
    target=sampling_target,
    regularization_weight=1.0,
)
# %%

inverse.formulate()

# %%

inverse.linear_solve()

# inverse.nonlinear_solve()

# %%

invhead = stage.copy(data=inverse.head.reshape(stage.shape))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
(ax0, ax1, ax2, ax3) = axes.ravel()

head.plot.imshow(ax=ax0, levels=30)
ax0.scatter(x=x, y=y, s=15, alpha=0.5)
invhead.plot.imshow(ax=ax1, levels=30)


invrate = stage.copy(data=inverse.recharge.reshape(stage.shape))
invrate.plot.imshow(ax=ax2)

diff = invhead - head
diff.plot.imshow(ax=ax3)

for ax in axes.ravel():
    ax.set_aspect(1.0)

ax0.set_title("Head")
ax1.set_title("Estimated Head")
ax2.set_title("Estimated R*")
ax3.set_title("Head Error")
# %%


# %%

xmin = stage.x.min() - 12.5 + 125.0
xmax = stage.x.max()
ymin = stage.y.min()
ymax = stage.y.max() + 12.5 - 125.0
tx = np.arange(xmin, xmax, 250.0)
ty = np.arange(ymax, ymin, -250.0)
template = xr.DataArray(
    np.zeros((ty.size, tx.size)), coords={"y": ty, "x": tx}, dims=("y", "x")
)

# %%

regridder = xu.OverlapRegridder(source=head, target=template)
coarse = regridder.regrid(head)
coarsegrid = xu.Ugrid2d.from_structured(coarse)
# %%

target = rsp.ModelTarget(head=coarse, grid=xu.Ugrid2d.from_structured(head))

# %%

inverse = rsp.InverseProblem(
    groundwatermodel=model,
    target=sampling_target,
    regularization_weight=1.0,
)
# %%

inverse.formulate()
# %%

inverse.linear_solve()
# %%

invhead = stage.copy(data=inverse.head.reshape(stage.shape))
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(20, 10))
invhead.plot.imshow(ax=ax0, levels=30)
coarsegrid.plot(ax=ax0, color="k", alpha=0.5, lw=0.5)

head.plot.imshow(ax=ax1, levels=30)

invrate = stage.copy(data=inverse.recharge.reshape(stage.shape))
invrate.plot.imshow(ax=ax2)

for ax in (ax0, ax1, ax2):
    ax.set_aspect(1.0)

# %%

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
(ax0, ax1, ax2, ax3) = axes.ravel()

coarse.sel(x=slice(238_000, 240_000)).sel(y=slice(564_000.0, 562_000.0)).plot(ax=ax0)
ax0.hlines(xmin=238_000.0, xmax=240_000.0, y=563000.0, color="r")
head.sel(x=slice(238_000, 240_000)).sel(y=slice(564_000.0, 562_000.0)).plot.contour(
    ax=ax1
)
ax1.hlines(xmin=238_000.0, xmax=240_000.0, y=563000.0, color="r")

ax0.set_aspect(1.0)
ax1.set_aspect(1.0)

invhead.sel(x=slice(238_000, 240_000)).sel(y=563000.0, method="nearest").plot.step(
    ax=ax2, lw=2, where="mid"
)
head.sel(x=slice(238_000, 240_000)).sel(y=563000.0, method="nearest").plot.step(
    ax=ax2, ls="-", where="mid"
)
coarse.sel(x=slice(238_000, 240_000)).sel(y=563000.0, method="nearest").plot.step(
    ax=ax2, where="mid"
)

invrate.sel(x=slice(238_000, 240_000)).sel(y=slice(564_000.0, 562_000.0)).plot(ax=ax3)
ax3.set_aspect(1.0)

ax0.set_title("Head coarse")
ax1.set_title("Estimated head")
ax2.set_title("Heads (y=563_000)")
ax3.set_title("R*")

# %%

half_coarse = coarse.sel(x=slice(244_000.0, None))
halfcoarsegrid = xu.Ugrid2d.from_structured(half_coarse)
select = x < 244_000.0
headvalues = head.sel(
    x=xr.DataArray(x[select]), y=xr.DataArray(y[select]), method="nearest"
).to_numpy()
# %%

coarsetarget = rsp.ModelTarget(head=half_coarse, grid=xu.Ugrid2d.from_structured(head))
pointtarget = rsp.CellSampling(x=x[select], y=y[select], head=headvalues, grid=grid)
target = rsp.CompositeFittingTarget([coarsetarget, pointtarget])
# %%

inverse = rsp.InverseProblem(
    groundwatermodel=model,
    target=target,
    regularization_weight=1.0,
)
# %%

inverse.formulate()
# %%

inverse.linear_solve()
# %%

invhead = stage.copy(data=inverse.head.reshape(stage.shape))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
(ax0, ax1, ax2, ax3) = axes.ravel()

coarse.plot.imshow(ax=ax0, levels=30)

invhead.plot.imshow(ax=ax1, levels=30)
halfcoarsegrid.plot(ax=ax1, color="k", alpha=0.5, lw=0.5)
ax1.scatter(x=x[select], y=y[select], s=15, alpha=0.5)
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)

(invhead - head).plot.imshow(ax=ax3)

ax0.set_title("Coarse head")
ax1.set_title("Estimated head")
ax2.set_title("R*")
ax3.set_title("Head error")

invrate = stage.copy(data=inverse.recharge.reshape(stage.shape))
invrate.plot.imshow(ax=ax2)

for ax in (ax0, ax1, ax2):
    ax.set_aspect(1.0)

# %%

diff = head - invhead
# %%
