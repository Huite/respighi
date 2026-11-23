"""
Horizontal Anisotropy
=====================

Basic axis aligned and non-axis aligned horizontal anisotropy.

This example shows horizontally anisotropic transmissivity for a single well
in a square domain, and how it interacts with a horizontal flow barrier.
"""
# %%

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
import xarray as xr

import respighi as rsp

# %%
# Set up a grid of 101 by 101 cells of 25 m. Rows run towards decreasing y,
# as in a raster; the model checks this, since the direction of the
# anisotropy depends on it.

cellsize = 25.0
nrow = ncol = 101
x = (np.arange(ncol) + 0.5) * cellsize
y = (np.arange(nrow) + 0.5)[::-1] * cellsize
like = xr.DataArray(np.ones((nrow, ncol)), coords={"y": y, "x": x}, dims=("y", "x"))

# %%
# The outer cells are kept at a head of 0.0 m by a stiff head boundary. A
# well in the centre extracts 1000 m3/d, given as a (negative) recharge rate
# in m/d.

edge = xr.zeros_like(like, dtype=bool)
edge[[0, -1], :] = True
edge[:, [0, -1]] = True
boundary = rsp.HeadBoundary(
    conductance=xr.where(edge, 1.0e4, 0.0).to_numpy(),
    head=xr.zeros_like(like).to_numpy(),
)

rate = xr.zeros_like(like)
rate[nrow // 2, ncol // 2] = -1000.0 / cellsize**2
recharge = rsp.Recharge(rate=rate.to_numpy())


def solve(transmissivity, barriers=()):
    gwf = rsp.GroundwaterModel(
        area=cellsize**2,
        initial=xr.zeros_like(like),
        recharge=recharge,
        head_boundaries=[boundary],
        transmissivity=transmissivity,
        horizontal_flow_barriers=barriers,
        linear_settings=rsp.PardisoSettings(),
    )
    gwf.nonlinear_solve()
    return gwf.head.isel(layer=0)


# %%
# For reference, an isotropic transmissivity of 500 m2/d: the drawdown
# contours are circles.

isotropic = solve(xr.full_like(like, 500.0))

# %%
# An anisotropic transmissivity is given by its principal values and the
# angle of the first principal axis, in degrees counter-clockwise from the
# x-axis. Here the aquifer is ten times more transmissive along a direction
# of 30 degrees: the contours become ellipses, elongated along that
# direction. The cross terms of the rotated tensor turn the usual 5-point
# stencil into a 9-point stencil (as in MODFLOW 6 XT3D).

anisotropic = rsp.AnisotropicTransmissivity(
    t1=xr.full_like(like, 500.0),
    t2=xr.full_like(like, 50.0),
    angle=30.0,
)
rotated = solve(anisotropic)

# %%
# Finally, add a horizontal flow barrier east of the well. The barrier is a
# line in a GeoDataFrame, snapped to the cell faces. Its resistance scales
# the whole flux across each face it is snapped to, cross terms included.

line = shapely.LineString([(1600.0, 700.0), (1600.0, 1800.0)])
barrier = rsp.HorizontalFlowBarrier.from_geodataframe(
    layer=0,
    barriers=gpd.GeoDataFrame({"resistance": [1.0]}, geometry=[line]),
    template=like,
    max_snap_distance=0.5 * cellsize,
)
barred = solve(anisotropic, barriers=[barrier])

# %%
# Let's compare the three. The 0.0 m contour is drawn in white.

levels = np.arange(-3.0, 0.51, 0.25)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, head, title in zip(
    axes,
    (isotropic, rotated, barred),
    ("Isotropic", "Anisotropic, 10:1 at 30°", "Anisotropic, with barrier"),
):
    filled = head.plot.contourf(
        ax=ax, levels=levels, extend="both", add_colorbar=False, cmap="viridis"
    )
    head.plot.contour(ax=ax, levels=levels, colors="k", linewidths=0.4)
    head.plot.contour(ax=ax, levels=[0.0], colors="w", linewidths=1.5)
    ax.set_title(title)
    ax.set_aspect(1.0)
axes[2].plot(*line.xy, color="red", linewidth=2.0)
fig.colorbar(filled, ax=axes, label="head (m)", shrink=0.8)

# %%
# The barrier diverts flow around its ends. Note also the heads *above*
# 0.0 m just east of the barrier: with only an extraction well and the edges
# at 0.0 m, that cannot be right. The cross terms of the faces next to the
# barrier estimate the tangential gradient by differencing across it, so
# the head jump over the barrier is mistaken for an aquifer gradient and
# drives spurious flow along it. MODFLOW 6 XT3D does the same; the effect
# grows with the barrier resistance and the degree of anisotropy.
