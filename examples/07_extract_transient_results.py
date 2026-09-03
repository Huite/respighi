# %%

import numpy as np
import xarray as xr

# %%
XMIN = 185_000.0
XMAX = 205_000.0
YMIN = 350_000.0
YMAX = 370_000.0
N_PIEZOMETERS = 100

# %%


def slice_dataset(ds):
    return ds.sel(x=slice(XMIN, XMAX), y=slice(YMAX, YMIN))


ibrahym_head = slice_dataset(
    xr.open_dataset("../case/ibrahym/ibrahym-head-l1-100m.nc")["head"]
).compute()

# %%


def make_piezometers(n_piezometers, xmin, xmax, ymin, ymax):
    rng = np.random.default_rng(seed=12345)
    x = xmin + (xmax - xmin) * rng.random(n_piezometers)
    y = ymin + (ymax - ymin) * rng.random(n_piezometers)
    return x, y


x, y = make_piezometers(
    n_piezometers=N_PIEZOMETERS,
    xmin=XMIN,
    xmax=XMAX,
    ymin=YMIN,
    ymax=YMAX,
)

# %%

head_timeseries = ibrahym_head.sel(
    x=xr.DataArray(x, dims=["observation"]),
    y=xr.DataArray(y, dims=["observation"]),
    method="nearest",
)
head_timeseries.to_netcdf(f"synthetic_piezometers-{N_PIEZOMETERS}.nc")

# %%
