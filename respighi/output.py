import contextlib

import dask.array
import numpy as np
import xarray as xr
import zarr


def initialize_zarr_store(path, time, dims, coords, mode="w"):
    """Use Xarray to initialize a Zarr store."""
    spatial_shape = tuple(coords[dim].size for dim in dims)
    xy_shape = spatial_shape[1:]
    ntime = len(time)

    template = xr.Dataset(
        {
            "head": (
                ("time", *dims),
                dask.array.empty(
                    (ntime, *spatial_shape),
                    chunks=(1, *spatial_shape),
                    dtype=float,
                ),
            ),
            "recharge": (
                ("time", *dims[1:]),
                dask.array.empty(
                    (ntime, *xy_shape),
                    chunks=(1, *xy_shape),
                    dtype=float,
                ),
            ),
            "converged": (
                ("time",),
                dask.array.zeros(
                    ntime,
                    chunks=(1,),
                    dtype=bool,
                ),
            ),
            "iterations": (
                ("time",),
                dask.array.zeros(
                    ntime,
                    chunks=(1,),
                    dtype=int,
                ),
            ),
        },
        coords={
            "time": time,
            **coords,
        },
    )

    encoding = {
        "head": {"_FillValue": np.nan},
        "recharge": {"_FillValue": np.nan},
    }

    # Create the Zarr arrays + Xarray metadata, but don't write
    # the dummy data arrays.
    template.to_zarr(
        path,
        mode=mode,
        compute=False,
        consolidated=False,
        encoding=encoding,
    )
    return


@contextlib.contextmanager
def zarr_writer(path, time, dims, coords: dict, mode: str = "w"):
    """Initialize a store and yield an open zarr group for direct array writes."""
    initialize_zarr_store(path, time, dims, coords, mode=mode)
    group = zarr.open_group(path, mode="r+")
    try:
        yield group
    finally:
        zarr.consolidate_metadata(path)
