# %%

import numpy as np

import respighi as rsp
import xarray as xr
import imod

# %%

riverds = xr.open_dataset("river.nc").astype(np.float64)
tubeds = xr.open_dataset("tube.nc").astype(np.float64)
ditchds = xr.open_dataset("ditch.nc").astype(np.float64)
olf = xr.open_dataarray("overlandflow.nc").astype(np.float64)
transmissivity = xr.open_dataarray("transmissivity.nc").astype(np.float64)
# %%

river = rsp.River(
    conductance=riverds["conductance"].fillna(0.0).to_numpy(),
    stage=riverds["stage"].fillna(0.0).to_numpy(),
    elevation=riverds["bottom"].fillna(0.0).to_numpy(),
)
# %%

ditch = rsp.Drainage(
    conductance=ditchds["conductance"].fillna(0.0).to_numpy(),
    elevation=ditchds["elevation"].fillna(0.0).to_numpy(),
)

overlandflow = rsp.Drainage(
    conductance=xr.full_like(olf, 500.0).to_numpy(),
    elevation=olf.to_numpy(),
)
# %%

tube = rsp.Drainage(
    conductance=tubeds["conductance"].fillna(0.0).to_numpy(),
    elevation=tubeds["elevation"].fillna(0.0).to_numpy(),
)

recharge = rsp.Recharge(
    rate=xr.full_like(transmissivity, 0.001).to_numpy(),
)


# %%

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

# %%

gwf.nonlinear_solve()

# %%

head = transmissivity.copy(data=gwf.head.reshape(transmissivity.shape))
head.plot.contourf(levels=30)

# %%


def set_layer1(da):
    return da.expand_dims({"layer": [1]})


bottom = set_layer1(xr.zeros_like(transmissivity))
idomain = set_layer1(xr.ones_like(transmissivity, dtype=int))

tubeds = set_layer1(tubeds)
ditchds = set_layer1(ditchds)
riverds = set_layer1(riverds)
olf = set_layer1(olf)
transmissivity = set_layer1(transmissivity)
rate = xr.full_like(transmissivity, 0.001)

# %%

riverds = riverds.where(riverds["conductance"] > 0)

# %%


gwf_model = imod.mf6.GroundwaterFlowModel()
gwf_model["dis"] = imod.mf6.StructuredDiscretization(
    top=1.0, bottom=bottom, idomain=idomain
)
gwf_model["tube"] = imod.mf6.Drainage(
    elevation=tubeds["elevation"],
    conductance=tubeds["conductance"],
)
gwf_model["ditch"] = imod.mf6.Drainage(
    elevation=ditchds["elevation"],
    conductance=ditchds["conductance"],
)
gwf_model["overland"] = imod.mf6.Drainage(
    elevation=olf,
    conductance=xr.full_like(olf, 500.0),
)
gwf_model["river"] = imod.mf6.River(
    conductance=riverds["conductance"],
    stage=riverds["stage"],
    bottom_elevation=riverds["bottom"],
)
gwf_model["ic"] = imod.mf6.InitialConditions(start=0.0)
gwf_model["npf"] = imod.mf6.NodePropertyFlow(
    icelltype=0,
    k=transmissivity,
    k33=1.0,
)
gwf_model["sto"] = imod.mf6.SpecificStorage(
    specific_storage=1.0e-5,
    specific_yield=0.15,
    transient=False,
    convertible=0,
)
gwf_model["oc"] = imod.mf6.OutputControl(save_head="all")
gwf_model["rch"] = imod.mf6.Recharge(rate=rate)


# Attach it to a simulation
simulation = imod.mf6.Modflow6Simulation("ex01-twri")
simulation["GWF_1"] = gwf_model
# Define solver settings
simulation["solver"] = imod.mf6.Solution(
    modelnames=["GWF_1"],
    print_option="summary",
    outer_dvclose=1.0e-6,
    outer_maximum=500,
    under_relaxation=None,
    inner_dvclose=1.0e-8,
    inner_rclose=0.001,
    inner_maximum=100,
    linear_acceleration="cg",
    scaling_method=None,
    reordering_method=None,
    relaxation_factor=0.97,
)
# %%
# Collect time discretization
simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])

# %%

simulation.write("reference")
# %%
simulation.run()

# %%
mf6heads = simulation.open_head().isel(time=0, layer=0).compute()
# %%

mf6heads.plot.contourf(levels=30)

# %%

diff = head - mf6heads

# %%

diff.plot.imshow()

# %%


simulation.dump("../testdata/mf6-simulation.toml")
# %%
