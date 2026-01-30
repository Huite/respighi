Respighi
========
REScaling & Piezometer Interpolation of Groundwater Heads, Inversely

Respighi is a Python library for downscaling and interpolating groundwater heads.
It solves inverse steady-state groundwater flow problems to estimate spatially
varying recharge, producing physically consistent head fields that honor the
governing PDE and boundary conditions.

Primary use cases:

1. **Downscaling coarse model results.** Given output from a regional groundwater
   model (e.g. MODFLOW 6), Respighi refines the heads to a finer resolution
   while respecting boundary conditions such as drainage, rivers, and fixed
   heads. The result is a more detailed, physically interpretable head field.

2. **Interpolating between piezometers.** Given point observations of
   groundwater head, Respighi interpolates between them by solving the
   groundwater flow equation. Unlike conventional spatial interpolation (e.g.
   kriging), the interpolation is constrained by the PDE and boundary
   conditions, producing head fields that are consistent with the physics.

3. **Data fusion.** When both piezometer observations and coarse model output
   are available, Respighi combines them into a single consistent head field.
   Point measurements anchor the solution locally, while the model output
   constrains the broader spatial pattern. The inverse formulation
   automatically balances the two data sources.

In all cases, Respighi formulates an inverse problem: it finds the recharge
field that, when the groundwater flow equation is solved, best reproduces the
observations or model output. Laplacian regularization ensures smooth recharge
estimates.

.. toctree::
    :maxdepth: 2

    examples/gwf.rst
    examples/interpolation.rst
