__version__ = "0.0.1"

from respighi.groundwaterflow import (
    Drainage,
    GroundwaterModel,
    HeadBoundary,
    HorizontalFlowBarrier,
    Recharge,
    River,
)
from respighi.inverse import InverseProblem
from respighi.inverse_minres import InverseProblemMINRES
from respighi.layer_reduction import (
    effective_transmissivity,
    two_layer_effective_transmisivity,
)
from respighi.linearsolvers.settings import (
    MumpsSettings,
    PardisoSettings,
    PCGSettings,
    ScipyLUSettings,
)
from respighi.nonlinear import NonlinearSettings
from respighi.relaxation import AitkenRelaxation, ScalarRelaxation
from respighi.surrogate import LinearInterpolationSurrogate
from respighi.target import (
    CellSampling,
    CompositeTarget,
    GridSampling,
    ModelTarget,
)
from respighi.tikhonov import (
    MaternSemivariogram,
    MinimumCurvature,
    UnscaledMinimumCurvature,
)

__all__ = (
    "Recharge",
    "HeadBoundary",
    "HorizontalFlowBarrier",
    "Drainage",
    "River",
    "GroundwaterModel",
    "LinearInterpolationSurrogate",
    "GridSampling",
    "CellSampling",
    "ModelTarget",
    "CompositeTarget",
    "MaternSemivariogram",
    "MinimumCurvature",
    "UnscaledMinimumCurvature",
    "InverseProblem",
    "InverseProblemMINRES",
    "effective_transmissivity",
    "two_layer_effective_transmisivity",
    "NonlinearSettings",
    "ScalarRelaxation",
    "AitkenRelaxation",
    "PCGSettings",
    "PardisoSettings",
    "MumpsSettings",
    "ScipyLUSettings",
)
