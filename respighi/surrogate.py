import xarray as xr

from respighi.target import FittingTarget


class LinearInterpolationSurrogate:
    def __init__(
        self,
        head_reference: xr.DataArray,
        observation_reference: xr.DataArray,
        weights: xr.DataArray,
    ):
        self.head_reference = head_reference.copy()
        self.observation_reference = observation_reference.copy()
        self.weights = weights.copy()

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        return cls(
            head_reference=dataset["head_reference"],
            observation_reference=dataset["observation_reference"],
            weights=dataset["weights"],
        )

    def interpolate(self, target: FittingTarget) -> xr.DataArray:
        # TODO: might make sense to store an offset such that delta
        # does not need to be computed each time.
        observations = self.observation_reference.copy(data=target.d)
        delta = observations - self.observation_reference
        head = self.head_reference + self.weights.dot(delta, dim="observation")
        return head.rename("head")
