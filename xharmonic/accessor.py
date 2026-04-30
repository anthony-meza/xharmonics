import xarray as xr

from .core import evaluate, fit


@xr.register_dataarray_accessor("harmonic")
class HarmonicDataArrayAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def fit(
        self,
        time_dim="time",
        n_harmonics=2,
        seasonal_period=None,
        sampling_frequency=None,
        weights=None,
        skipna=True,
        rtol=1e-3,
        allow_rechunk=False,
    ):
        return fit(
            self._obj,
            time_dim=time_dim,
            n_harmonics=n_harmonics,
            seasonal_period=seasonal_period,
            sampling_frequency=sampling_frequency,
            weights=weights,
            skipna=skipna,
            rtol=rtol,
            allow_rechunk=allow_rechunk,
        )


@xr.register_dataset_accessor("harmonic")
class HarmonicDatasetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def evaluate(self, time, time_dim="time"):
        return evaluate(self._obj, time=time, time_dim=time_dim)
