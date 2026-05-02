"""xarray accessors for the harmonic fitting API."""

from __future__ import annotations

import xarray as xr

from .core import evaluate, fit


@xr.register_dataarray_accessor("harmonic")
class HarmonicDataArrayAccessor:
    """DataArray accessor for fitting harmonic coefficients."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Store the accessed DataArray.

        Args:
            xarray_obj: DataArray with dimensions `(*dims)`.
        """
        self._obj = xarray_obj

    def fit(
        self,
        time_dim: str = "time",
        n_harmonics: int = 2,
        fundamental_period: float | None = None,
        weights: xr.DataArray | None = None,
        skipna: bool = True,
    ) -> xr.Dataset:
        """Fit harmonics to the accessed DataArray.

        See `xharmonics.fit` for parameter and return-value details.
        """
        return fit(
            self._obj,
            time_dim=time_dim,
            n_harmonics=n_harmonics,
            fundamental_period=fundamental_period,
            weights=weights,
            skipna=skipna,
        )


@xr.register_dataset_accessor("harmonic")
class HarmonicDatasetAccessor:
    """Dataset accessor for evaluating harmonic coefficient datasets."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Store the accessed Dataset.

        Args:
            xarray_obj: Dataset returned by `fit`.
        """
        self._obj = xarray_obj

    def evaluate(
        self,
        time: xr.DataArray,
        time_dim: str = "time",
    ) -> xr.DataArray:
        """Evaluate the accessed coefficient Dataset.

        See `xharmonics.evaluate` for parameter and return-value details.
        """
        return evaluate(self._obj, time=time, time_dim=time_dim)
