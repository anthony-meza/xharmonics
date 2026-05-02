"""xarray accessors for the harmonic fitting API."""

import xarray as xr

from .core import evaluate, fit


@xr.register_dataarray_accessor("harmonic")
class HarmonicDataArrayAccessor:
    """DataArray accessor for fitting harmonic coefficients."""

    def __init__(self, xarray_obj):
        """Store the accessed DataArray.

        Args:
            xarray_obj: DataArray with dimensions `(*dims)`.
        """
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
    ):
        """Fit harmonics to the accessed DataArray.

        Args:
            time_dim: Name of the time dimension.
            n_harmonics: Number of positive harmonics to fit.
            seasonal_period: Period of harmonic 1, or `None` to infer for
                datetime-like time.
            sampling_frequency: Sampling frequency for the Nyquist check, or
                `None` to infer when possible.
            weights: Optional finite positive weights with dims `(time_dim,)`.
            skipna: Whether to fit each vectorized signal using only finite
                data samples.
            rtol: Relative tolerance for sampling-frequency inference.

        Returns:
            coefficient_dataset: Dataset with `coef` dims
                `(*non_time_dims, harmonic, basis)` and `phase` dims
                `(*non_time_dims, harmonic)`.
        """
        return fit(
            self._obj,
            time_dim=time_dim,
            n_harmonics=n_harmonics,
            seasonal_period=seasonal_period,
            sampling_frequency=sampling_frequency,
            weights=weights,
            skipna=skipna,
            rtol=rtol,
        )


@xr.register_dataset_accessor("harmonic")
class HarmonicDatasetAccessor:
    """Dataset accessor for evaluating harmonic coefficient datasets."""

    def __init__(self, xarray_obj):
        """Store the accessed Dataset.

        Args:
            xarray_obj: Dataset returned by `fit`.
        """
        self._obj = xarray_obj

    def evaluate(self, time, time_dim="time"):
        """Evaluate the accessed coefficient Dataset.

        Args:
            time: One-dimensional datetime-like coordinate with shape
                `(n_time,)`.
            time_dim: Name of the output time dimension.

        Returns:
            fit_data_array: DataArray with dims `(harmonic, time_dim, *non_time_dims)`.
        """
        return evaluate(self._obj, time=time, time_dim=time_dim)
