"""Least-squares helpers for fitting harmonic coefficients."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.linalg import pinv


def _basis(time_offsets, seasonal_period, n_harmonics):
    """Build the real Fourier design matrix for one time axis.

    Args:
        time_offsets: Floating time coordinate with shape `(n_time,)`.
        seasonal_period: Period of harmonic 1 in `time_offsets` units.
        n_harmonics: Number of positive harmonics to include.

    Returns:
        basis: NumPy array with shape `(n_time, 2 * n_harmonics)`. Columns are
            `[cos1, sin1, cos2, sin2, ...]`.
    """
    if n_harmonics == 0:
        return np.empty((time_offsets.size, 0), dtype=float)

    harmonics = np.arange(1, n_harmonics + 1, dtype=float)
    angle = 2.0 * np.pi * time_offsets[:, None] * harmonics[None, :] / seasonal_period
    basis = np.empty((time_offsets.size, 2 * n_harmonics), dtype=float)
    basis[:, 0::2] = np.cos(angle)
    basis[:, 1::2] = np.sin(angle)
    return basis


def _lstsq_1d(data_values, weight_values, time_offsets, seasonal_period, n_harmonics, skipna):
    """Fit harmonic coefficients for a single one-dimensional signal.

    Fits `x(t) = mean + sum_k [a_k cos(2*pi*k*t/T) + b_k sin(2*pi*k*t/T)]`.
    Matrix form: `x = mean + B c`.
    Cost: `J = (x - mean - B c).T @ W @ (x - mean - B c)`.
    Minimizer: `c_hat = pinv(B.T @ W @ B) @ B.T @ W @ (x - mean)`.

    Symbols:
        x: Observed time series with shape `(n_time,)`.
        mean: Unweighted arithmetic mean of valid `x`.
        B: Harmonic basis with columns `[cos1, sin1, cos2, sin2, ...]`.
        c: Harmonic coefficients `[a_1, b_1, a_2, b_2, ...]`.
        W: Diagonal sample-weight matrix.

    Weights affect `c`, but not the returned `mean`.
    `pinv` returns a minimum-norm `c_hat` for rank-deficient harmonic systems.

    Args:
        data_values: Data values for one signal with shape `(n_time,)`.
        weight_values: Positive sample weights with shape `(n_time,)`.
        time_offsets: Floating time coordinate with shape `(n_time,)`.
        seasonal_period: Period of harmonic 1 in `time_offsets` units.
        n_harmonics: Number of positive harmonics to fit.
        skipna: If `True`, fit using samples where data are finite. If `False`,
            return NaN coefficients for any series with a nonfinite data value.

    Returns:
        coefficients: Coefficient array with shape `(n_harmonics + 1, 2)`. Axis
            0 is `harmonic`; axis 1 is `basis = [cos, sin]`.
            `coefficients[0, 0]` is the mean and `coefficients[0, 1]` is zero
            after a valid mean is found.

    Raises:
        ValueError: If inputs do not share shape `(n_time,)`.
    """
    coefficients = np.full((n_harmonics + 1, 2), np.nan, dtype=float)
    if data_values.shape != time_offsets.shape or data_values.shape != weight_values.shape:
        raise ValueError("Input arrays must share the same 1D shape.")

    finite_data = np.isfinite(data_values)

    if skipna:
        if not np.any(finite_data):
            return coefficients
        sample_selector = finite_data
    elif np.all(finite_data):
        sample_selector = slice(None)
    else:
        return coefficients

    valid_data_values = data_values[sample_selector]
    valid_weight_values = weight_values[sample_selector]
    valid_time_offsets = time_offsets[sample_selector]

    x = valid_data_values
    mean = np.mean(x)
    coefficients[0, 0] = mean
    coefficients[0, 1] = 0.0

    if n_harmonics == 0:
        return coefficients

    B = _basis(valid_time_offsets, seasonal_period, n_harmonics)
    W = np.diag(valid_weight_values)
    try:
        c_hat = pinv(B.T @ W @ B) @ B.T @ W @ (x - mean)
    except np.linalg.LinAlgError:
        return coefficients

    coefficients[1:, 0] = c_hat[0::2]
    coefficients[1:, 1] = c_hat[1::2]
    return coefficients


def _lstsq(data_array, weight_array, time_offsets, seasonal_period, n_harmonics, time_dim, skipna):
    """Vectorize one-dimensional harmonic fitting over non-time dimensions.

    Args:
        data_array: DataArray containing the input signal. It must include
            `time_dim`; all other dimensions are broadcast/vectorized over.
        weight_array: One-dimensional DataArray of positive weights with dims
            `(time_dim,)`.
        time_offsets: Floating time coordinate with shape `(n_time,)`, matching
            `data_array.sizes[time_dim]`.
        seasonal_period: Period of harmonic 1 in `time_offsets` units.
        n_harmonics: Number of positive harmonics to fit.
        time_dim: Name of the time dimension in `data` and `weights`.
        skipna: Passed through to `_lstsq_1d`.

    Returns:
        coefficients: Coefficients as a DataArray with dimensions
            `(*non_time_dims, harmonic, basis)`. `harmonic` has length
            `n_harmonics + 1`; `basis` has length 2.
    """
    return xr.apply_ufunc(
        _lstsq_1d,
        data_array,
        weight_array,
        xr.DataArray(time_offsets, dims=(time_dim,)),
        input_core_dims=[[time_dim], [time_dim], [time_dim]],
        output_core_dims=[["harmonic", "basis"]],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        kwargs={
            "seasonal_period": seasonal_period,
            "n_harmonics": n_harmonics,
            "skipna": skipna,
        },
        dask_gufunc_kwargs={
            "output_sizes": {
                "harmonic": n_harmonics + 1,
                "basis": 2,
            },
        },
    )
