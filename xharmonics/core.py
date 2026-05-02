"""Public fitting and evaluation API for xharmonics."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .harmonic_lstsq import _lstsq
from .harmonic_time import (
    _calendar_name,
    _is_datetime_like,
    _time_to_float,
    infer_sampling_frequency,
)


def fit(
    data_array,
    time_dim="time",
    n_harmonics=2,
    seasonal_period=None,
    sampling_frequency=None,
    weights=None,
    skipna=True,
    rtol=1e-3,
):
    """Fit harmonic coefficients to an xarray DataArray.

    Args:
        data_array: Input DataArray containing the signal to fit. It must include
            `time_dim`; any non-time dimensions are fit independently. Expected
            dimensions are `(time_dim, *non_time_dims)` in any order.
        time_dim: Name of the time dimension.
        n_harmonics: Number of positive harmonics to fit. Harmonic 0 is always
            the unweighted time mean.
        seasonal_period: Period of harmonic 1 in time-coordinate units. If
            omitted for datetime-like coordinates, one year is used. Numeric
            time coordinates require an explicit value.
        sampling_frequency: Optional sampling frequency used for the Nyquist
            check. If omitted, regular spacing is inferred when possible.
        weights: Optional weights. Accepts a DataArray with dims `(time_dim,)`
            or a Dataset variable matching `data_array.name` with dims
            `(time_dim,)`.
            Weights enter the harmonic least-squares objective but do not
            change the returned mean. Weights must be finite and positive.
        skipna: If `True`, fit each vectorized signal using finite data
            samples. If `False`, return NaN coefficients for any signal with a
            nonfinite data value.
        rtol: Relative tolerance used when inferring regular sampling
            frequency.

    Returns:
        coefficient_dataset: Dataset with these variables:
            `coef`: DataArray with dimensions
            `(*non_time_dims, harmonic, basis)`, where `harmonic` has length
            `n_harmonics + 1` and `basis` contains `"cos"` and `"sin"`.
            `coef.sel(harmonic=0, basis="cos")` is the mean;
            `coef.sel(harmonic=0, basis="sin")` is zero.
            `phase`: DataArray with dimensions `(*non_time_dims, harmonic)`.

    Raises:
        TypeError: If `data_array` is not a DataArray, or if Dataset input is supplied.
        ValueError: If dimensions, harmonic count, weights, seasonal period, or
            Nyquist constraints are invalid.
    """
    if not isinstance(data_array, xr.DataArray):
        if isinstance(data_array, xr.Dataset):
            raise TypeError("Dataset input support is not implemented yet. Pass an xarray.DataArray.")
        raise TypeError("data_array must be an xarray.DataArray.")
    if time_dim not in data_array.dims:
        raise ValueError(f"Dimension '{time_dim}' is not present in data_array.")
    if not isinstance(n_harmonics, int) or n_harmonics < 0:
        raise ValueError("n_harmonics must be an integer >= 0.")

    time_coordinate = data_array[time_dim]
    time_offsets, time_origin, is_datetime_like, calendar = _time_to_float(time_coordinate)
    if not np.all(np.isfinite(time_offsets)):
        raise ValueError("time coordinate must produce finite numeric offsets.")
    if seasonal_period is None and not is_datetime_like:
        raise ValueError(
            "Numeric time coordinates require an explicit seasonal_period during fit."
        )
    resolved_period = 1.0 if seasonal_period is None else float(seasonal_period)
    period_inferred = seasonal_period is None
    if is_datetime_like and calendar is None:
        calendar = _calendar_name(time_coordinate)

    if sampling_frequency is None:
        sampling_frequency, sampling_frequency_inferred = infer_sampling_frequency(time_coordinate, rtol=rtol)
    else:
        sampling_frequency = float(sampling_frequency)
        sampling_frequency_inferred = False

    if sampling_frequency is not None:
        nyquist_limit = 0.5 * sampling_frequency * resolved_period
        if n_harmonics > nyquist_limit:
            raise ValueError(
                "n_harmonics exceeds the Nyquist limit implied by sampling_frequency and seasonal_period."
            )

    time_chunks = getattr(data_array, "chunksizes", {}).get(time_dim, ())
    if len(time_chunks) > 1:
        message = (
            f"Dimension '{time_dim}' is chunked into multiple chunks. Harmonic fitting "
            f"treats '{time_dim}' as a core dimension. Rechunk with .chunk({{{time_dim!r}: -1}}) "
            "before calling fit."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        raise ValueError(message)

    if weights is None:
        weight_array = xr.DataArray(
            np.ones(data_array.sizes[time_dim], dtype=float),
            dims=(time_dim,),
            coords={time_dim: data_array[time_dim]},
        )
        is_weighted = False
    else:
        if isinstance(weights, xr.Dataset):
            if data_array.name is None:
                raise ValueError(
                    "weights was provided as a Dataset, but data_array has no name. "
                    "Pass weights as a DataArray or give data_array a name that matches a Dataset variable."
                )
            if data_array.name not in weights:
                raise ValueError(
                    f"weights Dataset does not contain a variable named {data_array.name!r}."
                )
            weight_array = weights[data_array.name]
        elif not isinstance(weights, xr.DataArray):
            raise TypeError(
                "weights must be None, an xarray.DataArray, or an xarray.Dataset "
                "with a variable matching data_array.name."
            )
        else:
            weight_array = weights

        if weight_array.dims != (time_dim,):
            raise ValueError(
                f"weights must be one-dimensional along '{time_dim}' only."
            )
        if weight_array.sizes[time_dim] != data_array.sizes[time_dim]:
            raise ValueError("weights must have the same length as data_array along time_dim.")
        weight_array = weight_array.astype(float)
        weight_values = np.asarray(weight_array.values)
        if not np.all(np.isfinite(weight_values) & (weight_values > 0)):
            raise ValueError("weights must be finite and positive.")
        is_weighted = True

    coefficients = _lstsq(
        data_array.astype(float),
        weight_array,
        time_offsets,
        resolved_period,
        n_harmonics,
        time_dim,
        skipna,
    )
    data_units = data_array.attrs.get("units")

    coefficient_array = coefficients.assign_coords(
        harmonic=np.arange(coefficients.sizes["harmonic"], dtype=int),
        basis=["cos", "sin"],
    )
    phase_array = np.arctan2(
        coefficient_array.sel(basis="sin"), coefficient_array.sel(basis="cos")
    )
    coefficient_dataset = xr.Dataset(
        data_vars={
            "coef": coefficient_array,
            "phase": phase_array,
        }
    )
    coefficient_dataset["coef"].attrs["description"] = (
        "Harmonic coefficients indexed by real Fourier basis. "
        "coef.sel(basis='cos') gives a_k; coef.sel(basis='sin') gives b_k."
    )
    coefficient_dataset["phase"].attrs["description"] = "Phase of each harmonic in radians. harmonic=0 phase is set to 0 by convention."
    coefficient_dataset["phase"].attrs["formula"] = "atan2(b_k, a_k)"
    coefficient_dataset["phase"].attrs["units"] = "radian"
    coefficient_dataset["harmonic"].attrs["description"] = (
        "Harmonic index k. k=0 is the mean. k=1 has period seasonal_period. "
        "k=2 has period seasonal_period/2. In general, harmonic k has period seasonal_period/k."
    )
    coefficient_dataset["basis"].attrs["description"] = (
        "Real Fourier basis function. basis='cos' gives the coefficient a_k multiplying "
        "cos(2*pi*k*t/seasonal_period). basis='sin' gives the coefficient b_k multiplying "
        "sin(2*pi*k*t/seasonal_period). For harmonic=0, basis='cos' is the mean and basis='sin' is zero."
    )
    if data_units is not None:
        coefficient_dataset["coef"].attrs["units"] = data_units

    coefficient_dataset.attrs.update(
        {
            "description": "Harmonic coefficients. Evaluate with xharmonics.evaluate(...) or coefficient_dataset.harmonic.evaluate(...).",
            "model": "x(t) = mean + sum_{k=1}^{K} [a_k cos(2*pi*k*t/seasonal_period) + b_k sin(2*pi*k*t/seasonal_period)]",
            "time_dim": time_dim,
            "seasonal_period": float(resolved_period),
            "seasonal_period_units": "years for datetime-like coordinates; native coordinate units for numeric coordinates",
            "seasonal_period_inferred": bool(period_inferred),
            "n_harmonics": int(n_harmonics),
            "sampling_frequency": None if sampling_frequency is None else float(sampling_frequency),
            "sampling_frequency_units": "samples per year for datetime-like coordinates; samples per native coordinate unit for numeric coordinates",
            "sampling_frequency_inferred": bool(sampling_frequency_inferred),
            "time_origin": str(time_origin),
            "centering": "harmonic=0 is the time mean. Harmonics k>=1 are fit to demeaned data.",
            "reconstruction": "fit = xharmonics.evaluate(coefficient_dataset, time, time_dim); seasonal = fit.sum('harmonic'); anomaly = original - seasonal",
            "phase_convention": "phase = atan2(b_k, a_k). harmonic=0 phase is set to 0 by convention. For k>=1, the maximum of harmonic k occurs when 2*pi*k*t/seasonal_period = phase modulo 2*pi.",
            "weighted": bool(is_weighted),
            "weights": "Weights enter the least-squares objective as sum_t weights(t) * residual(t)**2. Larger weights give a sample more influence. Use weights for irregular sampling, unequal time intervals, or inverse-variance observational uncertainty. If no weights were supplied, all valid samples were weighted equally.",
            "time_kind": "datetime-like" if is_datetime_like else "numeric",
        }
    )
    if calendar is not None:
        coefficient_dataset.attrs["calendar"] = calendar
    if data_units is not None:
        coefficient_dataset.attrs["units"] = data_units
    return coefficient_dataset


def evaluate(coefficient_dataset, time, time_dim="time"):
    """Evaluate fitted harmonic coefficients on a datetime-like time axis.

    Args:
        coefficient_dataset: Dataset returned by `fit`. It must contain `coef`
            and `phase` variables plus `seasonal_period` and `time_origin`
            attributes.
        time: One-dimensional datetime-like coordinate with shape `(n_time,)`.
            May be a DataArray or array-like object.
        time_dim: Name to use for the evaluation time dimension.

    Returns:
        fit_data_array: DataArray named `"fit"` with dimensions
            `(harmonic, time_dim, *non_time_dims)`. The `harmonic` dimension
            matches `coefficient_dataset["coef"].sizes["harmonic"]`; summing over
            `harmonic` gives the reconstructed seasonal cycle.

    Raises:
        TypeError: If `coefficient_dataset` is not a Dataset or `time` is not
            datetime-like.
        ValueError: If required variables/attributes are missing or `time` is
            not one-dimensional.
        UserWarning: Warns when the fitted calendar and evaluation calendar
            differ.
    """
    if not isinstance(coefficient_dataset, xr.Dataset):
        raise TypeError("coefficient_dataset must be the xarray.Dataset returned by fit.")

    required_variables = {"coef", "phase"}
    missing_variables = required_variables.difference(coefficient_dataset.data_vars)
    if missing_variables:
        raise ValueError(f"coefficient_dataset is missing required variables: {sorted(missing_variables)}")
    for attribute_name in ("seasonal_period", "time_origin"):
        if attribute_name not in coefficient_dataset.attrs:
            raise ValueError(f"coefficient_dataset is missing required attribute {attribute_name!r}.")

    if isinstance(time, xr.DataArray):
        time_array = time
    else:
        time_array = xr.DataArray(np.asarray(time), dims=(time_dim,), name=time_dim)

    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if not _is_datetime_like(time_array):
        raise TypeError("evaluate currently requires datetime-like time coordinates.")
    fit_calendar = coefficient_dataset.attrs.get("calendar")
    target_calendar = _calendar_name(time_array)
    if fit_calendar is not None and target_calendar is not None and fit_calendar != target_calendar:
        warnings.warn(
            f"Fitted coefficients were derived using calendar {fit_calendar!r} but evaluation time uses "
            f"calendar {target_calendar!r}. For non-monthly data this can shift fractional-year phase "
            "interpretation, especially around leap years.",
            UserWarning,
            stacklevel=2,
        )

    time_offsets, _, _, _ = _time_to_float(
        time_array,
        origin=coefficient_dataset.attrs["time_origin"],
        calendar=coefficient_dataset.attrs.get("calendar"),
    )
    time_offset_array = xr.DataArray(time_offsets, dims=(time_dim,), coords={time_dim: time_array.values})
    harmonic_index = coefficient_dataset["harmonic"].astype(float)
    harmonic_angle = 2.0 * np.pi * harmonic_index * time_offset_array / float(coefficient_dataset.attrs["seasonal_period"])
    fit_data_array = coefficient_dataset["coef"].sel(basis="cos") * np.cos(harmonic_angle) + coefficient_dataset["coef"].sel(
        basis="sin"
    ) * np.sin(harmonic_angle)
    fit_data_array = fit_data_array.transpose("harmonic", time_dim, *[dimension for dimension in coefficient_dataset["coef"].dims if dimension not in {"harmonic", "basis"}])
    fit_data_array.name = "fit"
    fit_data_array.attrs.update(
        {
            "description": "Reconstructed contribution from each harmonic. Sum over 'harmonic' to get the fitted seasonal cycle.",
            "formula": "fit[k,t] = a_k cos(2*pi*k*t/seasonal_period) + b_k sin(2*pi*k*t/seasonal_period); fit[0,t] = mean",
            "reconstruction": "seasonal = fit.sum('harmonic'); anomaly = original - seasonal",
            "seasonal_period": coefficient_dataset.attrs["seasonal_period"],
            "time_origin": coefficient_dataset.attrs["time_origin"],
        }
    )
    if "units" in coefficient_dataset.attrs:
        fit_data_array.attrs["units"] = coefficient_dataset.attrs["units"]
    return fit_data_array
