"""Public fitting and evaluation API for xharmonics."""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .lstsq import _lstsq
from .time import (
    _calendar_name,
    _is_datetime_like,
    _time_to_float,
    infer_sampling_frequency,
)


def fit(
    data_array: xr.DataArray,
    time_dim: str = "time",
    n_harmonics: int = 2,
    fundamental_period: float | None = None,
    weights: xr.DataArray | None = None,
    skipna: bool = True,
) -> xr.Dataset:
    """Fit harmonic coefficients to an xarray DataArray.

    Args:
        data_array: Input DataArray containing the signal to fit. It must include
            `time_dim`; any non-time dimensions are fit independently. Expected
            dimensions are `(time_dim, *non_time_dims)` in any order.
        time_dim: Name of the time dimension.
        n_harmonics: Number of positive harmonics to fit. Harmonic 0 is always
            the unweighted time mean.
        fundamental_period: Fundamental period in time-coordinate units. This is
            the longest fitted period. If omitted for datetime-like coordinates,
            one year is used. Numeric time coordinates require an explicit
            value.
        weights: Optional DataArray with dims `(time_dim,)`.
            Weights enter the harmonic least-squares objective but do not
            change the returned mean. Weights must be finite and positive.
        skipna: If `True`, fit each vectorized signal using finite data
            samples. If `False`, return NaN coefficients for any signal with a
            nonfinite data value.

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
        ValueError: If dimensions, harmonic count, weights, fundamental period, or
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
    if fundamental_period is None and not is_datetime_like:
        raise ValueError(
            "Numeric time coordinates require an explicit fundamental_period during fit."
        )
    resolved_fundamental_period = 1.0 if fundamental_period is None else float(fundamental_period)
    fundamental_period_inferred = fundamental_period is None
    if is_datetime_like and calendar is None:
        calendar = _calendar_name(time_coordinate)

    sampling_frequency, sampling_frequency_inferred = infer_sampling_frequency(time_coordinate)

    if sampling_frequency is not None:
        nyquist_limit = 0.5 * sampling_frequency * resolved_fundamental_period
        if n_harmonics > nyquist_limit:
            raise ValueError(
                "n_harmonics exceeds the Nyquist limit implied by sampling_frequency and fundamental_period."
            )

    time_chunks = getattr(data_array, "chunksizes", {}).get(time_dim, ())
    if len(time_chunks) > 1:
        raise ValueError(
            f"Dimension '{time_dim}' is chunked into multiple chunks. Harmonic fitting "
            f"treats '{time_dim}' as a core dimension. Rechunk with .chunk({{{time_dim!r}: -1}}) "
            "before calling fit."
        )

    if weights is None:
        weight_array = xr.DataArray(
            np.ones(data_array.sizes[time_dim], dtype=float),
            dims=(time_dim,),
            coords={time_dim: data_array[time_dim]},
        )
        is_weighted = False
    else:
        if not isinstance(weights, xr.DataArray):
            raise TypeError(
                "weights must be None or an xarray.DataArray."
            )
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

    coefficient_dataset = _lstsq(
        data_array.astype(float),
        weight_array,
        time_offsets,
        resolved_fundamental_period,
        n_harmonics,
        time_dim,
        skipna,
    )

    coefficient_dataset["coef"].attrs["description"] = (
        "Harmonic coefficients indexed by real Fourier basis. "
        "coef.sel(basis='cos') gives a_k; coef.sel(basis='sin') gives b_k."
    )
    coefficient_dataset["phase"].attrs["formula"] = "atan2(b_k, a_k)"
    coefficient_dataset["phase"].attrs["units"] = "radian"
    coefficient_dataset["harmonic"].attrs["description"] = (
        "Harmonic index k. k=0 is the mean. k=1 has the fundamental period "
        "fundamental_period, the longest fitted period. k=2 has period "
        "fundamental_period/2. In general, harmonic k has period fundamental_period/k."
    )

    data_units = data_array.attrs.get("units")
    if data_units is not None:
        coefficient_dataset["coef"].attrs["units"] = data_units

    coefficient_dataset.attrs.update(
        {
            "xharmonics": "fit",
            "time_dim": time_dim,
            "fundamental_period": resolved_fundamental_period,
            "fundamental_period_units": "years for datetime-like coordinates; native coordinate units for numeric coordinates",
            "fundamental_period_inferred": fundamental_period_inferred,
            "n_harmonics": n_harmonics,
            "sampling_frequency": sampling_frequency,
            "sampling_frequency_units": "samples per year for datetime-like coordinates; samples per native coordinate unit for numeric coordinates",
            "sampling_frequency_inferred": sampling_frequency_inferred,
            "time_origin": time_origin,
            "weighted": is_weighted,
            "time_kind": "datetime-like" if is_datetime_like else "numeric",
        }
    )
    if calendar is not None:
        coefficient_dataset.attrs["calendar"] = calendar
    if data_units is not None:
        coefficient_dataset.attrs["units"] = data_units

    return coefficient_dataset


def evaluate(
    coefficient_dataset: xr.Dataset,
    time: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """Evaluate fitted harmonic coefficients on a time axis.

    Args:
        coefficient_dataset: Dataset returned by `fit`.
        time: One-dimensional coordinate DataArray with shape `(n_time,)`. Its
            kind (datetime-like or numeric) must match the time kind the fit
            was performed on.
        time_dim: Name to use for the evaluation time dimension.

    Returns:
        fit_data_array: DataArray named `"fit"` with dimensions
            `(harmonic, time_dim, *non_time_dims)`. The `harmonic` dimension
            matches `coefficient_dataset["coef"].sizes["harmonic"]`; summing over
            `harmonic` gives the reconstructed seasonal cycle.

    Raises:
        TypeError: If `time` is not an `xarray.DataArray` or its kind does not
            match `coefficient_dataset.attrs["time_kind"]`.
        ValueError: If `time` is not one-dimensional (raised by the underlying
            time-coordinate conversion).
        UserWarning: Warns when the fitted calendar and evaluation calendar
            differ.
    """
    if getattr(coefficient_dataset, "attrs", {}).get("xharmonics") != "fit":
        raise TypeError("coefficient_dataset must be the xarray.Dataset returned by xharmonics.fit.")
    if not isinstance(time, xr.DataArray):
        raise TypeError("time must be an xarray.DataArray.")
    fit_kind = coefficient_dataset.attrs["time_kind"]
    if (fit_kind == "datetime-like") != _is_datetime_like(time):
        raise TypeError(
            f"time kind does not match the coefficient_dataset (fit on {fit_kind!r})."
        )
    time_array = time
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
    harmonic_angle = 2.0 * np.pi * coefficient_dataset["harmonic"] * time_offset_array / coefficient_dataset.attrs["fundamental_period"]
    fit_data_array = coefficient_dataset["coef"].sel(basis="cos") * np.cos(harmonic_angle) + coefficient_dataset["coef"].sel(
        basis="sin"
    ) * np.sin(harmonic_angle)
    fit_data_array = fit_data_array.transpose("harmonic", time_dim, *[dimension for dimension in coefficient_dataset["coef"].dims if dimension not in {"harmonic", "basis"}])
    fit_data_array.name = "fit"
    fit_data_array.attrs.update(
        {
            "formula": "fit[k,t] = a_k cos(2*pi*k*t/fundamental_period) + b_k sin(2*pi*k*t/fundamental_period); fit[0,t] = mean",
            "fundamental_period": coefficient_dataset.attrs["fundamental_period"],
            "time_origin": coefficient_dataset.attrs["time_origin"],
        }
    )
    if "units" in coefficient_dataset.attrs:
        fit_data_array.attrs["units"] = coefficient_dataset.attrs["units"]
    return fit_data_array
