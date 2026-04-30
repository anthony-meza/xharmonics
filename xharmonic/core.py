from __future__ import annotations

import numpy as np
import xarray as xr

from .harmonic_lstsq import (
    _as_weights,
    _check_core_dim_single_chunk,
    _lstsq,
)
from .harmonic_metadata import (
    _add_evaluate_attrs,
    _add_fit_attrs,
    _assemble_coef,
    _warn_on_calendar_mismatch,
)
from .harmonic_time import (
    _calendar_name,
    _is_datetime_like,
    _resolve_seasonal_period,
    _time_to_float,
    infer_sampling_frequency,
)


def fit(
    da,
    time_dim="time",
    n_harmonics=2,
    seasonal_period=None,
    sampling_frequency=None,
    weights=None,
    skipna=True,
    rtol=1e-3,
    allow_rechunk=False,
):
    if isinstance(da, xr.Dataset):
        raise TypeError("Dataset input support is not implemented yet. Pass an xarray.DataArray.")
    if not isinstance(da, xr.DataArray):
        raise TypeError("da must be an xarray.DataArray.")
    if time_dim not in da.dims:
        raise ValueError(f"Dimension '{time_dim}' is not present in da.")
    if not isinstance(n_harmonics, int) or n_harmonics < 0:
        raise ValueError("n_harmonics must be an integer >= 0.")

    time = da[time_dim]
    resolved_period, period_inferred = _resolve_seasonal_period(time, seasonal_period)
    t_float, time_origin, is_dt_like, calendar = _time_to_float(time)
    if is_dt_like and calendar is None:
        calendar = _calendar_name(time)

    if sampling_frequency is None:
        sampling_frequency, sf_inferred = infer_sampling_frequency(time, rtol=rtol)
    else:
        sampling_frequency = float(sampling_frequency)
        sf_inferred = False

    if sampling_frequency is not None:
        nyquist_limit = 0.5 * sampling_frequency * resolved_period
        if n_harmonics > nyquist_limit:
            raise ValueError(
                "n_harmonics exceeds the Nyquist limit implied by sampling_frequency and seasonal_period."
            )

    _check_core_dim_single_chunk(da, time_dim, allow_rechunk)
    weight_da, weighted = _as_weights(weights, da, time_dim)

    beta = _lstsq(
        da.astype(float),
        weight_da,
        t_float,
        resolved_period,
        n_harmonics,
        time_dim,
        skipna,
        allow_rechunk,
    )
    units = da.attrs.get("units")
    ds = _assemble_coef(beta, units=units)
    ds = _add_fit_attrs(
        ds,
        time_dim=time_dim,
        seasonal_period=resolved_period,
        seasonal_period_inferred=period_inferred,
        sampling_frequency=sampling_frequency,
        sampling_frequency_inferred=sf_inferred,
        time_origin=time_origin,
        weighted=weighted,
        is_datetime_like=is_dt_like,
        n_harmonics=n_harmonics,
        calendar=calendar,
        units=units,
    )
    return ds


def evaluate(coef_ds, time, time_dim="time"):
    if not isinstance(coef_ds, xr.Dataset):
        raise TypeError("coef_ds must be the xarray.Dataset returned by fit.")

    required = {"coef", "phase"}
    missing = required.difference(coef_ds.data_vars)
    if missing:
        raise ValueError(f"coef_ds is missing required variables: {sorted(missing)}")
    for attr in ("seasonal_period", "time_origin"):
        if attr not in coef_ds.attrs:
            raise ValueError(f"coef_ds is missing required attribute {attr!r}.")

    if isinstance(time, xr.DataArray):
        time_da = time
    else:
        time_da = xr.DataArray(np.asarray(time), dims=(time_dim,), name=time_dim)

    if time_da.ndim != 1:
        raise ValueError("time must be one-dimensional.")
    if not _is_datetime_like(time_da):
        raise TypeError("evaluate currently requires datetime-like time coordinates.")
    _warn_on_calendar_mismatch(coef_ds, time_da)

    t_float, _, _, _ = _time_to_float(
        time_da,
        origin=coef_ds.attrs["time_origin"],
        calendar=coef_ds.attrs.get("calendar"),
    )
    t = xr.DataArray(t_float, dims=(time_dim,), coords={time_dim: time_da.values})
    k = coef_ds["harmonic"].astype(float)
    angle = 2.0 * np.pi * k * t / float(coef_ds.attrs["seasonal_period"])
    fit_da = coef_ds["coef"].sel(basis="cos") * np.cos(angle) + coef_ds["coef"].sel(
        basis="sin"
    ) * np.sin(angle)
    fit_da = fit_da.transpose("harmonic", time_dim, *[d for d in coef_ds["coef"].dims if d not in {"harmonic", "basis"}])
    fit_da.name = "fit"
    return _add_evaluate_attrs(fit_da, coef_ds)
