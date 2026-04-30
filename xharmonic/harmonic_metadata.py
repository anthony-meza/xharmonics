from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .harmonic_time import _calendar_name


def _assemble_coef(beta, units=None):
    coef = beta.assign_coords(
        harmonic=np.arange(beta.sizes["harmonic"], dtype=int),
        basis=["cos", "sin"],
    )
    phase = np.arctan2(coef.sel(basis="sin"), coef.sel(basis="cos"))

    ds = xr.Dataset(
        data_vars={
            "coef": coef,
            "phase": phase,
        }
    )
    ds["coef"].attrs["description"] = (
        "Harmonic coefficients indexed by real Fourier basis. "
        "coef.sel(basis='cos') gives a_k; coef.sel(basis='sin') gives b_k."
    )
    ds["phase"].attrs["description"] = "Phase of each harmonic in radians. harmonic=0 phase is set to 0 by convention."
    ds["phase"].attrs["formula"] = "atan2(b_k, a_k)"
    ds["phase"].attrs["units"] = "radian"
    ds["harmonic"].attrs["description"] = (
        "Harmonic index k. k=0 is the mean. k=1 has period seasonal_period. "
        "k=2 has period seasonal_period/2. In general, harmonic k has period seasonal_period/k."
    )
    ds["basis"].attrs["description"] = (
        "Real Fourier basis function. basis='cos' gives the coefficient a_k multiplying "
        "cos(2*pi*k*t/seasonal_period). basis='sin' gives the coefficient b_k multiplying "
        "sin(2*pi*k*t/seasonal_period). For harmonic=0, basis='cos' is the mean and basis='sin' is zero."
    )
    if units is not None:
        ds["coef"].attrs["units"] = units
    return ds


def _add_fit_attrs(
    ds,
    *,
    time_dim,
    seasonal_period,
    seasonal_period_inferred,
    sampling_frequency,
    sampling_frequency_inferred,
    time_origin,
    weighted,
    is_datetime_like,
    n_harmonics,
    calendar=None,
    units=None,
):
    ds.attrs.update(
        {
            "description": "Harmonic coefficients. Evaluate with xharmonic.evaluate(...) or coef_ds.harmonic.evaluate(...).",
            "model": "x(t) = mean + sum_{k=1}^{K} [a_k cos(2*pi*k*t/seasonal_period) + b_k sin(2*pi*k*t/seasonal_period)]",
            "time_dim": time_dim,
            "seasonal_period": float(seasonal_period),
            "seasonal_period_units": "years for datetime-like coordinates; native coordinate units for numeric coordinates",
            "seasonal_period_inferred": bool(seasonal_period_inferred),
            "n_harmonics": int(n_harmonics),
            "sampling_frequency": None if sampling_frequency is None else float(sampling_frequency),
            "sampling_frequency_units": "samples per year for datetime-like coordinates; samples per native coordinate unit for numeric coordinates",
            "sampling_frequency_inferred": bool(sampling_frequency_inferred),
            "time_origin": str(time_origin),
            "centering": "harmonic=0 is the time mean. Harmonics k>=1 are fit to demeaned data.",
            "reconstruction": "fit = xharmonic.evaluate(coef_ds, time, time_dim); seasonal = fit.sum('harmonic'); anomaly = original - seasonal",
            "phase_convention": "phase = atan2(b_k, a_k). harmonic=0 phase is set to 0 by convention. For k>=1, the maximum of harmonic k occurs when 2*pi*k*t/seasonal_period = phase modulo 2*pi.",
            "weighted": bool(weighted),
            "weights": "Weights enter the least-squares objective as sum_t weights(t) * residual(t)**2. Larger weights give a sample more influence. Use weights for irregular sampling, unequal time intervals, or inverse-variance observational uncertainty. If no weights were supplied, all valid samples were weighted equally.",
            "time_kind": "datetime-like" if is_datetime_like else "numeric",
        }
    )
    if calendar is not None:
        ds.attrs["calendar"] = calendar
    if units is not None:
        ds.attrs["units"] = units
    return ds


def _add_evaluate_attrs(da, coef_ds):
    da.attrs.update(
        {
            "description": "Reconstructed contribution from each harmonic. Sum over 'harmonic' to get the fitted seasonal cycle.",
            "formula": "fit[k,t] = a_k cos(2*pi*k*t/seasonal_period) + b_k sin(2*pi*k*t/seasonal_period); fit[0,t] = mean",
            "reconstruction": "seasonal = fit.sum('harmonic'); anomaly = original - seasonal",
            "seasonal_period": coef_ds.attrs["seasonal_period"],
            "time_origin": coef_ds.attrs["time_origin"],
        }
    )
    if "units" in coef_ds.attrs:
        da.attrs["units"] = coef_ds.attrs["units"]
    return da


def _warn_on_calendar_mismatch(coef_ds, time_values):
    fit_calendar = coef_ds.attrs.get("calendar")
    target_calendar = _calendar_name(time_values)
    if fit_calendar is None or target_calendar is None or fit_calendar == target_calendar:
        return
    warnings.warn(
        f"Fitted coefficients were derived using calendar {fit_calendar!r} but evaluation time uses "
        f"calendar {target_calendar!r}. For non-monthly data this can shift fractional-year phase "
        "interpretation, especially around leap years.",
        UserWarning,
        stacklevel=3,
    )
