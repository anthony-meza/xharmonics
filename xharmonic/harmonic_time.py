from __future__ import annotations

import numpy as np
import xarray as xr


def _looks_like_cftime(values):
    arr = np.asarray(getattr(values, "values", values))
    if arr.size == 0:
        return False
    first = arr.reshape(-1)[0]
    return first.__class__.__module__.startswith("cftime")


def _is_datetime_like(values):
    arr = np.asarray(getattr(values, "values", values))
    return np.issubdtype(arr.dtype, np.datetime64) or _looks_like_cftime(arr)


def _days_per_year(calendar):
    mapping = {
        "360_day": 360.0,
        "365_day": 365.0,
        "noleap": 365.0,
        "366_day": 366.0,
        "all_leap": 366.0,
    }
    return mapping.get(calendar, 365.2425)


def _calendar_name(values):
    arr = np.asarray(getattr(values, "values", values))
    if np.issubdtype(arr.dtype, np.datetime64):
        return "proleptic_gregorian"
    if _looks_like_cftime(arr):
        if arr.size == 0:
            raise ValueError("Time coordinate must contain at least one value.")
        first = arr.reshape(-1)[0]
        return getattr(first, "calendar", "standard")
    return None


def _time_to_float(values, origin=None, calendar=None):
    arr = np.asarray(getattr(values, "values", values))
    if arr.ndim != 1:
        raise ValueError("Time coordinate must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("Time coordinate must contain at least one value.")

    if np.issubdtype(arr.dtype, np.datetime64):
        origin_value = arr.reshape(-1)[0] if origin is None else np.datetime64(origin)
        origin_str = np.datetime_as_string(origin_value, unit="ns")
        days = (arr - origin_value) / np.timedelta64(1, "D")
        return np.asarray(days, dtype=float) / 365.2425, origin_str, True, None

    if _looks_like_cftime(arr):
        try:
            import cftime
        except ImportError as exc:
            raise ImportError("cftime is required for cftime-based coordinates.") from exc

        first = arr.reshape(-1)[0]
        cal = calendar or getattr(first, "calendar", "standard")
        origin_str = str(first if origin is None else origin)
        units = f"days since {origin_str}"
        days = cftime.date2num(arr.tolist(), units=units, calendar=cal)
        return np.asarray(days, dtype=float) / _days_per_year(cal), origin_str, True, cal

    origin_value = arr.reshape(-1)[0] if origin is None else origin
    numeric = np.asarray(arr, dtype=float)
    return numeric - float(origin_value), str(origin_value), False, None


def _resolve_seasonal_period(time_values, seasonal_period):
    if seasonal_period is not None:
        return float(seasonal_period), False
    if _is_datetime_like(time_values):
        return 1.0, True
    raise ValueError(
        "Numeric time coordinates require an explicit seasonal_period during fit."
    )


def infer_sampling_frequency(time, rtol=1e-3):
    values = np.asarray(getattr(time, "values", time))
    if values.ndim != 1:
        raise ValueError("Time coordinate must be one-dimensional.")

    if _is_datetime_like(values):
        freq = (
            xr.infer_freq(time)
            if isinstance(time, xr.DataArray)
            else xr.infer_freq(xr.DataArray(values, dims="time"))
        )
        if freq is not None and "M" in freq:
            return 12.0, True

    t_float, _, _, _ = _time_to_float(values)
    positive_dt = np.diff(t_float)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if positive_dt.size == 0:
        return None, False
    median_dt = float(np.median(positive_dt))
    if median_dt == 0.0:
        return None, False
    if np.all(np.abs(positive_dt - median_dt) <= rtol * abs(median_dt)):
        return 1.0 / median_dt, True
    return None, False
