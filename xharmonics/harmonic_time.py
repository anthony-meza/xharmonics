"""Time-coordinate conversion and sampling-frequency helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _looks_like_cftime(values):
    """Return whether values appear to contain cftime datetime objects.

    Args:
        values: Array-like object or xarray object. The flattened first element
            is inspected when the array has at least one value.

    Returns:
        looks_like_cftime: `True` when the first value comes from a `cftime`
            module; otherwise `False`. Empty arrays return `False`.
    """
    value_array = np.asarray(getattr(values, "values", values))
    if value_array.size == 0:
        return False
    first_value = value_array.reshape(-1)[0]
    return first_value.__class__.__module__.startswith("cftime")


def _is_datetime_like(values):
    """Return whether values are NumPy or cftime datetimes.

    Args:
        values: Array-like object or xarray object with any shape.

    Returns:
        is_datetime_like: `True` for `datetime64` arrays and cftime-like object
            arrays; otherwise `False`.
    """
    value_array = np.asarray(getattr(values, "values", values))
    return np.issubdtype(value_array.dtype, np.datetime64) or _looks_like_cftime(value_array)


def _days_per_year(calendar):
    """Resolve a calendar name to the number of days in one model year.

    Args:
        calendar: Calendar name from a cftime object or xarray coordinate.

    Returns:
        days_per_year: Days per year as a float. Known fixed-length calendars
            return their exact length; other calendars use `365.2425`.
    """
    mapping = {
        "360_day": 360.0,
        "365_day": 365.0,
        "noleap": 365.0,
        "366_day": 366.0,
        "all_leap": 366.0,
    }
    return mapping.get(calendar, 365.2425)


def _calendar_name(values):
    """Infer a calendar name from a one-dimensional time coordinate.

    Args:
        values: Array-like object or xarray object containing time values.

    Returns:
        calendar_name: `"proleptic_gregorian"` for NumPy datetimes, a cftime
            calendar name for cftime values, or `None` for non-datetime values.

    Raises:
        ValueError: If cftime-like values are detected but the array is empty.
    """
    value_array = np.asarray(getattr(values, "values", values))
    if np.issubdtype(value_array.dtype, np.datetime64):
        return "proleptic_gregorian"
    if _looks_like_cftime(value_array):
        if value_array.size == 0:
            raise ValueError("Time coordinate must contain at least one value.")
        first_value = value_array.reshape(-1)[0]
        return getattr(first_value, "calendar", "standard")
    return None


def _time_to_float(values, origin=None, calendar=None):
    """Convert a one-dimensional time coordinate to floating offsets.

    Args:
        values: One-dimensional array-like object or xarray object with shape
            `(n_time,)`.
        origin: Optional time origin. If omitted, the first time value is used.
        calendar: Optional cftime calendar override used when converting cftime
            values.

    Returns:
        t_float: Finite NumPy float array with shape `(n_time,)`. Datetime-like
            coordinates are expressed in years since `origin`; numeric
            coordinates are expressed in native coordinate units since `origin`.
        origin_str: String representation of the time origin.
        is_datetime_like: Whether the input coordinate was datetime-like.
        calendar_name: Calendar used for cftime conversion, or `None`.

    Raises:
        ValueError: If `values` is not one-dimensional, is empty, or produces
            nonfinite numeric offsets.
        ImportError: If cftime values are supplied but `cftime` is not
            installed.
    """
    value_array = np.asarray(getattr(values, "values", values))
    if value_array.ndim != 1:
        raise ValueError("Time coordinate must be one-dimensional.")
    if value_array.size == 0:
        raise ValueError("Time coordinate must contain at least one value.")

    if np.issubdtype(value_array.dtype, np.datetime64):
        origin_value = value_array.reshape(-1)[0] if origin is None else np.datetime64(origin)
        origin_str = np.datetime_as_string(origin_value, unit="ns")
        days_since_origin = (value_array - origin_value) / np.timedelta64(1, "D")
        time_offsets = np.asarray(days_since_origin, dtype=float) / 365.2425
        if not np.all(np.isfinite(time_offsets)):
            raise ValueError("Time coordinate must produce finite numeric offsets.")
        return time_offsets, origin_str, True, None

    if _looks_like_cftime(value_array):
        try:
            import cftime
        except ImportError as exc:
            raise ImportError("cftime is required for cftime-based coordinates.") from exc

        first_value = value_array.reshape(-1)[0]
        calendar_name = calendar or getattr(first_value, "calendar", "standard")
        origin_str = str(first_value if origin is None else origin)
        units = f"days since {origin_str}"
        days_since_origin = cftime.date2num(value_array.tolist(), units=units, calendar=calendar_name)
        time_offsets = np.asarray(days_since_origin, dtype=float) / _days_per_year(calendar_name)
        if not np.all(np.isfinite(time_offsets)):
            raise ValueError("Time coordinate must produce finite numeric offsets.")
        return time_offsets, origin_str, True, calendar_name

    origin_value = value_array.reshape(-1)[0] if origin is None else origin
    numeric_values = np.asarray(value_array, dtype=float)
    time_offsets = numeric_values - float(origin_value)
    if not np.all(np.isfinite(time_offsets)):
        raise ValueError("Time coordinate must produce finite numeric offsets.")
    return time_offsets, str(origin_value), False, None


def infer_sampling_frequency(time, rtol=1e-3):
    """Infer sampling frequency from a one-dimensional coordinate.

    Args:
        time: One-dimensional time coordinate with shape `(n_time,)`. May be an
            xarray DataArray or an array-like object.
        rtol: Relative tolerance used to decide whether spacing is regular.

    Returns:
        sampling_frequency: Samples per year for datetime-like coordinates,
            samples per native coordinate unit for numeric coordinates, or
            `None` when regular spacing cannot be inferred.
        inferred: Whether `sampling_frequency` was inferred.

    Raises:
        ValueError: If `time` is not one-dimensional.
    """
    values = np.asarray(getattr(time, "values", time))
    if values.ndim != 1:
        raise ValueError("Time coordinate must be one-dimensional.")

    if _is_datetime_like(values):
        inferred_frequency_string = (
            xr.infer_freq(time)
            if isinstance(time, xr.DataArray)
            else xr.infer_freq(xr.DataArray(values, dims="time"))
        )
        if inferred_frequency_string is not None and "M" in inferred_frequency_string:
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
