import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xharmonics.time import (
    _calendar_name,
    _days_per_year,
    _is_datetime_like,
    _looks_like_cftime,
    _time_to_float,
    infer_sampling_frequency,
)


def test_days_per_year_known_calendars():
    assert _days_per_year("360_day") == 360.0
    assert _days_per_year("365_day") == 365.0
    assert _days_per_year("noleap") == 365.0
    assert _days_per_year("366_day") == 366.0
    assert _days_per_year("all_leap") == 366.0


def test_days_per_year_default_for_unknown_calendar():
    assert _days_per_year("standard") == 365.2425
    assert _days_per_year(None) == 365.2425


def test_is_datetime_like_for_numpy_datetime():
    values = pd.date_range("2000-01-01", periods=3, freq="D").to_numpy()
    assert _is_datetime_like(values) is True


def test_is_datetime_like_for_numeric_returns_false():
    assert _is_datetime_like(np.arange(5.0)) is False


def test_looks_like_cftime_empty_array_returns_false():
    assert _looks_like_cftime(np.array([], dtype=object)) is False


def test_calendar_name_for_numpy_datetime():
    values = pd.date_range("2000-01-01", periods=3, freq="D").to_numpy()
    assert _calendar_name(values) == "proleptic_gregorian"


def test_calendar_name_for_numeric_returns_none():
    assert _calendar_name(np.arange(5.0)) is None


def test_calendar_name_for_cftime():
    cftime = pytest.importorskip("cftime")
    times = xr.date_range(
        "2001-01-01", periods=3, freq="D", use_cftime=True, calendar="noleap"
    )
    assert _calendar_name(times) == "noleap"


def test_time_to_float_rejects_non_1d():
    with pytest.raises(ValueError, match="one-dimensional"):
        _time_to_float(np.zeros((2, 3), dtype=float))


def test_time_to_float_rejects_empty():
    with pytest.raises(ValueError, match="at least one value"):
        _time_to_float(np.array([], dtype="datetime64[ns]"))


def test_time_to_float_numpy_datetime_offsets_in_years():
    time = pd.date_range("2000-01-01", periods=2, freq="365D").to_numpy()
    offsets, origin, is_dt, calendar = _time_to_float(time)
    assert is_dt is True
    assert calendar is None
    assert offsets[0] == 0.0
    assert np.isclose(offsets[1], 365.0 / 365.2425)
    assert "2000" in origin


def test_time_to_float_numeric_uses_first_value_as_origin():
    values = np.array([10.0, 11.5, 13.0])
    offsets, origin, is_dt, calendar = _time_to_float(values)
    assert is_dt is False
    assert calendar is None
    np.testing.assert_allclose(offsets, [0.0, 1.5, 3.0])
    assert origin == "10.0"


def test_time_to_float_numeric_explicit_origin():
    values = np.array([10.0, 11.5, 13.0])
    offsets, origin, _, _ = _time_to_float(values, origin=0.0)
    np.testing.assert_allclose(offsets, [10.0, 11.5, 13.0])
    assert origin == "0.0"


def test_time_to_float_cftime_offsets_in_years():
    cftime = pytest.importorskip("cftime")
    times = xr.date_range(
        "2001-01-01", periods=2, freq="365D", use_cftime=True, calendar="noleap"
    )
    offsets, _, is_dt, calendar = _time_to_float(times)
    assert is_dt is True
    assert calendar == "noleap"
    np.testing.assert_allclose(offsets, [0.0, 1.0], atol=1e-12)


def test_infer_sampling_frequency_monthly_datetime():
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    coord = xr.DataArray(time, dims=("time",))
    freq, inferred = infer_sampling_frequency(coord)
    assert inferred is True
    assert freq == 12.0


def test_infer_sampling_frequency_uniform_numeric():
    coord = xr.DataArray(np.arange(0.0, 5.0, 0.5), dims=("time",))
    freq, inferred = infer_sampling_frequency(coord)
    assert inferred is True
    assert np.isclose(freq, 2.0)


def test_infer_sampling_frequency_irregular_returns_none():
    coord = xr.DataArray(np.array([0.0, 1.0, 1.5, 5.0]), dims=("time",))
    freq, inferred = infer_sampling_frequency(coord)
    assert inferred is False
    assert freq is None


def test_infer_sampling_frequency_rejects_non_1d():
    coord = xr.DataArray(np.zeros((2, 2)), dims=("a", "b"))
    with pytest.raises(ValueError, match="one-dimensional"):
        infer_sampling_frequency(coord)


def test_infer_sampling_frequency_rejects_non_dataarray():
    with pytest.raises(TypeError, match="xarray.DataArray"):
        infer_sampling_frequency(np.arange(5.0))
