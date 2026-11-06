import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xclimatology as xc
from xclimatology import harmonics


def signal(t, trend=0.0):
    return (
        10.0
        + 2.0 * np.cos(2 * np.pi * t)
        - 0.7 * np.sin(2 * np.pi * t)
        + 0.5 * np.cos(4 * np.pi * t)
        + 0.2 * np.sin(4 * np.pi * t)
        + trend * (t - t.mean())
    )


def monthly(start="2000-01-01", years=20, trend=0.0):
    time = xr.DataArray(
        pd.date_range(start, periods=12 * years, freq="MS"), dims="time"
    )
    t = time.dt.decimal_year.values
    return xr.DataArray(
        signal(t, trend),
        dims="time",
        coords={"time": time},
        name="sst",
        attrs={"units": "K"},
    )


def assert_known_coefs(coefs, atol=1e-8):
    np.testing.assert_allclose(coefs["mean"], 10.0, atol=atol)
    np.testing.assert_allclose(coefs["cos"], [2.0, 0.5], atol=atol)
    np.testing.assert_allclose(coefs["sin"], [-0.7, 0.2], atol=atol)


def test_monthly_recovers_coefficients_and_climatology():
    da = monthly()
    coefs = harmonics.coefficients(da)
    assert_known_coefs(coefs)
    assert coefs["mean"].attrs["units"] == coefs["amplitude"].attrs["units"] == "K"
    clim = harmonics.climatology(da)
    assert clim.dims == da.dims and clim.name == "sst" and clim.attrs["units"] == "K"
    xr.testing.assert_allclose(clim, da, atol=1e-8)


def test_daily_record_spanning_leap_years():
    time = xr.DataArray(
        pd.date_range("1999-01-01", "2002-12-31", freq="D"), dims="time"
    )
    da = xr.DataArray(
        signal(time.dt.decimal_year.values), dims="time", coords={"time": time}
    )
    assert_known_coefs(harmonics.coefficients(da))


def test_cftime_noleap():
    time = xr.date_range(
        "2001-01-01", periods=3 * 365, freq="D", calendar="noleap", use_cftime=True
    )
    da = xr.DataArray(
        signal(2001 + np.arange(time.size) / 365), dims="time", coords={"time": time}
    )
    assert_known_coefs(harmonics.coefficients(da))


def test_phase_is_measured_from_january_first():
    a, b = (
        harmonics.coefficients(monthly("2000-01-01")),
        harmonics.coefficients(monthly("2003-07-01")),
    )
    xr.testing.assert_allclose(a["phase"], b["phase"])
    time = xr.DataArray(
        pd.date_range("2000-01-01", periods=240, freq="MS"), dims="time"
    )
    peak_in_april = np.cos(2 * np.pi * (time.dt.decimal_year - 0.25))
    phase = harmonics.coefficients(peak_in_april.assign_coords(time=time), 1)[
        "phase"
    ].sel(harmonic=1)
    assert np.isclose(phase / (2 * np.pi), 0.25)


def test_linear_trend_is_fit_jointly():
    da = monthly(trend=0.3)
    coefs = harmonics.coefficients(da)
    assert_known_coefs(coefs)
    assert np.isclose(coefs["trend"].sel(degree=1), 0.3)  # c₁/L per year
    xr.testing.assert_allclose(harmonics.climatology(da), da, atol=1e-8)


def test_detrend_degree_none_fits_no_trend():
    assert "trend" not in harmonics.coefficients(monthly(), detrend_degree=None)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="detrend_degree"):
            harmonics.coefficients(monthly(), detrend_degree=bad)
    quadratic = harmonics.coefficients(monthly(), detrend_degree=2)
    assert list(quadratic["degree"].values) == [1, 2]
    assert np.isclose(quadratic["mean"], 10.0)


def test_numeric_time_decoded_with_decode_cf():
    days = np.arange(3 * 365.0)
    time = xr.Variable(
        "time", days, {"units": "days since 2001-01-01", "calendar": "noleap"}
    )
    da = xr.DataArray(signal(2001 + days / 365), dims="time", coords={"time": time})
    assert_known_coefs(
        harmonics.coefficients(xr.decode_cf(da.to_dataset(name="y"))["y"])
    )


def test_diurnal_cycle_in_hours():
    time = xr.DataArray(
        pd.date_range("2020-06-01", periods=60 * 24, freq="h"), dims="time"
    )
    hours = ((time - np.datetime64("1970-01-01")) / np.timedelta64(1, "h")).values
    da = xr.DataArray(
        3 * np.cos(2 * np.pi * (hours - 14) / 24), dims="time", coords={"time": time}
    )
    coefs = harmonics.coefficients(da, 1, period=24, period_units="hour")
    assert np.isclose(coefs["amplitude"].sel(harmonic=1), 3)
    peak_hour = (coefs["phase"].sel(harmonic=1) % (2 * np.pi)) * 24 / (2 * np.pi)
    assert np.isclose(peak_hour, 14)
    xr.testing.assert_allclose(
        da.climatology(n_harmonics=1, period=1, period_units="day"), da, atol=1e-8
    )
    with pytest.raises(ValueError, match="month"):
        harmonics.coefficients(da, period_units="month")


def test_missing_values_and_unfittable_series():
    da = monthly().expand_dims(x=5).copy()
    da[1, 5:9] = np.nan
    da[2, :] = np.nan  # land: no samples
    da[3, ::7] = np.nan
    da[4, :] = np.nan
    da[4, ::12] = 1.0  # every January only: cannot separate the harmonics
    coefs = harmonics.coefficients(da)
    for i in (0, 1, 3):
        assert_known_coefs(coefs.isel(x=i))
    assert coefs["mean"].isel(x=[2, 4]).isnull().all()


def test_fitting_all_series_at_once_matches_fitting_each():
    rng = np.random.default_rng(0)
    da = (monthly().expand_dims(x=4) + rng.normal(size=(4, 240))).copy()
    da[1, 10:20] = np.nan
    da[3, 100:] = np.nan
    together = harmonics.coefficients(da)
    for i in range(4):
        xr.testing.assert_allclose(
            together.isel(x=i), harmonics.coefficients(da.isel(x=i))
        )


def test_any_dims_in_any_order_and_datasets():
    da = monthly().expand_dims(member=2, depth=3).transpose("member", "time", "depth")
    assert_known_coefs(harmonics.coefficients(da).isel(member=1, depth=2))
    ds = xr.Dataset({"sst": da, "sss": da * 2})
    xr.testing.assert_allclose(ds.climatology(), ds, atol=1e-8)


def test_weights():
    rng = np.random.default_rng(2)
    da = monthly() + rng.normal(size=240)
    weights = da.time.dt.days_in_month
    assert not np.allclose(
        harmonics.coefficients(da, weights=weights)["cos"],
        harmonics.coefficients(da)["cos"],
    )
    xr.testing.assert_allclose(
        harmonics.coefficients(da, weights=weights * 0 + 3.0),
        harmonics.coefficients(da),
    )


def test_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        harmonics.coefficients(monthly(), 7)


def test_dask_is_lazy_and_matches_numpy():
    def forbid(dsk, keys, **kwargs):
        raise RuntimeError("computed eagerly")

    da = monthly().expand_dims(x=3)
    for chunks in ({"x": 1}, {"time": 60}):
        with dask.config.set(scheduler=forbid):
            coefs = harmonics.coefficients(da.chunk(chunks))
            clim = harmonics.climatology(da.chunk(chunks))
        xr.testing.assert_allclose(coefs.compute(), harmonics.coefficients(da))
        xr.testing.assert_allclose(clim.compute(), harmonics.climatology(da))


def test_generic_climatology_and_accessors():
    da = monthly()
    expected = harmonics.climatology(da, 2, detrend_degree=None)
    xr.testing.assert_identical(xc.climatology(da, detrend_degree=None), expected)
    xr.testing.assert_identical(da.climatology(detrend_degree=None), expected)
    xr.testing.assert_identical(
        da.climatology.coefficients(method="harmonic"), harmonics.coefficients(da)
    )
    with pytest.raises(ValueError, match="Unknown method"):
        da.climatology(method="monthly")
