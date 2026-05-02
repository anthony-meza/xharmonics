import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xharmonics as xh


def _monthly_signal(periods=24 * 12):
    time = pd.date_range("2000-01-01", periods=periods, freq="MS")
    t = (time - time[0]).days.to_numpy(dtype=float) / 365.2425
    values = (
        10.0
        + 2.0 * np.cos(2 * np.pi * t)
        - 0.7 * np.sin(2 * np.pi * t)
        + 0.5 * np.cos(4 * np.pi * t)
        + 0.2 * np.sin(4 * np.pi * t)
    )
    da = xr.DataArray(values, dims=("time",), coords={"time": time}, name="signal")
    return da


def _daily_signal(start="1999-01-01", end="2002-12-31"):
    time = xr.date_range(start, end, freq="D")
    t = np.arange(time.size, dtype=float) / 365.2425
    values = (
        5.0
        + 1.5 * np.cos(2 * np.pi * t)
        - 0.25 * np.sin(2 * np.pi * t)
        + 0.3 * np.cos(4 * np.pi * t)
        + 0.1 * np.sin(4 * np.pi * t)
    )
    return xr.DataArray(values, dims=("time",), coords={"time": time}, name="daily_signal")


def test_datetime_monthly_known_annual_signal():
    da = _monthly_signal()
    coef = xh.fit(da, time_dim="time", n_harmonics=2)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="cos"), 2.0, atol=5e-4)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="sin"), -0.7, atol=5e-4)


def test_harmonic_zero_equals_mean():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    assert np.isclose(
        coef["coef"].sel(harmonic=0, basis="cos"),
        da.mean("time"),
    )


def test_evaluate_dims():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    fit_da = xh.evaluate(coef, time=da["time"], time_dim="time")
    assert fit_da.dims == ("harmonic", "time")


def test_evaluated_sum_reconstructs_seasonal_cycle():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    fit_da = xh.evaluate(coef, time=da["time"], time_dim="time")
    seasonal = fit_da.sum("harmonic")
    xr.testing.assert_allclose(seasonal, da, atol=5e-3)


def test_annual_and_semiannual_coefficients():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="cos"), 2.0, atol=5e-4)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="sin"), -0.7, atol=5e-4)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="cos"), 0.5, atol=5e-4)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="sin"), 0.2, atol=5e-4)


def test_nyquist_error():
    da = _monthly_signal()
    with pytest.raises(ValueError, match="Nyquist"):
        xh.fit(da, n_harmonics=7)


def test_infers_datetime_defaults():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    assert coef.attrs["fundamental_period"] == 1.0
    assert coef.attrs["sampling_frequency"] == 12.0


def test_daily_data_recovers_known_coefficients():
    da = _daily_signal()
    coef = xh.fit(da, n_harmonics=2)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="cos"), 1.5, atol=0.02)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="sin"), -0.25, atol=0.02)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="cos"), 0.3, atol=0.02)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="sin"), 0.1, atol=0.02)


def test_daily_data_infers_sampling_frequency():
    da = _daily_signal()
    coef = xh.fit(da, n_harmonics=2)
    assert np.isclose(coef.attrs["sampling_frequency"], 365.2425, atol=1e-6)


def test_daily_reconstruction_spanning_leap_year():
    da = _daily_signal()
    coef = xh.fit(da, n_harmonics=2)
    fit_da = xh.evaluate(coef, time=da["time"], time_dim="time")
    xr.testing.assert_allclose(fit_da.sum("harmonic"), da, atol=2e-3)


def test_numeric_time_without_period_raises():
    da = xr.DataArray(np.arange(24.0), dims=("time",), coords={"time": np.arange(24.0)})
    with pytest.raises(ValueError, match="fundamental_period"):
        xh.fit(da, n_harmonics=1)


def test_numeric_time_with_period_works():
    t = np.arange(24.0)
    da = xr.DataArray(
        3.0 + np.cos(2 * np.pi * t / 12.0),
        dims=("time",),
        coords={"time": t},
    )
    coef = xh.fit(da, n_harmonics=1, fundamental_period=12.0)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="cos"), 1.0)


def test_evaluate_with_numeric_time_raises():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=2)
    with pytest.raises(TypeError, match="datetime-like"):
        xh.evaluate(coef, time=np.arange(da.sizes["time"]), time_dim="time")


def test_skipna_true_with_some_nans():
    da = _monthly_signal()
    da = da.where(~da["time"].isin(da["time"].values[[2, 10, 17]]))
    coef = xh.fit(da, n_harmonics=2, skipna=True)
    assert np.isfinite(coef["coef"].sel(harmonic=1, basis="cos"))


def test_dask_multichunk_time_warns_and_aborts():
    dask = pytest.importorskip("dask.array")
    da = _monthly_signal()
    chunked = da.chunk({"time": 12})
    with pytest.warns(UserWarning, match="core dimension"):
        with pytest.raises(ValueError, match="core dimension"):
            xh.fit(chunked, n_harmonics=2)


def test_dask_single_time_chunk_fits():
    dask = pytest.importorskip("dask.array")
    da = _monthly_signal()
    chunked = da.chunk({"time": -1})
    coef = xh.fit(chunked, n_harmonics=2).load()
    expected = xh.fit(da, n_harmonics=2)
    xr.testing.assert_allclose(coef["coef"], expected["coef"])


def test_weights_sets_weighted_attr():
    da = _monthly_signal()
    weights = da["time"].dt.days_in_month
    coef = xh.fit(da, n_harmonics=2, weights=weights)
    assert coef.attrs["weighted"] is True


def test_weights_do_not_change_returned_mean():
    da = _monthly_signal(periods=12)
    weights = xr.DataArray(
        np.linspace(1.0, 100.0, da.sizes["time"]),
        dims=("time",),
        coords={"time": da["time"]},
    )

    coef = xh.fit(da, n_harmonics=1, weights=weights)

    assert np.isclose(
        coef["coef"].sel(harmonic=0, basis="cos"),
        da.mean("time"),
    )
    assert not np.isclose(
        coef["coef"].sel(harmonic=0, basis="cos"),
        da.weighted(weights).mean("time"),
    )


def test_weights_dataset_raises():
    da = _monthly_signal()
    weights = xr.Dataset(
        {
            "signal": xr.DataArray(
                da["time"].dt.days_in_month,
                dims=("time",),
                coords={"time": da["time"]},
            )
        }
    )
    with pytest.raises(TypeError, match="xarray.DataArray"):
        xh.fit(da, n_harmonics=2, weights=weights)


def test_weights_numpy_array_raises():
    da = _monthly_signal()
    weights = np.ones(da.sizes["time"])
    with pytest.raises(TypeError, match="xarray.DataArray"):
        xh.fit(da, n_harmonics=2, weights=weights)


def test_weights_must_be_finite_and_positive():
    da = _monthly_signal()
    zero_weight = xr.DataArray(
        np.ones(da.sizes["time"]),
        dims=("time",),
        coords={"time": da["time"]},
    )
    zero_weight[0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        xh.fit(da, n_harmonics=2, weights=zero_weight)

    nan_weight = xr.DataArray(
        np.ones(da.sizes["time"]),
        dims=("time",),
        coords={"time": da["time"]},
    )
    nan_weight[0] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        xh.fit(da, n_harmonics=2, weights=nan_weight)


def test_n_harmonics_zero():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=0)
    assert coef.sizes["harmonic"] == 1
    assert np.isclose(coef["coef"].sel(harmonic=0, basis="cos"), da.mean("time"))


def test_phase_zero_is_zero():
    da = _monthly_signal()
    coef = xh.fit(da, n_harmonics=1)
    assert np.isclose(coef["phase"].sel(harmonic=0), 0.0)


def test_units_propagate_to_coef_and_fit():
    da = _daily_signal()
    da.attrs["units"] = "K"
    coef = xh.fit(da, n_harmonics=2)
    fit_da = xh.evaluate(coef, time=da["time"], time_dim="time")
    assert coef.attrs["units"] == "K"
    assert coef["coef"].attrs["units"] == "K"
    assert fit_da.attrs["units"] == "K"


def test_calendar_mismatch_warns_for_non_monthly_cftime():
    cftime = pytest.importorskip("cftime")
    fit_time = xr.date_range(
        "2001-01-01", periods=365, freq="D", use_cftime=True, calendar="noleap"
    )
    t = np.arange(fit_time.size, dtype=float) / 365.0
    da = xr.DataArray(
        2.0 + np.cos(2 * np.pi * t),
        dims=("time",),
        coords={"time": fit_time},
    )
    coef = xh.fit(da, n_harmonics=1)
    assert coef.attrs["calendar"] == "noleap"

    eval_time = xr.date_range(
        "2001-01-01", periods=366, freq="D", use_cftime=True, calendar="gregorian"
    )
    with pytest.warns(UserWarning, match="calendar"):
        xh.evaluate(coef, time=xr.DataArray(eval_time, dims="time"), time_dim="time")


def test_calendar_mismatch_warns_for_monthly_cftime():
    cftime = pytest.importorskip("cftime")
    fit_time = xr.date_range(
        "2001-01-01", periods=24, freq="MS", use_cftime=True, calendar="noleap"
    )
    fit_days = np.array([i * 30.0 for i in range(fit_time.size)], dtype=float)
    t = fit_days / 365.0
    da = xr.DataArray(
        2.0 + np.cos(2 * np.pi * t),
        dims=("time",),
        coords={"time": fit_time},
    )
    coef = xh.fit(da, n_harmonics=1)

    eval_time = xr.date_range(
        "2001-01-01", periods=24, freq="MS", use_cftime=True, calendar="gregorian"
    )
    with pytest.warns(UserWarning, match="calendar"):
        xh.evaluate(coef, time=xr.DataArray(eval_time, dims="time"), time_dim="time")


def test_noisy_recovery():
    rng = np.random.default_rng(0)
    time = pd.date_range("2000-01-01", periods=20 * 12, freq="MS")
    t = xr.DataArray((time - time[0]).days.to_numpy(dtype=float) / 365.2425, dims="time", coords={"time": time})
    annual = 2.0 * np.cos(2 * np.pi * t) - 0.7 * np.sin(2 * np.pi * t)
    semiannual = 0.5 * np.cos(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 2 * t)
    higher_frequency = 0.4 * np.cos(2 * np.pi * 5 * t)
    noise = xr.DataArray(rng.normal(scale=0.8, size=time.size), dims="time", coords={"time": time})
    da = 10.0 + annual + semiannual + higher_frequency + noise
    coef = xh.fit(da, time_dim="time", n_harmonics=2)
    fit_da = xh.evaluate(coef, time=da["time"], time_dim="time")
    recovered = fit_da.sum("harmonic")
    true_cycle = 10.0 + annual + semiannual

    assert np.isclose(coef["coef"].sel(harmonic=1, basis="cos"), 2.0, atol=0.15)
    assert np.isclose(coef["coef"].sel(harmonic=1, basis="sin"), -0.7, atol=0.15)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="cos"), 0.5, atol=0.18)
    assert np.isclose(coef["coef"].sel(harmonic=2, basis="sin"), 0.2, atol=0.15)

    recovered_rmse = float(np.sqrt(((recovered - true_cycle) ** 2).mean()))
    noisy_rmse = float(np.sqrt(((da - true_cycle) ** 2).mean()))
    assert recovered_rmse < noisy_rmse
