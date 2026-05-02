import xarray as xr
import numpy as np
import pandas as pd

import xharmonics


def _data():
    time = pd.date_range("2001-01-01", periods=36, freq="MS")
    t = np.arange(time.size) / 12.0
    da = xr.DataArray(
        1.5 + np.cos(2 * np.pi * t) + 0.2 * np.sin(2 * np.pi * t),
        dims=("time",),
        coords={"time": time},
        name="demo",
    )
    return da


def test_accessor_registration():
    da = _data()
    coef_ds = da.harmonic.fit(n_harmonics=2)
    fit_da = coef_ds.harmonic.evaluate(time=da["time"], time_dim="time")
    assert "coef" in coef_ds
    assert "phase" in coef_ds
    assert fit_da.name == "fit"


def test_function_and_accessor_equivalent():
    da = _data()
    coef_func = xharmonics.fit(da, n_harmonics=2)
    coef_acc = da.harmonic.fit(n_harmonics=2)
    xr.testing.assert_allclose(coef_func["coef"], coef_acc["coef"])
    xr.testing.assert_allclose(coef_func["phase"], coef_acc["phase"])
