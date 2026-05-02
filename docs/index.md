# xharmonics

`xharmonics` fits low-order harmonic models to `xarray.DataArray` objects and returns coefficient datasets that can be evaluated later on an explicit datetime-like time axis.

This package is currently alpha-stage. The implementation is intentionally narrow in this first version: DataArray input only, coefficient-only fitting, explicit evaluation on a supplied time axis, no rolling support yet, and limited calendar semantics outside the common monthly workflow.

## Install

```bash
pip install xharmonics
```

## Quickstart

```python
import xharmonics as xh

coef_ds = xh.fit(da, time_dim="time", n_harmonics=2)
fit_da = xh.evaluate(coef_ds, time=da["time"], time_dim="time")

seasonal = fit_da.sum("harmonic")
annual = fit_da.sel(harmonic=1)
semiannual = fit_da.sel(harmonic=2)
anom = da - seasonal
```

## Accessor API

```python
import xharmonics

coef_ds = da.harmonic.fit(time_dim="time", n_harmonics=2)
fit_da = coef_ds.harmonic.evaluate(time=da["time"], time_dim="time")
```

```{toctree}
:maxdepth: 1

../examples/noisy_seasonal_cycle
```

## Notes

- `fit` returns coefficients only.
- `evaluate` reconstructs the harmonic contributions from those coefficients.
- `harmonic=0` is the mean by construction.
- `basis` indexes the real Fourier basis functions `"cos"` and `"sin"`.
- Datetime-like coordinates are converted to fractional years internally.
- daily and monthly datetime-like data are supported.
- Numeric time requires an explicit `seasonal_period` during fitting.

## Calendar and leap-year behavior

- Datetime-like coordinates are converted using elapsed days expressed in fractional years.
- Leap years therefore affect the exact phase position of daily and monthly timestamps through elapsed time.
- For `cftime`, the conversion uses calendar-specific days per year for `360_day`, `noleap`/`365_day`, and `all_leap`/`366_day`, and otherwise falls back to `365.2425`.
- Coefficient datasets retain the fitted calendar, and `evaluate` warns on calendar mismatches.
- Cross-calendar evaluation should still be treated as provisional behavior in this version.
