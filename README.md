# xharmonics

[![Documentation Status](https://readthedocs.org/projects/xharmonics/badge/?version=latest)](https://xharmonics.readthedocs.io/en/latest/?badge=latest)

`xharmonics` is an alpha-stage package for fitting low-order harmonic models to `xarray.DataArray` objects.

It currently focuses on a narrow first release:

- `fit` returns harmonic coefficients only
- `evaluate` reconstructs harmonic contributions on an explicit time axis
- DataArray input is supported
- daily and monthly datetime-like data are supported
- Dataset input to `fit` is not implemented yet
- rolling support is not implemented yet

## Install

```bash
python -m pip install xharmonics
```

If you want xarray's full plotting and optional IO stack:

```bash
python -m pip install "xarray[complete]"
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

Accessor API:

```python
import xharmonics

coef_ds = da.harmonic.fit(time_dim="time", n_harmonics=2)
fit_da = coef_ds.harmonic.evaluate(time=da["time"], time_dim="time")
```

## Model

For `n_harmonics=K`, `fit` estimates

```text
x(t) = mean + sum_{k=1}^{K}
       [a_k cos(2*pi*k*t/fundamental_period)
      + b_k sin(2*pi*k*t/fundamental_period)]
```

`fundamental_period` is the fundamental period, i.e. the longest fitted period.
`harmonic=0` is the mean by construction. The returned `basis` coordinate indexes the real Fourier basis functions `"cos"` and `"sin"`.

## Status

This package is still in development. The current implementation is intentionally narrow. Daily and monthly datetime-like fitting are supported using elapsed-time fractional-year coordinates. Coefficient datasets retain calendar metadata and `evaluate` warns when the evaluation calendar does not match the fitted calendar.

Weights should be passed as xarray objects: either a `DataArray` or a `Dataset` containing a variable with the same name as the fitted `DataArray`.

## Documentation

Longer usage notes, examples, and the runnable daily-data notebook example belong in the Read the Docs site.
