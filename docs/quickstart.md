# Quickstart

## Installation

```bash
pip install xharmonics
```

## Example Usage

You can use `xharmonics` through its functional or accessor APIs.

The core functions are `fit` and `evaluate`:

```python
import xharmonics as xh

coef_ds = xh.fit(da, time_dim="time", n_harmonics=2)
fit_da = xh.evaluate(coef_ds, time=da["time"], time_dim="time")

seasonal = fit_da.sum("harmonic")
annual = fit_da.sel(harmonic=1)
semiannual = fit_da.sel(harmonic=2)
anom = da - seasonal
```

Alternatively, you can use the `.harmonic` accessor for a more chainable syntax:

```python
import xharmonics

coef_ds = da.harmonic.fit(time_dim="time", n_harmonics=2)
fit_da = coef_ds.harmonic.evaluate(time=da["time"], time_dim="time")
```
