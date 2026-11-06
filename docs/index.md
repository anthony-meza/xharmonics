# xharmonics

`xharmonics` fits low-order harmonic models to `xarray.DataArray` objects and returns coefficient datasets that can be evaluated later on an explicit datetime-like time axis.

This package is currently alpha-stage. The implementation is intentionally narrow in this first version: DataArray input only, coefficient-only fitting, explicit evaluation on a supplied time axis, no rolling support yet, and limited calendar semantics outside the common monthly workflow.

```{toctree}
:maxdepth: 2
:caption: "Contents:"

install
quickstart
examples
notes
```

