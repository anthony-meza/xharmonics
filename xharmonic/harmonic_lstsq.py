from __future__ import annotations

import numpy as np
import xarray as xr


def _basis(t, seasonal_period, n_harmonics):
    cols = []
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / seasonal_period
        cols.append(np.cos(angle))
        cols.append(np.sin(angle))
    if not cols:
        return np.empty((t.size, 0), dtype=float)
    return np.stack(cols, axis=-1)


def _weighted_mean(y, w):
    denom = np.sum(w)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return np.sum(w * y) / denom


def _lstsq_1d(y, w, t, seasonal_period, n_harmonics, skipna):
    beta = np.full((n_harmonics + 1, 2), np.nan, dtype=float)
    if y.shape != t.shape or y.shape != w.shape:
        raise ValueError("Input arrays must share the same 1D shape.")

    finite = np.isfinite(y) & np.isfinite(w) & np.isfinite(t)
    nonnegative = w >= 0
    if not np.all(nonnegative[np.isfinite(w)]):
        return beta

    if skipna:
        valid = finite & nonnegative
        yv = y[valid]
        wv = w[valid]
        tv = t[valid]
    else:
        if not np.all(finite & nonnegative):
            return beta
        yv = y
        wv = w
        tv = t

    if yv.size == 0:
        return beta

    mean = _weighted_mean(yv, wv)
    beta[0, 0] = mean
    beta[0, 1] = 0.0

    if n_harmonics == 0:
        return beta

    if yv.size < 2 * n_harmonics:
        return beta

    design = _basis(tv, seasonal_period, n_harmonics)
    sqrt_w = np.sqrt(wv)[:, None]
    target = (yv - mean) * np.sqrt(wv)
    try:
        coef, _, rank, _ = np.linalg.lstsq(design * sqrt_w, target, rcond=None)
    except np.linalg.LinAlgError:
        return beta

    if rank < 2 * n_harmonics:
        return beta

    beta[1:, 0] = coef[0::2]
    beta[1:, 1] = coef[1::2]
    return beta


def _lstsq(data, weights, t, seasonal_period, n_harmonics, time_dim, skipna, allow_rechunk):
    return xr.apply_ufunc(
        _lstsq_1d,
        data,
        weights,
        xr.DataArray(t, dims=(time_dim,)),
        input_core_dims=[[time_dim], [time_dim], [time_dim]],
        output_core_dims=[["harmonic", "basis"]],
        output_dtypes=[float],
        vectorize=True,
        dask="parallelized",
        kwargs={
            "seasonal_period": seasonal_period,
            "n_harmonics": n_harmonics,
            "skipna": skipna,
        },
        dask_gufunc_kwargs={
            "allow_rechunk": allow_rechunk,
            "output_sizes": {
                "harmonic": n_harmonics + 1,
                "basis": 2,
            },
        },
    )


def _as_weights(weights, da, time_dim):
    if weights is None:
        return xr.DataArray(
            np.ones(da.sizes[time_dim], dtype=float),
            dims=(time_dim,),
            coords={time_dim: da[time_dim]},
        ), False

    if isinstance(weights, xr.Dataset):
        if da.name is None:
            raise ValueError(
                "weights was provided as a Dataset, but da has no name. "
                "Pass weights as a DataArray or give da a name that matches a Dataset variable."
            )
        if da.name not in weights:
            raise ValueError(
                f"weights Dataset does not contain a variable named {da.name!r}."
            )
        weight_da = weights[da.name]
    elif isinstance(weights, xr.DataArray):
        weight_da = weights
    else:
        raise TypeError(
            "weights must be None, an xarray.DataArray, or an xarray.Dataset "
            "with a variable matching da.name."
        )

    if weight_da.dims != (time_dim,):
        raise ValueError(
            f"weights must be one-dimensional along '{time_dim}' only."
        )

    if weight_da.sizes[time_dim] != da.sizes[time_dim]:
        raise ValueError("weights must have the same length as da along time_dim.")

    if np.any(np.asarray(weight_da.values)[np.isfinite(weight_da.values)] < 0):
        raise ValueError("weights must be nonnegative.")

    return weight_da.astype(float), True


def _check_core_dim_single_chunk(obj, time_dim, allow_rechunk):
    chunks = getattr(obj, "chunksizes", None)
    if not chunks or time_dim not in chunks:
        return
    if len(chunks[time_dim]) > 1 and not allow_rechunk:
        raise ValueError(
            f"Dimension '{time_dim}' is chunked into multiple chunks. Harmonic fitting "
            f"treats '{time_dim}' as a core dimension. Rechunk with .chunk({{{time_dim!r}: -1}}) "
            "or pass allow_rechunk=True."
        )
