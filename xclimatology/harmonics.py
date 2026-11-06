"""Harmonic climatology: the mean, seasonal harmonics, and an optional trend.

Each series along the time dimension is fit by weighted least squares to

    y(t) = μ + Σₖ [aₖ cos(kωt) + bₖ sin(kωt)] + Σₚ cₚ τᵖ + ε(t),    ω = 2π/T

where T is the fundamental period, the harmonics run over k = 1, ..., K, the
trend is a polynomial of degree P in τ (time scaled to [-1, 1] over the record),
μ is the mean value, and ε is the residual. μ is the 0th harmonic (the sine term
vanishes at k = 0) and is always fit; K = 0 and P = 0 leave the mean alone.

In matrix form y = Xβ + ε. Each column of the design matrix X is one term,
[1, cos ωt, sin ωt, ..., cos Kωt, sin Kωt, τ, ..., τᴾ], and β = [μ, a₁, b₁, ...,
c_P] holds the m coefficients. β minimizes the weighted squared residual rᵀWr,
where r = y - Xβ and the diagonal weights W are zero at missing samples.

The trend is fit in τ, which keeps XᵀWX well conditioned, and reported per unit of t.
"""

from __future__ import annotations

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as xu
from xarray_einstats import linalg

# einstats options for the m × m matrix XᵀWX. Merging its dask chunks is cheap, and
# output_dtypes stops dask from probing the inverse with a singular dummy matrix.
_MATRIX_OPTIONS = {
    "dims": ["term", "term_"],
    "dask": "parallelized",
    "dask_gufunc_kwargs": {"allow_rechunk": True},
    "output_dtypes": [float],
}


def _time_in(time: xr.DataArray, units: str) -> xr.DataArray:
    """Convert a datetime coordinate to t in ``units``.

    ``"year"`` gives the calendar decimal year, which is an integer on January 1
    in every calendar, so annual phases are measured from January 1. Other units
    give elapsed time since 1970-01-01 00:00, so diurnal phases are measured from
    midnight.
    """
    if units == "year":
        return time.dt.decimal_year
    if time.dtype == object:
        epoch = cftime.datetime(1970, 1, 1, calendar=time.dt.calendar)
    else:
        epoch = np.datetime64("1970-01-01")
    return (time - epoch) / pd.Timedelta(1, unit=units)


def climatology(
    obj: xr.DataArray | xr.Dataset,
    n_harmonics: int = 2,
    *,
    period: float = 1.0,
    period_units: str = "year",
    detrend_degree: int | None = 1,
    time_dim: str = "time",
    weights: xr.DataArray | None = None,
) -> xr.DataArray | xr.Dataset:
    """Fit a harmonic climatology and return it on the input's time axis.

    The climatology is the mean plus the harmonics and the trend, so
    ``obj - climatology(obj)`` is the anomaly with all three removed.

    Parameters
    ----------
    obj : DataArray or Dataset
        Data with dimension ``time_dim`` and any other dimensions. For a Dataset,
        every variable is fit.
    n_harmonics : int, default: 2
        Number of harmonics K. The mean, the 0th harmonic, is always fit.
    period : float, default: 1.0
        Fundamental period T, in ``period_units``.
    period_units : str, default: "year"
        ``"year"``, or a :py:class:`pandas.Timedelta` unit such as ``"day"``,
        ``"hour"``, or ``"minute"``.
    detrend_degree : int or None, default: 1
        Degree P of a polynomial trend fit jointly with the harmonics, so the
        trend does not leak into the mean and the low harmonics. A positive
        integer (1 is linear, 2 is quadratic) or None, which fits no trend.
    time_dim : str, default: "time"
        Name of the time dimension. Its coordinate must hold datetimes; decode
        numeric CF times with :py:func:`xarray.decode_cf` first.
    weights : DataArray, optional
        Positive weights along ``time_dim``.

    Returns
    -------
    DataArray or Dataset
        Fitted climatology with the same dimensions, name, and attributes as
        ``obj``.

    See Also
    --------
    coefficients

    Examples
    --------
    >>> clim = ds.climatology()
    >>> anomaly = ds - clim
    >>> diurnal = da.climatology(n_harmonics=1, period=1, period_units="day")
    """

    def fit(da):
        beta, model = _fit(
            da,
            n_harmonics,
            period,
            period_units,
            detrend_degree,
            time_dim,
            weights,
        )
        X = _design_matrix(_time_in(da[time_dim], period_units), model)
        fitted = xr.dot(beta, X, dim="term")  # Xβ
        return fitted.transpose(*da.dims).rename(da.name).assign_attrs(da.attrs)

    return obj.map(fit) if isinstance(obj, xr.Dataset) else fit(obj)


def coefficients(
    da: xr.DataArray,
    n_harmonics: int = 2,
    *,
    period: float = 1.0,
    period_units: str = "year",
    detrend_degree: int | None = 1,
    time_dim: str = "time",
    weights: xr.DataArray | None = None,
) -> xr.Dataset:
    """Fit the mean, K harmonics, and a degree-P trend to every series in ``da``.

    Parameters
    ----------
    da : DataArray
        Data with dimension ``time_dim`` and any other dimensions.
    n_harmonics, period, period_units, detrend_degree, time_dim, weights
        As in :py:func:`climatology`.

    Returns
    -------
    Dataset
        With variables

        - ``mean``: μ, the mean value.
        - ``cos``, ``sin`` (harmonic): aₖ and bₖ.
        - ``amplitude`` (harmonic): √(aₖ² + bₖ²).
        - ``phase`` (harmonic): atan2(bₖ, aₖ) in radians. The harmonic k first
          peaks (phase mod 2π) / (kω) after January 1 (years) or midnight
          (shorter units).
        - ``trend`` (degree), unless ``detrend_degree=None``: the coefficient of
          (t - t₀)ᵖ, with t in ``period_units`` and t₀ = ``attrs["t0"]`` the
          middle of the record. Degree 1 is the linear trend per ``period_units``.

        A series is NaN if its samples cannot separate the terms, for example
        when it has fewer valid samples than coefficients, or is sampled only on
        January 1 and so cannot tell the annual cosine from the mean.

    See Also
    --------
    climatology
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("coefficients takes a DataArray; select one variable.")
    beta, model = _fit(
        da, n_harmonics, period, period_units, detrend_degree, time_dim, weights
    )

    units = {"units": da.attrs["units"]} if "units" in da.attrs else {}
    k = np.arange(1, n_harmonics + 1)
    a = beta.sel(term=[f"cos{i}" for i in k]).assign_coords(term=k)
    b = beta.sel(term=[f"sin{i}" for i in k]).assign_coords(term=k)
    a, b = a.rename(term="harmonic"), b.rename(term="harmonic")
    variables = {
        "mean": beta.sel(term="mean", drop=True).assign_attrs(units),
        "cos": a.assign_attrs(units),
        "sin": b.assign_attrs(units),
        "amplitude": xu.hypot(a, b).assign_attrs(units),
        "phase": xu.arctan2(b, a).assign_attrs(units="radian"),
    }
    if model["detrend_degree"]:
        # The fit uses τ = (t - t₀)/L, so the coefficient of (t - t₀)ᵖ is cₚ / Lᵖ.
        p = np.arange(1, model["detrend_degree"] + 1)
        c = beta.sel(term=[f"trend{i}" for i in p]).assign_coords(term=p)
        c = c.rename(term="degree")
        variables["trend"] = (c / model["L"] ** c["degree"]).assign_attrs(
            long_name=f"coefficient of (t - t0)**degree, t in {period_units}"
        )
    keep = ("period", "period_units", "n_harmonics", "detrend_degree", "t0")
    attrs = {key: model[key] for key in keep if model[key] is not None}
    return xr.Dataset(variables, attrs=attrs)


def _fit(da, n_harmonics, period, period_units, detrend_degree, time_dim, weights):
    """Return β (term), NaN where a series cannot be fit, and the model."""
    is_degree = isinstance(detrend_degree, int) and not isinstance(detrend_degree, bool)
    if detrend_degree is not None and not (is_degree and detrend_degree >= 1):
        raise ValueError("detrend_degree must be None or a positive integer.")
    t = _time_in(da[time_dim], period_units)
    mean_interval = float(t.diff(time_dim).mean())
    if 2 * mean_interval * n_harmonics > period:
        raise ValueError(
            "n_harmonics exceeds the Nyquist limit: "
            "the period T/K must span two sampling intervals."
        )

    model = {
        "period": period,
        "period_units": period_units,
        "n_harmonics": n_harmonics,
        "detrend_degree": detrend_degree,
        "t0": float(t.max() + t.min()) / 2,
        "L": float(t.max() - t.min()) / 2 or 1.0,
    }
    X = _design_matrix(t, model)
    # W = weights × valid-sample mask.
    w = 1 if weights is None else weights
    valid, y = da.notnull(), da.fillna(0)
    m = X.sizes["term"]

    # Normal equations (XᵀWX) β = XᵀWy for every series at once. y is zero where
    # missing, so XᵀWy needs no mask. Only y and the mask are as large as the data.
    XtWX = xr.dot(valid, w * X * X.rename(term="term_"), dim=time_dim)
    XtWy = xr.dot(y, w * X, dim=time_dim)
    # A series can be fit when XᵀWX has full rank m, so the samples determine
    # every coefficient. The rank counts singular values above
    # σ_max · m · machine epsilon (NumPy's default).
    fitted = linalg.matrix_rank(XtWX, **_MATRIX_OPTIONS) == m

    # β = (XᵀWX)⁻¹ XᵀWy. Series that cannot be fit get an identity matrix so the
    # batched inverse never fails, and are set to NaN at the end. (XᵀWX)⁻¹ is
    # symmetric, so contracting over either of its dims gives β.
    identity = xr.DataArray(np.eye(m), dims=("term", "term_"))
    XtWX_inv = linalg.inv(XtWX.where(fitted, identity), **_MATRIX_OPTIONS)
    beta = xr.dot(XtWX_inv, XtWy, dim="term").rename(term_="term")
    return beta.where(fitted), model


def _design_matrix(t: xr.DataArray, model: dict) -> xr.DataArray:
    """Return the design matrix X with dimensions (term, time)."""
    omega_t = 2 * np.pi * t / model["period"]
    # Scaling τ to [-1, 1] keeps cond(XᵀWX) near 1 to 100, so forming the normal
    # equations loses no precision. Removing each τᵖ's mean keeps μ the mean value.
    tau = (t - model["t0"]) / model["L"]
    columns = {"mean": xr.ones_like(t)}
    for k in range(1, model["n_harmonics"] + 1):
        columns[f"cos{k}"] = xu.cos(k * omega_t)
        columns[f"sin{k}"] = xu.sin(k * omega_t)
    for p in range(1, (model["detrend_degree"] or 0) + 1):
        columns[f"trend{p}"] = tau**p - (tau**p).mean()
    return xr.Dataset(columns).to_dataarray("term")
