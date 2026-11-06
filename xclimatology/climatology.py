"""Climatology of xarray objects, with a choice of method.

Each method lives in its own module and provides two functions with the same
names as the ones below: ``climatology(obj, **kwargs)``, which returns the
climatology on the input's time axis, and ``coefficients(da, **kwargs)``, which
returns the fitted parameters. Adding a method means adding a module and one
branch to ``_method``.
"""

from __future__ import annotations

from types import ModuleType

import xarray as xr

from . import harmonics


def _method(name: str) -> ModuleType:
    if name == "harmonic":
        return harmonics
    raise ValueError(f"Unknown method {name!r}. Available methods: 'harmonic'.")


def climatology(
    obj: xr.DataArray | xr.Dataset, *, method: str = "harmonic", **kwargs
) -> xr.DataArray | xr.Dataset:
    """Return the climatology of ``obj`` on its own time axis.

    Parameters
    ----------
    obj : DataArray or Dataset
        Data with a time dimension. For a Dataset, every variable is used.
    method : {"harmonic"}, default: "harmonic"
        ``"harmonic"``: the mean, seasonal harmonics, and an optional trend, fit
        by least squares. See :py:func:`xclimatology.harmonics.climatology`.
    **kwargs
        Arguments of the method, such as ``n_harmonics`` or ``period``.

    Returns
    -------
    DataArray or Dataset
        Climatology with the same dimensions, name, and attributes as ``obj``.

    See Also
    --------
    coefficients

    Examples
    --------
    >>> anomaly = ds - ds.climatology()
    >>> ds.climatology(n_harmonics=3, detrend_degree=None)
    """
    return _method(method).climatology(obj, **kwargs)


def coefficients(da: xr.DataArray, *, method: str = "harmonic", **kwargs) -> xr.Dataset:
    """Return the fitted parameters of the climatology of ``da``.

    Parameters
    ----------
    da : DataArray
        Data with a time dimension.
    method : {"harmonic"}, default: "harmonic"
        ``"harmonic"``: see :py:func:`xclimatology.harmonics.coefficients`.
    **kwargs
        Arguments of the method.

    Returns
    -------
    Dataset
        Fitted parameters. Their names depend on the method.

    See Also
    --------
    climatology
    """
    return _method(method).coefficients(da, **kwargs)


@xr.register_dataarray_accessor("climatology")
@xr.register_dataset_accessor("climatology")
class _ClimatologyAccessor:
    """``obj.climatology(...)`` and ``obj.climatology.coefficients(...)``.

    These call :py:func:`climatology` and :py:func:`coefficients` with ``obj`` as
    the data.
    """

    def __init__(self, obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = obj

    def __call__(self, **kwargs) -> xr.DataArray | xr.Dataset:
        return climatology(self._obj, **kwargs)

    def coefficients(self, **kwargs) -> xr.Dataset:
        return coefficients(self._obj, **kwargs)
