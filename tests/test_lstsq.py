import numpy as np
import pytest

from xharmonics.lstsq import _basis, _lstsq_1d


def test_basis_zero_harmonics_returns_empty_columns():
    t = np.linspace(0.0, 1.0, 8)
    B = _basis(t, fundamental_period=1.0, n_harmonics=0)
    assert B.shape == (8, 0)


def test_basis_column_layout_is_cos_sin_interleaved():
    t = np.linspace(0.0, 1.0, 16, endpoint=False)
    B = _basis(t, fundamental_period=1.0, n_harmonics=3)
    assert B.shape == (16, 6)
    expected_cos1 = np.cos(2 * np.pi * t)
    expected_sin1 = np.sin(2 * np.pi * t)
    expected_cos2 = np.cos(4 * np.pi * t)
    np.testing.assert_allclose(B[:, 0], expected_cos1)
    np.testing.assert_allclose(B[:, 1], expected_sin1)
    np.testing.assert_allclose(B[:, 2], expected_cos2)


def test_lstsq_1d_recovers_known_coefficients():
    t = np.linspace(0.0, 4.0, 200, endpoint=False)
    x = 7.0 + 1.5 * np.cos(2 * np.pi * t) - 0.4 * np.sin(2 * np.pi * t)
    weights = np.ones_like(t)
    coef = _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=1, skipna=True)
    assert coef.shape == (2, 2)
    assert np.isclose(coef[0, 0], 7.0)
    assert coef[0, 1] == 0.0
    assert np.isclose(coef[1, 0], 1.5, atol=1e-6)
    assert np.isclose(coef[1, 1], -0.4, atol=1e-6)


def test_lstsq_1d_n_harmonics_zero_returns_only_mean():
    t = np.linspace(0.0, 1.0, 24, endpoint=False)
    x = 3.5 + np.cos(2 * np.pi * t)
    weights = np.ones_like(t)
    coef = _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=0, skipna=True)
    assert coef.shape == (1, 2)
    assert np.isclose(coef[0, 0], np.mean(x))
    assert coef[0, 1] == 0.0


def test_lstsq_1d_skipna_true_handles_nans():
    t = np.linspace(0.0, 4.0, 200, endpoint=False)
    x = 2.0 + np.cos(2 * np.pi * t)
    x[5:10] = np.nan
    weights = np.ones_like(t)
    coef = _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=1, skipna=True)
    assert np.all(np.isfinite(coef))
    assert np.isclose(coef[1, 0], 1.0, atol=1e-2)


def test_lstsq_1d_skipna_false_propagates_nans():
    t = np.linspace(0.0, 4.0, 200, endpoint=False)
    x = 2.0 + np.cos(2 * np.pi * t)
    x[0] = np.nan
    weights = np.ones_like(t)
    coef = _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=1, skipna=False)
    assert np.all(np.isnan(coef))


def test_lstsq_1d_all_nan_returns_all_nan():
    t = np.linspace(0.0, 1.0, 50, endpoint=False)
    x = np.full_like(t, np.nan)
    weights = np.ones_like(t)
    coef = _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=2, skipna=True)
    assert coef.shape == (3, 2)
    assert np.all(np.isnan(coef))


def test_lstsq_1d_rejects_mismatched_shapes():
    t = np.linspace(0.0, 1.0, 10)
    x = np.zeros(11)
    weights = np.ones(10)
    with pytest.raises(ValueError, match="same 1D shape"):
        _lstsq_1d(x, weights, t, fundamental_period=1.0, n_harmonics=1, skipna=True)
