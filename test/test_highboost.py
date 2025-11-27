# test/test_highboost.py
import numpy as np
import pytest
from core.highboost import compose_highboost_mask, compose_highboost_from_mask_params

def test_highboost_r_validation():
    L = np.zeros((8, 8))
    with pytest.raises(ValueError):
        compose_highboost_mask(L, r=1.0)

def test_highboost_formula_correctness():
    L = np.zeros((8, 8))
    Hb = compose_highboost_mask(L, r=2.0)
    assert np.allclose(Hb, 2.0)  # since L = 0 everywhere

def test_highboost_symmetry_enforced():
    L = np.random.rand(32, 32)
    Hb = compose_highboost_mask(L, r=1.5)
    assert np.allclose(Hb, Hb[::-1, ::-1])

def test_compose_from_params():
    Hb = compose_highboost_from_mask_params(
        shape=(32, 32), D0=10, mask_type="gaussian", order=2, r=2.0
    )
    assert Hb.shape == (32, 32)
    assert np.all(Hb >= 1.0)  # r>1 so Hb >= 1 always
