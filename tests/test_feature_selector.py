import pytest
import numpy as np
from detection.feature_selector import DynamicFeatureSelector

def test_feature_selector_mask():
    sel = DynamicFeatureSelector(n_features=14, context_dim=8, seed=42)
    z = np.random.randn(8)
    mask = sel.compute_mask(z)
    assert mask.shape == (14,)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)

def test_feature_selector_apply():
    sel = DynamicFeatureSelector(n_features=4, context_dim=4, seed=42)
    x = np.array([10.0, 20.0, 30.0, 40.0])
    z = np.array([0.5, 0.5, 0.5, 0.5])
    
    x_masked, mask = sel.apply_mask(x, z)
    assert x_masked.shape == (4,)
    assert mask.shape == (4,)
    assert np.allclose(x_masked, x * mask)

def test_feature_selector_train_step():
    sel = DynamicFeatureSelector(n_features=4, context_dim=4, seed=42)
    X = np.random.randn(10, 4)
    Z = np.random.randn(10, 4)
    targets = np.ones((10, 4))
    loss = sel.train_step(X, Z, targets, lr=0.05)
    assert isinstance(loss, float)
    assert loss >= 0.0
