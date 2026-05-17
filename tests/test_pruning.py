"""Unit tests for magnitude-based weight pruning."""

import numpy as np
from src.pruning import MagnitudePruner


def test_magnitude_pruning_sparsity():
    pruner = MagnitudePruner()
    weights = np.random.randn(100, 100).astype(np.float32)
    p = pruner.prune_tensor(weights, sparsity=0.50)

    assert abs(p.achieved_sparsity - 0.50) < 0.02
    assert p.non_zero_parameters == int(weights.size * 0.50)
    assert p.binary_mask.shape == weights.shape
    # Check that masked values are exactly zero
    assert np.all(p.pruned_weights[p.binary_mask == 0.0] == 0.0)
