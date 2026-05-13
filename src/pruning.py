"""Magnitude-based weight pruning and sparse tensor compression."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class PrunedTensor:
    pruned_weights: np.ndarray
    binary_mask: np.ndarray
    target_sparsity: float
    achieved_sparsity: float
    total_parameters: int
    non_zero_parameters: int
    rmse_distortion: float


class MagnitudePruner:
    """Applies unstructured L1-norm magnitude pruning to eliminate low-salience parameters."""

    def prune_tensor(self, weights: np.ndarray, sparsity: float = 0.40) -> PrunedTensor:
        """
        Zeros out the lowest |W| parameters based on target sparsity quantile.
        """
        w = weights.copy()
        flat_abs = np.abs(w.flatten())
        threshold = float(np.percentile(flat_abs, sparsity * 100.0))

        mask = (np.abs(w) >= threshold).astype(np.float32)
        pruned_w = w * mask

        achieved_sparsity = float(np.mean(pruned_w == 0.0))
        rmse = float(np.sqrt(np.mean((weights - pruned_w) ** 2)))

        return PrunedTensor(
            pruned_weights=pruned_w,
            binary_mask=mask,
            target_sparsity=round(sparsity, 2),
            achieved_sparsity=round(achieved_sparsity, 4),
            total_parameters=int(weights.size),
            non_zero_parameters=int(np.count_nonzero(pruned_w)),
            rmse_distortion=round(rmse, 6),
        )
