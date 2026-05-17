"""Unit tests for INT8 uniform symmetric quantization."""

import numpy as np
from src.quantization import SymmetricInt8Quantizer


def test_quantizer_int8_bounds():
    quantizer = SymmetricInt8Quantizer()
    weights = np.random.randn(64, 64).astype(np.float32)
    q = quantizer.quantize_tensor(weights)

    assert q.int8_weights.dtype == np.int8
    assert -128 <= np.min(q.int8_weights)
    assert np.max(q.int8_weights) <= 127
    assert q.scale > 0.0
    assert q.compression_ratio >= 3.8  # ~4x compression
    assert q.cosine_similarity > 0.99  # High numerical fidelity


def test_dequantization():
    quantizer = SymmetricInt8Quantizer()
    weights = np.array([[1.0, -2.5], [0.5, -0.2]], dtype=np.float32)
    q = quantizer.quantize_tensor(weights)
    dequant = quantizer.dequantize(q.int8_weights, q.scale)

    assert dequant.shape == weights.shape
    np.testing.assert_allclose(weights, dequant, atol=0.05)
