"""Uniform symmetric affine INT8 quantization & numerical error tracking."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from src.config import QuantizationConfig, DEFAULT_QUANT_CONFIG


@dataclass
class QuantizedTensor:
    int8_weights: np.ndarray
    scale: float
    fp32_bytes: int
    int8_bytes: int
    compression_ratio: float
    rmse_distortion: float
    cosine_similarity: float


class SymmetricInt8Quantizer:
    """Performs post-training symmetric affine quantization to signed 8-bit integer domain."""

    def __init__(self, config: QuantizationConfig = DEFAULT_QUANT_CONFIG):
        self.config = config

    def quantize_tensor(self, weights_fp32: np.ndarray) -> QuantizedTensor:
        """
        Quantizes continuous FP32 tensor into INT8 representation:
        Scale S = max(|W|) / 127
        Q = round(W / S) clamped to [-128, 127]
        """
        w_flat = weights_fp32.astype(np.float32)
        max_abs = float(np.max(np.abs(w_flat)))
        scale = max_abs / self.config.quant_max if max_abs > 0 else 1.0

        # Discretize and clamp to signed INT8
        int8_quant = np.clip(np.round(w_flat / scale), self.config.quant_min, self.config.quant_max).astype(np.int8)

        # Dequantize to calculate numerical fidelity metrics
        dequant = int8_quant.astype(np.float32) * scale

        rmse = float(np.sqrt(np.mean((w_flat - dequant) ** 2)))
        norm_orig = np.linalg.norm(w_flat)
        norm_dequant = np.linalg.norm(dequant)
        if norm_orig > 0 and norm_dequant > 0:
            cos_sim = float(np.dot(w_flat.flatten(), dequant.flatten()) / (norm_orig * norm_dequant))
        else:
            cos_sim = 1.0

        fp32_bytes = int(weights_fp32.nbytes)
        # In INT8, each element is 1 byte + 4 bytes for single FP32 scalar scale
        int8_bytes = int(int8_quant.nbytes + 4)
        comp_ratio = float(fp32_bytes / max(int8_bytes, 1))

        return QuantizedTensor(
            int8_weights=int8_quant,
            scale=round(scale, 6),
            fp32_bytes=fp32_bytes,
            int8_bytes=int8_bytes,
            compression_ratio=round(comp_ratio, 2),
            rmse_distortion=round(rmse, 6),
            cosine_similarity=round(cos_sim, 6),
        )

    def dequantize(self, int8_weights: np.ndarray, scale: float) -> np.ndarray:
        """Dequantizes INT8 representation back to continuous FP32."""
        return int8_weights.astype(np.float32) * scale
