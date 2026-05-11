"""Configuration parameters for Edge Model Quantization & Pruning Benchmark."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class QuantizationConfig:
    # Model Architecture Benchmark Dimensions (Dense Linear Projection Stack)
    input_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 256
    batch_size: int = 32

    # Quantization Parameters
    target_bitwidth: int = 8  # INT8 symmetric uniform affine
    quant_min: int = -128
    quant_max: int = 127

    # Pruning Sparsity Levels for Benchmark
    sparsity_ratios: Tuple[float, ...] = (0.20, 0.40, 0.60)

    # Profiling iterations
    benchmark_warmup_runs: int = 20
    benchmark_timed_runs: int = 100

    # Storage paths
    artifacts_dir: Path = Path("artifacts")
    summary_filename: str = "quantization_benchmark_results.json"


DEFAULT_QUANT_CONFIG = QuantizationConfig()
