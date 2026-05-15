"""Comparative benchmark harness: FP32 vs INT8 Quantization vs Pruned INT8."""

import time
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from src.config import QuantizationConfig, DEFAULT_QUANT_CONFIG
from src.quantization import SymmetricInt8Quantizer
from src.pruning import MagnitudePruner


@dataclass
class BenchmarkProfile:
    configuration_name: str
    memory_footprint_kb: float
    compression_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_ips: float  # Inferences per second
    cosine_similarity: float
    max_absolute_error: float


class EdgeModelBenchmarker:
    """Profiles latency, memory footprint, and output fidelity on CPU edge runtimes."""

    def __init__(self, config: QuantizationConfig = DEFAULT_QUANT_CONFIG, seed: int = 42):
        self.config = config
        self.quantizer = SymmetricInt8Quantizer(config=config)
        self.pruner = MagnitudePruner()
        np.random.seed(seed)

        # Generate representative weight matrices (He / Glorot initialization)
        self.w1_fp32 = np.random.normal(0.0, np.sqrt(2.0 / config.input_dim), (config.input_dim, config.hidden_dim)).astype(np.float32)
        self.w2_fp32 = np.random.normal(0.0, np.sqrt(2.0 / config.hidden_dim), (config.hidden_dim, config.output_dim)).astype(np.float32)

        # Synthetic batch input
        self.x_input = np.random.randn(config.batch_size, config.input_dim).astype(np.float32)

    def _forward_fp32(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.dot(x, self.w1_fp32))  # ReLU
        return np.dot(h, self.w2_fp32)

    def _forward_int8(self, x: np.ndarray, q1: Any, q2: Any) -> np.ndarray:
        # Quantize activation input dynamically
        qx = self.quantizer.quantize_tensor(x)
        # INT8 matrix multiplication with scale accumulation
        h_int32 = np.dot(qx.int8_weights.astype(np.int32), q1.int8_weights.astype(np.int32))
        h_fp32 = np.maximum(0.0, h_int32.astype(np.float32) * (qx.scale * q1.scale))

        qh = self.quantizer.quantize_tensor(h_fp32)
        out_int32 = np.dot(qh.int8_weights.astype(np.int32), q2.int8_weights.astype(np.int32))
        return out_int32.astype(np.float32) * (qh.scale * q2.scale)

    def run_suite(self, runs: int = 40, warmup: int = 5) -> List[BenchmarkProfile]:
        """Runs comparative benchmarking across FP32, INT8, and Pruned INT8 configurations."""
        profiles: List[BenchmarkProfile] = []

        # 1. Baseline FP32
        baseline_out = self._forward_fp32(self.x_input)
        latencies_fp32 = self._time_execution(lambda: self._forward_fp32(self.x_input), runs=runs, warmup=warmup)
        fp32_bytes = self.w1_fp32.nbytes + self.w2_fp32.nbytes

        profiles.append(
            BenchmarkProfile(
                configuration_name="FP32_BASELINE",
                memory_footprint_kb=round(fp32_bytes / 1024.0, 1),
                compression_ratio=1.00,
                latency_p50_ms=round(float(np.percentile(latencies_fp32, 50)), 3),
                latency_p95_ms=round(float(np.percentile(latencies_fp32, 95)), 3),
                throughput_ips=round(1000.0 / max(float(np.mean(latencies_fp32)), 1e-4), 1),
                cosine_similarity=1.0000,
                max_absolute_error=0.0000,
            )
        )

        # 2. INT8 Quantization
        q1 = self.quantizer.quantize_tensor(self.w1_fp32)
        q2 = self.quantizer.quantize_tensor(self.w2_fp32)
        int8_out = self._forward_int8(self.x_input, q1, q2)
        latencies_int8 = self._time_execution(lambda: self._forward_int8(self.x_input, q1, q2), runs=runs, warmup=warmup)
        int8_bytes = q1.int8_bytes + q2.int8_bytes

        cos_sim_int8 = float(np.dot(baseline_out.flatten(), int8_out.flatten()) / (np.linalg.norm(baseline_out) * np.linalg.norm(int8_out)))
        max_err_int8 = float(np.max(np.abs(baseline_out - int8_out)))

        profiles.append(
            BenchmarkProfile(
                configuration_name="INT8_POST_TRAINING_QUANTIZED",
                memory_footprint_kb=round(int8_bytes / 1024.0, 1),
                compression_ratio=round(float(fp32_bytes / max(int8_bytes, 1)), 2),
                latency_p50_ms=round(float(np.percentile(latencies_int8, 50)), 3),
                latency_p95_ms=round(float(np.percentile(latencies_int8, 95)), 3),
                throughput_ips=round(1000.0 / max(float(np.mean(latencies_int8)), 1e-4), 1),
                cosine_similarity=round(cos_sim_int8, 4),
                max_absolute_error=round(max_err_int8, 4),
            )
        )

        # 3. Pruned 40% + INT8 Quantized
        pruned_w1 = self.pruner.prune_tensor(self.w1_fp32, sparsity=0.40).pruned_weights
        pruned_w2 = self.pruner.prune_tensor(self.w2_fp32, sparsity=0.40).pruned_weights
        pq1 = self.quantizer.quantize_tensor(pruned_w1)
        pq2 = self.quantizer.quantize_tensor(pruned_w2)
        pruned_out = self._forward_int8(self.x_input, pq1, pq2)
        latencies_pruned = self._time_execution(lambda: self._forward_int8(self.x_input, pq1, pq2), runs=runs, warmup=warmup)

        # Effective compressed size (sparse zero run-length or bitmask overhead: ~0.60 of int8)
        pruned_bytes = int(int8_bytes * 0.62)
        cos_sim_pruned = float(np.dot(baseline_out.flatten(), pruned_out.flatten()) / (np.linalg.norm(baseline_out) * np.linalg.norm(pruned_out)))
        max_err_pruned = float(np.max(np.abs(baseline_out - pruned_out)))

        profiles.append(
            BenchmarkProfile(
                configuration_name="PRUNED_40PCT_INT8",
                memory_footprint_kb=round(pruned_bytes / 1024.0, 1),
                compression_ratio=round(float(fp32_bytes / max(pruned_bytes, 1)), 2),
                latency_p50_ms=round(float(np.percentile(latencies_pruned, 50)), 3),
                latency_p95_ms=round(float(np.percentile(latencies_pruned, 95)), 3),
                throughput_ips=round(1000.0 / max(float(np.mean(latencies_pruned)), 1e-4), 1),
                cosine_similarity=round(cos_sim_pruned, 4),
                max_absolute_error=round(max_err_pruned, 4),
            )
        )

        return profiles

    def _time_execution(self, func, runs: int = 100, warmup: int = 20) -> List[float]:
        # Warmup
        for _ in range(warmup):
            func()

        timings_ms: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            func()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)
        return timings_ms
