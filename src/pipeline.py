"""CLI runner for Edge Model Quantization & Pruning Profiling Suite."""

import argparse
import json
from pathlib import Path
from src.config import DEFAULT_QUANT_CONFIG
from src.benchmark import EdgeModelBenchmarker


def run_pipeline(save_artifact: bool = True):
    print("=" * 68)
    print("EDGE AI MODEL QUANTIZATION & PRUNING RUNTIME BENCHMARK")
    print("=" * 68)

    print("[1/2] Initializing model layer stack (512 -> 1024 -> 256) and profiling...")
    benchmarker = EdgeModelBenchmarker(config=DEFAULT_QUANT_CONFIG, seed=42)
    profiles = benchmarker.run_suite()

    print("\n[2/2] Performance & Numerical Compression Profile:")
    print(f"  {'Configuration':30s} | {'Size (KB)':10s} | {'Comp Ratio':10s} | {'P50 (ms)':9s} | {'Throughput (IPS)':16s} | {'Cosine Sim':10s}")
    print("  " + "-" * 102)
    for p in profiles:
        print(f"  {p.configuration_name:30s} | {p.memory_footprint_kb:7.1f} KB | {p.compression_ratio:8.2f}x | {p.latency_p50_ms:6.3f} ms | {p.throughput_ips:13.1f} IPS | {p.cosine_similarity:10.4f}")

    if save_artifact:
        out_dir = DEFAULT_QUANT_CONFIG.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / DEFAULT_QUANT_CONFIG.summary_filename
        print(f"\n[Artifact] Serializing benchmark profile to: {out_path}")
        with open(out_path, "w") as f:
            json.dump([p.__dict__ for p in profiles], f, indent=2)

    print("\n[SUCCESS] Edge quantization benchmark finished successfully.")
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Quantization Benchmark CLI")
    args = parser.parse_args()
    run_pipeline(save_artifact=True)
