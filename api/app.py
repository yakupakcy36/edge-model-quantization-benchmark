"""FastAPI application for Edge Model Quantization & Pruning Runtime Profiler."""

from contextlib import asynccontextmanager
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_QUANT_CONFIG
from src.quantization import SymmetricInt8Quantizer
from src.pruning import MagnitudePruner
from src.benchmark import EdgeModelBenchmarker, BenchmarkProfile
from api.schemas import (
    QuantizeTensorRequest,
    QuantizeTensorResponse,
    PruneTensorRequest,
    PruneTensorResponse,
    ProfileItemSchema,
    BenchmarkSuiteResponse,
    HealthResponse,
)

quantizer: Optional[SymmetricInt8Quantizer] = None
pruner: Optional[MagnitudePruner] = None
cached_profiles: Optional[List[BenchmarkProfile]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global quantizer, pruner, cached_profiles
    quantizer = SymmetricInt8Quantizer(config=DEFAULT_QUANT_CONFIG)
    pruner = MagnitudePruner()
    # Pre-warm benchmark profile with 15 runs for instant sub-second serving
    benchmarker = EdgeModelBenchmarker(config=DEFAULT_QUANT_CONFIG, seed=42)
    cached_profiles = benchmarker.run_suite(runs=15, warmup=3)
    yield


app = FastAPI(
    title="Edge AI Model Quantization & Pruning Benchmark API",
    description=(
        "Production model compression & edge runtime benchmark service. Evaluates "
        "Post-Training Quantization (PTQ) to INT8 uniform symmetric representation and "
        "L1-magnitude weight pruning, measuring microsecond latencies, throughput (IPS), and memory savings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        engine_ready=(quantizer is not None),
        target_bitwidth=DEFAULT_QUANT_CONFIG.target_bitwidth,
        benchmark_dims=[DEFAULT_QUANT_CONFIG.input_dim, DEFAULT_QUANT_CONFIG.hidden_dim, DEFAULT_QUANT_CONFIG.output_dim],
    )


@app.post("/quantize/tensor", response_model=QuantizeTensorResponse, tags=["Quantization"])
def quantize_custom_tensor(req: QuantizeTensorRequest):
    """Simulates symmetric uniform INT8 quantization on generated tensor and returns error metrics."""
    if quantizer is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Quantizer not ready")

    np.random.seed(req.seed)
    w = np.random.randn(req.matrix_rows, req.matrix_cols).astype(np.float32)
    q = quantizer.quantize_tensor(w)

    return QuantizeTensorResponse(
        matrix_shape=[req.matrix_rows, req.matrix_cols],
        scale=q.scale,
        fp32_bytes=q.fp32_bytes,
        int8_bytes=q.int8_bytes,
        compression_ratio=q.compression_ratio,
        rmse_distortion=q.rmse_distortion,
        cosine_similarity=q.cosine_similarity,
    )


@app.post("/prune/tensor", response_model=PruneTensorResponse, tags=["Pruning"])
def prune_custom_tensor(req: PruneTensorRequest):
    """Applies unstructured magnitude L1 pruning to tensor and returns sparsity metrics."""
    if pruner is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pruner not ready")

    np.random.seed(req.seed)
    w = np.random.randn(req.matrix_rows, req.matrix_cols).astype(np.float32)
    p = pruner.prune_tensor(w, sparsity=req.target_sparsity)

    return PruneTensorResponse(
        matrix_shape=[req.matrix_rows, req.matrix_cols],
        target_sparsity=p.target_sparsity,
        achieved_sparsity=p.achieved_sparsity,
        total_parameters=p.total_parameters,
        non_zero_parameters=p.non_zero_parameters,
        rmse_distortion=p.rmse_distortion,
    )


@app.get("/benchmark/suite", response_model=BenchmarkSuiteResponse, tags=["Benchmarking"])
def get_benchmark_suite():
    """Returns comparative edge runtime profiling results across FP32, INT8, and Pruned configurations."""
    global cached_profiles
    if cached_profiles is None:
        benchmarker = EdgeModelBenchmarker(config=DEFAULT_QUANT_CONFIG, seed=42)
        cached_profiles = benchmarker.run_suite(runs=15, warmup=3)

    items = [ProfileItemSchema(**p.__dict__) for p in cached_profiles]
    max_comp = max(p.compression_ratio for p in cached_profiles)
    best_sim = min(p.cosine_similarity for p in cached_profiles if "INT8" in p.configuration_name)

    rec = (
        f"INT8 Quantization achieves {cached_profiles[1].compression_ratio}x memory compression "
        f"with {cached_profiles[1].cosine_similarity:.4f} cosine similarity retention, recommended for MCU/Gateway deployment."
    )

    return BenchmarkSuiteResponse(
        profiles=items,
        max_compression_ratio=max_comp,
        best_retention_similarity=best_sim,
        edge_recommendation=rec,
    )
