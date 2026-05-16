"""Pydantic schemas for Edge Model Quantization Benchmark API."""

from typing import List
from pydantic import BaseModel, Field


class QuantizeTensorRequest(BaseModel):
    matrix_rows: int = Field(default=256, ge=4, le=1024)
    matrix_cols: int = Field(default=256, ge=4, le=1024)
    seed: int = Field(default=42, ge=0)


class QuantizeTensorResponse(BaseModel):
    matrix_shape: List[int]
    scale: float
    fp32_bytes: int
    int8_bytes: int
    compression_ratio: float
    rmse_distortion: float
    cosine_similarity: float


class ProfileItemSchema(BaseModel):
    configuration_name: str
    memory_footprint_kb: float
    compression_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_ips: float
    cosine_similarity: float
    max_absolute_error: float


class BenchmarkSuiteResponse(BaseModel):
    profiles: List[ProfileItemSchema]
    max_compression_ratio: float
    best_retention_similarity: float
    edge_recommendation: str


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    target_bitwidth: int
    benchmark_dims: List[int]
