"""Unit tests for edge model benchmarking suite."""

from src.benchmark import EdgeModelBenchmarker


def test_benchmarker_run_suite():
    benchmarker = EdgeModelBenchmarker()
    # Fast test with 5 runs
    profiles = benchmarker.run_suite(runs=5, warmup=1)

    assert len(profiles) == 3
    names = [p.configuration_name for p in profiles]
    assert "FP32_BASELINE" in names
    assert "INT8_POST_TRAINING_QUANTIZED" in names
    assert "PRUNED_40PCT_INT8" in names

    for p in profiles:
        assert p.memory_footprint_kb > 0
        assert p.latency_p50_ms > 0
        assert p.throughput_ips > 0
        assert 0.90 <= p.cosine_similarity <= 1.0
