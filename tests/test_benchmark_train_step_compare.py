def _make_run(tokens_per_sec_mean: float, peak_vram_mb: float | None) -> dict:
    return {
        "schema_version": 1,
        "run_id": "2026-02-02T00:00:00Z__train_step_e2e",
        "benchmark_name": "train_step_e2e",
        "baseline_name": None,
        "git": {"repo_root": "/tmp", "commit": "abc", "branch": "main", "dirty": False},
        "env": {
            "hostname": "host",
            "os": "Linux",
            "python": "3.11.0",
            "torch_version": "2.x",
            "cuda_version": None,
            "cudnn_version": None,
            "driver": None,
            "gpu_name": None,
            "gpu_count": 0,
            "sm": None,
            "total_vram_mb": None,
        },
        "config": {},
        "metrics_summary": {
            "step_time_mean_s": 0.1,
            "tokens_per_sec_mean": tokens_per_sec_mean,
            "peak_vram_mb": peak_vram_mb,
        },
        "samples": [],
        "notes": "",
    }


def test_validate_benchmark_run_v1_roundtrip():
    from eval.benchmark_train_step import _validate_benchmark_run_v1

    run = _make_run(tokens_per_sec_mean=100.0, peak_vram_mb=1000.0)
    _validate_benchmark_run_v1(run)


def test_compare_pass_small_throughput_drop_ok():
    from eval.benchmark_train_step import compare_benchmark_runs

    baseline = _make_run(tokens_per_sec_mean=100.0, peak_vram_mb=1000.0)
    current = _make_run(tokens_per_sec_mean=97.0, peak_vram_mb=1000.0)
    result = compare_benchmark_runs(baseline, current, throughput_threshold_pct=5.0, vram_threshold_pct=5.0)
    assert result["verdict"] == "pass"


def test_compare_throughput_regression():
    from eval.benchmark_train_step import compare_benchmark_runs

    baseline = _make_run(tokens_per_sec_mean=100.0, peak_vram_mb=1000.0)
    current = _make_run(tokens_per_sec_mean=94.0, peak_vram_mb=1000.0)
    result = compare_benchmark_runs(baseline, current, throughput_threshold_pct=5.0, vram_threshold_pct=5.0)
    assert result["verdict"] == "regression"
    assert any("throughput regression" in r for r in result["reasons"])


def test_compare_vram_regression():
    from eval.benchmark_train_step import compare_benchmark_runs

    baseline = _make_run(tokens_per_sec_mean=100.0, peak_vram_mb=1000.0)
    current = _make_run(tokens_per_sec_mean=100.0, peak_vram_mb=1100.0)
    result = compare_benchmark_runs(baseline, current, throughput_threshold_pct=5.0, vram_threshold_pct=5.0)
    assert result["verdict"] == "regression"
    assert any("VRAM regression" in r for r in result["reasons"])
