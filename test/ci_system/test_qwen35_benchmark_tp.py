"""Behavioral checks for HW2 NPU HBM parsing and TP benchmark summaries."""

from local_tests.qwen35.benchmark_tp import parse_npu_smi_hbm, summarize


def test_parse_hbm_uses_physical_id_and_last_memory_fraction():
    sample = """
| 3     Ascend910           | OK            | 164.9       | 48 | 0 / 0 |
| 0     6                   | 0000:91:00.0  | 0           0 / 0          26130 / 65536 |
| 3     Ascend910           | OK            | -           | 48 | 0 / 0 |
| 1     7                   | 0000:93:00.0  | 0           0 / 0          25530 / 65536 |
"""
    assert parse_npu_smi_hbm(sample) == {6: 26130, 7: 25530}


def test_summary_counts_tokens_and_keeps_two_device_totals_separate():
    requests = [
        {
            "success": True,
            "completion_tokens": 20,
            "ttft_s": 0.2,
            "tpot_s": 0.1,
            "e2e_s": 2.1,
        },
        {
            "success": True,
            "completion_tokens": 30,
            "ttft_s": 0.4,
            "tpot_s": 0.2,
            "e2e_s": 6.2,
        },
        {
            "success": False,
            "completion_tokens": None,
            "ttft_s": None,
            "tpot_s": None,
            "e2e_s": 1.0,
        },
    ]
    samples = [
        {"hbm_mb": {"6": 100, "7": 150}},
        {"hbm_mb": {"6": 110, "7": 145}},
    ]
    result = summarize(requests, elapsed=10.0, samples=samples, devices=[6, 7])
    assert result["requests"] == 3
    assert result["successes"] == 2
    assert result["output_tokens_per_s"] == 5.0
    assert result["hbm_peak_total_mb"] == {"6": 110, "7": 150}
