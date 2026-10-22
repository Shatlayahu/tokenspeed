# Qwen3.5-0.8B Ascend assignment-one validation

This directory contains the reproducible deliverables for the integration and
end-to-end-testing role. All commands use the local model snapshot and keep
Hugging Face in offline mode.

## Files

- `serve_qwen35_08b_npu.sh`: single-chip TokenSpeed launch configuration.
- `hf_reference.py`: deterministic Hugging Face reference generation.
- `e2e_correctness.py`: factual checks, HF comparison, and request isolation.
- `benchmark.py`: streaming TTFT, decode latency, E2E latency, and HBM sampling.
- `run_operator_tests.sh`: isolated Conv1D, attention, GDN, and NPU-kernel tests.
- `run_assignment1.sh`: complete reference, server, correctness, and benchmark run.

## Full experiment

```bash
cd /home/u2601112087/tokenspeed
local_tests/qwen35/run_assignment1.sh
```

The default benchmark covers input lengths 32 and 1024, output lengths 32 and
128, three warmups, and ten measured runs. Override it for a smoke test:

```bash
GENERATE_HF_REFERENCE=0 \
RUN_OPERATOR_TESTS=0 \
BENCH_INPUT_LENGTHS=32 \
BENCH_OUTPUT_LENGTHS=8 \
BENCH_WARMUPS=1 \
BENCH_REPEATS=1 \
local_tests/qwen35/run_assignment1.sh
```

Results are written under `local_tests/qwen35/results/`. Raw JSON is retained
for reproducibility; `benchmark/benchmark.md` is ready to paste into the stage
report.

The server runs with `--language-model-only` because assignment one evaluates
text inference. Prefix caching is disabled so the TTFT measurement represents
an uncached Prefill rather than a prefix-cache hit. Qwen3.5 uses head dimension
256, which is outside the fused
Ascend TND attention operator's supported 64/128/192 set; the current correct
fallback is therefore also the main performance-optimization target.
