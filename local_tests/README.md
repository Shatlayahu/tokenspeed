# Ascend Qwen3.5 validation

The local suite covers the Qwen3.5-0.8B computation path on one Ascend 910C
chip:

- `conv1d/`: prefill, decode, multi-step state updates, and boundary shapes.
- `attention/`: RMSNorm, RoPE, GQA prologue, output gate, KV storage, and the
  integrated prefill/decode path.
- `qwen35/`: local-weight server launch, Hugging Face reference outputs,
  end-to-end correctness, request isolation, TTFT/decode latency, and HBM.

Run kernel and module tests with the repository, kernel, and NPU-kernel Python
directories on `PYTHONPATH`. Run the complete model experiment with:

```bash
local_tests/qwen35/run_assignment1.sh
```

The text-only assignment uses BF16 and `--language-model-only`. Numerical
tests use FP32 references with `atol=rtol=2e-2` or `3e-2` for recurrent state
and grouped/gated norm paths.
