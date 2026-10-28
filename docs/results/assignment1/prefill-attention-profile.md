# Qwen3.5 Full Attention Prefill profile

Configuration: BF16, Q heads=8, KV heads=2, warmups=3, repeats=10.

## Whole operator

| Tokens | Head dim | Path | P50 (ms) | P95 (ms) |
|---:|---:|---|---:|---:|
| 39 | 128 | fused | 0.1326 | 0.2543 |
| 39 | 256 | fallback | 0.2243 | 0.9077 |
| 1031 | 128 | fused | 0.1664 | 0.1708 |
| 1031 | 256 | fallback | 0.5762 | 0.5839 |

## D=256 fallback stage breakdown

Stage measurements intentionally synchronize between stages; use them for attribution rather than end-to-end latency.

| Tokens | Layout/repeat P50 (ms) | QK+mask+softmax P50 (ms) | PV P50 (ms) | QK stage share | Score matrix (MiB) |
|---:|---:|---:|---:|---:|---:|
| 39 | 0.0840 | 0.1395 | 0.0698 | 47.6% | 0.05 |
| 1031 | 0.0949 | 0.4305 | 0.0888 | 70.1% | 32.44 |
