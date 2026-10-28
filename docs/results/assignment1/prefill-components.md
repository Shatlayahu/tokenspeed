# Qwen3.5-0.8B representative Prefill component profile

Configuration: BF16, warmups=3, repeats=10.

The layer-count estimate is a diagnostic microbenchmark, not a sum of a whole-model trace. It excludes norms, cache writes, framework overhead, and kernel overlap.

| Tokens | Component | Layers | P50/layer (ms) | P95/layer (ms) | Count × P50 (ms) |
|---:|---|---:|---:|---:|---:|
| 39 | gdn_input_projection | 18 | 0.0609 | 0.1594 | 1.10 |
| 39 | gdn_core | 18 | 3.3240 | 3.4008 | 59.83 |
| 39 | gdn_output_projection | 18 | 0.0491 | 0.0662 | 0.88 |
| 39 | ffn_gate_up_projection | 24 | 0.0511 | 0.0673 | 1.23 |
| 39 | ffn_silu_multiply | 24 | 0.0665 | 0.1261 | 1.60 |
| 39 | ffn_down_projection | 24 | 0.0586 | 0.0722 | 1.41 |
| 39 | full_attention_core | 6 | 0.2289 | 0.3364 | 1.37 |
| 1031 | gdn_input_projection | 18 | 0.1474 | 0.1744 | 2.65 |
| 1031 | gdn_core | 18 | 41.8110 | 44.3250 | 752.60 |
| 1031 | gdn_output_projection | 18 | 0.0772 | 0.0942 | 1.39 |
| 1031 | ffn_gate_up_projection | 24 | 0.1262 | 0.1309 | 3.03 |
| 1031 | ffn_silu_multiply | 24 | 0.0711 | 0.0799 | 1.71 |
| 1031 | ffn_down_projection | 24 | 0.0915 | 0.0982 | 2.20 |
| 1031 | full_attention_core | 6 | 0.5967 | 0.6043 | 3.58 |
