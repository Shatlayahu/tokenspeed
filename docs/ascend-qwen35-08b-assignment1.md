# 作业 1：Qwen3.5-0.8B Ascend 单 Chip 模型适配与正确性

## 1. 成员分工与本人贡献

- 成员 A：Gated DeltaNet Prefill、recurrent Decode 与 state 数值验证。
- 成员 B：Causal Conv1D、QK Norm、RoPE、Attention Gate 与 MHA 兼容性。
- 成员 C（本人）：合并 A/B 分支，搭建 TokenSpeed 运行环境，完成模型注册与
  NPU 兼容修复、真实权重加载、Prefill/Decode 集成、Hugging Face 参考输出、
  端到端正确性与请求隔离测试，并建立 TTFT、Decode latency 和 HBM 基线。

## 2. 环境与模型配置

| 项目 | 配置 |
|---|---|
| NPU | Ascend 910C，单 chip，64 GB HBM |
| CANN | 9.0.0 |
| Python | 3.11.6 |
| PyTorch | 2.6.0+cpu（计算设备由 torch-npu 后端提供） |
| torch-npu | 2.6.0.post3 |
| Transformers | 5.12.0 |
| Triton-Ascend | Triton 3.2.0 |
| 模型 | Qwen3.5-0.8B，本地 safetensors |
| 精度 | BF16 |
| 并行配置 | world size 1，tensor parallel size 1 |

Qwen3.5-0.8B 的文本模型包含 24 层，其中 18 层为 Linear Attention/Gated
DeltaNet，6 层为 Full Attention。隐藏维度为 1024，FFN 中间维度为 3584；
Full Attention 使用 8 个 Q head、2 个 KV head 和 256 head dimension；GDN
使用 16 个 key/value head、128 head dimension 和 kernel size 4 的 Conv1D。

## 3. 计算路径与适配内容

文本推理调用链如下：

```text
OpenAI HTTP request
  -> TokenSpeed scheduler / model runner
  -> Qwen3_5ForCausalLM
  -> Qwen3_5Model
  -> 24 x DecoderLayer
       -> 18 x GatedDeltaNet
            -> Causal Conv1D
            -> GDN chunk Prefill / recurrent Decode
            -> gated RMSNorm
       -> 6 x Full Attention
            -> QK Norm + partial RoPE + KV cache write
            -> MHA Prefill / paged Decode
            -> output gate
       -> Gemma RMSNorm + FFN
  -> LM Head -> greedy sampling
```

本组直接复用了已有 Ascend RMSNorm、RoPE、KV cache 和融合注意力实现，并完成
了以下适配：

1. 在 NPU 上不再无条件创建 `torch.cuda.Stream`。
2. gated RMSNorm 的 device context 改为由输入 tensor 的设备模块决定。
3. NPU add+Gemma RMSNorm 使用已有 Ascend 实现，避开不兼容的 Triton tuple
   编译路径；NPU 不生成未使用的 FP8/FP4 副本。
4. CANN TND fused attention 仅支持 head dimension 64/128/192。Qwen3.5 的
   head dimension 为 256，因此为 Prefill 增加 PyTorch/NPU 数学等价 fallback；
   受支持的形状仍使用融合实现。
5. 服务以 `--language-model-only` 运行，跳过本阶段不要求的视觉编码器。

## 4. 算子支持与验证

| 模块 | 主要形状/状态 | 实现 | Ascend 状态 | 验证 |
|---|---|---|---|---|
| Embedding / LM Head | `[T,1024]` / `[T,vocab]` | TokenSpeed PyTorch | 支持 | 真实权重 E2E |
| Causal Conv1D | `[B,C,T]`，kernel=4 | Ascend/Triton | 支持 | 1/多步 Decode、36 组 Prefill |
| GDN Prefill | Q/K/V head dim 128 | Ascend Triton | 支持 | CPU token-by-token reference |
| GDN Decode | recurrent state `[B,HV,V,K]` | Ascend Triton | 支持 | FP32/BF16 各 16 步 state 对照 |
| QK Norm / partial RoPE | rotary factor 0.25 | Ascend kernel | 支持 | 12 组 RoPE、8 组 GQA |
| MHA Prefill | Q=8 heads, KV=2, D=256 | NPU fallback | 正确，待优化 | 变长 GQA FP32 reference |
| MHA Decode | paged KV cache | CANN fused op | 支持 | paged decode 与 NPUGraph 更新 |
| Attention Gate | `[T,8,256]` | Ascend Triton | 支持 | 2D/3D stride，8 组测试 |
| gated RMSNorm | group size 128 | Triton-Ascend | 支持 | BF16 FP32 reference |
| Gemma RMSNorm | hidden 1024 + residual | Ascend native | 支持 | residual/state 与 FP32 reference |
| FFN | 1024 -> 3584 -> 1024 | TokenSpeed/PyTorch NPU | 支持 | 真实权重 E2E |

BF16 测试通常使用 `atol=rtol=2e-2`；recurrent state 与 gated/grouped norm
允许 `3e-2`。这是依据 BF16 有效尾数以及多步状态累积设置的，不要求跨硬件逐位
一致。实测 GDN Prefill output 最大误差为 `2.44e-4`，final state 最大误差为
`1.64e-3`；BF16 recurrent Decode 最大 state 误差为 `3.91e-3`。

## 5. 端到端正确性

参考端使用 Transformers 原始 `Qwen3_5ForConditionalGeneration`、FP32 CPU、
greedy decoding；TokenSpeed 使用 BF16 NPU、temperature 0。四组真实权重输出如下：

| 用例 | Hugging Face | TokenSpeed | 结果 |
|---|---|---|---|
| 简单数学 | `1 加 1 等于 2。` | `1 加 1 等于 2。` | 完全一致 |
| 中文事实 | `北京` | `北京` | 完全一致 |
| 英文指令 | `BLUE` | `BLUE` | 完全一致 |
| 长上下文抽取 | `ASCEND-910C-27` | `ASCEND-910C-27` | 完全一致 |

状态隔离测试在请求 A 与重复请求 A 之间插入请求 B。两次 A 的 greedy 输出完全
相同，说明 recurrent state 没有跨请求泄漏。

## 6. 性能基线

测试关闭 prefix caching 和 KVStore，以测量未缓存 Prefill；temperature 为 0，
输出设置 `ignore_eos=true` 以固定为 32/128 token。每组预热 1 次、正式测量 3
次，下表为 P50。HBM 每 0.5 秒由 `npu-smi` 采样。

| 输入 token | 输出 token | TTFT P50 (s) | Decode P50 (s/token) | E2E P50 (s) | 物理 chip 峰值 HBM (MB) |
|---:|---:|---:|---:|---:|---:|
| 39 | 32 | 0.1568 | 0.0415 | 1.4449 | 26750 |
| 39 | 128 | 0.1216 | 0.0421 | 5.8246 | 26750 |
| 1031 | 32 | 1.1976 | 0.0416 | 2.6677 | 26750 |
| 1031 | 128 | 1.1983 | 0.0420 | 7.3594 | 26753 |

TokenSpeed 自身的启动日志显示：权重约 1.46 GB，KV cache 15.67 GB，Torch
allocated 17.38 GB，driver 视角进程占用约 22.89 GB。`npu-smi` 的约 26.75 GB
是物理 chip 总量，包含同一 chip 上的其他常驻进程，因此报告同时保留两种口径。

## 7. 瓶颈分析

长输入的 TTFT 从约 0.12--0.16 秒增加到约 1.20 秒，而 Decode latency 在所有
配置中稳定在约 0.042 秒/token。当前最值得优化的是 Full Attention Prefill：
Qwen3.5 的 head dimension 256 无法使用当前 CANN TND fused attention，必须走
矩阵乘、causal mask 和 softmax fallback。后续应优先实现/移植 head dimension
256 的 Ascend fused Prefill kernel，再比较 TTFT；Decode 目前已经使用 paged CANN
路径，不是首要瓶颈。

## 8. 复现方式

完整算子、HF reference、E2E 和默认 2x2 基准：

```bash
cd /home/u2601112087/tokenspeed
local_tests/qwen35/run_assignment1.sh
```

只启动模型服务：

```bash
local_tests/qwen35/serve_qwen35_08b_npu.sh
```

正式结果保存在服务器：

```text
/home/u2601112087/tokenspeed-logs/hw1/final-uncached/
```

其中 `e2e_correctness.json` 保存逐用例输出及 HF 对照，`benchmark.json` 保存每次
测量的原始数据，`benchmark.csv` 便于绘图，`benchmark.md` 为汇总表。
