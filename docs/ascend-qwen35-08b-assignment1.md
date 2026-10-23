# 作业 1：Qwen3.5-0.8B Ascend 单 Chip 模型适配与正确性

## 摘要

本阶段在单个 Ascend 910C chip 上完成了 Qwen3.5-0.8B 的 TokenSpeed
适配。系统能够加载真实 safetensors 权重，执行包含 18 层 Gated DeltaNet
（GDN）和 6 层 Full Attention 的 Prefill、recurrent Decode 与连续文本生成。
关键算子均以 CPU/PyTorch 或独立公式为参考进行了分层数值验证；四个端到端用例
与 Hugging Face FP32 CPU greedy 输出完全一致，请求隔离测试通过。性能实验覆盖
两档输入长度和两档生成长度，每组预热 3 次、正式测量 10 次。1031-token 输入
的 TTFT P50 为约 1.20--1.23 s，Decode P50 为约 0.041--0.042 s/token。
组件微基准进一步表明，长序列 GDN Prefill core 是当前主要计算开销。

## 1. 成员分工与贡献

| 成员 | 主要贡献 |
|---|---|
| `lilyChang409` | GDN chunk Prefill、recurrent Decode、recurrent state 数值验证与独立测试 |
| `Shatlayahu` | Causal Conv1D、QK Norm、RoPE、Attention Gate、MHA 兼容性与算子测试 |
| `shaofeiliu04` | 合并两个计算分支；搭建 TokenSpeed/Ascend 环境；修复模型注册、NPU device context、RMSNorm 和 D=256 MHA；完成真实权重加载、服务启动、HF 参考、端到端/请求隔离测试、性能与组件微基准，并整理本报告 |

## 2. 环境与模型配置

### 2.1 软件和硬件

| 项目 | 配置 |
|---|---|
| NPU | Ascend 910C，单 chip，64 GB HBM |
| CANN | 9.0.0 |
| Python | 3.11.6 |
| PyTorch | 2.6.0+cpu；NPU backend 由 `torch-npu` 注册 |
| torch-npu | 2.6.0.post3 |
| Transformers | 5.12.0 |
| Triton-Ascend | Triton 3.2.0 |
| 模型 | Qwen3.5-0.8B，本地 safetensors |
| 模型精度 | BF16 |
| 并行配置 | world size 1，tensor parallel size 1 |
| 推理范围 | 文本模型，`--language-model-only` |

虽然 PyTorch wheel 的版本字符串包含 `+cpu`，但 `torch-npu` 已注册 NPU device
module；模型权重、激活、KV cache、GDN state 和所有主要矩阵计算均位于 `npu:0`。

### 2.2 模型结构

Qwen3.5-0.8B 文本模型包含 24 个 Decoder Layer：

- 18 层 Linear Attention/GDN；
- 6 层 Full Attention；
- hidden size 1024，FFN intermediate size 3584；
- Full Attention：8 个 Q head、2 个 KV head、head dimension 256；
- GDN：16 个 key/value head、head dimension 128、Conv1D kernel size 4。

## 3. 计算路径与适配内容

### 3.1 端到端调用链

```text
OpenAI HTTP request
  -> TokenSpeed scheduler / model runner
  -> Qwen3_5ForCausalLM
  -> Qwen3_5Model
  -> 24 x DecoderLayer
       -> 18 x GatedDeltaNet
            -> input projection
            -> Causal Conv1D
            -> GDN chunk Prefill / recurrent Decode
            -> gated RMSNorm -> output projection
       -> 6 x Full Attention
            -> QK Norm -> partial RoPE -> KV cache write
            -> MHA Prefill / paged Decode
            -> output gate -> output projection
       -> residual + Gemma RMSNorm -> FFN
  -> final norm -> LM Head -> greedy sampling
```

### 3.2 本组适配

本组复用了 TokenSpeed 的模型框架、调度器、Embedding、LM Head、FFN，以及已有
Ascend RMSNorm、RoPE、KV cache 和融合注意力路径，新增或修正了以下内容：

1. 模型初始化仅在 CUDA 可用时创建 `torch.cuda.Stream`，避免 NPU 启动失败。
2. gated RMSNorm 的 device context 由输入 tensor 的设备模块决定，不再写死 CUDA。
3. NPU add+Gemma RMSNorm 复用 Ascend native 实现，绕过不兼容的 Triton tuple
   编译路径；NPU 不再生成未使用的 FP8/FP4 副本。
4. CANN TND fused attention 仅支持 head dimension 64/128/192。Qwen3.5 的
   head dimension 为 256，因此新增数学等价的 PyTorch/NPU Prefill fallback；
   支持的维度仍进入 CANN fused 实现。
5. 合并 GDN 和 Conv1D 分支，接通真实权重、Prefill/Decode state、模型注册和
   OpenAI-compatible 服务入口。
6. 服务使用 `--language-model-only`，跳过本阶段不要求的视觉编码器。

最大的适配障碍是 CUDA 假设散落在模型初始化和 norm 路径中，以及 CANN fused
attention 对 D=256 的限制。前者通过 device-module dispatch 解决；后者保留
NPU tensor，使用 NPU 上的 FP32 QK、causal mask、softmax 和 BF16 PV fallback，
没有回退到 CPU。

## 4. 关键模块详解

### 4.1 Full Attention Prefill：布局转换与执行

Full Attention Prefill 的输入采用 TND 逻辑布局：

- Q：`[T, 8, 256]`；
- K/V：`[T, 2, 256]`；
- `cu_seqlens_cpu` 给出 packed batch 中每条序列的起止 token offset。

对 D=256 fallback，每条序列依次执行：

1. 根据 `cu_seqlens_cpu` 切出序列，并把 `[T,H,D]` 转为 `[H,T,D]`；
2. GQA 的 KV head 数为 2，Q head 数为 8，因此沿 head 维将 K/V 重复 4 次；
3. Q/K 转为 FP32，计算 `QK^T / sqrt(256)`；
4. 写入上三角 causal mask，执行 softmax；
5. 概率转回 V 的 dtype，执行 `P @ V`；
6. 转回 `[T,8,256]`，进入 attention gate 和 output projection。

Decode 不走此 fallback。每个请求只产生一个 query token，K/V 已写入分页 KV
cache；后端以 BSH 输入布局调用 `npu_fused_infer_attention_score`，使用 block
table 和实际 cache length 访问历史 K/V。

### 4.2 GDN state 更新

GDN Prefill 使用 chunk kernel 扫描整段输入，并输出最后状态；Decode 每次消费
一个 token，在相同 state 上执行 recurrent update。公开 state 布局为
`[B, H_v, V, K]`，本模型中 `H_v=16`、`V=K=128`。独立参考实现内部转为
`[B,H_v,K,V]`，逐 token 执行：

```text
S_t = exp(g_t) * S_(t-1)
delta_t = beta_t * (v_t - k_t @ S_t)
S_t = S_t + k_t^T @ delta_t
o_t = scale * (q_t @ S_t)
```

Prefill final state 与逐 token CPU reference 对照；Decode 对 FP32/BF16 state
连续执行 16 步，每一步比较输出和更新后的 state。请求 A、B、A 的隔离测试还
验证了 scheduler/cache 生命周期没有把 B 的 state 泄漏给第二次 A。

### 4.3 设备同步

正常推理不在每个算子后显式同步。投影、Conv1D、GDN、norm、MHA 和 FFN 排入
同一 NPU execution stream，由 stream 顺序保证依赖；强制逐算子同步会破坏真实性能。
端到端计时以客户端收到 streaming token 为边界，读取响应自然形成可观测完成点。
独立微基准为隔离单个算子，在 warmup 后于每次样本前后调用
`torch.npu.synchronize()`；分阶段 Attention 计时还在阶段边界同步，因此只用于
归因，不与端到端延迟直接相加。

## 5. 算子支持矩阵与分层验证

| 模块 | 主要输入输出/状态 | dtype | 实现来源 | Ascend 状态 | 验证方法 |
|---|---|---|---|---|---|
| Embedding / LM Head | `[T,1024] -> [T,vocab]` | BF16 | TokenSpeed PyTorch | 支持 | 真实权重 E2E |
| Causal Conv1D | `[B,C,T]`，kernel=4 | FP32/BF16 | Ascend/Triton | 支持 | Decode、10 步 state、36 组 Prefill 边界形状 |
| GDN Prefill | Q/K/V head dim 128 | BF16，state FP32/BF16 | Ascend Triton | 支持 | CPU 逐 token reference，output/final state |
| GDN Decode | state `[B,H_v,V,K]` | BF16，state FP32/BF16 | Ascend Triton | 支持 | 连续 16 步逐步 state 对照 |
| QK Norm / partial RoPE | rotary factor 0.25 | FP32/BF16 | Ascend kernel | 支持 | token=1/7、factor=0.25/0.5/1.0，共 12 组 |
| MHA Prefill | Q `[T,8,256]`，KV `[T,2,256]` | BF16/FP32 accumulate | NPU fallback | 正确，待优化 | 变长 GQA 与 FP32 reference，含 D=256 |
| MHA Decode | paged KV cache | BF16 | CANN fused op | 支持 | paged Decode 和 graph state 更新 |
| Attention Gate | `[T,8,256]` | FP32/BF16 | Ascend Triton | 支持 | token=1/7，2D/3D stride，共 8 组 |
| gated RMSNorm | group size 128 | BF16 | Triton-Ascend | 支持 | FP32 reference |
| Gemma RMSNorm | hidden 1024 + residual | BF16 | Ascend native | 支持 | token=1/7、empty/residual/state reference |
| FFN | `1024 -> 2x3584 -> 1024` | BF16 | TokenSpeed/PyTorch NPU | 支持 | 真实权重 E2E |

边界覆盖包括：Conv1D `T=1`、小于/等于/大于 kernel 的长度、不规则 packed
batch `[1,17,3]`、channel 257；norm/rope 的单 token；MHA 的 GQA、D=256；
GDN 的多步 Prefill→Decode 和不同 state dtype。

### 5.1 误差判据与结果

FP32 reference 用于公式正确性；模型运行使用 BF16。BF16 测试通常采用
`atol=rtol=2e-2`，recurrent state 与 grouped/gated norm 放宽至 `3e-2`。
依据是 BF16 仅有 7-bit fraction，矩阵乘和 recurrent update 会累积舍入误差；
阈值仍明显小于随机输入的典型幅值。无需不同硬件逐位相同。

关键实测结果：

- GDN Prefill output 最大误差 `2.44e-4`；
- GDN Prefill final state 最大误差 `1.64e-3`；
- BF16 recurrent Decode 最大 state 误差 `3.91e-3`；
- Gated Attention 三步 Decode 最大误差 `3.91e-3`；
- 所有 Conv1D、GDN、MHA、RoPE、QK Norm、gate、KV write 和 norm 测试通过。

## 6. 端到端正确性

参考端为 Transformers 原始 `Qwen3_5ForConditionalGeneration`、FP32 CPU、
greedy decoding；TokenSpeed 端为 BF16 NPU、temperature 0。测试固定本地权重、
tokenizer 和最大生成长度。

| 用例 | Prompt tokens | Hugging Face | TokenSpeed | 结果 |
|---|---:|---|---|---|
| 简单数学 | 22 | `1 加 1 等于 2。` | `1 加 1 等于 2。` | 完全一致 |
| 中文事实 | 22 | `北京` | `北京` | 完全一致 |
| 英文指令 | 22 | `BLUE` | `BLUE` | 完全一致 |
| 长上下文抽取 | 1794 | `ASCEND-910C-27` | `ASCEND-910C-27` | 完全一致 |

状态隔离测试执行 A（数学）→B（长上下文）→A（数学）。两次 A 的 greedy 输出
完全一致，且 JSON 中 `request_isolation.passed=true`，说明 recurrent state 没有
跨请求泄漏。

## 7. 性能基线

### 7.1 测量方法

- 关闭 prefix caching 和 KVStore，避免缓存命中缩短 Prefill；
- temperature 0，`ignore_eos=true`，固定生成 32/128 token；
- 输入目标长度为 32/1024 token，加入 chat template 后实际为 39/1031 token；
- 每组预热 3 次，正式测量 10 次，报告 P50；
- TTFT 为请求发出至第一个 streaming token 到达；
- Decode latency 为第一个 token 后其余 token 的平均间隔；
- HBM 每 0.5 s 通过 `npu-smi` 采样。

### 7.2 结果

| 输入 token | 输出 token | TTFT P50 (s) | Decode P50 (s/token) | E2E P50 (s) | 物理 chip 峰值 HBM (MB) |
|---:|---:|---:|---:|---:|---:|
| 39 | 32 | 0.1600 | 0.0423 | 1.4716 | 29913 |
| 39 | 128 | 0.1218 | 0.0417 | 5.8686 | 29978 |
| 1031 | 32 | 1.1987 | 0.0420 | 2.6668 | 29914 |
| 1031 | 128 | 1.2320 | 0.0407 | 6.8252 | 29914 |

本次服务器为共享环境，物理 chip HBM 包含其他常驻进程，因此它不能直接解释为
本模型独占量。TokenSpeed 启动日志给出的本进程口径更稳定：权重约 1.46 GB，
KV cache 15.63 GB，Torch allocated 17.35 GB，driver view 22.98 GB。报告同时保留
物理总量和进程口径，避免把共享进程内存误归因于本模型。

## 8. 瓶颈分析

### 8.1 端到端现象

输入从 39 增加到 1031 token 后，TTFT 从约 0.12--0.16 s 增加到约
1.20--1.23 s；Decode 则在所有配置中稳定于约 0.041--0.042 s/token。
因此当前优化重点应放在 Prefill，而非 Decode。

### 8.2 代表性组件微基准

所有组件在 NPU BF16 上预热 3 次、测量 10 次。下表为 1031-token P50；
“层数×P50”只用于诊断，不是完整 trace 的精确加和，它未包含 norm、cache write、
框架开销和可能的 kernel overlap。

| 组件 | 层数 | P50/层 (ms) | 层数×P50 (ms) |
|---|---:|---:|---:|
| GDN input projection | 18 | 0.1474 | 2.65 |
| GDN Prefill core | 18 | 41.8110 | 752.60 |
| GDN output projection | 18 | 0.0772 | 1.39 |
| FFN gate/up projection | 24 | 0.1262 | 3.03 |
| FFN SiLU×up | 24 | 0.0711 | 1.71 |
| FFN down projection | 24 | 0.0915 | 2.20 |
| Full Attention core | 6 | 0.5967 | 3.58 |

GDN core 的诊断估计约 752.6 ms，占 1031-token TTFT 的约 61%--63%，并且从
39-token 的 3.324 ms/层增长到 1031-token 的 41.811 ms/层。这是隔离 NPU
kernel 后仍存在的耗时，不是 HTTP、tokenizer 或主机调度造成。GDN 每个 token
需要更新 `[16,128,128]` recurrent state，主要成本来自 chunk recurrence 的计算
以及 state 的反复读写。因此当前最值得优化的环节是 **GDN Prefill core**。

### 8.3 D=256 Full Attention 的次要局部瓶颈

D=256 fallback 在 1031 token 时为 0.5762 ms/层；作为支持路径对照，D=128
CANN fused op 为 0.1664 ms/层。二者 head dimension 不同，不能当作严格同形状
加速比，但说明缺少 D=256 fusion 确有局部代价。fallback 分阶段结果为：

| Tokens | Layout/repeat (ms) | QK+mask+softmax (ms) | PV (ms) | QK 阶段占比 | Score matrix |
|---:|---:|---:|---:|---:|---:|
| 39 | 0.0840 | 0.1395 | 0.0698 | 47.6% | 0.05 MiB |
| 1031 | 0.0949 | 0.4305 | 0.0888 | 70.1% | 32.44 MiB |

长序列中 QK、mask、softmax 的二次复杂度和中间 score matrix 访存成为该模块主要
开销。不过 6 层合计远小于 GDN 的 18 层 core，因此 D=256 fusion 是第二优先级。

## 9. 复现方式

### 9.1 完整正确性与 3×10 基线

```bash
cd /home/u2601112087/tokenspeed
RESULTS_DIR="$PWD/local_tests/qwen35/results/final-3x10" \
BENCH_WARMUPS=3 \
BENCH_REPEATS=10 \
local_tests/qwen35/run_assignment1.sh
```

该命令依次运行算子测试、生成 HF reference、启动单-chip 服务、执行 E2E/请求
隔离测试和 2×2 benchmark。只启动服务可运行：

```bash
local_tests/qwen35/serve_qwen35_08b_npu.sh
```

### 9.2 瓶颈微基准

服务停止后执行：

```bash
source /home/u2601112087/venvs/tokenspeed/bin/activate
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python"

python local_tests/qwen35/profile_prefill_components.py \
  --warmups 3 --repeats 10
python local_tests/qwen35/profile_prefill_attention.py \
  --warmups 3 --repeats 10
```

结果目录包含：

- `e2e_correctness.json`：逐用例输出、HF 对照和请求隔离；
- `benchmark/benchmark.json`：原始样本与配置；
- `benchmark/benchmark.csv`：表格数据；
- `benchmark/benchmark.md`：性能汇总；
- `prefill_components/`：GDN、FFN、Attention 代表性组件微基准；
- `prefill_profile/`：D=256 Attention 路径和阶段分解。

仓库同时保存了便于检查的 Markdown 汇总：

- [3×10 性能基线](results/assignment1/benchmark-3x10.md)；
- [Prefill 组件微基准](results/assignment1/prefill-components.md)；
- [Full Attention Prefill 分解](results/assignment1/prefill-attention-profile.md)。

## 10. 结论与后续优化

本阶段已经实现 Qwen3.5-0.8B 在单个 Ascend 910C chip 上的完整文本推理，关键
算子、recurrent state、请求隔离和端到端输出均通过验证，并建立了可重复的
TTFT、Decode latency 和 HBM 基线。

作业要求中的四个问题回答如下：

1. **直接复用与修改内容：**复用 TokenSpeed 模型/调度框架和已有 Ascend kernels；
   修改 NPU device dispatch、RMSNorm、D=256 MHA，并集成 GDN/Conv1D。
2. **最大适配障碍：**CUDA 假设与 CANN D=256 限制；分别通过 device-module
   dispatch 和 NPU math fallback 解决。
3. **状态正确性证明：**Prefill final state、16 步 Decode state 对 CPU reference
   数值对照，并通过 A→B→A 请求隔离验证生命周期。
4. **当前最值得优化的环节：**GDN Prefill core；1031-token 组件微基准为
   41.811 ms/层，18 层诊断估计约 752.6 ms，是模型 TTFT 的主要组成。

后续优化顺序为：先优化/融合 GDN chunk Prefill，减少 recurrent state 的读写与
kernel 开销；再实现 D=256 fused Attention Prefill；最后重新测量端到端 TTFT，
以相同权重、输入和 3×10 配置进行优化前后对照。

## 11. AI 工具使用说明

本阶段使用 Codex 辅助代码检索、环境故障定位、测试脚本搭建、日志汇总和报告
结构整理。所有代码修改均由成员结合 TokenSpeed 调用链复核，并在真实 Ascend
910C 上通过算子测试、HF 对照、端到端推理和性能实验验证；报告中的结论以实际
日志和可复现脚本为依据。
