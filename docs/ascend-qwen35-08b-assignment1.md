# 作业 1：Qwen3.5-0.8B Ascend 单 Chip 模型适配、正确性与性能基线

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

| 成员 | 分工 | 具体工作 | 交付内容 |
|---|---|---|---|
| 成员 A（`lilyChang409`） | Gated DeltaNet 适配，负责 18 个 Linear Attention Layer 的核心计算 | 梳理 `Qwen3_5GatedDeltaNet` 调用链；适配 GDN Prefill 和 recurrent Decode kernel；验证 recurrent state 的数值正确性；编写 GDN 独立测试 | 可在 Ascend 上运行的 GDN 算子及测试 |
| 成员 B（`Shatlayahu`） | Conv1D 与 Attention 适配，负责卷积计算、状态更新和 Full Attention 兼容性 | 适配 Causal Conv1D Prefill、Decode 和 state update；验证现有 Ascend MHA 对 Qwen3.5 的适用性；检查 QK Norm、RoPE 和 Attention Gate；编写 Conv1D 与完整 Attention Layer 测试 | 可在 Ascend 上运行的 Conv1D，以及 Qwen3.5 Attention 验证结果 |
| 成员 C（`shaofeiliu04`） | 模型集成与端到端测试，负责将成员 A、B 的模块组合为完整推理系统 | 下载并加载 Qwen3.5-0.8B 权重；处理模型注册、NPU device 和运行时依赖；检查 FFN、Embedding 和 LM Head；建立 Hugging Face 参考输出；集成 Prefill/Decode 并验证完整文本生成；记录 TTFT、Decode latency 和 HBM 占用 | 可运行的模型启动脚本、端到端正确性测试和性能基线 |

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

权重从服务器本地模型目录以 safetensors 形式离线加载。模型注册后，TokenSpeed
根据 Qwen3.5 配置构建 24 层文本模型；`load_weights` 将 checkpoint 中分离的
Q/K/V、gate/up 等权重映射到运行时的 packed projection 参数，并使用专用 loader
处理 Conv1D、`A_log` 和 `dt_bias`。服务启动日志确认权重加载完成，权重占用约
1.46 GB；随后四组真实权重端到端结果均与参考端一致。

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
| QK Norm / partial RoPE | rotary factor 0.25 | FP32/BF16 | Ascend kernel | 支持 | 8 组 GQA prologue；token=1/7、factor=0.25/0.5/1.0 的 12 组 RoPE |
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

FP32 reference 用于公式正确性，模型运行使用 BF16。误差阈值按计算链长度设置：
FP32 Conv1D 为 `atol=rtol=1e-4`；GDN BF16 output 为 `2e-2`，recurrent state
为 `3e-2`；BF16 Conv1D Prefill/Prefill→Decode 为 `5e-2`，10 步独立 Decode
为 `1e-1`；完整 gated Attention 为 `6e-2`。Conv1D state 在参考结果转为目标
dtype 后要求逐元素相同。阈值差异来自 BF16 仅有 7-bit fraction，卷积、矩阵乘、
gate 和 recurrent update 的舍入误差会沿多步计算累积；因此不要求不同硬件逐位
相同，但所有结果必须落在预先设定的模块阈值内。

关键实测结果：

- GDN Prefill output 最大误差 `2.44e-4`；
- GDN Prefill final state 最大误差 `1.64e-3`；
- BF16 recurrent Decode 最大 state 误差 `3.91e-3`；
- Causal Conv1D 的 36 组 Prefill、6 组 Prefill→Decode 和 6 组十步 Decode
  用例全部通过；前两类用例日志中的最大误差为 `0`，十步 Decode 均在预设阈值内；
- Gated Attention Prefill 最大误差 `7.812e-3`，三步 Decode 最大误差
  `3.906e-3`；
- Gemma RMSNorm BF16 最大误差 `1.5625e-2`，GQA prologue 最大误差
  `3.125e-2`，partial RoPE 最大误差 `1.5625e-2`，Attention Gate 最大误差
  `1.953e-3`；
- 5 组 MHA、KV cache write、gated/add RMSNorm 以及上述全部测试均通过。

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

状态隔离测试执行 A（简单数学）→B（中文事实）→A（简单数学）。两次 A 的
greedy 输出均为 `1 加 1 等于 2。`，测试结果为通过，说明 recurrent state 没有
跨请求泄漏。

## 7. 性能基线

### 7.1 测量方法

- 关闭 prefix caching 和 KVStore，避免缓存命中缩短 Prefill；
- temperature 0，`ignore_eos=true`，固定生成 32/128 token；
- 输入目标长度为 32/1024 token，加入 chat template 后实际为 39/1031 token；
- 每组预热 3 次，正式测量 10 次，报告 P50 和 P95；
- TTFT 为请求发出至第一个 streaming token 到达；
- Decode latency 为第一个 token 后其余 token 的平均间隔；
- HBM 每 0.5 s 通过 `npu-smi` 采样。

### 7.2 结果

| 输入 token | 输出 token | TTFT P50/P95 (s) | Decode P50/P95 (s/token) | E2E P50/P95 (s) | 物理 chip 峰值 HBM (MB) |
|---:|---:|---:|---:|---:|---:|
| 39 | 32 | 0.1600 / 0.1614 | 0.0423 / 0.0426 | 1.4716 / 1.4825 | 29913 |
| 39 | 128 | 0.1218 / 0.1239 | 0.0417 / 0.0422 | 5.8686 / 5.9415 | 29978 |
| 1031 | 32 | 1.1987 / 1.2012 | 0.0420 / 0.0424 | 2.6668 / 2.7048 | 29914 |
| 1031 | 128 | 1.2320 / 1.3636 | 0.0407 / 0.0423 | 6.8252 / 7.4421 | 29914 |

本次测量开始前的物理 chip HBM 基线为 29913 MB。服务器为共享环境，物理 HBM
还包含其他常驻进程，因此峰值不能直接解释为本模型独占量。TokenSpeed 启动日志
给出的本进程口径更稳定：权重约 1.46 GB，
KV cache 15.63 GB，Torch allocated 17.35 GB，driver view 22.98 GB。报告同时保留
物理总量和进程口径，避免把共享进程内存误归因于本模型。

## 8. 瓶颈分析

### 8.1 端到端现象

输入从 39 增加到 1031 token 后，TTFT 从约 0.12--0.16 s 增加到约
1.20--1.23 s；Decode 则在所有配置中稳定于约 0.041--0.042 s/token。
因此当前优化重点应放在 Prefill，而非 Decode。

### 8.2 代表性组件微基准

所有组件在 NPU BF16 上预热 3 次、测量 10 次。下表同时给出短、长输入的
P50/P95。“层数×P50”只用于诊断，不是完整 trace 的精确加和，它未包含 norm、
cache write、框架开销和可能的 kernel overlap。

| Tokens | 组件 | 层数 | P50/层 (ms) | P95/层 (ms) | 层数×P50 (ms) |
|---:|---|---:|---:|---:|---:|
| 39 | GDN input projection | 18 | 0.0609 | 0.1594 | 1.10 |
| 39 | GDN Prefill core | 18 | 3.3240 | 3.4008 | 59.83 |
| 39 | GDN output projection | 18 | 0.0491 | 0.0662 | 0.88 |
| 39 | FFN gate/up projection | 24 | 0.0511 | 0.0673 | 1.23 |
| 39 | FFN SiLU×up | 24 | 0.0665 | 0.1261 | 1.60 |
| 39 | FFN down projection | 24 | 0.0586 | 0.0722 | 1.41 |
| 39 | Full Attention core | 6 | 0.2289 | 0.3364 | 1.37 |
| 1031 | GDN input projection | 18 | 0.1474 | 0.1744 | 2.65 |
| 1031 | GDN Prefill core | 18 | 41.8110 | 44.3250 | 752.60 |
| 1031 | GDN output projection | 18 | 0.0772 | 0.0942 | 1.39 |
| 1031 | FFN gate/up projection | 24 | 0.1262 | 0.1309 | 3.03 |
| 1031 | FFN SiLU×up | 24 | 0.0711 | 0.0799 | 1.71 |
| 1031 | FFN down projection | 24 | 0.0915 | 0.0982 | 2.20 |
| 1031 | Full Attention core | 6 | 0.5967 | 0.6043 | 3.58 |

GDN core 的诊断估计约 752.6 ms，占 1031-token TTFT 的约 61%--63%，并且从
39-token 的 3.324 ms/层增长到 1031-token 的 41.811 ms/层。这是隔离 NPU
kernel 后仍存在的耗时，不是 HTTP、tokenizer 或主机调度造成。GDN 每个 token
需要更新 `[16,128,128]` recurrent state，主要成本来自 chunk recurrence 的计算
以及 state 的反复读写。因此当前最值得优化的环节是 **GDN Prefill core**。

### 8.3 D=256 Full Attention 的次要局部瓶颈

D=256 fallback 在 1031 token 时为 0.5762 ms/层；作为支持路径对照，D=128
CANN fused op 为 0.1664 ms/层。二者 head dimension 不同，不能当作严格同形状
加速比，但说明缺少 D=256 fusion 确有局部代价。完整算子结果为：

| Tokens | Head dim | 路径 | P50 (ms) | P95 (ms) |
|---:|---:|---|---:|---:|
| 39 | 128 | fused | 0.1326 | 0.2543 |
| 39 | 256 | fallback | 0.2243 | 0.9077 |
| 1031 | 128 | fused | 0.1664 | 0.1708 |
| 1031 | 256 | fallback | 0.5762 | 0.5839 |

D=256 fallback 分阶段结果为：

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
