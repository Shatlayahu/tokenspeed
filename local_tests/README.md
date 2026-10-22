1. 环境：Ascend 910C、CANN、PyTorch NPU、Triton-Ascend 版本。
2. 已验证算子：Conv1D、GemmaRMSNorm、RoPE、GQA Prologue、Output Gate、MHA 集成。
3. 测试命令及通过结果。
4. 已知限制：Conv1D 输入 stride、BF16 数值容差、PyTorch RMSNorm fallback。
5. 尚未验证：真实 Qwen3.5 模型权重加载、ForwardContext 和端到端推理。