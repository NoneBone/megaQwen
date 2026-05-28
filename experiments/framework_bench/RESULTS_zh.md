```markdown experiments/framework_bench/RESULTS_zh.md
# 框架基准测试结果 - Qwen3-0.6B

## 硬件环境
- **GPU**: RTX 3090 (24 GB)  
- **CPU**: AMD Threadripper  
- **CUDA**: 12.1  

## 基准配置
- **Prompt 数量**: 3 条  
- **每条 Prompt 解码长度**: 100 token  
- **Temperature**: 0.0（贪婪）  

## 吞吐量（简化表）

| Framework      | TTFT (s)  | Decode tok/s | Peak Memory | 速度提升 (vs HF) |
| -------------- | --------- | ------------ | ----------- | ---------------- |
| HuggingFace    | 0.071     | 81           | 1.4 GB      | 1.0×             |
| **Megakernel** | **0.004** | **239**      | 2.6 GB      | **2.95×**        |
| vLLM           | 0.047     | 107          | ~2.0 GB     | 1.32×            |
| SGLang         | –         | –            | –           | 需部署服务器     |
| llama.cpp      | –         | –            | –           | 需 GGUF 转换     |

## 能耗与效率（简化表）

| Framework      | 空闲功耗 (W) | 平均功耗 (W) | 峰值功耗 (W) | tok/s | tok/J    | 效率提升 (vs HF) |
| -------------- | ------------ | ------------ | ------------ | ----- | -------- | ---------------- |
| HuggingFace    | 169.1        | 185.3        | 190.2        | 58.4  | 0.31     | 1.0×             |
| **Megakernel** | 172.1        | 200.2        | 228.3        | 156.6 | **0.78** | **2.48×**        |
| vLLM           | 170.8        | 195.4        | 208.8        | 107.3 | 0.55     | 1.74×            |

## 关键发现

1. **Megakernel 的解码吞吐提升 2.95 倍**（239 tok/s vs 81 tok/s）。  
2. **能效提升 2.48 倍**（0.78 tok/J vs 0.31 tok/J）。  
3. **TTFT 加速 17.75 倍**（0.004 s vs 0.071 s），得益于 **fused kernel** 的一次性启动。  
4. vLLM 在能效上比 HuggingFace 高 1.74 倍。  
5. Megakernel 的显存占用更高（2.6 GB vs 1.4 GB），原因是：  
   - 预分配了中间缓冲区  
   - KV cache 在推理开始时一次性全部分配  

## 质量指标

| Framework   | KL Divergence | Perplexity | Argmax Match |
| ----------- | ------------- | ---------- | ------------ |
| HuggingFace | 0.0（基准）   | 35.51      | 100%         |
| Megakernel  | N/A           | N/A        | 100%         |

- Megakernel 在测试 Prompt 上的 **argmax 预测** 与 HuggingFace 完全一致。  
- 完整的 KL 散度对比需要从 Megakernel 暴露 logits，当前尚未实现。

## 运行基准

```bash
# 吞吐量基准
python experiments/framework_bench/benchmark_suite.py

# 质量指标
python experiments/framework_bench/quality_metrics.py
```

## 待办事项

- [x] 在独立进程中运行 vLLM（避免与其他模型共存导致 OOM）  
- [ ] 将模型转换为 GGUF，以便在 llama.cpp / Ollama 中使用  
- [ ] 部署 SGLang 服务器进行对比  
- [ ] 添加 TensorRT-LLM 基准  
- [ ] 添加 ExLlamaV2 基准
```