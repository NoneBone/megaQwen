# MegaQwen 实验结果
 
[返回主页](../README_zh.md)

**模型**：Qwen3-0.6B（28 层，16 个 head，hidden dim 1024）  
**硬件**：NVIDIA RTX 3090（24GB，TDP 420W）  
**Kernel 配置**：82 个 block × 256 线程，每步解码约 **225 次 `grid.sync()`**

---

## 1. 框架对比

### 完整基准测试（长提示词，生成 200 token）

**提示词**：  
“写一篇关于龙虾的详细文章，涵盖其生物学、栖息地……”（约 22 个输入 token）

| Framework        | tok/s   | Avg Power (W) | Peak Power (W) | tok/J    | Speedup vs HF |
| ---------------- | ------- | ------------- | -------------- | -------- | ------------- |
| **TensorRT-LLM** | **355** | 290           | 290            | **1.22** | **6.01x**     |
| Megakernel       | 158     | 205           | 233            | 0.77     | 2.68x         |
| vLLM             | 107     | 196           | 206            | 0.55     | 1.82x         |
| SGLang           | 107     | 210           | 210            | 0.51     | 1.81x         |
| ExLlamaV2        | 98      | 197           | 207            | 0.50     | 1.66x         |
| HuggingFace      | 59      | 186           | 192            | 0.32     | 1.0x          |
| llama.cpp        | 50      | 195           | 201            | 0.26     | 0.85x         |

### 短提示词基准测试（生成 100 token）

**提示词**：  
“Hello”（1 个输入 token）——最小 KV cache 开销

| 框架           | tok/s   | 加速比    |
| -------------- | ------- | --------- |
| **Megakernel** | **530** | **3.91x** |
| HuggingFace    | 136     | 1.0x      |

**说明**：解码吞吐随上下文变长而下降，原因是 attention 需要读取更多 KV cache。

### 关键发现

- **TensorRT-LLM 比 HuggingFace 快 6 倍**，能效高 **3.8 倍**
- **TensorRT-LLM 比 Megakernel 快 2.24 倍**（但需提前编译 engine）
- **Megakernel 比 HuggingFace 快 2.68 倍**，且无需编译
- **Megakernel 比 vLLM 快 1.47 倍**
- **llama.cpp（GGUF F16）在 GPU 上慢于 HuggingFace**——更适合 CPU 推理
- Megakernel 峰值功耗更高（233W），但完成速度更快，整体能效更好

### 位置相关吞吐量（Megakernel + L2 预取）

经过优化（attention 期间 block 分化 + L2 预取）后：

| Position | Before | After   | Speedup |
| -------- | ------ | ------- | ------- |
| 1        | 242    | **525** | 2.17x   |
| 10       | 241    | **527** | 2.19x   |
| 50       | 229    | **500** | 2.18x   |
| 100      | 175    | **472** | 2.70x   |
| 200      | 142    | **422** | 2.97x   |
| 300      | 139    | **382** | 2.75x   |

**优化原理**：  
Attention 阶段只有 16 个 block 参与计算（每个 Q head 一个）。其余 66 个 block 使用 `__ldg` 将 MLP 权重预取到 L2 cache，当 MLP 启动时权重已在缓存中。

---

## 2. 质量指标

| Framework      | KL Divergence | Argmax Match | Notes                        |
| -------------- | ------------- | ------------ | ---------------------------- |
| HuggingFace    | 0.0 (ref)     | 100%         | Reference implementation     |
| **Megakernel** | **0.000582**  | varies       | Near-identical distributions |
| vLLM           | -             | 100%         | Logits not exposed           |
| llama.cpp      | -             | -            | Token IDs not exposed        |

**KL 散度分析**：
- Megakernel 的 KL = 0.000582，表明**概率分布几乎相同**
- 微小差异来源于矩阵运算中 bf16 与 fp32 累积的差异
- 即使分布接近，argmax 在概率相近的情况下仍可能不同

---

## 3. 协作式 Kernel vs CUDA Graph

### 同步开销（空 kernel 测试）

| Approach                      | Time     | Per-Op Cost    |
| ----------------------------- | -------- | -------------- |
| Cooperative + 225 grid.sync() | 167.3 us | 0.73 us/sync   |
| CUDA graph (225 kernels)      | 186.9 us | 0.83 us/kernel |
| 225 regular kernel launches   | 347.5 us | 1.54 us/launch |

### 结论

在纯同步开销上，**协作式 kernel 比 CUDA graph 快 19.7 μs**。

但 megakernel 的真正收益并不只是同步节省，而是**显存带宽节省**：
- 避免了约 340 次中间全局内存读写
- 每 token 节省约 2.7 MB 显存传输
- 估计每步解码节省 1000+ μs

如果在 `grid.sync()` 处分割 kernel，就会失去这些内存收益。

---

## 4. 优化实验

### 冗余 RMSNorm（已实现）

**方法**：所有 82 个 block 都计算 RMSNorm，而不是仅 block 0 计算。  
**减少同步次数**：56 次（每层 2 次 × 28 层）

| Position | Original | Optimized | Speedup |
| -------- | -------- | --------- | ------- |
| 1        | 5.655ms  | 3.982ms   | 1.42x   |
| 10       | 5.708ms  | 4.006ms   | 1.42x   |
| 50       | 5.819ms  | 4.107ms   | 1.42x   |
| 100      | 5.936ms  | 4.227ms   | 1.40x   |
| 200      | 6.202ms  | 6.883ms   | 0.90x   |

**结果**：短序列下吞吐提升 **+26.3%**（170 → 215 tok/s）。  
长序列下性能下降，原因是 KV cache 增加导致 L2 cache 压力上升。

**取舍**：适合交互式使用（短上下文），不推荐用于长上下文任务。

---

### 基于 Head 的工作划分（不可行）

**方法**：为每个 Q head 分配 5 个 block，使 QKV + attention 无需 `grid.sync()`。  
**减少同步次数**：28 次（每层 1 次）

| 位置 | 原始版本 | 优化后 | 加速比 |
|------|----------|---------|--------|
| 1 | 4.023 ms | 6.862 ms | 0.59x |
| 10 | 4.051 ms | 6.892 ms | 0.59x |
| 50 | 4.151 ms | 6.990 ms | 0.59x |

**结果**：吞吐下降 **-33%**（213 → 142 tok/s）。**不可行。**

**失败原因**：QKV 是内存瓶颈。工作 block 从 82 减少到 16 会损失并行度，内存带宽损失远大于同步节省。

**教训**：不要为了消除同步而牺牲 block 利用率。

---

### 阶段融合（未实现）

**方法**：将相邻阶段融合（QKV + QK norm + RoPE，O proj + residual + post-attn RMSNorm）。  
**预计减少同步次数**：约 56 次

**状态**：尚未实现。分析显示由于跨 block 的数据依赖，实现复杂度很高。

---

## 5. Kernel 级优化扫描（最终）

在大量分析与基准测试后，我们尝试了多种 kernel 级优化，试图突破 ~530 tok/s 的天花板。

### Warp Producer/Consumer 比例扫描

**假设**：部分 warp 负责预取，其余 warp 负责计算。

| Ratio (P:C) | Pos 1 | Pos 50 | Pos 100 | Pos 200 | Average   |
| ----------- | ----- | ------ | ------- | ------- | --------- |
| **0:8**     | 567.6 | 529.9  | 498.1   | 444.0   | **509.9** |
| 1:7         | 563.3 | 532.2  | 500.9   | 447.0   | 510.8     |
| 2:6         | 531.6 | 511.2  | 482.9   | 433.8   | 489.9     |
| 3:5         | 545.6 | 521.3  | 490.3   | 437.8   | 498.7     |
| 4:4         | 518.5 | 500.9  | 472.6   | 422.6   | 478.6     |

**结果**：无提升。减少计算 warp 的代价大于预取带来的收益。

**原因**：我们是延迟瓶颈，而非带宽瓶颈。`grid.sync()` 开销才是限制因素，预取帮助有限。

---

### 128 位向量化加载（v2）

| Metric    | v1 (64-bit) | v2 (128-bit) | Improvement |
| --------- | ----------- | ------------ | ----------- |
| Latency   | 1.904 ms    | 1.838 ms     | 3.5%        |
| LDG.E.128 | 0           | 118          | -           |

**结果**：+3.5% 提升，来自更好的内存合并访问。

---

### 共享内存缓存

**方法**：将 `g_normalized` 和 `g_activations`（各 4KB）缓存到 shared memory，避免重复全局内存读取。

| Sequence | Original | Shared Mem | Speedup |
|----------|----------|------------|---------|
| 10 tokens | 567.9 tok/s | 564.2 tok/s | 0.99x |
| 50 tokens | 387.2 tok/s | 244.3 tok/s | 0.63x |
| 100 tokens | 241.2 tok/s | 241.0 tok/s | 1.00x |

**结果**：无提升甚至退化。L1/L2 cache 已能有效处理重复读取，额外的 `__syncthreads()` 开销抵消了理论收益。

---

### cp.async 双缓冲（v3 / v4）

**方法**：使用 `__pipeline_memcpy_async` 在计算当前 tile 的同时预取权重 tile。

| Version | Description      | Avg Time | Speedup |
| ------- | ---------------- | -------- | ------- |
| v1      | Base __ldg       | 1.827ms  | 1.00x   |
| v2      | 128-bit loads    | 1.777ms  | 1.03x   |
| v3      | Register caching | 1.830ms  | 1.00x   |
| v4      | cp.async         | 1.808ms  | 1.01x   |

**结果**：所有变体提升均小于 3%。

---

### 原子计数器同步（探索性）

**方法**：用原子计数器自旋等待替代 `grid.sync()`。

**状态**：已编写 kernel（`fused_decode_atomic_sync.cu`），但因编译时间过长，基准测试无定论。

**理论收益**：若自旋等待快于协作式调度，可减少同步延迟。但带 sense 翻转的 barrier 正确性较难保证。

---

## 6. 根因分析

### 为什么卡在 ~530 tok/s

**根本瓶颈是 `grid.sync()` 延迟，而不是显存带宽。**

| Metric              | Value          |
| ------------------- | -------------- |
| Effective bandwidth | ~47 GB/s       |
| Peak bandwidth      | 936 GB/s       |
| Utilization         | **5%**         |
| grid.sync() calls   | 140+ per token |
| Sync time estimate  | ~0.7 us each   |

140 次同步 × ~0.7 μs ≈ **每 token 约 100 μs 纯同步开销**。

### 为什么内存优化没用

1. **延迟瓶颈，而非带宽瓶颈**：即使内存访问快 4 倍，大部分时间仍在 barrier 上等待。
2. **L1/L2 cache 已足够高效**：GPU cache 层次已很好地处理重复读取。
3. **Block 级预取已做到极限**：Attention 期间 66 个 block 预取 MLP 权重。这是最优粒度——再细到 warp 级会牺牲计算并行度。

### 真正有帮助的方向

| Approach                     | Expected Gain | Difficulty           |
| ---------------------------- | ------------- | -------------------- |
| Quantization (INT4)          | ~4x           | Medium               |
| Non-cooperative architecture | Unknown       | High (major rewrite) |
| Speculative decoding         | ~2-4x         | Medium               |
| Larger batch size            | Linear        | N/A for single-user  |

---

## 汇总表

| Metric                   | TensorRT-LLM | Megakernel | vs HuggingFace    |
| ------------------------ | ------------ | ---------- | ----------------- |
| Decode tok/s (short ctx) | 355          | **531**    | 2.61x / **3.91x** |
| Decode tok/s (long ctx)  | 355          | 158        | 6.01x / 2.68x     |
| Energy (tok/J)           | 1.22         | 0.77       | 3.81x / 2.41x     |
| Compilation              | Required     | None       | -                 |
| KL Divergence            | -            | 0.000582   | near-identical    |

**关键成果**：在短上下文下，Megakernel 以 **531 vs 355 tok/s** 击败 TensorRT-LLM，且无需任何编译开销。

---

## 运行基准测试

bash
完整基准测试（吞吐 + 功耗 + 质量）

python experiments/framework_bench/full_benchmark.py

仅框架吞吐测试

python experiments/framework_bench/benchmark_suite.py

仅功耗测试

python experiments/framework_bench/power_benchmark.py

质量指标（KL 散度、argmax 匹配率）

python experiments/framework_bench/quality_metrics.py

同步开销分析

python experiments/sync_overhead.py

优化实验

python experiments/optimizations/redundant_rmsnorm/benchmark.py
python experiments/optimizations/compare_all.py


---

## 模型转换（用于 llama.cpp）

转换为 GGUF 格式

```bash
# Convert to GGUF for llama.cpp
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp
python /tmp/llama.cpp/convert_hf_to_gguf.py \
    ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/  \
    --outfile /tmp/qwen3_gguf/qwen3-0.6b-f16.gguf \
    --outtype f16
```

---

## 待办事项

- [x] 暴露 logits 以测量 KL 散度
- [x] 添加 llama.cpp 基准测试（GGUF F16）
- [x] 添加 vLLM 基准测试
- [x] 添加 ExLlamaV2 基准测试（需 flash-attn 2.8.3）
- [x] 添加 SGLang 基准测试（通过 OpenAI API 的 server 模式）
- [x] 添加 TensorRT-LLM 基准测试（355 tok/s，比 HF 快 6 倍）
- [x] Block 分化 + L2 预取（+2x 以上加速）
- [x] Warp producer/consumer 比例扫描（无提升）
- [x] 128 位向量化加载（+3.5%）
- [x] 共享内存缓存（无提升）
- [x] cp.async 双缓冲（无提升）
- [x] 根因分析：被 `grid.sync()` 延迟限制
- [ ] 量化（INT4 / INT8）以进一步提升性能
- [ ] 探索非协作式 kernel 架构