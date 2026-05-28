# MegaQwen 开发日志

[返回主页](./README_zh.md)

记录为 Qwen3-0.6B 推理构建并优化**协作式 CUDA megakernel**的完整过程。

**最终结果**：RTX 3090 上实现 **530 tok/s 解码速度**（比 HuggingFace 快 3.9 倍，短上下文下比 TensorRT-LLM 快 1.5 倍）

---

## 目录

1. [目标](#目标)
2. [架构概览](#架构概览)
3. [框架基准测试](#框架基准测试)
4. [优化历程](#优化历程)
5. [根因分析](#根因分析)
6. [有效与无效的优化](#有效与无效的优化)
7. [SASS / PTX 分析](#sass--ptx-分析)
8. [架构天花板](#架构天花板)
9. [经验教训](#经验教训)
10. [未来方向](#未来方向)

---

## 目标

构建一个**单一协作式 CUDA kernel**，将 Qwen3-0.6B 的解码前向（batch=1）完全融合。  
不做中间显存写入、不产生 kernel 启动开销——一个 kernel 从输入直接算到输出。

**模型**：Qwen3-0.6B
- 28 个 Transformer layer
- 16 个 query head，8 个 KV head（GQA）
- hidden dimension：1024
- head dimension：128
- MLP 中间层维度：3072

**硬件**：NVIDIA RTX 3090
- 82 个 SM，10496 个 CUDA core
- 显存带宽 936 GB/s
- FP32 算力 35.6 TFLOPS
- 24 GB VRAM

---

## 架构概览

Megakernel 使用 **cooperative groups** 在所有 82 个 SM 之间同步：

```
┌─────────────────────────────────────────────────────────────┐
│                    MEGAKERNEL（82 个 block）                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 0–27（循环）：                                        │
│    ├─ RMSNorm（仅 block 0）          → grid.sync()         │
│    ├─ QKV Projection（全部 block）   → grid.sync()         │
│    ├─ QK Norm + RoPE（全部 block）   → grid.sync()         │
│    ├─ Attention（16 个 block 对应 16 个 Q head）            │
│    │   └─ 其余 66 个 block：预取 MLP 权重到 L2               │
│    │                                    → grid.sync()     │
│    ├─ O Projection（全部 block）      → grid.sync()         │
│    ├─ Residual Add                                             │
│    ├─ RMSNorm（仅 block 0）          → grid.sync()         │
│    ├─ Gate/Up Projection（全部 block）→ grid.sync()         │
│    ├─ SiLU + Multiply                                            │
│    ├─ Down Projection（全部 block） → grid.sync()         │
│    └─ Residual Add                                               │
├─────────────────────────────────────────────────────────────┤
│  LM Head（全部 block）                 → grid.sync()         │
│  Softmax（仅 block 0）                                             │
└─────────────────────────────────────────────────────────────┘
```

**同步次数**：每步解码约 **225 次 `grid.sync()`**（每层 8 次 × 28 层 + 额外同步）

---

## 框架基准测试

### 长上下文（生成 200 token，提示词约 22 token）

| 框架 | tok/s | 平均功耗 | tok/J | 相对 HuggingFace |
|------|-------|----------|-------|------------------|
| **TensorRT-LLM** | **355** | 290W | **1.22** | **6.01x** |
| Megakernel | 158 | 205W | 0.77 | 2.68x |
| vLLM | 107 | 196W | 0.55 | 1.82x |
| SGLang | 107 | 210W | 0.51 | 1.81x |
| ExLlamaV2 | 98 | 197W | 0.50 | 1.66x |
| HuggingFace | 59 | 186W | 0.32 | 1.0x |
| llama.cpp | 50 | 195W | 0.26 | 0.85x |

### 短上下文（100 token，提示词 1 token）

| 框架 | tok/s | 相对 HuggingFace |
|------|-------|------------------|
| **Megakernel** | **530** | **3.91x** |
| TensorRT-LLM | 355 | 2.61x |
| HuggingFace | 136 | 1.0x |

**关键结论**：在短上下文下，megakernel 以 **530 vs 355 tok/s** 超过 TensorRT-LLM，且无编译开销。

### 位置相关吞吐量

解码吞吐随上下文变长而下降，因为 attention 需要读取更多 KV cache：

| 位置 | 优化前 | 优化后（L2 预取） | 加速比 |
|------|--------|------------------|--------|
| 1 | 242 | **525** | 2.17x |
| 10 | 241 | **527** | 2.19x |
| 50 | 229 | **500** | 2.18x |
| 100 | 175 | **472** | 2.70x |
| 200 | 142 | **422** | 2.97x |
| 300 | 139 | **382** | 2.75x |

---

## 优化历程

### 阶段 1：基线实现

从最直观的协作式 kernel 开始：
- 每 SM 一个 block（共 82 个 block）
- 每 block 256 线程（8 个 warp）
- 使用标准 `__ldg()` 缓存权重读取
- 每个阶段之间都调用 `grid.sync()`

**初始结果**：约 170 tok/s

---

### 阶段 2：冗余 RMSNorm

**假设**：RMSNorm 原本只让 block 0 计算，其余 block 等待。如果让所有 block 都计算呢？

cuda
// 优化前：只有 block 0 计算，其余等待
if (block_id == 0) {
    compute_rmsnorm(...);
}
grid.sync();  // 所有 block 等待

// 优化后：所有 block 都计算，无需同步
compute_rmsnorm(...);


**结果**：短上下文下 **+42%**（170 → 215 tok/s），但在长上下文下变差，原因是冗余权重读取导致 L2 cache 压力增大。

**减少同步次数**：56 次（每层 2 次 × 28 层）

---

### 阶段 3：Block 分化 + L2 预取

**关键洞察**：Attention 阶段只有 16 个 block 参与计算（每个 Q head 一个）。其余 66 个 block 都在 `grid.sync()` 处空等。

**方案**：让空闲 block 把下一阶段的 MLP 权重预取到 L2 cache 中：

cuda
if (block_id < NUM_Q_HEADS) {
    // 计算 attention
    compute_attention(block_id, ...);
} else {
    // 使用 __ldg() 预取 MLP 权重
    prefetch_mlp_weights(block_id - NUM_Q_HEADS, ...);
}
grid.sync();


**结果**：**+2x 加速**（位置 1：242 → 530 tok/s）

这是唯一带来显著收益的优化。

---

### 阶段 4：128 位向量化加载

**假设**：一次性加载 128 位（uint4）而不是 64 位（uint2），应该能改善内存合并访问。

cuda
// 优化前：64 位加载
uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));

// 优化后：128 位加载
uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));


**SASS 分析结果**：

优化前：LDG.E.128 = 0（无 128 位加载）
优化后：LDG.E.128 = 118（存在 128 位加载）


**结果**：+3.5%（1.904 ms → 1.838 ms）

提升较小——我们并不是带宽瓶颈。

---

### 阶段 5：Warp Producer / Consumer 分工

**假设**：部分 warp 负责预取，其余 warp 负责计算。

cuda
bool is_producer = (warp_id < NUM_PRODUCER_WARPS);
if (is_producer) {
    // 预取下阶段数据
    prefetch_weights(...);
} else {
    // 消费 warp 执行计算
    compute_matmul(...);
}


**参数扫描结果**：

| Producer : Consumer 比例 | 平均 tok/s |
|--------------------------|------------|
| **0:8（全计算）** | **509.9** |
| 1:7 | 510.8 |
| 2:6 | 489.9 |
| 3:5 | 498.7 |
| 4:4 | 478.6 |

**结果**：无提升。全计算模式最优。

**原因**：我们被 `grid.sync()` 的延迟限制，而不是带宽限制。减少计算并行度的代价大于预取带来的收益。

---

### 阶段 6：共享内存缓存

**假设**：将 `g_normalized` 和 `g_activations`（各 4KB）缓存到 shared memory，避免重复全局内存读取。

cuda
__shared__ float smem_activations[1024];

// block 开始时一次性加载
for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    smem_activations[i] = g_activations[i];
}
__syncthreads();

// 后续全部从 shared memory 读取
sum += weight * smem_activations[k];


**结果**：0% 提升（部分位置甚至略微下降）

**原因**：L1/L2 cache 已经很好地处理了重复读取。额外的 `__syncthreads()` 开销抵消了理论上的节省。

---

### 阶段 7：cp.async 双缓冲

**假设**：使用异步内存拷贝，在计算当前 tile 的同时加载下一个 tile。

cuda
include <cuda_pipeline.h>

// 计算当前 tile 的同时异步加载下一个 tile
__pipeline_memcpy_async(smem_next, global_next, sizeof(float) * TILE_SIZE);
__pipeline_commit();

compute_tile(smem_current);

__pipeline_wait_prior(0);
swap(smem_current, smem_next);


**结果**：+1%（1.827 ms → 1.808 ms）

**原因**：可重叠的计算量不足。matmul 内层循环太短，无法隐藏内存延迟。

---

### 阶段 8：原子计数器同步（探索性）

**假设**：用原子计数自旋等待替代 `grid.sync()`，以降低同步延迟。

cuda
__device__ void atomic_barrier(int counter, int sense, int num_blocks) {
    __shared__ int local_sense;
    if (threadIdx.x == 0) {
        local_sense = *sense;
        int arrived = atomicAdd(counter, 1);
        if (arrived == num_blocks - 1) {
            *counter = 0;
            *sense = 1 - local_sense;  // 翻转 sense
        }
        while (*sense == local_sense) {}  // 自旋等待
    }
    __syncthreads();
}


**状态**：无定论。Kernel 编译过慢，无法进行可靠基准测试。

---

## 根因分析

### 为什么卡在 ~530 tok/s

经过所有优化，我们发现了根本性瓶颈：

| 指标 | 数值 |
|------|------|
| 有效显存带宽 | ~47 GB/s |
| 峰值显存带宽 | 936 GB/s |
| **利用率** | **5%** |
| 每 token 的 `grid.sync()` 调用次数 | 140+ |
| 单次同步估计延迟 | ~0.7 μs |

**我们只用了可用显存带宽的 5%。**

140 次同步 × ~0.7 μs ≈ **每 token 约 100 μs 纯同步开销**。

### 为什么内存优化没用

1. **延迟瓶颈，而非带宽瓶颈**：即使内存访问快 4 倍，大部分时间仍在 barrier 上等待。
2. **L1/L2 cache 已足够高效**：GPU 的 cache 层次已经很好地处理了重复读取。
3. **Block 级预取已做到极限**：Attention 期间 66 个 block 预取 MLP 权重。这是最优粒度——再往下分到 warp 级会牺牲计算并行度。

### 协作式 vs CUDA Graph 开销

| 方案 | 耗时 | 单次操作成本 |
|------|------|--------------|
| 协作式 + 225 次 grid.sync() | 167.3 μs | 0.73 μs / sync |
| CUDA graph（225 个 kernel） | 186.9 μs | 0.83 μs / kernel |
| 普通 kernel 启动 | 347.5 μs | 1.54 μs / launch |

**协作式比 CUDA graph 快 19.7 μs**，但真正的收益在于避免了每 token 约 2.7 MB 的中间显存传输。

---

## 有效与无效的优化

| 优化方式 | 效果 | 原因 |
|----------|------|------|
| Block 分化 + L2 预取 | **+2x** | 利用 attention 期间的空闲 block |
| 冗余 RMSNorm | +42%（短上下文） | 消除 56 次同步 |
| 128 位向量化加载 | +3.5% | 更好的合并访问 |
| cp.async 双缓冲 | +1% | 重叠有限 |
| Warp producer/consumer | 0% | 减少计算并行度 |
| 共享内存缓存 | 0% | L1/L2 已足够高效 |
| 原子计数器同步 | 未知 | 编译过慢，无法评估 |

**经验**：当系统被同步延迟限制时，内存优化的边际收益会迅速递减。

---

## SASS / PTX 分析

### 原始 Kernel（v1）——64 位加载


LDG.E:              208  （32 位加载）
LDG.E.U16.CONSTANT: 107  （16 位标量加载）
LDG.E.64.CONSTANT:   67  （64 位加载）
LDG.E.CONSTANT:      60  （32 位加载）
LDG.E.128:            0  （无 128 位加载）
STL（寄存器溢出）:      0  （无溢出）


### 优化后 Kernel（v2）——128 位加载


LDG.E.128.CONSTANT:  66  （128 位加载）
LDG.E.128:           52  （128 位加载）
LDG.E.64.CONSTANT:   31  （64 位加载）
LDG.E:               26  （32 位加载）
STG.E.128:           29  （128 位存储）


128 位加载确实存在，但只带来了 3.5% 的提升。

---

## 架构天花板

**~530 tok/s 是 RTX 3090 上 batch=1、bf16 协作式 megakernel 的架构上限。**

### 为什么这是极限

1. **140+ 次 `grid.sync()`** 是协作式 kernel 的固有成本，所有 SM 都必须到达 barrier。
2. **每次同步约 0.7 μs**——这是硬件 / 驱动层面的开销，无法消除。
3. **每 token 约 100 μs 同步开销** 为延迟设定了下限，无论怎么优化计算或内存都无法绕过。

### 突破天花板的潜在方案

| 方案 | 预期收益 | 难度 |
|------|----------|------|
| **INT4 量化** | ~4x | 中等 |
| 非协作式架构 | 未知 | 高（需要大规模重构） |
| 推测解码（speculative decoding） | ~2–4x | 中等 |
| 更大 batch size | 线性提升 | 不适用于单用户场景 |

---

## 经验教训

### 1. 先 Profiling，再优化

我们最初以为瓶颈是显存带宽，事实并非如此。5% 的带宽利用率揭示了真正的延迟瓶颈。

### 2. 协作式 Kernel 有天然局限

`grid.sync()` 开销在高同步次数下占主导地位。每 token 140+ 次同步几乎不可避免。

### 3. Block 级并行至关重要

L2 预取带来的 +2x 加速，源于利用 attention 期间的空闲 block。这是唯一真正显著的收益来源。

### 4. Warp 级预取在此无效

只有在带宽瓶颈时才值得用计算换预取。我们不是。

### 5. GPU Cache 比你想象得更强

L1/L2 已能高效处理重复读取。显式共享内存缓存往往得不偿失。

### 6. SASS 分析必不可少

查看真实汇编代码，才能确认优化（如 128 位加载）是否真的生效。

---

## 未来方向

### 量化（最现实）

INT4 权重大幅减少内存流量 4 倍。结合现有优化，有望达到 2000+ tok/s。

### 非协作式架构

彻底取消 `grid.sync()`，让每个 SM 处理独立工作。需要大规模重构，收益不确定。

### 推测解码

用小草稿模型预测多个 token，再由主模型并行验证。等效于提升 batch size。

### 连续批处理（Continuous Batching）

将同步开销分摊到多个请求上。不适用于单用户交互式场景。

---

## 文件参考

```
csrc/megakernel/
├── fused_decode_ldg.cu           # 生产级 kernel（含 L2 预取）
├── fused_decode_ldg_v2.cu        # 128 位向量化加载
├── fused_decode_ldg_v3.cu        # cp.async 双缓冲
├── fused_decode_ldg_smem.cu      # 共享内存缓存
├── fused_decode_atomic_sync.cu   # 原子计数器同步实验
└── benchmark_*.py                # 各类基准测试脚本

experiments/
├── warp_sweep/                   # Producer/Consumer 比例扫描
│   ├── sweep.py
│   └── kernel_warp_spec.cu
├── framework_bench/              # 框架对比测试
│   ├── benchmark_suite.py
│   ├── power_benchmark.py
│   └── quality_metrics.py
├── optimizations/
│   ├── redundant_rmsnorm/
│   └── head_based_distribution/
├── RESULTS.md                    # 完整基准测试数据
└── sync_overhead.py              # 协作式 vs CUDA graph 对比

docs/
├── ARCHITECTURE.md               # Kernel 架构细节
└── MEMORY_ANALYSIS.md            # 显存带宽分析
```

---

## 运行实验

```bash
# 完整基准测试套件

python experiments/framework_bench/full_benchmark.py

# Warp 比例扫描

python experiments/warp_sweep/sweep.py

# 同步开销分析

python experiments/sync_overhead.py

# 显存分析（需要 cuobjdump）

./analyze_sass.sh csrc/megakernel/fused_decode_ldg.cu

# 交互式聊天

python chat.py
```

---

## 致谢

本项目探索了协作式 megakernel 在 LLM 推理上的极限。虽然我们撞上了架构天花板，但这一过程揭示了 GPU 同步开销的基本规律，以及内存带宽与延迟之间的权衡。

在**短上下文、单用户、低延迟推理**场景中，megakernel 仍具有价值——它甚至在短上下文下击败了 TensorRT-LLM（530 vs 355 tok/s）。

---

*最后更新：2026 年 2 月*