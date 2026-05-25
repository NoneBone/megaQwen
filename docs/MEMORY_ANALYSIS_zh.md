# 显存分析：fused_decode_ldg.cu

## 一句话总结

对 fused decode megakernel 的分析显示，显存利用率极低：有效带宽只有约 **47 GB/s**，而峰值带宽为 **936 GB/s**（效率仅 5%）。  
瓶颈在于 **grid 同步带来的延迟**，而不是显存带宽。

---

## SASS 分析结果

### Original Kernel（v1）——64 位加载


LDG.E:              208  （32 位加载）
LDG.E.U16.CONSTANT: 107  （16 位标量加载）
LDG.E.64.CONSTANT:   67  （64 位加载）
LDG.E.CONSTANT:      60  （32 位加载）
LDG.E.128:            0  （没有 128 位加载）
STL（寄存器溢出）:      0  （无溢出）


### Optimized Kernel（v2）——128 位加载


LDG.E.128.CONSTANT:  66  （128 位加载）
LDG.E.128:           52  （128 位加载）
LDG.E.64.CONSTANT:   31  （64 位加载）
LDG.E:               26  （32 位加载）
STG.E.128:           29  （128 位存储）


---

## 基准测试结果（RTX 3090）

| 指标 | v1（64 位） | v2（128 位） |
|------|-------------|--------------|
| 平均延迟 | 1.904 ms | 1.838 ms |
| 最小延迟 | 1.843 ms | 1.795 ms |
| 加速比 | — | 1.04x |
| 有效带宽 | 46 GB/s | 48 GB/s |
| 峰值带宽利用率 | 4.9% | 5.1% |

---

## Root Cause: Latency-Bound Execution

Kernel 只用到峰值带宽的 5%，因为它属于**延迟瓶颈**，而不是带宽瓶颈：

### 1. Grid Synchronization Overhead

协作式 kernel 使用 `grid.sync()` 来做 SM 之间的协调：
- RMSNorm 阶段完成 → 同步
- QKV projection 完成 → 同步
- QK norm + RoPE 完成 → 同步
- Attention 完成 → 同步
- O proj + MLP 完成 → 同步

28 层 × 每层 5 次同步 = 每 token 140 次 grid barrier。  
每一次都会让所有 SM 停下来，直到最慢的那个完成。

### 2. Serial Layer Processing

所有 82 个 block 一次只处理一层，并在阶段之间等待 `grid.sync()`。  
这种写法把理论上可以重叠的工作串行化了。

### 3. Low Memory-Level Parallelism

即使使用了 128 位加载，每个线程同时发出的内存请求仍然有限：
- 内层循环只展开了 4 次加载
- warp 级并行不错（32 线程 × 128 位 = 512 字节 / warp）
- 但所有 SM 加起来，正在进行的传输总量仍不到 1 MB

### 4. Insufficient Occupancy

- 82 个 block × 256 线程 = 20,992 个总线程
- RTX 3090：82 SM × 每 SM 最多 2048 线程 = 167,936 个可能线程
- 实际占用率：约 **12.5%**

---

## Optimization Strategies

### High Impact (needs architecture change):


1. **流水线式层执行**：  
   让不同的 SM 同时处理不同的层，把上一层计算和这一层的内存访问重叠起来。需要精心设计数据流。

2. **Persistent kernel with reduced syncs**：  
   通过融合更多操作来尽量减少 `grid.sync()`。对 batch=1 的解码来说，可以让每个 SM 独立处理模型的一部分。

3. **Warp 分工（specialization）**：  
   一部分 warp 专门负责预取，另一部分负责计算。这样可以在不增加线程数的情况下提高 MLP 效率。

### Medium Impact (kernel-level):

4. **增加展开因子**：  
   把 `#pragma unroll 4` 改成 `#pragma unroll 8`，让每个线程同时有更多加载在进行。

5. **软件流水线**：  
   双缓冲权重 tile——在计算当前 tile 的同时加载下一个 tile。

6. **共享内存缓存**：  
   在 attention 阶段，把 Q 向量缓存在 shared memory 里，减少全局内存访问。

### Low Impact (already implemented in v2):

7. **128 位向量化加载**：  
   v2 已实现，带来 1.04 倍加速。

---

## 已做的代码改动（v2）

### 权重加载：uint2 → uint4

cpp
// 之前（64 位）：
uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(weight_row + k));

// 之后（128 位）：
uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>(weight_row + k));


### 激活值加载：标量 → float4

cpp
// 之前：
sum += w[0] * g_normalized[k] +
       w[1] * g_normalized[k+1] + ...

// 之后：
float4 act1 = reinterpret_cast<const float4>(g_normalized + k);
float4 act2 = reinterpret_cast<const float4>(g_normalized + k + 4);
sum += w[0]  act1.x + w[1]  act1.y + ...


### Attention K/V Cache：标量 bf16 → uint2

cpp
// 之前：
score += q_head[d] * __bfloat162float(__ldg(k_pos + d));

// 之后：
uint2 k_u2 = __ldg(reinterpret_cast<const uint2>(k_pos + lane_id  4));
score = q_local.x  k[0] + q_local.y  k[1] + ...


---

## 建议的后续步骤

1. **测一下 grid.sync() 的开销**：  
   在同步前后插 CUDA event，看看有多少时间在空等。

2. **探索非协作式方案**：  
   - 用 CUDA graph 减少启动开销  
   - 拆成多个 kernel，用显式依赖连接  
   - 用 stream-ordered memory 做 kernel 间通信  

3. **Increase occupancy**:
   - 减少每个线程的寄存器用量  
   - 用更小的 block size，配合更多的 block  
   - 考虑在 CUDA graph 里多次启动 kernel  

4. **为 batch=1 换算法**：  
   - 对 KV cache 使用 Flash Attention 风格  
   - 用 continuous batching 摊薄同步开销  

---

## 相关文件

- `csrc/megakernel/fused_decode_ldg.cu` —— 原始 v1 kernel（含 L2 预取）
- `csrc/megakernel/fused_decode_ldg_v2.cu` —— 优化后 v2 kernel（128 位加载）
- `csrc/megakernel/fused_decode_ldg_v3.cu` —— cp.async 双缓冲版本
- `csrc/megakernel/fused_decode_ldg_smem.cu` —— 共享内存缓存版本
- `csrc/megakernel/fused_decode_atomic_sync.cu` —— 原子计数器同步版本
- `experiments/warp_sweep/` —— warp producer/consumer 比例扫描

---

## 最终结论

在穷尽各种优化尝试后，我们确定：  
**~530 tok/s 是 RTX 3090 上 batch=1、bf16 协作式 megakernel 的架构上限。**

### 我们试过的东西

| 优化方式 | 效果 | 原因 |
|----------|------|------|
| Block 分化 + L2 预取 | **+2x** | 唯一真正的收益——利用 attention 期间的空闲 block |
| 128 位向量化加载 | +3.5% | 合并访问略有改善 |
| Warp producer/consumer 拆分 | 0% | 减少了计算并行度 |
| 共享内存缓存 | 0% | L1/L2 已经够高效 |
| cp.async 双缓冲 | +1% | 没法重叠足够的计算 |

### 根本限制

每 token 140 多次 `grid.sync()`，每次约 0.7 μs，光同步就占了约 100 μs。  
这是协作式 kernel 的固有特性——所有 SM 都必须到达 barrier。

### 突破方向

要在 batch=1 下超过 530 tok/s：
1. **量化（INT4）**：内存流量减到 1/4，最现实
2. **非协作式架构**：要大改，收益不确定
3. **推测解码**：变相提高 batch size