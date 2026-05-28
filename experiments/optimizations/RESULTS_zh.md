experiments/optimizations/RESULTS_zh.md
# Megakernel 优化实验

## 目标
将每层的 `grid.sync()` 调用次数从 8 次降低到理论最小的 5 次。

当前：8 syncs/层 × 28 层 = 每个解码步骤 224 次 sync。

## 优化 1：冗余 RMSNorm

**状态**：已实现并测试

**思路**：让 **所有 82 个 block** 都冗余计算 RMSNorm，而不是仅在 block 0 上计算。隐藏状态只有 1024 元素（2 KB）——在第一个 block 读取后即可放入 L2 缓存。

**消除的 sync 次数**：56 次（每层 2 次 × 28 层）

**结果**：
```
Position    Original    Optimized    Speedup
-------------------------------------------------
1           5.655ms     3.982ms      1.42x
10          5.708ms     4.006ms      1.42x
50          5.819ms     4.107ms      1.42x
100         5.936ms     4.227ms      1.40x
200         6.202ms     6.883ms      0.90x
-------------------------------------------------
Average     5.864ms     4.641ms      1.26x

Original:  170.5 tok/s
Optimized: 215.5 tok/s
Improvement: +44.9 tok/s (26.3%)
```

**关键发现**：在短‑中等序列上提升显著，但在更长序列（position 200+）上出现退化。这可能是因为冗余的 RMSNorm 读取与 KV 缓存读取竞争 L2 缓存，导致缓存压力增大。

**权衡**：适用于交互式、短上下文的使用场景；对长上下文工作负载可能会降低吞吐。

## 优化 2：基于 Head 的工作分配

**状态**：已实现并测试 —— **不可行**

**思路**：将 block 分配给注意力 head（每个 Q head 负责 5 个 block），使 QKV + attention 能在 head 本地完成，无需在 QKV 与 attention 之间进行 `grid.sync`。

**消除的 sync 次数**：28 次（每层 1 次）

**结果**：
```
Position    Original    Optimized    Speedup
-------------------------------------------------
1           4.023ms     6.862ms      0.59x
10          4.051ms     6.892ms      0.59x
50          4.151ms     6.990ms      0.59x
100         4.272ms     7.039ms      0.61x
200         6.921ms     7.293ms      0.95x
-------------------------------------------------
Average     4.683ms     7.015ms      0.67x

Original:  213.5 tok/s
Optimized: 142.5 tok/s
Result:    -33% throughput (WORSE)
```

**失败原因**：
1. QKV 是 **memory‑bound**，需要全部 82 个 block 参与才能获得最大内存并行度  
2. 在 QKV 阶段仅有 16 个 leader block 工作——其余 66 个 block 处于空闲状态  
3. 消除每层 1 次 sync（约 5‑10 µs × 28 = 140‑280 µs）不足以弥补并行度的损失  
4. 内存带宽的下降主导了性能下降，而不是 sync 开销

**经验教训**：不要为了消除 sync 而牺牲 block 的利用率。相较于 sync 开销，块利用率对性能的影响更大。

## 优化 3：相邻阶段融合（Fused Phases）

**状态**：尚未实现

**思路**：将相邻的计算阶段融合：
- **QKV 投影 + QK 归一化 + RoPE**
- **O 投影 + 残差 + post‑attention RMSNorm**

**预计可消除的 sync 次数**：约 56 次（每层 2 次）

## 总结

| Optimization            | Syncs Eliminated | Speedup                        | Status          |
| ----------------------- | ---------------- | ------------------------------ | --------------- |
| Redundant RMSNorm       | 56               | 1.26x（平均），1.42x（短序列）   | 已完成          |
| Head‑Based Distribution | 28               | **0.67x（更慢）**              | 已完成 – 不可行 |
| Fused Phases            | 56               | 待评估                         | 未开始          |

## 运行实验

```bash
# 冗余 RMSNorm
python experiments/optimizations/redundant_rmsnorm/benchmark.py

# 对比全部（基线）
python experiments/optimizations/compare_all.py
```

## 文件列表

- `redundant_rmsnorm/kernel.cu` – 实现冗余 RMSNorm 的优化 kernel  
- `redundant_rmsnorm/benchmark.py` – 正确性验证与基准测试脚本  
- `compare_all.py` – 基线对比脚本