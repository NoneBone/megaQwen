# MegaQwen

面向 Qwen3-0.6B 推理的自定义 CUDA **megakernel**，在 RTX 3090 上实现 **530 tok/s decode速度**（比 HuggingFace 快 3.9 倍）。

## 性能表现

| 后端 | decode速度（tok/s） | 加速比 |
|------|------------------|--------|
| **Megakernel** | **531** | **3.9x** |
| TensorRT-LLM | 355 | 2.6x |
| vLLM | 107 | 0.8x |
| SGLang | 107 | 0.8x |
| HuggingFace | 136 | 1.0x |

**说明**：decode吞吐受上下文长度影响。第 1 个为 525 tok/s，第 200 个为 422 tok/s。完整基准测试见 [experiments/RESULTS.md](experiments/RESULTS_zh.md)。

## 公平对比（反方视角）

必须承认：**TensorRT-LLM、vLLM、SGLang 及其他框架在生产负载下的优化已非常成熟**，支持动态 shape、可变 batch size 和长上下文。本 megakernel 利用了它们有意不采用的几类优势：

1. **静态 shape**：所有维度（hidden size、head 数、MLP 宽度）均为编译期常量。生产框架必须在运行时适配任意模型结构。
2. **短上下文偏向**：基准测试集中在位置 1–100，此时 KV cache 开销极小。更长上下文中，TensorRT-LLM 稳定保持 355 tok/s，优于 megakernel 下降至 158 tok/s 的表现。
3. **单模型、单 GPU**：无张量并行、无连续批处理、无动态内存分配。真实服务系统需要全部这些能力。
4. **学习性质**：本项目旨在理解 GPU 优化，而非替代生产级推理引擎。

加速效果真实存在，但来自对**窄场景（batch=1、短上下文、静态 shape）**的极致利用：  
**texture cache（`__ldg()`）通过将权重保留在只读缓存路径中带来巨大收益**，同时由 L1/L2 处理激活值。生产框架无法做此类假设。

**一句话总结**：生产环境用 TensorRT-LLM 或 vLLM；想搞懂 GPU 实际怎么跑，可以用这个项目练手。

## 什么是 Megakernel？

Megakernel 将整个 Transformer block 融合为一次 CUDA kernel 启动，消除 kernel 启动开销与中间显存传输。本实现包含：

- 将 RMSNorm、QKV projection、RoPE、attention、O projection 和 MLP 融合为单个 kernel
- 使用 `__ldg()` 通过 texture cache 读取权重
- 采用 cooperative groups 实现网格级同步
- 实现 online softmax，降低注意力内存占用

## 运行要求

- 计算能力 8.6+ 的 NVIDIA GPU（RTX 3090、A100 等）
- CUDA 11.8+
- Python 3.10+

## 安装步骤
```bash
git clone https://github.com/Infatoshi/MegaQwen.git
cd MegaQwen

# 创建虚拟环境

uv venv
source .venv/bin/activate

# 安装依赖

uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers triton
```

## 使用方法

### 交互式聊天
```bash
python chat.py
```

### 运行基准测试
```bash
python benchmark_suite.py
```

### 验证正确性
```bash
python verify_correctness.py
```

## 关键发现

经过充分优化后，我们发现该 kernel 的性能瓶颈是**同步延迟，而非显存带宽**：

- **仅 5% 显存带宽利用率**（有效 47 GB/s，峰值 936 GB/s）
- 每个 token 触发 **140+ 次 `grid.sync()` 调用**，单次约 0.7μs，累计同步开销约 100μs
- 在 RTX 3090 上，**batch=1、bf16 的协作式 megakernel 架构上限约为 530 tok/s**

### 有效的优化手段

| 优化方式 | 效果 |
|----------|------|
| Block divergence + L2 prefetch | **+2x** |
| 128-bit 向量化加载 | +3.5% |

### 无效的优化尝试

| 优化方式 | 效果 | 原因 |
|----------|------|------|
| Warp producer/consumer 拆分 | 0% | 降低计算并行度 |
| 共享内存缓存 | 0% | L1/L2 已足够高效 |
| cp.async 双缓冲 | +1% | 无法重叠足够多的计算 |

完整优化历程见 [DEVLOG.md](DEVLOG.md)。

## 文档说明

- **[开发日志](DEVLOG_zh.md)** —— 完整优化过程与经验总结
- [基准测试结果](experiments/RESULTS.md) —— 完整测试数据
- [显存分析](docs/MEMORY_ANALYSIS.md) —— 带宽与 SASS 分析
- [架构设计](docs/ARCHITECTURE.md) —— Kernel 架构细节
- [技术规范](SPEC.md) —— 技术规格说明

## 许可证

MIT