# W4实验

在补充了在线量化 `_online_quantize_bnb4_weight` 后， qwen3_asr_megakernel的 W4旋钮得以正常工作。实验结果如下：

| 方法           | e2e/ms |
| -------------- | ------ |
| torch          | 220.1  |
| megaKernel，MK | 199.6  |
| MK-W4          | 153.3  |

W4 量化的影响，包括qkv_gemm（0.97->0.71）,gateup_gemm（1.4->0.9）,down_gemm(0.8->0.5)及silu_mul等，符合预期。

- NF4（NormalFloat 4-bit）专门为 LLM 权重分布​ 设计的 4-bit 格式, 假设权重服从 均值为 0 的正态分布, 量化间隔 不均匀（非线性）, 在低比特下, 精度损失最小.

- FP4（Float4）传统 4-bit 浮点表示, 类似 IEEE 浮点（sign + exponent + mantissa）, 不假设权重分布, 表达能力不如 NF4，但更通用.
