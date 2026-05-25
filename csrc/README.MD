# Unit testing is implemented using C++.

```sh
cd ./csrc/
mkdir ./build 

/usr/local/cuda-12.8/bin/nvcc \
-std=c++17 \
-O2 \
-arch=sm_89 \
-Iinclude \
-o build/test_attention \
test_unit/test_attention.cu \
kernels/attention_decode.cu \
kernels/attention_decode_paged.cu \
kernels/attention_prefill.cu

./build/test_attention
```

# prefill 单元测试

总结: half 转换为 bfloat 16 会存在精度丢失，而 seq_major 转换为 head_major 的布局转换，不影响精度， 也不影响性能。

- prefill seq_major (half)

```sh
(cp310) l8w@l8w:/media/l8w/Linux118/PROJECTS/29-vllm-serials/09-MegaQwenNew/csrc$ ./build/test_attention
=== Flash Attention kernel tests ===

[PASS] Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)              max_err=0.000152  
[PASS] Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)             max_err=0.000189  
[PASS] Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles)    max_err=0.000184  
[PASS] Prefill B=2 S=96  H_q=4  H_kv=2 (batched)                    max_err=0.000173  
[PASS] Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)                 max_err=0.000181  

=== Prefill benchmarks (warmup=10, iters=10) ===
[BENCH] Prefill B=1 S=2048 H_q=16 H_kv=8                    3102.89 us  11.07 TFLOPS
[BENCH] Prefill B=8 S=2048 H_q=16 H_kv=8                    23489.68 us  11.70 TFLOPS
[BENCH] Prefill B=8 S=4096 H_q=16 H_kv=8                    86492.16 us  12.71 TFLOPS
[BENCH] Prefill B=8 S=8192 H_q=16 H_kv=8                    353525.25 us  12.44 TFLOPS
[BENCH] Prefill B=8 S=16384 H_q=16 H_kv=8                   1383366.38 us  12.72 TFLOPS
[BENCH] Prefill B=8 S=32768 H_q=16 H_kv=8                   5334377.00 us  13.19 TFLOPS
```

- prefill head_major ( bfloat16 )

下方为测试与最新代码匹配。误差有点大，但问题不大，因为：（1）将 bfloat16 全部替换为 __half 后测试，误差恢复正常；（2）集成测试已经通过， 即便单元测试误差略大（但始终小于5e3）。 

```sh

(cp310) l8w@l8w:/media/l8w/Linux118/PROJECTS/29-vllm-serials/09-MegaQwenNew/csrc$ ./build/test_attention
=== Flash Attention kernel tests ===

[PASS] Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)              max_err=0.001253  
[PASS] Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)             max_err=0.001225  
[PASS] Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles)    max_err=0.001375  
[PASS] Prefill B=2 S=96  H_q=4  H_kv=2 (batched)                    max_err=0.001248  
[PASS] Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)                 max_err=0.001489  

=== Prefill benchmarks (warmup=10, iters=10) ===
[BENCH] Prefill B=1 S=2048 H_q=16 H_kv=8                    3041.41 us  11.30 TFLOPS   8.27 GB/s
[BENCH] Prefill B=8 S=2048 H_q=16 H_kv=8                    24127.30 us  11.39 TFLOPS   8.34 GB/s
[BENCH] Prefill B=8 S=4096 H_q=16 H_kv=8                    89990.04 us  12.22 TFLOPS   4.47 GB/s
[BENCH] Prefill B=8 S=8192 H_q=16 H_kv=8                    345730.34 us  12.72 TFLOPS   2.33 GB/s
[BENCH] Prefill B=8 S=16384 H_q=16 H_kv=8                   1381868.62 us  12.73 TFLOPS   1.17 GB/s
[BENCH] Prefill B=8 S=32768 H_q=16 H_kv=8                   5481308.50 us  12.84 TFLOPS   0.59 GB/s
```

## 为什么精度丢失？

“半精度”（FP16）格式拥有**10 位小数位** → 相对量化误差约为 0.1%。

“双精度浮点数”（BF16）格式拥有**7 位小数位** → 相对量化误差约为 0.8%。

因为内核将 **Q、K、V** 存储在 BF16 格式中，所以在进行任何运算之前，每个值的精度就已经大约降低了 8 倍。其余的处理流程（矩阵乘法、软最大化、缩放等）都在 FP32 格式下进行，因此额外的误差来源仅仅是 **初始的 BF16 四舍五入** 以及在 WMMA 片段内部发生的 **BF16 到 FP32 的转换**。因此，您所看到的最大绝对误差（约为 1.2×10⁻³）正是与 FP16 格式相比大约粗 8 倍的这种格式所应有的误差值。

简而言之：**内核本身并无任何问题——精度的降低是 BF16 的固有特性**。

| **格式**                    | **符号** | **指数位数** | **小数位数（含隐藏位）** | **相对间距（约等于 2 的负 n 次方）** | **动态范围**                   |
| :-------------------------- | :------- | :----------- | :----------------------- | :----------------------------------- | :----------------------------- |
| **FP16**（“半精度”）        | 1        | 5            | 10                       | 2⁻¹⁰ 约等于 0.001（0.1%）            | 2⁻¹⁴ 至 2¹⁶                    |
| **BF16**（“NVIDIA 半精度”） | 1        | 8            | 7                        | 2⁻⁷ 约等于 0.008（0.8%）             | 2⁻¹³⁷ 至 2¹²⁸ （与 FP32 相同） |


# decode 单元测试

总结：v1作为baseline, v2 全面超越 v1, v3仅在大批尺寸场景超越 v2, v4 采用split-K, 具有最好的单元测试性能, 但集成测试暂未通过。

- v1: one block per query head, Q 被重复读取 cache_len × head_dim 次
  - 每个线程把 Q 向量的一部分​ 拷进寄存器数组 q_reg[8]
  - 每个 KV 位置都重新从全局内存读取 Q（q_head[i]）
  - 每个线程负责一部分 KV 位置, 输出维度分散在不同线程 → 需要额外同步

- v2: Each block handles one query head, block 级两阶段 softmax
    - 一次性将整个 Q 向量加载到 shared memory, 减少 global memory 访问
    - 提高 cache 命中率, 降低 DRAM 带宽压力（decode 阶段非常关键）
    - 每个线程负责 一个或一组 head_dim 维度, 直接在目标维度上串行累加所有 KV

- v3: warp 级在线 softmax + 多 warp 归约
    - 在 decode 场景下显著减少了同步、shared memory 压力和全局访存，同时提升了数值稳定性和指令效率
    - 每个 warp 独立维护 online softmax 状态

- v4: FlashDecoding‑style 的 Split‑K 并行与两阶段归约 (适用于短序列, 建议根据 HQ * B 调整 num_splits)
    - 显式支持 block table, 与 PagedAttention / vLLM​ 对齐, 更接近生产级推理引擎 ( TODO: 解耦 block table )
    - 128‑bit 对齐加载, 更高 L2 / DRAM 利用率, 更少 load instruction
    - 注意： 集成测试采用的是另外的 launch_attention_decode_paged_splitk

```sh
=== Decode-v1 benchmarks (warmup=10, iters=10) ===
[BENCH] Decode  B=1   ctx=512   H_q=16 H_kv=8                      93.16 us    22.6 GB/s
[BENCH] Decode  B=1   ctx=2048  H_q=16 H_kv=8                     366.69 us    22.9 GB/s
[BENCH] Decode  B=16  ctx=512   H_q=16 H_kv=8                     348.88 us    96.6 GB/s
[BENCH] Decode  B=16  ctx=2048  H_q=16 H_kv=8                    1617.40 us    83.1 GB/s
[BENCH] Decode  B=64  ctx=512   H_q=16 H_kv=8                    1037.31 us   129.9 GB/s
[BENCH] Decode  B=64  ctx=2048  H_q=16 H_kv=8                    4537.95 us   118.4 GB/s
[BENCH] Decode  B=128 ctx=2048  H_q=16 H_kv=8                    8354.61 us   128.6 GB/s

=== Decode-v2 benchmarks (warmup=10, iters=10) ===
[BENCH] Decode  B=1   ctx=512   H_q=16 H_kv=8                      32.77 us    64.3 GB/s
[BENCH] Decode  B=1   ctx=2048  H_q=16 H_kv=8                     123.89 us    67.8 GB/s
[BENCH] Decode  B=16  ctx=512   H_q=16 H_kv=8                     221.88 us   151.8 GB/s
[BENCH] Decode  B=16  ctx=2048  H_q=16 H_kv=8                    1006.81 us   133.4 GB/s
[BENCH] Decode  B=64  ctx=512   H_q=16 H_kv=8                     858.00 us   157.0 GB/s
[BENCH] Decode  B=64  ctx=2048  H_q=16 H_kv=8                    3759.92 us   142.9 GB/s
[BENCH] Decode  B=128 ctx=2048  H_q=16 H_kv=8                    7337.88 us   146.5 GB/s

=== Decode-v3 benchmarks (warmup=10, iters=10) ===
[BENCH] Decode  B=1   ctx=512   H_q=16 H_kv=8                      55.19 us    38.1 GB/s
[BENCH] Decode  B=1   ctx=2048  H_q=16 H_kv=8                     212.38 us    39.5 GB/s
[BENCH] Decode  B=16  ctx=512   H_q=16 H_kv=8                     101.17 us   333.0 GB/s
[BENCH] Decode  B=16  ctx=2048  H_q=16 H_kv=8                     549.36 us   244.6 GB/s
[BENCH] Decode  B=64  ctx=512   H_q=16 H_kv=8                     589.72 us   228.5 GB/s
[BENCH] Decode  B=64  ctx=2048  H_q=16 H_kv=8                    2415.99 us   222.4 GB/s
[BENCH] Decode  B=128 ctx=2048  H_q=16 H_kv=8                    4402.59 us   244.1 GB/s

=== Decode-v4 benchmarks (warmup=10, iters=10) ===
[BENCH] Decode  B=1   ctx=512   H_q=16 H_kv=8                      18.43 us   114.2 GB/s
[BENCH] Decode  B=1   ctx=2048  H_q=16 H_kv=8                      40.65 us   206.5 GB/s
[BENCH] Decode  B=16  ctx=512   H_q=16 H_kv=8                      58.46 us   576.2 GB/s
[BENCH] Decode  B=16  ctx=2048  H_q=16 H_kv=8                     527.13 us   254.9 GB/s
[BENCH] Decode  B=64  ctx=512   H_q=16 H_kv=8                     504.43 us   267.1 GB/s
[BENCH] Decode  B=64  ctx=2048  H_q=16 H_kv=8                    2544.03 us   211.2 GB/s
[BENCH] Decode  B=128 ctx=2048  H_q=16 H_kv=8                    6262.17 us   171.6 GB/s
```

