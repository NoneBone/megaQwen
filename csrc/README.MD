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
kernels/attention_decode_v3.cu \
kernels/attention_prefill.cu

./build/test_attention
```
# prefill单元测试
## prefill seq_major性能

(cp310) l8w@l8w:/media/l8w/Linux118/PROJECTS/29-vllm-serials/09-MegaQwenNew/csrc$ ./build/test_attention
=== Flash Attention kernel tests ===

[PASS] Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)    max_err=0.000152  
[PASS] Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)   max_err=0.000189  
[PASS] Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles) max_err=0.000184  
[PASS] Prefill B=2 S=96  H_q=4  H_kv=2 (batched)          max_err=0.000173  
[PASS] Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)       max_err=0.000181  

=== Prefill benchmarks (warmup=10, iters=10) ===
[BENCH] Prefill B=1 S=2048 H_q=16 H_kv=8                    3102.89 us  11.07 TFLOPS
[BENCH] Prefill B=8 S=2048 H_q=16 H_kv=8                    23489.68 us  11.70 TFLOPS
[BENCH] Prefill B=8 S=4096 H_q=16 H_kv=8                    86492.16 us  12.71 TFLOPS
[BENCH] Prefill B=8 S=8192 H_q=16 H_kv=8                    353525.25 us  12.44 TFLOPS
[BENCH] Prefill B=8 S=16384 H_q=16 H_kv=8                   1383366.38 us  12.72 TFLOPS
[BENCH] Prefill B=8 S=32768 H_q=16 H_kv=8                   5334377.00 us  13.19 TFLOPS

## prefill head_major性能

(cp310) l8w@l8w:/media/l8w/Linux118/PROJECTS/29-vllm-serials/09-MegaQwenNew/csrc$ ./build/test_attention
=== Flash Attention kernel tests ===

[PASS] Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)    max_err=0.001253  
[PASS] Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)   max_err=0.001225  
[PASS] Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles) max_err=0.001375  
[PASS] Prefill B=2 S=96  H_q=4  H_kv=2 (batched)          max_err=0.001248  
[PASS] Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)       max_err=0.001489  

=== Prefill benchmarks (warmup=10, iters=10) ===
[BENCH] Prefill B=1 S=2048 H_q=16 H_kv=8                    3041.41 us  11.30 TFLOPS   8.27 GB/s
[BENCH] Prefill B=8 S=2048 H_q=16 H_kv=8                    24127.30 us  11.39 TFLOPS   8.34 GB/s
[BENCH] Prefill B=8 S=4096 H_q=16 H_kv=8                    89990.04 us  12.22 TFLOPS   4.47 GB/s
[BENCH] Prefill B=8 S=8192 H_q=16 H_kv=8                    345730.34 us  12.72 TFLOPS   2.33 GB/s
[BENCH] Prefill B=8 S=16384 H_q=16 H_kv=8                   1381868.62 us  12.73 TFLOPS   1.17 GB/s
[BENCH] Prefill B=8 S=32768 H_q=16 H_kv=8                   5481308.50 us  12.84 TFLOPS   0.59 GB/s

## 解释
“半精度”（FP16）格式拥有**10 位小数位** → 相对量化误差约为 0.1%。
“双精度浮点数”（BF16）格式拥有**7 位小数位** → 相对量化误差约为 0.8%。
因为内核将 **Q、K、V** 存储在 BF16 格式中，所以在进行任何运算之前，每个值的精度就已经大约降低了 8 倍。其余的处理流程（矩阵乘法、软最大化、缩放等）都在 FP32 格式下进行，因此额外的误差来源仅仅是 **初始的 BF16 四舍五入** 以及在 WMMA 片段内部发生的 **BF16 到 FP32 的转换**。因此，您所看到的最大绝对误差（约为 1.2×10⁻³）正是与 FP16 格式相比大约粗 8 倍的这种格式所应有的误差值。
简而言之：**内核本身并无任何问题——精度的降低是 BF16 的固有特性**。

| **格式**                    | **符号** | **指数位数** | **小数位数（含隐藏位）** | **相对间距（约等于 2 的负 n 次方）** | **动态范围**                   |
| :-------------------------- | :------- | :----------- | :----------------------- | :----------------------------------- | :----------------------------- |
| **FP16**（“半精度”）        | 1        | 5            | 10                       | 2⁻¹⁰ 约等于 0.001（0.1%）            | 2⁻¹⁴ 至 2¹⁶                    |
| **BF16**（“NVIDIA 半精度”） | 1        | 8            | 7                        | 2⁻⁷ 约等于 0.008（0.8%）             | 2⁻¹³⁷ 至 2¹²⁸ （与 FP32 相同） |