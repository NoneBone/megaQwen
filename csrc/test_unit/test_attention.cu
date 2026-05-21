// test_attention.cu
// Validates Flash Attention 2 (prefill) and Paged Attention (decode) kernels
// against a CPU FP32 naive reference.
//
// Pass criterion (from DESIGN.md): max absolute error < 5e-3 vs naive attention.
//
// Prefill test cases:
//   1. B=1, S=16,  H_q=2, H_kv=1 — small, GQA ratio 2:1
//   2. B=1, S=64,  H_q=4, H_kv=2 — single KV tile (Bc=64)
//   3. B=1, S=128, H_q=16, H_kv=8 — Qwen3 head config, two KV tiles
//   4. B=2, S=96,  H_q=4, H_kv=2 — batched
//
// Decode test cases:
//   5. batch_size=1, context_len=32,  H_q=2, H_kv=1
//   6. batch_size=2, context_len=128, H_q=16, H_kv=8

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h> 
#include <cuda_runtime.h>
#include <functional>
#include "../include/attention_unit.cuh"
using AttentionDecodeLaunchFunc = std::function<void(
    __nv_bfloat16* d_q,
    __nv_bfloat16* d_k,
    __nv_bfloat16* d_v,
    __nv_bfloat16* d_out,
    int batch_size,
    int context_len,
    int H_q,
    int H_kv,
    int D,
    int max_seq_len,
    float scale,
    cudaStream_t stream
)>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Deterministic LCG (no srand dependency)
static unsigned int lcg_state = 42u;
static float lcg_randf() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(lcg_state >> 1) / (float)0x7fffffffu) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// CPU reference: naive causal attention
// Q, K, V layout: [B, H, S, D]   (H = num_heads)
// O        layout: [B, H_q, S, D]
// FP32 throughout.
// ---------------------------------------------------------------------------
static void ref_attention(
    float*       O,
    const float* Q,
    const float* K,
    const float* V,
    int B, int S, int H_q, int H_kv, int D
) {
    // -----------------------------------------------------------------
    // Head‑major layout:
    //   Q, K, V : [B, H, S, D]   → index = ((b*H + h)*S + s)*D + d
    //   O      : [B, H_q, S, D]
    // -----------------------------------------------------------------
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S];

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H_q; ++h) {
            const int hkv = h * H_kv / H_q;   // map query‑head → kv‑head
            for (int i = 0; i < S; ++i) {
                // qi : [D]
                const float* qi = Q + ((b * H_q + h) * S + i) * D;

                // ---------------------------------------------------------
                // Compute dot‑products Q_i · K_j for j = 0 .. i
                // ---------------------------------------------------------
                float max_s = -1e30f;
                for (int j = 0; j <= i; ++j) {
                    const float* kj = K + ((b * H_kv + hkv) * S + j) * D;
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_s) max_s = scores[j];
                }

                // ---------------------------------------------------------
                // Softmax over the prefix [0..i]
                // ---------------------------------------------------------
                float sum_e = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_e += scores[j];
                }
                for (int j = 0; j <= i; ++j) scores[j] /= sum_e;

                // ---------------------------------------------------------
                // Weighted sum of V[0..i] → O_i
                // ---------------------------------------------------------
                float* oi = O + ((b * H_q + h) * S + i) * D;
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j <= i; ++j) {
                        const float* vj = V + ((b * H_kv + hkv) * S + j) * D;
                        acc += scores[j] * vj[d];
                    }
                    oi[d] = acc;
                }
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// CPU reference: non‑causal attention (for decode, S_q = 1)
// q layout:   [batch_size, H_q, D]
// K, V layout (new): [batch_size, H_kv, S_kv, D]  — head‑major KV context
// O layout:   [batch_size, H_q, D]
// ---------------------------------------------------------------------------
static void ref_decode_attention(
    float*       O,
    const float* q,
    const float* K,
    const float* V,
    int batch_size, int S_kv, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S_kv];   // temporary soft‑max scores for one query

    for (int s = 0; s < batch_size; s++) {
        for (int h = 0; h < H_q; h++) {
            // 将 Q‑head 映射到对应的 KV‑head（可能是多‑to‑one）
            const int hkv = h * H_kv / H_q;
            const float* qi = q + (s * H_q + h) * D;   // [D]

            // -------------------------------------------------------------
            // 1) 计算 Q·K，得到长度为 S_kv 的分数向量
            // -------------------------------------------------------------
            float max_s = -1e30f;
            for (int j = 0; j < S_kv; j++) {
                // K layout: [batch_size, H_kv, S_kv, D]
                // offset = ((s * H_kv + hkv) * S_kv + j) * D
                const float* kj = K + ((s * H_kv + hkv) * S_kv + j) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }

            // -------------------------------------------------------------
            // 2) Softmax (stable)
            // -------------------------------------------------------------
            float sum_e = 0.0f;
            for (int j = 0; j < S_kv; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_e += scores[j];
            }
            for (int j = 0; j < S_kv; j++) scores[j] /= sum_e;

            // -------------------------------------------------------------
            // 3) 加权求和得到输出向量 O_i
            // -------------------------------------------------------------
            float* oi = O + (s * H_q + h) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < S_kv; j++) {
                    // V layout: [batch_size, H_kv, S_kv, D]
                    const float* vj = V + ((s * H_kv + hkv) * S_kv + j) * D;
                    acc += scores[j] * vj[d];
                }
                oi[d] = acc;
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// Prefill test runner
// ---------------------------------------------------------------------------
static bool run_prefill_test(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, float tol = 5e-3f
) {
    const long N_q  = (long)B * S * H_q  * D;
    const long N_kv = (long)B * S * H_kv * D;

    // Host buffers (FP32 for reference, BF16 for kernel)
    float* h_Q_f32 = new float[N_q];
    float* h_K_f32 = new float[N_kv];
    float* h_V_f32 = new float[N_kv];
    float* h_ref   = new float[N_q];

    __nv_bfloat16* h_Q = new __nv_bfloat16[N_q];
    __nv_bfloat16* h_K = new __nv_bfloat16[N_kv];
    __nv_bfloat16* h_V = new __nv_bfloat16[N_kv];
    __nv_bfloat16* h_out = new __nv_bfloat16[N_q];

    // ------------------- 随机填充 Q -------------------
    for (long b = 0; b < B; ++b) {
        for (long h = 0; h < H_q; ++h) {
            for (long s = 0; s < S; ++s) {
                for (long d = 0; d < D; ++d) {
                    long idx = ((b * H_q + h) * S + s) * D + d;
                    h_Q[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                }
            }
        }
    }
    // ------------------- 随机填充 K / V -------------------
    for (long b = 0; b < B; ++b) {
        for (long h = 0; h < H_kv; ++h) {
            for (long s = 0; s < S; ++s) {
                for (long d = 0; d < D; ++d) {
                    long idx = ((b * H_kv + h) * S + s) * D + d;
                    h_K[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                    h_V[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                }
            }
        }
    }

    // ------------------- FP32 reference 使用量化后的 BF16 -------------------
    for (long i = 0; i < N_q;  ++i) h_Q_f32[i] = __bfloat162float(h_Q[i]);
    for (long i = 0; i < N_kv; ++i) h_K_f32[i] = __bfloat162float(h_K[i]);
    for (long i = 0; i < N_kv; ++i) h_V_f32[i] = __bfloat162float(h_V[i]);

    // 参考实现（仍然是 FP32 计算）
    ref_attention(h_ref, h_Q_f32, h_K_f32, h_V_f32, B, S, H_q, H_kv, D);

    // ------------------- Device buffers -------------------
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // launch wrapper
    launch_flash_attention_prefill(d_Q, d_K, d_V, d_O,
                                 B, S, H_q, H_kv, D,
                                 0 /*default stream*/, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_O, N_q * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // ------------------- 误差检查 -------------------
    float max_err = 0.0f;
    for (long i = 0; i < N_q; i++) {
        float diff = fabsf(__bfloat162float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    // ------------------- 清理 -------------------
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_Q;    delete[] h_K;    delete[] h_V;    delete[] h_out;

    return passed;
}
// ---------------------------------------------------------------------------
// Split‑K decode test runner (now using BF16 kernel) – KV layout is head‑major
// ---------------------------------------------------------------------------
static bool run_decode_test(
    const char* name,
    int batch_size, int context_len, int H_q, int H_kv,
    int /*num_splits*/,               // kept for API compatibility – unused
    AttentionDecodeLaunchFunc launch_func = launch_attention_decode,
    int D = 128, int /*BLOCK_SIZE*/ = 16, float tol = 5e-3f
) {
    // -----------------------------------------------------------------------
    // 1) Allocate host buffers (FP32 reference + BF16 device inputs/outputs)
    //    KV layout: [batch_size, H_kv, context_len, D]  (head‑major)
    // -----------------------------------------------------------------------
    const long Q_total = (long)batch_size * H_q * D;
    const long KV_total = (long)batch_size * H_kv * context_len * D; // note the order
    const long O_total = Q_total;

    // FP32 buffers for reference
    float* h_q_f32 = new float[Q_total];
    float* h_K_f32 = new float[KV_total];
    float* h_V_f32 = new float[KV_total];
    float* h_ref   = new float[O_total];

    // BF16 buffers for device
    __nv_bfloat16* h_q_bf   = new __nv_bfloat16[Q_total];
    __nv_bfloat16* h_k_bf   = new __nv_bfloat16[KV_total];
    __nv_bfloat16* h_v_bf   = new __nv_bfloat16[KV_total];
    __nv_bfloat16* h_out_bf = new __nv_bfloat16[O_total];

    // -----------------------------------------------------------------------
    // 2) Fill host FP32 data with deterministic random numbers
    // -----------------------------------------------------------------------
    for (long i = 0; i < Q_total; i++) h_q_f32[i] = lcg_randf() * 0.5f;
    for (long i = 0; i < KV_total; i++) {
        h_K_f32[i] = lcg_randf() * 0.5f;
        h_V_f32[i] = lcg_randf() * 0.5f;
    }

    // -----------------------------------------------------------------------
    // 3) Compute FP32 reference output (uses the head‑major layout)
    // -----------------------------------------------------------------------
    ref_decode_attention(h_ref, h_q_f32, h_K_f32, h_V_f32,
                         batch_size, context_len, H_q, H_kv, D);

    // -----------------------------------------------------------------------
    // 4) Convert FP32 inputs to BF16 (device format)
    // -----------------------------------------------------------------------
    for (long i = 0; i < Q_total; i++) h_q_bf[i] = __float2bfloat16(h_q_f32[i]);
    for (long i = 0; i < KV_total; i++) {
        h_k_bf[i] = __float2bfloat16(h_K_f32[i]);
        h_v_bf[i] = __float2bfloat16(h_V_f32[i]);
    }

    // -----------------------------------------------------------------------
    // 5) Allocate device memory and copy inputs
    // -----------------------------------------------------------------------
    __nv_bfloat16 *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q,   Q_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k,   KV_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v,   KV_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, O_total * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_q,   h_q_bf, Q_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k,   h_k_bf, KV_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,   h_v_bf, KV_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // 6) Launch BF16 decode kernel (kernel expects head‑major KV layout)
    // -----------------------------------------------------------------------
    const float scale = 1.0f / sqrtf((float)D);
    const int max_seq_len = context_len;   // decode 时 cache_len == max_seq_len

    launch_func(
        d_q, d_k, d_v, d_out,
        batch_size,          // batch_size
        context_len,       // cache_len (same as context_len)
        H_q,               // n_q_heads
        H_kv,              // n_kv_heads
        D,                 // head_dim
        max_seq_len,       // max_seq_len
        scale,
        0                  // default stream
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // 7) Copy result back and compute max absolute error
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(h_out_bf, d_out,
                          O_total * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (long i = 0; i < O_total; i++) {
        float diff = fabsf(__bfloat162float(h_out_bf[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-70s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS tolerance");

    // -----------------------------------------------------------------------
    // 8) Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    delete[] h_q_f32;
    delete[] h_K_f32;
    delete[] h_V_f32;
    delete[] h_ref;
    delete[] h_q_bf;
    delete[] h_k_bf;
    delete[] h_v_bf;
    delete[] h_out_bf;

    return passed;
}

// ---------------------------------------------------------------------------
// Prefill benchmark runner (BF16 version)
// ---------------------------------------------------------------------------
static void run_prefill_benchmark(
    const char* name, int B, int S, int H_q, int H_kv,
    int D = 128, int warmup = 10, int iters = 10
) {
    // --------------------------------------------------------------
    // 1. 计算张量大小
    // --------------------------------------------------------------
    const long N_q  = (long)B * S * H_q  * D;   // Q / O 元素数
    const long N_kv = (long)B * S * H_kv * D;   // K / V 元素数

    // --------------------------------------------------------------
    // 2. 主机端 BF16 缓冲区并随机填充（与 run_prefill_test 保持一致）
    // --------------------------------------------------------------
    __nv_bfloat16 *h_Q = new __nv_bfloat16[N_q];
    __nv_bfloat16 *h_K = new __nv_bfloat16[N_kv];
    __nv_bfloat16 *h_V = new __nv_bfloat16[N_kv];

    // 随机填充 Q
    for (long b = 0; b < B; ++b) {
        for (long h = 0; h < H_q; ++h) {
            for (long s = 0; s < S; ++s) {
                for (long d = 0; d < D; ++d) {
                    long idx = ((b * H_q + h) * S + s) * D + d;
                    h_Q[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                }
            }
        }
    }
    // 随机填充 K / V
    for (long b = 0; b < B; ++b) {
        for (long h = 0; h < H_kv; ++h) {
            for (long s = 0; s < S; ++s) {
                for (long d = 0; d < D; ++d) {
                    long idx = ((b * H_kv + h) * S + s) * D + d;
                    h_K[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                    h_V[idx] = __float2bfloat16(lcg_randf() * 0.5f);
                }
            }
        }
    }

    // --------------------------------------------------------------
    // 3. 设备端 BF16 缓冲区
    // --------------------------------------------------------------
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, N_q  * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, N_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, N_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_O, N_q  * sizeof(__nv_bfloat16)));

    // 将随机数据拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, N_q  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, N_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // --------------------------------------------------------------
    // 4. 计时事件准备
    // --------------------------------------------------------------
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // --------------------------------------------------------------
    // 5. Warm‑up（确保 JIT、缓存等已就绪）
    // --------------------------------------------------------------
    for (int i = 0; i < warmup; ++i) {
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O,
                                      B, S, H_q, H_kv, D,
                                      0 /*default stream*/, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------------
    // 6. 正式计时
    // --------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        launch_flash_attention_prefill(d_Q, d_K, d_V, d_O,
                                      B, S, H_q, H_kv, D,
                                      0 /*default stream*/, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    // --------------------------------------------------------------
    // 7. 计算平均耗时（us）
    // --------------------------------------------------------------
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;   // µs / iteration

    // --------------------------------------------------------------
    // 8. FLOPs 与 TFLOPs 统计
    //    QKᵀ  : 2 * B * H_q * S² * D
    //    Softmax·V : 2 * B * H_q * S² * D
    //    合计 4 * B * H_q * S² * D
    // --------------------------------------------------------------
    const double flops   = 4.0 * B * H_q * (double)S * S * D;
    const double tflops  = flops / (us * 1e-6) / 1e12;   // TFLOP/s

    // --------------------------------------------------------------
    // 9. 带宽统计（仅统计设备端读写：Q、K、V 读取一次，O 写入一次）
    // --------------------------------------------------------------
    const double bytes   = (2.0 * N_q + 2.0 * N_kv) * sizeof(__nv_bfloat16);
    const double gb_per_s = bytes / (us * 1e-6) / 1e9;   // GB/s

    // --------------------------------------------------------------
    // 10. 输出结果
    // --------------------------------------------------------------
    printf("[BENCH] %-50s  %7.2f us  %5.2f TFLOPS  %5.2f GB/s\n",
           name, us, tflops, gb_per_s);

    // --------------------------------------------------------------
    // 11. 资源释放
    // --------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    delete[] h_Q; delete[] h_K; delete[] h_V;
}

// ---------------------------------------------------------------------------
// Split-K decode benchmark
// ---------------------------------------------------------------------------
static void run_attention_decode_benchmark(
    const char* name,
    int batch_size, int context_len, int H_q, int H_kv,
    AttentionDecodeLaunchFunc launch_func = launch_attention_decode,
    int D = 128,
    int warmup = 10, int iters = 10
) {
    // -----------------------------------------------------------------------
    // 1) Host buffers (FP32 reference not needed for benchmark)
    // -----------------------------------------------------------------------
    const long Q_total = (long)batch_size * H_q * D;                 // [B, H_q, D]
    const long KV_total = (long)batch_size * context_len * H_kv * D; // [B, S, H_kv, D]
    const long O_total = Q_total;                                 // same shape as Q

    // BF16 host buffers (filled with deterministic random data)
    __nv_bfloat16 *h_q   = new __nv_bfloat16[Q_total];
    __nv_bfloat16 *h_k   = new __nv_bfloat16[KV_total];
    __nv_bfloat16 *h_v   = new __nv_bfloat16[KV_total];
    __nv_bfloat16 *h_out = new __nv_bfloat16[O_total]; // output buffer (unused on host)

    for (long i = 0; i < Q_total; i++) h_q[i] = __float2bfloat16(lcg_randf() * 0.5f);
    for (long i = 0; i < KV_total; i++) {
        h_k[i] = __float2bfloat16(lcg_randf() * 0.5f);
        h_v[i] = __float2bfloat16(lcg_randf() * 0.5f);
    }

    // -----------------------------------------------------------------------
    // 2) Device buffers
    // -----------------------------------------------------------------------
    __nv_bfloat16 *d_q, *d_k, *d_v, *d_out;
    CUDA_CHECK(cudaMalloc(&d_q,   Q_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k,   KV_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v,   KV_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, O_total * sizeof(__nv_bfloat16)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_q,   h_q,   Q_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k,   h_k,   KV_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,   h_v,   KV_total * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // 3) Benchmark loop (warm‑up + timed runs)
    // -----------------------------------------------------------------------
    const float scale = 1.0f / sqrtf((float)D);
    const int max_seq_len = context_len;   // for decode the max seq length equals cache_len

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Warm‑up
    for (int i = 0; i < warmup; ++i) {
        launch_func(
            d_q, d_k, d_v, d_out,
            batch_size,          // batch_size
            context_len,       // cache_len
            H_q,               // n_q_heads
            H_kv,              // n_kv_heads
            D,                 // head_dim
            max_seq_len,       // max_seq_len
            scale,
            0                  // default stream
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        launch_func(
            d_q, d_k, d_v, d_out,
            batch_size,
            context_len,
            H_q,
            H_kv,
            D,
            max_seq_len,
            scale,
            0
        );
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;   // microseconds per iteration

    // -----------------------------------------------------------------------
    // 4) Throughput calculation (GB/s)
    // -----------------------------------------------------------------------
    const double bytes_kv = 2.0 * (double)batch_size * H_kv * context_len * D *
                           sizeof(__nv_bfloat16);   // K + V
    const double bytes_qo = 2.0 * (double)batch_size * H_q * D *
                           sizeof(__nv_bfloat16);   // Q + O
    const double gbs = (bytes_kv + bytes_qo) / (us * 1e-6) / 1e9;

    printf("[BENCH] %-55s  %7.2f us  %6.1f GB/s\n", name, us, gbs);

    // -----------------------------------------------------------------------
    // 5) Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);

    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_out;
}

// ---------------------------------------------------------------------------
// CPU reference: non-causal attention (for decode, S_q=1)
// q layout: [num_seqs, H_q, D]
// New layout: K, V layout = [num_seqs, H_kv, S_kv, D]  (head‑major)
// O layout: [num_seqs, H_q, D]
// ---------------------------------------------------------------------------
static void ref_decode_attention_head_major(
    float*       O,
    const float* q,
    const float* K,
    const float* V,
    int num_seqs, int S_kv, int H_q, int H_kv, int D
) {
    const float scale = 1.0f / sqrtf((float)D);
    float* scores = new float[S_kv];

    for (int s = 0; s < num_seqs; s++) {
        for (int h = 0; h < H_q; h++) {
            // map query head -> KV head (GQA support)
            const int hkv = h * H_kv / H_q;
            const float* qi = q + (s * H_q + h) * D;

            // ---------- compute scaled dot‑product scores ----------
            float max_s = -1e30f;
            for (int j = 0; j < S_kv; j++) {
                // head‑major offset: ((s * H_kv + hkv) * S_kv + j) * D
                const float* kj = K + ((s * H_kv + hkv) * S_kv + j) * D;
                float dot = 0.0f;
                for (int d = 0; d < D; d++) dot += qi[d] * kj[d];
                scores[j] = dot * scale;
                if (scores[j] > max_s) max_s = scores[j];
            }

            // ---------- softmax ----------
            float sum_e = 0.0f;
            for (int j = 0; j < S_kv; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_e += scores[j];
            }
            for (int j = 0; j < S_kv; j++) scores[j] /= sum_e;

            // ---------- weighted sum of V ----------
            float* oi = O + (s * H_q + h) * D;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int j = 0; j < S_kv; j++) {
                    // head‑major offset for V
                    const float* vj = V + ((s * H_kv + hkv) * S_kv + j) * D;
                    acc += scores[j] * vj[d];
                }
                oi[d] = acc;
            }
        }
    }

    delete[] scores;
}

// ---------------------------------------------------------------------------
// Split‑K decode test runner (correctness vs CPU reference)
// ---------------------------------------------------------------------------
static bool run_splitk_decode_test_head_major(
    const char* name,
    int num_seqs, int context_len, int H_q, int H_kv,
    int num_splits,
    int D = 128, int BLOCK_SIZE = 16, float tol = 5e-3f
) {
    const long Q_total = (long)num_seqs * H_q * D;
    const long O_total = Q_total;

    float* h_q_f32 = new float[Q_total];
    float* h_K_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_V_f32 = new float[(long)num_seqs * context_len * H_kv * D];
    float* h_ref   = new float[O_total];

    // BF16 buffers (host side)
    __nv_bfloat16* h_q   = new __nv_bfloat16[Q_total];
    __nv_bfloat16* h_out = new __nv_bfloat16[O_total];

    // -----------------------------------------------------------------------
    // Random data generation (layout‑agnostic – we just fill the linear buffer)
    // -----------------------------------------------------------------------
    for (long i = 0; i < Q_total; i++) h_q[i] = __float2bfloat16(lcg_randf() * 0.5f);
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = lcg_randf() * 0.5f;
        h_V_f32[i] = lcg_randf() * 0.5f;
    }

    // Convert Q to FP32 for the reference
    for (long i = 0; i < Q_total; i++) h_q_f32[i] = __bfloat162float(h_q[i]);
    // Cast K/V to BF16 → FP32 to guarantee the same rounding as the kernel
    for (long i = 0; i < (long)num_seqs * context_len * H_kv * D; i++) {
        h_K_f32[i] = __bfloat162float(__float2bfloat16(h_K_f32[i]));
        h_V_f32[i] = __bfloat162float(__float2bfloat16(h_V_f32[i]));
    }

    // -----------------------------------------------------------------------
    // Run the **head‑major** reference implementation
    // -----------------------------------------------------------------------
    ref_decode_attention_head_major(
        h_ref, h_q_f32, h_K_f32, h_V_f32,
        num_seqs, context_len, H_q, H_kv, D
    );

    // -----------------------------------------------------------------------
    // KV‑cache init (head‑major per‑block layout expected by the kernel)
    // -----------------------------------------------------------------------
    const int blocks_per_seq    = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int total_phys_blocks = num_seqs * blocks_per_seq;
    const long cache_elems      = (long)total_phys_blocks * H_kv * BLOCK_SIZE * D;

    __nv_bfloat16* h_k_cache = new __nv_bfloat16[cache_elems]();   // zero‑init
    __nv_bfloat16* h_v_cache = new __nv_bfloat16[cache_elems]();

    // Copy from the **head‑major** K/V tensors into the block‑wise cache
    for (int s = 0; s < num_seqs; s++) {
        for (int t = 0; t < context_len; t++) {
            for (int h = 0; h < H_kv; h++) {
                int logical_block = t / BLOCK_SIZE;
                int tok_offset    = t % BLOCK_SIZE;
                int phys_block    = s * blocks_per_seq + logical_block;

                // ----- head‑major source offsets -----
                const float* ksrc = h_K_f32 + ((s * H_kv + h) * context_len + t) * D;
                const float* vsrc = h_V_f32 + ((s * H_kv + h) * context_len + t) * D;

                // ----- destination offsets inside the cache -----
                __nv_bfloat16* kdst = h_k_cache + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D + tok_offset * D;
                __nv_bfloat16* vdst = h_v_cache + ((long)phys_block * H_kv + h) * BLOCK_SIZE * D + tok_offset * D;

                for (int d = 0; d < D; d++) {
                    kdst[d] = __float2bfloat16(ksrc[d]);
                    vdst[d] = __float2bfloat16(vsrc[d]);
                }
            }
        }
    }

    // logical‑to‑physical block mapping (identity mapping for this test)
    int* h_block_tables = new int[num_seqs * blocks_per_seq];
    int* h_seq_lens     = new int[num_seqs];
    for (int s = 0; s < num_seqs; s++) {
        h_seq_lens[s] = context_len;
        for (int b = 0; b < blocks_per_seq; b++)
            h_block_tables[s * blocks_per_seq + b] = s * blocks_per_seq + b;
    }

    // -----------------------------------------------------------------------
    // Device allocations & kernel launch
    // -----------------------------------------------------------------------
    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out;
    int  *d_block_tables, *d_seq_lens;
    float *d_partial_out, *d_partial_max, *d_partial_sum;

    const long ws_out = (long)num_seqs * H_q * num_splits * D;
    const long ws_ms  = (long)num_seqs * H_q * num_splits;

    CUDA_CHECK(cudaMalloc(&d_q,           Q_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_cache,     cache_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v_cache,     cache_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out,         O_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, num_seqs * blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    num_seqs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_out, ws_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, ws_ms  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, ws_ms  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q,            h_q,            Q_total * sizeof(__nv_bfloat16),                     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache,      h_k_cache,      cache_elems * sizeof(__nv_bfloat16),                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache,      h_v_cache,      cache_elems * sizeof(__nv_bfloat16),                 cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables,  num_seqs * blocks_per_seq * sizeof(int),            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens,     h_seq_lens,      num_seqs * sizeof(int),                             cudaMemcpyHostToDevice));

    // NOTE: launch function now expects a stream argument; we use the default stream (0)
    launch_flash_decode_splitk_seq_major(
        d_q, d_k_cache, d_v_cache, d_out,
        d_partial_out, d_partial_max, d_partial_sum,
        d_block_tables, d_seq_lens,
        num_seqs, H_q, H_kv, D,
        blocks_per_seq, BLOCK_SIZE, num_splits,
        0   // cudaStreamDefault
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, O_total * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Verify results
    // -----------------------------------------------------------------------
    float max_err = 0.0f;
    for (long i = 0; i < O_total; i++) {
        float diff = fabsf(__bfloat162float(h_out[i]) - h_ref[i]);
        if (diff > max_err) max_err = diff;
    }

    bool passed = (max_err < tol);
    printf("[%s] %-50s max_err=%.6f  %s\n",
           passed ? "PASS" : "FAIL", name, max_err,
           passed ? "" : "<-- EXCEEDS 5e-3");

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_out); cudaFree(d_block_tables); cudaFree(d_seq_lens);
    cudaFree(d_partial_out); cudaFree(d_partial_max); cudaFree(d_partial_sum);

    delete[] h_q_f32; delete[] h_K_f32; delete[] h_V_f32; delete[] h_ref;
    delete[] h_q; delete[] h_out;
    delete[] h_k_cache; delete[] h_v_cache;
    delete[] h_block_tables; delete[] h_seq_lens;

    return passed;
}

static void run_flash_decode_splitk_benchmark(
    const char* name,
    int batch_size, int context_len, int H_q, int H_kv,
    int num_splits,
    int D = 128,
    int BLOCK_SIZE = 16,
    int warmup = 10,
    int iters = 10
) {
    // -----------------------------------------------------------------------
    // 1) Host buffers (BF16)
    // -----------------------------------------------------------------------
    const long Q_total = (long)batch_size * H_q * D;                     // [B, H_q, D]
    const long KV_total = (long)batch_size * context_len * H_kv * D;    // [B, H_kv, S, D]
    const long O_total = Q_total;                                        // same shape as Q

    __nv_bfloat16 *h_q   = new __nv_bfloat16[Q_total];
    __nv_bfloat16 *h_k   = new __nv_bfloat16[KV_total];
    __nv_bfloat16 *h_v   = new __nv_bfloat16[KV_total];
    __nv_bfloat16 *h_out = new __nv_bfloat16[O_total];   // device‑output copy‑back buffer

    // Fill Q, K, V with deterministic random BF16 data
    for (long i = 0; i < Q_total; ++i) h_q[i] = __float2bfloat16(lcg_randf() * 0.5f);
    for (long i = 0; i < KV_total; ++i) {
        h_k[i] = __float2bfloat16(lcg_randf() * 0.5f);
        h_v[i] = __float2bfloat16(lcg_randf() * 0.5f);
    }

    // -----------------------------------------------------------------------
    // 2) KV‑cache layout (block‑major) expected by the kernel
    // -----------------------------------------------------------------------
    const int blocks_per_seq = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const long total_phys_blocks = (long)batch_size * blocks_per_seq;
    const long cache_elems = total_phys_blocks * H_kv * BLOCK_SIZE * D; // may be larger than KV_total

    __nv_bfloat16 *h_k_cache = new __nv_bfloat16[cache_elems]();   // zero‑init
    __nv_bfloat16 *h_v_cache = new __nv_bfloat16[cache_elems]();

    // Copy head‑major K/V into the block‑major cache
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < context_len; ++t) {
            int logical_block = t / BLOCK_SIZE;
            int token_offset  = t % BLOCK_SIZE;
            long phys_block   = (long)b * blocks_per_seq + logical_block;

            for (int h = 0; h < H_kv; ++h) {
                // source offsets (head‑major)
                const __nv_bfloat16* ksrc = h_k + ((long)(b * H_kv + h) * context_len + t) * D;
                const __nv_bfloat16* vsrc = h_v + ((long)(b * H_kv + h) * context_len + t) * D;

                // destination offsets (block‑major)
                __nv_bfloat16* kdst = h_k_cache + ((phys_block * H_kv + h) * BLOCK_SIZE + token_offset) * D;
                __nv_bfloat16* vdst = h_v_cache + ((phys_block * H_kv + h) * BLOCK_SIZE + token_offset) * D;

                for (int d = 0; d < D; ++d) {
                    kdst[d] = ksrc[d];
                    vdst[d] = vsrc[d];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3) Block‑table & sequence‑length arrays (identity mapping)
    // -----------------------------------------------------------------------
    int *h_block_tables = new int[batch_size * blocks_per_seq];
    int *h_seq_lens     = new int[batch_size];
    for (int b = 0; b < batch_size; ++b) {
        h_seq_lens[b] = context_len;
        for (int blk = 0; blk < blocks_per_seq; ++blk) {
            h_block_tables[b * blocks_per_seq + blk] = b * blocks_per_seq + blk;
        }
    }

    // -----------------------------------------------------------------------
    // 4) Device allocations
    // -----------------------------------------------------------------------
    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out;
    int  *d_block_tables, *d_seq_lens;
    float *d_partial_out, *d_partial_max, *d_partial_sum;

    const long ws_out = (long)batch_size * H_q * num_splits * D; // partial output buffer
    const long ws_ms  = (long)batch_size * H_q * num_splits;    // partial max / sum buffers

    CUDA_CHECK(cudaMalloc(&d_q,           Q_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_cache,     cache_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v_cache,     cache_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out,         O_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_block_tables, batch_size * blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_out, ws_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, ws_ms  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, ws_ms  * sizeof(float)));

    // -----------------------------------------------------------------------
    // 5) Copy host data to device
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_q,          h_q,          Q_total * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache,    h_k_cache,    cache_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache,    h_v_cache,    cache_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_tables, h_block_tables, batch_size * blocks_per_seq * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens,   h_seq_lens,   batch_size * sizeof(int), cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // 6) Warm‑up + timed runs
    // -----------------------------------------------------------------------
    // Warm‑up
    for (int i = 0; i < warmup; ++i) {
        launch_flash_decode_splitk_seq_major(
            d_q, d_k_cache, d_v_cache, d_out,
            d_partial_out, d_partial_max, d_partial_sum,
            d_block_tables, d_seq_lens,
            batch_size, H_q, H_kv, D,
            blocks_per_seq, BLOCK_SIZE, num_splits,
            0   // default stream
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) {
        launch_flash_decode_splitk_seq_major(
            d_q, d_k_cache, d_v_cache, d_out,
            d_partial_out, d_partial_max, d_partial_sum,
            d_block_tables, d_seq_lens,
            batch_size, H_q, H_kv, D,
            blocks_per_seq, BLOCK_SIZE, num_splits,
            0
        );
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    const float us = ms * 1000.0f / iters;   // µs per iteration

    // -----------------------------------------------------------------------
    // 7) Throughput (GB/s)
    // -----------------------------------------------------------------------
    const double bytes_kv = 2.0 * (double)batch_size * H_kv * context_len * D *
                            sizeof(__nv_bfloat16);   // K + V
    const double bytes_qo = 2.0 * (double)batch_size * H_q * D *
                            sizeof(__nv_bfloat16);   // Q + O
    const double gbs = (bytes_kv + bytes_qo) / (us * 1e-6) / 1e9;

    printf("[BENCH] %-55s  %7.2f us  %6.1f GB/s\n", name, us, gbs);

    // -----------------------------------------------------------------------
    // 8) Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_out);
    cudaFree(d_block_tables);
    cudaFree(d_seq_lens);
    cudaFree(d_partial_out);
    cudaFree(d_partial_max);
    cudaFree(d_partial_sum);

    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_out;
    delete[] h_k_cache;
    delete[] h_v_cache;
    delete[] h_block_tables;
    delete[] h_seq_lens;
}

// ---------------------------------------------------------------------------
// main – only a single decode test is kept for brevity (add more as needed)
// ---------------------------------------------------------------------------
int main() {
    printf("=== Flash Attention kernel tests ===\n\n");

    // ── Prefill tests ────────────────────────────────────────────────────────
    printf("--- Prefill (FA2) ---\n");
    bool all_pass = true;

    all_pass &= run_prefill_test(
        "Prefill B=1 S=16  H_q=2  H_kv=1 (small GQA 2:1)",
        1, 16, 2, 1);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=64  H_q=4  H_kv=2 (single KV tile)",
        1, 64, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=128 H_q=16 H_kv=8 (Qwen3 heads, 2 KV tiles)",
        2, 128, 16, 8);

    all_pass &= run_prefill_test(
        "Prefill B=2 S=96  H_q=4  H_kv=2 (batched)",
        2, 96, 4, 2);

    all_pass &= run_prefill_test(
        "Prefill B=1 S=256 H_q=16 H_kv=8 (4 KV tiles)",
        1, 256, 16, 8);

    // ── Split-K decode tests ──────────────────────────────────────────────────
    printf("\n--- Decode (Flash Decoding Split-K) ---\n");

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=1 ctx=32  H_q=2  H_kv=1  splits=2",
        1, 32, 2, 1, 2);

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=1 ctx=128 H_q=16 H_kv=8  splits=4",
        1, 128, 16, 8, 4);

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=2 ctx=64  H_q=4  H_kv=2  splits=4",
        2, 64, 4, 2, 4);

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=4 ctx=128 H_q=16 H_kv=8  splits=8",
        4, 128, 16, 8, 8);

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=1 ctx=2048 H_q=16 H_kv=8 splits=16",
        1, 2048, 16, 8, 16);

    all_pass &= run_splitk_decode_test_head_major(
        "SplitK num_seqs=16 ctx=2048 H_q=16 H_kv=8 splits=16",
        16, 2048, 16, 8, 16);

    printf("\n--- Decode (Custom 1 kernel) ---\n");
    AttentionDecodeLaunchFunc launch_func = launch_attention_decode;// launch_attention_decode_splitk;
    
    all_pass &= run_decode_test(
        "BF16 Decode batch_size=1 ctx=32  H_q=2  H_kv=1  splits=1",
        1, 32, 2, 1, 1, launch_func
    );

    all_pass &= run_decode_test(
    "BF16 SplitK batch_size=1 ctx=128  H_q=16 H_kv=8  splits=4",
        1, 128, 16, 8, 1, launch_func);

    all_pass &= run_decode_test(
        "BF16 SplitK batch_size=2 ctx=64   H_q=4  H_kv=2  splits=4",
        2, 64, 4, 2, 4, launch_func);

    all_pass &= run_decode_test(
        "BF16 SplitK batch_size=4 ctx=128  H_q=16 H_kv=8  splits=8",
        4, 128, 16, 8, 8, launch_func);

    all_pass &= run_decode_test(
        "BF16 SplitK batch_size=1 ctx=2048 H_q=16 H_kv=8 splits=16",
        1, 2048, 16, 8, 16, launch_func);
        
    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n%s\n", all_pass ? "All tests PASSED." : "Some tests FAILED.");

    // // ── Benchmarks ───────────────────────────────────────────────────────────
    printf("\n=== Prefill benchmarks (warmup=10, iters=10) ===\n");
    run_prefill_benchmark("Prefill B=1 S=2048 H_q=16 H_kv=8",  1, 2048, 16, 8);// draft
    run_prefill_benchmark("Prefill B=8 S=2048 H_q=16 H_kv=8",  8, 2048, 16, 8);
    run_prefill_benchmark("Prefill B=8 S=4096 H_q=16 H_kv=8",  8, 4096, 16, 8);
    run_prefill_benchmark("Prefill B=8 S=8192 H_q=16 H_kv=8",  8, 8192, 16, 8);
    // run_prefill_benchmark("Prefill B=8 S=16384 H_q=16 H_kv=8",  8, 16384, 16, 8);
    // run_prefill_benchmark("Prefill B=8 S=32768 H_q=16 H_kv=8",  8, 32768, 16, 8);

    printf("\n=== Decode benchmarks — Decoding-v2 (warmup=10, iters=10) ===\n");// 待解读v1，v2源码
    run_attention_decode_benchmark("SplitK  B=1   ctx=512   H_q=16 H_kv=8  S=8",   1,   512, 16, 8, launch_func, 8);
    run_attention_decode_benchmark("SplitK  B=1   ctx=2048  H_q=16 H_kv=8  S=16",  1,  2048, 16, 8, launch_func, 16);// draft
    run_attention_decode_benchmark("SplitK  B=16  ctx=512   H_q=16 H_kv=8  S=8",  16,   512, 16, 8, launch_func, 8);
    run_attention_decode_benchmark("SplitK  B=16  ctx=2048  H_q=16 H_kv=8  S=16", 16,  2048, 16, 8, launch_func, 16);
    run_attention_decode_benchmark("SplitK  B=64  ctx=512   H_q=16 H_kv=8  S=8",  64,   512, 16, 8, launch_func, 8);
    run_attention_decode_benchmark("SplitK  B=64  ctx=2048  H_q=16 H_kv=8  S=16", 64,  2048, 16, 8, launch_func, 16);
    run_attention_decode_benchmark("SplitK  B=128 ctx=2048  H_q=16 H_kv=8  S=16",128,  2048, 16, 8, launch_func, 16);

    printf("\n=== Decode benchmarks — Flash Decoding Split-K (warmup=10, iters=10) ===\n");
    run_flash_decode_splitk_benchmark("SplitK  B=1   ctx=512   H_q=16 H_kv=8  S=8",   1,   512, 16, 8, 8);// 性能最优
    run_flash_decode_splitk_benchmark("SplitK  B=1   ctx=2048  H_q=16 H_kv=8  S=16",  1,  2048, 16, 8, 16);// 性能最优
    run_flash_decode_splitk_benchmark("SplitK  B=16  ctx=512   H_q=16 H_kv=8  S=8",  16,   512, 16, 8, 8);
    run_flash_decode_splitk_benchmark("SplitK  B=16  ctx=2048  H_q=16 H_kv=8  S=16", 16,  2048, 16, 8, 16);
    run_flash_decode_splitk_benchmark("SplitK  B=64  ctx=512   H_q=16 H_kv=8  S=8",  64,   512, 16, 8, 8);
    run_flash_decode_splitk_benchmark("SplitK  B=64  ctx=2048  H_q=16 H_kv=8  S=16", 64,  2048, 16, 8, 16);
    run_flash_decode_splitk_benchmark("SplitK  B=128 ctx=2048  H_q=16 H_kv=8  S=16",128,  2048, 16, 8, 16);

    return all_pass ? 0 : 1;
}