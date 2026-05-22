#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
template<int HEAD_DIM, int Br, int Bc, int NUM_WARPS>
__global__ void flash_attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16*       __restrict__ O,
    float*              __restrict__ LSE,
    int B,
    int S,
    int H_q,
    int H_kv,
    float scale
);

extern "C" void launch_flash_attention_prefill(
    const void* Q,
    const void* K,
    const void* V,
    void*       O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream,
    float*       lse    // [B, S, H_q] logsumexp — optional, for backward
);

// ---------------------------------------------------------------------------
// Declaration of the BF16 decode kernel wrapper (defined in attention_decode.cu)
// ---------------------------------------------------------------------------
extern "C" void launch_attention_decode_v1(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    cudaStream_t stream
);

extern "C" void launch_attention_decode_v2(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    cudaStream_t stream
);

extern "C" void launch_attention_decode_v3(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    cudaStream_t stream
);

extern "C" void launch_attention_decode_v4(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale,
    cudaStream_t stream
);

