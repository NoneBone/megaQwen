#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

void launch_flash_attention_prefill(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16*       O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream,
    float*       lse    // [B, S, H_q] logsumexp — optional, for backward
);

// ---------------------------------------------------------------------------
// Declaration of the BF16 decode kernel wrapper (defined in attention_decode.cu)
// ---------------------------------------------------------------------------
extern "C" void launch_attention_decode(
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

void launch_flash_decode_splitk_seq_major(
    const __nv_bfloat16*  q,               
    const __nv_bfloat16*  k_cache,         
    const __nv_bfloat16*  v_cache,         
    __nv_bfloat16*        out,              
    float*                partial_out,
    float*                partial_max,
    float*                partial_sum,
    const int*            block_tables,
    const int*            seq_lens,
    int num_seqs, int H_q, int H_kv, int head_dim,
    int max_blocks_per_seq, int block_size,
    int num_splits,
    cudaStream_t stream
);