/**
 * Attention decode kernel for single-token inference.
 *
 * Computes attention for a single query token against the full KV cache.
 * This is optimized for the decode phase where seq_len_q = 1.
 *
 * out = softmax(Q @ K^T / sqrt(head_dim)) @ V
 *
 * Supports GQA (grouped query attention) where n_q_heads > n_kv_heads.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

#define ATTN_WARP_SIZE 32

// Warp reduce max (for attention decode)
__device__ __forceinline__ float attn_warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// Warp reduce sum (for attention decode)
__device__ __forceinline__ float attn_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce max using shared memory - broadcasts result to all threads
__device__ __forceinline__ float attn_block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % ATTN_WARP_SIZE;
    int wid = threadIdx.x / ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    // Reduce within warp
    val = attn_warp_reduce_max(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Warp 0 reduces across warps
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : -INFINITY;
        val = attn_warp_reduce_max(val);
        // Write final result to shared[0]
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    // All threads read the result
    return shared[0];
}

// Block reduce sum using shared memory - broadcasts result to all threads
__device__ __forceinline__ float attn_block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % ATTN_WARP_SIZE;
    int wid = threadIdx.x / ATTN_WARP_SIZE;
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    // Reduce within warp
    val = attn_warp_reduce_sum(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Warp 0 reduces across warps
    if (wid == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = attn_warp_reduce_sum(val);
        // Write final result to shared[0]
        if (lane == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();

    // All threads read the result
    return shared[0];
}

/**
 * Attention decode kernel - one block per query head
 *
 * @param q Query tensor (batch, n_q_heads, 1, head_dim) in bf16
 * @param k_cache K cache (batch, n_kv_heads, max_seq_len, head_dim) in bf16
 * @param v_cache V cache (batch, n_kv_heads, max_seq_len, head_dim) in bf16
 * @param out Output tensor (batch, n_q_heads, 1, head_dim) in bf16
 * @param cache_len Number of valid tokens in cache (including current)
 * @param n_q_heads Number of query heads
 * @param n_kv_heads Number of KV heads
 * @param head_dim Head dimension
 * @param max_seq_len Maximum sequence length in cache
 * @param scale Attention scale (1 / sqrt(head_dim))
 */
__global__ void attention_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    extern __shared__ float shared[];

    int batch_idx = blockIdx.y;
    int q_head_idx = blockIdx.x;

    // GQA: map query head to kv head
    int n_groups = n_q_heads / n_kv_heads;
    int kv_head_idx = q_head_idx / n_groups;

    // Pointers for this head
    const __nv_bfloat16* q_head = q + (batch_idx * n_q_heads + q_head_idx) * head_dim;
    const __nv_bfloat16* k_head = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    const __nv_bfloat16* v_head = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    __nv_bfloat16* out_head = out + (batch_idx * n_q_heads + q_head_idx) * head_dim;

    // Load Q into registers (each thread loads part of the vector)
    float q_reg[8];  // Assume max 8 elements per thread
    int elems_per_thread = (head_dim + blockDim.x - 1) / blockDim.x;
    elems_per_thread = min(elems_per_thread, 8);

    for (int i = 0; i < elems_per_thread; i++) {
        int idx = threadIdx.x * elems_per_thread + i;
        if (idx < head_dim) {
            q_reg[i] = __bfloat162float(q_head[idx]);
        } else {
            q_reg[i] = 0.0f;
        }
    }

    // Shared memory layout:
    // [0..num_warps): for reductions
    // [num_warps..num_warps + BLOCK_K): for attention weights
    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    float* reduce_shared = shared;
    float* attn_weights = shared + num_warps;

    // Phase 1: Compute all attention scores and find max
    float max_score = -INFINITY;

    // Each thread processes multiple KV positions
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        const __nv_bfloat16* k_pos = k_head + kv_pos * head_dim;

        // Compute dot product Q @ K^T for this position
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float q_val = __bfloat162float(q_head[i]);
            float k_val = __bfloat162float(k_pos[i]);
            score += q_val * k_val;
        }
        score *= scale;

        attn_weights[kv_pos] = score;
        max_score = fmaxf(max_score, score);
    }

    __syncthreads();

    // Reduce max across block
    max_score = attn_block_reduce_max(max_score, reduce_shared);
    __syncthreads();

    // Phase 2: Compute exp(score - max) and sum
    float sum_exp = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        float score = attn_weights[kv_pos];
        float exp_score = expf(score - max_score);
        attn_weights[kv_pos] = exp_score;
        sum_exp += exp_score;
    }

    __syncthreads();

    // Reduce sum across block
    sum_exp = attn_block_reduce_sum(sum_exp, reduce_shared);
    __syncthreads();

    // Phase 3: Normalize and compute weighted sum of V
    float inv_sum = 1.0f / sum_exp;

    // Each thread accumulates its part of the output
    float out_accum[8] = {0.0f};

    for (int kv_pos = 0; kv_pos < cache_len; kv_pos++) {
        float weight = attn_weights[kv_pos] * inv_sum;
        const __nv_bfloat16* v_pos = v_head + kv_pos * head_dim;

        // Accumulate weighted V
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = threadIdx.x * elems_per_thread + i;
            if (idx < head_dim) {
                float v_val = __bfloat162float(v_pos[idx]);
                out_accum[i] += weight * v_val;
            }
        }
    }

    // Store output
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = threadIdx.x * elems_per_thread + i;
        if (idx < head_dim) {
            out_head[idx] = __float2bfloat16(out_accum[i]);
        }
    }
}

/**
 * Attention decode kernel v2 - simpler single-pass approach
 * Each block handles one query head, threads cooperate on dot products
 */
__global__ void attention_decode_kernel_v2(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    int cache_len,// 21
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,// 533
    float scale
) {
    // Shared memory layout:
    // - q_shared: head_dim floats for Q vector
    // - scores: cache_len floats for attention scores
    // - reduce_shared: num_warps floats for reductions
    extern __shared__ float shared[];

    int batch_idx = blockIdx.y;
    int q_head_idx = blockIdx.x;

    // GQA: map query head to kv head
    int n_groups = n_q_heads / n_kv_heads;
    int kv_head_idx = q_head_idx / n_groups;

    // Pointers
    const __nv_bfloat16* q_head = q + (batch_idx * n_q_heads + q_head_idx) * head_dim;
    const __nv_bfloat16* k_head = k_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    const __nv_bfloat16* v_head = v_cache + (batch_idx * n_kv_heads + kv_head_idx) * max_seq_len * head_dim;
    __nv_bfloat16* out_head = out + (batch_idx * n_q_heads + q_head_idx) * head_dim;

    int num_warps = (blockDim.x + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;

    float* q_shared = shared;
    float* scores = shared + head_dim;
    float* reduce_shared = shared + head_dim + cache_len;

    // Load Q into shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_shared[i] = __bfloat162float(q_head[i]);
    }
    __syncthreads();

    // Phase 1: Compute all attention scores Q @ K^T
    float local_max = -INFINITY;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        const __nv_bfloat16* k_pos = k_head + kv_pos * head_dim;

        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += q_shared[i] * __bfloat162float(k_pos[i]);
        }
        score *= scale;
        scores[kv_pos] = score;
        local_max = fmaxf(local_max, score);
    }
    __syncthreads();

    // Reduce to find global max
    float max_score = attn_block_reduce_max(local_max, reduce_shared);
    __syncthreads();

    // Phase 2: Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < cache_len; kv_pos += blockDim.x) {
        float exp_score = expf(scores[kv_pos] - max_score);
        scores[kv_pos] = exp_score;
        local_sum += exp_score;
    }
    __syncthreads();

    float sum_exp = attn_block_reduce_sum(local_sum, reduce_shared);
    __syncthreads();

    float inv_sum = 1.0f / sum_exp;

    // Phase 3: Compute weighted sum of V values
    // Each thread is responsible for a subset of output dimensions
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < cache_len; kv_pos++) {
            float weight = scores[kv_pos] * inv_sum;
            float v_val = __bfloat162float(v_head[kv_pos * head_dim + d]);
            acc += weight * v_val;
        }
        out_head[d] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// Helper: warp‑wide reduction (sum)
// ---------------------------------------------------------------------------
#ifndef ATTN_WARP_SIZE
#define ATTN_WARP_SIZE 32
#endif

__inline__ __device__ float warp_reduce_sum_v3(float val) {
    // Reduce within a warp using shuffle instructions.
    #pragma unroll
    for (int offset = ATTN_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Simple attention decode kernel (no FP8, no debug)
// ---------------------------------------------------------------------------
/**
 *  Compute one‑step (decode) attention for a batch of queries.
 *
 *  Layouts (same as the original wrapper):
 *    q        : [batch, n_q_heads, head_dim]          (bf16)
 *    k_cache  : [batch, n_kv_heads, max_seq_len, head_dim] (bf16)
 *    v_cache  : [batch, n_kv_heads, max_seq_len, head_dim] (bf16)
 *    out      : [batch, n_q_heads, head_dim]          (bf16)
 *
 *  The kernel is launched with:
 *      dim3 grid(n_q_heads, batch);
 *      int   threads = 128;   // 4 warps per block
 *
 *  Each block processes a single (batch, q_head) pair.
 *  Warps cooperatively iterate over the KV cache, performing an
 *  online soft‑max and accumulating the weighted V vectors.
 *
 *  No FP8 handling, no block‑table paging, and no debug instrumentation.
 */
extern "C" __global__ void attention_decode_kernel_v3(
    const __nv_bfloat16* __restrict__ q,          // [B, n_q_heads, D]
    const __nv_bfloat16* __restrict__ k_cache,   // [B, n_kv_heads, S, D]
    const __nv_bfloat16* __restrict__ v_cache,   // [B, n_kv_heads, S, D]
    __nv_bfloat16* __restrict__ out,             // [B, n_q_heads, D]
    int cache_len,                               // 当前 KV 长度（已包含新 token）
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale                                   // 1/√D
) {
    // --------------------------------------------------------------
    // 1) block / thread identification
    // --------------------------------------------------------------
    const int batch_id = blockIdx.y;
    const int q_head   = blockIdx.x;

    // GQA mapping: 每个 KV head 负责 kv_ratio 个 Q head
    int kv_ratio = n_q_heads / n_kv_heads;
    if (kv_ratio <= 0) kv_ratio = 1;
    int kv_head = q_head / kv_ratio;
    if (kv_head >= n_kv_heads) kv_head = n_kv_heads - 1;

    const int lane_id = threadIdx.x % ATTN_WARP_SIZE;          // 0‑31
    const int warp_id = threadIdx.x / ATTN_WARP_SIZE;          // 0‑(warps_per_block‑1)
    const int warps_per_block = blockDim.x / ATTN_WARP_SIZE;   // e.g. 128/32 = 4

    // --------------------------------------------------------------
    // 2) pointers to the relevant tensors
    // --------------------------------------------------------------
    const __nv_bfloat16* q_vec = q
        + ((batch_id * n_q_heads + q_head) * head_dim);
    const __nv_bfloat16* k_base = k_cache
        + ((batch_id * n_kv_heads + kv_head) * max_seq_len * head_dim);
    const __nv_bfloat16* v_base = v_cache
        + ((batch_id * n_kv_heads + kv_head) * max_seq_len * head_dim);
    __nv_bfloat16* out_vec = out
        + ((batch_id * n_q_heads + q_head) * head_dim);

    // --------------------------------------------------------------
    // 3) load Q into registers (lane‑wise)
    // --------------------------------------------------------------
    constexpr int MAX_ELEMS_PER_LANE = 8;                     // 支持 up to 256‑dim
    int elems_per_lane = (head_dim + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    if (elems_per_lane > MAX_ELEMS_PER_LANE) elems_per_lane = MAX_ELEMS_PER_LANE;

    float q_local[MAX_ELEMS_PER_LANE] = {0.f};
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
        if (i < elems_per_lane) {
            int d = lane_id + i * ATTN_WARP_SIZE;
            if (d < head_dim) {
                q_local[i] = __bfloat162float(q_vec[d]);
            }
        }
    }

    // --------------------------------------------------------------
    // 4) per‑warp online‑softmax state
    // --------------------------------------------------------------
    float m_w = -INFINITY;                     // 当前 warp 的 max
    float s_w = 0.f;                           // 当前 warp 的 Σexp
    float out_w[MAX_ELEMS_PER_LANE] = {0.f};   // warp‑局部输出累加

    // --------------------------------------------------------------
    // 5) 主循环：遍历 KV cache（每个 warp 负责 stride‑step）
    // --------------------------------------------------------------
    for (int t = warp_id; t < cache_len; t += warps_per_block) {
        const __nv_bfloat16* k_vec = k_base + t * head_dim;
        const __nv_bfloat16* v_vec = v_base + t * head_dim;

        // ---- dot(Q, K[t]) ----
        float dot = 0.f;
        #pragma unroll
        for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
            if (i < elems_per_lane) {
                int d = lane_id + i * ATTN_WARP_SIZE;
                if (d < head_dim) {
                    float kval = __bfloat162float(k_vec[d]);
                    dot += q_local[i] * kval;
                }
            }
        }
        // warp‑wide reduction
        dot = warp_reduce_sum_v3(dot);
        float score = dot * scale;

        // ---- online softmax (max‑sum trick) ----
        float m_new = fmaxf(m_w, score);
        float old_scale = expf(m_w - m_new);
        float new_scale = expf(score - m_new);
        s_w = s_w * old_scale + new_scale;

        // ---- accumulate weighted V[t] ----
        #pragma unroll
        for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
            if (i < elems_per_lane) {
                int d = lane_id + i * ATTN_WARP_SIZE;
                if (d < head_dim) {
                    float vval = __bfloat162float(v_vec[d]);
                    out_w[i] = out_w[i] * old_scale + vval * new_scale;
                }
            }
        }
        m_w = m_new;   // 为下一次迭代准备
    }

    // --------------------------------------------------------------
    // 6) 写入共享内存（每个 warp 的局部结果）
    // --------------------------------------------------------------
    extern __shared__ float shmem[];
    // layout:
    //   sm_m[warps]          : per‑warp max
    //   sm_s[warps]          : per‑warp sum
    //   sm_out[warps * head_dim] : per‑warp partial output
    //   sm_global_m, sm_global_s   : block‑wide max / Σexp
    float* sm_m   = shmem;                                 // size = warps_per_block
    float* sm_s   = sm_m + warps_per_block;                // size = warps_per_block
    float* sm_out = sm_s + warps_per_block;                // size = warps_per_block * head_dim
    float* sm_global_m = sm_out + warps_per_block * head_dim; // 1 float
    float* sm_global_s = sm_global_m + 1;                     // 1 float

    // lane 0 of each warp writes its max / sum
    if (lane_id == 0) {
        sm_m[warp_id] = m_w;
        sm_s[warp_id] = s_w;
    }

    // each lane writes its slice of the partial output
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
        if (i < elems_per_lane) {
            int d = lane_id + i * ATTN_WARP_SIZE;
            if (d < head_dim) {
                sm_out[warp_id * head_dim + d] = out_w[i];
            }
        }
    }

    __syncthreads();

    // --------------------------------------------------------------
    // 7) block‑wide reduction (max & Σexp) – only one thread does it
    // --------------------------------------------------------------
    if (threadIdx.x == 0) {
        float block_max = -INFINITY;
        for (int w = 0; w < warps_per_block; ++w) {
            block_max = fmaxf(block_max, sm_m[w]);
        }
        float block_sum = 0.f;
        for (int w = 0; w < warps_per_block; ++w) {
            float scale_w = (sm_s[w] > 0.f) ? expf(sm_m[w] - block_max) : 0.f;
            block_sum += sm_s[w] * scale_w;
        }
        *sm_global_m = block_max;
        *sm_global_s = block_sum;
    }

    __syncthreads();   // 确保所有 warp 能看到 block‑wide max / sum

    float global_m = *sm_global_m;
    float global_s = *sm_global_s;
    float inv_s    = (global_s > 0.f) ? 1.f / global_s : 0.f;

    // --------------------------------------------------------------
    // 8) 最终输出：把所有 warp 的贡献合并
    // --------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < MAX_ELEMS_PER_LANE; ++i) {
        if (i < elems_per_lane) {
            int d = lane_id + i * ATTN_WARP_SIZE;
            if (d < head_dim) {
                float acc = 0.f;
                for (int w = 0; w < warps_per_block; ++w) {
                    float scale_w = (sm_s[w] > 0.f) ? expf(sm_m[w] - global_m) : 0.f;
                    acc += sm_out[w * head_dim + d] * scale_w;
                }
                float out_f = acc * inv_s;
                out_vec[d] = __float2bfloat16(out_f);
            }
        }
    }
}

#include <cuda_bf16.h>          // <-- added for bfloat16
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;
template<int HEAD_DIM, int BLOCK_SIZE, int NUM_WARPS>
__global__ void flash_decode_splitk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float*      __restrict__ partial_out,
    float*      __restrict__ partial_max,
    float*      __restrict__ partial_sum,
    const int*  __restrict__ block_tables,
    const int*  __restrict__ seq_lens,
    float scale,
    int max_blocks_per_seq,
    int H_q, int H_kv,
    int head_dim,
    long head_stride,               // <-- new argument: true stride (max_seq_len * head_dim)
    int num_splits
) {
    // -----------------------------------------------------------------
    // 0) sanity checks (optional, can be compiled out with -DNDEBUG)
    // -----------------------------------------------------------------
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    constexpr int ELEMS = HEAD_DIM / 32;

    const int seq_idx  = blockIdx.x;          // which sequence in the batch
    const int h_q      = blockIdx.y;          // which Q‑head
    const int split_id = blockIdx.z;          // which split
    const int lane     = threadIdx.x % 32;
    const int warp_id  = threadIdx.x / 32;
    const int h_kv     = h_q * H_kv / H_q;    // KV‑head that belongs to this Q‑head

    const int context_len = seq_lens[seq_idx];
    const int num_blocks  = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // -------------------------------------------------------------
    // 1) Split‑K bookkeeping
    // -------------------------------------------------------------
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start      = split_id * blocks_per_split;
    const int split_end        = min(split_start + blocks_per_split, num_blocks);

    // -------------------------------------------------------------
    // 2) Load Q (same for every warp / split of this head)
    // -------------------------------------------------------------
    float q_reg[ELEMS];
    {
        const __nv_bfloat16* q_ptr = q + (seq_idx * H_q + h_q) * HEAD_DIM + lane * ELEMS;
        const int2 raw   = *reinterpret_cast<const int2*>(q_ptr);
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const float2 lo_f = __bfloat1622float2(lo);
        const float2 hi_f = __bfloat1622float2(hi);
        q_reg[0] = lo_f.x;  q_reg[1] = lo_f.y;
        q_reg[2] = hi_f.x;  q_reg[3] = hi_f.y;
    }

    // -------------------------------------------------------------
    // 3) Per‑warp soft‑max state
    // -------------------------------------------------------------
    float warp_max = -INFINITY;
    float warp_sum = 0.0f;
    float warp_acc[ELEMS];
    #pragma unroll
    for (int d = 0; d < ELEMS; d++) warp_acc[d] = 0.0f;

    // -------------------------------------------------------------
    // 4) Pointer to this sequence’s block table (logical block ids)
    // -------------------------------------------------------------
    const int* seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // -------------------------------------------------------------
    // 5) Main loop – iterate over the KV blocks that belong to this split
    // -------------------------------------------------------------
    for (int blk = split_start + warp_id; blk < split_end; blk += NUM_WARPS) {
        const int logical_block = seq_block_table[blk];          // 0 … max_blocks_per_seq‑1
        // ----- compute the *global* offset of the first token of this block -----
        // layout: [batch, H_kv, max_seq_len, HEAD_DIM]   (head‑major)
        const long block_base =
            // batch‑head stride (uses the true stride of the KV cache)
            ((long)seq_idx * H_kv + h_kv) * head_stride
            // block‑within‑sequence stride
            + (long)logical_block * BLOCK_SIZE * HEAD_DIM;

        const int tokens_in_block = min(BLOCK_SIZE, context_len - blk * BLOCK_SIZE);

        for (int tok = 0; tok < tokens_in_block; tok++) {
            const __nv_bfloat16* k_ptr = k_cache + block_base + (long)tok * HEAD_DIM + lane * ELEMS;
            const __nv_bfloat16* v_ptr = v_cache + block_base + (long)tok * HEAD_DIM + lane * ELEMS;

            // ---- load K ----------------------------------------------------
            const int2   k_raw = *reinterpret_cast<const int2*>(k_ptr);
            const __nv_bfloat162 k_lo = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.x);
            const __nv_bfloat162 k_hi = *reinterpret_cast<const __nv_bfloat162*>(&k_raw.y);
            const float2 k_lo_f = __bfloat1622float2(k_lo);
            const float2 k_hi_f = __bfloat1622float2(k_hi);

            // ---- dot‑product ------------------------------------------------
            float partial = q_reg[0] * k_lo_f.x + q_reg[1] * k_lo_f.y
                          + q_reg[2] * k_hi_f.x + q_reg[3] * k_hi_f.y;

            // ---- warp‑wide reduction of the dot product --------------------
            #pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1)
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, mask);
            const float score = partial * scale;

            // ---- online soft‑max update ------------------------------------
            const float new_max = fmaxf(warp_max, score);
            const float alpha   = expf(warp_max - new_max);
            const float p       = expf(score - new_max);
            warp_sum = warp_sum * alpha + p;
            warp_max = new_max;

            // ---- load V ----------------------------------------------------
            const int2   v_raw = *reinterpret_cast<const int2*>(v_ptr);
            const __nv_bfloat162 v_lo = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.x);
            const __nv_bfloat162 v_hi = *reinterpret_cast<const __nv_bfloat162*>(&v_raw.y);
            const float2 v_lo_f = __bfloat1622float2(v_lo);
            const float2 v_hi_f = __bfloat1622float2(v_hi);

            // ---- accumulate weighted V --------------------------------------
            warp_acc[0] = warp_acc[0] * alpha + p * v_lo_f.x;
            warp_acc[1] = warp_acc[1] * alpha + p * v_lo_f.y;
            warp_acc[2] = warp_acc[2] * alpha + p * v_hi_f.x;
            warp_acc[3] = warp_acc[3] * alpha + p * v_hi_f.y;
        }
    }

    // -----------------------------------------------------------------
    // 6) Cross‑warp reduction (identical to the original implementation)
    // -----------------------------------------------------------------
    extern __shared__ float smem[];
    float* smem_max = smem;
    float* smem_sum = smem_max + NUM_WARPS;
    float* smem_acc = smem_sum + NUM_WARPS;

    #pragma unroll
    for (int e = 0; e < ELEMS; e++)
        smem_acc[warp_id * HEAD_DIM + lane * ELEMS + e] = warp_acc[e];

    if (lane == 0) {
        smem_max[warp_id] = warp_max;
        smem_sum[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        // ---- global max across warps ------------------------------------
        float global_max = smem_max[0];
        #pragma unroll
        for (int w = 1; w < NUM_WARPS; w++)
            global_max = fmaxf(global_max, smem_max[w]);

        // ---- global sum across warps (with scaling) --------------------
        float global_sum = 0.0f;
        float final_acc[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) final_acc[e] = 0.0f;

        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            const float alpha = expf(smem_max[w] - global_max);
            global_sum += smem_sum[w] * alpha;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                final_acc[e] += smem_acc[w * HEAD_DIM + lane * ELEMS + e] * alpha;
        }

        // ---- write per‑split partial results ----------------------------
        const long out_offset = ((long)seq_idx * H_q + h_q) * num_splits + split_id;
        if (lane == 0) {
            partial_max[out_offset] = global_max;
            partial_sum[out_offset] = global_sum;
        }
        float* po = partial_out + out_offset * HEAD_DIM + lane * ELEMS;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            po[e] = final_acc[e];
    }
}
template<int HEAD_DIM>
__global__ void flash_decode_reduce_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ out,          // <-- changed
    int H_q,
    int num_splits
) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be divisible by 32");
    constexpr int ELEMS = HEAD_DIM / 32;

    const int seq_idx = blockIdx.x;
    const int h_q     = blockIdx.y;
    const int lane    = threadIdx.x;

    const long base = ((long)seq_idx * H_q + h_q) * num_splits;

    // Find global max across splits
    float global_max = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        float m = partial_max[base + s];
        global_max = fmaxf(global_max, m);
    }

    // Merge: rescale each split's partial and accumulate
    float global_sum = 0.0f;
    float acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        const float alpha = expf(partial_max[base + s] - global_max);
        global_sum += partial_sum[base + s] * alpha;

        const float* po = partial_out + (base + s) * HEAD_DIM + lane * ELEMS;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            acc[e] += po[e] * alpha;
    }

    // Normalize and write final output
    const float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    __nv_bfloat16* out_ptr = out + (seq_idx * H_q + h_q) * HEAD_DIM + lane * ELEMS;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++)
        out_ptr[e] = __float2bfloat16(acc[e] * inv);   // <-- changed
}

// Wrapper function callable from PyTorch
// q layout:   [num_seqs, H_q, D]
// K, V layout (new): [num_seqs, H_kv, S_kv, D]  — head‑major KV context
// O layout:   [num_seqs, H_q, D]
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
) {
    // Use 128 threads per block (4 warps)
    int threads = 128;

    // Shared memory: Q (head_dim) + scores (cache_len) + reduce buffer (num_warps)
    int num_warps = (threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    int shared_mem = (head_dim + cache_len + num_warps) * sizeof(float);

    dim3 grid(n_q_heads, batch_size);

    attention_decode_kernel<<<grid, threads, shared_mem, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (__nv_bfloat16*)out,
        cache_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        scale
    );
}


// Wrapper function callable from PyTorch
// q layout:   [num_seqs, H_q, D]
// K, V layout (new): [num_seqs, H_kv, S_kv, D]  — head‑major KV context
// O layout:   [num_seqs, H_q, D]
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
) {
    // Use 128 threads per block (4 warps)
    int threads = 128;

    // Shared memory: Q (head_dim) + scores (cache_len) + reduce buffer (num_warps)
    int num_warps = (threads + ATTN_WARP_SIZE - 1) / ATTN_WARP_SIZE;
    int shared_mem = (head_dim + cache_len + num_warps) * sizeof(float);

    dim3 grid(n_q_heads, batch_size);

    attention_decode_kernel_v2<<<grid, threads, shared_mem, stream>>>(
        (const __nv_bfloat16*)q,
        (const __nv_bfloat16*)k_cache,
        (const __nv_bfloat16*)v_cache,
        (__nv_bfloat16*)out,
        cache_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        scale
    );
}

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
) {
    // 128 threads → 4 warps per block (can be tuned)
    const int threads = 128;
    const int warps_per_block = threads / ATTN_WARP_SIZE;   // ATTN_WARP_SIZE = 32

    // Dynamic shared memory layout:
    //   per‑warp max (float)      -> warps_per_block
    //   per‑warp sum (float)      -> warps_per_block
    //   per‑warp output vector    -> warps_per_block * head_dim
    const size_t shared_mem_bytes =
        (size_t)(warps_per_block * (2 + head_dim)+ 2) * sizeof(float);

    dim3 grid(n_q_heads, batch_size);

    attention_decode_kernel_v3<<<grid, threads, shared_mem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_cache),
        static_cast<const __nv_bfloat16*>(v_cache),
        static_cast<__nv_bfloat16*>(out),
        cache_len,
        n_q_heads,
        n_kv_heads,
        head_dim,
        max_seq_len,
        scale
    );
}

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

    extern "C" void launch_attention_decode_v4_new(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    int batch_size,
    int cache_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len_total,          // <-- renamed to avoid shadowing
    float scale,
    cudaStream_t stream,
    const int* d_block_tables,
    const int* d_seq_lens,
    bool use_page_cache
) {
    // ------------------------------------------------------------------
    // 1.  Basic constants
    // ------------------------------------------------------------------
    constexpr int BLOCK_SIZE = 16;   // KV‑cache block size
    constexpr int NUM_WARPS  = 8;    // 8 warps = 256 threads per CTA
    const int num_splits = 2;        // can be tuned

    // ------------------------------------------------------------------
    // 2.  Derived sizes
    // ------------------------------------------------------------------
    const int max_blocks_per_seq = (cache_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // The *actual* stride (tokens per head) of the KV cache – comes from the
    // tensor shape supplied by the Python wrapper.
    const long head_stride = static_cast<long>(max_seq_len_total) * head_dim;

    // ------------------------------------------------------------------
    // 3.  Temporary buffers (only needed when !use_page_cache)
    // ------------------------------------------------------------------
    float *d_partial_out = nullptr;
    float *d_partial_max = nullptr;
    float *d_partial_sum = nullptr;
    int   *d_block_tables_local = nullptr;
    int   *d_seq_lens_local = nullptr;

    CUDA_CHECK(cudaMalloc(&d_partial_out,
        static_cast<size_t>(batch_size) *
        n_q_heads * num_splits * head_dim *
        sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max,
        static_cast<size_t>(batch_size) *
        n_q_heads * num_splits *
        sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum,
        static_cast<size_t>(batch_size) *
        n_q_heads * num_splits *
        sizeof(float)));

    if (!use_page_cache) {
        // --------------------------------------------------------------
        // Allocate and fill block‑tables + seq_lens on the host, then copy.
        // --------------------------------------------------------------
        CUDA_CHECK(cudaMalloc(&d_block_tables_local,
                 static_cast<size_t>(batch_size) *
                 max_blocks_per_seq * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seq_lens_local,
                 static_cast<size_t>(batch_size) *
                 sizeof(int)));

        // Host‑side temporary buffers
        int *h_block_tables = new int[static_cast<size_t>(batch_size) *
                                      max_blocks_per_seq];
        int *h_seq_lens    = new int[batch_size];

        for (int b = 0; b < batch_size; ++b) {
            h_seq_lens[b] = cache_len;               // actual length of each sequence
            // **Logical** block IDs: 0 … max_blocks_per_seq‑1
            for (int blk = 0; blk < max_blocks_per_seq; ++blk) {
                h_block_tables[b * max_blocks_per_seq + blk] = blk;
            }
        }

        // Asynchronously copy to device
        CUDA_CHECK(cudaMemcpyAsync(d_block_tables_local, h_block_tables,
                 static_cast<size_t>(batch_size) *
                 max_blocks_per_seq * sizeof(int),
                 cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_local, h_seq_lens,
                 batch_size * sizeof(int),
                 cudaMemcpyHostToDevice, stream));

        delete[] h_block_tables;
        delete[] h_seq_lens;
    }

    // ------------------------------------------------------------------
    // 4.  Choose head_dim and launch kernels
    // ------------------------------------------------------------------
    auto launch_for_head = [&](auto HEAD) {
        constexpr int HD = decltype(HEAD)::value;   // compile‑time head_dim

        const size_t smem_bytes = (2 * NUM_WARPS + NUM_WARPS * HD) * sizeof(float);

        // split‑K kernel
        dim3 grid_splitk(batch_size, n_q_heads, num_splits);
        dim3 block_splitk(32 * NUM_WARPS);   // 256 threads

        flash_decode_splitk_kernel<HD, BLOCK_SIZE, NUM_WARPS>
        <<<grid_splitk, block_splitk, smem_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q),
            static_cast<const __nv_bfloat16*>(k_cache),
            static_cast<const __nv_bfloat16*>(v_cache),
            d_partial_out, d_partial_max, d_partial_sum,
            // use the caller‑provided tables if they exist, otherwise the locals we just built
            use_page_cache ? d_block_tables : d_block_tables_local,
            use_page_cache ? d_seq_lens   : d_seq_lens_local,
            scale,
            max_blocks_per_seq,
            n_q_heads,
            n_kv_heads,
            head_dim,
            // <<<--- NEW: pass the true head stride for address calculation
            head_stride,
            num_splits
        );

        // reduce kernel
        dim3 grid_reduce(batch_size, n_q_heads);
        dim3 block_reduce(32);   // 1 warp

        flash_decode_reduce_kernel<HD>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            d_partial_out, d_partial_max, d_partial_sum,
            static_cast<__nv_bfloat16*>(out),
            n_q_heads,
            num_splits
        );
    };

    if (head_dim == 64) {
        launch_for_head(std::integral_constant<int, 64>{});
    } else if (head_dim == 128) {
        launch_for_head(std::integral_constant<int, 128>{});
    } else if (head_dim == 256) {
        launch_for_head(std::integral_constant<int, 256>{});
    } else {
        fprintf(stderr,
                "[launch_attention_decode_v4_new] Unsupported head_dim %d. "
                "Supported values: 64, 128, 256.\n",
                head_dim);
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------
    // 5.  Synchronize, error‑check and free temporaries
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));

    if (!use_page_cache) {
        CUDA_CHECK(cudaFree(d_block_tables_local));
        CUDA_CHECK(cudaFree(d_seq_lens_local));
    }
}