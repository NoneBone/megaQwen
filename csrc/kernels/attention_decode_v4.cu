
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
        // max_seq_len = max_blocks_per_seq * BLOCK_SIZE
        const long max_seq_len = (long)max_blocks_per_seq * BLOCK_SIZE;
        const long block_base =
            // batch‑head stride
            ((long)seq_idx * H_kv + h_kv) * max_seq_len * HEAD_DIM
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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Deprecated  version. Even though the unit test performance is excellent, 
// the current integration test performance is not satisfactory because of the poor performance of the page attn system.
extern "C" void launch_attention_decode_v4(
    const void* q,               // [batch, n_q_heads, 1, head_dim]   (bfloat16)
    const void* k_cache,         // [batch, n_kv_heads, max_seq_len, head_dim] (bfloat16, block‑major)
    const void* v_cache,         // 同上
    void* out,                   // [batch, n_q_heads, head_dim]   (bfloat16)
    int batch_size,
    int cache_len,               // 已经缓存的 token 数目 (= seq_len)
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq_len,             // 与 k_cache/v_cache 第三维相同（这里不直接使用，只用于检查）
    float scale,
    cudaStream_t stream
) {
    // ------------------------------------------------------------------
    // 1. 基本指针转换
    // ------------------------------------------------------------------
    const __nv_bfloat16* d_q       = static_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* d_k_cache = static_cast<const __nv_bfloat16*>(k_cache);
    const __nv_bfloat16* d_v_cache = static_cast<const __nv_bfloat16*>(v_cache);
    __nv_bfloat16*       d_out     = static_cast<__nv_bfloat16*>(out);

    // ------------------------------------------------------------------
    // 2. 常量配置（与 v3 中保持一致）
    // ------------------------------------------------------------------
    constexpr int BLOCK_SIZE = 16;   // KV‑cache block size
    constexpr int NUM_WARPS  = 8;    // 8 warps = 256 threads per CTA
    const int num_splits = 2;        // 可自行调节，2 在大多数长度下效果不错

    // ------------------------------------------------------------------
    // 3. 派生尺寸
    // ------------------------------------------------------------------
    const int max_blocks_per_seq = (cache_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // partial buffers: [batch, n_q_heads, num_splits, head_dim] (float)
    const size_t ws_out = static_cast<size_t>(batch_size) * n_q_heads * num_splits * head_dim;
    const size_t ws_ms  = static_cast<size_t>(batch_size) * n_q_heads * num_splits; // max / sum

    // ------------------------------------------------------------------
    // 4. 申请临时显存
    // ------------------------------------------------------------------
    float *d_partial_out = nullptr;
    float *d_partial_max = nullptr;
    float *d_partial_sum = nullptr;
    int   *d_block_tables = nullptr;
    int   *d_seq_lens = nullptr;

    CUDA_CHECK(cudaMalloc(&d_partial_out, ws_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_max, ws_ms  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_sum, ws_ms  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_tables,
                         static_cast<size_t>(batch_size) * max_blocks_per_seq * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, batch_size * sizeof(int)));

    // ------------------------------------------------------------------
    // 5. 初始化 block_tables（逻辑 → 物理）和 seq_lens
    //    这里使用最简单的 “identity” 映射：物理 block = batch * max_blocks_per_seq + logical
    // ------------------------------------------------------------------
    {
        // Allocate host‑side temporary buffers
    int *h_block_tables = new int[static_cast<size_t>(batch_size) * max_blocks_per_seq];
    int *h_seq_lens    = new int[batch_size];

    for (int b = 0; b < batch_size; ++b) {
        h_seq_lens[b] = cache_len;               // actual length of each sequence
        for (int blk = 0; blk < max_blocks_per_seq; ++blk) {
            // Store **logical** block index only.
            // The kernel will combine it with the sequence id (seq_idx) later.
            h_block_tables[b * max_blocks_per_seq + blk] = blk;
        }
    }

    // Asynchronously copy to device
    CUDA_CHECK(cudaMemcpyAsync(d_block_tables, h_block_tables,
                               static_cast<size_t>(batch_size) * max_blocks_per_seq * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens, h_seq_lens,
                               batch_size * sizeof(int),
                               cudaMemcpyHostToDevice, stream));

    delete[] h_block_tables;
    delete[] h_seq_lens;
    }

    // ------------------------------------------------------------------
    // 6. 根据 head_dim 进行模板实例化并启动 kernel
    //    flash_decode_splitk_kernel<HEAD_DIM, BLOCK_SIZE, NUM_WARPS>
    //    flash_decode_reduce_kernel<HEAD_DIM>
    // ------------------------------------------------------------------
    // 为了在运行时根据 head_dim 选择编译期常量，使用 std::integral_constant 包装
    auto launch_for_head = [&](auto HEAD) {
        constexpr int HD = decltype(HEAD)::value;   // 编译期 head_dim

        // 共享内存大小： (2 * NUM_WARPS + NUM_WARPS * HD) * sizeof(float)
        const size_t smem_bytes = (2 * NUM_WARPS + NUM_WARPS * HD) * sizeof(float);

        // ------------------- split‑K kernel -------------------
        dim3 grid_splitk(batch_size, n_q_heads, num_splits);
        dim3 block_splitk(32 * NUM_WARPS);   // 256 线程

        flash_decode_splitk_kernel<HD, BLOCK_SIZE, NUM_WARPS>
        <<<grid_splitk, block_splitk, smem_bytes, stream>>>(
            d_q, d_k_cache, d_v_cache,
            d_partial_out, d_partial_max, d_partial_sum,
            d_block_tables, d_seq_lens,
            scale, max_blocks_per_seq, n_q_heads, n_kv_heads, num_splits
        );

        // ------------------- reduce kernel -------------------
        dim3 grid_reduce(batch_size, n_q_heads);
        dim3 block_reduce(32);   // 1 warp

        flash_decode_reduce_kernel<HD>
        <<<grid_reduce, block_reduce, 0, stream>>>(
            d_partial_out, d_partial_max, d_partial_sum,
            d_out, n_q_heads, num_splits
        );
    };

    // 只支持常见的 64 / 128 / 256 三种 head_dim（必须能被 32 整除）
    if (head_dim == 64) {
        launch_for_head(std::integral_constant<int, 64>{});
    } else if (head_dim == 128) {
        launch_for_head(std::integral_constant<int, 128>{});
    } else if (head_dim == 256) {
        launch_for_head(std::integral_constant<int, 256>{});
    } else {
        fprintf(stderr,
                "[launch_attention_decode_v4] Unsupported head_dim %d. "
                "Supported values: 64, 128, 256.\n",
                head_dim);
        exit(EXIT_FAILURE);
    }

    // ------------------------------------------------------------------
    // 7. 同步、错误检查并释放临时显存
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(d_partial_out));
    CUDA_CHECK(cudaFree(d_partial_max));
    CUDA_CHECK(cudaFree(d_partial_sum));
    CUDA_CHECK(cudaFree(d_block_tables));
    CUDA_CHECK(cudaFree(d_seq_lens));
}
