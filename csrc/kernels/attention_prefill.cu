#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;
static constexpr int WARP_SIZE = 32;

// FA2 Prefill Kernel  (WMMA Tensor Core, 4 warps, Bc=64, async double-buffer K/V)
// Grid:  (ceil(S/Br), H_q, B)
// Block: (NUM_WARPS * 32 = 128)  — 4 warps
//
// Warp roles (each warp handles a non-overlapping tile):
//   QK^T : warp w → S_smem[:, w*16:(w+1)*16]   (1 WMMA tile of [Br=16, 16])
//   PV   : warp w → O_smem[:, w*32:(w+1)*32]   (2 WMMA tiles of [Br=16, 16])
//
// Shared memory per block (~95 KB): compile-time offsets (no runtime allocation chain).
//   K/V are double-buffered (ping-pong) with cp.async to hide memory latency.
//   Buffer 0 is loaded while buffer 1 is computed, and vice versa.
//
//   Q_smem  [16][ 136] half:         4352 B  @ offset 0
//   K_smem[0][64][136] half:        17408 B  @ offset 4352
//   K_smem[1][64][136] half:        17408 B  @ offset 21760
//   V_smem[0][64][136] half:        17408 B  @ offset 39168
//   V_smem[1][64][136] half:        17408 B  @ offset 56576
//   S_smem  [16][  64] float:        4096 B  @ offset 73984
//   P_smem  [16][  64] half:         2048 B  @ offset 78080
//   O_smem  [16][ 128] float:        8192 B  @ offset 80128
//   warp_tmp[ 4][ 16][32] float:     8192 B  @ offset 88320
//   s_row_max/sum/alpha[16] float:    192 B  @ offset 96512
//   Total: 96704 B (~94.4 KB)
//
// Pipeline per KV tile:
//   1. Prologue (before loop): cp.async tile-0 → buf-0, commit_group
//   2. Each iteration i: cp.async tile-(i+1) → buf-(1-cur), commit_group,
//      wait_group(1) [cur buf ready], __syncthreads, compute, __syncthreads
//   3. Last iteration: wait_group(0) [drain final group]
//
// OOB K/V rows (kv_row >= S): cp.async skipped (smem retains prev. tile's data,
//   correctness preserved: OOB K scores masked to -INF; OOB V rows have P=0).
//
// Barriers per KV tile: 7 (same as single-buffer; async hides load latency)

template<int HEAD_DIM, int Br, int Bc, int NUM_WARPS>
__global__ void flash_attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q,   // [B, S, H_q,  D]
    const __nv_bfloat16* __restrict__ K,   // [B, S, H_kv, D]
    const __nv_bfloat16* __restrict__ V,   // [B, S, H_kv, D]
    __nv_bfloat16*       __restrict__ O,   // [B, S, H_q,  D]
    float*      __restrict__ LSE, // [B, S, H_q] or nullptr
    int B, int S, int H_q, int H_kv, float scale
) {
    static_assert(Br == 16 && HEAD_DIM % 16 == 0,
                  "WMMA FA2: Br must be 16; HEAD_DIM must be divisible by 16");
    static_assert(Bc == NUM_WARPS * 16,
                  "WMMA FA2: each warp handles exactly 16 KV cols → Bc = NUM_WARPS*16");
    static_assert(HEAD_DIM % (NUM_WARPS * 16) == 0,
                  "WMMA FA2: HEAD_DIM must be divisible by NUM_WARPS*16");

    // Each warp owns KV_COLS_PER_WARP=16 score cols and OUT_COLS_PER_WARP=32 output cols
    constexpr int TOTAL_THREADS     = NUM_WARPS * WARP_SIZE;    // 128
    constexpr int KV_COLS_PER_WARP  = Bc / NUM_WARPS;           // 64/4 = 16
    constexpr int OUT_COLS_PER_WARP = HEAD_DIM / NUM_WARPS;     // 128/4 = 32
    constexpr int OUT_FRAGS         = OUT_COLS_PER_WARP / 16;   // 32/16 = 2

    const int warp_id     = threadIdx.x / WARP_SIZE;// 0-3
    const int tx          = threadIdx.x;
    const int q_tile      = blockIdx.x;// 0-7
    const int h_q         = blockIdx.y;// 0-15
    const int b           = blockIdx.z;// 0-1
    const int h_kv        = h_q * H_kv / H_q;// 0-15 * 8/16 = 0-7
    const int q_row_start = q_tile * Br;// 0-7 * 16

    constexpr int PAD = 8;   // half‑row padding for bank‑conflict avoidance

    // Dynamic smem (~94 KB); K/V double‑buffered with cp.async.  Launcher sets carveout.
    extern __shared__ char dyn_smem[];

    // Double‑buffered K/V smem — all offsets computed at compile time from template params.
    // Layout (bytes): Q=0  K0=4352  K1=21760  V0=39168  V1=56576
    //                 S=73984  P=78080  O=80128  WT=88320  stats=96512
    constexpr int KV_STRIDE   = HEAD_DIM + PAD;   // half‑elements per padded row (136)
    constexpr int SZ_Q        = Br * KV_STRIDE * sizeof(__nv_bfloat16);   //  4352 B
    constexpr int SZ_KV       = Bc * KV_STRIDE * sizeof(__nv_bfloat16);   // 17408 B
    constexpr int OFF_K0      = SZ_Q;
    constexpr int OFF_K1      = OFF_K0 + SZ_KV;
    constexpr int OFF_V0      = OFF_K1 + SZ_KV;
    constexpr int OFF_V1      = OFF_V0 + SZ_KV;
    constexpr int OFF_S       = OFF_V1 + SZ_KV;            // 73984: first float region
    constexpr int OFF_P       = OFF_S  + Br * Bc * sizeof(float);             // 78080
    constexpr int OFF_O       = OFF_P  + Br * Bc * sizeof(__nv_bfloat16);    // 80128
    constexpr int OFF_WT      = OFF_O  + Br * HEAD_DIM * sizeof(float);       // 88320
    constexpr int OFF_RMAX    = OFF_WT + NUM_WARPS * Br * OUT_COLS_PER_WARP * sizeof(float); // 96512
    constexpr int OFF_RSUM    = OFF_RMAX + Br * sizeof(float);   // 96576
    constexpr int OFF_ALPHA   = OFF_RSUM + Br * sizeof(float);   // 96640

    __nv_bfloat16* Q_smem    = ( __nv_bfloat16*)(dyn_smem + 0);
    __nv_bfloat16* K_buf[2]  = { ( __nv_bfloat16*)(dyn_smem + OFF_K0),
                                 ( __nv_bfloat16*)(dyn_smem + OFF_K1) };
    __nv_bfloat16* V_buf[2]  = { ( __nv_bfloat16*)(dyn_smem + OFF_V0),
                                 ( __nv_bfloat16*)(dyn_smem + OFF_V1) };
    float*          S_smem    = (float*)(dyn_smem + OFF_S);
    __nv_bfloat16*  P_smem    = (__nv_bfloat16*)(dyn_smem + OFF_P);
    float*          O_smem    = (float*)(dyn_smem + OFF_O);
    float*          warp_tmp  = (float*)(dyn_smem + OFF_WT);
    float*          s_row_max = (float*)(dyn_smem + OFF_RMAX);
    float*          s_row_sum = (float*)(dyn_smem + OFF_RSUM);
    float*          s_alpha   = (float*)(dyn_smem + OFF_ALPHA);

    // 2D indexing helpers
    auto Qs = [&](int r, int c) -> __nv_bfloat16& { return Q_smem[r*KV_STRIDE + c]; };
    auto Ss = [&](int r, int c) -> float&          { return S_smem[r*Bc + c]; };
    auto Ps = [&](int r, int c) -> __nv_bfloat16& { return P_smem[r*Bc + c]; };
    auto Os = [&](int r, int c) -> float&          { return O_smem[r*HEAD_DIM + c]; };
    auto Wt = [&](int w, int r, int c) -> float&   { return warp_tmp[(w*Br + r)*OUT_COLS_PER_WARP + c]; };

    // Zero‑initialize K/V double buffers to prevent NaN from uninitialized smem.
    constexpr int KV_BUF_ELEMS = Bc * KV_STRIDE;
    for (int i = tx; i < KV_BUF_ELEMS; i += TOTAL_THREADS) {
        K_buf[0][i] = __float2bfloat16(0.0f);
        K_buf[1][i] = __float2bfloat16(0.0f);
        V_buf[0][i] = __float2bfloat16(0.0f);
        V_buf[1][i] = __float2bfloat16(0.0f);
    }

    // Initialize O_smem and softmax stats
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS)
        Os(i / HEAD_DIM, i % HEAD_DIM) = 0.0f;
    if (tx < Br) { s_row_max[tx] = -INFINITY; s_row_sum[tx] = 0.0f; }

    // Load Q tile [Br, HEAD_DIM] — cooperatively across all 4 warps
    //   Q layout: [B, H_q, S, D]
    //   index = ((b * H_q + h_q) * S + q_row) * HEAD_DIM + c
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, c = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        Qs(r, c) = (q_row < S)
            ? Q[((long)(b * H_q + h_q) * S + q_row) * HEAD_DIM + c]
            : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int num_kv_active = min((S + Bc - 1) / Bc, (q_row_start + Br - 1) / Bc + 1);

    // -------------------------------------------------------------------------
    // Async K/V load: cp.async.ca (16-byte / 8-half chunks, L1-cached).
    // OOB rows (kv_row >= S) are skipped; correctness is preserved because:
    //   - OOB K scores are masked to -INF after QK^T
    //   - OOB V rows have P=0 so contribute nothing to the output
    // The smem for skipped rows retains stale data from a previous tile, but
    // that data is never read (P=0 or score=-INF).
    // Pipeline structure (stage depth = 2):
    //   Prologue : issue tile-0 → buf-0, commit_group
    //   Iter i   : issue tile-(i+1) → buf-(1-cur), commit_group,
    //              wait_group(1) [cur ready], sync, compute, sync
    //   Last iter: wait_group(0)  [drain final group]
    // -------------------------------------------------------------------------
    constexpr int ASYNC_STEPS = Bc * HEAD_DIM / 8;  // 16‑byte (8‑bhalf) chunks per tile

    // Prologue: issue tile-0 into buf-0 before the loop
    {
        const int kv_base = 0;
        for (int i = tx; i < ASYNC_STEPS; i += TOTAL_THREADS) {
            const int r  = i / (HEAD_DIM / 8);
            const int c8 = (i % (HEAD_DIM / 8)) * 8;
            const int kv_row = kv_base + r;
            if (kv_row < S) {
                const long g = ((long)(b * H_kv + h_kv) * S + kv_row) * HEAD_DIM + c8;
                unsigned int sk = __cvta_generic_to_shared(&K_buf[0][r*KV_STRIDE + c8]);
                unsigned int sv = __cvta_generic_to_shared(&V_buf[0][r*KV_STRIDE + c8]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(sk), "l"((unsigned long long)(K + g)) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(sv), "l"((unsigned long long)(V + g)) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    for (int kv_tile = 0; kv_tile < num_kv_active; kv_tile++) {
        const int kv_start = kv_tile * Bc;
        const int cur      = kv_tile & 1;
        __nv_bfloat16* K_cur = K_buf[cur];
        __nv_bfloat16* V_cur = V_buf[cur];

        // Issue prefetch for next tile into the opposite buffer, then wait for cur.
        if (kv_tile + 1 < num_kv_active) {
            const int nxt_base = (kv_tile + 1) * Bc;
            const int nxt      = 1 - cur;
            for (int i = tx; i < ASYNC_STEPS; i += TOTAL_THREADS) {
                const int r  = i / (HEAD_DIM / 8);
                const int c8 = (i % (HEAD_DIM / 8)) * 8;
                const int kv_row = nxt_base + r;
                if (kv_row < S) {
                    const long g = ((long)(b * H_kv + h_kv) * S + kv_row) * HEAD_DIM + c8;
                    unsigned int sk = __cvta_generic_to_shared(&K_buf[nxt][r*KV_STRIDE + c8]);
                    unsigned int sv = __cvta_generic_to_shared(&V_buf[nxt][r*KV_STRIDE + c8]);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                                 :: "r"(sk), "l"((unsigned long long)(K + g)) : "memory");
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                                 :: "r"(sv), "l"((unsigned long long)(V + g)) : "memory");
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            // wait_group(1): all but the last committed group (next tile's) must be done
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        __syncthreads();

        // --- QK^T: warp w → S_smem[:, w*KV_COLS_PER_WARP] ---
        {// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-description
            const int kvc = warp_id * KV_COLS_PER_WARP;// id * 16
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag;// 寄存器抽象容器，映射到寄存器的逻辑切片，M N K (K是隐藏维度) , half,行主序列
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;// 列主序列
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc;
            wmma::fill_fragment(s_acc, 0.0f);
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 16) {
                wmma::load_matrix_sync(q_frag, Q_smem + d, KV_STRIDE);
                wmma::load_matrix_sync(k_frag, K_cur + kvc*KV_STRIDE + d, KV_STRIDE);
                wmma::mma_sync(s_acc, q_frag, k_frag, s_acc);// acc=q*k+acc
            }
            // Each warp stores to its non-overlapping column slice; stride = Bc
            wmma::store_matrix_sync(S_smem + kvc, s_acc, Bc, wmma::mem_row_major);
        }
        __syncthreads();

        // --- Scale + causal mask (128 threads, 8 elements each for Br*Bc=1024) ---
        for (int i = tx; i < Br * Bc; i += TOTAL_THREADS) {
            const int r = i / Bc, c = i % Bc;
            const int q_pos = q_row_start + r;
            const int kv_pos = kv_start + c;
            float val = Ss(r, c) * scale;
            if (kv_pos > q_pos || q_pos >= S || kv_pos >= S) val = -INFINITY;// 只有第一个才是因果mask，后面两个是序列长度裁剪
            Ss(r, c) = val;
        }
        __syncthreads();

        // --- Softmax: all 128 threads participate ---
        // TPR=8 threads per row, CPT=8 cols per thread.
        // shfl_xor(4,2,1) reduces within each 8-thread row-group without crossing rows:
        //   row r occupies warp lanes (r%4)*8 .. (r%4)*8+7; xor with 4,2,1 stays in [0..7] sub-group.
        {
            constexpr int TPR = TOTAL_THREADS / Br;   // 128/16 = 8 threads per row
            constexpr int CPT = Bc / TPR;             // 64/8  = 8 cols per thread
            static_assert(Bc % TPR == 0, "Bc must divide evenly across TOTAL_THREADS/Br");

            const int row = tx / TPR;
            const int col_start = (tx % TPR) * CPT;
            const bool valid_row = (q_row_start + row < S);

            // Phase 1: find new row-max (initialize with running max from previous tiles)
            const float old_max = s_row_max[row];
            float thread_max = old_max;
            if (valid_row) {
                for (int c = col_start; c < col_start + CPT; c++)
                    thread_max = fmaxf(thread_max, Ss(row, c));
            }
            // 3-step butterfly reduce within 8-thread row-group
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 4));
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 2));
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, 1));
            // all 8 threads now hold the same new_max for their row

            const float alpha = expf(old_max - thread_max);

            // Phase 2: compute exp(x - new_max) and sum
            float thread_sum = 0.0f;
            if (valid_row) {
                for (int c = col_start; c < col_start + CPT; c++) {
                    const float p = expf(Ss(row, c) - thread_max);
                    Ps(row, c) = __float2bfloat16(p);
                    thread_sum += p;
                }
            } else {
                for (int c = col_start; c < col_start + CPT; c++)
                    Ps(row, c) = __float2bfloat16(0.0f);
            }
            // Reduce sum within 8-thread row-group
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 4);
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 2);
            thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 1);

            // First thread in each row-group writes back softmax stats
            if (tx % TPR == 0) {
                s_row_max[row] = thread_max;
                s_row_sum[row] = s_row_sum[row] * alpha + thread_sum;
                s_alpha[row]   = valid_row ? alpha : 1.0f;
            }
        }
        __syncthreads();

        // --- Rescale O_smem by alpha (128 threads, all warps, 16 elements each) ---
        for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS)
            Os(i / HEAD_DIM, i % HEAD_DIM) *= s_alpha[i / HEAD_DIM];
        __syncthreads();

        // --- PV: warp w → warp_tmp[w][:][0..OUT_COLS_PER_WARP] ---
        // P_smem[16, 64] × V_smem[cur_buf][64, w*32:(w+1)*32] → O contribution [16, 32]
        {
            const int outc = warp_id * OUT_COLS_PER_WARP;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> v_frag[OUT_FRAGS];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_acc[OUT_FRAGS];
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++) wmma::fill_fragment(o_acc[f], 0.0f);

            // Inner loop over Bc=64 K-dimension in steps of 16 (4 iterations)
            #pragma unroll
            for (int k = 0; k < Bc; k += 16) {
                wmma::load_matrix_sync(p_frag, P_smem + k, Bc);
                #pragma unroll
                for (int f = 0; f < OUT_FRAGS; f++) {
                    wmma::load_matrix_sync(v_frag[f],
                                            V_cur + k*KV_STRIDE + outc + f*16,
                                            KV_STRIDE);
                    wmma::mma_sync(o_acc[f], p_frag, v_frag[f], o_acc[f]);
                }
            }
            // Store [16, 32] result into per-warp staging buffer
            #pragma unroll
            for (int f = 0; f < OUT_FRAGS; f++)
                wmma::store_matrix_sync(&Wt(warp_id, 0, f * 16),
                                        o_acc[f], OUT_COLS_PER_WARP, wmma::mem_row_major);
        }
        __syncthreads();

        // --- Accumulate warp_tmp into O_smem (128 threads, 16 elements each) ---
        for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {// 16*128, 128
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            Os(r, c) += Wt(c / OUT_COLS_PER_WARP, r, c % OUT_COLS_PER_WARP);
        }
        __syncthreads();
    } // end KV tiles

    // --- Normalize and write output ---
    for (int i = tx; i < Br * HEAD_DIM; i += TOTAL_THREADS) {
        const int r = i / HEAD_DIM, d = i % HEAD_DIM;
        const int q_row = q_row_start + r;
        if (q_row >= S) continue;
        const float inv = (s_row_sum[r] > 0.0f) ? 1.0f / s_row_sum[r] : 0.0f;
        O[((long)(b * H_q + h_q) * S + q_row) * HEAD_DIM + d] =
            __float2bfloat16(Os(r, d) * inv);
    }

    // Write LSE for backward pass Log-Sum-Exp（LSE）
    if (LSE && tx < Br) {
        const int q_row = q_row_start + tx;
        if (q_row < S)
            LSE[((long)(b * H_q + h_q) * S + q_row)] =
                s_row_max[tx] + logf(fmaxf(s_row_sum[tx], 1e-10f));
    }
}


// =============================================================================
// Launch wrapper
// =============================================================================
extern "C" void launch_flash_attention_prefill(
    const void* Q,
    const void* K,
    const void* V,
    void*       O,
    int B, int S, int H_q, int H_kv, int head_dim,
    cudaStream_t stream,
    float*       lse
) {
    constexpr int Br        = 16;   // Q rows per tile (WMMA M dim)
    constexpr int NUM_WARPS = 4;    // warps per block
    constexpr int Bc        = NUM_WARPS * 16;  // = 64: KV cols per tile
    const float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid((S + Br - 1) / Br, H_q, B);
    dim3 block(NUM_WARPS * WARP_SIZE);  // 128 threads

    // Request dynamic smem > 48 KB default limit (sm_120 supports up to 228 KB).
    // K/V are double-buffered → 2× the single-buffer size.
    constexpr int PAD_LAUNCHER = 8;
    constexpr int KV_STRIDE_L  = 128 + PAD_LAUNCHER;// TODO 修复 headdim
    const int smem_bytes =
        Br * KV_STRIDE_L * sizeof(__nv_bfloat16)               // Q_smem
      + 2 * Bc * KV_STRIDE_L * sizeof(__nv_bfloat16)           // K_buf[2]
      + 2 * Bc * KV_STRIDE_L * sizeof(__nv_bfloat16)           // V_buf[2]
      + Br * Bc              * sizeof(float)                   // S_smem
      + Br * Bc              * sizeof(__nv_bfloat16)            // P_smem
      + Br * 128             * sizeof(float)                   // O_smem
      + NUM_WARPS * Br * (128 / NUM_WARPS) * sizeof(float)    // warp_tmp
      + Br * 3               * sizeof(float);                  // s_row_max/sum/alpha

    static bool fa2_attr_set = false;
    auto kernel_fn = flash_attention_prefill_kernel<128, Br, Bc, NUM_WARPS>;
    if (!fa2_attr_set) {
        cudaError_t _e = cudaFuncSetAttribute(kernel_fn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        if (_e != cudaSuccess) {
            fprintf(stderr, "[FA2] FATAL: cudaFuncSetAttribute failed (%s) "
                    "requesting %d B smem.\n", cudaGetErrorString(_e), smem_bytes);
            fflush(stderr);
        } else {
            fa2_attr_set = true;
        }
    }

    kernel_fn<<<grid, block, smem_bytes, stream>>>(
        (const __nv_bfloat16*)Q, 
        (const __nv_bfloat16*)K, 
        (const __nv_bfloat16*)V, 
        (__nv_bfloat16*)O,
         lse, B, S, H_q, H_kv, scale
    );
}

// Explicit instantiation
template __global__ void
flash_attention_prefill_kernel<128, 16, 64, 4>(
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    const __nv_bfloat16*,
    __nv_bfloat16*,
    float*,
    int, int, int, int, float
);
