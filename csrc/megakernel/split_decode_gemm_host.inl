// Decode launch entry
#include "split_decode_gemm_host_utils.inl"
extern "C" void launch_split_decode_gemm(
    const int* input_token_id,    // device [1]
    int* output_token_id,         // device [1]
    const void* embed_weight,     // bf16
    const PrefillLayerWeights* layer_weights,  // host array (device pointers)
    const void* const* qkv_weight_packed_ptrs,     // host [num_layers], each [Q+KV+KV, H]
    const void* const* gateup_weight_packed_ptrs,  // host [num_layers], each [2*I, H]
    const void* const* down_weight_ptrs,           // host [num_layers], each [H, I]
    const void* const* q_w4_packed_ptrs,
    const void* const* q_w4_scales_ptrs,
    const void* const* q_w4_codebook_ptrs,
    const void* const* k_w4_packed_ptrs,
    const void* const* k_w4_scales_ptrs,
    const void* const* k_w4_codebook_ptrs,
    const void* const* v_w4_packed_ptrs,
    const void* const* v_w4_scales_ptrs,
    const void* const* v_w4_codebook_ptrs,
    const void* const* o_w4_packed_ptrs,
    const void* const* o_w4_scales_ptrs,
    const void* const* o_w4_codebook_ptrs,
    const void* const* gateup_w4_packed_ptrs,      // host [num_layers], each uint8 packed [2*I*H/2]
    const void* const* gateup_w4_scales_ptrs,      // host [num_layers], float [2*I*H/64]
    const void* const* gateup_w4_codebook_ptrs,    // host [num_layers], float [16]
    const void* const* down_w4_packed_ptrs,        // host [num_layers], each uint8 packed [H*I/2]
    const void* const* down_w4_scales_ptrs,        // host [num_layers], float [H*I/64]
    const void* const* down_w4_codebook_ptrs,      // host [num_layers], float [16]
    const int* kv_block_table,                     // optional [num_kv_blocks], logical->physical
    int kv_block_size,                             // token block size for paged KV
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,        // bf16 [max_seq, head_dim]
    const void* sin_table,        // bf16 [max_seq, head_dim]
    void* k_cache,                // bf16
    void* v_cache,                // bf16
    void* k_cache_fp8,            // optional fp8 (e4m3)
    void* v_cache_fp8,            // optional fp8 (e4m3)
    void* k_scale_cache,          // optional fp32 scale
    void* v_scale_cache,          // optional fp32 scale
    int kv_fp8_enabled,           // 0/1
    void* split_attn_partial_m,   // [split_max_chunks, num_q_heads] float
    void* split_attn_partial_s,   // [split_max_chunks, num_q_heads] float
    void* split_attn_partial_out, // [split_max_chunks, num_q_heads, head_dim] float
    int split_attn_chunk_size,
    int split_attn_max_chunks,
    // Scratch: FP32 activation chain + BF16 GEMM I/O buffers
    void* hidden_f32,             // [hidden] float
    void* residual_f32,           // [hidden] float
    void* normalized_bf16,        // [hidden]
    void* qkv_proj_bf16,          // [q_size + kv_size + kv_size]
    void* attn_out_bf16,          // [q_size]
    void* o_proj_out_bf16,        // [hidden]
    void* mlp_norm_bf16,          // [hidden]
    void* gateup_out_bf16,        // [2*intermediate]
    void* mlp_intermediate_bf16,  // [intermediate]
    void* down_out_bf16,          // [hidden]
    void* final_hidden_bf16,      // [hidden]
    void* block_max_vals,         // [PREFILL_LM_NUM_BLOCKS] float
    void* block_max_idxs,         // [PREFILL_LM_NUM_BLOCKS] int
    int num_layers,
    const int* position_ptr,      // device scalar (absolute position)
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cublasLtHandle_t cublaslt_handle,
    void* cublaslt_workspace,
    size_t cublaslt_workspace_bytes,
    cudaStream_t stream
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    constexpr int QKV_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    constexpr int GATEUP_ROWS = INTERMEDIATE_SIZE + INTERMEDIATE_SIZE;
    static SplitLtMatmulPlan qkv_lt_plan;
    static SplitLtMatmulPlan o_lt_plan;
    static SplitLtMatmulPlan gateup_lt_plan;
    static SplitLtMatmulPlan down_lt_plan;
    constexpr int kMaxDebugEvents = 2048;
    SplitDebugEventRec debug_events[kMaxDebugEvents];
    int debug_event_count = 0;
    bool debug_stage_detail = _split_debug_take_ticket();
    bool debug_stage_avg = _split_debug_avg_enabled();
    bool debug_stage_avg_exact = _split_debug_avg_exact();
    int debug_stage_avg_stride = _split_debug_avg_stride();
    if (debug_stage_avg_stride < 1) debug_stage_avg_stride = 1;
    bool debug_stage_avg_sample = false;
    if (debug_stage_avg) {
        auto& agg = _split_debug_agg();
        long long tok_idx = agg.tokens;
        agg.tokens += 1;
        debug_stage_avg_sample =
            debug_stage_avg_exact ||
            debug_stage_avg_stride <= 1 ||
            (tok_idx % static_cast<long long>(debug_stage_avg_stride) == 0);
    }
    bool debug_stage = (debug_stage_detail || debug_stage_avg_sample);
    auto dbg_begin = [&](int stage, int layer_id = -1) -> int {
        if (!debug_stage) return -1;
        if (debug_event_count >= kMaxDebugEvents) return -1;
        int idx = debug_event_count++;
        debug_events[idx].stage = stage;
        debug_events[idx].layer = layer_id;
        if (cudaEventCreateWithFlags(&debug_events[idx].start, cudaEventDefault) != cudaSuccess) {
            debug_events[idx].start = nullptr;
            debug_events[idx].stop = nullptr;
            return -1;
        }
        if (cudaEventCreateWithFlags(&debug_events[idx].stop, cudaEventDefault) != cudaSuccess) {
            cudaEventDestroy(debug_events[idx].start);
            debug_events[idx].start = nullptr;
            debug_events[idx].stop = nullptr;
            return -1;
        }
        cudaEventRecord(debug_events[idx].start, stream);
        return idx;
    };
    auto dbg_end = [&](int idx) {
        if (idx < 0 || idx >= debug_event_count) return;
        if (debug_events[idx].stop == nullptr) return;
        cudaEventRecord(debug_events[idx].stop, stream);
    };

    if (!_cublas_check(cublasSetStream(cublas_handle, stream), "cublasSetStream")) return;
    if (qkv_weight_packed_ptrs == nullptr || gateup_weight_packed_ptrs == nullptr || down_weight_ptrs == nullptr) {
        printf("[MEGAQWEN_SPLIT] packed weight ptrs are null\n");
        return;
    }
    if (kv_fp8_enabled &&
        (k_cache_fp8 == nullptr || v_cache_fp8 == nullptr ||
         k_scale_cache == nullptr || v_scale_cache == nullptr)) {
        static bool warned_kv_fp8_ptr = false;
        if (!warned_kv_fp8_ptr) {
            warned_kv_fp8_ptr = true;
            std::printf("[MEGAQWEN_SPLIT] kv_fp8 requested but fp8 cache/scale buffers are null; fallback to bf16 KV.\n");
        }
        kv_fp8_enabled = 0;
    }
    int k_fp8_enabled = (kv_fp8_enabled && _split_k_fp8_enabled()) ? 1 : 0;
    int kv_fp8_only = (kv_fp8_enabled && _split_kv_fp8_only_enabled()) ? 1 : 0;
    if (!k_fp8_enabled) kv_fp8_only = 0;
    const char* kv_mode_desc =
        kv_fp8_enabled
            ? (k_fp8_enabled ? "fp8_spec" : "kbf16_vfp8")
            : "bf16_spec";

    bool debug_paged_kv = _split_debug_paged_kv_take_ticket();
    bool debug_flash_decode = _split_debug_flash_decode_take_ticket();
    bool debug_ffn_w4_fine = _split_debug_ffn_w4_fine_take_ticket();
    int debug_flash_clock_khz = 0;
    int debug_ffn_w4_clock_khz = 0;
    if (debug_flash_decode) {
        static int s_flash_clock_khz = -1;
        if (s_flash_clock_khz < 0) {
            int dev = 0;
            int khz = 0;
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) == cudaSuccess &&
                khz > 0) {
                s_flash_clock_khz = khz;
            } else {
                s_flash_clock_khz = 0;
            }
        }
        debug_flash_clock_khz = s_flash_clock_khz;
    }
    if (debug_ffn_w4_fine) {
        static int s_ffn_w4_clock_khz = -1;
        if (s_ffn_w4_clock_khz < 0) {
            int dev = 0;
            int khz = 0;
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) == cudaSuccess &&
                khz > 0) {
                s_ffn_w4_clock_khz = khz;
            } else {
                s_ffn_w4_clock_khz = 0;
            }
        }
        debug_ffn_w4_clock_khz = s_ffn_w4_clock_khz;
    }
    int flash_parts = _split_flash_parts();
    if (flash_parts < 1) flash_parts = 1;
    int debug_position_host = -1;
    if ((debug_paged_kv || debug_flash_decode) && position_ptr != nullptr) {
        cudaMemcpyAsync(&debug_position_host, position_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    SplitFlashDecodeDebugRec* flash_debug_dev = nullptr;
    SplitFlashDecodeDebugRec* flash_phase_debug_dev = nullptr;
    SplitFlashDecodeDebugRec* flash_phase_debug_host = nullptr;
    int flash_phase_debug_cap = 0;
    float* flash_part_m_dev = nullptr;
    float* flash_part_s_dev = nullptr;
    float* flash_part_out_dev = nullptr;
    int flash_part_cap = 0;
    SplitFfnW4FineDebugRec* ffn_w4_fine_dev = nullptr;
    if (debug_flash_decode) {
        static SplitFlashDecodeDebugRec* s_flash_debug_dev = nullptr;
        static int s_flash_debug_cap = 0;
        static SplitFlashDecodeDebugRec* s_flash_phase_debug_dev = nullptr;
        static SplitFlashDecodeDebugRec* s_flash_phase_debug_host = nullptr;
        static int s_flash_phase_debug_cap = 0;
        if (s_flash_debug_cap < NUM_Q_HEADS) {
            if (s_flash_debug_dev != nullptr) {
                cudaFree(s_flash_debug_dev);
                s_flash_debug_dev = nullptr;
                s_flash_debug_cap = 0;
            }
            if (cudaMalloc((void**)&s_flash_debug_dev, sizeof(SplitFlashDecodeDebugRec) * NUM_Q_HEADS) == cudaSuccess) {
                s_flash_debug_cap = NUM_Q_HEADS;
            }
        }
        flash_debug_dev = s_flash_debug_dev;
        flash_phase_debug_dev = s_flash_phase_debug_dev;
        flash_phase_debug_host = s_flash_phase_debug_host;
        flash_phase_debug_cap = s_flash_phase_debug_cap;
        if (flash_debug_dev == nullptr) {
            std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE debug buffer alloc failed, disable debug output.\n");
            debug_flash_decode = false;
        }
        int need_phase_slots = NUM_Q_HEADS * split_attn_max_chunks * ((flash_parts > 0) ? flash_parts : 1);
        if (debug_flash_decode && need_phase_slots > 0 && flash_phase_debug_cap < need_phase_slots) {
            if (s_flash_phase_debug_dev != nullptr) {
                cudaFree(s_flash_phase_debug_dev);
                s_flash_phase_debug_dev = nullptr;
            }
            if (s_flash_phase_debug_host != nullptr) {
                std::free(s_flash_phase_debug_host);
                s_flash_phase_debug_host = nullptr;
            }
            s_flash_phase_debug_cap = 0;
            if (cudaMalloc((void**)&s_flash_phase_debug_dev, sizeof(SplitFlashDecodeDebugRec) * need_phase_slots) == cudaSuccess) {
                s_flash_phase_debug_host = (SplitFlashDecodeDebugRec*)std::malloc(sizeof(SplitFlashDecodeDebugRec) * need_phase_slots);
                if (s_flash_phase_debug_host != nullptr) {
                    s_flash_phase_debug_cap = need_phase_slots;
                } else {
                    cudaFree(s_flash_phase_debug_dev);
                    s_flash_phase_debug_dev = nullptr;
                }
            }
            flash_phase_debug_dev = s_flash_phase_debug_dev;
            flash_phase_debug_host = s_flash_phase_debug_host;
            flash_phase_debug_cap = s_flash_phase_debug_cap;
            if (flash_phase_debug_dev == nullptr || flash_phase_debug_host == nullptr) {
                std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE phase12 debug buffer alloc failed, fallback to light debug.\n");
            }
        }
        flash_phase_debug_dev = s_flash_phase_debug_dev;
        flash_phase_debug_host = s_flash_phase_debug_host;
        flash_phase_debug_cap = s_flash_phase_debug_cap;
    }
    if (debug_ffn_w4_fine) {
        static SplitFfnW4FineDebugRec* s_ffn_w4_fine_dev = nullptr;
        if (s_ffn_w4_fine_dev == nullptr) {
            if (cudaMalloc((void**)&s_ffn_w4_fine_dev, sizeof(SplitFfnW4FineDebugRec)) != cudaSuccess) {
                s_ffn_w4_fine_dev = nullptr;
            }
        }
        ffn_w4_fine_dev = s_ffn_w4_fine_dev;
        if (ffn_w4_fine_dev == nullptr) {
            std::printf("[MEGAQWEN_DEBUG] FFN_W4_FINE debug buffer alloc failed, disable fine output.\n");
            debug_ffn_w4_fine = false;
        } else {
            cudaMemsetAsync(ffn_w4_fine_dev, 0, sizeof(SplitFfnW4FineDebugRec), stream);
        }
    }
    if (flash_parts > 1) {
        static float* s_flash_part_m_dev = nullptr;
        static float* s_flash_part_s_dev = nullptr;
        static float* s_flash_part_out_dev = nullptr;
        static int s_flash_part_cap = 0;
        int need_part_entries = NUM_Q_HEADS * split_attn_max_chunks * flash_parts;
        if (need_part_entries > 0 && s_flash_part_cap < need_part_entries) {
            if (s_flash_part_m_dev != nullptr) { cudaFree(s_flash_part_m_dev); s_flash_part_m_dev = nullptr; }
            if (s_flash_part_s_dev != nullptr) { cudaFree(s_flash_part_s_dev); s_flash_part_s_dev = nullptr; }
            if (s_flash_part_out_dev != nullptr) { cudaFree(s_flash_part_out_dev); s_flash_part_out_dev = nullptr; }
            s_flash_part_cap = 0;
            bool ok = true;
            ok = ok && (cudaMalloc((void**)&s_flash_part_m_dev, sizeof(float) * need_part_entries) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&s_flash_part_s_dev, sizeof(float) * need_part_entries) == cudaSuccess);
            ok = ok && (cudaMalloc((void**)&s_flash_part_out_dev, sizeof(float) * need_part_entries * HEAD_DIM) == cudaSuccess);
            if (ok) {
                s_flash_part_cap = need_part_entries;
            } else {
                if (s_flash_part_m_dev != nullptr) { cudaFree(s_flash_part_m_dev); s_flash_part_m_dev = nullptr; }
                if (s_flash_part_s_dev != nullptr) { cudaFree(s_flash_part_s_dev); s_flash_part_s_dev = nullptr; }
                if (s_flash_part_out_dev != nullptr) { cudaFree(s_flash_part_out_dev); s_flash_part_out_dev = nullptr; }
            }
        }
        flash_part_m_dev = s_flash_part_m_dev;
        flash_part_s_dev = s_flash_part_s_dev;
        flash_part_out_dev = s_flash_part_out_dev;
        flash_part_cap = s_flash_part_cap;
        if (flash_part_m_dev == nullptr || flash_part_s_dev == nullptr || flash_part_out_dev == nullptr) {
            flash_parts = 1;
            if (debug_flash_decode) {
                std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE partitions buffer alloc failed, fallback parts=1.\n");
            }
        }
    }
    if (debug_paged_kv) {
        int position_host = debug_position_host;
        int cache_len = position_host + 1;
        if (cache_len < 0) cache_len = 0;
        if (cache_len > max_seq_len) cache_len = max_seq_len;
        int logical_block = -1;
        int physical_block = -1;
        int physical_pos = -1;
        if (kv_block_table != nullptr && kv_block_size > 0 && position_host >= 0) {
            logical_block = position_host / kv_block_size;
            if (logical_block >= 0) {
                cudaMemcpyAsync(&physical_block, kv_block_table + logical_block, sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                int in_block = position_host - logical_block * kv_block_size;
                physical_pos = physical_block * kv_block_size + in_block;
            }
        } else {
            physical_pos = position_host;
        }
        std::printf(
            "[MEGAQWEN_DEBUG] PAGED_KV token pos=%d cache_len=%d block_size=%d logical_block=%d physical_block=%d physical_pos=%d\n",
            position_host, cache_len, kv_block_size, logical_block, physical_block, physical_pos);

        if (kv_block_table != nullptr && kv_block_size > 0) {
            static bool dumped_table = false;
            if (!dumped_table) {
                dumped_table = true;
                constexpr int kDumpN = 16;
                int table_host[kDumpN];
                int dump_n = (max_seq_len + kv_block_size - 1) / kv_block_size;
                if (dump_n > kDumpN) dump_n = kDumpN;
                if (dump_n < 0) dump_n = 0;
                if (dump_n > 0) {
                    cudaMemcpyAsync(table_host, kv_block_table, sizeof(int) * dump_n, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    std::printf("[MEGAQWEN_DEBUG] PAGED_KV block_table_head:");
                    for (int i = 0; i < dump_n; i++) std::printf(" %d->%d", i, table_host[i]);
                    std::printf("\n");
                }
            }
        }
    }

    const int* kv_block_table_attn = kv_block_table;
    bool kv_identity_fastpath = false;
    if (kv_block_table != nullptr && kv_block_size > 0) {
        static const int* s_kv_table_ptr = nullptr;
        static int s_kv_table_identity = -1;  // -1 unknown, 0 non-identity, 1 identity
        if (s_kv_table_ptr != kv_block_table || s_kv_table_identity < 0) {
            s_kv_table_identity = 0;
            int kv_blocks = (max_seq_len + kv_block_size - 1) / kv_block_size;
            if (kv_blocks > 0) {
                int* host_blocks = (int*)std::malloc(sizeof(int) * kv_blocks);
                if (host_blocks != nullptr) {
                    cudaMemcpyAsync(host_blocks, kv_block_table, sizeof(int) * kv_blocks, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    s_kv_table_identity = 1;
                    for (int i = 0; i < kv_blocks; i++) {
                        if (host_blocks[i] != i) {
                            s_kv_table_identity = 0;
                            break;
                        }
                    }
                    std::free(host_blocks);
                }
            }
            s_kv_table_ptr = kv_block_table;
        }
        if (s_kv_table_identity == 1) {
            kv_block_table_attn = nullptr;
            kv_identity_fastpath = true;
        }
    }

    // 0) Embed lookup -> hidden_f32
    int dbg_embed = dbg_begin(SPLIT_STAGE_EMBED);
    split_embed_to_hidden_f32_kernel<<<1, 256, 0, stream>>>(
        input_token_id,
        (const __nv_bfloat16*)embed_weight,
        (float*)hidden_f32
    );
    dbg_end(dbg_embed);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;
    int kv_scale_layer_stride = NUM_KV_HEADS;

    for (int layer = 0; layer < num_layers; layer++) {
        const PrefillLayerWeights& w = layer_weights[layer];
        const __nv_bfloat16* qkv_weight = (const __nv_bfloat16*)qkv_weight_packed_ptrs[layer];
        const __nv_bfloat16* gateup_weight = (const __nv_bfloat16*)gateup_weight_packed_ptrs[layer];
        const __nv_bfloat16* down_weight = (const __nv_bfloat16*)down_weight_ptrs[layer];
        const uint8_t* q_w4_packed = q_w4_packed_ptrs ? (const uint8_t*)q_w4_packed_ptrs[layer] : nullptr;
        const float* q_w4_scales = q_w4_scales_ptrs ? (const float*)q_w4_scales_ptrs[layer] : nullptr;
        const float* q_w4_codebook = q_w4_codebook_ptrs ? (const float*)q_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* k_w4_packed = k_w4_packed_ptrs ? (const uint8_t*)k_w4_packed_ptrs[layer] : nullptr;
        const float* k_w4_scales = k_w4_scales_ptrs ? (const float*)k_w4_scales_ptrs[layer] : nullptr;
        const float* k_w4_codebook = k_w4_codebook_ptrs ? (const float*)k_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* v_w4_packed = v_w4_packed_ptrs ? (const uint8_t*)v_w4_packed_ptrs[layer] : nullptr;
        const float* v_w4_scales = v_w4_scales_ptrs ? (const float*)v_w4_scales_ptrs[layer] : nullptr;
        const float* v_w4_codebook = v_w4_codebook_ptrs ? (const float*)v_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* o_w4_packed = o_w4_packed_ptrs ? (const uint8_t*)o_w4_packed_ptrs[layer] : nullptr;
        const float* o_w4_scales = o_w4_scales_ptrs ? (const float*)o_w4_scales_ptrs[layer] : nullptr;
        const float* o_w4_codebook = o_w4_codebook_ptrs ? (const float*)o_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* gateup_w4_packed = gateup_w4_packed_ptrs ? (const uint8_t*)gateup_w4_packed_ptrs[layer] : nullptr;
        const float* gateup_w4_scales = gateup_w4_scales_ptrs ? (const float*)gateup_w4_scales_ptrs[layer] : nullptr;
        const float* gateup_w4_codebook = gateup_w4_codebook_ptrs ? (const float*)gateup_w4_codebook_ptrs[layer] : nullptr;
        const uint8_t* down_w4_packed = down_w4_packed_ptrs ? (const uint8_t*)down_w4_packed_ptrs[layer] : nullptr;
        const float* down_w4_scales = down_w4_scales_ptrs ? (const float*)down_w4_scales_ptrs[layer] : nullptr;
        const float* down_w4_codebook = down_w4_codebook_ptrs ? (const float*)down_w4_codebook_ptrs[layer] : nullptr;
        bool qkv_w4 = _split_qkv_w4_enabled() &&
                      q_w4_packed != nullptr && q_w4_scales != nullptr && q_w4_codebook != nullptr &&
                      k_w4_packed != nullptr && k_w4_scales != nullptr && k_w4_codebook != nullptr &&
                      v_w4_packed != nullptr && v_w4_scales != nullptr && v_w4_codebook != nullptr;
        bool o_w4 = _split_o_w4_enabled() &&
                    o_w4_packed != nullptr && o_w4_scales != nullptr && o_w4_codebook != nullptr;
        bool ffn_w4 = _split_ffn_w4_enabled() &&
                      gateup_w4_packed != nullptr && gateup_w4_scales != nullptr && gateup_w4_codebook != nullptr &&
                      down_w4_packed != nullptr && down_w4_scales != nullptr && down_w4_codebook != nullptr;

        __nv_bfloat16* layer_k_cache = (__nv_bfloat16*)k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = (__nv_bfloat16*)v_cache + layer * kv_cache_layer_stride;
        __nv_fp8_e4m3* layer_k_cache_fp8 =
            (kv_fp8_enabled && k_cache_fp8 != nullptr)
                ? ((__nv_fp8_e4m3*)k_cache_fp8 + layer * kv_cache_layer_stride)
                : nullptr;
        __nv_fp8_e4m3* layer_v_cache_fp8 =
            (kv_fp8_enabled && v_cache_fp8 != nullptr)
                ? ((__nv_fp8_e4m3*)v_cache_fp8 + layer * kv_cache_layer_stride)
                : nullptr;
        float* layer_k_scale_cache =
            (kv_fp8_enabled && k_scale_cache != nullptr)
                ? ((float*)k_scale_cache + layer * kv_scale_layer_stride)
                : nullptr;
        float* layer_v_scale_cache =
            (kv_fp8_enabled && v_scale_cache != nullptr)
                ? ((float*)v_scale_cache + layer * kv_scale_layer_stride)
                : nullptr;

        // 1) RMSNorm (FP32 chain, BF16 output for GEMM)
        int dbg_rms1 = dbg_begin(SPLIT_STAGE_RMS1, layer);
        split_rmsnorm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
            (const float*)hidden_f32,
            (const __nv_bfloat16*)w.input_layernorm_weight,
            (__nv_bfloat16*)normalized_bf16,
            (float*)residual_f32
        );
        dbg_end(dbg_rms1);

        // 2) QKV projection (merged): [Q+KV+KV, H] x [H, 1] -> [Q+KV+KV, 1]
        int dbg_qkv = dbg_begin(SPLIT_STAGE_QKV_GEMM, layer);
        bool qkv_done = false;
        bool qkv_lt_done = false;
        bool qkv_gemmex_done = false;
        bool qkv_gemv_done = false;
        bool qkv_use_gemv = _split_qkv_use_gemv();
        bool qkv_try_lt = _split_qkv_try_cublaslt();
        if (qkv_w4) {
            split_w4_qkv_matvec_bf16_out_kernel<<<QKV_ROWS, 256, 0, stream>>>(
                q_w4_packed,
                q_w4_scales,
                q_w4_codebook,
                k_w4_packed,
                k_w4_scales,
                k_w4_codebook,
                v_w4_packed,
                v_w4_scales,
                v_w4_codebook,
                (const __nv_bfloat16*)normalized_bf16,
                (__nv_bfloat16*)qkv_proj_bf16,
                HIDDEN_SIZE
            );
            qkv_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.qkv_w4 += 1;
            }
        } else if (!qkv_use_gemv && qkv_try_lt) {
            qkv_done = _split_lt_matmul(
                cublaslt_handle,
                qkv_lt_plan,
                QKV_ROWS,
                HIDDEN_SIZE,
                qkv_weight,
                (const __nv_bfloat16*)normalized_bf16,
                (__nv_bfloat16*)qkv_proj_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            qkv_lt_done = qkv_done;
        }
        if (!qkv_done) {
            if (qkv_use_gemv) {
                qkv_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    qkv_weight,
                    QKV_ROWS,
                    HIDDEN_SIZE,
                    (const __nv_bfloat16*)normalized_bf16,
                    (__nv_bfloat16*)qkv_proj_bf16,
                    "qkv_gemv"
                );
                qkv_done = qkv_gemv_done;
            }
            if (!qkv_done) {// attn后端为2时发生出错的地方
                qkv_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    QKV_ROWS, 1, HIDDEN_SIZE,
                    &alpha,
                    qkv_weight, CUDA_R_16BF, HIDDEN_SIZE,
                    normalized_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    &beta,
                    qkv_proj_bf16, CUDA_R_16BF, QKV_ROWS,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "qkv_gemm")) return;
            }
        }
        if (debug_stage_avg && !qkv_w4) {
            auto& agg = _split_debug_agg();
            if (!qkv_use_gemv && qkv_try_lt) {
                if (qkv_lt_done) agg.qkv_lt_ok += 1;
                else agg.qkv_lt_fb += 1;
            }
            if (qkv_gemv_done) agg.qkv_gemv += 1;
            else if (qkv_gemmex_done) agg.qkv_gemmex += 1;
        }
        dbg_end(dbg_qkv);

        __nv_bfloat16* q_proj_bf16 = (__nv_bfloat16*)qkv_proj_bf16;
        __nv_bfloat16* k_proj_bf16 = q_proj_bf16 + Q_SIZE;
        __nv_bfloat16* v_proj_bf16 = k_proj_bf16 + KV_SIZE;

        // 3) QK norm + RoPE + KV cache (position dynamic via position_ptr)
        int dbg_qk = dbg_begin(SPLIT_STAGE_QK_ROPE_CACHE, layer);
        int heads = (NUM_Q_HEADS > NUM_KV_HEADS) ? NUM_Q_HEADS : NUM_KV_HEADS;
        int flash_fp8_tc_qk = _split_flash_fp8_tc_qk_enabled() ? 1 : 0;
        decode_qk_norm_rope_cache_kernel<<<heads, 128, 0, stream>>>(
            (__nv_bfloat16*)q_proj_bf16,
            (__nv_bfloat16*)k_proj_bf16,
            (const __nv_bfloat16*)v_proj_bf16,
            (const __nv_bfloat16*)w.q_norm_weight,
            (const __nv_bfloat16*)w.k_norm_weight,
            (const __nv_bfloat16*)cos_table,
            (const __nv_bfloat16*)sin_table,
            layer_k_cache,
            layer_v_cache,
            layer_k_cache_fp8,
            layer_v_cache_fp8,
            layer_k_scale_cache,
            layer_v_scale_cache,
            kv_fp8_enabled ? 1 : 0,
            kv_fp8_only,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            max_seq_len,
            kv_block_table,
            kv_block_size,
            position_ptr
        );
        dbg_end(dbg_qk);

        // 4) Decode attention over cache (BF16 in/out)
        int dbg_attn = dbg_begin(SPLIT_STAGE_ATTN, layer);
        int attn_impl = _split_attn_impl();
        int attn_warps = (attn_impl == 3) ? _split_flash_attn_warps_per_head()
                                          : _split_attn_warps_per_head();
        bool flash_gqa_share =
            _split_flash_gqa_share_enabled() &&
            (NUM_Q_HEADS == 2 * NUM_KV_HEADS);
        if (attn_impl == 3 && flash_gqa_share && !_split_flash_warps_user_override()) {
            // GQA2 kernel holds two q-head accumulators per thread; use fewer warps
            // by default to reduce register pressure / occupancy collapse on AD102.
            if (attn_warps > 3) attn_warps = 3;
        }
        if (attn_impl == 3) {// 包含3种变体：V3-GQA、V3-S0、V3-Combine, 以及FP8、TC 旋钮.
            bool flash_use_split_reduce = // 置零以使能 decode_attention_cache_flash_decode_kernel_t
                (split_attn_chunk_size > 0 && split_attn_max_chunks > 0 &&
                 split_attn_partial_m != nullptr && split_attn_partial_s != nullptr &&
                 split_attn_partial_out != nullptr);
            if (flash_use_split_reduce) {
                int parts = flash_parts;
                if (parts < 1) parts = 1;
                bool use_gqa_phase1 = flash_gqa_share;
                bool use_gqa_simt_fast = (use_gqa_phase1 && _split_flash_gqa_mode() == 1);
                int blocks = (use_gqa_phase1 ? NUM_KV_HEADS : NUM_Q_HEADS) * split_attn_max_chunks * parts;
                int threads = attn_warps * WARP_SIZE;
                constexpr size_t kFlashGqaTcSharedBytes = size_t(8) * size_t(16) * size_t(16) *
                                                          size_t(2 + 2 + 4);
                size_t flash_gqa_tc_shared_bytes =
                    (use_gqa_phase1 && !use_gqa_simt_fast && kv_fp8_enabled && k_fp8_enabled && flash_fp8_tc_qk)
                        ? kFlashGqaTcSharedBytes
                        : 0;
                SplitFlashDecodeDebugRec* gqa_phase_debug_ptr =
                    (debug_flash_decode && use_gqa_phase1 && flash_phase_debug_dev != nullptr &&
                     flash_phase_debug_cap >= (NUM_Q_HEADS * split_attn_max_chunks * parts))
                        ? flash_phase_debug_dev
                        : nullptr;
                if (kv_fp8_enabled) {
                    if (use_gqa_phase1) {
                        if (use_gqa_simt_fast || !k_fp8_enabled) { // share 且 tc
                            if (gqa_phase_debug_ptr != nullptr) {
                                if (k_fp8_enabled) {//kV_FP8 + 关闭 TC + 开启调试模式
                                    decode_attention_cache_flash_phase1_gqa2_kernel_t<true, true, false, true><<<blocks, threads, 0, stream>>>(
                                        (const __nv_bfloat16*)q_proj_bf16,
                                        (const __nv_bfloat16*)layer_k_cache,
                                        (const __nv_bfloat16*)layer_v_cache,
                                        (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                        (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                        (const float*)layer_k_scale_cache,
                                        (const float*)layer_v_scale_cache,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_m_dev
                                            : (float*)split_attn_partial_m,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_s_dev
                                            : (float*)split_attn_partial_s,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_out_dev
                                            : (float*)split_attn_partial_out,
                                        NUM_Q_HEADS,
                                        NUM_KV_HEADS,
                                        HEAD_DIM,
                                        max_seq_len,
                                        attn_scale,
                                        attn_warps,
                                        parts,
                                        split_attn_chunk_size,
                                        split_attn_max_chunks,
                                        kv_block_table_attn,
                                        kv_block_size,
                                        position_ptr,
                                        0,
                                        gqa_phase_debug_ptr
                                    );
                                } else {// V_FP8 + 调试模式
                                    decode_attention_cache_flash_phase1_gqa2_kernel_t<false, true, false, true><<<blocks, threads, 0, stream>>>( 
                                    (const __nv_bfloat16*)q_proj_bf16,
                                    (const __nv_bfloat16*)layer_k_cache,
                                    (const __nv_bfloat16*)layer_v_cache,
                                    (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                    (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                    (const float*)layer_k_scale_cache,
                                    (const float*)layer_v_scale_cache,
                                    (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                        ? (float*)flash_part_m_dev
                                        : (float*)split_attn_partial_m,
                                    (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                        ? (float*)flash_part_s_dev
                                        : (float*)split_attn_partial_s,
                                    (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                        ? (float*)flash_part_out_dev
                                        : (float*)split_attn_partial_out,
                                    NUM_Q_HEADS,
                                    NUM_KV_HEADS,
                                    HEAD_DIM,
                                    max_seq_len,
                                    attn_scale,
                                    attn_warps,
                                    parts,
                                    split_attn_chunk_size,
                                    split_attn_max_chunks,
                                    kv_block_table_attn,
                                    kv_block_size,
                                    position_ptr,
                                    0,
                                    gqa_phase_debug_ptr
                                );
                                }
                            } else {
                                if (k_fp8_enabled) {// kV_FP8 + 关闭 TC + 关闭调试模式
                                    decode_attention_cache_flash_phase1_gqa2_kernel_t<true, true, false, false><<<blocks, threads, 0, stream>>>(
                                        (const __nv_bfloat16*)q_proj_bf16,
                                        (const __nv_bfloat16*)layer_k_cache,
                                        (const __nv_bfloat16*)layer_v_cache,
                                        (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                        (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                        (const float*)layer_k_scale_cache,
                                        (const float*)layer_v_scale_cache,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_m_dev
                                            : (float*)split_attn_partial_m,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_s_dev
                                            : (float*)split_attn_partial_s,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_out_dev
                                            : (float*)split_attn_partial_out,
                                        NUM_Q_HEADS,
                                        NUM_KV_HEADS,
                                        HEAD_DIM,
                                        max_seq_len,
                                        attn_scale,
                                        attn_warps,
                                        parts,
                                        split_attn_chunk_size,
                                        split_attn_max_chunks,
                                        kv_block_table_attn,
                                        kv_block_size,
                                        position_ptr,
                                        0,
                                        nullptr
                                    );
                                } else {// V_FP8 + 关闭 TC + 关闭调试模式
                                    decode_attention_cache_flash_phase1_gqa2_kernel_t<false, true, false, false><<<blocks, threads, 0, stream>>>(
                                        (const __nv_bfloat16*)q_proj_bf16,
                                        (const __nv_bfloat16*)layer_k_cache,
                                        (const __nv_bfloat16*)layer_v_cache,
                                        (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                        (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                        (const float*)layer_k_scale_cache,
                                        (const float*)layer_v_scale_cache,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_m_dev
                                            : (float*)split_attn_partial_m,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_s_dev
                                            : (float*)split_attn_partial_s,
                                        (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                            ? (float*)flash_part_out_dev
                                            : (float*)split_attn_partial_out,
                                        NUM_Q_HEADS,
                                        NUM_KV_HEADS,
                                        HEAD_DIM,
                                        max_seq_len,
                                        attn_scale,
                                        attn_warps,
                                        parts,
                                        split_attn_chunk_size,
                                        split_attn_max_chunks,
                                        kv_block_table_attn,
                                        kv_block_size,
                                        position_ptr,
                                        0,
                                        nullptr
                                    );
                                }
                            }
                        } else {
                            if (gqa_phase_debug_ptr != nullptr) {// kV_FP8 + 开启 TC + 开启调试模式
                                decode_attention_cache_flash_phase1_gqa2_kernel_t<true, true, true, true><<<blocks, threads, flash_gqa_tc_shared_bytes, stream>>>(
                                (const __nv_bfloat16*)q_proj_bf16,
                                (const __nv_bfloat16*)layer_k_cache,
                                (const __nv_bfloat16*)layer_v_cache,
                                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                (const float*)layer_k_scale_cache,
                                (const float*)layer_v_scale_cache,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_m_dev
                                    : (float*)split_attn_partial_m,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_s_dev
                                    : (float*)split_attn_partial_s,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_out_dev
                                    : (float*)split_attn_partial_out,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                HEAD_DIM,
                                max_seq_len,
                                attn_scale,
                                attn_warps,
                                parts,
                                split_attn_chunk_size,
                                split_attn_max_chunks,
                                kv_block_table_attn,
                                kv_block_size,
                                position_ptr,
                                flash_fp8_tc_qk,
                                gqa_phase_debug_ptr
                            );
                            } else {// kV_FP8 + 开启 TC
                                decode_attention_cache_flash_phase1_gqa2_kernel_t<true, true, true, false><<<blocks, threads, flash_gqa_tc_shared_bytes, stream>>>(
                            (const __nv_bfloat16*)q_proj_bf16,
                            (const __nv_bfloat16*)layer_k_cache,
                            (const __nv_bfloat16*)layer_v_cache,
                            (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                            (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                            (const float*)layer_k_scale_cache,
                            (const float*)layer_v_scale_cache,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_m_dev
                                : (float*)split_attn_partial_m,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_s_dev
                                : (float*)split_attn_partial_s,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_out_dev
                                : (float*)split_attn_partial_out,
                            NUM_Q_HEADS,
                            NUM_KV_HEADS,
                            HEAD_DIM,
                            max_seq_len,
                            attn_scale,
                            attn_warps,
                            parts,
                            split_attn_chunk_size,
                            split_attn_max_chunks,
                            kv_block_table_attn,
                            kv_block_size,
                            position_ptr,
                            flash_fp8_tc_qk,
                            nullptr
                        );
                            }
                        }
                    } else {// v3-S0
                        if (k_fp8_enabled) {
                            decode_attention_cache_flash_phase1_kernel_t<true><<<blocks, threads, 0, stream>>>(
                                (const __nv_bfloat16*)q_proj_bf16,
                                (const __nv_bfloat16*)layer_k_cache,
                                (const __nv_bfloat16*)layer_v_cache,
                                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                (const float*)layer_k_scale_cache,
                                (const float*)layer_v_scale_cache,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_m_dev
                                    : (float*)split_attn_partial_m,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_s_dev
                                    : (float*)split_attn_partial_s,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_out_dev
                                    : (float*)split_attn_partial_out,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                HEAD_DIM,
                                max_seq_len,
                                attn_scale,
                                attn_warps,
                                parts,
                                split_attn_chunk_size,
                                split_attn_max_chunks,
                                kv_block_table_attn,
                                kv_block_size,
                                position_ptr,
                                (debug_flash_decode && flash_phase_debug_dev != nullptr &&
                                 flash_phase_debug_cap >= blocks)
                                    ? flash_phase_debug_dev
                                    : nullptr
                            );
                        } else {
                            decode_attention_cache_flash_phase1_kernel_t<false><<<blocks, threads, 0, stream>>>(
                                (const __nv_bfloat16*)q_proj_bf16,
                                (const __nv_bfloat16*)layer_k_cache,
                                (const __nv_bfloat16*)layer_v_cache,
                                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                (const float*)layer_k_scale_cache,
                                (const float*)layer_v_scale_cache,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_m_dev
                                    : (float*)split_attn_partial_m,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_s_dev
                                    : (float*)split_attn_partial_s,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_out_dev
                                    : (float*)split_attn_partial_out,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                HEAD_DIM,
                                max_seq_len,
                                attn_scale,
                                attn_warps,
                                parts,
                                split_attn_chunk_size,
                                split_attn_max_chunks,
                                kv_block_table_attn,
                                kv_block_size,
                                position_ptr,
                                (debug_flash_decode && flash_phase_debug_dev != nullptr &&
                                 flash_phase_debug_cap >= blocks)
                                    ? flash_phase_debug_dev
                                    : nullptr
                            );
                        }
                    }
                } else {// 完全未启用 FP8
                    if (use_gqa_phase1) {// 无 FP8 的GQA
                        if (gqa_phase_debug_ptr != nullptr) {
                            decode_attention_cache_flash_phase1_gqa2_kernel_t<false, false, false, true><<<blocks, threads, 0, stream>>>(
                                (const __nv_bfloat16*)q_proj_bf16,
                                (const __nv_bfloat16*)layer_k_cache,
                                (const __nv_bfloat16*)layer_v_cache,
                                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                                (const float*)layer_k_scale_cache,
                                (const float*)layer_v_scale_cache,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_m_dev
                                    : (float*)split_attn_partial_m,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_s_dev
                                    : (float*)split_attn_partial_s,
                                (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                    ? (float*)flash_part_out_dev
                                    : (float*)split_attn_partial_out,
                                NUM_Q_HEADS,
                                NUM_KV_HEADS,
                                HEAD_DIM,
                                max_seq_len,
                                attn_scale,
                                attn_warps,
                                parts,
                                split_attn_chunk_size,
                                split_attn_max_chunks,
                                kv_block_table_attn,
                                kv_block_size,
                                position_ptr,
                                flash_fp8_tc_qk,
                                gqa_phase_debug_ptr
                            );
                        } else {
                            decode_attention_cache_flash_phase1_gqa2_kernel_t<false, false, false, false><<<blocks, threads, 0, stream>>>(
                            (const __nv_bfloat16*)q_proj_bf16,
                            (const __nv_bfloat16*)layer_k_cache,
                            (const __nv_bfloat16*)layer_v_cache,
                            (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                            (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                            (const float*)layer_k_scale_cache,
                            (const float*)layer_v_scale_cache,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_m_dev
                                : (float*)split_attn_partial_m,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_s_dev
                                : (float*)split_attn_partial_s,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_out_dev
                                : (float*)split_attn_partial_out,
                            NUM_Q_HEADS,
                            NUM_KV_HEADS,
                            HEAD_DIM,
                            max_seq_len,
                            attn_scale,
                            attn_warps,
                            parts,
                            split_attn_chunk_size,
                            split_attn_max_chunks,
                            kv_block_table_attn,
                            kv_block_size,
                            position_ptr,
                            flash_fp8_tc_qk,
                            nullptr
                        );
                        }
                    } else {
                        decode_attention_cache_flash_phase1_kernel_t<false><<<blocks, threads, 0, stream>>>(
                            (const __nv_bfloat16*)q_proj_bf16,
                            (const __nv_bfloat16*)layer_k_cache,
                            (const __nv_bfloat16*)layer_v_cache,
                            (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                            (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                            (const float*)layer_k_scale_cache,
                            (const float*)layer_v_scale_cache,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_m_dev
                                : (float*)split_attn_partial_m,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_s_dev
                                : (float*)split_attn_partial_s,
                            (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr)
                                ? (float*)flash_part_out_dev
                                : (float*)split_attn_partial_out,
                            NUM_Q_HEADS,
                            NUM_KV_HEADS,
                            HEAD_DIM,
                            max_seq_len,
                            attn_scale,
                            attn_warps,
                            parts,
                            split_attn_chunk_size,
                            split_attn_max_chunks,
                            kv_block_table_attn,
                            kv_block_size,
                            position_ptr,
                            (debug_flash_decode && flash_phase_debug_dev != nullptr &&
                             flash_phase_debug_cap >= blocks)
                                ? flash_phase_debug_dev
                                : nullptr
                        );
                    }
                }
                // part > 1 会插入以下kernel，TODO: part指的时什么？
                if (parts > 1 && flash_part_m_dev != nullptr && flash_part_s_dev != nullptr && flash_part_out_dev != nullptr) {
                    decode_attention_cache_flash_reduce_parts_kernel<<<NUM_Q_HEADS * split_attn_max_chunks, 128, 0, stream>>>(
                        (const float*)flash_part_m_dev,
                        (const float*)flash_part_s_dev,
                        (const float*)flash_part_out_dev,
                        (float*)split_attn_partial_m,
                        (float*)split_attn_partial_s,
                        (float*)split_attn_partial_out,
                        NUM_Q_HEADS,
                        HEAD_DIM,
                        split_attn_max_chunks,
                        parts,
                        split_attn_chunk_size,
                        position_ptr
                    );
                }
                // 上方的phase 1 内核，全部采用相同的 reduce phase2
                decode_attention_cache_splitk_phase2_kernel<<<NUM_Q_HEADS, 128, 0, stream>>>(
                    (const float*)split_attn_partial_m,
                    (const float*)split_attn_partial_s,
                    (const float*)split_attn_partial_out,
                    (__nv_bfloat16*)attn_out_bf16,
                    NUM_Q_HEADS,
                    HEAD_DIM,
                    split_attn_chunk_size,
                    split_attn_max_chunks,
                    position_ptr
                );
                if (debug_flash_decode) {
                    int cache_len_dbg = debug_position_host + 1;
                    if (cache_len_dbg < 0) cache_len_dbg = 0;
                    if (cache_len_dbg > max_seq_len) cache_len_dbg = max_seq_len;
                    int chunks_dbg = (cache_len_dbg + split_attn_chunk_size - 1) / split_attn_chunk_size;
                    if (chunks_dbg > split_attn_max_chunks) chunks_dbg = split_attn_max_chunks;
                    if (chunks_dbg < 1) chunks_dbg = 1;
                    int used_blocks = NUM_Q_HEADS * chunks_dbg * parts;
                    int launched_blocks = (use_gqa_phase1 ? NUM_KV_HEADS : NUM_Q_HEADS) * chunks_dbg * parts;
                    int phase_slots = NUM_Q_HEADS * split_attn_max_chunks * parts;
                    bool have_phase_debug =
                        (flash_phase_debug_dev != nullptr && flash_phase_debug_host != nullptr &&
                         flash_phase_debug_cap >= phase_slots);
                    if (have_phase_debug) {
                        cudaMemcpyAsync(
                            flash_phase_debug_host,
                            flash_phase_debug_dev,
                            sizeof(SplitFlashDecodeDebugRec) * phase_slots,
                            cudaMemcpyDeviceToHost,
                            stream);
                        cudaStreamSynchronize(stream);
                    }

                    unsigned long long head_cycles[NUM_Q_HEADS];
                    int head_tokens[NUM_Q_HEADS];
                    int head_switches[NUM_Q_HEADS];
                    int head_samples[NUM_Q_HEADS];
                    for (int h = 0; h < NUM_Q_HEADS; h++) {
                        head_cycles[h] = 0;
                        head_tokens[h] = 0;
                        head_switches[h] = 0;
                        head_samples[h] = 0;
                    }
                    long long sum_tokens = 0;
                    long long sum_switches = 0;
                    long long sum_oob = 0;
                    long long sum_tiles = 0;
                    unsigned long long sum_cycles = 0;
                    unsigned long long sum_map_cycles = 0;
                    unsigned long long sum_qk_cycles = 0;
                    unsigned long long sum_qk_load_cycles = 0;
                    unsigned long long sum_qk_load_mem_cycles = 0;
                    unsigned long long sum_qk_load_cvt_cycles = 0;
                    unsigned long long sum_qk_load_bf16_cycles = 0;
                    unsigned long long sum_qk_fma_cycles = 0;
                    unsigned long long sum_qk_fma_bf16_cycles = 0;
                    unsigned long long sum_qk_reduce_cycles = 0;
                    unsigned long long sum_qk_tc_pack_cycles = 0;
                    unsigned long long sum_qk_tc_mma_cycles = 0;
                    unsigned long long sum_qk_tc_unpack_cycles = 0;
                    unsigned long long sum_softmax_cycles = 0;
                    unsigned long long sum_value_cycles = 0;
                    unsigned long long sum_value_load_cycles = 0;
                    unsigned long long sum_value_fma_cycles = 0;
                    unsigned long long sum_value_dequant_cycles = 0;
                    unsigned long long sum_value_rescale_cycles = 0;
                    unsigned long long sum_value_accum_cycles = 0;
                    unsigned long long sum_value_bookkeep_cycles = 0;
                    unsigned long long sum_merge_cycles = 0;
                    int cache_len_from_rec = cache_len_dbg;
                    int warps_from_rec = attn_warps;
                    if (have_phase_debug) {
                        for (int p = 0; p < parts; p++) {
                            for (int c = 0; c < chunks_dbg; c++) {
                                for (int h = 0; h < NUM_Q_HEADS; h++) {
                                    int idx = (p * split_attn_max_chunks + c) * NUM_Q_HEADS + h;
                                    const SplitFlashDecodeDebugRec& rec = flash_phase_debug_host[idx];
                                    head_cycles[h] += rec.cycles;
                                    head_tokens[h] += rec.tokens_processed;
                                    head_switches[h] += rec.block_switches;
                                    head_samples[h] += 1;
                                    sum_cycles += rec.cycles;
                                    sum_tokens += rec.tokens_processed;
                                    sum_switches += rec.block_switches;
                                    sum_oob += rec.oob_skips;
                                    sum_tiles += rec.tiles_processed;
                                    sum_map_cycles += rec.map_cycles;
                                    sum_qk_cycles += rec.qk_cycles;
                                    sum_qk_load_cycles += rec.qk_load_cycles;
                                    sum_qk_load_mem_cycles += rec.qk_load_mem_cycles;
                                    sum_qk_load_cvt_cycles += rec.qk_load_cvt_cycles;
                                    sum_qk_load_bf16_cycles += rec.qk_load_bf16_cycles;
                                    sum_qk_fma_cycles += rec.qk_fma_cycles;
                                    sum_qk_fma_bf16_cycles += rec.qk_fma_bf16_cycles;
                                    sum_qk_reduce_cycles += rec.qk_reduce_cycles;
                                    sum_qk_tc_pack_cycles += rec.qk_tc_pack_cycles;
                                    sum_qk_tc_mma_cycles += rec.qk_tc_mma_cycles;
                                    sum_qk_tc_unpack_cycles += rec.qk_tc_unpack_cycles;
                                    sum_softmax_cycles += rec.softmax_cycles;
                                    sum_value_cycles += rec.value_cycles;
                                    sum_value_load_cycles += rec.value_load_cycles;
                                    sum_value_fma_cycles += rec.value_fma_cycles;
                                    sum_value_dequant_cycles += rec.value_dequant_cycles;
                                    sum_value_rescale_cycles += rec.value_rescale_cycles;
                                    sum_value_accum_cycles += rec.value_accum_cycles;
                                    sum_value_bookkeep_cycles += rec.value_bookkeep_cycles;
                                    sum_merge_cycles += rec.merge_cycles;
                                    if (p == 0 && c == 0 && h == 0) {
                                        cache_len_from_rec = rec.cache_len;
                                        warps_from_rec = rec.active_warps;
                                    }
                                }
                            }
                        }
                    }

                    bool gqa_mode_simt_effective = (use_gqa_simt_fast || !k_fp8_enabled);
                    int flash_fp8_tc_qk_effective =
                        (kv_fp8_enabled && use_gqa_phase1 && !gqa_mode_simt_effective && flash_fp8_tc_qk) ? 1 : 0;
                    double avg_cycles = (used_blocks > 0) ? (double(sum_cycles) / double(used_blocks)) : 0.0;
                    std::printf(
                        "[MEGAQWEN_DEBUG] FLASH_DECODE layer=%02d token_pos=%d mode=phase12 kv_mode=%s cache_len=%d warps=%d chunk=%d chunks=%d parts=%d blocks=%d launched=%d gqa_share=%d gqa_mode=%s tc_qk=%d tc_qk_eff=%d avg_cycles=%.1f tokens=%lld block_switches=%lld oob=%lld tiles=%lld\n",
                        layer,
                        debug_position_host,
                        kv_mode_desc,
                        cache_len_from_rec,
                        warps_from_rec,
                        split_attn_chunk_size,
                        chunks_dbg,
                        parts,
                        used_blocks,
                        launched_blocks,
                        use_gqa_phase1 ? 1 : 0,
                        gqa_mode_simt_effective ? "simt_fast" : "tc",
                        flash_fp8_tc_qk,
                        flash_fp8_tc_qk_effective,
                        avg_cycles,
                        sum_tokens,
                        sum_switches,
                        sum_oob,
                        sum_tiles);
                    if (debug_flash_decode && layer == 0) {
                        std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE kv_map=%s\n",
                                    kv_identity_fastpath ? "contiguous_fastpath" : "paged_table");
                    }
                    if (have_phase_debug && _split_debug_flash_breakdown_enabled()) {
                        long long sum_known = (long long)sum_map_cycles + (long long)sum_qk_cycles +
                                              (long long)sum_softmax_cycles + (long long)sum_value_cycles +
                                              (long long)sum_merge_cycles;
                        long long sum_other = (long long)sum_cycles - sum_known;
                        if (sum_other < 0) sum_other = 0;
                        double denom_overlap = (sum_cycles > 0) ? (double)sum_cycles : 1.0;
                        double denom_norm = (sum_known > 0) ? (double)sum_known : 1.0;
                        double cyc_to_ms = (debug_flash_clock_khz > 0) ? (1.0 / double(debug_flash_clock_khz)) : 0.0;
                        auto to_ms = [&](unsigned long long cycles) -> double {
                            return (cyc_to_ms > 0.0) ? (double(cycles) * cyc_to_ms) : 0.0;
                        };
                        auto to_ms_ll = [&](long long cycles) -> double {
                            return (cyc_to_ms > 0.0) ? (double(cycles) * cyc_to_ms) : 0.0;
                        };
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown (phase12): map=%llu/%.3fms ov=%.1f%% norm=%.1f%% qk=%llu/%.3fms ov=%.1f%% norm=%.1f%% softmax=%llu/%.3fms ov=%.1f%% norm=%.1f%% value=%llu/%.3fms ov=%.1f%% norm=%.1f%% merge=%llu/%.3fms ov=%.1f%% norm=%.1f%% other=%lld/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_map_cycles, to_ms(sum_map_cycles),
                            100.0 * (double)sum_map_cycles / denom_overlap, 100.0 * (double)sum_map_cycles / denom_norm,
                            (unsigned long long)sum_qk_cycles, to_ms(sum_qk_cycles),
                            100.0 * (double)sum_qk_cycles / denom_overlap, 100.0 * (double)sum_qk_cycles / denom_norm,
                            (unsigned long long)sum_softmax_cycles, to_ms(sum_softmax_cycles),
                            100.0 * (double)sum_softmax_cycles / denom_overlap, 100.0 * (double)sum_softmax_cycles / denom_norm,
                            (unsigned long long)sum_value_cycles, to_ms(sum_value_cycles),
                            100.0 * (double)sum_value_cycles / denom_overlap, 100.0 * (double)sum_value_cycles / denom_norm,
                            (unsigned long long)sum_merge_cycles, to_ms(sum_merge_cycles),
                            100.0 * (double)sum_merge_cycles / denom_overlap, 100.0 * (double)sum_merge_cycles / denom_norm,
                            sum_other, to_ms_ll(sum_other),
                            100.0 * (double)sum_other / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_fine (phase12): qk_load=%llu/%.3fms ov=%.1f%% qk_fma=%llu/%.3fms ov=%.1f%% qk_reduce=%llu/%.3fms ov=%.1f%% value_load=%llu/%.3fms ov=%.1f%% value_fma=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_load_cycles, to_ms(sum_qk_load_cycles), 100.0 * (double)sum_qk_load_cycles / denom_overlap,
                            (unsigned long long)sum_qk_fma_cycles, to_ms(sum_qk_fma_cycles), 100.0 * (double)sum_qk_fma_cycles / denom_overlap,
                            (unsigned long long)sum_qk_reduce_cycles, to_ms(sum_qk_reduce_cycles), 100.0 * (double)sum_qk_reduce_cycles / denom_overlap,
                            (unsigned long long)sum_value_load_cycles, to_ms(sum_value_load_cycles), 100.0 * (double)sum_value_load_cycles / denom_overlap,
                            (unsigned long long)sum_value_fma_cycles, to_ms(sum_value_fma_cycles), 100.0 * (double)sum_value_fma_cycles / denom_overlap);
                        double denom_qk_load = (sum_qk_load_cycles > 0) ? (double)sum_qk_load_cycles : 1.0;
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_qk_load_fine (phase12): qk_load_mem=%llu/%.3fms ov=%.1f%% qk_load_cvt=%llu/%.3fms ov=%.1f%% qk_load_bf16=%llu/%.3fms ov=%.1f%% mem_ratio=%.1f%% cvt_ratio=%.1f%% bf16_ratio=%.1f%% qk_fma_bf16=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_load_mem_cycles, to_ms(sum_qk_load_mem_cycles), 100.0 * (double)sum_qk_load_mem_cycles / denom_overlap,
                            (unsigned long long)sum_qk_load_cvt_cycles, to_ms(sum_qk_load_cvt_cycles), 100.0 * (double)sum_qk_load_cvt_cycles / denom_overlap,
                            (unsigned long long)sum_qk_load_bf16_cycles, to_ms(sum_qk_load_bf16_cycles), 100.0 * (double)sum_qk_load_bf16_cycles / denom_overlap,
                            100.0 * (double)sum_qk_load_mem_cycles / denom_qk_load,
                            100.0 * (double)sum_qk_load_cvt_cycles / denom_qk_load,
                            100.0 * (double)sum_qk_load_bf16_cycles / denom_qk_load,
                            (unsigned long long)sum_qk_fma_bf16_cycles, to_ms(sum_qk_fma_bf16_cycles), 100.0 * (double)sum_qk_fma_bf16_cycles / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_value_fine (phase12): value_dequant=%llu/%.3fms ov=%.1f%% value_rescale=%llu/%.3fms ov=%.1f%% value_accum=%llu/%.3fms ov=%.1f%% value_bookkeep=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_value_dequant_cycles, to_ms(sum_value_dequant_cycles), 100.0 * (double)sum_value_dequant_cycles / denom_overlap,
                            (unsigned long long)sum_value_rescale_cycles, to_ms(sum_value_rescale_cycles), 100.0 * (double)sum_value_rescale_cycles / denom_overlap,
                            (unsigned long long)sum_value_accum_cycles, to_ms(sum_value_accum_cycles), 100.0 * (double)sum_value_accum_cycles / denom_overlap,
                            (unsigned long long)sum_value_bookkeep_cycles, to_ms(sum_value_bookkeep_cycles), 100.0 * (double)sum_value_bookkeep_cycles / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_tc (phase12): qk_tc_pack=%llu/%.3fms ov=%.1f%% qk_tc_mma=%llu/%.3fms ov=%.1f%% qk_tc_unpack=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_tc_pack_cycles, to_ms(sum_qk_tc_pack_cycles), 100.0 * (double)sum_qk_tc_pack_cycles / denom_overlap,
                            (unsigned long long)sum_qk_tc_mma_cycles, to_ms(sum_qk_tc_mma_cycles), 100.0 * (double)sum_qk_tc_mma_cycles / denom_overlap,
                            (unsigned long long)sum_qk_tc_unpack_cycles, to_ms(sum_qk_tc_unpack_cycles), 100.0 * (double)sum_qk_tc_unpack_cycles / denom_overlap);
                        unsigned long long sum_qk_mutual_cycles =
                            sum_qk_load_cycles + sum_qk_fma_cycles + sum_qk_reduce_cycles +
                            sum_qk_tc_pack_cycles + sum_qk_tc_mma_cycles + sum_qk_tc_unpack_cycles;
                        double denom_qk = (sum_qk_cycles > 0) ? (double)sum_qk_cycles : 1.0;
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_qk_mutual (phase12): qk_known=%llu/%.3fms qk_total=%llu/%.3fms norm_qk=%.1f%%\n",
                            (unsigned long long)sum_qk_mutual_cycles, to_ms(sum_qk_mutual_cycles),
                            (unsigned long long)sum_qk_cycles, to_ms(sum_qk_cycles),
                            100.0 * (double)sum_qk_mutual_cycles / denom_qk);
                        std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_note: ov=percent_vs_total_cycles (may >100%%), norm=percent_vs_known_components (<=100%%), qk_mutual=non-overlap qk components\n");
                    }

                    if (have_phase_debug) {
                        struct FlashPair {
                            unsigned long long cycles;
                            int head;
                        };
                        FlashPair pairs[NUM_Q_HEADS];
                        for (int h = 0; h < NUM_Q_HEADS; h++) {
                            pairs[h] = FlashPair{head_cycles[h], h};
                        }
                        for (int i = 0; i < NUM_Q_HEADS; i++) {
                            for (int j = i + 1; j < NUM_Q_HEADS; j++) {
                                if (pairs[j].cycles > pairs[i].cycles) {
                                    FlashPair tmp = pairs[i];
                                    pairs[i] = pairs[j];
                                    pairs[j] = tmp;
                                }
                            }
                        }
                        int topk = _split_debug_flash_head_topk();
                        if (topk > NUM_Q_HEADS) topk = NUM_Q_HEADS;
                        std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE head hotspot (phase12 cycles/tokens/switches/samples):");
                        for (int i = 0; i < topk; i++) {
                            int h = pairs[i].head;
                            std::printf(" H%02d=%llu/%d/%d/%d",
                                        h,
                                        (unsigned long long)head_cycles[h],
                                        head_tokens[h],
                                        head_switches[h],
                                        head_samples[h]);
                        }
                        std::printf("\n");
                    }
                }
            } else {// 合并的内核 V3
                int threads = attn_warps * WARP_SIZE;
                if (kv_fp8_enabled) {
                    decode_attention_cache_flash_decode_kernel_t<true><<<NUM_Q_HEADS, threads, 0, stream>>>(
                        (const __nv_bfloat16*)q_proj_bf16,
                        (const __nv_bfloat16*)layer_k_cache,
                        (const __nv_bfloat16*)layer_v_cache,
                        (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                        (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                        (const float*)layer_k_scale_cache,
                        (const float*)layer_v_scale_cache,
                        (__nv_bfloat16*)attn_out_bf16,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        max_seq_len,
                        attn_scale,
                        attn_warps,
                        kv_block_table_attn,
                        kv_block_size,
                        position_ptr,
                        debug_flash_decode ? flash_debug_dev : nullptr
                    );
                } else {
                    decode_attention_cache_flash_decode_kernel_t<false><<<NUM_Q_HEADS, threads, 0, stream>>>(
                        (const __nv_bfloat16*)q_proj_bf16,
                        (const __nv_bfloat16*)layer_k_cache,
                        (const __nv_bfloat16*)layer_v_cache,
                        (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                        (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                        (const float*)layer_k_scale_cache,
                        (const float*)layer_v_scale_cache,
                        (__nv_bfloat16*)attn_out_bf16,
                        NUM_Q_HEADS,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                        max_seq_len,
                        attn_scale,
                        attn_warps,
                        kv_block_table_attn,
                        kv_block_size,
                        position_ptr,
                        debug_flash_decode ? flash_debug_dev : nullptr
                    );
                }
                if (debug_flash_decode && flash_debug_dev != nullptr) {
                    SplitFlashDecodeDebugRec flash_debug_host[NUM_Q_HEADS];
                    cudaMemcpyAsync(
                        flash_debug_host,
                        flash_debug_dev,
                        sizeof(SplitFlashDecodeDebugRec) * NUM_Q_HEADS,
                        cudaMemcpyDeviceToHost,
                        stream);
                    cudaStreamSynchronize(stream);

                    unsigned long long sum_cycles = 0;
                    long long sum_tokens = 0;
                    long long sum_switches = 0;
                    long long sum_oob = 0;
                    long long sum_tiles = 0;
                    unsigned long long sum_map_cycles = 0;
                    unsigned long long sum_qk_cycles = 0;
                    unsigned long long sum_qk_load_cycles = 0;
                    unsigned long long sum_qk_load_mem_cycles = 0;
                    unsigned long long sum_qk_load_cvt_cycles = 0;
                    unsigned long long sum_qk_load_bf16_cycles = 0;
                    unsigned long long sum_qk_fma_cycles = 0;
                    unsigned long long sum_qk_fma_bf16_cycles = 0;
                    unsigned long long sum_qk_reduce_cycles = 0;
                    unsigned long long sum_qk_tc_pack_cycles = 0;
                    unsigned long long sum_qk_tc_mma_cycles = 0;
                    unsigned long long sum_qk_tc_unpack_cycles = 0;
                    unsigned long long sum_softmax_cycles = 0;
                    unsigned long long sum_value_cycles = 0;
                    unsigned long long sum_value_load_cycles = 0;
                    unsigned long long sum_value_fma_cycles = 0;
                    unsigned long long sum_value_dequant_cycles = 0;
                    unsigned long long sum_value_rescale_cycles = 0;
                    unsigned long long sum_value_accum_cycles = 0;
                    unsigned long long sum_value_bookkeep_cycles = 0;
                    unsigned long long sum_merge_cycles = 0;
                    int cache_len_dbg = 0;
                    int active_warps_dbg = 0;
                    for (int h = 0; h < NUM_Q_HEADS; h++) {
                        sum_cycles += flash_debug_host[h].cycles;
                        sum_tokens += flash_debug_host[h].tokens_processed;
                        sum_switches += flash_debug_host[h].block_switches;
                        sum_oob += flash_debug_host[h].oob_skips;
                        sum_tiles += flash_debug_host[h].tiles_processed;
                        sum_map_cycles += flash_debug_host[h].map_cycles;
                        sum_qk_cycles += flash_debug_host[h].qk_cycles;
                        sum_qk_load_cycles += flash_debug_host[h].qk_load_cycles;
                        sum_qk_load_mem_cycles += flash_debug_host[h].qk_load_mem_cycles;
                        sum_qk_load_cvt_cycles += flash_debug_host[h].qk_load_cvt_cycles;
                        sum_qk_load_bf16_cycles += flash_debug_host[h].qk_load_bf16_cycles;
                        sum_qk_fma_cycles += flash_debug_host[h].qk_fma_cycles;
                        sum_qk_fma_bf16_cycles += flash_debug_host[h].qk_fma_bf16_cycles;
                        sum_qk_reduce_cycles += flash_debug_host[h].qk_reduce_cycles;
                        sum_qk_tc_pack_cycles += flash_debug_host[h].qk_tc_pack_cycles;
                        sum_qk_tc_mma_cycles += flash_debug_host[h].qk_tc_mma_cycles;
                        sum_qk_tc_unpack_cycles += flash_debug_host[h].qk_tc_unpack_cycles;
                        sum_softmax_cycles += flash_debug_host[h].softmax_cycles;
                        sum_value_cycles += flash_debug_host[h].value_cycles;
                        sum_value_load_cycles += flash_debug_host[h].value_load_cycles;
                        sum_value_fma_cycles += flash_debug_host[h].value_fma_cycles;
                        sum_value_dequant_cycles += flash_debug_host[h].value_dequant_cycles;
                        sum_value_rescale_cycles += flash_debug_host[h].value_rescale_cycles;
                        sum_value_accum_cycles += flash_debug_host[h].value_accum_cycles;
                        sum_value_bookkeep_cycles += flash_debug_host[h].value_bookkeep_cycles;
                        sum_merge_cycles += flash_debug_host[h].merge_cycles;
                        if (h == 0) {
                            cache_len_dbg = flash_debug_host[h].cache_len;
                            active_warps_dbg = flash_debug_host[h].active_warps;
                        }
                    }
                    double avg_cycles = double(sum_cycles) / double(NUM_Q_HEADS);
                    std::printf(
                        "[MEGAQWEN_DEBUG] FLASH_DECODE layer=%02d token_pos=%d mode=single kv_mode=%s cache_len=%d warps=%d avg_cycles=%.1f tokens=%lld block_switches=%lld oob=%lld tiles=%lld\n",
                        layer,
                        debug_position_host,
                        kv_mode_desc,
                        cache_len_dbg,
                        active_warps_dbg,
                        avg_cycles,
                        sum_tokens,
                        sum_switches,
                        sum_oob,
                        sum_tiles);
                    if (_split_debug_flash_breakdown_enabled()) {
                        long long sum_known = (long long)sum_map_cycles + (long long)sum_qk_cycles +
                                              (long long)sum_softmax_cycles + (long long)sum_value_cycles +
                                              (long long)sum_merge_cycles;
                        long long sum_other = (long long)sum_cycles - sum_known;
                        if (sum_other < 0) sum_other = 0;
                        double denom_overlap = (sum_cycles > 0) ? (double)sum_cycles : 1.0;
                        double denom_norm = (sum_known > 0) ? (double)sum_known : 1.0;
                        double cyc_to_ms = (debug_flash_clock_khz > 0) ? (1.0 / double(debug_flash_clock_khz)) : 0.0;
                        auto to_ms = [&](unsigned long long cycles) -> double {
                            return (cyc_to_ms > 0.0) ? (double(cycles) * cyc_to_ms) : 0.0;
                        };
                        auto to_ms_ll = [&](long long cycles) -> double {
                            return (cyc_to_ms > 0.0) ? (double(cycles) * cyc_to_ms) : 0.0;
                        };
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown: map=%llu/%.3fms ov=%.1f%% norm=%.1f%% qk=%llu/%.3fms ov=%.1f%% norm=%.1f%% softmax=%llu/%.3fms ov=%.1f%% norm=%.1f%% value=%llu/%.3fms ov=%.1f%% norm=%.1f%% merge=%llu/%.3fms ov=%.1f%% norm=%.1f%% other=%lld/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_map_cycles, to_ms(sum_map_cycles),
                            100.0 * (double)sum_map_cycles / denom_overlap, 100.0 * (double)sum_map_cycles / denom_norm,
                            (unsigned long long)sum_qk_cycles, to_ms(sum_qk_cycles),
                            100.0 * (double)sum_qk_cycles / denom_overlap, 100.0 * (double)sum_qk_cycles / denom_norm,
                            (unsigned long long)sum_softmax_cycles, to_ms(sum_softmax_cycles),
                            100.0 * (double)sum_softmax_cycles / denom_overlap, 100.0 * (double)sum_softmax_cycles / denom_norm,
                            (unsigned long long)sum_value_cycles, to_ms(sum_value_cycles),
                            100.0 * (double)sum_value_cycles / denom_overlap, 100.0 * (double)sum_value_cycles / denom_norm,
                            (unsigned long long)sum_merge_cycles, to_ms(sum_merge_cycles),
                            100.0 * (double)sum_merge_cycles / denom_overlap, 100.0 * (double)sum_merge_cycles / denom_norm,
                            sum_other, to_ms_ll(sum_other),
                            100.0 * (double)sum_other / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_fine: qk_load=%llu/%.3fms ov=%.1f%% qk_fma=%llu/%.3fms ov=%.1f%% qk_reduce=%llu/%.3fms ov=%.1f%% value_load=%llu/%.3fms ov=%.1f%% value_fma=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_load_cycles, to_ms(sum_qk_load_cycles), 100.0 * (double)sum_qk_load_cycles / denom_overlap,
                            (unsigned long long)sum_qk_fma_cycles, to_ms(sum_qk_fma_cycles), 100.0 * (double)sum_qk_fma_cycles / denom_overlap,
                            (unsigned long long)sum_qk_reduce_cycles, to_ms(sum_qk_reduce_cycles), 100.0 * (double)sum_qk_reduce_cycles / denom_overlap,
                            (unsigned long long)sum_value_load_cycles, to_ms(sum_value_load_cycles), 100.0 * (double)sum_value_load_cycles / denom_overlap,
                            (unsigned long long)sum_value_fma_cycles, to_ms(sum_value_fma_cycles), 100.0 * (double)sum_value_fma_cycles / denom_overlap);
                        double denom_qk_load = (sum_qk_load_cycles > 0) ? (double)sum_qk_load_cycles : 1.0;
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_qk_load_fine: qk_load_mem=%llu/%.3fms ov=%.1f%% qk_load_cvt=%llu/%.3fms ov=%.1f%% qk_load_bf16=%llu/%.3fms ov=%.1f%% mem_ratio=%.1f%% cvt_ratio=%.1f%% bf16_ratio=%.1f%% qk_fma_bf16=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_load_mem_cycles, to_ms(sum_qk_load_mem_cycles), 100.0 * (double)sum_qk_load_mem_cycles / denom_overlap,
                            (unsigned long long)sum_qk_load_cvt_cycles, to_ms(sum_qk_load_cvt_cycles), 100.0 * (double)sum_qk_load_cvt_cycles / denom_overlap,
                            (unsigned long long)sum_qk_load_bf16_cycles, to_ms(sum_qk_load_bf16_cycles), 100.0 * (double)sum_qk_load_bf16_cycles / denom_overlap,
                            100.0 * (double)sum_qk_load_mem_cycles / denom_qk_load,
                            100.0 * (double)sum_qk_load_cvt_cycles / denom_qk_load,
                            100.0 * (double)sum_qk_load_bf16_cycles / denom_qk_load,
                            (unsigned long long)sum_qk_fma_bf16_cycles, to_ms(sum_qk_fma_bf16_cycles), 100.0 * (double)sum_qk_fma_bf16_cycles / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_value_fine: value_dequant=%llu/%.3fms ov=%.1f%% value_rescale=%llu/%.3fms ov=%.1f%% value_accum=%llu/%.3fms ov=%.1f%% value_bookkeep=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_value_dequant_cycles, to_ms(sum_value_dequant_cycles), 100.0 * (double)sum_value_dequant_cycles / denom_overlap,
                            (unsigned long long)sum_value_rescale_cycles, to_ms(sum_value_rescale_cycles), 100.0 * (double)sum_value_rescale_cycles / denom_overlap,
                            (unsigned long long)sum_value_accum_cycles, to_ms(sum_value_accum_cycles), 100.0 * (double)sum_value_accum_cycles / denom_overlap,
                            (unsigned long long)sum_value_bookkeep_cycles, to_ms(sum_value_bookkeep_cycles), 100.0 * (double)sum_value_bookkeep_cycles / denom_overlap);
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_tc: qk_tc_pack=%llu/%.3fms ov=%.1f%% qk_tc_mma=%llu/%.3fms ov=%.1f%% qk_tc_unpack=%llu/%.3fms ov=%.1f%%\n",
                            (unsigned long long)sum_qk_tc_pack_cycles, to_ms(sum_qk_tc_pack_cycles), 100.0 * (double)sum_qk_tc_pack_cycles / denom_overlap,
                            (unsigned long long)sum_qk_tc_mma_cycles, to_ms(sum_qk_tc_mma_cycles), 100.0 * (double)sum_qk_tc_mma_cycles / denom_overlap,
                            (unsigned long long)sum_qk_tc_unpack_cycles, to_ms(sum_qk_tc_unpack_cycles), 100.0 * (double)sum_qk_tc_unpack_cycles / denom_overlap);
                        unsigned long long sum_qk_mutual_cycles =
                            sum_qk_load_cycles + sum_qk_fma_cycles + sum_qk_reduce_cycles +
                            sum_qk_tc_pack_cycles + sum_qk_tc_mma_cycles + sum_qk_tc_unpack_cycles;
                        double denom_qk = (sum_qk_cycles > 0) ? (double)sum_qk_cycles : 1.0;
                        std::printf(
                            "[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_qk_mutual: qk_known=%llu/%.3fms qk_total=%llu/%.3fms norm_qk=%.1f%%\n",
                            (unsigned long long)sum_qk_mutual_cycles, to_ms(sum_qk_mutual_cycles),
                            (unsigned long long)sum_qk_cycles, to_ms(sum_qk_cycles),
                            100.0 * (double)sum_qk_mutual_cycles / denom_qk);
                        std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE breakdown_note: ov=percent_vs_total_cycles (may >100%%), norm=percent_vs_known_components (<=100%%), qk_mutual=non-overlap qk components\n");
                    }

                    struct FlashPair {
                        unsigned long long cycles;
                        int head;
                    };
                    FlashPair pairs[NUM_Q_HEADS];
                    for (int h = 0; h < NUM_Q_HEADS; h++) {
                        pairs[h] = FlashPair{flash_debug_host[h].cycles, h};
                    }
                    for (int i = 0; i < NUM_Q_HEADS; i++) {
                        for (int j = i + 1; j < NUM_Q_HEADS; j++) {
                            if (pairs[j].cycles > pairs[i].cycles) {
                                FlashPair tmp = pairs[i];
                                pairs[i] = pairs[j];
                                pairs[j] = tmp;
                            }
                        }
                    }
                    int topk = _split_debug_flash_head_topk();
                    if (topk > NUM_Q_HEADS) topk = NUM_Q_HEADS;
                    std::printf("[MEGAQWEN_DEBUG] FLASH_DECODE head hotspot (cycles/tokens/switches):");
                    for (int i = 0; i < topk; i++) {
                        int h = pairs[i].head;
                        std::printf(" H%02d=%llu/%d/%d",
                                    h,
                                    (unsigned long long)flash_debug_host[h].cycles,
                                    flash_debug_host[h].tokens_processed,
                                    flash_debug_host[h].block_switches);
                    }
                    std::printf("\n");
                }
            }
        } else if (attn_impl == 2 && split_attn_chunk_size > 0 && split_attn_max_chunks > 0 &&
            split_attn_partial_m != nullptr && split_attn_partial_s != nullptr && split_attn_partial_out != nullptr) {
            int blocks = NUM_Q_HEADS * split_attn_max_chunks;
            int threads = attn_warps * WARP_SIZE;// 下方核函数会导致后续出错；如果替换为其他正常的kernel，则没有该问题
            decode_attention_cache_splitk_phase1_kernel<<<blocks, threads, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                (const float*)layer_k_scale_cache,
                (const float*)layer_v_scale_cache,
                kv_fp8_enabled ? 1 : 0,
                (float*)split_attn_partial_m,
                (float*)split_attn_partial_s,
                (float*)split_attn_partial_out,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                attn_warps,
                split_attn_chunk_size,
                split_attn_max_chunks,
                kv_block_table,
                kv_block_size,
                position_ptr
            );
            decode_attention_cache_splitk_phase2_kernel<<<NUM_Q_HEADS, 128, 0, stream>>>(
                (const float*)split_attn_partial_m,
                (const float*)split_attn_partial_s,
                (const float*)split_attn_partial_out,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                HEAD_DIM,
                split_attn_chunk_size,
                split_attn_max_chunks,
                position_ptr
            );
        } else if (attn_impl == 1) {
            int threads = attn_warps * WARP_SIZE;
            decode_attention_cache_splitk_kernel<<<NUM_Q_HEADS, threads, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                (const float*)layer_k_scale_cache,
                (const float*)layer_v_scale_cache,
                kv_fp8_enabled ? 1 : 0,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                attn_warps,
                kv_block_table,
                kv_block_size,
                position_ptr
            );
        } else {
            int warps_per_block = 8;
            int threads_per_block = warps_per_block * WARP_SIZE;  // 256
            int num_warps = NUM_Q_HEADS;
            int num_blocks = (num_warps + warps_per_block - 1) / warps_per_block;
            decode_attention_cache_legacy_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                (const __nv_bfloat16*)q_proj_bf16,
                (const __nv_bfloat16*)layer_k_cache,
                (const __nv_bfloat16*)layer_v_cache,
                (const __nv_fp8_e4m3*)layer_k_cache_fp8,
                (const __nv_fp8_e4m3*)layer_v_cache_fp8,
                (const float*)layer_k_scale_cache,
                (const float*)layer_v_scale_cache,
                kv_fp8_enabled ? 1 : 0,
                (__nv_bfloat16*)attn_out_bf16,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                max_seq_len,
                attn_scale,
                kv_block_table,
                kv_block_size,
                position_ptr
            );
        }
        dbg_end(dbg_attn);

        // 5) O projection: [1,q_size] @ [hidden,q_size]^T -> [1,hidden]
        int dbg_o = dbg_begin(SPLIT_STAGE_O_GEMM, layer);
        bool o_done = false;
        bool o_lt_done = false;
        bool o_gemmex_done = false;
        bool o_gemv_done = false;
        bool o_use_gemv = _split_o_use_gemv();
        bool o_try_lt = _split_o_try_cublaslt();
        if (o_w4) {
            split_w4_matvec_bf16_out_kernel_t<256><<<HIDDEN_SIZE, 256, 0, stream>>>(
                o_w4_packed,
                o_w4_scales,
                o_w4_codebook,
                (const __nv_bfloat16*)attn_out_bf16,
                (__nv_bfloat16*)o_proj_out_bf16,
                HIDDEN_SIZE,
                Q_SIZE,
                nullptr
            );
            o_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.o_w4 += 1;
            }
        } else if (!o_use_gemv && o_try_lt) {
            o_done = _split_lt_matmul(
                cublaslt_handle,
                o_lt_plan,
                HIDDEN_SIZE,
                Q_SIZE,
                (const __nv_bfloat16*)w.o_proj_weight,
                (const __nv_bfloat16*)attn_out_bf16,
                (__nv_bfloat16*)o_proj_out_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            o_lt_done = o_done;
        }
        if (!o_done) {
            if (o_use_gemv) {
                o_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    (const __nv_bfloat16*)w.o_proj_weight,
                    HIDDEN_SIZE,
                    Q_SIZE,
                    (const __nv_bfloat16*)attn_out_bf16,
                    (__nv_bfloat16*)o_proj_out_bf16,
                    "o_gemv"
                );
                o_done = o_gemv_done;
            }
            if (!o_done) {// 首次出错地方
                o_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_SIZE, 1, Q_SIZE,
                    &alpha,
                    w.o_proj_weight, CUDA_R_16BF, Q_SIZE,
                    attn_out_bf16, CUDA_R_16BF, Q_SIZE,
                    &beta,
                    o_proj_out_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "o_gemm")) return;
            }
        }
        if (debug_stage_avg && !o_w4) {
            auto& agg = _split_debug_agg();
            if (!o_use_gemv && o_try_lt) {
                if (o_lt_done) agg.o_lt_ok += 1;
                else agg.o_lt_fb += 1;
            }
            if (o_gemv_done) agg.o_gemv += 1;
            else if (o_gemmex_done) agg.o_gemmex += 1;
        }
        dbg_end(dbg_o);

        // 6) Residual add -> hidden
        int dbg_resid1 = dbg_begin(SPLIT_STAGE_RESID1, layer);
        split_residual_add_bf16_to_f32_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
            (const __nv_bfloat16*)o_proj_out_bf16,
            (const float*)residual_f32,
            (float*)hidden_f32,
            HIDDEN_SIZE
        );
        dbg_end(dbg_resid1);

        // 7) Post-attn RMSNorm (FP32 chain, BF16 output for GEMM)
        int dbg_rms2 = dbg_begin(SPLIT_STAGE_RMS2, layer);
        split_rmsnorm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
            (const float*)hidden_f32,
            (const __nv_bfloat16*)w.post_attn_layernorm_weight,
            (__nv_bfloat16*)mlp_norm_bf16,
            (float*)residual_f32
        );
        dbg_end(dbg_rms2);

        // 8) GateUp projection (merged): [2*I, H] x [H, 1] -> [2*I, 1]
        int dbg_gateup = dbg_begin(SPLIT_STAGE_GATEUP_GEMM, layer);
        bool gateup_done = false;
        bool gateup_lt_done = false;
        bool gateup_gemmex_done = false;
        bool gateup_gemv_done = false;
        bool gateup_use_gemv = _split_gateup_use_gemv();
        bool gateup_try_lt = _split_gateup_try_cublaslt();
        int gateup_w4_block = _split_gateup_w4_block_threads();
        bool gateup_w4_warp_row = _split_gateup_w4_warp_row_enabled();
        if (ffn_w4) {
            if (gateup_w4_warp_row) {
                if (gateup_w4_block == 128) {
                    constexpr int kWarps = 128 / WARP_SIZE;
                    int grid = (GATEUP_ROWS + kWarps - 1) / kWarps;
                    split_w4_matvec_warp_per_row_kernel_t<128><<<grid, 128, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else if (gateup_w4_block == 192) {
                    constexpr int kWarps = 192 / WARP_SIZE;
                    int grid = (GATEUP_ROWS + kWarps - 1) / kWarps;
                    split_w4_matvec_warp_per_row_kernel_t<192><<<grid, 192, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else {
                    constexpr int kWarps = 256 / WARP_SIZE;
                    int grid = (GATEUP_ROWS + kWarps - 1) / kWarps;
                    split_w4_matvec_warp_per_row_kernel_t<256><<<grid, 256, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                }
            } else {
                if (gateup_w4_block == 128) {
                    split_w4_matvec_bf16_out_kernel_t<128><<<GATEUP_ROWS, 128, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else if (gateup_w4_block == 192) {
                    split_w4_matvec_bf16_out_kernel_t<192><<<GATEUP_ROWS, 192, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else {
                    split_w4_matvec_bf16_out_kernel_t<256><<<GATEUP_ROWS, 256, 0, stream>>>(
                        gateup_w4_packed,
                        gateup_w4_scales,
                        gateup_w4_codebook,
                        (const __nv_bfloat16*)mlp_norm_bf16,
                        (__nv_bfloat16*)gateup_out_bf16,
                        GATEUP_ROWS,
                        HIDDEN_SIZE,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                }
            }
            gateup_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.gateup_w4 += 1;
            }
        } else if (!gateup_use_gemv && gateup_try_lt) {
            gateup_done = _split_lt_matmul(
                cublaslt_handle,
                gateup_lt_plan,
                GATEUP_ROWS,
                HIDDEN_SIZE,
                gateup_weight,
                (const __nv_bfloat16*)mlp_norm_bf16,
                (__nv_bfloat16*)gateup_out_bf16,
                cublaslt_workspace,
                cublaslt_workspace_bytes,
                stream
            );
            gateup_lt_done = gateup_done;
        }
        if (!gateup_done) {
            if (gateup_use_gemv) {
                gateup_gemv_done = _split_try_gemv_matvec(
                    cublas_handle,
                    gateup_weight,
                    GATEUP_ROWS,
                    HIDDEN_SIZE,
                    (const __nv_bfloat16*)mlp_norm_bf16,
                    (__nv_bfloat16*)gateup_out_bf16,
                    "gateup_gemv"
                );
                gateup_done = gateup_gemv_done;
            }
            if (!gateup_done) {
                gateup_gemmex_done = true;
                if (!_cublas_check(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    GATEUP_ROWS, 1, HIDDEN_SIZE,
                    &alpha,
                    gateup_weight, CUDA_R_16BF, HIDDEN_SIZE,
                    mlp_norm_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                    &beta,
                    gateup_out_bf16, CUDA_R_16BF, GATEUP_ROWS,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                ), "gateup_gemm")) return;
            }
        }
        if (debug_stage_avg && !ffn_w4) {
            auto& agg = _split_debug_agg();
            if (!gateup_use_gemv && gateup_try_lt) {
                if (gateup_lt_done) agg.gateup_lt_ok += 1;
                else agg.gateup_lt_fb += 1;
            }
            if (gateup_gemv_done) agg.gateup_gemv += 1;
            else if (gateup_gemmex_done) agg.gateup_gemmex += 1;
        }
        dbg_end(dbg_gateup);

        __nv_bfloat16* gate_out_bf16 = (__nv_bfloat16*)gateup_out_bf16;
        __nv_bfloat16* up_out_bf16 = gate_out_bf16 + INTERMEDIATE_SIZE;

        int ffn_impl = _split_ffn_impl();
        bool ffn_w4_fused_silu_down = ffn_w4 && _split_ffn_w4_fused_silu_down_enabled();
        bool ffn_fused_tail = (!ffn_w4) && (ffn_impl != SPLIT_FFN_CUBLAS);

        // 9) SiLU(gate) * up
        int dbg_silu = dbg_begin(SPLIT_STAGE_SILU_MUL, layer);
        if (ffn_impl != SPLIT_FFN_FUSED_SILU_DOWN && !ffn_w4_fused_silu_down) {
            prefill_silu_mul_kernel<<<(INTERMEDIATE_SIZE + 255) / 256, 256, 0, stream>>>(
                (const __nv_bfloat16*)gate_out_bf16,
                (const __nv_bfloat16*)up_out_bf16,
                (__nv_bfloat16*)mlp_intermediate_bf16,
                INTERMEDIATE_SIZE
            );
        }
        dbg_end(dbg_silu);

        // 10) Down projection (or fused down+residual tail)
        int dbg_down = dbg_begin(SPLIT_STAGE_DOWN_GEMM, layer);
        bool down_done = false;
        bool down_lt_done = false;
        bool down_gemmex_done = false;
        bool down_gemv_done = false;
        bool down_try_lt = false;
        bool down_use_gemv = false;
        bool down_w4_done = false;
        bool down_w4_fast_exp = _split_down_w4_fast_exp_enabled();
        int down_w4_row_tile = _split_down_w4_row_tile();
        if (ffn_w4_fused_silu_down) {
            int grid_rows = (down_w4_row_tile == 4) ? ((HIDDEN_SIZE + 3) / 4)
                                                     : ((down_w4_row_tile == 2) ? ((HIDDEN_SIZE + 1) / 2) : HIDDEN_SIZE);
            if (down_w4_fast_exp) {
                if (down_w4_row_tile == 4) {
                    split_w4_silu_downproj_residual_rowtile_kernel_t<true, 4><<<grid_rows, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else if (down_w4_row_tile == 2) {
                    split_w4_silu_downproj_residual_rowtile_kernel_t<true, 2><<<grid_rows, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else {
                    split_w4_silu_downproj_residual_kernel_t<true><<<HIDDEN_SIZE, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                }
            } else {
                if (down_w4_row_tile == 4) {
                    split_w4_silu_downproj_residual_rowtile_kernel_t<false, 4><<<grid_rows, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else if (down_w4_row_tile == 2) {
                    split_w4_silu_downproj_residual_rowtile_kernel_t<false, 2><<<grid_rows, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                } else {
                    split_w4_silu_downproj_residual_kernel_t<false><<<HIDDEN_SIZE, 256, 0, stream>>>(
                        (const __nv_bfloat16*)gateup_out_bf16,
                        down_w4_packed,
                        down_w4_scales,
                        down_w4_codebook,
                        (const float*)residual_f32,
                        (float*)hidden_f32,
                        debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                    );
                }
            }
            down_done = true;
            down_w4_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.down_fused_silu_w4 += 1;
                agg.down_w4 += 1;
            }
        } else if (ffn_impl == SPLIT_FFN_FUSED_SILU_DOWN) {
            split_silu_downproj_residual_fused_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                (const __nv_bfloat16*)gateup_out_bf16,
                down_weight,
                (const float*)residual_f32,
                (float*)hidden_f32
            );
            down_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.down_fused_silu += 1;
            }
        } else if (ffn_fused_tail) {
            split_downproj_residual_fused_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                (const __nv_bfloat16*)mlp_intermediate_bf16,
                down_weight,
                (const float*)residual_f32,
                (float*)hidden_f32
            );
            down_done = true;
            if (debug_stage_avg) {
                auto& agg = _split_debug_agg();
                agg.down_fused_tail += 1;
            }
        } else {
            if (ffn_w4) {
                split_w4_downproj_residual_kernel<<<HIDDEN_SIZE, 256, 0, stream>>>(
                    down_w4_packed,
                    down_w4_scales,
                    down_w4_codebook,
                    (const __nv_bfloat16*)mlp_intermediate_bf16,
                    (const float*)residual_f32,
                    (float*)hidden_f32,
                    debug_ffn_w4_fine ? ffn_w4_fine_dev : nullptr
                );
                down_done = true;
                down_w4_done = true;
                if (debug_stage_avg) {
                    auto& agg = _split_debug_agg();
                    agg.down_w4 += 1;
                }
            } else {
                down_try_lt = _split_down_try_cublaslt();
                down_use_gemv = _split_down_use_gemv();
            }
            if (!down_done && !down_use_gemv && down_try_lt) {
                down_done = _split_lt_matmul(
                    cublaslt_handle,
                    down_lt_plan,
                    HIDDEN_SIZE,
                    INTERMEDIATE_SIZE,
                    down_weight,
                    (const __nv_bfloat16*)mlp_intermediate_bf16,
                    (__nv_bfloat16*)down_out_bf16,
                    cublaslt_workspace,
                    cublaslt_workspace_bytes,
                    stream
                );
                down_lt_done = down_done;
            }
            if (!down_done) {
                if (down_use_gemv) {
                    down_gemv_done = _split_try_gemv_matvec(
                        cublas_handle,
                        down_weight,
                        HIDDEN_SIZE,
                        INTERMEDIATE_SIZE,
                        (const __nv_bfloat16*)mlp_intermediate_bf16,
                        (__nv_bfloat16*)down_out_bf16,
                        "down_gemv"
                    );
                    down_done = down_gemv_done;
                }
                if (!down_done) {
                    down_gemmex_done = true;
                    if (!_cublas_check(cublasGemmEx(
                        cublas_handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        HIDDEN_SIZE, 1, INTERMEDIATE_SIZE,
                        &alpha,
                        down_weight, CUDA_R_16BF, INTERMEDIATE_SIZE,
                        mlp_intermediate_bf16, CUDA_R_16BF, INTERMEDIATE_SIZE,
                        &beta,
                        down_out_bf16, CUDA_R_16BF, HIDDEN_SIZE,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                    ), "down_gemm")) return;
                }
            }
            if (debug_stage_avg && !down_w4_done) {
                auto& agg = _split_debug_agg();
                if (!down_use_gemv && down_try_lt) {
                    if (down_lt_done) agg.down_lt_ok += 1;
                    else agg.down_lt_fb += 1;
                }
                if (down_gemv_done) agg.down_gemv += 1;
                else if (down_gemmex_done) agg.down_gemmex += 1;
            }
        }
        dbg_end(dbg_down);

        // 11) Residual add -> hidden
        int dbg_resid2 = dbg_begin(SPLIT_STAGE_RESID2, layer);
        if (!ffn_fused_tail && !down_w4_done) {
            split_residual_add_bf16_to_f32_kernel<<<(HIDDEN_SIZE + 255) / 256, 256, 0, stream>>>(
                (const __nv_bfloat16*)down_out_bf16,
                (const float*)residual_f32,
                (float*)hidden_f32,
                HIDDEN_SIZE
            );
        }
        dbg_end(dbg_resid2);
    }

    // Final RMSNorm
    int dbg_final_norm = dbg_begin(SPLIT_STAGE_FINAL_NORM);
    split_final_norm_f32_to_bf16_kernel<<<1, 256, 0, stream>>>(
        (const float*)hidden_f32,
        (const __nv_bfloat16*)final_norm_weight,
        (__nv_bfloat16*)final_hidden_bf16
    );
    dbg_end(dbg_final_norm);

    // LM head argmax (reuse prefill kernels)
    int dbg_lm1 = dbg_begin(SPLIT_STAGE_LM_PHASE1);
    prefill_lm_head_phase1<<<PREFILL_LM_NUM_BLOCKS, PREFILL_LM_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)final_hidden_bf16,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    dbg_end(dbg_lm1);

    int dbg_lm2 = dbg_begin(SPLIT_STAGE_LM_PHASE2);
    prefill_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_LM_NUM_BLOCKS
    );
    dbg_end(dbg_lm2);

    if (debug_stage) {
        cudaStreamSynchronize(stream);
        float stage_ms[SPLIT_STAGE_COUNT] = {0.0f};
        int stage_count[SPLIT_STAGE_COUNT] = {0};
        for (int i = 0; i < debug_event_count; i++) {
            if (debug_events[i].start == nullptr || debug_events[i].stop == nullptr) continue;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, debug_events[i].start, debug_events[i].stop);
            int st = debug_events[i].stage;
            if (st >= 0 && st < SPLIT_STAGE_COUNT) {
                stage_ms[st] += ms;
                stage_count[st] += 1;
            }
            int layer_id = debug_events[i].layer;
            if (st >= 0 && st < SPLIT_STAGE_COUNT && layer_id >= 0 && layer_id < SPLIT_DEBUG_MAX_LAYERS) {
                auto& agg = _split_debug_agg();
                agg.stage_layer_ms[st][layer_id] += double(ms);
                agg.stage_layer_calls[st][layer_id] += 1;
                long long seen = (long long)layer_id + 1;
                if (seen > agg.max_layer_seen) agg.max_layer_seen = seen;
            }
            cudaEventDestroy(debug_events[i].start);
            cudaEventDestroy(debug_events[i].stop);
        }

        float total_ms = 0.0f;
        for (int s = 0; s < SPLIT_STAGE_COUNT; s++) total_ms += stage_ms[s];

        if (debug_stage_avg && (debug_stage_avg_sample || debug_stage_detail)) {
            auto& agg = _split_debug_agg();
            agg.sampled_tokens += 1;
            agg.total_ms += double(total_ms);
            for (int s = 0; s < SPLIT_STAGE_COUNT; s++) {
                agg.stage_ms[s] += double(stage_ms[s]);
                agg.stage_calls[s] += static_cast<long long>(stage_count[s]);
            }
        }

        if (debug_stage_detail) {
            const char* impl = _split_use_cublaslt() ? "cublasLt(+fallback)" : "cublasGemmEx";
            const char* qkv_impl = _split_qkv_impl_name();
            const char* attn_impl = _split_attn_impl_name();
            float wk_mib = float(cublaslt_workspace_bytes) / (1024.0f * 1024.0f);
            printf("[MEGAQWEN_DEBUG] SPLIT stage timing (first decode token) impl=%s qkv=%s attn=%s chunk=%d chunks=%d layers=%d flash_gqa_share=%d flash_gqa_mode=%s flash_fp8_tc_qk=%d lt_workspace=%.1f MiB\n",
                   impl, qkv_impl, attn_impl, split_attn_chunk_size, split_attn_max_chunks, num_layers,
                   _split_flash_gqa_share_enabled() ? 1 : 0,
                   _split_flash_gqa_mode_name(),
                   _split_flash_fp8_tc_qk_enabled() ? 1 : 0, wk_mib);
            for (int s = 0; s < SPLIT_STAGE_COUNT; s++) {
                if (stage_count[s] == 0) continue;
                float pct = (total_ms > 0.0f) ? (stage_ms[s] * 100.0f / total_ms) : 0.0f;
                float avg_ms = stage_ms[s] / float(stage_count[s]);
                printf("  %-16s total=%8.3f ms  avg=%7.4f ms  calls=%3d  (%5.1f%%)\n",
                       kSplitStageNames[s], stage_ms[s], avg_ms, stage_count[s], pct);
            }
            printf("  %-16s total=%8.3f ms\n", "decode_step_sum", total_ms);
        }
    }

    if (debug_ffn_w4_fine && ffn_w4_fine_dev != nullptr) {
        SplitFfnW4FineDebugRec rec{};
        cudaMemcpyAsync(&rec, ffn_w4_fine_dev, sizeof(rec), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        auto& agg = _split_debug_agg();
        agg.ffn_w4_fine_samples += 1;
        if (debug_ffn_w4_clock_khz > 0) {
            agg.ffn_w4_clock_khz = static_cast<long long>(debug_ffn_w4_clock_khz);
        }
        agg.gateup_dequant_cycles += rec.gateup_dequant_cycles;
        agg.gateup_fma_cycles += rec.gateup_fma_cycles;
        agg.gateup_reduce_cycles += rec.gateup_reduce_cycles;
        agg.down_dequant_cycles += rec.down_dequant_cycles;
        agg.down_act_cycles += rec.down_act_cycles;
        agg.down_fma_cycles += rec.down_fma_cycles;
    }
}
