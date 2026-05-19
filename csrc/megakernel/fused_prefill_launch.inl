// =============================================================================
// Main prefill launch function (all BF16)
// =============================================================================

static inline bool prefill_causal_attention_flash_ext_sdpa(
    void* q_proj,
    void* k_proj,
    void* v_proj,
    void* attn_out,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float attn_scale,
    bool force_fp32
) {
    try {
        int dev = 0;
        cudaGetDevice(&dev);
        auto opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, dev);

        auto q3 = torch::from_blob(q_proj, {seq_len, num_q_heads, head_dim}, opts);
        auto k3 = torch::from_blob(k_proj, {seq_len, num_kv_heads, head_dim}, opts);
        auto v3 = torch::from_blob(v_proj, {seq_len, num_kv_heads, head_dim}, opts);
        auto out3 = torch::from_blob(attn_out, {seq_len, num_q_heads, head_dim}, opts);

        auto q4 = q3.permute({1, 0, 2}).unsqueeze(0);
        auto k4 = k3.permute({1, 0, 2}).unsqueeze(0);
        auto v4 = v3.permute({1, 0, 2}).unsqueeze(0);

        bool enable_gqa = (num_q_heads != num_kv_heads);
        at::Tensor q4_attn = q4;
        at::Tensor k4_attn = k4;
        at::Tensor v4_attn = v4;
        if (force_fp32) {
            q4_attn = q4_attn.to(torch::kFloat32);
            k4_attn = k4_attn.to(torch::kFloat32);
            v4_attn = v4_attn.to(torch::kFloat32);
        }

        auto out4 = at::scaled_dot_product_attention(
            q4_attn,
            k4_attn,
            v4_attn,
            c10::nullopt,
            0.0,
            true,
            std::optional<double>(attn_scale)
        );// enable_gqa

        auto out_write = out4;
        if (out_write.scalar_type() != torch::kBFloat16) {
            out_write = out_write.to(torch::kBFloat16);
        }
        out3.copy_(out_write.squeeze(0).permute({1, 0, 2}));
        return true;
    } catch (const c10::Error& e) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf(
                "[MEGAQWEN_PREFILL] flash_ext(sdpa) failed once and will fallback to legacy attention: %s\n",
                e.what_without_backtrace()
            );
        }
        return false;
    }
}

extern "C" void launch_prefill_float(
    const int* input_token_ids,      // [seq_len] on device
    int* output_token_id,            // [1] on device
    int seq_len,
    const void* embed_weight,
    const void* audio_embeds,        // [audio_len, hidden_size] bf16 on device (optional)
    int audio_start_idx,             // start row in sequence to patch
    int audio_len,                   // number of rows to patch
    const PrefillLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    // BF16 buffers (renamed from float for compatibility)
    void* hidden_bf16,       // [seq_len, hidden_size] bf16
    void* residual,          // [seq_len, hidden_size] bf16
    void* normalized,        // [seq_len, hidden_size] bf16
    void* q_proj,            // [seq_len, q_size] bf16
    void* k_proj,            // [seq_len, kv_size] bf16
    void* v_proj,            // [seq_len, kv_size] bf16
    void* attn_out,          // [seq_len, q_size] bf16
    void* o_proj_out,        // [seq_len, hidden_size] bf16
    void* mlp_norm,          // [seq_len, hidden_size] bf16
    void* gate_out,          // [seq_len, intermediate_size] bf16
    void* up_out,            // [seq_len, intermediate_size] bf16
    void* mlp_intermediate,  // [seq_len, intermediate_size] bf16
    void* down_out,          // [seq_len, hidden_size] bf16
    void* final_hidden,      // [hidden_size] bf16
    void* block_max_vals,
    void* block_max_idxs,
    void* hidden_bf16_out,   // [hidden_size] bf16 - for decode continuation
    int num_layers,
    int max_seq_len,
    float attn_scale,
    cublasHandle_t cublas_handle,
    cudaStream_t stream
) {
    // cuBLAS parameters for BF16 x BF16 -> BF16 with FP32 accumulation
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    constexpr int kMaxDebugEvents = 16384;
    PrefillDebugEventRec debug_events[kMaxDebugEvents];
    int debug_event_count = 0;
    bool debug_stage = _prefill_debug_take_ticket();
    int attn_impl_mode = _prefill_attn_impl_mode();
    bool splitk_requested = (attn_impl_mode == 1) || (attn_impl_mode == 2 && seq_len >= _prefill_attn_auto_min_seq());
    bool flash_requested = (attn_impl_mode == 3);
    bool flash_ext_requested = (attn_impl_mode == 4);
    bool splitk_experimental = _prefill_attn_experimental_enabled();
    bool flash_experimental = splitk_experimental;
    bool use_splitk_attn = splitk_requested && splitk_experimental;
    bool use_flash_attn = flash_requested && flash_experimental;
    bool use_flash_ext_attn = flash_ext_requested && flash_experimental;
    bool flash_ext_force_fp32 = _prefill_flash_ext_force_fp32();
    int flash_ext_tail_legacy_layers = _prefill_flash_ext_tail_legacy_layers();
    if (flash_ext_tail_legacy_layers < 0) flash_ext_tail_legacy_layers = 0;
    if (flash_ext_tail_legacy_layers > num_layers) flash_ext_tail_legacy_layers = num_layers;
    bool debug_flash = use_flash_attn && _prefill_flash_debug_take_ticket();
    int legacy_warps = _prefill_legacy_attn_warps();
    int splitk_warps = _prefill_attn_warps();
    if (splitk_requested && !splitk_experimental) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf(
                "[MEGAQWEN_PREFILL] splitk attention is experimental and disabled by default; "
                "falling back to legacy. Set MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL=1 to enable.\n"
            );
        }
    }
    if (flash_requested && !flash_experimental) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf(
                "[MEGAQWEN_PREFILL] flash attention is experimental and disabled by default; "
                "falling back to legacy. Set MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL=1 to enable.\n"
            );
        }
    }
    if (flash_ext_requested && !flash_experimental) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf(
                "[MEGAQWEN_PREFILL] flash_ext attention is experimental and disabled by default; "
                "falling back to legacy. Set MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL=1 to enable.\n"
            );
        }
    }
    if (use_flash_ext_attn && flash_ext_tail_legacy_layers > 0) {
        static bool informed = false;
        if (!informed) {
            informed = true;
            std::printf(
                "[MEGAQWEN_PREFILL] flash_ext hybrid enabled: tail %d layer(s) use legacy attention for quality.\n",
                flash_ext_tail_legacy_layers
            );
        }
    }
    const int total_attn_elems = seq_len * Q_SIZE;
    __nv_bfloat16* flash_ref_attn = nullptr;
    double *dbg_sum_abs_diff = nullptr, *dbg_sum_sq_diff = nullptr, *dbg_sum_abs_ref = nullptr;
    unsigned int* dbg_max_abs_bits = nullptr;
    int *dbg_bad = nullptr, *dbg_gt_1e2 = nullptr, *dbg_gt_5e2 = nullptr, *dbg_gt_1e1 = nullptr;
    float* flash_ms_per_layer = nullptr;
    float* legacy_ms_per_layer = nullptr;
    double* mae_per_layer = nullptr;
    double* rel_l1_per_layer = nullptr;
    int* bad_per_layer = nullptr;
    if (debug_flash) {
        bool alloc_ok = true;
        auto _alloc = [&](void** p, size_t n, const char* name) {
            if (cudaMalloc(p, n) != cudaSuccess) {
                std::printf("[MEGAQWEN_DEBUG] flash debug alloc failed: %s (%zu bytes)\n", name, n);
                alloc_ok = false;
            }
        };
        _alloc((void**)&flash_ref_attn, (size_t)total_attn_elems * sizeof(__nv_bfloat16), "flash_ref_attn");
        _alloc((void**)&dbg_sum_abs_diff, sizeof(double), "dbg_sum_abs_diff");
        _alloc((void**)&dbg_sum_sq_diff, sizeof(double), "dbg_sum_sq_diff");
        _alloc((void**)&dbg_sum_abs_ref, sizeof(double), "dbg_sum_abs_ref");
        _alloc((void**)&dbg_max_abs_bits, sizeof(unsigned int), "dbg_max_abs_bits");
        _alloc((void**)&dbg_bad, sizeof(int), "dbg_bad");
        _alloc((void**)&dbg_gt_1e2, sizeof(int), "dbg_gt_1e2");
        _alloc((void**)&dbg_gt_5e2, sizeof(int), "dbg_gt_5e2");
        _alloc((void**)&dbg_gt_1e1, sizeof(int), "dbg_gt_1e1");
        if (!alloc_ok) {
            debug_flash = false;
        } else {
            flash_ms_per_layer = new float[num_layers]();
            legacy_ms_per_layer = new float[num_layers]();
            mae_per_layer = new double[num_layers]();
            rel_l1_per_layer = new double[num_layers]();
            bad_per_layer = new int[num_layers]();
            std::printf(
                "[MEGAQWEN_DEBUG] PREFILL_FLASH debug on: seq_len=%d layers=%d elems/layer=%d head_dim=%d q_heads=%d kv_heads=%d\n",
                seq_len, num_layers, total_attn_elems, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS
            );
        }
    }

    auto dbg_begin = [&](int stage) -> int {
        if (!debug_stage) return -1;
        if (debug_event_count >= kMaxDebugEvents) return -1;
        int idx = debug_event_count++;
        debug_events[idx].stage = stage;
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

    cublasSetStream(cublas_handle, stream);

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    // Embedding lookup (to BF16)
    int dbg_embed = dbg_begin(PREFILL_STAGE_EMBED_PATCH);
    prefill_embed_kernel<<<seq_len, 256, 0, stream>>>(
        input_token_ids,
        (const __nv_bfloat16*)embed_weight,
        (__nv_bfloat16*)hidden_bf16,
        seq_len,
        HIDDEN_SIZE
    );

    // Patch audio span (optional).
    if (audio_embeds != nullptr && audio_len > 0) {
        prefill_patch_audio_kernel<<<audio_len, 256, 0, stream>>>(
            (__nv_bfloat16*)hidden_bf16,
            (const __nv_bfloat16*)audio_embeds,
            audio_start_idx,
            audio_len,
            HIDDEN_SIZE
        );
    }
    dbg_end(dbg_embed);

    for (int layer = 0; layer < num_layers; layer++) {
        const PrefillLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = (__nv_bfloat16*)k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = (__nv_bfloat16*)v_cache + layer * kv_cache_layer_stride;

        // 1. Input LayerNorm (bf16 -> bf16, save residual)
        int dbg_rms1 = dbg_begin(PREFILL_STAGE_RMS1);
        prefill_rmsnorm_kernel<<<seq_len, PREFILL_BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)hidden_bf16,
            (const __nv_bfloat16*)w.input_layernorm_weight,
            (__nv_bfloat16*)normalized,
            (__nv_bfloat16*)residual,
            seq_len,
            HIDDEN_SIZE
        );
        dbg_end(dbg_rms1);

        // 2. Q Projection: [seq_len, hidden] @ [q_size, hidden]^T -> [seq_len, q_size]
        // cuBLAS col-major: C = alpha * op(A) * op(B) + beta * C
        // Row-major: C[seq,q] = input[seq,hid] @ W[q,hid]^T
        // Col-major: C^T[q,seq] = W[q,hid] @ input^T[hid,seq]
        // => cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, q_size, seq_len, hidden, W, input, C)
        int dbg_qkv = dbg_begin(PREFILL_STAGE_QKV_GEMM);
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            Q_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.q_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            q_proj, CUDA_R_16BF, Q_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 3. K Projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            KV_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.k_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            k_proj, CUDA_R_16BF, KV_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 4. V Projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            KV_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.v_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            normalized, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            v_proj, CUDA_R_16BF, KV_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
        dbg_end(dbg_qkv);

        // 5. QK Norm + RoPE + KV Cache
        int dbg_rope = dbg_begin(PREFILL_STAGE_QK_ROPE_CACHE);
        dim3 grid_rope(seq_len, max(NUM_Q_HEADS, NUM_KV_HEADS));
        prefill_qk_norm_rope_kernel<<<grid_rope, 128, 0, stream>>>(
            (__nv_bfloat16*)q_proj,
            (__nv_bfloat16*)k_proj,
            (const __nv_bfloat16*)v_proj,
            (const __nv_bfloat16*)w.q_norm_weight,
            (const __nv_bfloat16*)w.k_norm_weight,
            (const __nv_bfloat16*)cos_table,
            (const __nv_bfloat16*)sin_table,
            layer_k_cache,
            layer_v_cache,
            seq_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            max_seq_len,
            0  // position_offset = 0 for prefill
        );
        dbg_end(dbg_rope);

        // 6. Causal Attention
        int dbg_attn = dbg_begin(PREFILL_STAGE_ATTN);
        int total_queries = seq_len * NUM_Q_HEADS;
        bool layer_use_flash_ext = use_flash_ext_attn && (layer < (num_layers - flash_ext_tail_legacy_layers));
        if (layer_use_flash_ext) {
            bool ok = prefill_causal_attention_flash_ext_sdpa(
                q_proj,
                k_proj,
                v_proj,
                attn_out,
                seq_len,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                attn_scale,
                flash_ext_force_fp32
            );
            if (!ok) {
                int threads_per_block = legacy_warps * WARP_SIZE;
                int num_blocks_attn = (total_queries + legacy_warps - 1) / legacy_warps;
                prefill_causal_attention_kernel<<<num_blocks_attn, threads_per_block, 0, stream>>>(
                    (const __nv_bfloat16*)q_proj,
                    (const __nv_bfloat16*)k_proj,
                    (const __nv_bfloat16*)v_proj,
                    (__nv_bfloat16*)attn_out,
                    seq_len,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    attn_scale
                );
            }
        } else if (use_flash_attn) {
            int threads_per_block = 128;
            int num_blocks_attn = total_queries;
            cudaEvent_t ev_flash_begin = nullptr, ev_flash_end = nullptr;
            cudaEvent_t ev_legacy_begin = nullptr, ev_legacy_end = nullptr;
            if (debug_flash) {
                cudaEventCreateWithFlags(&ev_flash_begin, cudaEventDefault);
                cudaEventCreateWithFlags(&ev_flash_end, cudaEventDefault);
                cudaEventCreateWithFlags(&ev_legacy_begin, cudaEventDefault);
                cudaEventCreateWithFlags(&ev_legacy_end, cudaEventDefault);
                cudaEventRecord(ev_flash_begin, stream);
            }
            prefill_causal_attention_flash_tile_kernel<<<num_blocks_attn, threads_per_block, 0, stream>>>(
                (const __nv_bfloat16*)q_proj,
                (const __nv_bfloat16*)k_proj,
                (const __nv_bfloat16*)v_proj,
                (__nv_bfloat16*)attn_out,
                seq_len,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                attn_scale
            );
            if (debug_flash) {
                cudaEventRecord(ev_flash_end, stream);

                int ref_threads = legacy_warps * WARP_SIZE;
                int ref_blocks = (total_queries + legacy_warps - 1) / legacy_warps;
                cudaEventRecord(ev_legacy_begin, stream);
                prefill_causal_attention_kernel<<<ref_blocks, ref_threads, 0, stream>>>(
                    (const __nv_bfloat16*)q_proj,
                    (const __nv_bfloat16*)k_proj,
                    (const __nv_bfloat16*)v_proj,
                    flash_ref_attn,
                    seq_len,
                    NUM_Q_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    attn_scale
                );
                cudaEventRecord(ev_legacy_end, stream);

                cudaMemsetAsync(dbg_sum_abs_diff, 0, sizeof(double), stream);
                cudaMemsetAsync(dbg_sum_sq_diff, 0, sizeof(double), stream);
                cudaMemsetAsync(dbg_sum_abs_ref, 0, sizeof(double), stream);
                cudaMemsetAsync(dbg_max_abs_bits, 0, sizeof(unsigned int), stream);
                cudaMemsetAsync(dbg_bad, 0, sizeof(int), stream);
                cudaMemsetAsync(dbg_gt_1e2, 0, sizeof(int), stream);
                cudaMemsetAsync(dbg_gt_5e2, 0, sizeof(int), stream);
                cudaMemsetAsync(dbg_gt_1e1, 0, sizeof(int), stream);
                int cmp_block = 256;
                int cmp_grid = (total_attn_elems + cmp_block - 1) / cmp_block;
                prefill_attn_compare_stats_kernel<<<cmp_grid, cmp_block, 0, stream>>>(
                    (const __nv_bfloat16*)attn_out,
                    (const __nv_bfloat16*)flash_ref_attn,
                    total_attn_elems,
                    dbg_sum_abs_diff,
                    dbg_sum_sq_diff,
                    dbg_sum_abs_ref,
                    dbg_max_abs_bits,
                    dbg_bad,
                    dbg_gt_1e2,
                    dbg_gt_5e2,
                    dbg_gt_1e1
                );

                cudaStreamSynchronize(stream);
                float flash_ms = -1.0f, legacy_ms = -1.0f;
                cudaEventElapsedTime(&flash_ms, ev_flash_begin, ev_flash_end);
                cudaEventElapsedTime(&legacy_ms, ev_legacy_begin, ev_legacy_end);
                cudaEventDestroy(ev_flash_begin);
                cudaEventDestroy(ev_flash_end);
                cudaEventDestroy(ev_legacy_begin);
                cudaEventDestroy(ev_legacy_end);

                double h_sum_abs = 0.0, h_sum_sq = 0.0, h_sum_ref = 0.0;
                unsigned int h_max_bits = 0;
                int h_bad = 0, h_gt_1e2 = 0, h_gt_5e2 = 0, h_gt_1e1 = 0;
                cudaMemcpy(&h_sum_abs, dbg_sum_abs_diff, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_sum_sq, dbg_sum_sq_diff, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_sum_ref, dbg_sum_abs_ref, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_max_bits, dbg_max_abs_bits, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_bad, dbg_bad, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_gt_1e2, dbg_gt_1e2, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_gt_5e2, dbg_gt_5e2, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_gt_1e1, dbg_gt_1e1, sizeof(int), cudaMemcpyDeviceToHost);

                double denom = (double)total_attn_elems;
                double mae = h_sum_abs / (denom > 0.0 ? denom : 1.0);
                double rmse = std::sqrt(h_sum_sq / (denom > 0.0 ? denom : 1.0));
                double rel_l1 = h_sum_abs / (h_sum_ref + 1e-12);
                float max_abs = 0.0f;
                std::memcpy(&max_abs, &h_max_bits, sizeof(float));
                flash_ms_per_layer[layer] = flash_ms;
                legacy_ms_per_layer[layer] = legacy_ms;
                mae_per_layer[layer] = mae;
                rel_l1_per_layer[layer] = rel_l1;
                bad_per_layer[layer] = h_bad;

                std::printf(
                    "[MEGAQWEN_DEBUG] PREFILL_FLASH layer=%02d flash_ms=%.3f legacy_ms=%.3f mae=%.6e rmse=%.6e max_abs=%.6e rel_l1=%.6e bad=%d gt1e-2=%d gt5e-2=%d gt1e-1=%d\n",
                    layer, flash_ms, legacy_ms, mae, rmse, (double)max_abs, rel_l1, h_bad, h_gt_1e2, h_gt_5e2, h_gt_1e1
                );
            }
        } else if (use_splitk_attn) {
            int threads_per_block = splitk_warps * WARP_SIZE;
            int num_blocks_attn = total_queries;
            prefill_causal_attention_splitk_kernel<<<num_blocks_attn, threads_per_block, 0, stream>>>(
                (const __nv_bfloat16*)q_proj,
                (const __nv_bfloat16*)k_proj,
                (const __nv_bfloat16*)v_proj,
                (__nv_bfloat16*)attn_out,
                seq_len,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                attn_scale,
                splitk_warps
            );
        } else {
            int threads_per_block = legacy_warps * WARP_SIZE;
            int num_blocks_attn = (total_queries + legacy_warps - 1) / legacy_warps;
            prefill_causal_attention_kernel<<<num_blocks_attn, threads_per_block, 0, stream>>>(
                (const __nv_bfloat16*)q_proj,
                (const __nv_bfloat16*)k_proj,
                (const __nv_bfloat16*)v_proj,
                (__nv_bfloat16*)attn_out,
                seq_len,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                attn_scale
            );
        }
        dbg_end(dbg_attn);

        // 7. O Projection: [seq_len, q_size] @ [hidden, q_size]^T -> [seq_len, hidden]
        int dbg_o = dbg_begin(PREFILL_STAGE_O_GEMM);
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            HIDDEN_SIZE, seq_len, Q_SIZE,
            &alpha_f,
            w.o_proj_weight, CUDA_R_16BF, Q_SIZE,
            attn_out, CUDA_R_16BF, Q_SIZE,
            &beta_f,
            o_proj_out, CUDA_R_16BF, HIDDEN_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
        dbg_end(dbg_o);

        // 8. Residual add
        int dbg_res1 = dbg_begin(PREFILL_STAGE_RESID1);
        int total_hidden = seq_len * HIDDEN_SIZE;
        int block_size_res = 256;
        int num_blocks_res = (total_hidden + block_size_res - 1) / block_size_res;
        prefill_residual_add_kernel<<<num_blocks_res, block_size_res, 0, stream>>>(
            (const __nv_bfloat16*)o_proj_out,
            (const __nv_bfloat16*)residual,
            (__nv_bfloat16*)hidden_bf16,
            total_hidden
        );
        dbg_end(dbg_res1);

        // 9. Post-attention LayerNorm (save residual)
        int dbg_rms2 = dbg_begin(PREFILL_STAGE_RMS2);
        prefill_rmsnorm_kernel<<<seq_len, PREFILL_BLOCK_SIZE, 0, stream>>>(
            (__nv_bfloat16*)hidden_bf16,
            (const __nv_bfloat16*)w.post_attn_layernorm_weight,
            (__nv_bfloat16*)mlp_norm,
            (__nv_bfloat16*)residual,
            seq_len,
            HIDDEN_SIZE
        );
        dbg_end(dbg_rms2);

        // 10. Gate projection
        int dbg_gateup = dbg_begin(PREFILL_STAGE_GATEUP_GEMM);
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            INTERMEDIATE_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.gate_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            mlp_norm, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            gate_out, CUDA_R_16BF, INTERMEDIATE_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );

        // 11. Up projection
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            INTERMEDIATE_SIZE, seq_len, HIDDEN_SIZE,
            &alpha_f,
            w.up_proj_weight, CUDA_R_16BF, HIDDEN_SIZE,
            mlp_norm, CUDA_R_16BF, HIDDEN_SIZE,
            &beta_f,
            up_out, CUDA_R_16BF, INTERMEDIATE_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
        dbg_end(dbg_gateup);

        // 12. SiLU + multiply
        int dbg_silu = dbg_begin(PREFILL_STAGE_SILU);
        int total_inter = seq_len * INTERMEDIATE_SIZE;
        int num_blocks_silu = (total_inter + 255) / 256;
        prefill_silu_mul_kernel<<<num_blocks_silu, 256, 0, stream>>>(
            (const __nv_bfloat16*)gate_out,
            (const __nv_bfloat16*)up_out,
            (__nv_bfloat16*)mlp_intermediate,
            total_inter
        );
        dbg_end(dbg_silu);

        // 13. Down projection
        int dbg_down = dbg_begin(PREFILL_STAGE_DOWN_GEMM);
        cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            HIDDEN_SIZE, seq_len, INTERMEDIATE_SIZE,
            &alpha_f,
            w.down_proj_weight, CUDA_R_16BF, INTERMEDIATE_SIZE,
            mlp_intermediate, CUDA_R_16BF, INTERMEDIATE_SIZE,
            &beta_f,
            down_out, CUDA_R_16BF, HIDDEN_SIZE,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
        );
        dbg_end(dbg_down);

        // 14. Residual add
        int dbg_res2 = dbg_begin(PREFILL_STAGE_RESID2);
        prefill_residual_add_kernel<<<num_blocks_res, block_size_res, 0, stream>>>(
            (const __nv_bfloat16*)down_out,
            (const __nv_bfloat16*)residual,
            (__nv_bfloat16*)hidden_bf16,
            total_hidden
        );
        dbg_end(dbg_res2);
    }

    // Final norm (only last token)
    int dbg_final_norm = dbg_begin(PREFILL_STAGE_FINAL_NORM);
    prefill_final_norm_kernel<<<1, PREFILL_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)hidden_bf16,
        (const __nv_bfloat16*)final_norm_weight,
        (__nv_bfloat16*)final_hidden,
        seq_len,
        HIDDEN_SIZE
    );
    dbg_end(dbg_final_norm);

    // Copy final hidden to output for decode continuation
    int dbg_hidden_copy = dbg_begin(PREFILL_STAGE_HIDDEN_COPY);
    cudaMemcpyAsync(
        hidden_bf16_out,
        final_hidden,
        HIDDEN_SIZE * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice,
        stream
    );
    dbg_end(dbg_hidden_copy);

    // LM Head
    int dbg_lm1 = dbg_begin(PREFILL_STAGE_LM1);
    prefill_lm_head_phase1<<<PREFILL_LM_NUM_BLOCKS, PREFILL_LM_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)final_hidden,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    dbg_end(dbg_lm1);

    int dbg_lm2 = dbg_begin(PREFILL_STAGE_LM2);
    prefill_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        PREFILL_LM_NUM_BLOCKS
    );
    dbg_end(dbg_lm2);

    if (debug_flash) {
        int worst_layer = -1;
        double worst_rel = -1.0;
        double mean_rel = 0.0;
        double mean_mae = 0.0;
        int bad_total = 0;
        for (int i = 0; i < num_layers; i++) {
            mean_rel += rel_l1_per_layer[i];
            mean_mae += mae_per_layer[i];
            bad_total += bad_per_layer[i];
            if (rel_l1_per_layer[i] > worst_rel) {
                worst_rel = rel_l1_per_layer[i];
                worst_layer = i;
            }
        }
        if (num_layers > 0) {
            mean_rel /= (double)num_layers;
            mean_mae /= (double)num_layers;
        }
        std::printf(
            "[MEGAQWEN_DEBUG] PREFILL_FLASH summary worst_layer=%d worst_rel_l1=%.6e avg_rel_l1=%.6e avg_mae=%.6e bad_total=%d\n",
            worst_layer, worst_rel, mean_rel, mean_mae, bad_total
        );
    }

    if (debug_stage) {
        cudaStreamSynchronize(stream);
        float stage_ms[PREFILL_STAGE_COUNT] = {0.0f};
        int stage_count[PREFILL_STAGE_COUNT] = {0};
        for (int i = 0; i < debug_event_count; i++) {
            if (debug_events[i].start == nullptr || debug_events[i].stop == nullptr) continue;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, debug_events[i].start, debug_events[i].stop);
            int st = debug_events[i].stage;
            if (st >= 0 && st < PREFILL_STAGE_COUNT) {
                stage_ms[st] += ms;
                stage_count[st] += 1;
            }
            cudaEventDestroy(debug_events[i].start);
            cudaEventDestroy(debug_events[i].stop);
        }
        float total_ms = 0.0f;
        for (int s = 0; s < PREFILL_STAGE_COUNT; s++) total_ms += stage_ms[s];
        printf(
            "[MEGAQWEN_DEBUG] PREFILL(cublas) stage timing seq_len=%d layers=%d attn=%s legacy_warps=%d splitk_warps=%d auto_min_seq=%d exp=%d flash_ext_fp32=%d flash_ext_tail_legacy=%d\n",
            seq_len, num_layers, _prefill_attn_impl_name(use_splitk_attn, use_flash_attn, use_flash_ext_attn), legacy_warps, splitk_warps, _prefill_attn_auto_min_seq(),
            splitk_experimental ? 1 : 0, flash_ext_force_fp32 ? 1 : 0, flash_ext_tail_legacy_layers
        );
        for (int s = 0; s < PREFILL_STAGE_COUNT; s++) {
            if (stage_count[s] == 0) continue;
            float pct = (total_ms > 0.0f) ? (stage_ms[s] * 100.0f / total_ms) : 0.0f;
            float avg_ms = stage_ms[s] / float(stage_count[s]);
            printf("  %-16s total=%8.3f ms  avg=%7.4f ms  calls=%3d  (%5.1f%%)\n",
                   kPrefillStageNames[s], stage_ms[s], avg_ms, stage_count[s], pct);
        }
        printf("  %-16s total=%8.3f ms\n", "prefill_sum", total_ms);
    }

    if (flash_ref_attn != nullptr) cudaFree(flash_ref_attn);
    if (dbg_sum_abs_diff != nullptr) cudaFree(dbg_sum_abs_diff);
    if (dbg_sum_sq_diff != nullptr) cudaFree(dbg_sum_sq_diff);
    if (dbg_sum_abs_ref != nullptr) cudaFree(dbg_sum_abs_ref);
    if (dbg_max_abs_bits != nullptr) cudaFree(dbg_max_abs_bits);
    if (dbg_bad != nullptr) cudaFree(dbg_bad);
    if (dbg_gt_1e2 != nullptr) cudaFree(dbg_gt_1e2);
    if (dbg_gt_5e2 != nullptr) cudaFree(dbg_gt_5e2);
    if (dbg_gt_1e1 != nullptr) cudaFree(dbg_gt_1e1);
    delete[] flash_ms_per_layer;
    delete[] legacy_ms_per_layer;
    delete[] mae_per_layer;
    delete[] rel_l1_per_layer;
    delete[] bad_per_layer;
}
