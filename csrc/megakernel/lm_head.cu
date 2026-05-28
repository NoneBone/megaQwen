// megakernel/lm_head.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define HIDDEN 1024

extern "C" __global__ void rmsnorm_final_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int seq_len
) {
    int tid = threadIdx.x;
    if (tid < HIDDEN) {
        output[tid] = input[tid];
    }
}

extern "C" __global__ void lm_head_last_token_kernel(
    const __nv_bfloat16* normalized,
    const __nv_bfloat16* weight,
    __nv_bfloat16* logits,
    int vocab_size
) {
    int tid = threadIdx.x;
    if (tid < vocab_size) {
        logits[tid] = __float2bfloat16(0.0f);
    }
}

extern "C" void launch_rmsnorm_final(
    const void* input, const void* weight, void* output,
    int seq_len, cudaStream_t stream
) {
    rmsnorm_final_kernel<<<1, 256, 0, stream>>>(
        (__nv_bfloat16*)input,
        (__nv_bfloat16*)weight,
        (__nv_bfloat16*)output,
        seq_len
    );
}

extern "C" void launch_lm_head_last_token(
    const void* normalized, const void* weight, void* logits,
    int seq_len, cudaStream_t stream
) {
    constexpr int VOCAB = 151936;
    lm_head_last_token_kernel<<<1, 256, 0, stream>>>(
        (__nv_bfloat16*)normalized,
        (__nv_bfloat16*)weight,
        (__nv_bfloat16*)logits,
        VOCAB
    );
}