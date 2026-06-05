"""
End-to-end demo comparing Torch vs Triton vs CUDA backends for Qwen3-0.6B.
Streams tokens to screen as they are generated.
"""

import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
import triton
import triton.language as tl
from kernels import get_kernels
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

class Attn_method:
    use_flash_attn = 0
    PREFILL = {"v1":"naive","v2":"cuda","v3":"flash"}
    DECODE = {"v1":"v1","v2":"best","v3":"largeBatch", "v4":"splitk", "paged_splitk":"paged"}

@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960


# ---------------------------------------------------------------------------
# Helper utilities (new)
# ---------------------------------------------------------------------------

def make_input_ids(tokenizer: AutoTokenizer, length: int, device: torch.device) -> torch.Tensor:
    """
    Build a dummy ``input_ids`` tensor that contains exactly ``length`` tokens.
    We simply repeat the token for the word ``"Hello"`` – it is guaranteed to be
    in the vocab of Qwen‑3‑0.6B.
    """
    token = tokenizer.encode("Hello", add_special_tokens=False)[0]
    return torch.full((1, length), token, dtype=torch.long, device=device)

def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)


def gpu_mem_snapshot(device: int) -> Dict[str, int]:
    total = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return {
        "total": total,
        "reserved": reserved,
        "allocated": allocated,
        "free": total - reserved,
    }


def add_mem_stats(stats: Dict[str, object], prefix: str, snapshot: Dict[str, int]) -> None:
    stats[f"{prefix}_allocated_mib"] = bytes_to_mib(snapshot["allocated"])
    stats[f"{prefix}_reserved_mib"] = bytes_to_mib(snapshot["reserved"])
    stats[f"{prefix}_free_mib"] = bytes_to_mib(snapshot["free"])

def benchmark_model(
    model,                     # Qwen3Model instance
    input_ids: torch.Tensor,   # prompt tensor
    max_new_tokens: int,       # number of decode steps
) -> Tuple[int, Dict[str, object]]:
    """
    Run a full generation (prefill + decode) and return:
        * total generated tokens (including prompt)
        * a dict with the detailed timings:
            - prompt_tokens
            - prefill_time (ms)
            - prefill_speed (tok/s)
            - decode_steps
            - decode_time (ms)
            - decode_speed (tok/s)
    """
    device = torch.cuda.current_device()
    stats: Dict[str, object] = {}

    torch.cuda.synchronize()
    init_mem = gpu_mem_snapshot(device)
    add_mem_stats(stats, "init", init_mem)

    # ---------- Prefill ----------
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, kv_caches, prompt_len = model.prefill(input_ids, reserve_decode_steps=max_new_tokens)
    torch.cuda.synchronize()
    prefill_time = (time.perf_counter() - t0) * 1e3          # ms

    prefill_mem = gpu_mem_snapshot(device)
    add_mem_stats(stats, "prefill_end", prefill_mem)
    prefill_peak_allocated = torch.cuda.max_memory_allocated(device)
    prefill_peak_reserved = torch.cuda.max_memory_reserved(device)
    stats["prefill_peak_allocated_mib"] = bytes_to_mib(prefill_peak_allocated)
    stats["prefill_peak_reserved_mib"] = bytes_to_mib(prefill_peak_reserved)

    # ---------- Decode ----------
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    decode_steps = max_new_tokens
    decode_start_mem = gpu_mem_snapshot(device)
    add_mem_stats(stats, "decode_start", decode_start_mem)

    torch.cuda.reset_peak_memory_stats(device)
    decode_peak_allocated = torch.cuda.max_memory_allocated(device)
    decode_peak_reserved = torch.cuda.max_memory_reserved(device)
    decode_peak_step = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(decode_steps):
        logits, kv_caches = model.decode_step(
            next_token, kv_caches, prompt_len + step
        )
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        current_peak_allocated = torch.cuda.max_memory_allocated(device)
        current_peak_reserved = torch.cuda.max_memory_reserved(device)
        if (
            current_peak_allocated > decode_peak_allocated
            or current_peak_reserved > decode_peak_reserved
        ):
            decode_peak_allocated = current_peak_allocated
            decode_peak_reserved = current_peak_reserved
            decode_peak_step = step + 1

    torch.cuda.synchronize()
    decode_time = (time.perf_counter() - t0) * 1e3          # ms
    decode_end_mem = gpu_mem_snapshot(device)
    add_mem_stats(stats, "decode_end", decode_end_mem)
    stats["decode_peak_allocated_mib"] = bytes_to_mib(decode_peak_allocated)
    stats["decode_peak_reserved_mib"] = bytes_to_mib(decode_peak_reserved)

    # ---------- Stats ----------
    total_time_ms = prefill_time + decode_time
    stats.update({
        "prompt_tokens": prompt_len,
        "prefill_time_ms": prefill_time,
        "prefill_speed_toks": (prompt_len / (prefill_time / 1e3)),
        "decode_steps": decode_steps,
        "decode_time_ms": decode_time,
        "decode_speed_toks": (decode_steps / (decode_time / 1e3)) if decode_steps else 0.0,
        "total_time_ms": total_time_ms,
        "total_speed_toks": ((prompt_len + decode_steps) / (total_time_ms / 1e3)),
    })

    overall_peak_mib = stats["prefill_peak_allocated_mib"]
    if decode_peak_allocated > prefill_peak_allocated:
        overall_peak_mib = stats["decode_peak_allocated_mib"]
    stats["overall_peak_allocated_mib"] = overall_peak_mib
    stats["overall_peak_reserved_mib"] = stats["prefill_peak_reserved_mib"]
    if decode_peak_reserved > prefill_peak_reserved:
        stats["overall_peak_reserved_mib"] = stats["decode_peak_reserved_mib"]

    total_generated = prompt_len + decode_steps
    return total_generated, stats


# ============================================================================
# Torch RMS Norm (sequential sum to match Triton/CUDA exactly)
# ============================================================================

def torch_rms_norm_sequential(x, weight, eps):
    """RMS norm using sequential sum - matches Triton/CUDA exactly for n_cols <= 128."""
    original_shape = x.shape
    n_cols = original_shape[-1]
    x_flat = x.view(-1, n_cols)
    x_f32 = x_flat.float()

    # Sequential sum of squares - matches Triton/CUDA accumulation order
    sum_sq = torch.zeros(x_flat.shape[0], 1, device=x.device, dtype=torch.float32)
    for i in range(n_cols):
        sum_sq += x_f32[:, i:i+1] ** 2

    variance = sum_sq / n_cols
    rstd = torch.rsqrt(variance + eps)
    x_normed = x_f32 * rstd
    result = (weight.float() * x_normed).to(x.dtype)
    return result.view(original_shape)


# ============================================================================
# Triton Kernels (copied from qwen3-0.6b.py)
# ============================================================================

@triton.jit
def rms_norm_sequential(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with sequential sum for small sizes (n_cols <= 128) - matches PyTorch."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    # Sequential sum of squares - matches PyTorch for small sizes
    sum_sq = 0.0
    for i in range(n_cols):
        x_i = tl.load(x_ptr + row_start + i).to(tl.float32)
        sum_sq += x_i * x_i
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    # Vectorized output
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * (x * rstd)
    tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def rms_norm_tree(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with tree reduction for medium sizes (128 < n_cols <= 4096) - matches PyTorch."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = w * (x * rstd)
    tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def rms_norm_multi_pass(
    x_ptr, weight_ptr, out_ptr, stride_x_row, n_cols, eps, BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm with multi-pass for very large sizes (n_cols > 4096)."""
    row_idx = tl.program_id(0)
    row_start = row_idx * stride_x_row
    # Accumulate sum of squares across blocks
    sum_sq = 0.0
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    variance = sum_sq / n_cols
    rstd = tl.math.rsqrt(variance + eps)
    # Output pass
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = w * (x * rstd)
        tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm wrapper with hybrid sum method to match PyTorch exactly."""
    assert x.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if n_cols <= 128:
        # Sequential sum for head_dim=128 (QK norm)
        rms_norm_sequential[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    elif BLOCK_SIZE <= 4096:
        # Tree reduction for hidden_size=1024 (layer norm)
        rms_norm_tree[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # Multi-pass for very large hidden sizes
        rms_norm_multi_pass[(n_rows,)](x, weight, out, x.stride(0), n_cols, eps, BLOCK_SIZE=4096)
    return out.view(shape)


@triton.jit
def silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    gate = tl.load(gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up
    tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(gate)
    n_elements = gate.numel()
    BLOCK_SIZE = 1024
    silu_mul_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        gate.view(-1), up.view(-1), out.view(-1), n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 1000000.0, device="cuda"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope_torch(q, k, cos, sin, position_ids):
    """PyTorch RoPE implementation."""
    pos = position_ids[0]
    cos_pos = cos[pos].unsqueeze(0).unsqueeze(0)
    sin_pos = sin[pos].unsqueeze(0).unsqueeze(0)
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos_fp32 = cos_pos.float()
    sin_fp32 = sin_pos.float()
    half = q.shape[-1] // 2
    q1, q2 = q_fp32[..., :half], q_fp32[..., half:]
    k1, k2 = k_fp32[..., :half], k_fp32[..., half:]
    cos1, sin1 = cos_fp32[..., :half], sin_fp32[..., :half]
    q_rot = torch.cat([q1 * cos1 - q2 * sin1, q2 * cos1 + q1 * sin1], dim=-1)
    k_rot = torch.cat([k1 * cos1 - k2 * sin1, k2 * cos1 + k1 * sin1], dim=-1)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def attention_decode_torch(q, k_cache, v_cache, cache_len):
    """PyTorch attention decode."""
    batch, n_heads_q, _, head_dim = q.shape
    n_heads_kv = k_cache.shape[1]
    n_groups = n_heads_q // n_heads_kv
    k = k_cache[:, :, :cache_len, :]
    v = v_cache[:, :, :cache_len, :]
    k = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(batch, n_heads_q, cache_len, head_dim)
    v = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1).reshape(batch, n_heads_q, cache_len, head_dim)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())
    return out.to(q.dtype)


# ============================================================================
# Backend enum
# ============================================================================

class Backend:
    TORCH = "torch"
    TRITON = "triton"
    CUDA = "cuda"

# ============================================================================
# Model Components (Backend-switchable)
# ============================================================================

class Qwen3Attention:
    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int, backend: str, cuda_kernels=None):
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.q_proj_weight = layer_weights["q_proj.weight"]
        self.k_proj_weight = layer_weights["k_proj.weight"]
        self.v_proj_weight = layer_weights["v_proj.weight"]
        self.o_proj_weight = layer_weights["o_proj.weight"]
        self.q_norm_weight = layer_weights["q_norm.weight"]
        self.k_norm_weight = layer_weights["k_norm.weight"]

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def apply_rope(self, q, k, cos, sin, position_ids):
        if self.backend == Backend.TORCH:
            return apply_rope_torch(q, k, cos, sin, position_ids)
        elif self.backend == Backend.TRITON:
            return apply_rope_torch(q, k, cos, sin, position_ids)  # Use torch for triton too
        else:  # CUDA
            pos = position_ids[0]
            cos_pos = cos[pos].contiguous()
            sin_pos = sin[pos].contiguous()
            return self.cuda_kernels.rope(q.contiguous(), k.contiguous(), cos_pos, sin_pos)

    def attention_decode(self, q, k_cache, v_cache, cache_len):
        if Attn_method.use_flash_attn:
            q_fa = q.permute(0, 2, 1, 3).contiguous()
            k_fa = k_cache.permute(0, 2, 1, 3).contiguous()
            v_fa = v_cache.permute(0, 2, 1, 3).contiguous()

            from flash_attn import flash_attn_with_kvcache
            o_fa = flash_attn_with_kvcache(
                q_fa,
                k_fa,
                v_fa,
                cache_seqlens=cache_len, block_table=None, 
                softmax_scale=1.0 / (self.config.head_dim ** 0.5), causal=False)
            o = o_fa.permute(0, 2, 1, 3).contiguous()
            return o
        else:
            if self.backend == Backend.CUDA:
                o = self.cuda_kernels.attention_decode_v4(q.contiguous(), k_cache.contiguous(), v_cache.contiguous(), 
                                                          cache_len) # decode kernel调度
                return o
            else:
                o = attention_decode_torch(q, k_cache, v_cache, cache_len)
                return o

    def forward(self, hidden_states, cos, sin, position_ids, k_cache=None, v_cache=None, cache_position=0, is_prefill=True):
        batch, seq_len, _ = hidden_states.shape
        q = torch.nn.functional.linear(hidden_states, self.q_proj_weight)
        k = torch.nn.functional.linear(hidden_states, self.k_proj_weight)
        v = torch.nn.functional.linear(hidden_states, self.v_proj_weight)
        q = q.view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2).contiguous()

        # QK norm
        q = self.rms_norm(q.contiguous(), self.q_norm_weight)
        k = self.rms_norm(k.contiguous(), self.k_norm_weight)

        # RoPE
        q, k = self.apply_rope(q, k, cos, sin, position_ids)
        if is_prefill:
            if Attn_method.use_flash_attn:
                from flash_attn import flash_attn_func, flash_attn_varlen_func
                batch, n_head_q, seq_len_q, _ = q.shape
                _, n_head_k, seq_len_k, _ = k.shape
                _, n_head_v, seq_len_v, _ = v.shape
                if 1:# NOTE: 用于 batch 变长序列处理 
                    q_ = q.permute(0, 2, 1, 3)[0]
                    k_ = k.permute(0, 2, 1, 3)[0]
                    v_ = v.permute(0, 2, 1, 3)[0]

                    attn_out = flash_attn_varlen_func(
                        q_,
                        k_,
                        v_,
                        max_seqlen_q=seq_len_q,
                        cu_seqlens_q=torch.tensor([0, seq_len_q], dtype=torch.int32, device="cuda"),# 20
                        max_seqlen_k=seq_len_k,
                        cu_seqlens_k=torch.tensor([0, seq_len_k], dtype=torch.int32, device="cuda"),# 20
                        causal=True,
                        block_table=None,
                    )
                else:# NOTE: 用于 batch 定长序列处理
                    q_ = q.permute(0, 2, 1, 3)
                    k_ = k.permute(0, 2, 1, 3)
                    v_ = v.permute(0, 2, 1, 3)
                    attn_out = flash_attn_func(
                        q_, k_, v_,
                        causal=True
                    )
                attn_out = attn_out.view(batch, seq_len_q, n_head_q, self.config.head_dim)   # (batch, seq_len_q, n_head_q, head_dim)
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
                
            else:
                n_groups = self.config.num_attention_heads // self.config.num_key_value_heads
                k_expanded = k.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
                k_expanded = k_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)
                v_expanded = v.unsqueeze(2).expand(-1, -1, n_groups, -1, -1)
                v_expanded = v_expanded.reshape(batch, self.config.num_attention_heads, seq_len, self.config.head_dim)
                if self.backend != Backend.CUDA:
                    scale = 1.0 / (self.config.head_dim ** 0.5)
                    scores = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * scale # [1，16，21，128] * 
                    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()# [21，21]
                    scores = scores.masked_fill(mask, float('-inf')) # [1，16，21，21]
                    attn = torch.softmax(scores, dim=-1)
                    attn_out = torch.matmul(attn, v_expanded.float()).to(q.dtype)
                else:
                    attn_out = self.cuda_kernels.attention_prefill(q.contiguous(), k_expanded.contiguous(), v_expanded.contiguous(), seq_len)  # prefill kernel 调度
            if k_cache is not None:
                k_cache[:, :, :seq_len, :] = k
                v_cache[:, :, :seq_len, :] = v
        else:
            k_cache[:, :, cache_position:cache_position+1, :] = k
            v_cache[:, :, cache_position:cache_position+1, :] = v
            attn_out = self.attention_decode(q, k_cache, v_cache, cache_position + 1) # cuda or torch

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)# [1，21，2048]
        output = torch.nn.functional.linear(attn_out, self.o_proj_weight)
        return output, k_cache, v_cache


class Qwen3MLP:
    def __init__(self, layer_weights, config: Qwen3Config, backend: str, cuda_kernels=None):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.gate_proj_weight = layer_weights["gate_proj.weight"]
        self.up_proj_weight = layer_weights["up_proj.weight"]
        self.down_proj_weight = layer_weights["down_proj.weight"]

    def silu_mul(self, gate, up):
        if self.backend == Backend.TORCH:
            return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
        elif self.backend == Backend.TRITON:
            return triton_silu_mul(gate, up)
        else:  # CUDA
            return self.cuda_kernels.silu_mul(gate, up)

    def forward(self, hidden_states):
        gate = torch.nn.functional.linear(hidden_states, self.gate_proj_weight)
        up = torch.nn.functional.linear(hidden_states, self.up_proj_weight)
        hidden = self.silu_mul(gate, up)
        return torch.nn.functional.linear(hidden, self.down_proj_weight)


class Qwen3Layer:
    def __init__(self, layer_weights, config: Qwen3Config, layer_idx: int, backend: str, cuda_kernels=None):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.input_layernorm_weight = layer_weights["input_layernorm.weight"]
        self.post_attention_layernorm_weight = layer_weights["post_attention_layernorm.weight"]
        attn_weights = {k.replace("self_attn.", ""): v for k, v in layer_weights.items() if "self_attn" in k}
        mlp_weights = {k.replace("mlp.", ""): v for k, v in layer_weights.items() if "mlp" in k}
        self.self_attn = Qwen3Attention(attn_weights, config, layer_idx, backend, cuda_kernels)
        self.mlp = Qwen3MLP(mlp_weights, config, backend, cuda_kernels)

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, position_ids, k_cache=None, v_cache=None, cache_position=0, is_prefill=True):
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states, self.input_layernorm_weight)
        hidden_states, k_cache, v_cache = self.self_attn.forward(
            hidden_states, cos, sin, position_ids, k_cache, v_cache, cache_position, is_prefill
        )# Prefill or Decode 调度
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states, self.post_attention_layernorm_weight)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, k_cache, v_cache


class Qwen3Model:
    def __init__(
        self,
        hf_model,
        config: Qwen3Config,
        backend: str,
        cuda_kernels=None,
        max_new_tokens: int = 256,
    ):
        self.config = config
        self.backend = backend
        self.cuda_kernels = cuda_kernels
        self.max_new_tokens = max_new_tokens
        self.device = next(hf_model.parameters()).device
        self.dtype = torch.bfloat16
        state_dict = hf_model.state_dict()
        self.embed_tokens = state_dict["model.embed_tokens.weight"]
        self.final_norm_weight = state_dict["model.norm.weight"]
        self.lm_head_weight = self.embed_tokens
        self.layers = []
        for i in range(config.num_hidden_layers):
            layer_weights = {k.replace(f"model.layers.{i}.", ""): v for k, v in state_dict.items() if f"model.layers.{i}." in k}
            self.layers.append(Qwen3Layer(layer_weights, config, i, backend, cuda_kernels))
        self.cos, self.sin = precompute_rope_freqs(config.head_dim, config.max_position_embeddings, config.rope_theta, self.device)

    def _required_cache_len(self, prompt_len: int, reserve_decode_steps: int) -> int:
        total_len = prompt_len + max(reserve_decode_steps, 0)
        if prompt_len > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt length {prompt_len} exceeds max_position_embeddings {self.config.max_position_embeddings}"
            )
        if total_len > self.config.max_position_embeddings:
            raise ValueError(
                f"prompt+decode length {total_len} exceeds max_position_embeddings {self.config.max_position_embeddings}"
            )
        return total_len

    def rms_norm(self, x, weight):
        if self.backend == Backend.TORCH:
            return torch_rms_norm_sequential(x, weight, self.config.rms_norm_eps)
        elif self.backend == Backend.TRITON:
            return triton_rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)
        else:  # CUDA
            return self.cuda_kernels.rms_norm(x.contiguous(), weight, self.config.rms_norm_eps)

    def prefill(self, input_ids, reserve_decode_steps: Optional[int] = None):
        batch, seq_len = input_ids.shape
        hidden_states = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
        kv_caches = []
        if reserve_decode_steps is None:
            reserve_decode_steps = self.max_new_tokens
        max_cache_len = self._required_cache_len(seq_len, reserve_decode_steps)
        for layer in self.layers:
            k_cache = torch.zeros(batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim, device=self.device, dtype=self.dtype)
            v_cache = torch.zeros(batch, self.config.num_key_value_heads, max_cache_len, self.config.head_dim, device=self.device, dtype=self.dtype)
            hidden_states, k_cache, v_cache = layer.forward(hidden_states, self.cos, self.sin, position_ids, k_cache, v_cache, 0, is_prefill=True)
            kv_caches.append((k_cache, v_cache))
        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)
        return logits, kv_caches, seq_len

    def decode_step(self, input_id, kv_caches, cache_position):
        hidden_states = torch.nn.functional.embedding(input_id, self.embed_tokens)
        position_ids = torch.tensor([[cache_position]], device=self.device)
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            if cache_position >= k_cache.size(2):
                raise ValueError(
                    f"decode cache_position {cache_position} exceeds KV cache capacity {k_cache.size(2)} "
                    f"for layer {i}; increase reserved decode steps"
                )
            hidden_states, k_cache, v_cache = layer.forward(hidden_states, self.cos, self.sin, position_ids, k_cache, v_cache, cache_position, is_prefill=False)# decode 调度
            new_kv_caches.append((k_cache, v_cache))
        hidden_states = self.rms_norm(hidden_states, self.final_norm_weight)
        logits = torch.nn.functional.linear(hidden_states, self.lm_head_weight)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate_streaming(self, input_ids, tokenizer, max_new_tokens=100):
        """Generate tokens with streaming output."""
        # Prefill
        logits, kv_caches, cache_len = self.prefill(input_ids, reserve_decode_steps=max_new_tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # Decode token and print
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)

            # Check for EOS
            if next_token.item() == 151645:
                break

            # Generate next token
            logits, kv_caches = self.decode_step(next_token, kv_caches, cache_len + tokens_generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens_generated += 1

        print()  # Newline after generation
        return tokens_generated

def fast_answer_test(tokenizer, model):
    # out test
    prompt="list all prime numbers within 100"
    # prompt = input("Enter your question (or 'quit' to exit): ").strip()
    # if prompt.lower() in ['quit', 'exit', 'q']:
    #     break
    # if not prompt:
    #     continue

    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    print(f"\nInput tokens: {input_ids.shape[1]}")

    # Generate with each backend
    max_new_tokens = 256
    # CUDA
    print("\n" + "=" * 70)
    print(" CUDA")
    print("=" * 70)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cuda_tokens = model.generate_streaming(input_ids.clone(), tokenizer, max_new_tokens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    cuda_time = t1 - t0
    print(f"[Generated {cuda_tokens} tokens in {cuda_time:.3f}s ({cuda_tokens/cuda_time:.1f} tok/s)]")

    # Summary
    print("\n" + "-" * 70)
    print("Performance Summary:")
    print(f"  XXXX:   {cuda_time:.3f}s ({cuda_tokens/cuda_time:.1f} tok/s)")
    print()

# ============================================================================
# Main Demo
# ============================================================================
def run_demo():
    print("=" * 70)
    print(" End-to-End Demo: Torch vs Triton vs CUDA (with detailed timing)")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-0.6B model...")
    model_name = "/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    hf_model.eval()

    # Compile CUDA kernels
    print("Compiling CUDA kernels...")
    cuda_kernels = get_kernels()
    print("Done!\n")

    # -----------------------------------------------------------------------
    # Test scenarios (Prompt Tokens, Decode Steps, Total Context, Dominant Phase)
    # -----------------------------------------------------------------------
    config = Qwen3Config()
    scenarios = [
        # name,   prompt_len, decode_steps, total_context, dominant
        ("Short",   64,   16,   80,   "decode"),
        ("Short",   128,   32,   160,   "decode"),
        ("Short",   256,   64,   320,   "decode"),
        ("Medium", 512,  128,  640,   "balanced"),
        ("Long",  1024,  256, 1280,   "prefill"),
        ("Very Long1", 4096, 1024, 5120, "prefill+cache"),
        ("Very Long2", 8192, 2048, 10240, "prefill-8k"),
        ("Very Long3", 16384, 2048, 20480, "near OOM1"),
        ("Very Long4", 32768, 2048, 40960, "near OOM2"),# TODO：11.8能测，而在12.8不能测了
    ]
    model_max_new_tokens = max(256, max(decode_steps for _, _, decode_steps, _, _ in scenarios))

    # Create config & backend models (only CUDA is exercised in the demo)
    torch_model = Qwen3Model(hf_model, config, Backend.TORCH, max_new_tokens=model_max_new_tokens)
    triton_model = Qwen3Model(hf_model, config, Backend.TRITON, max_new_tokens=model_max_new_tokens)
    cuda_model = Qwen3Model(hf_model, config, Backend.CUDA, cuda_kernels, max_new_tokens=model_max_new_tokens)

    mod = cuda_model
    # for mod in (cuda_model):# triton_model, torch_model
    if mod:
        fast_answer_test(tokenizer, mod)
        # header
        print(
            "prompts,"
            "steps,"
            "total_ms,"
            "prefill_ms,"
            "decode_ms,"
            "total_tok_s,"
            "prefill_tok_s,"
            "decode_tok_s,"
            "peakAloc_MB,"
            "peakRes_MB"
        )
        for name, prompt_len, decode_steps, total_ctx, dominant in scenarios:
            # print("-" * 70)
            # print(f"Scenario: {name} (prompt={prompt_len}, decode={decode_steps}) – dominant: {dominant}")
            # Build a dummy prompt of the requested length
            input_ids = make_input_ids(tokenizer, prompt_len, torch.device("cuda"))

            # Run benchmark (CUDA backend)
            torch.cuda.synchronize()
            _, stats = benchmark_model(mod, input_ids, decode_steps)
            # data
            print(
                f"{stats['prompt_tokens']},"
                f"{stats['decode_steps']},"
                f"{stats['total_time_ms']:.1f},"
                f"{stats['prefill_time_ms']:.1f},"
                f"{stats['decode_time_ms']:.1f},"
                f"{stats['total_speed_toks']:.0f},"
                f"{stats['prefill_speed_toks']:.0f},"
                f"{stats['decode_speed_toks']:.0f},"
                f"{stats['overall_peak_allocated_mib']:.1f},"
                f"{stats['overall_peak_reserved_mib']:.1f}"
            )

    print("=" * 70)
    print("All scenarios completed.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
