from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import AutoFeatureExtractor, AutoTokenizer

from .qwen3_asr import (
    Qwen3ASRAudioTower,
    Qwen3ASRSpecialIds,
    _pick_device_dtype,
    load_wav_mono,
)
from .profiler import StageTimer
from .safetensors_loader import load_tensors
from .whisper_fbank_torch import WhisperFbankConfig, WhisperFbankTorch
from .asr_result import ASRResult


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_tensor_keys(model_dir: Path) -> set[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return {str(key) for key in weight_map.keys()}
    ckpt = model_dir / "model.safetensors"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    with safe_open(str(ckpt), framework="pt", device="cpu") as f:
        return set(f.keys())


def _collect_bnb4_meta_keys(weight_key: str, available_keys: set[str]) -> list[str]:
    keys: list[str] = []
    for suffix in [".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map"]:
        full = weight_key + suffix
        if full in available_keys:
            keys.append(full)
    nf4 = weight_key + ".quant_state.bitsandbytes__nf4"
    fp4 = weight_key + ".quant_state.bitsandbytes__fp4"
    if nf4 in available_keys:
        keys.append(nf4)
    elif fp4 in available_keys:
        keys.append(fp4)
    return keys


def _dequantize_bnb4_weight(
    weight_key: str,
    packed_weight: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    try:
        from bitsandbytes import functional as bnbf
    except Exception as exc:
        raise RuntimeError(
            f"Detected bitsandbytes 4bit tensor for {weight_key}, but bitsandbytes is unavailable."
        ) from exc

    if packed_weight.dtype != torch.uint8:
        return packed_weight

    qkey_nf4 = weight_key + ".quant_state.bitsandbytes__nf4"
    qkey_fp4 = weight_key + ".quant_state.bitsandbytes__fp4"
    qkey = qkey_nf4 if qkey_nf4 in tensors else qkey_fp4
    if qkey not in tensors:
        raise RuntimeError(f"Missing quant_state tensor for {weight_key}")

    qdict: dict[str, torch.Tensor] = {qkey: tensors[qkey]}
    for suffix in [".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map"]:
        full = weight_key + suffix
        if full in tensors:
            qdict[full] = tensors[full]

    try:
        quant_state = bnbf.QuantState.from_dict(qdict, device=packed_weight.device)
        dequant = bnbf.dequantize_4bit(packed_weight, quant_state=quant_state)
    except Exception:
        if packed_weight.device.type == "cuda" or not torch.cuda.is_available():
            raise
        moved = {k: v.to("cuda") for k, v in qdict.items()}
        quant_state = bnbf.QuantState.from_dict(moved, device=torch.device("cuda"))
        dequant = bnbf.dequantize_4bit(packed_weight.to("cuda"), quant_state=quant_state).to("cpu")

    return dequant.contiguous()


_ASR_MEGAKERNEL_DEFAULT_KNOBS: dict[str, str] = {
    # Prefill attention: use flash_ext path by default (with experimental gate),
    # and keep tail legacy layers for quality stability.
    "MEGAQWEN_PREFILL_ATTN_IMPL": "flash",     # -1 unknown, 0 legacy, 1 splitk, 2 auto, 3 flash, 4 flash_ext
    "MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL": "1", # 
    "MEGAQWEN_PREFILL_FLASH_EXT_TAIL_LEGACY_LAYERS": "2",# 2 for flash_ext layer, 0 for all flash
    # Decode: use split_gemm + GPU-side greedy loop.
    "MEGAQWEN_DECODE_BACKEND": "split_gemm",
    "MEGAQWEN_DECODE_GPU_LOOP": "1",
    # Aggressive flash-decode defaults (can still be overridden by env).
    "MEGAQWEN_SPLIT_ATTN_IMPL": "flash_decode", #  0=legacy, 1=splitk(v1, one block per head), 2=splitk2(seq-split two-phase), 3=flash_decode(tile-online)
    "MEGAQWEN_SPLIT_FLASH_WARPS": "4",
    "MEGAQWEN_SPLIT_FLASH_PARTS": "1",          # 当 part >1, 会插入 decode_attention_cache_flash_reduce_parts_kernel
    "MEGAQWEN_SPLIT_ATTN_CHUNK": "64",          # FD 算法中二阶段归约的 CHUNK 数目，16 / 设定值 / max_seq_len_
    "MEGAQWEN_SPLIT_QKV_GEMM_IMPL": "gemmex",   # 为1时，采用cublasLt(+fallback), 为0或者 gemmex 或者 cublas 或者false，则采用 cublasGemmEx 函数
    # Config for PagedAttn
    "MEGAQWEN_SPLIT_KV_LAYOUT": "paged",
    "MEGAQWEN_SPLIT_KV_BLOCK_SIZE": "16",
    "MEGAQWEN_SPLIT_KV_FP8": "1",               # FP8 KV 初始化
    "MEGAQWEN_SPLIT_KV_FP8_ONLY": "1",          # 在启用FP8时，另外配置 decode_qk_norm_rope_cache_kernel
    # MEGAQWEN_SPLIT_ATTN_IMPL ==3 的变体控制
    "MEGAQWEN_SPLIT_FLASH_GQA_SHARE": "1",      # 当 MEGAQWEN_SPLIT_ATTN_IMPL ==3，控制GQA变体的有效性
    "MEGAQWEN_SPLIT_FLASH_GQA_MODE": "tc",      # 总控开关。当 MEGAQWEN_SPLIT_FLASH_GQA_SHARE == 1时，0 for tensorCore, 1 for simt_fast TODO: simt_fast 用了什么指令
    "MEGAQWEN_SPLIT_FLASH_FP8_TC_QK": "1",      # 细粒度使能开关。
    # W4A16量化与反量化, 用于权重加载 load_qwen3_weights
    "MEGAQWEN_SPLIT_QKV_W4": "1",
    "MEGAQWEN_SPLIT_O_W4": "1",
    "MEGAQWEN_SPLIT_FFN_W4": "1",
    "MEGAQWEN_SPLIT_FFN_W4_FUSED": "1",
    # Audio tower inference defaults.
    "MEGAQWEN_AUDIO_TOWER_ATTN_IMPL": "flash_attention_2",
    "MEGAQWEN_AUDIO_TOWER_ALLOW_TF32": "1",
    "MEGAQWEN_AUDIO_TOWER_CUDNN_BENCHMARK": "1",
    "MEGAQWEN_AUDIO_TOWER_ENABLE_FLASH_SDP": "1",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN": "0",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN_FORCE": "0",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN_DEBUG": "0",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN_TIMING": "0",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN_TIMING_EVERY": "128",
    "MEGAQWEN_AUDIO_TOWER_CUDA_ATTN_TIMING_LAYER": "0",
    "MEGAQWEN_AUDIO_TOWER_COMPILE": "1",
    "MEGAQWEN_AUDIO_TOWER_COMPILE_WITH_FLASH_ATTN": "0",
    "MEGAQWEN_AUDIO_TOWER_COMPILE_WITH_CUDA_ATTN": "0",
    "MEGAQWEN_AUDIO_TOWER_COMPILE_MODE": "reduce-overhead",
    "MEGAQWEN_AUDIO_TOWER_DEBUG": "0",
    # Long-audio chunking for stable prefill length (seconds per chunk).
    "MEGAQWEN_ASR_CHUNK_SEC": "60",
    # Debug logs are OFF by default; enable explicitly when profiling internals.
    "MEGAQWEN_DEBUG_PREFILL_STAGE": "0",
    "MEGAQWEN_DEBUG_SPLIT_STAGE": "0",
    "MEGAQWEN_DEBUG_SPLIT_STAGE_AVG": "0",      # 
    "MEGAQWEN_DEBUG_FLASH_DECODE": "0",         # 调试总控开关
}

def _apply_asr_megakernel_default_knobs(ATTN_METHOD=33) -> None:
    # ENVIRONMENT MANAGE
    #  0=legacy, 1=splitk(v1, one block per head), 2=splitk2(seq-split two-phase), 3=flash_decode(tile-online)
    if(ATTN_METHOD==0):
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="legacy"
    elif(ATTN_METHOD==10):
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="splitk"
    elif(ATTN_METHOD==20):
        print("ERROR cause by decode_attention_cache_splitk_phase1_kernel !")
        # _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="splitk2"
        pass
    elif(ATTN_METHOD==30):# V3 Warning：need manually change "置零"
        # _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="flash_decode"
        pass
    elif(ATTN_METHOD==31):# v3S0
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="flash_decode"
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_FLASH_GQA_SHARE"]="0"
    elif(ATTN_METHOD==32):# v3GQA
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="flash_decode"
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_FLASH_GQA_MODE"]="simt_fast"
    elif(ATTN_METHOD==33):
        _ASR_MEGAKERNEL_DEFAULT_KNOBS["MEGAQWEN_SPLIT_ATTN_IMPL"]="flash_decode"

    for key, value in _ASR_MEGAKERNEL_DEFAULT_KNOBS.items():
        os.environ.setdefault(key, value)


def _split_audio_into_chunks(
    wav: np.ndarray,
    *,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> list[np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    if wav.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape={wav.shape}")

    total_len = int(wav.shape[0])
    if total_len <= 0:
        return [wav]
    if max_chunk_sec <= 0:
        return [wav]

    max_len = max(1, int(float(max_chunk_sec) * float(sr)))
    if total_len <= max_len:
        return [wav]

    expand = max(1, int(float(search_expand_sec) * float(sr)))
    win = max(4, int((float(min_window_ms) / 1000.0) * float(sr)))

    out: list[np.ndarray] = []
    start = 0
    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)
        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)
            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
            min_pos = int(np.argmin(window_sums))
            wstart = min_pos
            wend = min_pos + win
            local = seg_abs[wstart:wend]
            inner = int(np.argmin(local))
            boundary = left + wstart + inner
        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))
        out.append(wav[start:boundary].astype(np.float32, copy=False))
        start = boundary
    out.append(wav[start:total_len].astype(np.float32, copy=False))
    return out


def _merge_chunk_languages(langs: list[str | None], user_language: str | None) -> str | None:
    if user_language:
        return user_language
    freq: dict[str, int] = {}
    order: list[str] = []
    for item in langs:
        if item is None:
            continue
        s = str(item).strip()
        if not s:
            continue
        if s not in freq:
            freq[s] = 0
            order.append(s)
        freq[s] += 1
    if not freq:
        return None
    return max(order, key=lambda k: (freq[k], -order.index(k)))


def _maybe_print_ldg_stage_debug(cycles_cuda: torch.Tensor, *, kernel_ms: float | None = None) -> None:
    """Pretty-print per-stage cycle stats from MegakernelPrefillDecoder.debug_ldg_decode_cycles()."""
    if not (torch.cuda.is_available() and cycles_cuda.is_cuda):
        return
    # Print once per process to keep benchmark output readable.
    if getattr(_maybe_print_ldg_stage_debug, "_printed", False):
        return
    setattr(_maybe_print_ldg_stage_debug, "_printed", True)

    stage_names = [
        "matvec_rmsnorm",
        "matvec_sync1",
        "matvec_qkv",
        "matvec_sync2",
        "qk_norm_rope_cache",
        "qk_sync",
        "attn",
        "attn_sync",
        "oproj_resid",
        "oproj_sync",
        "postnorm",
        "postnorm_sync",
        "gate_up",
        "gate_up_sync",
        "downproj",
        "downproj_sync",
    ]
    sync_stage = {1, 3, 5, 7, 9, 11, 13, 15}

    cycles = cycles_cuda.to(device="cpu", non_blocking=False)
    if cycles.dtype != torch.int64 or cycles.ndim != 3 or cycles.shape[-1] != len(stage_names):
        print(f"[MEGAQWEN_DEBUG] unexpected stage debug tensor: dtype={cycles.dtype} shape={tuple(cycles.shape)}")
        return
    # Detect the actually-used cooperative grid size: unused blocks remain all-zero.
    blk_sum = cycles.abs().sum(dim=(1, 2))  # [blocks]
    used = int((blk_sum > 0).sum().item())
    if used <= 0:
        print("[MEGAQWEN_DEBUG] stage debug buffer is all-zero (did the kernel write the buffer?)")
        return
    # Keep a contiguous prefix; any holes would indicate a write bug.
    last = int((blk_sum > 0).nonzero()[-1].item())
    b = last + 1
    cycles = cycles[:b]
    _, l, _ = cycles.shape
    # Wall-time proxy per stage in *cycles*: max over blocks per layer.
    stage_wall_cycles_per_layer = cycles.max(dim=0).values.to(dtype=torch.float64)  # [layers, stages]
    stage_wall_cycles = stage_wall_cycles_per_layer.sum(dim=0)  # [stages]
    total_cycles = float(stage_wall_cycles.sum().item())
    if total_cycles <= 0:
        print("[MEGAQWEN_DEBUG] stage debug total is zero (did the kernel write the buffer?)")
        return

    # Optionally convert cycles to milliseconds by scaling to a measured kernel time.
    if kernel_ms is not None and kernel_ms > 0:
        stage_ms = stage_wall_cycles * (float(kernel_ms) / total_cycles)
        total_ms = float(stage_ms.sum().item())
    else:
        stage_ms = None
        total_ms = None

    # Per-layer wall proxy (cycles).
    layer_wall_cycles = stage_wall_cycles_per_layer.sum(dim=1)  # [layers]
    topk = min(5, int(layer_wall_cycles.numel()))
    vals, idxs = torch.topk(layer_wall_cycles, k=topk, largest=True)

    header = f"[MEGAQWEN_DEBUG] LDG stage timing (one decode token) blocks={b} (used~{used}) layers={l}"
    if kernel_ms is not None and kernel_ms > 0:
        header += f" kernel~{kernel_ms:.3f} ms"
    print(header)
    print("[MEGAQWEN_DEBUG] stage wall-time proxy = sum_l max_b(cycles[b,l,stage])")
    if total_ms is not None:
        print(f"[MEGAQWEN_DEBUG] total (scaled): {total_ms:.3f} ms")
    else:
        print(f"[MEGAQWEN_DEBUG] total cycles: {total_cycles:.3e}")
    print("[MEGAQWEN_DEBUG] stages:")
    for i, name in enumerate(stage_names):
        v_cycles = float(stage_wall_cycles[i].item())
        pct = 100.0 * v_cycles / total_cycles
        tag = "sync" if i in sync_stage else "comp"
        if stage_ms is not None:
            v_ms = float(stage_ms[i].item())
            print(f"  {name:18s} {tag}  {v_ms:8.3f} ms  ({pct:5.1f}%)")
        else:
            print(f"  {name:18s} {tag}  {v_cycles:11.3e} cyc  ({pct:5.1f}%)")

    sync_cycles = float(stage_wall_cycles[list(sync_stage)].sum().item())
    print(f"[MEGAQWEN_DEBUG] sync share: {100.0 * sync_cycles / total_cycles:.1f}%")
    print("[MEGAQWEN_DEBUG] slowest layers (wall proxy):")
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if stage_ms is not None:
            v_ms = float(v) * (float(kernel_ms) / total_cycles)
            print(f"  layer {int(i):2d}: {v_ms:8.3f} ms")
        else:
            print(f"  layer {int(i):2d}: {float(v):11.3e} cyc")


@dataclass(frozen=True)
class Qwen3ASRMegakernelOptions:
    max_seq_len: int = 4096
    max_prefill_len: int = 4096
    max_new_tokens: int = 256


class Qwen3ASRMegakernelModel:
    """Qwen3-ASR end-to-end (audio tower + megakernel prefill/decode).

    Constraints:
    - Megakernel path is compile-time specialized per text config (hidden/intermediate/heads/layers).
      Supported specs depend on successful JIT compile on the current GPU/toolchain.
    - Audio tower runs in PyTorch (Conv/GEMM/FlashAttention via SDPA).
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        opts: Qwen3ASRMegakernelOptions,
    ):
        _apply_asr_megakernel_default_knobs()

        self.model_dir = Path(model_dir)
        self.device = device
        self.dtype = dtype
        self.opts = opts

        cfg = _read_json(self.model_dir / "config.json")
        if cfg.get("model_type") != "qwen3_asr":
            raise ValueError(f"Not a Qwen3-ASR checkpoint: {self.model_dir}")

        thinker = cfg["thinker_config"]
        self._audio_cfg = thinker["audio_config"]
        self._text_cfg = thinker["text_config"]

        # Tokenizer / feature extractor
        # Some recent tokenizers ship with a known-bad regex; `fix_mistral_regex`
        # (when available) patches it to avoid incorrect tokenization.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), fix_mistral_regex=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.model_dir))
        self._fbank_torch: WhisperFbankTorch | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self._fbank_torch = WhisperFbankTorch(
                cfg=WhisperFbankConfig(
                    n_fft=int(self.feature_extractor.n_fft),
                    hop_length=int(self.feature_extractor.hop_length),
                    n_samples=int(self.feature_extractor.n_samples),
                    nb_max_frames=int(self.feature_extractor.nb_max_frames),
                    dither=float(getattr(self.feature_extractor, "dither", 0.0) or 0.0),
                ),
                mel_filters=self.feature_extractor.mel_filters,
                device=self.device,
            )

        self.special = Qwen3ASRSpecialIds(
            im_start=self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            im_end=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            audio_start=self.tokenizer.convert_tokens_to_ids("<|audio_start|>"),
            audio_pad=self.tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
            audio_end=self.tokenizer.convert_tokens_to_ids("<|audio_end|>"),
            asr_text=(self.tokenizer.convert_tokens_to_ids("<asr_text>") if "<asr_text>" in self.tokenizer.get_vocab() else None),
        )

        # Audio tower
        out_hidden = int(self._text_cfg["hidden_size"])
        self.audio_tower = Qwen3ASRAudioTower(audio_config=self._audio_cfg, out_hidden=out_hidden)
        self._load_audio_weights()
        self.audio_tower.eval().to(device=self.device, dtype=self.dtype)
        self.audio_tower.prepare_for_inference()

        # Megakernel (prefill+decode)
        self.decoder = self._init_megakernel_decoder()

        # Stop tokens
        eos_ids = set()
        gen_cfg_path = self.model_dir / "generation_config.json"
        if gen_cfg_path.exists():
            gen_cfg = _read_json(gen_cfg_path)
            eos = gen_cfg.get("eos_token_id")
            if isinstance(eos, list):
                eos_ids.update(int(x) for x in eos)
            elif eos is not None:
                eos_ids.add(int(eos))
        if not eos_ids:
            eos_ids.add(int(self.tokenizer.eos_token_id))
        self.eos_ids = eos_ids

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | None = None,
        dtype: str | None = None,
        opts: Qwen3ASRMegakernelOptions | None = None,
    ) -> "Qwen3ASRMegakernelModel":
        dev, dt = _pick_device_dtype(device, dtype)
        return cls(model_dir, device=dev, dtype=dt, opts=opts or Qwen3ASRMegakernelOptions())

    def _load_audio_weights(self) -> None:
        audio_prefix = "thinker.audio_tower."
        expected_local_keys = list(self.audio_tower.state_dict().keys())
        audio_keys = [audio_prefix + k for k in expected_local_keys]
        tensors = load_tensors(self.model_dir, audio_keys, device="cpu")

        quantized_audio_keys = [k for k in audio_keys if k in tensors and tensors[k].dtype == torch.uint8]
        if quantized_audio_keys:
            available = _available_tensor_keys(self.model_dir)
            meta_keys: list[str] = []
            for key in quantized_audio_keys:
                meta_keys.extend(_collect_bnb4_meta_keys(key, available))
            meta_keys = list(dict.fromkeys(meta_keys))
            if meta_keys:
                tensors.update(load_tensors(self.model_dir, meta_keys, device="cpu"))
            for key in quantized_audio_keys:
                tensors[key] = _dequantize_bnb4_weight(key, tensors[key], tensors)

        audio_sd = {k: tensors[audio_prefix + k] for k in expected_local_keys}
        missing, unexpected = self.audio_tower.load_state_dict(audio_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Audio tower state mismatch: missing={missing}, unexpected={unexpected}")

    def _init_megakernel_decoder(self):
        # Import megakernel module (JIT compile on first use).
        import sys

        sys.path.insert(0, str((Path(__file__).parent.parent / "csrc" / "megakernel").resolve()))
        from megakernel_decode import _compile_prefill_kernel, configure_model_spec, get_active_model_spec, load_qwen3_weights  # type: ignore

        configure_model_spec(
            hidden_size=int(self._text_cfg["hidden_size"]),
            intermediate_size=int(self._text_cfg["intermediate_size"]),
            num_q_heads=int(self._text_cfg["num_attention_heads"]),
            num_kv_heads=int(self._text_cfg["num_key_value_heads"]),
            head_dim=int(self._text_cfg["head_dim"]),
            num_layers=int(self._text_cfg["num_hidden_layers"]),
            vocab_size=int(self._text_cfg.get("vocab_size", 151936)),
        )
        spec = get_active_model_spec()

        weights = load_qwen3_weights(str(self.model_dir), max_seq_len=self.opts.max_seq_len)
        kernel = _compile_prefill_kernel()
        return kernel.MegakernelPrefillDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            weights.get("split_q_w4_packed", []),
            weights.get("split_q_w4_scales", []),
            weights.get("split_q_w4_codebook", []),
            weights.get("split_k_w4_packed", []),
            weights.get("split_k_w4_scales", []),
            weights.get("split_k_w4_codebook", []),
            weights.get("split_v_w4_packed", []),
            weights.get("split_v_w4_scales", []),
            weights.get("split_v_w4_codebook", []),
            weights.get("split_o_w4_packed", []),
            weights.get("split_o_w4_scales", []),
            weights.get("split_o_w4_codebook", []),
            weights.get("split_gateup_w4_packed", []),
            weights.get("split_gateup_w4_scales", []),
            weights.get("split_gateup_w4_codebook", []),
            weights.get("split_down_w4_packed", []),
            weights.get("split_down_w4_scales", []),
            weights.get("split_down_w4_codebook", []),
            int(spec["num_layers"]),
            self.opts.max_seq_len,
            self.opts.max_prefill_len,
        )

    def _build_prompt_ids(
        self,
        audio_token_count: int,
        *,
        system_text: str = "",
        force_language: str | None = None,
    ) -> tuple[list[int], int]:
        # Match official template and language forcing behavior.
        prefix = "<|im_start|>system\n" + (system_text or "") + "<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        ids.append(self.special.audio_start)
        audio_start_idx = len(ids)  # first audio_pad will be written here
        ids.extend([self.special.audio_pad] * audio_token_count)
        ids.append(self.special.audio_end)
        ids.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
        if force_language:
            ids.extend(self.tokenizer.encode(f"language {force_language}", add_special_tokens=False))
            if self.special.asr_text is not None:
                ids.append(self.special.asr_text)
            else:
                ids.extend(self.tokenizer.encode("<asr_text>", add_special_tokens=False))
        return ids, audio_start_idx

    @torch.no_grad()
    def _decode_greedy_tokens(
        self,
        first_token_id: int,
        *,
        max_new_tokens: int,
        timer: StageTimer,
    ) -> tuple[list[int], str]:
        """Greedy decode on the megakernel decoder.

        Implementation goals:
        - Keep tokens on GPU during generation to avoid per-chunk D2H copies.
        - Stop on EOS, excluding the EOS token from the returned list (matches Torch baseline).
        """
        max_new = int(max_new_tokens)
        if max_new <= 0:
            return [], "max_new_tokens"

        chunk = int(os.environ.get("MEGAQWEN_DECODE_CHUNK", "32"))
        if chunk <= 0:
            chunk = 32

        eos_ids = self.eos_ids
        eos_sorted = sorted(int(x) for x in eos_ids)
        eos_cuda = torch.tensor(eos_sorted, device=self.device, dtype=torch.int32) if eos_sorted else None

        generated_cuda = torch.empty((max_new,), device=self.device, dtype=torch.int32)
        gen_len = 0
        next_tok = int(first_token_id)
        stop_reason = "max_new_tokens"

        with timer.cuda("text_decode"):
            use_graph = os.environ.get("MEGAQWEN_DECODE_GRAPH", "0") not in ("", "0", "false", "False")
            use_gpu_loop = os.environ.get("MEGAQWEN_DECODE_GPU_LOOP", "0") not in ("", "0", "false", "False")

            # Fast path: run the greedy decode loop inside the extension to reduce
            # Python-level synchronization (especially `.item()` in the chunk loop).
            if use_gpu_loop and eos_cuda is not None and hasattr(self.decoder, "decode_greedy_cuda"):
                if next_tok in eos_ids:
                    stop_reason = "eos"
                    gen_len = 0
                else:
                    out_cuda, out_len, stop_code = self.decoder.decode_greedy_cuda(
                        int(next_tok), int(max_new - gen_len), eos_cuda, int(chunk)
                    )
                    out_len = int(out_len)
                    if out_len > 0:
                        generated_cuda[gen_len : gen_len + out_len].copy_(out_cuda)
                    gen_len += out_len
                    stop_reason = "eos" if int(stop_code) == 1 else "max_new_tokens"
            else:
                while gen_len < max_new and next_tok not in eos_ids:
                    steps = min(chunk, max_new - gen_len)
                    if use_graph and hasattr(self.decoder, "decode_steps_cuda_graph"):
                        tail = self.decoder.decode_steps_cuda_graph(int(next_tok), int(steps))  # CUDA int32, [steps]
                    else:
                        tail = self.decoder.decode_steps_cuda(int(next_tok), int(steps))  # CUDA int32, [steps]

                    # consumed = [next_tok] + tail[:-1]  (len=steps)
                    generated_cuda[gen_len].fill_(int(next_tok))
                    if steps > 1:
                        generated_cuda[gen_len + 1 : gen_len + steps].copy_(tail[:-1])

                    # Chunk-level EOS detection (caps wasted compute to <chunk tokens).
                    if eos_cuda is not None and steps > 1:
                        mask = (tail[:-1].unsqueeze(1) == eos_cuda.unsqueeze(0)).any(dim=1)
                        if bool(mask.any().item()):
                            # EOS is at chunk position (m+1); exclude it from output.
                            m = int(torch.argmax(mask.to(torch.int32)).item())  # first True, 0..steps-2
                            gen_len += (m + 1)
                            stop_reason = "eos"
                            break

                    gen_len += steps
                    # The next token to consume is the last output of this chunk.
                    next_tok = int(tail[-1].item())

                if next_tok in eos_ids:
                    stop_reason = "eos"

        out = generated_cuda[:gen_len].to(device="cpu", non_blocking=False).tolist()
        return [int(x) for x in out], stop_reason

    def _asr_chunk_seconds(self) -> float:
        s = os.environ.get("MEGAQWEN_ASR_CHUNK_SEC", "")
        if s.strip() == "":
            return 60.0
        try:
            v = float(s)
        except Exception:
            return 60.0
        if v <= 0:
            return 0.0
        return v

    @torch.no_grad()
    def _extract_features(
        self,
        audio: np.ndarray,
        *,
        sr: int,
        timer: StageTimer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._fbank_torch is not None:
            with timer.cuda("feature_extract"):
                input_features, attention_mask = self._fbank_torch.extract(audio, sampling_rate=sr)
            with timer.cuda("h2d_features"):
                input_features = input_features.to(dtype=self.dtype)
            return input_features, attention_mask

        with timer.cpu("feature_extract"):
            feats = self.feature_extractor(
                audio,
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )

        with timer.cuda("h2d_features"):
            input_features = feats["input_features"].to(self.device, dtype=self.dtype)
            attention_mask = feats.get("attention_mask")
            if attention_mask is not None:
                if int(attention_mask.sum().item()) == int(attention_mask.numel()):
                    attention_mask = None
                else:
                    attention_mask = attention_mask.to(self.device)
        return input_features, attention_mask

    @torch.no_grad()
    def _transcribe_chunk_result(
        self,
        audio: np.ndarray,
        *,
        sr: int,
        max_new_tokens: int,
        language: str | None,
        timer: StageTimer,
    ) -> ASRResult:
        input_features, attention_mask = self._extract_features(audio, sr=sr, timer=timer)

        with timer.cuda("audio_tower"):
            audio_embeds = self.audio_tower(input_features, attention_mask)[0].contiguous()
        t_audio = int(audio_embeds.shape[0])

        with timer.cpu("prompt_build"):
            prompt_ids, audio_start_idx = self._build_prompt_ids(
                audio_token_count=t_audio,
                system_text="",
                force_language=language,
            )
            input_ids = torch.tensor(prompt_ids, dtype=torch.long)

        self.decoder.reset()
        with timer.cuda("text_prefill"):
            next_token = int(self.decoder.prefill_asr(input_ids, audio_embeds, int(audio_start_idx)))

        if os.environ.get("MEGAQWEN_DEBUG_LDG_STAGE", "0") not in ("", "0", "false", "False"):
            try:
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    cycles = self.decoder.debug_ldg_decode_cycles(int(next_token))
                    end.record()
                    torch.cuda.synchronize()
                    km = float(start.elapsed_time(end))
                else:
                    cycles = self.decoder.debug_ldg_decode_cycles(int(next_token))
                    km = None
                _maybe_print_ldg_stage_debug(cycles, kernel_ms=km)
            finally:
                self.decoder.reset()
                with timer.cuda("text_prefill"):
                    next_token = int(self.decoder.prefill_asr(input_ids, audio_embeds, int(audio_start_idx)))

        generated, stop_reason = self._decode_greedy_tokens(int(next_token), max_new_tokens=max_new_tokens, timer=timer)
        from .qwen3_asr import _parse_asr_output
        with timer.cpu("token_decode"):
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        lang, text = _parse_asr_output(raw, user_language=language)
        return ASRResult(text=text, language=lang, generated_tokens=len(generated), stop_reason=stop_reason)

    @torch.no_grad()
    def transcribe(
        self,
        wav_path: str | Path,
        *,
        max_new_tokens: int | None = None,
        language: str | None = None,
        return_language: bool = False,
        timer: StageTimer | None = None,
    ) -> str | tuple[str | None, str]:
        res = self.transcribe_result(
            wav_path,
            max_new_tokens=max_new_tokens,
            language=language,
            timer=timer,
        )
        return (res.language, res.text) if return_language else res.text

    @torch.no_grad()
    def transcribe_result(
        self,
        wav_path: str | Path,
        *,
        max_new_tokens: int | None = None,
        language: str | None = None,
        timer: StageTimer | None = None,
    ) -> ASRResult:
        timer = timer if timer is not None else StageTimer(enabled=False)

        with timer.cpu("wav_load"):
            audio, sr = load_wav_mono(wav_path, target_sr=16000)

        max_new = int(max_new_tokens if max_new_tokens is not None else self.opts.max_new_tokens)
        chunk_sec = self._asr_chunk_seconds()
        chunks: list[np.ndarray]
        if chunk_sec > 0.0:
            with timer.cpu("chunk_split"):
                chunks = _split_audio_into_chunks(audio, sr=sr, max_chunk_sec=chunk_sec)
        else:
            chunks = [np.asarray(audio, dtype=np.float32)]

        results: list[ASRResult] = []
        pending: list[np.ndarray] = [np.asarray(c, dtype=np.float32, copy=False) for c in chunks]
        min_samples_to_split = max(int(sr), 2)
        while pending:
            piece = pending.pop(0)
            try:
                result = self._transcribe_chunk_result(
                    piece,
                    sr=sr,
                    max_new_tokens=max_new,
                    language=language,
                    timer=timer,
                )
                results.append(result)
            except RuntimeError as exc:
                msg = str(exc)
                can_split = (
                    ("Prefill sequence length exceeds maximum" in msg)
                    or ("decode_steps would exceed max_seq_len" in msg)
                    or ("decode_steps_cuda_graph would exceed max_seq_len" in msg)
                )
                if can_split and int(piece.shape[0]) > min_samples_to_split:
                    mid = int(piece.shape[0] // 2)
                    if mid <= 0 or mid >= int(piece.shape[0]):
                        raise
                    pending.insert(0, piece[mid:])
                    pending.insert(0, piece[:mid])
                    continue
                raise

        merged_text = "".join(r.text for r in results if r.text)
        merged_lang = _merge_chunk_languages([r.language for r in results], user_language=language)
        merged_tokens = int(sum(int(r.generated_tokens) for r in results))
        merged_stop = "max_new_tokens" if any(r.stop_reason != "eos" for r in results) else "eos"
        return ASRResult(text=merged_text, language=merged_lang, generated_tokens=merged_tokens, stop_reason=merged_stop)
