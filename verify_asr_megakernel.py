"""Sanity-check Qwen3-ASR megakernel vs torch baseline on the same audio.

Usage:
  python verify_asr_megakernel.py --model Qwen3-ASR-0.6B --audio your.wav --device cuda --dtype bf16
  python verify_asr_megakernel.py --model Qwen3-ASR-1.7B --audio your.wav --device cuda --dtype bf16
"""

import argparse
import time
import wave
import os

import torch
from torch.profiler import ProfilerActivity, profile

from megaqwen.qwen3_asr import Qwen3ASRModel
from megaqwen.qwen3_asr_megakernel import (
    Qwen3ASRMegakernelModel,
    Qwen3ASRMegakernelOptions,
    _apply_asr_megakernel_default_knobs,
)
from megaqwen.profiler import StageTimer, StageSummary


def _wav_duration_s(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wf:
            n = wf.getnframes()
            sr = wf.getframerate()
        return float(n) / float(sr) if sr > 0 else None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--audio", default="/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/AUDIO/hello.wav")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument(
        "--target",
        default="both",
        choices=["mk", "torch", "both"],
        help="Which implementation(s) to run (default: both). Use mk to skip loading torch baseline (helps avoid OOM).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--language", default=None, help='Force language, e.g. "Chinese" or "English" (default: auto)')
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-prefill-len", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=1, help="Timed runs (excluding warmup)")
    ap.add_argument("--warmup", type=int, default=0, help="Warmup runs (not timed)")
    ap.add_argument("--ENV_METHOD", type=int, default=33, help="Conviently decode beckend control")
    ap.add_argument("--profile", action="store_true", help="Print per-stage timing breakdown")
    ap.add_argument("--stats", action="store_true", help="Print generated token count and stop reason")
    ap.add_argument(
        "--torch-prof",
        default="",
        help="If set, write a Chrome trace (json) from torch.profiler for quick bottleneck debugging.",
    )
    ap.add_argument(
        "--torch-prof-summary",
        action="store_true",
        help="With --torch-prof, also print a small op summary table sorted by CUDA time.",
    )
    ap.add_argument(
        "--torch-prof-target",
        default="mk",
        choices=["mk", "torch", "both"],
        help="Which path to capture with --torch-prof (default: mk).",
    )
    ap.add_argument(
        "--order",
        default="torch-first",
        choices=["torch-first", "mk-first", "alternate"],
        help="Run order to reduce drift/thermal bias (recommend: alternate)",
    )
    args = ap.parse_args()

    _apply_asr_megakernel_default_knobs(args.ENV_METHOD)

    # Step-0 style: print key knobs so perf numbers are reproducible.
    print(
        "knobs:",
        "MEGAQWEN_DECODE_CHUNK=" + os.environ.get("MEGAQWEN_DECODE_CHUNK", ""),
        "MEGAQWEN_PREFILL_BACKEND=" + os.environ.get("MEGAQWEN_PREFILL_BACKEND", ""),
        "MEGAQWEN_PREFILL_ATTN_IMPL=" + os.environ.get("MEGAQWEN_PREFILL_ATTN_IMPL", ""),
        "MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL=" + os.environ.get("MEGAQWEN_PREFILL_ATTN_EXPERIMENTAL", ""),
        "MEGAQWEN_PREFILL_ATTN_WARPS=" + os.environ.get("MEGAQWEN_PREFILL_ATTN_WARPS", ""),
        "MEGAQWEN_PREFILL_LEGACY_ATTN_WARPS=" + os.environ.get("MEGAQWEN_PREFILL_LEGACY_ATTN_WARPS", ""),
        "MEGAQWEN_PREFILL_ATTN_AUTO_MIN_SEQ=" + os.environ.get("MEGAQWEN_PREFILL_ATTN_AUTO_MIN_SEQ", ""),
        "MEGAQWEN_PREFILL_FLASH_EXT_FORCE_FP32=" + os.environ.get("MEGAQWEN_PREFILL_FLASH_EXT_FORCE_FP32", ""),
        "MEGAQWEN_PREFILL_FLASH_EXT_TAIL_LEGACY_LAYERS=" + os.environ.get("MEGAQWEN_PREFILL_FLASH_EXT_TAIL_LEGACY_LAYERS", ""),
        "MEGAQWEN_DEBUG_PREFILL_FLASH=" + os.environ.get("MEGAQWEN_DEBUG_PREFILL_FLASH", ""),
        "MEGAQWEN_DEBUG_PREFILL_FLASH_SKIP=" + os.environ.get("MEGAQWEN_DEBUG_PREFILL_FLASH_SKIP", ""),
        "MEGAQWEN_DEBUG_PREFILL_STAGE=" + os.environ.get("MEGAQWEN_DEBUG_PREFILL_STAGE", ""),
        "MEGAQWEN_DEBUG_PREFILL_STAGE_SKIP=" + os.environ.get("MEGAQWEN_DEBUG_PREFILL_STAGE_SKIP", ""),
        "MEGAQWEN_DECODE_BACKEND=" + os.environ.get("MEGAQWEN_DECODE_BACKEND", ""),
        "MEGAQWEN_DECODE_GPU_LOOP=" + os.environ.get("MEGAQWEN_DECODE_GPU_LOOP", ""),
        "MEGAQWEN_ASR_CHUNK_SEC=" + os.environ.get("MEGAQWEN_ASR_CHUNK_SEC", ""),
        "MEGAQWEN_DECODE_V2_CHECK_STRIDE=" + os.environ.get("MEGAQWEN_DECODE_V2_CHECK_STRIDE", ""),
        "MEGAQWEN_SPLIT_GEMM_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_GEMM_IMPL", ""),
        "MEGAQWEN_SPLIT_O_GEMM_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_O_GEMM_IMPL", ""),
        "MEGAQWEN_SPLIT_GATEUP_GEMM_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_GATEUP_GEMM_IMPL", ""),
        "MEGAQWEN_SPLIT_DOWN_GEMM_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_DOWN_GEMM_IMPL", ""),
        "MEGAQWEN_SPLIT_FFN_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_FFN_IMPL", ""),
        "MEGAQWEN_SPLIT_QKV_W4=" + os.environ.get("MEGAQWEN_SPLIT_QKV_W4", ""),
        "MEGAQWEN_SPLIT_O_W4=" + os.environ.get("MEGAQWEN_SPLIT_O_W4", ""),
        "MEGAQWEN_SPLIT_FFN_W4=" + os.environ.get("MEGAQWEN_SPLIT_FFN_W4", ""),
        "MEGAQWEN_SPLIT_FFN_W4_FUSED=" + os.environ.get("MEGAQWEN_SPLIT_FFN_W4_FUSED", ""),
        "MEGAQWEN_SPLIT_KV_LAYOUT=" + os.environ.get("MEGAQWEN_SPLIT_KV_LAYOUT", ""), # SPLIT_KV
        "MEGAQWEN_SPLIT_KV_PAGED=" + os.environ.get("MEGAQWEN_SPLIT_KV_PAGED", ""),
        "MEGAQWEN_SPLIT_FLASH_GQA_SHARE=" + os.environ.get("MEGAQWEN_SPLIT_FLASH_GQA_SHARE", ""),
        "MEGAQWEN_SPLIT_FLASH_GQA_MODE=" + os.environ.get("MEGAQWEN_SPLIT_FLASH_GQA_MODE", ""),
        "MEGAQWEN_SPLIT_KV_BLOCK_SIZE=" + os.environ.get("MEGAQWEN_SPLIT_KV_BLOCK_SIZE", ""),
        "MEGAQWEN_DEBUG_PAGED_KV=" + os.environ.get("MEGAQWEN_DEBUG_PAGED_KV", ""),
        "MEGAQWEN_DEBUG_PAGED_KV_TOKENS=" + os.environ.get("MEGAQWEN_DEBUG_PAGED_KV_TOKENS", ""),
        "MEGAQWEN_SPLIT_LTWK_MB=" + os.environ.get("MEGAQWEN_SPLIT_LTWK_MB", ""),
        "MEGAQWEN_SPLIT_ATTN_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_ATTN_IMPL", ""),
        "MEGAQWEN_SPLIT_ATTN_WARPS=" + os.environ.get("MEGAQWEN_SPLIT_ATTN_WARPS", ""),
        "MEGAQWEN_SPLIT_ATTN_CHUNK=" + os.environ.get("MEGAQWEN_SPLIT_ATTN_CHUNK", ""),
        "MEGAQWEN_SPLIT_QKV_GEMM_IMPL=" + os.environ.get("MEGAQWEN_SPLIT_QKV_GEMM_IMPL", ""),
        "MEGAQWEN_DEBUG_SPLIT_STAGE=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_STAGE", ""),
        "MEGAQWEN_DEBUG_SPLIT_STAGE_SKIP=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_STAGE_SKIP", ""),
        "MEGAQWEN_DEBUG_SPLIT_STAGE_AVG=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG", ""),
        "MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_STRIDE=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_STRIDE", ""),
        "MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_EXACT=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_STAGE_AVG_EXACT", ""),
        "MEGAQWEN_DEBUG_SPLIT_LAYER_TOPK=" + os.environ.get("MEGAQWEN_DEBUG_SPLIT_LAYER_TOPK", ""),
        "MEGAQWEN_FUSE_LM_HEAD=" + os.environ.get("MEGAQWEN_FUSE_LM_HEAD", ""),
        "MEGAQWEN_ATTN_ALL_BLOCKS=" + os.environ.get("MEGAQWEN_ATTN_ALL_BLOCKS", ""),
        "MEGAQWEN_DEBUG_LDG_STAGE=" + os.environ.get("MEGAQWEN_DEBUG_LDG_STAGE", ""),
        "MEGAQWEN_LDG_NUM_BLOCKS=" + os.environ.get("MEGAQWEN_LDG_NUM_BLOCKS", ""),
        "MEGAQWEN_LDG_LM_NUM_BLOCKS=" + os.environ.get("MEGAQWEN_LDG_LM_NUM_BLOCKS", ""),
    )

    audio_dur = _wav_duration_s(args.audio)

    want_torch = args.target in ("torch", "both")
    want_mk = args.target in ("mk", "both")

    torch_model = None
    if want_torch:
        torch_model = Qwen3ASRModel.from_pretrained(args.model, device=args.device, dtype=args.dtype)

    opts = Qwen3ASRMegakernelOptions(
        max_seq_len=args.max_seq_len,
        max_prefill_len=args.max_prefill_len,
        max_new_tokens=args.max_new_tokens,
    )
    mk_model = None
    if want_mk:
        mk_model = Qwen3ASRMegakernelModel.from_pretrained(args.model, device=args.device, dtype=args.dtype, opts=opts)

    if args.torch_prof:
        if args.torch_prof_target in ("torch", "both") and torch_model is None:
            raise SystemExit("--torch-prof-target includes torch, but --target is not torch/both")
        if args.torch_prof_target in ("mk", "both") and mk_model is None:
            raise SystemExit("--torch-prof-target includes mk, but --target is not mk/both")
        acts: list[ProfilerActivity] = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        # Warm up once to avoid capturing cuDNN algorithm search / initial CUDA graph creation
        # in the profiler trace.
        if args.torch_prof_target in ("mk", "both"):
            _ = mk_model.transcribe(args.audio, max_new_tokens=min(args.max_new_tokens, 16), language=args.language)
        if args.torch_prof_target in ("torch", "both"):
            _ = torch_model.transcribe(args.audio, max_new_tokens=min(args.max_new_tokens, 16), language=args.language)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with profile(
            activities=acts,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            if args.torch_prof_target in ("mk", "both"):
                _ = mk_model.transcribe(
                    args.audio,
                    max_new_tokens=args.max_new_tokens,
                    language=args.language,
                )
            if args.torch_prof_target in ("torch", "both"):
                _ = torch_model.transcribe(
                    args.audio,
                    max_new_tokens=args.max_new_tokens,
                    language=args.language,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.torch_prof)
        print(f"torch.profiler trace saved: {args.torch_prof}")
        if args.torch_prof_summary:
            try:
                # Note: This summary does not require hardware perf counters.
                print(
                    prof.key_averages().table(
                        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                        row_limit=25,
                    )
                )
            except Exception as e:
                print(f"(torch-prof-summary failed: {e})")
        return

    def _run_torch() -> tuple[str, float, StageSummary, object | None]:
        timer = StageTimer(enabled=args.profile)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.stats:
            res = torch_model.transcribe_result(args.audio, max_new_tokens=args.max_new_tokens, language=args.language, timer=timer)
            text = f"{res.text}"
        else:
            res = None
            text = torch_model.transcribe(args.audio, max_new_tokens=args.max_new_tokens, language=args.language, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return text, (t1 - t0), timer.summary(), res

    def _run_mk() -> tuple[str, float, StageSummary, object | None]:
        timer = StageTimer(enabled=args.profile)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.stats:
            res = mk_model.transcribe_result(args.audio, max_new_tokens=args.max_new_tokens, language=args.language, timer=timer)
            text = f"{res.text}"
        else:
            res = None
            text = mk_model.transcribe(args.audio, max_new_tokens=args.max_new_tokens, language=args.language, timer=timer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return text, (t1 - t0), timer.summary(), res

    # Warmup
    for _ in range(int(args.warmup)):
        if want_torch:
            _ = torch_model.transcribe(args.audio, max_new_tokens=min(args.max_new_tokens, 16), language=args.language)
        if want_mk:
            _ = mk_model.transcribe(args.audio, max_new_tokens=min(args.max_new_tokens, 16), language=args.language)

    def _accumulate(dst: dict[str, float], src: dict[str, float]) -> None:
        for k, v in src.items():
            dst[k] = dst.get(k, 0.0) + float(v)

    torch_times_s: list[float] = []
    mk_times_s: list[float] = []
    torch_cpu_ms: dict[str, float] = {}
    torch_cuda_ms: dict[str, float] = {}
    mk_cpu_ms: dict[str, float] = {}
    mk_cuda_ms: dict[str, float] = {}
    torch_text = ""
    mk_text = ""
    torch_res = None
    mk_res = None
    runs = int(args.runs)
    for i in range(runs):
        if want_torch and want_mk and (args.order == "mk-first" or (args.order == "alternate" and (i % 2 == 1))):
            mk_text, dt, summ, mk_res = _run_mk()
            mk_times_s.append(dt)
            _accumulate(mk_cpu_ms, summ.cpu_ms)
            _accumulate(mk_cuda_ms, summ.cuda_ms)

            torch_text, dt, summ, torch_res = _run_torch()
            torch_times_s.append(dt)
            _accumulate(torch_cpu_ms, summ.cpu_ms)
            _accumulate(torch_cuda_ms, summ.cuda_ms)
        elif want_torch and want_mk:
            torch_text, dt, summ, torch_res = _run_torch()
            torch_times_s.append(dt)
            _accumulate(torch_cpu_ms, summ.cpu_ms)
            _accumulate(torch_cuda_ms, summ.cuda_ms)

            mk_text, dt, summ, mk_res = _run_mk()
            mk_times_s.append(dt)
            _accumulate(mk_cpu_ms, summ.cpu_ms)
            _accumulate(mk_cuda_ms, summ.cuda_ms)
        elif want_mk:
            mk_text, dt, summ, mk_res = _run_mk()
            mk_times_s.append(dt)
            _accumulate(mk_cpu_ms, summ.cpu_ms)
            _accumulate(mk_cuda_ms, summ.cuda_ms)
        else:
            torch_text, dt, summ, torch_res = _run_torch()
            torch_times_s.append(dt)
            _accumulate(torch_cpu_ms, summ.cpu_ms)
            _accumulate(torch_cuda_ms, summ.cuda_ms)

    if want_torch:
        print("Torch:", torch_text)
    if want_mk:
        print("MegaK:", mk_text)
    if args.stats:
        if want_torch and torch_res is not None:
            print(f"Torch tokens: {torch_res.generated_tokens} stop={torch_res.stop_reason} lang={torch_res.language}")
        if want_mk and mk_res is not None:
            print(f"MegaK tokens: {mk_res.generated_tokens} stop={mk_res.stop_reason} lang={mk_res.language}")
    torch_avg_s = (sum(torch_times_s) / max(1, len(torch_times_s))) if want_torch else 0.0
    mk_avg_s = (sum(mk_times_s) / max(1, len(mk_times_s))) if want_mk else 0.0
    if want_torch:
        print(f"Torch latency: {torch_avg_s * 1000:.1f} ms (avg over {runs} runs)")
    if want_mk:
        print(f"MegaK latency: {mk_avg_s * 1000:.1f} ms (avg over {runs} runs)")
    if audio_dur is not None:
        print(f"Audio duration: {audio_dur:.3f} s")
        if want_torch:
            print(f"Torch RTF: {torch_avg_s / audio_dur:.6f}")
        if want_mk:
            print(f"MegaK RTF: {mk_avg_s / audio_dur:.6f}")

    if args.profile:
        def _avg(d: dict[str, float]) -> dict[str, float]:
            return {k: v / float(runs) for k, v in d.items()}

        def _print(name: str, cpu_ms: dict[str, float], cuda_ms: dict[str, float]) -> None:
            print(f"\n[{name}] stage breakdown (avg ms)")
            if cpu_ms:
                print("CPU:")
                for k in sorted(cpu_ms.keys()):
                    print(f"  {k}: {cpu_ms[k]:.3f}")
            if cuda_ms:
                print("CUDA:")
                for k in sorted(cuda_ms.keys()):
                    print(f"  {k}: {cuda_ms[k]:.3f}")

        if want_torch:
            _print("Torch", _avg(torch_cpu_ms), _avg(torch_cuda_ms))
        if want_mk:
            _print("MegaK", _avg(mk_cpu_ms), _avg(mk_cuda_ms))

    if args.stats and args.profile:
        # Rough decode efficiency (ms/token) from averaged CUDA stage time.
        torch_decode_ms = (torch_cuda_ms.get("text_decode", 0.0) / float(runs)) if torch_cuda_ms else 0.0
        mk_decode_ms = (mk_cuda_ms.get("text_decode", 0.0) / float(runs)) if mk_cuda_ms else 0.0
        if want_torch and torch_res is not None and torch_res.generated_tokens > 0 and torch_decode_ms > 0:
            print(f"Torch decode: {torch_decode_ms / torch_res.generated_tokens:.3f} ms/token")
        if want_mk and mk_res is not None and mk_res.generated_tokens > 0 and mk_decode_ms > 0:
            print(f"MegaK decode: {mk_decode_ms / mk_res.generated_tokens:.3f} ms/token")


if __name__ == "__main__":
    main()
