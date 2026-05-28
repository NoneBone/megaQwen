import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
import os
from pathlib import Path

import torch
from safetensors.torch import safe_open

import megakernel_kernels as mk_kernels


def _collect_layer_keys(text_prefix: str, lm_head_key: str | None, num_layers: int) -> list[str]:
    keys = [text_prefix + "embed_tokens.weight", text_prefix + "norm.weight"]
    if lm_head_key is not None:
        keys.append(lm_head_key)
    for i in range(num_layers):
        p = f"{text_prefix}layers.{i}."
        keys.extend(
            [
                p + "input_layernorm.weight",
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.q_norm.weight",
                p + "self_attn.k_norm.weight",
                p + "self_attn.o_proj.weight",
                p + "post_attention_layernorm.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
        )
    return keys


def _collect_linear_weight_keys(text_prefix: str, lm_head_key: str | None, num_layers: int) -> list[str]:
    keys: list[str] = []
    for i in range(num_layers):
        p = f"{text_prefix}layers.{i}."
        keys.extend(
            [
                p + "self_attn.q_proj.weight",
                p + "self_attn.k_proj.weight",
                p + "self_attn.v_proj.weight",
                p + "self_attn.o_proj.weight",
                p + "mlp.gate_proj.weight",
                p + "mlp.up_proj.weight",
                p + "mlp.down_proj.weight",
            ]
        )
    if lm_head_key is not None:
        keys.append(lm_head_key)
    return keys


def _as_bf16_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    return tensor.contiguous()


def _parse_bnb_quant_state_json(blob: torch.Tensor) -> dict:
    data = blob.detach().cpu().contiguous().view(-1).numpy().tobytes()
    return json.loads(data.decode("utf-8"))


def _bnb4_block_scales_and_codebook(
    weight_key: str,
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    state_key_nf4 = weight_key + ".quant_state.bitsandbytes__nf4"
    state_key_fp4 = weight_key + ".quant_state.bitsandbytes__fp4"
    state_key = state_key_nf4 if state_key_nf4 in tensors else state_key_fp4
    if state_key not in tensors:
        raise RuntimeError(f"Missing quant_state tensor for {weight_key}")

    qs = _parse_bnb_quant_state_json(tensors[state_key])
    blocksize = int(qs.get("blocksize", 64))
    nested_blocksize = int(qs.get("nested_blocksize", 256))
    nested_offset = float(qs.get("nested_offset", 0.0))

    absmax_key = weight_key + ".absmax"
    quant_map_key = weight_key + ".quant_map"
    if absmax_key not in tensors or quant_map_key not in tensors:
        raise RuntimeError(f"Missing absmax/quant_map tensors for {weight_key}")

    absmax = tensors[absmax_key]
    codebook = tensors[quant_map_key].to(torch.float32).contiguous()
    if absmax.dtype == torch.uint8:
        nabs_key = weight_key + ".nested_absmax"
        nmap_key = weight_key + ".nested_quant_map"
        if nabs_key not in tensors or nmap_key not in tensors:
            raise RuntimeError(f"Missing nested quant tensors for {weight_key}")
        nested_absmax = tensors[nabs_key].to(torch.float32)
        nested_quant_map = tensors[nmap_key].to(torch.float32)
        idx = absmax.to(torch.long).view(-1)
        block_scales = nested_quant_map[idx]
        groups = torch.arange(block_scales.numel(), device=block_scales.device) // nested_blocksize
        block_scales = block_scales * nested_absmax[groups] + nested_offset
    else:
        block_scales = absmax.to(torch.float32).view(-1)

    if block_scales.numel() <= 0:
        raise RuntimeError(f"Invalid block scales for {weight_key}")
    if blocksize != 64:
        raise RuntimeError(f"Unsupported blocksize for {weight_key}: {blocksize}, expected 64")
    return block_scales.contiguous(), codebook


def _collect_layer_weights(tensors: dict[str, torch.Tensor], text_prefix: str, num_layers: int) -> list[torch.Tensor]:
    layer_weights = []
    for i in range(num_layers):
        p = f"{text_prefix}layers.{i}."
        layer_weights.extend(
            [
                _as_bf16_contiguous(tensors[p + "input_layernorm.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.q_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.k_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.v_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.q_norm.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.k_norm.weight"]),
                _as_bf16_contiguous(tensors[p + "self_attn.o_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "post_attention_layernorm.weight"]),
                _as_bf16_contiguous(tensors[p + "mlp.gate_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "mlp.up_proj.weight"]),
                _as_bf16_contiguous(tensors[p + "mlp.down_proj.weight"]),
            ]
        )
    return layer_weights


def _build_rope_tables(max_seq_len: int, rope_theta: float, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device="cuda", dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device="cuda", dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
    return cos_table, sin_table


def _resolve_local_model_dir(model_name: str) -> Path | None:
    model_dir = Path(model_name)
    if (not model_dir.exists()) and ("/" in str(model_name)):
        fallback = Path(str(model_name).split("/")[-1])
        if fallback.exists() and fallback.is_dir():
            model_dir = fallback
    if model_dir.exists() and model_dir.is_dir():
        return model_dir
    return None


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
    base = [
        weight_key + ".absmax",
        weight_key + ".quant_map",
    ]
    nested = [
        weight_key + ".nested_absmax",
        weight_key + ".nested_quant_map",
    ]
    packed_nf4 = weight_key + ".quant_state.bitsandbytes__nf4"
    packed_fp4 = weight_key + ".quant_state.bitsandbytes__fp4"
    found = [key for key in base if key in available_keys]
    found.extend([key for key in nested if key in available_keys])
    if packed_nf4 in available_keys:
        found.append(packed_nf4)
    elif packed_fp4 in available_keys:
        found.append(packed_fp4)
    return found


def _dequantize_bnb4_weight(
    weight_key: str,
    packed_weight: torch.Tensor,
    all_tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    try:
        from bitsandbytes import functional as bnbf
    except Exception as exc:
        raise RuntimeError(
            f"Detected bitsandbytes 4bit tensor for {weight_key}, but bitsandbytes is not available."
        ) from exc

    if packed_weight.dtype != torch.uint8:
        raise RuntimeError(f"Expected packed uint8 tensor for {weight_key}, got {packed_weight.dtype}")

    quant_state_key_nf4 = weight_key + ".quant_state.bitsandbytes__nf4"
    quant_state_key_fp4 = weight_key + ".quant_state.bitsandbytes__fp4"
    quant_state_key = quant_state_key_nf4 if quant_state_key_nf4 in all_tensors else quant_state_key_fp4
    if quant_state_key not in all_tensors:
        raise RuntimeError(f"Missing quant_state tensor for {weight_key}")

    qdict: dict[str, torch.Tensor] = {
        quant_state_key: all_tensors[quant_state_key],
    }
    for suffix in [".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map"]:
        full_key = weight_key + suffix
        if full_key in all_tensors:
            qdict[full_key] = all_tensors[full_key]

    quant_state = bnbf.QuantState.from_dict(qdict, device=packed_weight.device)
    dequant = bnbf.dequantize_4bit(packed_weight, quant_state=quant_state)
    return _as_bf16_contiguous(dequant)


def load_qwen3_weights(model_name: str = "/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B", max_seq_len: int = 2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = _resolve_local_model_dir(model_name)
    if model_dir is not None:
        print(f"Loading local weights: {model_dir} ...")
        cfg_path = model_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        model_type = cfg.get("model_type")

        if model_type == "qwen3_asr":
            text_prefix = "thinker.model."
            lm_head_key = "thinker.lm_head.weight"
            text_cfg = cfg["thinker_config"]["text_config"]
            mk_kernels.configure_model_spec(
                hidden_size=int(text_cfg["hidden_size"]),
                intermediate_size=int(text_cfg["intermediate_size"]),
                num_q_heads=int(text_cfg["num_attention_heads"]),
                num_kv_heads=int(text_cfg["num_key_value_heads"]),
                head_dim=int(text_cfg.get("head_dim", int(text_cfg["hidden_size"]) // int(text_cfg["num_attention_heads"]))),
                num_layers=int(text_cfg["num_hidden_layers"]),
                vocab_size=int(text_cfg.get("vocab_size", cfg.get("vocab_size", 151936))),
            )
            rope_theta = float(text_cfg.get("rope_theta", 1000000.0))
        else:
            text_prefix = "model."
            lm_head_key = "lm_head.weight"
            rope_theta = float(cfg.get("rope_theta", 1000000.0))
            model_cfg = cfg
            if "hidden_size" in model_cfg and "intermediate_size" in model_cfg:
                mk_kernels.configure_model_spec(
                    hidden_size=int(model_cfg["hidden_size"]),
                    intermediate_size=int(model_cfg["intermediate_size"]),
                    num_q_heads=int(model_cfg["num_attention_heads"]),
                    num_kv_heads=int(model_cfg["num_key_value_heads"]),
                    head_dim=int(model_cfg.get("head_dim", int(model_cfg["hidden_size"]) // int(model_cfg["num_attention_heads"]))),
                    num_layers=int(model_cfg["num_hidden_layers"]),
                    vocab_size=int(model_cfg.get("vocab_size", 151936)),
                )

        spec = mk_kernels.get_active_model_spec()
        num_layers = int(spec["num_layers"])
        head_dim = int(spec["head_dim"])

        try:
            from megaqwen.safetensors_loader import load_tensors
        except Exception as e:
            raise RuntimeError("Local safetensors loader import failed; ensure repo root is on PYTHONPATH") from e

        available_keys = _available_tensor_keys(model_dir)
        resolved_lm_head_key = lm_head_key
        if resolved_lm_head_key not in available_keys:
            resolved_lm_head_key = text_prefix + "embed_tokens.weight"

        tensors = load_tensors(model_dir, _collect_layer_keys(text_prefix, lm_head_key=None, num_layers=num_layers), device="cuda")
        if resolved_lm_head_key not in tensors:
            lm_head_tensor = load_tensors(model_dir, [resolved_lm_head_key], device="cuda")[resolved_lm_head_key]
            tensors[resolved_lm_head_key] = lm_head_tensor

        linear_weight_keys = _collect_linear_weight_keys(text_prefix, resolved_lm_head_key, num_layers=num_layers)
        quantized_weight_keys = [
            key
            for key in linear_weight_keys
            if key in tensors and tensors[key].dtype == torch.uint8
        ]
        runtime_ffn_w4_enabled = str(os.environ.get("MEGAQWEN_SPLIT_FFN_W4", "")).lower() not in ("", "0", "false")
        runtime_qkv_w4_enabled = str(os.environ.get("MEGAQWEN_SPLIT_QKV_W4", "")).lower() not in ("", "0", "false")
        runtime_o_w4_enabled = str(os.environ.get("MEGAQWEN_SPLIT_O_W4", "")).lower() not in ("", "0", "false")
        runtime_w4_enabled = runtime_ffn_w4_enabled or runtime_qkv_w4_enabled or runtime_o_w4_enabled
        split_q_w4_packed: list[torch.Tensor] = []
        split_q_w4_scales: list[torch.Tensor] = []
        split_q_w4_codebook: list[torch.Tensor] = []
        split_k_w4_packed: list[torch.Tensor] = []
        split_k_w4_scales: list[torch.Tensor] = []
        split_k_w4_codebook: list[torch.Tensor] = []
        split_v_w4_packed: list[torch.Tensor] = []
        split_v_w4_scales: list[torch.Tensor] = []
        split_v_w4_codebook: list[torch.Tensor] = []
        split_o_w4_packed: list[torch.Tensor] = []
        split_o_w4_scales: list[torch.Tensor] = []
        split_o_w4_codebook: list[torch.Tensor] = []
        split_gateup_w4_packed: list[torch.Tensor] = []
        split_gateup_w4_scales: list[torch.Tensor] = []
        split_gateup_w4_codebook: list[torch.Tensor] = []
        split_down_w4_packed: list[torch.Tensor] = []
        split_down_w4_scales: list[torch.Tensor] = []
        split_down_w4_codebook: list[torch.Tensor] = []
        if quantized_weight_keys:
            print(
                f"[MEGAQWEN_W4] detected {len(quantized_weight_keys)} packed 4bit weights; "
                "dequantizing to BF16 on load (compat mode)"
            )
            quant_meta_keys: list[str] = []
            for weight_key in quantized_weight_keys:
                quant_meta_keys.extend(_collect_bnb4_meta_keys(weight_key, available_keys))
            quant_meta_keys = list(dict.fromkeys(quant_meta_keys))
            if quant_meta_keys:
                tensors.update(load_tensors(model_dir, quant_meta_keys, device="cuda"))

            if runtime_w4_enabled:
                try:
                    for i in range(num_layers):
                        p = f"{text_prefix}layers.{i}."
                        q_key = p + "self_attn.q_proj.weight"
                        k_key = p + "self_attn.k_proj.weight"
                        v_key = p + "self_attn.v_proj.weight"
                        o_key = p + "self_attn.o_proj.weight"
                        gate_key = p + "mlp.gate_proj.weight"
                        up_key = p + "mlp.up_proj.weight"
                        down_key = p + "mlp.down_proj.weight"

                        if runtime_qkv_w4_enabled:
                            if not (
                                q_key in tensors and k_key in tensors and v_key in tensors and
                                tensors[q_key].dtype == torch.uint8 and
                                tensors[k_key].dtype == torch.uint8 and
                                tensors[v_key].dtype == torch.uint8
                            ):
                                raise RuntimeError(f"Layer {i} missing W4 packed QKV tensors")
                            q_scales, q_code = _bnb4_block_scales_and_codebook(q_key, tensors)
                            k_scales, k_code = _bnb4_block_scales_and_codebook(k_key, tensors)
                            v_scales, v_code = _bnb4_block_scales_and_codebook(v_key, tensors)
                            split_q_w4_packed.append(tensors[q_key].view(-1).contiguous())
                            split_q_w4_scales.append(q_scales.contiguous())
                            split_q_w4_codebook.append(q_code.contiguous())
                            split_k_w4_packed.append(tensors[k_key].view(-1).contiguous())
                            split_k_w4_scales.append(k_scales.contiguous())
                            split_k_w4_codebook.append(k_code.contiguous())
                            split_v_w4_packed.append(tensors[v_key].view(-1).contiguous())
                            split_v_w4_scales.append(v_scales.contiguous())
                            split_v_w4_codebook.append(v_code.contiguous())

                        if runtime_o_w4_enabled:
                            if not (o_key in tensors and tensors[o_key].dtype == torch.uint8):
                                raise RuntimeError(f"Layer {i} missing W4 packed O tensor")
                            o_scales, o_code = _bnb4_block_scales_and_codebook(o_key, tensors)
                            split_o_w4_packed.append(tensors[o_key].view(-1).contiguous())
                            split_o_w4_scales.append(o_scales.contiguous())
                            split_o_w4_codebook.append(o_code.contiguous())

                        if runtime_ffn_w4_enabled:
                            if not (
                                gate_key in tensors and up_key in tensors and down_key in tensors and
                                tensors[gate_key].dtype == torch.uint8 and
                                tensors[up_key].dtype == torch.uint8 and
                                tensors[down_key].dtype == torch.uint8
                            ):
                                raise RuntimeError(f"Layer {i} missing W4 packed FFN tensors")
                            gate_scales, gate_code = _bnb4_block_scales_and_codebook(gate_key, tensors)
                            up_scales, up_code = _bnb4_block_scales_and_codebook(up_key, tensors)
                            down_scales, down_code = _bnb4_block_scales_and_codebook(down_key, tensors)
                            split_gateup_w4_packed.append(
                                torch.cat([tensors[gate_key].view(-1), tensors[up_key].view(-1)], dim=0).contiguous()
                            )
                            split_gateup_w4_scales.append(torch.cat([gate_scales, up_scales], dim=0).contiguous())
                            split_gateup_w4_codebook.append(torch.cat([gate_code, up_code], dim=0).contiguous())
                            split_down_w4_packed.append(tensors[down_key].view(-1).contiguous())
                            split_down_w4_scales.append(down_scales.contiguous())
                            split_down_w4_codebook.append(down_code.contiguous())

                    msg_parts: list[str] = []
                    if runtime_qkv_w4_enabled:
                        msg_parts.append("QKV")
                    if runtime_o_w4_enabled:
                        msg_parts.append("O")
                    if runtime_ffn_w4_enabled:
                        msg_parts.append("FFN")
                    msg = "+".join(msg_parts) if msg_parts else "none"
                    print(f"[MEGAQWEN_W4] runtime {msg} W4 tensors prepared for split decode")
                except Exception as exc:
                    split_q_w4_packed = []
                    split_q_w4_scales = []
                    split_q_w4_codebook = []
                    split_k_w4_packed = []
                    split_k_w4_scales = []
                    split_k_w4_codebook = []
                    split_v_w4_packed = []
                    split_v_w4_scales = []
                    split_v_w4_codebook = []
                    split_o_w4_packed = []
                    split_o_w4_scales = []
                    split_o_w4_codebook = []
                    split_gateup_w4_packed = []
                    split_gateup_w4_scales = []
                    split_gateup_w4_codebook = []
                    split_down_w4_packed = []
                    split_down_w4_scales = []
                    split_down_w4_codebook = []
                    print(f"[MEGAQWEN_W4] runtime FFN W4 prepare failed, fallback to BF16 split path: {exc}")

            for weight_key in quantized_weight_keys:
                tensors[weight_key] = _dequantize_bnb4_weight(weight_key, tensors[weight_key], tensors)

        sample_key = text_prefix + "layers.0.self_attn.q_proj.weight"
        if tensors[sample_key].dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise RuntimeError(f"Unsupported dtype for {sample_key}: {tensors[sample_key].dtype}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        cos_table, sin_table = _build_rope_tables(max_seq_len=max_seq_len, rope_theta=rope_theta, head_dim=head_dim)

        embed_weight = _as_bf16_contiguous(tensors[text_prefix + "embed_tokens.weight"])
        final_norm_weight = _as_bf16_contiguous(tensors[text_prefix + "norm.weight"])
        lm_head_weight = _as_bf16_contiguous(tensors[resolved_lm_head_key])

        return {
            "embed_weight": embed_weight,
            "layer_weights": _collect_layer_weights(tensors, text_prefix=text_prefix, num_layers=num_layers),
            "final_norm_weight": final_norm_weight,
            "lm_head_weight": lm_head_weight,
            "cos_table": cos_table,
            "sin_table": sin_table,
            "tokenizer": tokenizer,
            "split_q_w4_packed": split_q_w4_packed,
            "split_q_w4_scales": split_q_w4_scales,
            "split_q_w4_codebook": split_q_w4_codebook,
            "split_k_w4_packed": split_k_w4_packed,
            "split_k_w4_scales": split_k_w4_scales,
            "split_k_w4_codebook": split_k_w4_codebook,
            "split_v_w4_packed": split_v_w4_packed,
            "split_v_w4_scales": split_v_w4_scales,
            "split_v_w4_codebook": split_v_w4_codebook,
            "split_o_w4_packed": split_o_w4_packed,
            "split_o_w4_scales": split_o_w4_scales,
            "split_o_w4_codebook": split_o_w4_codebook,
            "split_gateup_w4_packed": split_gateup_w4_packed,
            "split_gateup_w4_scales": split_gateup_w4_scales,
            "split_gateup_w4_codebook": split_gateup_w4_codebook,
            "split_down_w4_packed": split_down_w4_packed,
            "split_down_w4_scales": split_down_w4_scales,
            "split_down_w4_codebook": split_down_w4_codebook,
        }

    print(f"Loading {model_name} (HuggingFace) ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()
    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and hasattr(model_cfg, "hidden_size"):
        mk_kernels.configure_model_spec(
            hidden_size=int(model_cfg.hidden_size),
            intermediate_size=int(model_cfg.intermediate_size),
            num_q_heads=int(model_cfg.num_attention_heads),
            num_kv_heads=int(model_cfg.num_key_value_heads),
            head_dim=int(getattr(model_cfg, "head_dim", int(model_cfg.hidden_size) // int(model_cfg.num_attention_heads))),
            num_layers=int(model_cfg.num_hidden_layers),
            vocab_size=int(model_cfg.vocab_size),
        )
    spec = mk_kernels.get_active_model_spec()
    num_layers = int(spec["num_layers"])
    head_dim = int(spec["head_dim"])

    rope_theta = float(getattr(getattr(model, "config", None), "rope_theta", 1000000.0))
    cos_table, sin_table = _build_rope_tables(max_seq_len=max_seq_len, rope_theta=rope_theta, head_dim=head_dim)
    layer_weights = _collect_layer_weights(state, text_prefix="model.", num_layers=num_layers)

    del model
    torch.cuda.empty_cache()

    lm_head_weight = state.get("lm_head.weight", state["model.embed_tokens.weight"])
    return {
        "embed_weight": _as_bf16_contiguous(state["model.embed_tokens.weight"]),
        "layer_weights": layer_weights,
        "final_norm_weight": _as_bf16_contiguous(state["model.norm.weight"]),
        "lm_head_weight": _as_bf16_contiguous(lm_head_weight),
        "cos_table": cos_table,
        "sin_table": sin_table,
        "tokenizer": tokenizer,
        "split_q_w4_packed": [],
        "split_q_w4_scales": [],
        "split_q_w4_codebook": [],
        "split_k_w4_packed": [],
        "split_k_w4_scales": [],
        "split_k_w4_codebook": [],
        "split_v_w4_packed": [],
        "split_v_w4_scales": [],
        "split_v_w4_codebook": [],
        "split_o_w4_packed": [],
        "split_o_w4_scales": [],
        "split_o_w4_codebook": [],
        "split_gateup_w4_packed": [],
        "split_gateup_w4_scales": [],
        "split_gateup_w4_codebook": [],
        "split_down_w4_packed": [],
        "split_down_w4_scales": [],
        "split_down_w4_codebook": [],
    }
