import torch

from megakernel_kernels import (
    NUM_LAYERS,
    _compile_decode_kernel,
    _compile_fused_prefill_kernel,
    _compile_prefill_kernel,
)
from megakernel_weights import load_qwen3_weights


class MegakernelGenerator:
    def __init__(self, model_name="/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B", max_seq_len=2048):
        weights = load_qwen3_weights(model_name, max_seq_len=max_seq_len)
        kernel = _compile_decode_kernel()
        self.decoder = kernel.MegakernelDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
        )
        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len

    def generate(self, prompt, max_new_tokens=100, temperature=1.0, stop_tokens=None):
        self.decoder.reset()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for token_id in input_ids[:-1]:
            self.decoder.decode_step(token_id)

        generated = []
        current_token = input_ids[-1]
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        for _ in range(max_new_tokens):
            next_token = self.decoder.decode_step(current_token)
            if next_token in stop_tokens:
                break
            generated.append(next_token)
            current_token = next_token
            if self.decoder.position() >= self.max_seq_len - 1:
                break
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        self.decoder.reset()
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for token_id in input_ids[:-1]:
            self.decoder.decode_step(token_id)

        current_token = input_ids[-1]
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]
        for _ in range(max_new_tokens):
            next_token = self.decoder.decode_step(current_token)
            if next_token in stop_tokens:
                break
            yield self.tokenizer.decode([next_token])
            current_token = next_token
            if self.decoder.position() >= self.max_seq_len - 1:
                break


class MegakernelPrefillGenerator:
    def __init__(self, model_name="/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B", max_seq_len=2048, max_prefill_len=512):
        weights = load_qwen3_weights(model_name, max_seq_len=max_seq_len)
        kernel = _compile_prefill_kernel()
        self.decoder = kernel.MegakernelPrefillDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            max_prefill_len,
        )
        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.max_prefill_len = max_prefill_len

    def _tokenize(self, prompt):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")
        return input_ids

    def generate(self, prompt, max_new_tokens=100, stop_tokens=None):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        generated = []
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break
            generated.append(next_token)
            if self.decoder.position() >= self.max_seq_len - 1:
                break
            next_token = self.decoder.decode_step(next_token)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break
            yield self.tokenizer.decode([next_token])
            if self.decoder.position() >= self.max_seq_len - 1:
                break
            next_token = self.decoder.decode_step(next_token)

    def prefill_only(self, prompt):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        return next_token, self.decoder.position()


class MegakernelFusedPrefillGenerator:
    def __init__(self, model_name="/media/l8w/Linux118/PROJECTS/29-vllm-serials/00-COMMON/Qwen/Qwen3-0.6B", max_seq_len=2048, max_prefill_len=64):
        weights = load_qwen3_weights(model_name, max_seq_len=max_seq_len)
        kernel = _compile_fused_prefill_kernel()
        self.decoder = kernel.MegakernelFusedPrefillDecoder(
            weights["embed_weight"],
            weights["layer_weights"],
            weights["final_norm_weight"],
            weights["lm_head_weight"],
            weights["cos_table"],
            weights["sin_table"],
            NUM_LAYERS,
            max_seq_len,
            max_prefill_len,
        )
        self.tokenizer = weights["tokenizer"]
        self.max_seq_len = max_seq_len
        self.max_prefill_len = max_prefill_len

    def _tokenize(self, prompt):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(input_ids) > self.max_prefill_len:
            raise ValueError(f"Prompt length {len(input_ids)} exceeds max prefill length {self.max_prefill_len}")
        return input_ids

    def generate(self, prompt, max_new_tokens=100, stop_tokens=None):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        generated = []
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break
            generated.append(next_token)
            if self.decoder.position() >= self.max_seq_len - 1:
                break
            next_token = self.decoder.decode_step(next_token)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(self, prompt, max_new_tokens=100, stop_tokens=None):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        stop_tokens = stop_tokens or [self.tokenizer.eos_token_id]

        for _ in range(max_new_tokens):
            if next_token in stop_tokens:
                break
            yield self.tokenizer.decode([next_token])
            if self.decoder.position() >= self.max_seq_len - 1:
                break
            next_token = self.decoder.decode_step(next_token)

    def prefill_only(self, prompt):
        self.decoder.reset()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.int32)
        next_token = self.decoder.prefill_step(input_tensor)
        return next_token, self.decoder.position()


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill", action="store_true", help="Use cuBLAS prefill kernel")
    parser.add_argument("--fused-prefill", action="store_true", help="Use fused prefill megakernel (no cuBLAS)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--seq-len", type=int, default=None, help="Test with specific sequence length")
    args = parser.parse_args()

    if args.fused_prefill:
        print("Using fused prefill megakernel + decode kernels...")
        gen = MegakernelFusedPrefillGenerator(max_prefill_len=64)
    elif args.prefill:
        print("Using cuBLAS prefill + decode kernels...")
        gen = MegakernelPrefillGenerator()
    else:
        print("Using decode-only kernel...")
        gen = MegakernelGenerator()

    prompt = "Hello, my name is"
    if not args.benchmark:
        print("Testing generation...")
        output = gen.generate(prompt, max_new_tokens=20)
        print(f"Output: {output}")
        return

    use_prefill = args.prefill or args.fused_prefill
    if args.seq_len is not None:
        base_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)
        if args.seq_len > len(base_ids):
            pad_token = base_ids[-1]
            input_ids = base_ids + [pad_token] * (args.seq_len - len(base_ids))
        else:
            input_ids = base_ids[:args.seq_len]
    else:
        input_ids = gen.tokenizer.encode(prompt, add_special_tokens=True)

    input_tensor = torch.tensor(input_ids, dtype=torch.int32)
    for _ in range(3):
        gen.decoder.reset()
        if use_prefill:
            gen.decoder.prefill_step(input_tensor)
        else:
            for tid in input_ids[:-1]:
                gen.decoder.decode_step(tid)
            gen.decoder.decode_step(input_ids[-1])

    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        gen.decoder.reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if use_prefill:
            gen.decoder.prefill_step(input_tensor)
        else:
            for tid in input_ids[:-1]:
                gen.decoder.decode_step(tid)
            gen.decoder.decode_step(input_ids[-1])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    mode = "Fused megakernel" if args.fused_prefill else ("cuBLAS prefill" if args.prefill else "Decode-only")
    print(f"{mode} ({len(input_ids)} tokens):")
    print(f"  Mean: {sum(times)/len(times):.2f} ms")
    print(f"  Min: {min(times):.2f} ms")
    print(f"  Max: {max(times):.2f} ms")
    print(f"  Throughput: {len(input_ids) * 1000 / (sum(times)/len(times)):.1f} tokens/s")


if __name__ == "__main__":
    main()

