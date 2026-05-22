"""
KV‑Cache manager for paged attention (decode_v4).

The manager owns:
* `k_cache` / `v_cache` – contiguous tensors of shape
  [batch, num_kv_heads, max_cache_len, head_dim] (bfloat16).
* `block_table` – int32 tensor [batch, max_blocks] where each entry is a
  **logical** block id (0 … max_blocks‑1).  The CUDA kernel will combine it
  with the batch index internally.
* `seq_lens` – int32 tensor [batch] storing the current logical length
  of each sequence.

Both pre‑fill and decode helpers keep `seq_lens` up‑to‑date.
"""

from __future__ import annotations
from typing import Union

import torch


class KVCache:
    """
    KV‑cache for a single transformer layer.

    Parameters
    ----------
    batch_size : int
        Number of sequences processed in parallel.
    num_kv_heads : int
        Number of KV heads (usually ``num_key_value_heads`` from the model config).
    head_dim : int
        Dimension of each head.
    max_cache_len : int
        Maximum number of tokens that can be stored in the cache.
    block_size : int, optional
        Size of a KV block (default 16, the same as the kernel).
    device : torch.device, optional
        Device where the tensors live (default ``torch.device('cuda')``).
    dtype : torch.dtype, optional
        Data type for the KV tensors (default ``torch.bfloat16``).
    """

    def __init__(
        self,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        block_size: int = 16,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # Pad the cache length to a multiple of BLOCK_SIZE – the kernel assumes
        # block‑major layout.
        self.padded_cache_len = (
            (max_cache_len + block_size - 1) // block_size
        ) * block_size
        self.max_blocks = self.padded_cache_len // block_size

        # ------------------------------------------------------------------
        # Allocate KV tensors (padded)
        # ------------------------------------------------------------------
        self.k_cache = torch.empty(
            batch_size,
            num_kv_heads,
            self.padded_cache_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty_like(self.k_cache)

        # ------------------------------------------------------------------
        # Block table – logical block ids (0 … max_blocks‑1) for each batch.
        # The CUDA kernel will add the batch offset internally.
        # ------------------------------------------------------------------
        self.block_table = torch.arange(
            self.max_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0).repeat(batch_size, 1)  # [batch, max_blocks]

        # ------------------------------------------------------------------
        # Sequence lengths (int32) – initially zero.
        # ------------------------------------------------------------------
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.current_len = 0

    # ------------------------------------------------------------------
    # Prefill – write the whole KV for the initial prompt.
    # ------------------------------------------------------------------
    def write_kv_prefill(self, k: torch.Tensor, v: torch.Tensor, seq_len: int) -> None:
        """
        Write the KV tensors for a prefill.

        Parameters
        ----------
        k, v : torch.Tensor
            Shape ``[batch, num_kv_heads, seq_len, head_dim]``.
        seq_len : int
            Logical length of the sequence (must be ≤ ``max_cache_len``).
        """
        assert k.shape == (self.batch_size, self.num_kv_heads, seq_len, self.head_dim)
        assert v.shape == k.shape
        self.k_cache[:, :, :seq_len, :] = k
        self.v_cache[:, :, :seq_len, :] = v
        self.seq_lens[:] = seq_len

    # ------------------------------------------------------------------
    # Decode – append a single token.
    # ------------------------------------------------------------------
    def write_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        position: Union[int, torch.Tensor],
    ) -> None:
        """
        Append a single token to the KV cache.

        Parameters
        ----------
        k, v : torch.Tensor
            Shape ``[batch, num_kv_heads, 1, head_dim]``.
        position : int or torch.Tensor
            Logical token index (0‑based).  If an ``int`` is given it is broadcast
            to all batches; otherwise a ``[batch]`` tensor is expected.
        """
        if isinstance(position, int):
            position = torch.full(
                (self.batch_size,),
                position,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            position = position.to(self.device)

        # Remove the singleton sequence dimension.
        k_val = k.squeeze(2)  # [batch, num_kv_heads, head_dim]
        v_val = v.squeeze(2)

        # Index tensors.
        batch_idx = torch.arange(self.batch_size, device=self.device)[:, None].expand(
            -1, self.num_kv_heads
        )
        head_idx = torch.arange(self.num_kv_heads, device=self.device)[None, :].expand(
            self.batch_size, -1
        )
        pos_idx = position[:, None].expand(-1, self.num_kv_heads)

        # Write into the cache.
        self.k_cache[batch_idx, head_idx, pos_idx, :] = k_val
        self.v_cache[batch_idx, head_idx, pos_idx, :] = v_val

        # Update sequence lengths (position is 0‑based → length = position + 1).
        new_len = position + 1
        self.seq_lens = torch.maximum(self.seq_lens, new_len)

    # ------------------------------------------------------------------
    # Utility getters (used by the CUDA kernel).
    # ------------------------------------------------------------------
    def get_kv(self):
        """Return the K and V tensors."""
        return self.k_cache, self.v_cache

    def get_block_table(self):
        """Return the block‑table tensor (int32)."""
        return self.block_table

    def get_seq_lens(self):
        """Return the sequence‑length tensor (int32)."""
        return self.seq_lens

    # ------------------------------------------------------------------
    # Reset (useful for repeated generations).
    # ------------------------------------------------------------------
    def reset(self):
        self.current_len = 0
        self.block_table.zero_()
        
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_lens.zero_()