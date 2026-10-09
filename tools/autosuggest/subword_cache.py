"""Stateless, bounded KV-cache graphs shared by ONNX and Core ML exports.

The host owns all cache buffers. A cache belongs to one exact token prefix and
model version; cursor edits, resets and model changes invalidate it. Never wrap
positions modulo capacity: prefill a truncated context when capacity is reached.
"""

import torch
from torch import nn
from torch.nn import functional as F


def attention_parts(block, x, cos, sin, heads):
    batch, length, width = x.shape
    q, k, v = (
        block.qkv(block.attn_norm(x))
        .view(batch, length, 3, heads, width // heads)
        .unbind(2)
    )

    def rotate(value):
        value = value.transpose(1, 2)
        left, right = value.chunk(2, dim=-1)
        return value * cos.to(value.dtype) + torch.cat((-right, left), dim=-1) * sin.to(
            value.dtype
        )

    return rotate(q), rotate(k), v.transpose(1, 2)


def finish_block(block, x, attention):
    batch, length, width = x.shape
    x = x + block.attn_out(attention.transpose(1, 2).reshape(batch, length, width))
    gate, up = block.gate_up(block.ff_norm(x)).chunk(2, dim=-1)
    return x + block.down(F.silu(gate) * up)


class PrefillScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, last_index):
        x = self.model.embedding(tokens.long())
        length = tokens.shape[1]
        keys = []
        values = []
        for block in self.model.blocks:
            q, k, v = attention_parts(
                block,
                x,
                self.model.cos[:, :, :length],
                self.model.sin[:, :, :length],
                self.model.config.heads,
            )
            x = finish_block(
                block, x, F.scaled_dot_product_attention(q, k, v, is_causal=True)
            )
            keys.append(k)
            values.append(v)
        logits = (
            self.model.norm(x[:, last_index[0].long()]) @ self.model.embedding.weight.T
        )
        return logits, torch.stack(keys), torch.stack(values)


class CachedStepScorer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "slots", torch.arange(model.config.sequence_length), persistent=False
        )

    def forward(self, token, position, key_cache, value_cache):
        x = self.model.embedding(token.long())
        keys = []
        values = []
        cos = self.model.cos.index_select(2, position.long())
        sin = self.model.sin.index_select(2, position.long())
        replace = (self.slots == position[0]).view(1, 1, -1, 1)
        allowed = (self.slots <= position[0]).view(1, 1, 1, -1)
        for index, block in enumerate(self.model.blocks):
            q, k, v = attention_parts(block, x, cos, sin, self.model.config.heads)
            k = torch.where(replace, k, key_cache[index].to(k.dtype))
            v = torch.where(replace, v, value_cache[index].to(v.dtype))
            attention = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=False
            )
            x = finish_block(block, x, attention)
            keys.append(k)
            values.append(v)
        logits = self.model.norm(x[:, 0]) @ self.model.embedding.weight.T
        return logits, torch.stack(keys), torch.stack(values)


def cache_shape(config, batch=1):
    return (
        config.layers,
        batch,
        config.heads,
        config.sequence_length,
        config.width // config.heads,
    )
