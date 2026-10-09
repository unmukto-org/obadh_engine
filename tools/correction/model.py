"""Compact bidirectional Transformer encoder and recurrent copy decoder.

The decoder returns vocabulary logits, source attention and a generation gate.
This keeps the host-side copy mixture portable and avoids a vocabulary-sized
copy tensor in training. Sentence/prefix mode is explicit. No beam search or
unbounded generation is required by this architecture.
"""

from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F

from tools.autosuggest.subword_lm import ModelConfig, make_model
from tools.autosuggest.subword_cache import attention_parts, finish_block


@dataclass(frozen=True)
class CorrectionConfig:
    vocab_size: int = 16384
    width: int = 384
    layers: int = 8
    heads: int = 6
    ff_width: int = 1024
    sequence_length: int = 128
    decoder_layers: int = 2


class Corrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_config = {k: v for k, v in asdict(config).items() if k != "decoder_layers"}
        self.encoder = make_model(ModelConfig(**encoder_config))
        self.mode_embedding = nn.Embedding(2, config.width)
        self.decoder = nn.LSTM(config.width, config.width, config.decoder_layers, batch_first=True)
        self.initial = nn.Linear(config.width, config.width, bias=False)
        self.query = nn.Linear(config.width, config.width, bias=False)
        self.combine = nn.Linear(2 * config.width, config.width, bias=False)
        self.generation_gate = nn.Linear(2 * config.width, 1)
        self.output_norm = nn.LayerNorm(config.width, bias=False)
        self.register_buffer("forbidden", torch.isin(torch.arange(config.vocab_size), torch.tensor([0, 1, 3])), persistent=False)
        nn.init.normal_(self.mode_embedding.weight, std=0.02)

    def initialize_encoder(self, checkpoint):
        expected = {k: v for k, v in asdict(self.config).items() if k != "decoder_layers"}
        if checkpoint["config"] != expected:
            raise ValueError("encoder initialization config mismatch")
        self.encoder.load_state_dict(checkpoint["state_dict"], strict=True)

    def encode(self, source, mode):
        valid = source != 0
        x = self.encoder.embedding(source) + self.mode_embedding(mode)[:, None, :]
        length = source.shape[1]
        allowed = valid[:, None, None, :]
        for block in self.encoder.blocks:
            q, k, v = attention_parts(block, x, self.encoder.cos[:, :, :length], self.encoder.sin[:, :, :length], self.config.heads)
            x = finish_block(block, x, F.scaled_dot_product_attention(q, k, v, attn_mask=allowed))
        memory = self.encoder.norm(x)
        pooled = (memory * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        hidden = self.initial(pooled).tanh()[None].repeat(self.config.decoder_layers, 1, 1).float()
        return memory, valid & (source != 1) & (source != 3), (hidden, torch.zeros_like(hidden))

    def decode(self, memory, valid, previous, state):
        # cuDNN recurrent BF16 support varies by runtime; keep this small
        # recurrent component in FP32 while the encoder/projection use BF16.
        with torch.autocast(device_type=memory.device.type, enabled=False):
            decoded, next_state = self.decoder(self.encoder.embedding(previous).float(), state)
        query = self.query(decoded)
        scores = query @ memory.transpose(1, 2) / self.config.width**0.5
        attention = scores.float().masked_fill(~valid[:, None], float("-inf")).softmax(-1)
        context = attention.to(memory.dtype) @ memory
        joined = torch.cat((decoded.to(context.dtype), context), dim=-1)
        hidden = self.output_norm(self.combine(joined).tanh())
        logits = hidden @ self.encoder.embedding.weight.T
        logits = logits.float().masked_fill(self.forbidden, -1e9)
        gate = self.generation_gate(joined).float()
        return logits, attention, gate, next_state

    def forward(self, source, mode, previous):
        memory, valid, state = self.encode(source, mode)
        return self.decode(memory, valid, previous, state)[:3]

    @staticmethod
    def target_log_prob(logits, attention, gate, source, target):
        generated = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
        copied = (attention * (source[:, None, :] == target[:, :, None])).sum(-1)
        copy_log = copied.clamp_min(1e-30).log()
        return torch.logaddexp(F.logsigmoid(gate.squeeze(-1)) + generated, F.logsigmoid(-gate.squeeze(-1)) + copy_log)

    @torch.no_grad()
    def greedy(self, source, mode, maximum=None):
        maximum = maximum or self.config.sequence_length
        if not 1 <= maximum <= self.config.sequence_length:
            raise ValueError("generation capacity exceeded")
        memory, valid, state = self.encode(source, mode)
        previous = torch.ones((len(source), 1), dtype=torch.long, device=source.device)
        ended = torch.zeros(len(source), dtype=torch.bool, device=source.device)
        tokens, probabilities = [], []
        for _ in range(maximum):
            logits, attention, gate, state = self.decode(memory, valid, previous, state)
            mixture = logits[:, 0].softmax(-1) * gate[:, 0].sigmoid()
            mixture.scatter_add_(1, source, attention[:, 0] * (-gate[:, 0]).sigmoid())
            mixture[:, self.forbidden] = 0
            probability, token = mixture.max(-1)
            token = torch.where(ended, torch.full_like(token, 2), token)
            tokens.append(token)
            probabilities.append(probability)
            ended |= token == 2
            previous = token[:, None]
            if bool(ended.all()):
                break
        return torch.stack(tokens, 1), torch.stack(probabilities, 1), ended
