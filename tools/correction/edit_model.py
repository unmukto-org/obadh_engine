"""Parallel token editing with explicit KEEP and bounded insertion slots.

Unedited tokens are copied exactly, without recurrent regeneration. This is a
GECToR-inspired subword comparator, not a claim to reproduce that architecture.
Unsupported training alignments are counted and excluded, never relabeled.
"""

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import torch
from torch import nn
from torch.nn import functional as F

from tools.autosuggest.subword_lm import ModelConfig, make_model
from tools.autosuggest.subword_cache import attention_parts, finish_block
from tools.correction.edit_policy import admit_atomic


@dataclass(frozen=True)
class EditConfig:
    vocab_size: int = 16384
    width: int = 384
    layers: int = 8
    heads: int = 6
    ff_width: int = 1024
    sequence_length: int = 128
    max_append: int = 3


def align(source, target, max_append=3):
    """Source/target include BOS/EOS. Payload slot zero replaces a token."""
    if source[:1] != [1] or target[:1] != [1] or source[-1:] != [2] or target[-1:] != [2]:
        raise ValueError("alignment requires BOS/EOS")
    action, count = [0] * len(source), [0] * len(source)
    payload = [[-100] * (max_append + 1) for _ in source]
    for tag, i, j, k, l in SequenceMatcher(a=source, b=target, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i == 0 or j > len(source) - 1 or any(t < 4 for t in target[k:l]):
            return None
        if tag == "delete":
            action[i:j] = [1] * (j - i)
        elif tag == "insert":
            if l - k > max_append:
                return None
            count[i - 1] = l - k
            payload[i - 1][1:1 + l - k] = target[k:l]
        else:
            common = min(j - i, l - k)
            for offset in range(common):
                action[i + offset] = 2
                payload[i + offset][0] = target[k + offset]
            for pos in range(i + common, j):
                action[pos] = 1
            remaining = target[k + common:l]
            if len(remaining) > max_append:
                return None
            if remaining:
                count[j - 1] = len(remaining)
                payload[j - 1][1:1 + len(remaining)] = remaining
    return action, count, payload


def apply_edits(source, action, count, payload):
    result = []
    for token, op, n, values in zip(source, action, count, payload):
        if op == 0:
            result.append(token)
        elif op == 2:
            result.append(values[0])
        result.extend(values[1:1 + n])
    return result


class EditCorrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = make_model(ModelConfig(**{k: v for k, v in asdict(config).items() if k != "max_append"}))
        self.mode_embedding = nn.Embedding(2, config.width)
        self.action = nn.Linear(config.width, 3)
        self.append = nn.Linear(config.width, config.max_append + 1)
        self.slots = nn.ModuleList(nn.Sequential(nn.Linear(config.width, config.width, bias=False), nn.Tanh(), nn.LayerNorm(config.width, bias=False)) for _ in range(config.max_append + 1))

    def initialize(self, checkpoint):
        if checkpoint.get("kind") == "obadh-contextual-edit-corrector":
            if checkpoint["config"] != asdict(self.config):
                raise ValueError("edit initialization config mismatch")
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            return
        expected = {k: v for k, v in asdict(self.config).items() if k != "max_append"}
        supplied = {k: v for k, v in checkpoint["config"].items() if k != "decoder_layers"}
        if expected != supplied or checkpoint.get("kind") != "obadh-contextual-corrector":
            raise ValueError("edit encoder initialization mismatch")
        weights = checkpoint["state_dict"]
        self.encoder.load_state_dict({k[len("encoder."):]: v for k, v in weights.items() if k.startswith("encoder.")}, strict=True)
        self.mode_embedding.load_state_dict({"weight": weights["mode_embedding.weight"]})

    def forward(self, source, mode):
        valid = source != 0
        x = self.encoder.embedding(source) + self.mode_embedding(mode)[:, None]
        length = source.shape[1]
        for block in self.encoder.blocks:
            q, k, v = attention_parts(block, x, self.encoder.cos[:, :, :length], self.encoder.sin[:, :, :length], self.config.heads)
            x = finish_block(block, x, F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :]))
        memory = self.encoder.norm(x)
        return memory, self.action(memory), self.append(memory)

    def loss(self, batch):
        source, target = batch["source"], batch["target"]
        actions, counts, payloads, supported = [], [], [], []
        length = source.shape[1]
        for i, (s, t) in enumerate(zip(source.tolist(), target.tolist())):
            s, t = [n for n in s if n], [1] + [n for n in t if n]
            aligned = align(s, t, self.config.max_append)
            if aligned is None:
                continue
            action, count, payload = aligned
            pad = length - len(s)
            actions.append(action + [-100] * pad)
            counts.append(count + [-100] * pad)
            payloads.append(payload + [[-100] * (self.config.max_append + 1)] * pad)
            supported.append(i)
        if not supported:
            raise ValueError("no representable edit examples")
        device = next(self.parameters()).device
        indices = torch.tensor(supported)
        x, action_logits, count_logits = self(source[indices].to(device), batch["mode"][indices].to(device))
        action, count, payload = (torch.tensor(v, device=device) for v in (actions, counts, payloads))
        mask = action != -100
        losses = []
        for logits, labels in ((action_logits, action), (count_logits, count)):
            ce = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").reshape_as(labels)
            weight = torch.where(labels > 0, 4., 1.) * mask
            losses.append((ce * weight).sum() / weight.sum())
        token_loss = x.sum() * 0
        total = 0
        for slot, projection in enumerate(self.slots):
            selected = payload[:, :, slot] != -100
            if bool(selected.any()):
                logits = projection(x[selected]) @ self.encoder.embedding.weight.T
                logits = logits.float()
                logits[:, :4] = -1e9
                token_loss = token_loss + F.cross_entropy(logits, payload[:, :, slot][selected], reduction="sum")
                total += int(selected.sum())
        return 2 * sum(losses) + token_loss / max(1, total), len(supported), len(source) - len(supported)

    @torch.no_grad()
    def greedy(self, source, mode, maximum=None, threshold=.5, confidence="joint", edit_groups=None):
        maximum = maximum or self.config.sequence_length
        if not 0 < threshold <= 1 or confidence not in ("action", "joint") or not 1 <= maximum <= self.config.sequence_length:
            raise ValueError("invalid edit policy or capacity")
        x, action_logits, count_logits = self(source, mode)
        action_probability, action = action_logits.float().softmax(-1).max(-1)
        count_probability, count = count_logits.float().softmax(-1).max(-1)
        # Keep all proposed operations until admission, including uncertain
        # replacements whose confident deletions must be rejected with them.
        action = torch.where(source < 4, 0, action)
        count = torch.where((source == 0) | (source == 2), 0, count)
        payload = torch.full((*source.shape, self.config.max_append + 1), -100, device=source.device, dtype=torch.long)
        payload_probability = torch.ones_like(payload, dtype=torch.float32)
        for slot, projection in enumerate(self.slots):
            selected = action == 2 if slot == 0 else count >= slot
            if bool(selected.any()):
                logits = (projection(x[selected]) @ self.encoder.embedding.weight.T).float()
                logits[:, :4] = -1e9
                probability, tokens = logits.softmax(-1).max(-1)
                payload[:, :, slot][selected] = tokens
                payload_probability[:, :, slot][selected] = probability
        if confidence == "joint":
            # An edit detector alone does not establish that its proposed
            # replacement is reliable. Gate on both decisions; these scores
            # still require held-out calibration before automatic application.
            action_probability = torch.where(action == 2, action_probability * payload_probability[:, :, 0], action_probability)
            count_probability = count_probability * payload_probability[:, :, 1:].prod(-1)
        outputs = []
        group_rows = edit_groups.tolist() if edit_groups is not None else [[0] * source.shape[1] for _ in range(len(source))]
        for s, a, c, p, ac, ic, groups in zip(source.tolist(), action.tolist(), count.tolist(), payload.tolist(), action_probability.tolist(), count_probability.tolist(), group_rows):
            end = s.index(2) + 1
            a, c = admit_atomic(a, c, ac, ic, groups, threshold)
            output = apply_edits(s[:end], a[:end], c[:end], p[:end])[1:]
            outputs.append(output[:maximum])
        size = max(len(ids) for ids in outputs)
        ended = torch.tensor([2 in ids for ids in outputs], device=source.device)
        tokens = torch.tensor([ids + [2] * (size - len(ids)) for ids in outputs], device=source.device)
        # No calibrated sequence confidence is available for parallel edits.
        return tokens, None, ended
