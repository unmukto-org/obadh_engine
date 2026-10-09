"""Score externally retrieved corrections with one shared contextual encoder.

The unchanged input competes with every candidate. These factorized model scores
are ranking diagnostics, not calibrated probabilities of safe autocorrection.
Retrieval is backend-independent; a production host can supply the engine's FST
candidates without allowing unrestricted neural spelling generation.
"""

import unicodedata
import torch
from tools.correction.edit_model import align


class EditCandidateScorer:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        tokenizer.encode_special_tokens = True

    @torch.no_grad()
    def rank(self, source, candidates, mode=0):
        source = unicodedata.normalize("NFC", source)
        candidates = list(dict.fromkeys([source] + [unicodedata.normalize("NFC", s) for s in candidates]))
        if not 1 <= len(candidates) <= 33 or mode not in (0, 1):
            raise ValueError("invalid candidate scoring budget")
        ids = [1] + self.tokenizer.encode(source, add_special_tokens=False).ids + [2]
        if len(ids) > self.model.config.sequence_length or any(t < 4 for t in ids[1:-1]):
            raise ValueError("invalid candidate source capacity or control tokens")
        memory, action, count = self.model(torch.tensor([ids], device=self.device), torch.tensor([mode], device=self.device))
        action, count = action[0].float().log_softmax(-1), count[0].float().log_softmax(-1)
        cache = {}
        result = []
        for candidate in candidates:
            target = [1] + self.tokenizer.encode(candidate, add_special_tokens=False).ids + [2]
            labels = align(ids, target, self.model.config.max_append) if len(target) <= self.model.config.sequence_length and not any(t < 4 for t in target[1:-1]) else None
            if labels is None:
                continue
            actions, counts, payload = labels
            score = sum(float(action[i, a]) + float(count[i, c]) for i, (a, c) in enumerate(zip(actions, counts)))
            for i, values in enumerate(payload):
                for slot, token in enumerate(values):
                    if token == -100:
                        continue
                    key = (i, slot)
                    if key not in cache:
                        logits = (self.model.slots[slot](memory[0, i]) @ self.model.encoder.embedding.weight.T).float()
                        logits[:4] = -1e9
                        cache[key] = logits.log_softmax(-1)
                    score += float(cache[key][token])
            result.append(dict(text=candidate, score=score, unchanged=candidate == source))
        result.sort(key=lambda r: (-r["score"], not r["unchanged"], r["text"]))
        return result
