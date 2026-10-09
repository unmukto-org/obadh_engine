"""Atomic edit admission: never apply half of a subword replacement.

Edits touching the same source word, or adjacent edited tokens, share one
confidence decision. Rejected replacements cannot leave their deletions behind.
"""

import regex


def token_groups(text, offsets):
    regions = [(m.start(), m.end()) for m in regex.finditer(r"[\p{L}\p{M}\p{N}\u200c\u200d]+|[^\s]", text)]
    result = []
    for i, (start, end) in enumerate(offsets):
        overlap = [j for j, (a, b) in enumerate(regions) if start < b and a < end]
        result.append(overlap[0] if overlap else len(regions) + i)
    return result


def admit_atomic(action, count, action_confidence, insertion_confidence, groups, threshold):
    n = len(action)
    if any(len(x) != n for x in (count, action_confidence, insertion_confidence, groups)) or not 0 < threshold <= 1:
        raise ValueError("invalid atomic edit inputs")
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    edited = {i for i, (a, c) in enumerate(zip(action, count)) if a != 0 or c > 0}
    words = {}
    for i in sorted(edited):
        if groups[i] in words:
            union(i, words[groups[i]])
        words[groups[i]] = i
        if i - 1 in edited:
            union(i, i - 1)
    rejected = set()
    for i in edited:
        if (action[i] and not action_confidence[i] >= threshold) or (count[i] and not insertion_confidence[i] >= threshold):
            rejected.add(find(i))
    action, count = list(action), list(count)
    for i in edited:
        if find(i) in rejected:
            action[i], count[i] = 0, 0
    return action, count
