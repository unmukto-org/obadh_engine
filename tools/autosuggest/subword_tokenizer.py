"""Versioned, lossless byte BPE pre-tokenization for Indic scripts."""

UNICODE_MARKS_PATTERN = (
    r" ?[\p{L}\p{M}\u200c\u200d]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}\u200c\u200d]+|\s+"
)


def make_tokenizer(profile):
    from tokenizers import (
        Tokenizer,
        Regex,
        models,
        normalizers,
        pre_tokenizers,
        decoders,
    )

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    if profile == "unicode-marks-v1":
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(UNICODE_MARKS_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    elif profile == "gpt2-regex-v0":
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    else:
        raise ValueError("unsupported tokenizer contract")
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer
