import unittest
from tools.autosuggest.decode_subword_words import byte_alphabet, lexical_word


class SubwordDecoderTests(unittest.TestCase):
    def test_byte_contract_and_strict_utf8_output(self):
        alphabet = byte_alphabet()
        self.assertEqual(set(alphabet.values()), set(range(256)))
        self.assertEqual(alphabet["Ġ"], 32)
        self.assertEqual(lexical_word("কোথায়".encode()), "কোথায়")
        self.assertEqual(lexical_word("র\u200c্যাব".encode()), "র\u200c্যাব")
        self.assertEqual(lexical_word(b"meeting"), "meeting")
        for raw in (b"", b"\xe0\xa6", b"one two", b"!", b"\x00abc", "ি".encode()):
            self.assertIsNone(lexical_word(raw))


if __name__ == "__main__":
    unittest.main()
