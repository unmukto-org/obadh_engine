from pathlib import Path
import tempfile
import unittest

from tools.corpus.provenance import sha256_file


class ExportFixtureTests(unittest.TestCase):
    def test_short_documents_never_supply_padding_or_adjacent_context(self):
        from tools.tests.dependencies import require
        require("numpy")
        import numpy as np
        from tools.autosuggest.export_fixtures import ValidationSequences

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.asarray([[1, 10, 2, 0, 0], [1, 20, 21, 22, 2]], dtype="<u2").tofile(
                root / "validation.u16"
            )
            np.asarray([2, 4], dtype="<u2").tofile(root / "validation.targets.u16")
            counts = root / "validation.targets.u16"
            receipt = dict(
                kind="obadh-document-training-data",
                sequence_length=4,
                partitions={
                    "validation": {"sha256": sha256_file(root / "validation.u16")}
                },
                files=[
                    dict(
                        path=counts.name,
                        sha256=sha256_file(counts),
                        bytes=counts.stat().st_size,
                    )
                ],
            )
            sampler = ValidationSequences(root, receipt, 4)
            rng = np.random.default_rng(17)
            for continuation in (1, 2, 3):
                for _ in range(100):
                    seq, length = sampler.sample(
                        rng, maximum_prefix=4, continuation=continuation
                    )
                    self.assertEqual(len(seq), length + continuation)
                    self.assertNotIn(0, seq)
                    self.assertFalse(10 in seq and any(x in seq for x in (20, 21, 22)))
                    if 2 in seq:
                        self.assertEqual(int(seq[-1]), 2)
            with self.assertRaises(ValueError):
                sampler.sample(rng, maximum_prefix=4, continuation=5)
            counts.write_bytes(b"\x04\x00\x04\x00")
            with self.assertRaisesRegex(ValueError, "integrity"):
                ValidationSequences(root, receipt, 4)


if __name__ == "__main__":
    unittest.main()
