from tools.tests.dependencies import require

require('regex')

import json
import gzip
from pathlib import Path
import tempfile
import unittest
from tools.correction.teacher_roundtrip import candidate, critic_accepts, select_bank
from tools.corpus.provenance import digest_json, sha256_file


class RoundtripTests(unittest.TestCase):
    def test_continuation_selects_new_distinct_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(id=str(i), split='train', kind='identity', clean_sha256=str(i)) for i in range(4)]
            with gzip.open(root / 'train.jsonl.gz', 'wt') as out:
                for row in [rows[0], rows[0], *rows[1:]]:
                    out.write(json.dumps(row) + '\n')
            receipt = dict(files=[dict(path='train.jsonl.gz', sha256=sha256_file(root / 'train.jsonl.gz'))])
            receipt['data_id'] = digest_json(receipt)
            (root / 'manifest.json').write_text(json.dumps(receipt))
            _, first = select_bank(root, 2)
            _, next_batch = select_bank(root, 2, 2)
            self.assertEqual([r['parent_id'] for r in first], ['0', '1'])
            self.assertEqual([r['parent_id'] for r in next_batch], ['2', '3'])
            with self.assertRaises(ValueError):
                select_bank(root, 3, 2)

    def test_valid_emphasis_and_negation_changes_cannot_be_training_errors(self):
        self.assertEqual(candidate(json.dumps(dict(source="আমি এখানেই থাকিই।")), "আমি এখানেই থাকি।", "verb_form")[1], "optional_emphasis")
        self.assertEqual(candidate(json.dumps(dict(source="ওটা নতুন নয়া।")), "ওটা নতুন নয়।", "verb_form")[1], "negation_changed")
        valid = dict(source_has_clear_error=True, target_is_acceptable=True, intent_preserved=True)
        self.assertTrue(critic_accepts(json.dumps(valid)))
        self.assertTrue(critic_accepts("```json\n" + json.dumps(valid) + "\n```"))
        self.assertFalse(critic_accepts("Explanation\n```json\n" + json.dumps(valid) + "\n```"))
        self.assertFalse(critic_accepts('{"source_has_clear_error":false,"source_has_clear_error":true,"target_is_acceptable":true,"intent_preserved":true}'))
        valid["source_has_clear_error"] = False
        self.assertFalse(critic_accepts(json.dumps(valid)))
        valid["source_has_clear_error"] = "true"
        self.assertFalse(critic_accepts(json.dumps(valid)))
    def test_minimal_grammar_error_and_punctuation_remain_distinct(self):
        source = "আমি সকালে স্কুলে যায়।"
        self.assertEqual(candidate(json.dumps(dict(source=source)), "আমি সকালে স্কুলে যাই।", "agreement"), (source, None))
        self.assertEqual(candidate(json.dumps(dict(source="তুমি কোথায় যাচ্ছ।")), "তুমি কোথায় যাচ্ছ?", "punctuation"), ("তুমি কোথায় যাচ্ছ।", None))
        self.assertEqual(candidate(json.dumps(dict(source="তুমি কোথায় যাচ্ছ।")), "তুমি কোথায় যাচ্ছ?", "agreement")[1], "terminal_punctuation_changed")

    def test_skips_rewrites_protected_changes_and_unchanged_are_rejected(self):
        self.assertIsNone(candidate('{"source":null}', "আমি যাই।", "agreement")[0])
        self.assertIsNone(candidate('{"source":"আমি যাই।"}', "আমি যাই।", "agreement")[0])
        self.assertEqual(candidate(json.dumps(dict(source="ID AB124, আমি যাই।")), "ID AB123, আমি যাই।", "agreement")[1], "protected_span_changed")
        self.assertEqual(candidate(json.dumps(dict(source="তুমি কেন যাচ্ছ?")), "তুমি কোথায় যাচ্ছ?", "punctuation")[1], "punctuation_task_changed_words")
        self.assertIsNone(candidate('```json\n{}\n```', "আমি যাই।", "agreement")[0])


if __name__ == "__main__":
    unittest.main()
