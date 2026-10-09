import argparse
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.tests.dependencies import require
require('regex')
from tools.correction.contextual_bank import build
from tools.corpus.provenance import sha256_file


class ContextualBankTests(unittest.TestCase):
    def test_held_context_exclusion_lineage_and_clean_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lexical = root / 'lexical.csv'
            with lexical.open('w') as out:
                writer = csv.DictWriter(out, fieldnames=['Word', 'Error', 'ErrorType'])
                writer.writeheader()
                writer.writerow(dict(Word='বন্ধু', Error='বনধু', ErrorType='Typo Deletion'))
            rows = [dict(id=str(i), source=text, target=text, family='base', mode=0)
                    for i, text in enumerate(['আমার বন্ধু এসেছে।', 'তোমার বন্ধু এসেছে।'])]
            with gzip.open(root / 'validation.jsonl.gz', 'wt') as out:
                for row in rows:
                    out.write(json.dumps(row) + '\n')
            excluded = root / 'exclude.jsonl'
            excluded.write_text(json.dumps(rows[0]) + '\n')
            args = argparse.Namespace(data=root, lexical_csv=lexical, exclude=[excluded], seed=1,
                                      per_word=1, count=1, minimum=1, output=root / 'bank')
            receipt = dict(data_id='fixture', lexical_csv_sha256=sha256_file(lexical), byte_limit=768)
            with patch('tools.correction.contextual_bank.verify', return_value=receipt), \
                 patch('tools.correction.contextual_bank.word_partition', return_value='validation'):
                build(args)
            bank = [json.loads(s) for s in (args.output / 'bank.jsonl').read_text().splitlines()]
            self.assertEqual(len(bank), 2)
            self.assertEqual(bank[0]['source'], 'তোমার বনধু এসেছে।')
            self.assertEqual(bank[0]['target'], 'তোমার বন্ধু এসেছে।')
            self.assertEqual(bank[1]['source'], bank[1]['target'])
            manifest = json.loads((args.output / 'manifest.json').read_text())
            self.assertEqual(manifest['lineage'][0]['context_parent'], '1')
            self.assertFalse((args.output / '.building').exists())
