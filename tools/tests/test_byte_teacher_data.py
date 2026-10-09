import argparse
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.tests.dependencies import require
require('regex', 'numpy')
from tools.correction.byte_teacher_data import prepare
from tools.corpus.provenance import digest_json, sha256_file


class ByteTeacherDataTests(unittest.TestCase):
    def test_heldout_and_conflicting_labels_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, teacher = root / 'parent', root / 'teacher'
            parent.mkdir();teacher.mkdir()
            clean = dict(id='existing', source='সে যাবে।', target='সে যাবে।', mode=0, kind='base/identity', family='base')
            held = dict(clean, source='আমি যাব।', target='আমি যাব।')
            for split, rows in [('train', [clean]), ('validation', [held])]:
                with gzip.open(parent / (split + '.jsonl.gz'), 'wt') as out:
                    for row in rows: out.write(json.dumps(row) + '\n')
            (teacher / 'rows.jsonl').write_text('')
            contract = dict(count=1, data_id='original', bank_sha256=digest_json([]))
            (teacher / 'contract.json').write_text(json.dumps(contract))
            (teacher / 'report.json').write_text(json.dumps(dict(completed=True, contract=contract, rows_sha256=sha256_file(teacher / 'rows.jsonl'))))
            receipt = dict(kind='obadh-byte-correction-data', data_id='context', parent_data_id='byte-base', byte_limit=768, prefixes={'0':'sentence: '}, counts={'validation':{'base/identity':1}})
            proposed = [dict(id=str(i), parent_id='p'+str(i), source_text=s, target=t, mode=0, kind='teacher_grammar/agreement') for i,(s,t) in enumerate([
                ('সে যাবে।','সে যায়।'), ('আমি যায়।','আমি যাব।'), ('আমরা করে।','আমরা করি।')])]
            args = argparse.Namespace(parent=parent, byte_base=root/'byte-base', teacher=teacher, source_data=root/'original', exclude=[], output=root/'output', minimum_errors=1)
            with patch('tools.correction.byte_teacher_data.verify', side_effect=[receipt, dict(data_id='byte-base',base_data_id='original')]), \
                 patch('tools.correction.byte_teacher_data.select_bank', return_value=(dict(data_id='original'),[])), \
                 patch('tools.correction.byte_teacher_data.admitted', return_value=proposed):
                prepare(args)
            with gzip.open(args.output/'train.jsonl.gz','rt') as handle: result=[json.loads(s) for s in handle]
            self.assertEqual(len(result),2)
            self.assertEqual(result[1]['target'],'আমরা করি।')
            self.assertEqual(sha256_file(args.output/'validation.jsonl.gz'),sha256_file(parent/'validation.jsonl.gz'))
            self.assertFalse((args.output/'.building').exists())
