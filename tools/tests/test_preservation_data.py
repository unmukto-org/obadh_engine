import argparse
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from tools.tests.dependencies import require
require('regex')
from tools.correction.preservation_data import bank, prepare, difficult_keys
from tools.corpus.provenance import digest_json,sha256_file
from tools.correction.byte_data import read_rows


class PreservationTests(unittest.TestCase):
    def test_only_unwanted_safe_edits_are_mined(self):
        def row(s,t,a,f=None):return dict(source=s,target=t,answer=a,fallback_reason=f,mode=0)
        rows=[row('a','a','b'),row('c','c','c'),row('d','e','e'),row('f','f',None),row('g','g','h','protected')]
        self.assertEqual(difficult_keys(rows),{('a',0)})

    def test_bank_exclusions_relabel_only_and_immutable_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'parent';parent.mkdir()
            train=[dict(id=str(i),source=str(i),target=str(i),mode=0,kind='base/identity',family='base') for i in range(6)]
            valid=[dict(train[0],source='held',target='held')]
            for split,rows in [('train',train),('validation',valid)]:
                with gzip.open(parent/(split+'.jsonl.gz'),'wt') as f:
                    for r in rows:f.write(json.dumps(r)+'\n')
            receipt=dict(kind='obadh-byte-correction-data',byte_limit=768,prefixes={'0':'sentence: '},counts={'validation':{'base/identity':1}},files=[dict(path=p.name,sha256=sha256_file(p)) for p in sorted(parent.glob('*.gz'))])
            receipt['data_id']=digest_json(receipt);(parent/'manifest.json').write_text(json.dumps(receipt))
            exclude=root/'exclude.jsonl';exclude.write_text(json.dumps(dict(source='0',target='1'))+'\n')
            b=root/'bank';bank(argparse.Namespace(parent=parent,exclude=[exclude],count=4,seed=42,output=b))
            selected=list(map(json.loads,(b/'bank.jsonl').read_text().splitlines()))
            self.assertEqual({r['source'] for r in selected},{'2','3','4','5'})
            predictions=root/'predictions';predictions.mkdir()
            rows=[dict(r,answer=r['source']+'x',fallback_reason=None) for r in selected]
            (predictions/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report=dict(completed=True,contract={'bank_sha256':digest_json(selected)},rows_sha256=sha256_file(predictions/'rows.jsonl'))
            (predictions/'report.json').write_text(json.dumps(report))
            output=root/'data';args=argparse.Namespace(parent=parent,bank=b,predictions=predictions,minimum=2,exclude=[exclude],output=output)
            prepare(args)
            self.assertEqual((parent/'validation.jsonl.gz').read_bytes(),(output/'validation.jsonl.gz').read_bytes())
            after=list(read_rows(output/'train.jsonl.gz'))
            self.assertEqual([(r['source'],r['target'],r['mode']) for r in train],[(r['source'],r['target'],r['mode']) for r in after])
            self.assertEqual(sum(r['kind']=='base/mined_preservation_0' for r in after),4)
            exclude.write_text(json.dumps(dict(source='2',target='2'))+'\n');args.output=root/'rejected'
            with self.assertRaisesRegex(ValueError,'overlap'):prepare(args)
            self.assertFalse(args.output.exists())
            (predictions/'.building').touch()
            with self.assertRaisesRegex(ValueError,'incomplete'):prepare(args)
