import unittest
from tools.tests.dependencies import require
require('numpy')
from tools.correction.candidate_selection import grouped,vector,winner

class CandidateSelectionTests(unittest.TestCase):
    def row(self,**kw):
        return dict(dict(case_id='one',source='ক',target='খ',mode=0,kind='fixture',features=[1.],raw_answer='খ',fallback_reason=None,beam_rank=1,candidate_count=3,original_beam_top1=True),**kw)
    def test_group_alignment_and_invalid_candidate(self):
        with self.assertRaises(ValueError):grouped([self.row(),self.row(target='গ')])
        self.assertIsNone(vector(self.row(fallback_reason='protected')))
        self.assertIsNone(vector(self.row(raw_answer='ক')))
        with self.assertRaises(ValueError):vector(self.row(beam_rank=None))
    def test_label_independent_ranking_and_copy_fallback(self):
        model=dict(mean=[0.]*4,scale=[1.]*4,beta=[0.,1.,0.,0.,0.])
        rows=[self.row(features=[-1.]),self.row(raw_answer='গ',features=[2.],beam_rank=2)]
        self.assertEqual(winner(model,rows)['answer'],'গ')
        for r in rows:r['target']='unseen reference'
        self.assertEqual(winner(model,rows)['answer'],'গ')
        self.assertEqual(winner(model,[self.row(features=None)])['answer'],'ক')
    def test_fit_and_evaluate_freezes_policy_and_reconstructs_cases(self):
        require('torch','tokenizers','regex')
        import argparse,json,tempfile
        from pathlib import Path
        from tools.correction.acceptance import FEATURES
        from tools.correction.candidate_selection import fit,evaluate
        from tools.corpus.provenance import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def bank(name):
                p=root/name;p.mkdir();rows=[]
                for i in range(64):
                    source=name+str(i);correct=i%2==0
                    for rank in (1,2):
                        candidate=source+str(rank)
                        rows.append(dict(id=source+':'+str(rank),case_id=source,source=source,target=source+'1' if correct else source,
                            raw_answer=candidate,answer=candidate,mode=0,kind='fixture',fallback_reason=None,capacity_exceeded=False,
                            features=[3. if correct and rank==1 else -3.]+[0.]*(len(FEATURES)-1),beam_rank=rank,candidate_count=3,original_beam_top1=rank==1))
                (p/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (p/'report.json').write_text(json.dumps(dict(completed=True,features=FEATURES,checkpoint_sha256='generator',
                    generator_contract={'case_bank_sha256':name},rows_sha256=sha256_file(p/'rows.jsonl'))))
                return p
            train,cal,held=[bank(n) for n in ('train','cal','held')]
            model=root/'model';fit(argparse.Namespace(train=train,calibration=cal,output=model))
            snapshot=(model/'model.json').read_bytes();output=root/'evaluation'
            evaluate(argparse.Namespace(model=model,scored=held,output=output))
            self.assertEqual(snapshot,(model/'model.json').read_bytes())
            rows=list(map(json.loads,(output/'rows.jsonl').read_text().splitlines()))
            self.assertEqual(len(rows),64);self.assertTrue(all(r['answer']==r['target'] for r in rows))
            self.assertEqual(json.loads((output/'report.json').read_text())['contract']['bank_sha256'],'held')
            with self.assertRaisesRegex(ValueError,'overlap'):evaluate(argparse.Namespace(model=model,scored=train,output=root/'bad'))
