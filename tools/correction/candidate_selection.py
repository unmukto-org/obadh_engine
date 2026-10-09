"""Fit a reference-exactness selector over generated candidate sets.

Candidate features use input/proposed text and model likelihoods only. Fit on
training cases, choose one threshold on calibration, then freeze for evaluation.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
from tools.corpus.provenance import digest_json,sha256_file,write_json
from tools.correction.acceptance import load_scored,fit_logistic,probabilities
from tools.correction.acceptance_union import text_hash,select_threshold

EXTRA_FEATURES=['beam_rank','candidate_count','original_beam_top1']


def flatten(args):
    report=json.loads((args.candidates/'report.json').read_text())
    path=args.candidates/'rows.jsonl'
    if (args.candidates/'.building').exists() or not report['completed'] or sha256_file(path)!=report['rows_sha256']:
        raise ValueError('Incomplete candidate bank')
    cases=list(map(json.loads,path.read_text().splitlines()))
    if len({r['id'] for r in cases})!=len(cases):raise ValueError('Duplicate case identity')
    rows=[]
    for case in cases:
        for index,candidate in enumerate(case['candidates']):
            rows.append(dict(id=case['id']+':candidate:'+str(index),case_id=case['id'],kind=case['kind'],source=case['source'],
                target=case['target'],mode=case['mode'],raw_answer=candidate['raw_answer'],answer=candidate['answer'],
                fallback_reason=candidate['fallback_reason'],capacity_exceeded=False,
                beam_rank=candidate['rank'],candidate_count=len(case['candidates']),original_beam_top1=index==0))
    contract=dict(report['contract'],case_bank_sha256=report['contract']['bank_sha256'],
        bank_sha256=digest_json([{k:r[k] for k in ('id','kind','source','target','mode')} for r in rows]),
        candidate_report_sha256=sha256_file(args.candidates/'report.json'))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'.building').touch()
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    write_json(args.output/'report.json',dict(completed=True,contract=contract,rows_sha256=sha256_file(args.output/'rows.jsonl')))
    (args.output/'.building').unlink()


def grouped(rows):
    groups=OrderedDict()
    for row in rows:
        group=groups.setdefault(row['case_id'],[])
        if group and any(group[0][k]!=row[k] for k in ('source','target','mode','kind')):raise ValueError('Inconsistent candidate labels')
        group.append(row)
    return list(groups.values())


def vector(row):
    if row['features'] is None or row['fallback_reason'] or row['raw_answer']==row['source']:return None
    rank=row['beam_rank']
    if type(rank) is not int or rank<1 or not 1<=row['candidate_count']<=9:raise ValueError('Invalid beam metadata')
    return row['features']+[rank,row['candidate_count'],int(row['original_beam_top1'])]


def winner(model,group):
    eligible=[r for r in group if vector(r) is not None]
    if not eligible:return dict(answer=group[0]['source'],probability=None)
    scores=probabilities(model,[vector(r) for r in eligible]);i=int(np.argmax(scores))
    return dict(answer=eligible[i]['raw_answer'],probability=float(scores[i]))


def identity(report):
    return dict(checkpoint_sha256=report['checkpoint_sha256'],features=report['features'],
                vocabulary_sha256=report.get('lexical_vocabulary_sha256'))


def hashes(rows):return {text_hash(r[k]) for r in rows for k in ('source','target')}


def fit(args):
    train_receipt,train=load_scored(args.train);cal_receipt,cal=load_scored(args.calibration)
    if identity(train_receipt)!=identity(cal_receipt):raise ValueError('Generator/feature identity mismatch')
    if hashes(train)&hashes(cal):raise ValueError('Training/calibration overlap')
    train_groups=grouped(train);cal_groups=grouped(cal)
    if min(len(train_groups),len(cal_groups))<50:raise ValueError('Insufficient independent training/calibration cases')
    proposals=[r for r in train if vector(r) is not None]
    if len(proposals)<50 or len({r['raw_answer']==r['target'] for r in proposals})<2:raise ValueError('Insufficient contrastive examples')
    model=fit_logistic(np.asarray([vector(r) for r in proposals]),np.asarray([r['raw_answer']==r['target'] for r in proposals],dtype=float))
    chosen=select_threshold([winner(model,g) for g in cal_groups],[g[0] for g in cal_groups])
    model.update(threshold=chosen['threshold'],calibration=chosen,generator=identity(train_receipt),
        features=train_receipt['features']+EXTRA_FEATURES,excluded_text_hashes=sorted(hashes(train)|hashes(cal)),
        receipts=dict(train=train_receipt,calibration=cal_receipt),fitter_sha256=sha256_file(Path(__file__)),
        selection='Sentence-level exact-reference F0.5 on separate calibration cases',scope=__doc__)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'.building').touch()
    write_json(args.output/'model.json',model)
    write_json(args.output/'report.json',dict(completed=True,model_sha256=sha256_file(args.output/'model.json'),
        train_cases=len(train_groups),calibration_cases=len(cal_groups),train_proposals=len(proposals),calibration=chosen))
    (args.output/'.building').unlink()
    print(json.dumps(dict(event='candidate_selector_fitted',train_cases=len(train_groups),calibration=chosen)),flush=True)


def evaluate(args):
    receipt,rows=load_scored(args.scored)
    model_report=json.loads((args.model/'report.json').read_text())
    if (args.model/'.building').exists() or not model_report['completed'] or model_report['model_sha256']!=sha256_file(args.model/'model.json'):raise ValueError('Invalid selector artifact')
    model=json.loads((args.model/'model.json').read_text())
    if identity(receipt)!=model['generator']:raise ValueError('Generator/feature identity mismatch')
    if hashes(rows)&set(model['excluded_text_hashes']):raise ValueError('Evaluation overlaps training/calibration')
    results=[]
    for group in grouped(rows):
        r=group[0];proposal=winner(model,group)
        accepted=proposal['probability'] is not None and proposal['probability']>=model['threshold']
        answer=proposal['answer'] if accepted else r['source']
        results.append(dict(id=r['case_id'],kind=r['kind'],source=r['source'],target=r['target'],mode=r['mode'],
            raw_answer=answer,answer=answer,proposal_answer=proposal['answer'],acceptance_score=proposal['probability'],
            fallback_reason=None if accepted else 'candidate_selector_abstained',capacity_exceeded=False))
    from tools.correction.evaluate import measure
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'.building').touch()
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    contract=dict(bank_sha256=receipt['generator_contract']['case_bank_sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],
                  selector_sha256=model_report['model_sha256'],candidate_feature_report_sha256=sha256_file(args.scored/'report.json'))
    write_json(args.output/'report.json',dict(completed=True,contract=contract,groups=measure(results),
        rows_sha256=sha256_file(args.output/'rows.jsonl'),scope=__doc__))
    (args.output/'.building').unlink()
    print(json.dumps(dict(event='candidate_selector_evaluated',rows=len(results),groups=measure(results)['all'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    for command,names in [('flatten',('candidates','output')),('fit',('train','calibration','output')),('evaluate',('model','scored','output'))]:
        parser=s.add_parser(command)
        for name in names:parser.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();{'flatten':flatten,'fit':fit,'evaluate':evaluate}[args.command](args)


if __name__=='__main__':main()
