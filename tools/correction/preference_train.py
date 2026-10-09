"""Cached-reference edit-focused preference training with a matched SFT control.

Bangla grapheme-aligned weighting is an adaptation, not published EPO replication.
No evaluation labels enter preferences, reference caches, or training objectives.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from tools.autosuggest.checkpoints import atomic_torch_save,capture_rng,restore_rng
from tools.corpus.provenance import digest_json,sha256_file,write_json
from tools.correction.byte_train import load_model,collate
from tools.correction.preference_data import edit_weights


def preference_loss(chosen,rejected,reference_chosen,reference_rejected,beta=.1):
    advantage=beta*((chosen-reference_chosen)-(rejected-reference_rejected))
    return -F.logsigmoid(advantage).mean(),advantage.detach().mean()


def scores(model,rows,device):
    n=len(rows)
    targets=[r['chosen'] for r in rows]+[r['rejected'] for r in rows]
    pairs=[dict(r,target=t) for r,t in zip(rows+rows,targets)]
    tensors=collate(pairs,device)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(**tensors,use_cache=False)
    labels=tensors['labels'];valid=labels.ne(-100)
    logp=out.logits.float().log_softmax(-1).gather(-1,labels.clamp_min(0)[...,None]).squeeze(-1)*valid
    weights=[edit_weights(r['chosen'],r['rejected']) for r in rows]+[edit_weights(r['rejected'],r['chosen']) for r in rows]
    w=torch.tensor([v+[0.]*(labels.shape[1]-len(v)) for v in weights],device=device)
    weighted=(logp*w).sum(-1)
    ce=-logp[:n].sum()/valid[:n].sum()
    return weighted[:n],weighted[n:],ce


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','pretrained','initialize-from','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-only',action='store_true')
    p.add_argument('--objective',choices=('sft','preference'),default='preference')
    p.add_argument('--steps',type=int,default=600);p.add_argument('--batch',type=int,default=8);p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=3e-6);p.add_argument('--beta',type=float,default=.1);p.add_argument('--sft-weight',type=float,default=.1)
    p.add_argument('--warmup',type=int,default=30);p.add_argument('--seed',type=int,default=20261006)
    p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int);p.add_argument('--stop-epoch',type=float,required=True)
    a=p.parse_args()
    if min(a.steps,a.batch,a.accumulate,a.warmup)<1 or not all(math.isfinite(x) and x>0 for x in (a.learning_rate,a.beta,a.sft_weight)):p.error('Invalid schedule')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(8);random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda')
    manifest=json.loads((a.data/'manifest.json').read_text())
    if (a.data/'.building').exists() or manifest['data_id']!=digest_json({k:v for k,v in manifest.items() if k!='data_id'}) or manifest['pairs_sha256']!=sha256_file(a.data/'pairs.jsonl'):raise ValueError('Preference data integrity failure')
    rows=list(map(json.loads,(a.data/'pairs.jsonl').read_text().splitlines()))
    source_hash=sha256_file(a.initialize_from)
    weights_hash=sha256_file(Path(__file__).with_name('preference_data.py'))
    cache_contract=dict(data_id=manifest['data_id'],checkpoint_sha256=source_hash,weights_sha256=weights_hash,
                        scorer_sha256=sha256_file(Path(__file__)),batch=a.batch,precision='bf16 autocast FP32 master')
    state=torch.load(a.initialize_from,map_location='cpu',weights_only=False)
    pretrained_files={f.name:sha256_file(f) for f in a.pretrained.iterdir() if f.is_file()}
    if state['kind']!='obadh-byte-accuracy-reference' or state['contract']['pretrained_files']!=pretrained_files or state['contract']['byte_limit']!=manifest['byte_limit']:raise ValueError('Incompatible initial model')
    model=load_model(a.pretrained);model.load_state_dict(state['state_dict'],strict=True);del state
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.to(device);model.config.use_cache=False
    if not a.cache.exists():
        model.eval();chosen_logp=[];rejected_logp=[]
        with torch.inference_mode():
            for start in range(0,len(rows),a.batch):
                if time.time()>=a.stop_epoch:raise RuntimeError('Cache deadline reached')
                c,r,_=scores(model,rows[start:start+a.batch],device);chosen_logp.extend(c.tolist());rejected_logp.extend(r.tolist())
                if start%(a.batch*25)==0:print(json.dumps(dict(event='preference_cache',rows=min(start+a.batch,len(rows)),total=len(rows))),flush=True)
        atomic_torch_save(dict(contract=cache_contract,chosen=chosen_logp,rejected=rejected_logp),a.cache)
    cached=torch.load(a.cache,map_location='cpu',weights_only=False)
    if cached['contract']!=cache_contract or len(cached['chosen'])!=len(rows) or len(cached['rejected'])!=len(rows) or not all(math.isfinite(x) for x in cached['chosen']+cached['rejected']):raise ValueError('Reference cache mismatch')
    if a.cache_only:
        print(json.dumps(dict(event='preference_cache_complete',rows=len(rows),cache_sha256=sha256_file(a.cache))),flush=True);return
    contract=dict(kind='obadh-byte-accuracy-reference',data_id=manifest['data_id'],byte_limit=manifest['byte_limit'],pretrained_files=pretrained_files,
        initialization_sha256=source_hash,cache_contract=cache_contract,cache_sha256=sha256_file(a.cache),objective=a.objective,
        steps=a.steps,batch=a.batch,accumulate=a.accumulate,learning_rate=a.learning_rate,warmup=a.warmup,beta=a.beta,sft_weight=a.sft_weight,
        seed=a.seed,dropout=0.,group_sampling='equal repair and preservation',script_sha256=sha256_file(Path(__file__)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,betas=(.9,.999),weight_decay=.01,fused=True)
    rng=random.Random(a.seed+1);step=0;history=[]
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['contract']!=contract:raise ValueError('Preference resume contract mismatch')
        model.load_state_dict(state['state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
        rng.setstate(state['sampling_rng']);restore_rng(state['rng'],device);step=state['step'];history=state['history'];del state
    elif a.output.exists() and any(a.output.iterdir()):raise ValueError('Existing training requires explicit resume')
    a.output.mkdir(parents=True,exist_ok=True)
    pools={group:[i for i,r in enumerate(rows) if r['group']==group] for group in ('repair','preserve')}
    if not all(pools.values()):raise ValueError('Both preference groups required')
    def save(status):
        atomic_torch_save(dict(kind='obadh-byte-accuracy-reference',contract=contract,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            sampling_rng=rng.getstate(),rng=capture_rng(device),step=step,history=history,status=status),a.output/'latest.pt')
    model.train();save('running');last_save=time.monotonic();started=last_save
    while step<a.steps and (a.stop_after is None or step<a.stop_after):
        if time.time()>=a.stop_epoch:break
        progress=max(0.,(step-a.warmup)/max(1,a.steps-a.warmup));rate=(step+1)/a.warmup if step<a.warmup else .1+.9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=a.learning_rate*rate
        optimizer.zero_grad(set_to_none=True);losses=[];advantages=[]
        for _ in range(a.accumulate):
            indices=[rng.choice(pools['repair' if i%2 else 'preserve']) for i in range(a.batch)];batch=[rows[i] for i in indices]
            chosen,rejected,ce=scores(model,batch,device)
            rc=torch.tensor([cached['chosen'][i] for i in indices],device=device);rr=torch.tensor([cached['rejected'][i] for i in indices],device=device)
            preference,advantage=preference_loss(chosen,rejected,rc,rr,a.beta)
            loss=a.sft_weight*ce+(preference if a.objective=='preference' else 0.)
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite preference loss')
            (loss/a.accumulate).backward();losses.append(float(loss.detach()));advantages.append(float(advantage))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();step+=1
        if step%25==0:
            record=dict(event='preference_train',objective=a.objective,step=step,loss=sum(losses)/len(losses),advantage=sum(advantages)/len(advantages),gradient_norm=float(norm),seconds=time.monotonic()-started)
            history.append(record);print(json.dumps(record),flush=True)
        if step%100==0 or time.monotonic()-last_save>=300:save('running');last_save=time.monotonic()
    status='completed' if step==a.steps else 'stopped';save(status)
    reloaded=torch.load(a.output/'latest.pt',map_location='cpu',weights_only=False)
    if reloaded['step']!=step or reloaded['contract']!=contract:raise RuntimeError('Checkpoint reload failed')
    del reloaded
    write_json(a.output/'report.json',dict(status=status,step=step,contract=contract,checkpoint_reload_verified=True,peak_cuda_bytes=torch.cuda.max_memory_allocated(),history=history))
    print(json.dumps(dict(event='preference_training_stage_complete',objective=a.objective,step=step,status=status)),flush=True)


if __name__=='__main__':main()
