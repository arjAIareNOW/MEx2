"""Isolated current-schema V4-B preparation and fixed-protocol training."""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, random, subprocess, time, wave
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from .tc_dscnn_v4 import IntentTCDSCNNV4B, architecture_config
from .v2_experiment_common import V2IntentDataset, ConservativeWaveformAugmentor, augmentation_config, padding_difference
from .data import collate_variable_audio
from .train import set_reproducible, class_weights, make_loader, train_epoch, evaluate, write_history
from .dscnn_exp1 import collect_environment, sha256_file
PROJECT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT/'models/vcm_tc_dscnn_v4_b_19class_final'
import sys
sys.path.insert(0,str(PROJECT/'vcm-benchmark'))
from vcmbench.schema import INTENTS, SLOTTED, VARIATIONS
CLASSES=sorted(INTENTS)
COLS=['command','speaker_id','source','is_synthetic','duration_s','file','out_of_scope']

def git_revision(path):
    # Optional metadata: compute nodes may not provide the Git executable.
    try:
        p=subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
    except (OSError, subprocess.SubprocessError):  # Includes FileNotFoundError and nonzero Git exits.
        return 'UNAVAILABLE_LOCAL_GIT_METADATA'
    return p.stdout.strip() or 'UNAVAILABLE_LOCAL_GIT_METADATA'

def dump(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n')

def read(files, audio=False):
    return pa.concat_tables([pq.read_table(p,columns=COLS+(['audio'] if audio else [])) for p in files])

def eligible(table):
    rows=table.to_pylist() if 'audio' not in table.column_names else table.select(COLS).to_pylist()
    for r in rows:
        assert r['command'] in CLASSES+['OUT_OF_SCOPE'],r['command']
        assert bool(r['out_of_scope']) == (r['command']=='OUT_OF_SCOPE')
        assert r['is_synthetic'] in (0,1)
        if r['command'] in CLASSES: assert r['speaker_id'], 'Missing in-scope speaker ID'
    return [i for i,r in enumerate(rows) if r['command'] in CLASSES]

def stats(table,indices):
    rows=table.select(COLS).take(pa.array(indices,type=pa.int64())).to_pylist()
    return {'count':len(rows),'hours':sum(r['duration_s'] for r in rows)/3600,
            'speakers':len({r['speaker_id'] for r in rows if r['speaker_id']}),
            'missing_speaker_rows':sum(not r['speaker_id'] for r in rows),
            'human':sum(r['is_synthetic']==0 for r in rows),'synthetic':sum(r['is_synthetic']==1 for r in rows),
            'class_counts':dict(sorted(Counter(r['command'] for r in rows).items()))}

def prepare(args):
    main_files=sorted((args.data/'data').glob('train-*.parquet'))
    fil_files=sorted((args.data/'supplemental_fil').glob('train-*.parquet'))
    assert len(main_files)==2 and len(fil_files)==3
    main=read(main_files,True);fil=read(fil_files,True)
    mi=eligible(main);fi=eligible(fil)
    speakers=main['speaker_id'].to_pylist(); filspeakers=set(fil['speaker_id'].to_pylist())
    # Speaker-level 15% validation, independently shuffled within human/synthetic source strata.
    strata={}
    for i in mi:
        if speakers[i] in filspeakers: continue
        key=(main['source'][i].as_py(),main['is_synthetic'][i].as_py())
        strata.setdefault(key,set()).add(speakers[i])
    rng=random.Random(args.seed); vs=set()
    for key,group in sorted(strata.items()):
        group=sorted(group);rng.shuffle(group)
        if len(group)>1: vs.update(group[:max(1,round(len(group)*.15))])
    vi=[i for i in mi if speakers[i] in vs]
    ti=[i for i in mi if speakers[i] not in vs]+[len(main)+i for i in fi]
    table=pa.concat_tables([main,fil])
    ts={table['speaker_id'][i].as_py() for i in ti}
    assert not ts & vs
    assert set(stats(table,ti)['class_counts'])==set(CLASSES)
    assert set(stats(table,vi)['class_counts'])==set(CLASSES)
    # Audio byte identity and PCM identity checks; no test labels influence split or choices.
    def identities(tab, inds):
        byte=[]; pcm=[]
        for i in inds:
            b=tab['audio'][i].as_py()['bytes']
            with wave.open(io.BytesIO(b),'rb') as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
                raw=w.readframes(w.getnframes()); assert raw
            byte.append(hashlib.sha256(b).hexdigest());pcm.append(hashlib.sha256(raw).hexdigest())
        return set(byte),set(pcm)
    tb,tp=identities(table,ti);vb,vp=identities(table,vi)
    assert not tb & vb and not tp & vp,'train/validation duplicate audio'
    test_files=sorted((args.data/'data').glob('test-*.parquet'))
    test=read(test_files,True);ei=eligible(test)
    es=set(test['speaker_id'].to_pylist()); eb,ep=identities(test,list(range(len(test))))
    overlap={'train_validation_speakers':sorted(ts & vs),'train_test_speakers':sorted(ts & es),
             'validation_test_speakers':sorted(vs & es),'train_validation_audio_bytes':len(tb & vb),
             'train_validation_pcm':len(tp & vp),'train_test_audio_bytes':len(tb & eb),
             'validation_test_audio_bytes':len(vb & eb),'train_test_pcm':len(tp & ep),
             'validation_test_pcm':len(vp & ep),
             'holdout':'NOT_INSPECTED: footer row count only; no rows/audio loaded',
             'reference_voice_identity':'Metadata IDs only; supplemental reference identities versus main voice clones are unverified'}
    assert not any(v for k,v in overlap.items() if k not in ('holdout','reference_voice_identity')),overlap
    files=main_files+fil_files+test_files
    detail={'main_train_raw':stats(main,list(range(len(main)))), 'main_train_19class':stats(main,mi),
            'supplemental_fil_raw':stats(fil,list(range(len(fil)))),'supplemental_fil_19class':stats(fil,fi),
            'train':stats(table,ti),'validation':stats(table,vi),'test_raw':stats(test,list(range(len(test)))),
            'test_19class':stats(test,ei),'holdout_footer_count':sum(pq.ParquetFile(p).metadata.num_rows for p in (args.data/'data').glob('holdout-*.parquet')),
            'leakage':overlap,'dataset_revision':git_revision(args.data),
            'file_sha256':{str(p):sha256_file(p) for p in files},
            'split_strategy':'seed 23101; source/is_synthetic strata; shuffled speakers; 15% rounded, minimum one when >=2; supplemental speaker IDs forced train only',
            'train_indices':ti,'validation_indices':vi,'train_speaker_ids':sorted(ts),'validation_speaker_ids':sorted(vs)}
    detail['dataset_fingerprint']=hashlib.sha256(json.dumps(detail['file_sha256'],sort_keys=True).encode()).hexdigest()
    return table,ti,vi,detail

def metadata(args,detail,model,aug):
    arch=architecture_config('B');arch['classifier']='Linear(96,19) raw logits'
    return {'status':'PREPARED_NOT_SUBMITTED','classes':CLASSES,'class_to_index':{c:i for i,c in enumerate(CLASSES)},
        'input':{'sample_rate':16000,'waveform':'mono PCM16; max 80000 samples; pad short clips to 400',
                 'features':'power mel -> dB top_db=80 -> per-utterance mean/std',
                 'dimensions':'[B,1,64,T], T<=501; frame_lengths [B] int64',
                 'n_fft':512,'window':400,'hop':160,'n_mels':64,'f_min':20,'f_max':8000},
        'architecture':arch,'layer_dimensions':str(model),'parameter_count':sum(p.numel() for p in model.parameters()),
        'float32_parameter_bytes':4*sum(p.numel() for p in model.parameters()),
        'checkpoint_size_bytes':'PENDING_TRAINING','checkpoint_sha256':'PENDING_TRAINING',
        'dataset':detail,'slot_schema':{c:sorted({v.value for v in VARIATIONS if v.intent==c}) for c in sorted(SLOTTED)},
        'slots':'Separate transcript/ASR slot parser; no slot head; runtime requires an external transcript provider for slotted commands',
        'training':{'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'scheduler':'CosineAnnealingLR T_max=60',
                    'maximum_epochs':60,'batch_size':64,'patience':10,'seed':23101,'dropout':.2,
                    'loss':'TRAIN-only inverse-frequency weighted CrossEntropyLoss, weights normalized to mean 1; no label smoothing',
                    'selection':'validation macro-F1 only; fresh initialization','steps_per_epoch':math.ceil(len(detail['train_indices'])/64),
                    'epochs_completed':'PENDING_TRAINING','steps_completed':'PENDING_TRAINING'},
        'augmentation':augmentation_config(args,aug),'benchmark_commit':git_revision(PROJECT/'vcm-benchmark'),
        'training_git_commit':git_revision(PROJECT),'source_sha256':{str(p.relative_to(PROJECT)):sha256_file(p) for p in [Path(__file__),PROJECT/'vcm/final19_runtime.py',PROJECT/'vcm/tc_dscnn_v4.py',PROJECT/'vcm/tc_dscnn_v2.py',PROJECT/'vcm/v2_experiment_common.py',PROJECT/'vcm/train.py',PROJECT/'vcm/data.py',PROJECT/'vcm/dscnn_exp1.py',PROJECT/'vcm/final19_evaluate.py',PROJECT/'scripts/train_vcm_tc_dscnn_v4_b_19class_final.slurm']},
        'environment':collect_environment(torch.device('cpu')),
        'a100_node':'PENDING_TRAINING','gpu':'PENDING_TRAINING','slurm_job_id':'PENDING_TRAINING',
        'training_wall_seconds':'PENDING_TRAINING','best_epoch':'PENDING_TRAINING',
        'validation_metrics':'PENDING_TRAINING','test_metrics':'PENDING_FINAL_TEST_EVALUATION',
        'holdout_metrics':'PENDING_RPI_BENCHMARK',
        'rpi':{k:'PENDING_RPI_BENCHMARK' for k in ['latency','infer_ms','real_time_factor','cpu','rss','temperature','power','wake_metrics','intent_and_slot_metrics']},
        'oos':'Excluded classifier target; source preserved. No rejection threshold selected; runtime external rejection gate required for OOS handling.'}

def summary(meta):
    lines=['# Final V4-B 19-class submission','',f"Status: {meta['status']}",'',
           '19 intent labels: '+', '.join(CLASSES), '',f"Parameters: {meta['parameter_count']}",
           'Input: mono 16 kHz, log-mel [B,1,64,T], T <= 501.', '',
           'Fixed V4-B backbone and training protocol. Official test reserved for final evaluation; holdout rows/audio never loaded.',
           'Slots require a separate ASR transcript provider at deployment; the intent head cannot predict slot values.',
           'Pi measurements: PENDING_RPI_BENCHMARK.', '', 'Full submission fields:', '', '```json',json.dumps(meta,indent=2),'```']
    (OUTPUT/'submission_summary.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path.home()/'ai231');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    vars(a).update(seed=23101,cpu_threads=4,batch_size=64,workers=0,epochs=60,noise_dir=PROJECT/'dataset/speech_commands/_background_noise_')
    set_reproducible(a.seed,a.cpu_threads)
    table,ti,vi,detail=prepare(a);mapping={c:i for i,c in enumerate(CLASSES)}
    aug=ConservativeWaveformAugmentor(a.noise_dir);model=IntentTCDSCNNV4B(19,.2)
    if a.preflight:
        OUTPUT.mkdir(exist_ok=True)
        assert not (OUTPUT/'training').exists()
        dump(OUTPUT/'split_manifest.json',detail)
        meta=metadata(a,detail,model,aug)
        import shutil
        for relative in meta['source_sha256']:
            dest=OUTPUT/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/relative,dest)
        samples=[V2IntentDataset(table,[ti[0],ti[len(ti)//2],ti[-1]],mapping)[i] for i in range(3)]
        x,lengths,y=collate_variable_audio(samples);weights=class_weights(CLASSES,detail['train']['class_counts'],torch.device('cpu'))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);before=model.stem.weight.detach().clone()
        logits=model(x,lengths);loss=torch.nn.CrossEntropyLoss(weight=weights)(logits,y);loss.backward();opt.step()
        assert logits.shape==(3,19) and torch.isfinite(loss) and not torch.equal(before,model.stem.weight)
        diff=padding_difference(model,x[:1],int(lengths[0]));assert diff<1e-5
        augmented=V2IntentDataset(table,[ti[-1]],mapping,aug)[0][0];assert torch.isfinite(augmented).all()
        from .final19_runtime import check_contract
        check_contract()
        smoke={'passed':True,'shape':list(x.shape),'lengths':lengths.tolist(),'output_shape':list(logits.shape),'loss':float(loss.detach()),'padding_difference':diff,'forward_backward_adamw':True,'augmentation':True,'runtime_contract':True}
        dump(OUTPUT/'preflight.json',smoke);dump(OUTPUT/'submission_metadata.json',meta);summary(meta)
        print(json.dumps({'smoke':smoke,'counts':{k:v for k,v in detail.items() if k not in ['train_indices','validation_indices','file_sha256','train_speaker_ids','validation_speaker_ids']},'parameters':meta['parameter_count']},indent=2));return
    frozen=json.loads((OUTPUT/'split_manifest.json').read_text())
    # Git availability is informational; all dataset hashes and split fields must match.
    assert frozen==dict(detail,dataset_revision=frozen['dataset_revision']),'Dataset/split changed since preflight'
    prepared=json.loads((OUTPUT/'submission_metadata.json').read_text())
    for relative,digest in prepared['source_sha256'].items():
        assert sha256_file(PROJECT/relative)==digest, f'Source changed since preflight: {relative}'
    target=OUTPUT/'training';assert not target.exists(),'Refusing overwrite'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
    device=torch.device('cuda');model.to(device);weights=class_weights(CLASSES,detail['train']['class_counts'],device)
    criterion=torch.nn.CrossEntropyLoss(weight=weights);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    tl=make_loader(V2IntentDataset(table,ti,mapping,aug),a,device,True);vl=make_loader(V2IntentDataset(table,vi,mapping),a,device,False)
    target.mkdir();meta=json.loads((OUTPUT/'submission_metadata.json').read_text());meta['status']='TRAINING'
    meta['environment']=collect_environment(device);meta['a100_node']=os.environ.get('SLURMD_NODENAME',os.uname().nodename);meta['gpu']=torch.cuda.get_device_name(0);meta['slurm_job_id']=os.environ.get('SLURM_JOB_ID','UNAVAILABLE')
    dump(target/'config.json',meta);start=time.perf_counter();history=[];best=-1;stale=0
    for epoch in range(1,61):
        loss,accuracy=train_epoch(model,tl,criterion,opt,device);report,vloss=evaluate(model,vl,criterion,device,CLASSES)
        row={'epoch':epoch,'learning_rate':opt.param_groups[0]['lr'],'train_loss':loss,'train_accuracy':accuracy,'validation_loss':vloss,**{'validation_'+k:v for k,v in report.summary.items()}}
        history.append(row);write_history(target/'training_history.csv',history);print(json.dumps(row),flush=True)
        if report.summary['macro_f1']>best:
            best=report.summary['macro_f1'];stale=0
            torch.save({'model_state':model.state_dict(),'model_config':{'num_classes':19,'dropout':.2},'class_to_index':mapping,'epoch':epoch,'validation_report':{'loss':vloss,**report.summary,'per_class':report.per_class,'confusion_matrix':report.confusion_matrix}},target/'best.pt')
        else:stale+=1
        scheduler.step()
        if stale>=10:break
    ck=torch.load(target/'best.pt',map_location='cpu',weights_only=True)
    meta.update(status='TRAINED_VALIDATION_ONLY',training_wall_seconds=time.perf_counter()-start,best_epoch=ck['epoch'],validation_metrics=ck['validation_report'],checkpoint_size_bytes=(target/'best.pt').stat().st_size,checkpoint_sha256=sha256_file(target/'best.pt'))
    meta['training'].update(epochs_completed=len(history),steps_completed=len(history)*len(tl));dump(OUTPUT/'submission_metadata.json',meta);summary(meta)
    (target/'COMPLETED').write_text('Validation-only training completed. Test and holdout not evaluated.\n')
if __name__=='__main__':main()
