"""Portable, feature-input-only inference for frozen MD-SOP v2.2 weights.

This wrapper does not extract ESM-C/HMM features, train, clip, or smooth outputs.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from model_bilstm_window_v2_2 import MDSoPBiLSTMRegressor,ModelConfig,state_dict_sha256

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=('A_MD_w1','A_MD_w15','B_SOPPE_w1','B_SOPPE_w15','C_MIX_w1','C_MIX_w15')

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4194304),b''):h.update(block)
    return h.hexdigest()

def checked_features(array,dimension):
    array=np.asarray(array)
    if array.ndim!=2 or array.shape[0]<1 or array.shape[1]!=dimension:
        raise ValueError('Expected nonempty [L,%d] features'%dimension)
    if array.dtype.kind!='f':raise TypeError('Features must be floating-point arrays')
    if not np.isfinite(array).all():raise ValueError('Features contain NaN or infinity')
    result=np.asarray(array,dtype=np.float32,order='C')
    if not np.isfinite(result).all():raise ValueError('Features overflow float32')
    return result

def concatenate_features(esmc,hmm):
    esmc=checked_features(esmc,1152);hmm=checked_features(hmm,30)
    if esmc.shape[0]!=hmm.shape[0]:raise ValueError('ESM-C/HMM residue row counts differ')
    return np.concatenate([esmc,hmm],axis=1)

def load_model(experiment_id,device='cpu',root=ROOT):
    root=Path(root).resolve();registry=json.loads((root/'checkpoints/model_registry.json').read_text(encoding='utf-8'))
    if experiment_id not in EXPERIMENTS:raise ValueError('Unknown frozen experiment')
    item=registry['models'][experiment_id];path=(root/item['file']).resolve()
    if not path.is_relative_to(root/'checkpoints'):raise ValueError('Checkpoint escaped its directory')
    if sha256_file(path)!=item['file_sha256']:raise ValueError('Checkpoint file digest mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('format')!='md_sop_inference_v2_2' or payload.get('experiment_id')!=experiment_id:
        raise ValueError('Unexpected inference checkpoint identity')
    if payload.get('run_tag')!='v2_2_20261107':raise ValueError('Wrong experiment release')
    model=MDSoPBiLSTMRegressor(ModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state_dict'],strict=True)
    state_hash=state_dict_sha256(model.state_dict())
    if state_hash!=payload['model_state_sha256'] or state_hash!=item['model_state_sha256']:
        raise ValueError('Model tensor digest mismatch')
    model.eval();model.to(torch.device(device))
    return model,item

def predict_array(model,array,device='cpu'):
    array=checked_features(array,1182)
    x=torch.from_numpy(array).unsqueeze(0).to(torch.device(device))
    with torch.inference_mode():prediction=model(x,torch.tensor([len(array)],dtype=torch.int64))[0,:len(array)]
    result=prediction.detach().cpu().numpy()
    if not np.isfinite(result).all():raise RuntimeError('Nonfinite model prediction')
    return result

def fasta_sequence(path):
    lines=[line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or not lines[0].startswith('>') or sum(s.startswith('>') for s in lines)!=1:
        raise ValueError('FASTA must contain exactly one sequence')
    sequence=''.join(lines[1:]).upper()
    if not sequence or any(not ('A'<=s<='Z') for s in sequence):raise ValueError('Invalid FASTA sequence')
    return sequence

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True,choices=EXPERIMENTS)
    ap.add_argument('--features',type=Path,help='float .npy [L,1182], ESM-C followed by HMM30')
    ap.add_argument('--esmc',type=Path,help='float .npy [L,1152]')
    ap.add_argument('--hmm',type=Path,help='float .npy [L,30]')
    ap.add_argument('--fasta',type=Path,help='Optional one-chain sequence for length validation; row identity remains the preparer responsibility')
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    if args.output.exists():ap.error('Output exists; use a new output file')
    if args.threads<1:ap.error('threads must be positive')
    if args.features and (args.esmc or args.hmm):ap.error('Use --features OR the --esmc/--hmm pair')
    if not args.features and not (args.esmc and args.hmm):ap.error('Provide --features OR both --esmc and --hmm')
    torch.set_num_threads(args.threads)
    if args.features:array=checked_features(np.load(args.features,allow_pickle=False),1182)
    else:array=concatenate_features(np.load(args.esmc,allow_pickle=False),np.load(args.hmm,allow_pickle=False))
    sequence=fasta_sequence(args.fasta) if args.fasta else None
    if sequence is not None and len(sequence)!=len(array):ap.error('FASTA and feature lengths differ')
    model,item=load_model(args.model,args.device);prediction=predict_array(model,array,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='') as f:
        f.write('sequence_position_1based\tamino_acid\tpredicted_s2\n')
        for i,value in enumerate(prediction):f.write('%d\t%s\t%.9g\n'%(i+1,sequence[i] if sequence else '',value))
    print(json.dumps({'model':args.model,'run_tag':'v2_2_20261107','sequence_length':len(array),'checkpoint_sha256':item['file_sha256'],
        'model_state_sha256':item['model_state_sha256'],'device':args.device,'prediction_transform':'none','output':str(args.output)},indent=2))
if __name__=='__main__':main()
