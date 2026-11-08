"""Verify the frozen staging contents without loading models or training data."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,statistics
from collections import defaultdict
from pathlib import Path

def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def verify(root):
    root=root.resolve();rows=read(root/'manifests/release_files.tsv');known=set()
    for item in rows:
        rel=item['relative_path'];p=(root/rel).resolve()
        assert p.is_relative_to(root) and rel not in known and p.is_file(),rel;known.add(rel)
        assert p.stat().st_size==int(item['bytes']) and sha(p)==item['sha256'],rel
        assert p.stat().st_size<50*1024**2,'Large normal-Git file: '+rel
    ignore={'manifests/release_files.tsv'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.relative_to(root).parts[0] not in ['outputs','.pytest_cache']}
    assert actual==known|ignore,(actual-known-ignore,known-actual)
    scores=read(root/'results/primary_18_scores.tsv');assert len(scores)==18
    seen=set();maximum_error=0.0
    for row in scores:
        key=(row['experiment_id'],row['endpoint']);assert key not in seen;seen.add(key)
        data=read(root/f'results/per_protein/{key[1]}__{key[0]}.tsv')
        count,positions={'md_test':(136,18939),'soppe_test':(273,32673),'experimental10':(10,709)}[key[1]]
        assert len(data)==count and sum(int(r['eligible_residues']) for r in data)==positions
        for metric in ['rmse','mae','bias','pcc','spearman','out_of_range_fraction']:
            values=[float(r[metric]) for r in data if r[metric] and math.isfinite(float(r[metric]))]
            if values:
                error=abs(statistics.mean(values)-float(row['macro_'+metric]));maximum_error=max(maximum_error,error);assert error<1e-9
    model_registry=json.loads((root/'checkpoints/model_registry.json').read_text())
    assert len(model_registry['models'])==6 and len({e for e,t in seen})==6
    security_patterns=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),re.compile(r'https?://[^\s/@:]+:[^\s/@]+@'),re.compile(r'(?i)(?:password|passwd|api_key|access_token)\s*[:=]\s*[\"\x27](?!<|\{|\$|YOUR_|example)[^\"\x27\n]{6,}[\"\x27]')]
    forbidden_addresses=['.'.join(map(str,parts)) for parts in [(100,76,76,11),(100,79,36,68),(222,20,96,163)]]
    forbidden_addresses.append('login.'+'tailscale.com/')
    text_files=0
    for item in rows:
        p=root/item['relative_path']
        if p.suffix.lower() not in ['.py','.sh','.md','.txt','.json','.yaml','.yml','.tsv','.csv','.svg',''] and p.name not in ['.gitignore','.gitattributes']:continue
        try:text=p.read_text(encoding='utf-8')
        except UnicodeDecodeError:continue
        text_files+=1
        assert not any(pattern.search(text) for pattern in security_patterns),'Possible secret; inspect privately: '+item['relative_path']
        assert not any(address in text for address in forbidden_addresses),'Private endpoint in: '+item['relative_path']
    return {'status':'PASS','files_verified':len(rows),'total_payload_bytes':sum(int(r['bytes']) for r in rows),'model_count':6,'primary_score_cells':18,
        'per_protein_records_reaggregated':2514,'maximum_macro_error':maximum_error,'text_files_scanned':text_files,
        'secret_scan':'heuristic_pass_not_a_security_guarantee','network_contacted':False,'model_inference_or_training':False}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=ap.parse_args();print(json.dumps(verify(args.root),indent=2))
