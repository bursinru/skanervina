"""One bounded experiment: export -> synthetic queries -> frozen-encoder adapter -> report.

Run: PYTHONPATH=backend python -m training.run --run backend/data/training/adapter-v1
No production tables or active model configuration are modified.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from dotenv import load_dotenv
from app.database import connect
from app.vision import ImageEncoder, MODEL_ID, MODEL_REVISION
from training.adapter import QueryAdapter
from training.augment import augment, VERSION
from training.metrics import retrieval_metrics


def write_json(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temp.replace(path)


def export_dataset(root, image_root, seed, sizes):
    with connect() as db:
        rows=db.execute('''SELECT w.slug,w.card,e.image_hash,e.embedding::text AS embedding
            FROM wines w JOIN wine_embeddings e USING(slug) WHERE e.model=%s ORDER BY w.slug''',(MODEL_ID,)).fetchall()
    if not rows:
        raise ValueError('No indexed catalog images')
    gallery=np.asarray([json.loads(row.pop('embedding')) for row in rows],dtype=np.float32)
    groups={}
    for i,row in enumerate(rows):
        groups.setdefault(row['image_hash'],[]).append(i)
    # Ambiguous identical images cannot honestly supervise a unique slug.
    unique=[indices[0] for indices in groups.values() if len(indices)==1]
    partitions={name:[] for name in sizes}
    for index in unique:
        bucket=int(hashlib.sha256(f"{seed}:{rows[index]['image_hash']}".encode()).hexdigest()[:8],16)%10
        partitions['train' if bucket<8 else 'validation' if bucket==8 else 'test'].append(index)
    selected={}
    for split,count in sizes.items():
        choices=sorted(partitions[split],key=lambda i:hashlib.sha256(f"{seed}:{rows[i]['slug']}".encode()).hexdigest())
        if len(choices)<count:
            raise ValueError(f'Only {len(choices)} eligible {split} identities; requested {count}')
        selected[split]=choices[:count]
    examples=[]
    for split,indices in selected.items():
        for index in indices:
            row=rows[index]
            filename=unquote(Path(urlparse(row['card'].get('image_url') or '').path).name)
            path=(image_root/filename).resolve()
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError(f'Missing image for {row["slug"]}')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_hash']:
                raise ValueError(f'Image changed since indexing: {row["slug"]}')
            for view in range(2):
                aug_seed=int(hashlib.sha256(f"{seed}:{split}:{row['slug']}:{view}".encode()).hexdigest()[:8],16)
                examples.append({'split':split,'slug':row['slug'],'gallery_index':index,'image_hash':row['image_hash'],
                                 'image_path':str(path),'view':view,'seed':aug_seed})
    manifest={'seed':seed,'model':MODEL_ID,'revision':MODEL_REVISION,'augmentation':VERSION,
              'gallery_slugs':[r['slug'] for r in rows], 'gallery_size':len(rows),
              'exact_duplicate_groups':sum(len(v)>1 for v in groups.values()),
              'excluded_ambiguous_identities':len(rows)-len(unique),
              'split_population':{k:len(v) for k,v in partitions.items()},
              'selected_identities':{k:len(v) for k,v in selected.items()},
              'training_gallery_indices':partitions['train'], 'examples':examples,
              'limitations':['Synthetic query images derived from catalog references, not real shelf photos.',
                             'Exact file hashes grouped; near-duplicate photos may remain.',
                             'Validation/test wine identities never used in gradient updates, including negatives.',
                             'All catalog references are legitimate retrieval gallery entries for evaluation.',
                             'The three unlabelled organizer queries are excluded entirely.']}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()+gallery.tobytes()).hexdigest()
    manifest['fingerprint']=digest
    path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())['fingerprint']!=digest:
        raise ValueError('Dataset changed; use a new run directory')
    write_json(path,manifest)
    np.save(root/'gallery.npy',gallery)
    return manifest,gallery


def encode_queries(root,manifest,batch_size):
    examples=manifest['examples']
    cache=root/'queries.npz'
    features=[]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved['fingerprint'])!=manifest['fingerprint']:
                raise ValueError('Cache fingerprint mismatch')
            features=list(saved['features'])
    if len(features)<len(examples):
        encoder=ImageEncoder()
        start=time.monotonic()
        for offset in range(len(features),len(examples),batch_size):
            batch=examples[offset:offset+batch_size]
            images=[]
            for example in batch:
                with Image.open(example['image_path']) as image:
                    images.append(augment(image,example['seed']))
            features.extend(np.asarray(encoder.encode(images),dtype=np.float32))
            with (root/'queries.tmp').open('wb') as handle:
                np.savez_compressed(handle,features=np.asarray(features),fingerprint=manifest['fingerprint'])
            (root/'queries.tmp').replace(cache)
            status={'stage':'encoding','completed':len(features),'total':len(examples),'elapsed_seconds':round(time.monotonic()-start,2)}
            write_json(root/'status.json',status)
            print(json.dumps(status),flush=True)
        del encoder
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return np.asarray(features,dtype=np.float32)


def train(root,manifest,gallery,features,args):
    # This small head trains efficiently on CPU. Only feature extraction uses MPS/CUDA.
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model=QueryAdapter()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    splits={name:np.asarray([i for i,e in enumerate(manifest['examples']) if e['split']==name]) for name in ('train','validation','test')}
    labels=np.asarray([e['gallery_index'] for e in manifest['examples']])
    x=torch.tensor(features)
    g=F.normalize(torch.tensor(gallery),dim=-1)
    gallery=g.numpy()
    train_gallery=manifest['training_gallery_indices']
    local={index:i for i,index in enumerate(train_gallery)}
    targets=torch.tensor([local[int(labels[i])] for i in splits['train']],dtype=torch.long)
    refs=g[train_gallery]
    baseline=retrieval_metrics(features[splits['validation']],gallery,labels[splits['validation']])
    write_json(root/'baseline-validation.json',baseline)
    history=[]
    best_score=baseline['mrr']
    best_epoch=0
    torch.save({'state_dict':model.state_dict(),'model_id':MODEL_ID,'revision':MODEL_REVISION,
                'manifest_fingerprint':manifest['fingerprint'],'hidden':128,'epoch':0},root/'best.pt')
    start=time.monotonic()
    train_x=x[splits['train']]
    for epoch in range(1,args.epochs+1):
        model.train()
        permutation=torch.randperm(len(train_x))
        losses=[]
        for positions in permutation.split(64):
            query=model(train_x[positions])
            logits=query @ refs.T/.07
            positive=refs[targets[positions]]
            # All training-gallery negatives, including the nearest confusing wines.
            loss=F.cross_entropy(logits,targets[positions])+.2*(1-(query*positive).sum(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            validation=retrieval_metrics(model(x[splits['validation']]).numpy(),gallery,labels[splits['validation']])
        record={'epoch':epoch,'loss':sum(losses)/len(losses),'validation':validation,'elapsed_seconds':round(time.monotonic()-start,2)}
        history.append(record)
        write_json(root/'history.json',history)
        write_json(root/'status.json',{'stage':'training',**record})
        print(json.dumps(record),flush=True)
        if validation['mrr']>best_score+1e-6:
            best_score=validation['mrr']; best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'model_id':MODEL_ID,'revision':MODEL_REVISION,
                        'manifest_fingerprint':manifest['fingerprint'],'hidden':128,'epoch':epoch},root/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'rng':torch.get_rng_state(),'manifest_fingerprint':manifest['fingerprint']},root/'last.pt')
    # Select checkpoint using validation only. The held-out test is read only now.
    checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.inference_mode():
        adapted=model(x).numpy()
    report={'experiment':'Frozen SigLIP 2 + trained residual query adapter',
            'model':MODEL_ID,'revision':MODEL_REVISION,'manifest_fingerprint':manifest['fingerprint'],
            'selected_epoch':best_epoch,'epochs_run':args.epochs,
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'training_seconds':round(time.monotonic()-start,2),
            'baseline_validation':baseline,
            'trained_validation':retrieval_metrics(adapted[splits['validation']],gallery,labels[splits['validation']]),
            'baseline_test':retrieval_metrics(features[splits['test']],gallery,labels[splits['test']]),
            'trained_test':retrieval_metrics(adapted[splits['test']],gallery,labels[splits['test']]),
            'selected_identities':manifest['selected_identities'],
            'promotion':'not_deployed_requires_real_labelled_validation',
            'limitations':manifest['limitations']}
    write_json(root/'metrics.json',report)
    write_json(root/'status.json',{'stage':'complete','selected_epoch':best_epoch,'metrics':'metrics.json'})
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--train',type=int,default=256)
    p.add_argument('--validation',type=int,default=64)
    p.add_argument('--test',type=int,default=64)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--images',type=Path,default=Path('Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads'))
    args=p.parse_args()
    if min(args.train,args.validation,args.test,args.epochs,args.batch_size)<=0:
        p.error('Counts must be positive')
    load_dotenv('backend/.env.local')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=True)
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(torch=torch.__version__,python=platform.python_version(),device=os.getenv('CV_DEVICE','cpu'),
                  source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text())!=config:
        raise ValueError('Configuration/code changed; use a new run directory')
    write_json(root/'config.json',config)
    if (root/'metrics.json').exists():
        print('Completed experiment already exists; not overwriting.');return
    write_json(root/'status.json',{'stage':'preparing','started_at':datetime.now(timezone.utc).isoformat()})
    manifest,gallery=export_dataset(root,args.images.resolve(),args.seed,{'train':args.train,'validation':args.validation,'test':args.test})
    features=encode_queries(root,manifest,args.batch_size)
    train(root,manifest,gallery,features,args)


if __name__=='__main__':
    main()
