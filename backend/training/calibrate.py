"""Calibrate on validation only; evaluate once on held-out synthetic test queries."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.adapter import QueryAdapter
from training.metrics import retrieval_metrics


def wilson_lower(successes, n, z=1.96):
    if not n:
        return 0.
    p=successes/n
    return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def choose_threshold(queries,gallery,labels):
    choices=[]
    for threshold in np.arange(.60,.991,.01):
        for margin in np.arange(.00,.151,.01):
            m=retrieval_metrics(queries,gallery,labels,float(round(threshold,2)),float(round(margin,2)))
            if m['accepted']>=20 and m['accepted_precision']>=.98:
                lower=wilson_lower(m['accepted']-m['false_accepts'],m['accepted'])
                if lower>=.90:
                    choices.append({**m,'precision_wilson_lower_95':lower})
    if not choices:
        return None
    return max(choices,key=lambda m:(m['coverage'],m['accepted_precision'],m['threshold'],m['margin']))


def paired_interval(base,adapted,gallery,labels,groups,seed=20260916):
    a=(base @ gallery.T).argmax(axis=1)==labels
    b=(adapted @ gallery.T).argmax(axis=1)==labels
    group_list=np.unique(groups)
    deltas=np.asarray([(b[groups==g].astype(float)-a[groups==g]).mean() for g in group_list])
    rng=np.random.default_rng(seed)
    sampled=deltas[rng.integers(0,len(deltas),size=(2000,len(deltas)))].mean(axis=1)
    return {'top1_delta':float((b.astype(float)-a).mean()),'bootstrap_identity_groups':len(group_list),
            'paired_95_percentile_interval':np.quantile(sampled,[.025,.975]).tolist(),
            'note':'Bootstrap groups both synthetic views by source identity; not a real-photo confidence interval.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    root=args.run
    m=json.loads((root/'manifest.json').read_text())
    features=np.load(root/'queries.npz',allow_pickle=False)['features']
    gallery=np.load(root/'gallery.npy',allow_pickle=False)
    gallery=gallery/np.linalg.norm(gallery,axis=1,keepdims=True)
    model=QueryAdapter()
    checkpoint=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint['manifest_fingerprint']!=m['fingerprint']:
        raise ValueError('Checkpoint does not belong to this dataset')
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.inference_mode():
        adapted=model(torch.tensor(features)).numpy()
    labels=np.array([e['gallery_index'] for e in m['examples']])
    val=np.array([e['split']=='validation' for e in m['examples']])
    test=np.array([e['split']=='test' for e in m['examples']])
    report={'selection_data':'validation only','target_empirical_precision':.98,'minimum_accepts':20,
            'minimum_wilson_precision_lower_95':.90,
            'warning':'Operating thresholds remain experimental; synthetic calibration is not production validation.'}
    for name,values in [('baseline',features),('adapter',adapted)]:
        choice=choose_threshold(values[val],gallery,labels[val])
        report[name]={'validation_choice':choice,'test':retrieval_metrics(values[test],gallery,labels[test],choice['threshold'],choice['margin']) if choice else None}
    report['paired_test_top1']=paired_interval(features[test],adapted[test],gallery,labels[test],labels[test])
    (root/'calibration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
