"""Offline, paired ablation on a frozen, reviewed manifest and isolated gallery.

No production state is changed. Gallery construction and report interpretation
are documented in reports/recognition/ablation-2026-09-20/README.md.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageOps

from app.import_catalog import load_catalog
from app.label_detection import crop_label, detect_label, enhance_label
from app.label_signals import color_delta, crop_color_features, ocr_delta, _lock_visual_leader
from app.recognition import Recognizer
from app.settings import settings
from app.vision import ImageEncoder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('reports/recognition/ablation-2026-09-20'))
    parser.add_argument('--cache-dir', type=Path, default=Path('backend/data/ablation-2026-09-20'))
    args = parser.parse_args()
    out, cache = args.run_dir, args.cache_dir
    manifest = json.loads((out/'manifest.json').read_text())
    gallery_meta = json.loads((cache/'gallery.json').read_text())
    gallery = np.load(cache/'gallery.npy')
    gallery_rows = gallery_meta['rows']
    slugs = list(dict.fromkeys(r['slug'] for r in gallery_rows))
    slug_index = {s:i for i,s in enumerate(slugs)}
    row_slug = np.array([slug_index[r['slug']] for r in gallery_rows])
    kinds = np.array([r['kind'] for r in gallery_rows])
    catalog = load_catalog(settings.catalog_csv)
    os.environ['CV_ENABLED'] = 'false'
    recognizer = Recognizer(catalog, settings)
    encoder = ImageEncoder()
    encoder.encode([Image.new('RGB',(224,224),'white')])
    all_cases=[]
    config_names=['full_full','full_both','label_full','label_crop','label_both','bbox_both','enhanced_both','gray_both','dual_max','dual_mean']

    def search(vector):
        similarities=gallery @ np.asarray(vector,dtype=np.float32)
        by_kind={}
        for kind in ['full','crop']:
            values=np.full(len(slugs),-1.0,dtype=np.float32)
            mask=kinds==kind
            np.maximum.at(values,row_slug[mask],similarities[mask])
            by_kind[kind]=values
        by_kind['both']=np.maximum(by_kind['full'],by_kind['crop'])
        return by_kind

    def rank(scores, full, crops, color, text='', use_color=False, use_ocr=False, lock=False, limit=8):
        # Same candidate limit and additive weights as production; exact search
        # replaces pgvector ANN so gallery/ANN changes cannot confound factors.
        ids=np.argsort(-scores,kind='stable')[:limit]
        rows=[]
        for i in ids:
            wine=catalog.get(slugs[i]);c=color_delta(color,wine) if use_color else 0.0;o=ocr_delta(text,wine) if use_ocr else 0.0
            rows.append({'slug':slugs[i],'score':float(np.clip(scores[i]+c+o,0,1)),'siglip':float(scores[i]),'full_score':float(full[i]) if full[i]>-1 else None,'crop_score':float(crops[i]) if crops[i]>-1 else None,'color_delta':c,'ocr_delta':o})
        rows.sort(key=lambda r:r['score'],reverse=True)
        return _lock_visual_leader(rows) if lock else rows

    for case in manifest:
        casefile=cache/f"query-{case['sha256']}.json"
        vectorfile=cache/f"query-{case['sha256']}.npz"
        if casefile.exists() and vectorfile.exists():
            data=json.loads(casefile.read_text());vectors=dict(np.load(vectorfile))
        else:
            image,decode_ms=timed(lambda:ImageOps.exif_transpose(Image.open(case['path'])).convert('RGB'))
            detection,detect_ms=timed(lambda:detect_label(image))
            label,crop_ms=timed(lambda:crop_label(image,detection))
            x0,y0,x1,y1=detection.bbox
            bbox,bbox_ms=timed(lambda:image.crop((int(x0*image.width),int(y0*image.height),int(x1*image.width),int(y1*image.height))))
            enhanced,enhance_ms=timed(lambda:enhance_label(label))
            gray,gray_ms=timed(lambda:ImageOps.grayscale(label).convert('RGB'))
            views={'full':image,'label':label,'bbox':bbox,'enhanced':enhanced,'gray':gray}
            prep={'full':decode_ms,'label':decode_ms+detect_ms+crop_ms,'bbox':decode_ms+detect_ms+bbox_ms,'enhanced':decode_ms+detect_ms+crop_ms+enhance_ms,'gray':decode_ms+detect_ms+crop_ms+gray_ms}
            vectors={};times={};colors={}
            for name,im in views.items():
                embeddings,ms=timed(lambda im=im:encoder.encode([im]));vectors[name]=np.array(embeddings[0],dtype=np.float32);times[name]=ms+prep[name];colors[name]=crop_color_features(im)
            ocr={}
            for name in ['full','label','enhanced']:
                (text,status),ms=timed(lambda name=name:recognizer._ocr(recognizer._image_bytes(views[name]),psm=6))
                ocr[name]={'text':text,'status':status,'ms':ms}
            label.thumbnail((600,600));label.save(cache/f"crop-{case['id']:02d}.jpg",quality=90)
            data={'prep_ms':prep,'image_ms':times,'ocr':ocr,'color':colors,'detection':asdict(detection)}
            casefile.write_text(json.dumps(data,ensure_ascii=False,indent=2));np.savez(vectorfile,**vectors)
        searches={name:search(v) for name,v in vectors.items()}
        expected=case['expected_slug'];modes={}

        def save(name,rows,latency):
            top=[r['slug'] for r in rows]
            modes[name]={'slug':top[0] if top else None,'correct':bool(top and top[0]==expected),'top5_hit':expected in top[:5],'top8_hit':expected in top,'rank':top.index(expected)+1 if expected in top else None,'score':rows[0]['score'] if rows else None,'gap':rows[0]['score']-rows[1]['score'] if len(rows)>1 else None,'latency_ms':latency,'top5':rows[:5]}

        for cfg in config_names:
            if cfg.startswith('dual_'):
                a,b=searches['full'],searches['label'];fuse=np.maximum if cfg=='dual_max' else lambda x,y:(x+y)/2
                scores=fuse(a['both'],b['both']);full=fuse(a['full'],b['full']);crop=fuse(a['crop'],b['crop']);image_ms=data['image_ms']['full']+data['image_ms']['label']-data['prep_ms']['full'];color=data['color']['label']
            else:
                view,gallery_kind=cfg.split('_');item=searches[view];scores=item[gallery_kind];full=item['full'];crop=item['crop'];image_ms=data['image_ms'][view];color=data['color'][view]
            rows,rank_ms=timed(lambda:rank(scores,full,crop,color));save(cfg,rows,image_ms+rank_ms)
        # Fully paired 2 x 2 color/OCR factorial, with and without visual lock.
        item=searches['label']
        for use_color in [False,True]:
            for use_ocr in [False,True]:
                for lock in [False,True]:
                    name=f"label_color{int(use_color)}_ocr{int(use_ocr)}_lock{int(lock)}"
                    rows,ms=timed(lambda:rank(item['both'],item['full'],item['crop'],data['color']['label'],data['ocr']['label']['text'],use_color,use_ocr,lock))
                    save(name,rows,data['image_ms']['label']+ms+(data['ocr']['label']['ms'] if use_ocr else 0))
        # Test where the color is measured: full image versus label crop.
        rows,ms=timed(lambda:rank(item['both'],item['full'],item['crop'],data['color']['full'],use_color=True))
        save('label_color_from_full',rows,data['image_ms']['label']+ms)
        # Alternative OCR view and a predeclared gate (not tuned on labels).
        for view in ['full','enhanced']:
            rows,ms=timed(lambda:rank(item['both'],item['full'],item['crop'],data['color']['label'],data['ocr'][view]['text'],False,True))
            extra_prep=max(0,data['prep_ms'][view]-data['prep_ms']['label'])
            save('label_ocr_from_'+view,rows,data['image_ms']['label']+extra_prep+data['ocr'][view]['ms']+ms)
        plain=modes['label_both'];gate=plain['gap'] is None or plain['gap']<.04
        modes['selective_ocr_gap004']=dict(modes['label_color0_ocr1_lock0'] if gate else plain,ocr_used=gate)
        # OCR-only retrieval, separate from OCR reranking of 8 CV candidates.
        for view in ['full','label']:
            matches,ms=timed(lambda:recognizer._text_matches(data['ocr'][view]['text']))
            save('ocr_only_'+view,[{'slug':m.wine.slug,'score':m.score} for m in matches[:8]],data['prep_ms'][view]+data['ocr'][view]['ms']+ms)
        all_cases.append({'id':case['id'],'file':case['file'],'expected_slug':expected,'subset':case['subset'],'scene':case['scene'],'sha256':case['sha256'],'gallery_exact_hash_overlap':case['sha256'] in {r['sha256'] for r in gallery_rows},'color':data['color'],'ocr':data['ocr'],'detection':data['detection'],'modes':modes})
        (out/'results.json').write_text(json.dumps(all_cases,ensure_ascii=False,indent=2))
        print(json.dumps({'completed':len(all_cases),'total':len(manifest),'case':case['id'],'crop_ms':round(data['image_ms']['label']),'ocr_ms':round(data['ocr']['label']['ms'])}),flush=True)
    meta={'manifest_sha256':digest(out/'manifest.json'),'gallery_metadata_sha256':digest(cache/'gallery.json'),'model':gallery_meta['model'],'revision':gallery_meta['revision'],'gallery_version':gallery_meta['version'],'gallery_vectors':len(gallery_rows),'gallery_wines':len(slugs),'missing_gallery_slugs':gallery_meta['missing'],'device':'cpu','threads':4,'timing_note':'Warm offline stages, one timing sample per query/view; no HTTP, pgvector, response serialization or UI. Reused identical vectors/OCR across paired variants; latency is sum of measured stages, not endpoint SLA.','source_hashes':{str(p):digest(p)for p in [Path('backend/app/label_detection.py'),Path('backend/app/label_signals.py'),Path('backend/app/vision.py'),Path(__file__)]}}
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
