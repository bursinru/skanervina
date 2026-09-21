import os,json,hashlib,time,sys
from pathlib import Path
from urllib.parse import unquote,urlparse
os.environ.update(HF_HOME=str(Path('backend/data/models').resolve()),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CV_DEVICE='cpu',CV_THREADS='4')
import numpy as np
from PIL import Image,ImageOps
from app.import_catalog import load_catalog,label_crop_if_useful
from app.settings import settings
from app.vision import ImageEncoder,MODEL_ID,MODEL_REVISION
out=Path('backend/data/ablation-2026-09-20');root=Path('Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads');catalog=load_catalog(settings.catalog_csv)
source_files=['backend/app/label_detection.py','backend/app/vision.py','backend/app/import_catalog.py',str(settings.catalog_csv)]
version=hashlib.sha256(''.join(hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files).encode()).hexdigest()
print(json.dumps({'phase':'load_model','model':MODEL_ID,'version':version}),flush=True)
encoder=ImageEncoder(); encoder.encode([Image.new('RGB',(224,224),'white')])
rows=[];images=[];vectors=[];missing=[];failed=[];t=time.perf_counter();done=0

def flush():
 global images
 if not images:return
 vectors.extend(encoder.encode(images));images=[]
for w in catalog:
 path=root/(unquote(Path(urlparse(w.direct_image_url or '').path).name) or w.image_name)
 if not path.is_file():missing.append(w.slug);continue
 try:
  rgb=ImageOps.exif_transpose(Image.open(path)).convert('RGB');digest=hashlib.sha256(path.read_bytes()).hexdigest()
  views=[('full',rgb)];crop=label_crop_if_useful(rgb)
  if crop is not None:views.append(('crop',crop))
  for kind,im in views:
   rows.append({'slug':w.slug,'kind':kind,'path':str(path),'sha256':digest});images.append(im)
   if len(images)>=16:flush()
  done+=1
  if done%100==0:print(json.dumps({'gallery_wines':done,'vectors':len(vectors),'seconds':round(time.perf_counter()-t,1)}),flush=True)
 except Exception as e:failed.append({'slug':w.slug,'error':str(e)})
flush();np.save(out/'gallery.npy',np.array(vectors,dtype=np.float32));meta={'model':MODEL_ID,'revision':MODEL_REVISION,'version':version,'rows':rows,'missing':missing,'failed':failed,'seconds':time.perf_counter()-t,'device':'cpu','threads':4,'catalog_size':catalog.size,'indexed_wines':done}
(out/'gallery.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in meta.items()if k not in ['rows','missing','failed']}),flush=True)
