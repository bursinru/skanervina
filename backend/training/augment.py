"""Seeded synthetic camera distortions; no external images or eval photos."""
from io import BytesIO
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

VERSION = 'camera-v1'


def augment(image, seed):
    rng = random.Random(seed)
    im = ImageOps.exif_transpose(image).convert('RGB')
    # Small working images avoid retaining multi-megapixel originals in a batch.
    im.thumbnail((512, 512))
    im = ImageOps.pad(im, (320, 320), color=(235, 231, 223))
    # Moderate projective warp with visible bottle retained.
    im = im.transform(im.size, Image.Transform.PERSPECTIVE,
                      (1, rng.uniform(-.10,.10), rng.uniform(-8,8),
                       rng.uniform(-.08,.08), 1, rng.uniform(-8,8),
                       rng.uniform(-.0003,.0003), rng.uniform(-.0003,.0003)),
                      Image.Resampling.BICUBIC, fillcolor=(225,220,210))
    im = im.rotate(rng.uniform(-10,10), Image.Resampling.BICUBIC, fillcolor=(225,220,210))
    cut = rng.randint(0, 16)
    im = im.crop((cut, cut, 320-cut, 320-cut)).resize((320,320), Image.Resampling.BILINEAR)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(.60,1.15))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(.7,1.25))
    im = ImageEnhance.Color(im).enhance(rng.uniform(.7,1.15))
    pixels = np.asarray(im).astype(np.float32)
    pixels *= np.array([rng.uniform(.90,1.10),1,rng.uniform(.90,1.10)], dtype=np.float32)
    pixels += np.random.default_rng(seed).normal(0,rng.uniform(0,4),pixels.shape)
    im = Image.fromarray(np.clip(pixels,0,255).astype(np.uint8))
    if rng.random() < .5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.25,.9)))
    if rng.random() < .35:
        overlay = Image.new('RGBA',im.size,(0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        x=rng.randint(30,270)
        draw.ellipse((x-10,0,x+20,320),fill=(255,255,255,rng.randint(20,55)))
        im=Image.alpha_composite(im.convert('RGBA'),overlay.filter(ImageFilter.GaussianBlur(9))).convert('RGB')
    output=BytesIO()
    im.save(output,format='JPEG',quality=rng.randint(45,90))
    output.seek(0)
    with Image.open(output) as decoded:
        return decoded.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
