// User-selected label area: the exported pixels are the actual recognition input.
export function initLabelCrop({ getPhoto, onApply, onError }) {
  const byId = id => document.getElementById(id);
  const dialog = byId('crop-dialog');
  const canvas = byId('crop-canvas');
  const context = canvas.getContext('2d');
  const fields = ['left', 'top', 'right', 'bottom'].map(key => byId('crop-' + key));
  let image, rect = [0, 0, 1, 1], start = null, sourceUrl;
  function draw(syncFields = true) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const [l, t, r, b] = rect;
    context.fillStyle = '#161012b8';
    context.fillRect(0, 0, canvas.width, t * canvas.height);
    context.fillRect(0, b * canvas.height, canvas.width, (1 - b) * canvas.height);
    context.fillRect(0, t * canvas.height, l * canvas.width, (b - t) * canvas.height);
    context.fillRect(r * canvas.width, t * canvas.height, (1 - r) * canvas.width, (b - t) * canvas.height);
    context.strokeStyle = '#f6daa4'; context.lineWidth = 3;
    context.strokeRect(l * canvas.width, t * canvas.height, (r - l) * canvas.width, (b - t) * canvas.height);
    if (syncFields) fields.forEach((field, i) => { field.value = Math.round(rect[i] * 100); });
    byId('crop-apply').disabled = (r - l) * image.naturalWidth < 32 || (b - t) * image.naturalHeight < 32;
  }
  function point(event) {
    const bounds = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)), Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))];
  }
  canvas.onpointerdown = event => { if (!image) return; start = point(event); canvas.setPointerCapture(event.pointerId); };
  canvas.onpointermove = event => {
    if (!start) return;
    const end = point(event);
    rect = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]; draw();
  };
  canvas.onpointerup = canvas.onpointercancel = () => { start = null; };
  fields.forEach(field => field.oninput = () => {
    const values = fields.map(input => Math.max(0, Math.min(100, Number(input.value))) / 100);
    rect = [Math.min(values[0], values[2]), Math.min(values[1], values[3]), Math.max(values[0], values[2]), Math.max(values[1], values[3])]; draw(false);
  });
  byId('crop-reset').onclick = () => { rect = [0, 0, 1, 1]; draw(); };
  byId('crop-close').onclick = () => dialog.close();
  dialog.addEventListener('close', () => { start = null; if (sourceUrl) URL.revokeObjectURL(sourceUrl); sourceUrl = null; });
  byId('crop-label').onclick = async () => {
    const photo = getPhoto(); if (!photo) return;
    sourceUrl = URL.createObjectURL(photo);
    try {
      image = new Image(); image.src = sourceUrl; await image.decode();
      const scale = Math.min(1, 900 / image.naturalWidth, 650 / image.naturalHeight);
      canvas.width = Math.round(image.naturalWidth * scale); canvas.height = Math.round(image.naturalHeight * scale);
      rect = [0, 0, 1, 1]; draw(); dialog.showModal();
    } catch { URL.revokeObjectURL(sourceUrl); sourceUrl = null; onError('Не удалось открыть фото для выделения.'); }
  };
  byId('crop-apply').onclick = async () => {
    byId('crop-apply').disabled = true;
    try {
      const [l, t, r, b] = rect;
      const width = (r - l) * image.naturalWidth, height = (b - t) * image.naturalHeight;
      if (width < 32 || height < 32) return;
      const output = document.createElement('canvas');
      const scale = Math.min(1, 2400 / Math.max(width, height));
      output.width = Math.round(width * scale); output.height = Math.round(height * scale);
      output.getContext('2d').drawImage(image, l * image.naturalWidth, t * image.naturalHeight, width, height, 0, 0, output.width, output.height);
      const blob = await new Promise(resolve => output.toBlob(resolve, 'image/jpeg', .95));
      if (!blob) throw new Error('Encoding failed');
      await onApply(new File([blob], 'label-crop.jpg', { type: 'image/jpeg' }));
      dialog.close();
    } catch { onError('Не удалось вырезать этикетку. Попробуйте снова.'); }
    finally { byId('crop-apply').disabled = false; }
  };
}
