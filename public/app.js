const $ = (id) => document.getElementById(id);
const paths = {
  wine: '<path d="M7 3h10l1 7a6 6 0 0 1-12 0zM12 16v5M8 21h8M7 9h10"/>',
  camera: '<path d="M9 4h6l2 3h4v13H3V7h4z"/><circle cx="12" cy="13" r="4"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-6 5 7"/>',
  scan: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M7 8h10M7 12h10M7 16h7"/>',
  focus: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="12" r="3"/>',
  bookmark: '<path d="M6 3h12v19l-6-4-6 4z"/>',
  star: '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2-5.6-3-5.6 3 1.1-6.2L3 9.6l6.2-.9z"/>',
  utensils: '<path d="M4 3v6a3 3 0 0 0 6 0V3M7 3v18M19 3c-4 4-4 10 0 10V3v18"/>',
  sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5zM20 2v4m-2-2h4"/>',
  'arrow-right': '<path d="M4 12h16m-6-6 6 6-6 6"/>',
  'arrow-left': '<path d="M20 12H4m6-6-6 6 6 6"/>',
  'arrow-up-right': '<path d="M6 18 18 6M6 6h12v12"/>',
  x: '<path d="m6 6 12 12M6 18 18 6"/>',
  check: '<path d="m5 12 4 4L19 6"/>'
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.wine}</svg>`;
document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const example = {
  slug: 'fanagoriya-blanc-de-blancs-shardone-beloe-bryut-12', name: 'Blanc de Blancs', winery: 'Фанагория',
  category: 'Игристое белое · Брют', region: 'Кубань', grapes: ['Шардоне'], image_url: '/assets/blanc-de-blancs.webp',
  summary: 'Свежее, изящное вино с нотами белых цветов, персика и грейпфрута. Сбалансированная кислотность и лёгкая сливочность во вкусе.',
  description: 'Игра пузырьков изящная и утонченная — непрерывный жемчужный перляж. Аромат элегантный и тонкий, завораживающий. Вначале раскрывается нотами легких белых цветов и прохладой цветущего утреннего сада. Затем проявляются сочные оттенки мякоти белых фруктов и розового персика, свежие нотки грейпфрута и легкий травяной акцент. Фоном звучит тонкий оттенок миндального пралине. Вкус изящный, свежий и ажурный. Прекрасно сбалансированная, стройная кислотность отлично гармонирует с тонкой сливочностью и ведет к чистому и стройному послевкусию. Идеальный аперитив.',
  aromas: ['Белые цветы', 'Персик', 'Грейпфрут', 'Миндаль'], source: 'Описание из каталога «Своё Вино»', demo: true
};
let stream = null, cameraGeneration = 0, photo = null, photoUrl = null, controller = null, currentWine = null, toastTimer;
let saved = [];
try { const raw = JSON.parse(localStorage.getItem('svoe-wines') || '[]'); if (Array.isArray(raw)) saved = raw.filter(w => w && typeof w.slug === 'string' && typeof w.name === 'string').slice(0,100); } catch {}
const syncSavedCount = () => $('saved-count').textContent = saved.length;
syncSavedCount();
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function toast(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => $('toast').hidden = true, 3500); }
function stopCamera() { cameraGeneration++; stream?.getTracks().forEach(t => t.stop()); stream = null; $('camera-video').srcObject = null; $('camera-video').hidden = true; $('open-camera').disabled = false; }
function updateControls(mode) {
  ['initial','preview','camera'].forEach(name => $(name + '-actions').hidden = name !== mode);
  $('control-title').textContent = { initial:'Начнём с этикетки', preview:'Этикетка хорошо видна?', camera:'Поймайте этикетку в рамку' }[mode];
  $('control-description').textContent = { initial:'Сфотографируйте бутылку или выберите снимок.', preview:'Название и производитель должны читаться.', camera:'Держите телефон ровно и избегайте бликов.' }[mode];
  $('file-note').hidden = mode !== 'initial';
}
async function selectPhoto(file) {
  if (!file) return;
  notice('');
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) { notice('Выберите JPG, PNG или WebP. Для HEIC сохраните фотографию в JPG.'); return; }
  if (file.size > 15 * 1024 * 1024) { notice('Фотография слишком большая. Выберите файл до 15 МБ.'); return; }
  const candidateUrl = URL.createObjectURL(file);
  try {
    const img = new Image(); img.src = candidateUrl; await img.decode();
    if (!img.naturalWidth || !img.naturalHeight) throw new Error('Invalid image');
  } catch { URL.revokeObjectURL(candidateUrl); notice('Не удалось открыть изображение. Попробуйте другую фотографию.'); return; }
  stopCamera(); if (photoUrl) URL.revokeObjectURL(photoUrl);
  photo = file; photoUrl = candidateUrl;
  $('photo-preview').src = photoUrl; $('photo-preview').hidden = false;
  $('example-bottle').hidden = true; $('sample-label').hidden = true;
  $('viewfinder-caption').textContent = 'Ваше фото · проверьте читаемость этикетки';
  updateControls('preview');
}
$('upload').onclick = $('replace-photo').onclick = () => { $('file-input').value = ''; $('file-input').click(); };
$('file-input').onchange = e => selectPhoto(e.target.files[0]);
const dropzone = $('dropzone');
for (const name of ['dragenter','dragover']) dropzone.addEventListener(name, e => { e.preventDefault(); dropzone.classList.add('dragover'); });
for (const name of ['dragleave','drop']) dropzone.addEventListener(name, e => { e.preventDefault(); dropzone.classList.remove('dragover'); });
dropzone.addEventListener('drop', e => { if (!controller) selectPhoto(e.dataTransfer.files[0]); });
$('open-camera').onclick = async () => {
  notice('');
  if (!navigator.mediaDevices?.getUserMedia) { notice('Камера недоступна в этом браузере. Загрузите фото из галереи. Для камеры нужен HTTPS или localhost.'); return; }
  const generation = ++cameraGeneration; $('open-camera').disabled = true;
  try {
    const acquired = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1600 }, height: { ideal: 1200 } }, audio: false });
    if (generation !== cameraGeneration) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; $('camera-video').srcObject = stream; $('camera-video').hidden = false;
    await $('camera-video').play(); $('example-bottle').hidden = true; $('sample-label').hidden = true;
    $('photo-preview').hidden = true; updateControls('camera');
  } catch (err) { stopCamera(); notice(err.name === 'NotAllowedError' ? 'Доступ к камере не разрешён. Разрешите его в настройках браузера или загрузите фото.' : 'Не удалось включить камеру. Проверьте, что она свободна, или загрузите фото.'); }
  finally { $('open-camera').disabled = false; }
};
$('close-camera').onclick = () => { stopCamera(); $('photo-preview').hidden = !photo; $('example-bottle').hidden = !!photo; $('sample-label').hidden = !!photo; updateControls(photo ? 'preview' : 'initial'); };
$('capture').onclick = () => {
  const video = $('camera-video'); if (!video.videoWidth) { notice('Камера ещё запускается. Попробуйте через секунду.'); return; }
  const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => { if (blob) selectPhoto(new File([blob], 'wine-label.jpg', { type:'image/jpeg' })); }, 'image/jpeg', .92);
};
function safeImage(url) { try { const u = new URL(url, location.origin); return ['http:','https:'].includes(u.protocol) ? u.href : ''; } catch { return ''; } }
function resolveImageUrl(imageUrl, fileName) {
  const raw = String(imageUrl || fileName || '').trim();
  if (!raw) return '';
  if (/^https?:\/\//i.test(raw) || raw.startsWith('/')) return safeImage(raw);
  const base = window.SCANNER_CONFIG?.imageBaseUrl;
  if (!base) return '';
  try {
    const relative = raw.replace(/^\/?(?:uploads\/)?/, '').split('/').map(encodeURIComponent).join('/');
    return safeImage(new URL(relative, base).href);
  } catch { return ''; }
}
function formatRating(value) {
  const rating = Number(value);
  if (!Number.isFinite(rating) || rating < 0 || rating > 5) return '';
  return `${rating.toFixed(1).replace(/\.0$/, '')} / 5`;
}
$('recognize').onclick = async () => {
  if (!photo || controller) return;
  const endpoint = window.SCANNER_CONFIG?.recognitionEndpoint;
  if (!endpoint) { notice('Фото готово. Распознавание ещё не подключено — снимок никуда не отправлен. Пока можно открыть пример карточки ниже.'); return; }
  notice(''); controller = new AbortController(); const active = controller;
  const timer = setTimeout(() => active.abort('timeout'), 30000);
  $('processing').hidden = false;
  try {
    const body = new FormData(); body.append('image', photo);
    const response = await fetch(endpoint, { method:'POST', body, signal:active.signal });
    if (!response.ok) throw new Error('service');
    const result = await response.json();
    if (result.status === 'unknown' || result.status === 'uncertain') { notice('Не удалось определить вино точно. Снимите этикетку ближе, без бликов, и попробуйте ещё раз.'); return; }
    if (result.status !== 'matched' || !result.wine?.name || !(result.wine.slug || result.slug)) throw new Error('contract');
    showWine({ ...result.wine, slug:result.wine.slug || result.slug, image_url:result.wine.image_url || result.wine.imageUrl || result.wine.photo_name, demo:false });
  } catch (err) {
    notice(active.signal.aborted ? (active.signal.reason === 'timeout' ? 'Поиск занял слишком много времени. Попробуйте ещё раз.' : 'Поиск отменён. Можно выбрать другое фото.') : 'Сервис распознавания сейчас недоступен или вернул неполную карточку. Попробуйте позже.');
  } finally { clearTimeout(timer); controller = null; $('processing').hidden = true; }
};
$('cancel-request').onclick = () => controller?.abort('user');
function showWine(wine) {
  stopCamera(); currentWine = wine; $('scanner-screen').hidden = true; $('result-screen').hidden = false;
  const image = resolveImageUrl(wine.image_url, wine.photo_name || wine.image_name || wine.photoName);
  const rating = formatRating(wine.public_rating);
  const dishes = Array.isArray(wine.dishes) ? wine.dishes.filter(Boolean) : [];
  $('result-screen').innerHTML = `
    <div class="result-top"><button class="text-button" id="back-to-scanner">${icon('arrow-left')} К сканеру</button><span class="demo-badge">${wine.demo ? 'Пример карточки · не результат сканирования' : 'Вино найдено'}</span></div>
    <div class="result-layout"><div class="wine-portrait">${image ? `<img src="${escape(image)}" alt="${escape(wine.name)}"/>` : '<span>Фото пока нет</span>'}<span class="portrait-caption">СВОЁ ВИНО · РОССИЙСКИЕ ВИНОДЕЛЬНИ</span></div>
    <div class="wine-details"><p class="eyebrow">${escape(wine.winery)}</p><h1>${escape(wine.name)}</h1><p class="wine-category">${escape(wine.category || '')}${wine.region ? ' · ' + escape(wine.region) : ''}</p>
    <div class="rating-budget"><div class="data-block"><span>НАРОДНЫЙ РЕЙТИНГ</span><strong>${escape(rating || 'Пока нет данных')}</strong><small>${rating ? 'По публичному каталогу «Своё Вино»' : 'Рейтинг появится с источником'}</small></div><div class="data-block"><label for="shelf-price">ВАША ЦЕНА С ПОЛКИ</label><div class="price-input"><input id="shelf-price" inputmode="numeric" type="number" min="0" max="1000000" step="1" placeholder="Укажите цену" aria-label="Цена с полки в рублях"/><span>₽</span></div><small>Можно сохранить вместе с вином</small></div></div>
    <p class="wine-summary">${escape(wine.summary || wine.description || 'Описание пока не добавлено.')}</p><div class="taste-tags">${(Array.isArray(wine.aromas) ? wine.aromas : []).map(tag => `<span>${escape(tag)}</span>`).join('')}</div><p class="source-note">${escape(wine.source || 'Данные сервиса распознавания')}</p>${dishes.length ? `<p class="source-note">Сочетания: ${dishes.map(dish => escape(dish)).join(', ')}</p>` : ''}
    <div class="wine-facts"><span><small>СОРТ ВИНОГРАДА</small>${escape(Array.isArray(wine.grapes) ? wine.grapes.join(', ') : wine.grapes || 'Не указан')}</span><span><small>РЕГИОН</small>${escape(wine.region || 'Не указан')}</span>${wine.temperature ? `<span><small>ПОДАВАТЬ</small>${escape(wine.temperature)}</span>` : ''}${wine.alcohol ? `<span><small>КРЕПОСТЬ</small>${escape(wine.alcohol)}</span>` : ''}</div>
    <div class="result-actions"><button class="button primary" id="save-wine">${icon('bookmark')} Сохранить вино</button><button class="button secondary" id="scan-again">${icon('camera')} Ещё одно вино</button></div></div></div>
    ${wine.demo ? `<section class="pairings"><div class="pairings-heading">${icon('utensils')}<h2>Что у вас на ужин?</h2></div><p class="muted">Выберите блюдо — подскажем, как оно сочетается с этим стилем вина.</p><div class="food-buttons"><button data-food="fish" aria-pressed="true">Рыба и морепродукты</button><button data-food="cheese" aria-pressed="false">Мягкий сыр</button><button data-food="salad" aria-pressed="false">Лёгкий салат</button><button data-food="steak" aria-pressed="false">Стейк</button><button data-food="dessert" aria-pressed="false">Десерт</button></div><p class="pairing-explanation" id="pairing-explanation"></p><p class="source-note">Общая рекомендация для стиля «белое игристое брют», а не экспертная оценка этой бутылки.</p></section>` : ''}
    <details class="description"><summary>Полное описание вина</summary><p>${escape(wine.description || 'Описание пока не добавлено.')}</p></details>`;
  $('back-to-scanner').onclick = $('scan-again').onclick = backToScanner;
  const existing = saved.find(w => w.slug === wine.slug); if (existing?.price != null) $('shelf-price').value = existing.price;
  $('save-wine').onclick = saveWine;
  if (wine.demo) {
    const explanations = {
      fish:'Хорошая пара: свежесть брюта поддержит нежный вкус рыбы и морепродуктов. Выберите лёгкую подачу с лимоном, без сладкого или тяжёлого соуса.',
      cheese:'Попробуйте с мягким сливочным сыром: свежесть и пузырьки помогут уравновесить его нежную, жирную текстуру.',
      salad:'Подойдёт к лёгкому салату с зеленью и нежным сыром. Сильно кислую заправку лучше добавлять умеренно.',
      steak:'Для насыщенного стейка это более лёгкая пара. Если хочется поддержать жареный вкус мяса, рассмотрите более полнотелое вино.',
      dessert:'Сладкий десерт может сделать сухое вино ощутимо кислее. Для десерта лучше рассмотреть вино с большей сладостью.'
    };
    $('pairing-explanation').textContent = explanations.fish;
    document.querySelectorAll('[data-food]').forEach(btn => btn.onclick = () => { document.querySelectorAll('[data-food]').forEach(b => b.setAttribute('aria-pressed', b === btn)); $('pairing-explanation').textContent = explanations[btn.dataset.food]; });
  }
  $('result-screen').focus({ preventScroll:true }); window.scrollTo({ top:0, behavior:'instant' });
}
function backToScanner() { $('result-screen').hidden = true; $('scanner-screen').hidden = false; $('show-example').focus({ preventScroll:true }); window.scrollTo({ top:0, behavior:'instant' }); }
$('show-example').onclick = () => showWine(example);
function persist() { try { localStorage.setItem('svoe-wines', JSON.stringify(saved)); syncSavedCount(); return true; } catch { toast('Не удалось сохранить: хранилище браузера недоступно.'); return false; } }
function saveWine() {
  const input = $('shelf-price'); if (!input.checkValidity()) { input.reportValidity(); return; }
  const price = input.value === '' ? null : Number(input.value);
  const item = { ...currentWine, price }; const previous = [...saved];
  saved = [item, ...saved.filter(w => w.slug !== item.slug)].slice(0,100);
  if (persist()) { $('save-wine').innerHTML = icon('check') + ' Сохранено'; toast('Вино сохранено в «Мои вина»'); } else saved = previous;
}
function renderSaved() {
  $('saved-list').innerHTML = saved.length ? saved.map((w,i) => { const image = resolveImageUrl(w.image_url, w.photo_name || w.image_name || w.photoName); return `<div class="saved-item">${image ? `<img src="${escape(image)}" alt=""/>` : ''}<button data-saved="${i}">${escape(w.name)}<small>${escape(w.winery)}${w.price != null ? ' · ' + escape(w.price) + ' ₽' : ''}${w.demo ? ' · пример' : ''}</small></button><button class="remove-saved" data-remove="${i}" aria-label="Удалить ${escape(w.name)}">Удалить</button></div>`; }).join('') : '<p class="muted">Здесь будут вина, к которым захочется вернуться. Откройте карточку и нажмите «Сохранить вино».</p>';
  document.querySelectorAll('[data-saved]').forEach(b => b.onclick = () => { $('saved-dialog').close(); showWine(saved[Number(b.dataset.saved)]); });
  document.querySelectorAll('[data-remove]').forEach(b => b.onclick = () => { const previous = [...saved]; saved.splice(Number(b.dataset.remove),1); if (!persist()) saved = previous; renderSaved(); });
}
$('saved-open').onclick = () => { renderSaved(); $('saved-dialog').showModal(); };
$('saved-close').onclick = () => $('saved-dialog').close();
$('saved-dialog').addEventListener('click', e => { if (e.target === $('saved-dialog')) { const rect = e.target.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) e.target.close(); } });
window.addEventListener('pagehide', () => { stopCamera(); controller?.abort('user'); if (photoUrl) URL.revokeObjectURL(photoUrl); });
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) $('close-camera').click(); });
