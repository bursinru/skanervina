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
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 5 5"/>',
  compare: '<rect x="3" y="5" width="7" height="14" rx="1"/><rect x="14" y="5" width="7" height="14" rx="1"/>',
  x: '<path d="m6 6 12 12M6 18 18 6"/>',
  check: '<path d="m5 12 4 4L19 6"/>'
};
const icon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.wine}</svg>`;
document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const catalogAssets = {
  region: {
    'крым': '/assets/catalog/region-krym.webp',
    'кубань': '/assets/catalog/region-kuban.webp'
  },
  grape: {
    'мальвазия': '/assets/catalog/grape-malvaziya.webp',
    'первенец магарача': '/assets/catalog/grape-pervenets-magaracha.webp'
  },
  dish: {
    'блюда из птицы': '/assets/catalog/dish-poultry.webp',
    'птица': '/assets/catalog/dish-poultry.webp',
    'овощи гриль': '/assets/catalog/dish-grilled-vegetables.webp',
    'овощи на гриле': '/assets/catalog/dish-grilled-vegetables.webp',
    'легкие закуски': '/assets/catalog/dish-light-snacks.webp'
  }
};
const normalizeCatalogValue = value => String(value ?? '').trim().toLocaleLowerCase('ru-RU').replace(/ё/g, 'е').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
function catalogAsset(type, value) {
  const normalized = normalizeCatalogValue(value);
  if (!normalized) return '';
  const assets = catalogAssets[type] || {};
  const key = Object.keys(assets).find(candidate => normalized === candidate || normalized.includes(candidate) || candidate.includes(normalized));
  return key ? safeImage(assets[key]) : '';
}
function inferWineColor(wine) {
  const value = normalizeCatalogValue(`${wine.color || ''} ${wine.category || ''}`);
  if (value.includes('красн')) return 'Красное';
  if (value.includes('оранж')) return 'Оранжевое';
  if (value.includes('розов')) return 'Розовое';
  if (value.includes('бел')) return 'Белое';
  return wine.color || '';
}
function colorTone(color) {
  const value = normalizeCatalogValue(color);
  if (value.includes('красн')) return 'red';
  if (value.includes('оранж')) return 'orange';
  if (value.includes('розов')) return 'rose';
  return 'white';
}
const tasteDimensions = [
  { key: 'sweetness', label: 'Сладость' },
  { key: 'acidity', label: 'Кислотность' },
  { key: 'aromaticity', label: 'Ароматичность' },
  { key: 'body', label: 'Тело' }
];
const clampTaste = value => Math.max(1, Math.min(5, Math.round(Number(value) || 1)));
const tastingNoteRules = [
  [/лимон|лайм|цитрус|грейпфрут/, 'Цитрус'],
  [/персик|абрикос|нектарин/, 'Персик'],
  [/яблок|груш/, 'Яблоко и груша'],
  [/цветоч|белых цветов|роза|лаванд/, 'Белые цветы'],
  [/миндал|орех|пралине/, 'Миндаль'],
  [/ягод|вишн|слив|смород/, 'Красные ягоды'],
  [/ванил|дуб|карамел/, 'Ваниль и дуб'],
  [/трав|зелён|зелен|мят|перец/, 'Травы и специи']
];
function extractAromaNotes(wine) {
  const explicit = Array.isArray(wine.aromas) ? wine.aromas.map(item => String(item).trim()).filter(Boolean) : [];
  if (explicit.length) return [...new Set(explicit)].slice(0, 6);
  const text = normalizeCatalogValue(`${wine.name || ''} ${wine.grapes || ''} ${wine.description || ''}`);
  const notes = tastingNoteRules.filter(([pattern]) => pattern.test(text)).map(([, label]) => label);
  if (notes.length) return [...new Set(notes)].slice(0, 6);
  const color = inferWineColor(wine);
  return colorTone(color) === 'red' ? ['Красные ягоды'] : colorTone(color) === 'orange' ? ['Цедра и специи'] : colorTone(color) === 'rose' ? ['Ягоды'] : ['Свежие фрукты'];
}
function alcoholNumber(value) {
  const match = String(value || '').replace(',', '.').match(/\d+(?:\.\d+)?/);
  return match ? Number(match[0]) : 0;
}
function deriveTasteProfile(wine) {
  const supplied = wine.taste_profile || wine.tasteProfile;
  const text = normalizeCatalogValue(`${wine.color || ''} ${wine.category || ''} ${wine.grapes || ''} ${wine.description || ''}`);
  const notes = extractAromaNotes(wine);
  let sweetness = 1;
  if (text.includes('сладк')) sweetness = 5;
  else if (text.includes('полуслад')) sweetness = 3;
  else if (text.includes('полусух')) sweetness = 2;
  else if (text.includes('сух') || text.includes('брют')) sweetness = 1;
  let acidity = /белое/.test(text) ? 4 : /розовое/.test(text) ? 3 : 3;
  if (/освеж|ярк.*кислот|высок.*кислот|лимон|цитрус|грейпфрут/.test(text)) acidity += 1;
  if (/мягк|округл|сливоч|низк.*кислот/.test(text)) acidity -= 1;
  let aromaticity = 1 + Math.min(4, notes.length);
  if (/яркий аромат|насыщен|интенсив/.test(text)) aromaticity += 1;
  let body = 3;
  if (/игрист|брют/.test(text)) body -= 1;
  if (/красное|оранжевое|плотн|полнот|маслян|сливоч/.test(text)) body += 1;
  const alcohol = alcoholNumber(wine.alcohol);
  if (alcohol >= 13.5) body += 1;
  if (alcohol > 0 && alcohol <= 11) body -= 1;
  const calculated = { sweetness, acidity, aromaticity, body };
  return Object.fromEntries(tasteDimensions.map(({ key }) => [key, clampTaste(Number(supplied?.[key]) || calculated[key])]));
}
function getPreferenceProfile() {
  const profileWines = [...saved];
  if (currentWine && !profileWines.some(wine => wine.slug === currentWine.slug)) profileWines.push(currentWine);
  const rated = profileWines.map(wine => ({ wine, rating: Number(ratings[wine.slug]) })).filter(item => Number.isInteger(item.rating) && item.rating >= 1 && item.rating <= 5);
  if (!rated.length) return null;
  const weight = rated.reduce((sum, item) => sum + item.rating, 0);
  const values = Object.fromEntries(tasteDimensions.map(({ key }) => [key, rated.reduce((sum, item) => sum + deriveTasteProfile(item.wine)[key] * item.rating, 0) / weight]));
  return { values, count: rated.length };
}
function calculateFit(wine, preference = getPreferenceProfile()) {
  if (!preference) return null;
  const profile = deriveTasteProfile(wine);
  const difference = tasteDimensions.reduce((sum, { key }) => sum + Math.abs(profile[key] - preference.values[key]) / 4, 0) / tasteDimensions.length;
  const base = 100 - difference * 100;
  return Math.max(0, Math.min(100, Math.round(base)));
}
function fitMarkupFor(wine) {
  const preference = getPreferenceProfile();
  const fit = calculateFit(wine, preference);
  if (fit === null) return '';
  return `<section class="fit-card"><strong>${fit}%</strong><div><small>ПОДХОДИТ ВАМ</small><b>Похоже на то, что вы любите</b><p>На основе ${preference.count} ${preference.count === 1 ? 'вашей оценки' : preference.count < 5 ? 'ваших оценок' : 'оценок в вашем профиле'} и вкусового профиля вина.</p></div></section>`;
}
const example = {
  slug: 'fanagoriya-blanc-de-blancs-shardone-beloe-bryut-12', name: 'Blanc de Blancs', winery: 'Фанагория',
  category: 'Игристое белое · Брют', color: 'Белое', region: 'Кубань', grapes: ['Шардоне'], image_url: '/assets/blanc-de-blancs.webp',
  summary: 'Свежее, изящное вино с нотами белых цветов, персика и грейпфрута. Сбалансированная кислотность и лёгкая сливочность во вкусе.',
  description: 'Игра пузырьков изящная и утонченная — непрерывный жемчужный перляж. Аромат элегантный и тонкий, завораживающий. Вначале раскрывается нотами легких белых цветов и прохладой цветущего утреннего сада. Затем проявляются сочные оттенки мякоти белых фруктов и розового персика, свежие нотки грейпфрута и легкий травяной акцент. Фоном звучит тонкий оттенок миндального пралине. Вкус изящный, свежий и ажурный. Прекрасно сбалансированная, стройная кислотность отлично гармонирует с тонкой сливочностью и ведет к чистому и стройному послевкусию. Идеальный аперитив.',
  aromas: ['Белые цветы', 'Персик', 'Грейпфрут', 'Миндаль'], demo: true
};
let stream = null, cameraGeneration = 0, photo = null, photoUrl = null, controller = null, currentWine = null, toastTimer;
let processingStartedAt = 0, processingTimer = null;
let saved = [];
try { const raw = JSON.parse(localStorage.getItem('svoe-wines') || '[]'); if (Array.isArray(raw)) saved = raw.filter(w => w && typeof w.slug === 'string' && typeof w.name === 'string').slice(0,100); } catch {}
let ratings = {};
try { const raw = JSON.parse(localStorage.getItem('svoe-ratings') || '{}'); if (raw && typeof raw === 'object') ratings = raw; } catch {}
const profileEndpoint = window.SCANNER_CONFIG?.profileEndpoint;
let profileReady = !profileEndpoint;
let profileWrites = Promise.resolve();
if (profileEndpoint) {
  try {
    const response = await fetch(profileEndpoint, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error('profile');
    const state = await response.json();
    saved = state.saved;
    ratings = state.ratings;
    profileReady = true;
    document.querySelector('#saved-dialog > .muted').textContent = 'Хранятся на сервере. Доступ привязан к этому браузеру.';
  } catch { toast('Сервер сохранений недоступен. Изменения останутся в браузере; обновите страницу для подключения.'); }
}
function syncProfile() {
  if (!profileEndpoint || !profileReady) return;
  const body = JSON.stringify({ saved, ratings });
  profileWrites = profileWrites.then(async () => {
    const response = await fetch(profileEndpoint, {
      method: 'PUT', headers: { 'Content-Type': 'application/json', 'X-Scanner-Client': 'web' },
      body, signal: AbortSignal.timeout(10000)
    });
    if (!response.ok) throw new Error('save');
  }).catch(() => toast('Не удалось записать изменения на сервер. Копия осталась в браузере. Повторите сохранение.'));
}
let comparison = [];
try { const raw = JSON.parse(localStorage.getItem('svoe-comparison') || '[]'); if (Array.isArray(raw)) comparison = raw.filter(wine => wine && typeof wine.slug === 'string' && typeof wine.name === 'string').slice(0, 3); } catch {}
const syncSavedCount = () => $('saved-count').textContent = saved.length;
syncSavedCount();
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function toast(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => $('toast').hidden = true, 3500); }
function isSaved(wine) { return saved.some(item => item.slug === wine.slug); }
function isCompared(wine) { return comparison.some(item => item.slug === wine.slug); }
function persistComparison() { try { localStorage.setItem('svoe-comparison', JSON.stringify(comparison)); return true; } catch { toast('Сравнение показано только до закрытия страницы.'); return false; } }
function syncSaveButton() {
  const button = $('save-wine');
  if (!button || !currentWine) return;
  const savedState = isSaved(currentWine);
  button.classList.toggle('is-saved', savedState);
  button.setAttribute('aria-pressed', String(savedState));
  button.setAttribute('aria-label', savedState ? 'Убрать из сохранённых' : 'Сохранить вино');
  button.title = savedState ? 'Убрать из сохранённых' : 'Сохранить вино';
  button.innerHTML = icon('bookmark');
}
function syncCompareButton() {
  const button = $('compare-wine');
  if (!button || !currentWine) return;
  const compared = isCompared(currentWine);
  button.classList.toggle('is-selected', compared);
  button.setAttribute('aria-pressed', String(compared));
  button.innerHTML = icon('compare') + ' Сравнить';
}
function renderCompareTray() {
  const tray = $('compare-tray');
  const headerButton = $('compare-open');
  if (!tray || !headerButton) return;
  $('compare-count').textContent = comparison.length;
  $('compare-tray-count').textContent = comparison.length;
  $('compare-tray-copy').textContent = comparison.length === 1 ? 'Выбрано одно вино' : 'Сравните вкус, рейтинг и характеристики';
  tray.hidden = comparison.length === 0;
  headerButton.hidden = comparison.length === 0;
  headerButton.setAttribute('aria-label', `Сравнение: ${comparison.length}`);
}
function renderCompareDialog() {
  const list = $('compare-list');
  if (!list) return;
  const preference = getPreferenceProfile();
  list.innerHTML = comparison.length ? comparison.map((wine, index) => {
    const image = resolveImageUrl(wine.image_url, wine.photo_name || wine.image_name || wine.photoName);
    const fit = calculateFit(wine, preference);
    const profile = deriveTasteProfile(wine);
    return `<article class="compare-card"><button class="compare-remove" type="button" data-compare-remove="${index}" aria-label="Убрать ${escape(wine.name)} из сравнения">${icon('x')}</button>${image ? `<img src="${escape(image)}" alt="" loading="lazy">` : '<span class="compare-card-placeholder">Фото<br>нет</span>'}<small>${escape(wine.winery)}</small><h3>${escape(wine.name)}</h3><strong class="compare-fit">${fit === null ? '—' : `${fit}%`}<small>${fit === null ? 'оцените вина' : 'подходит вам'}</small></strong><div class="compare-bars">${tasteDimensions.map(({ key, label }) => `<span><b>${escape(label)}</b><i><em style="width:${profile[key] * 20}%"></em></i></span>`).join('')}</div></article>`;
  }).join('') : '<p class="muted">Здесь появятся выбранные вина для сравнения.</p>';
  document.querySelectorAll('[data-compare-remove]').forEach(button => button.onclick = () => {
    comparison.splice(Number(button.dataset.compareRemove), 1);
    persistComparison();
    renderCompareTray();
    renderCompareDialog();
    syncCompareButton();
  });
}
function toggleCompare(wine) {
  const index = comparison.findIndex(item => item.slug === wine.slug);
  if (index >= 0) {
    comparison.splice(index, 1);
    toast('Вино убрано из сравнения');
  } else if (comparison.length >= 3) {
    toast('Можно выбрать максимум три вина');
    return;
  } else {
    comparison.push({ ...wine });
    toast('Вино добавлено в сравнение');
  }
  persistComparison();
  renderCompareTray();
  syncCompareButton();
}
function stopCamera() { cameraGeneration++; stream?.getTracks().forEach(t => t.stop()); stream = null; $('camera-video').srcObject = null; $('camera-video').hidden = true; $('open-camera').disabled = false; }
function updateControls(mode) {
  ['initial','preview','camera'].forEach(name => $(name + '-actions').hidden = name !== mode);
  const title = $('control-title');
  const description = $('control-description');
  const fileNote = $('file-note');
  if (title) title.textContent = { initial:'Начнём с этикетки', preview:'Этикетка хорошо видна?', camera:'Поймайте этикетку в рамку' }[mode];
  if (description) description.textContent = { initial:'Сфотографируйте бутылку или выберите снимок.', preview:'Система автоматически выделит этикетку перед поиском.', camera:'Держите телефон ровно и избегайте бликов.' }[mode];
  if (fileNote) fileNote.hidden = mode !== 'initial';
}
async function selectPhoto(file) {
  if (!file || controller) return;
  notice('');
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) { notice('Выберите JPG, PNG или WebP. Для HEIC сохраните фотографию в JPG.'); return; }
  if (file.size > 15 * 1024 * 1024) { notice('Фотография слишком большая. Выберите файл до 15 МБ.'); return; }
  const candidateUrl = URL.createObjectURL(file);
  try {
    const img = new Image(); img.src = candidateUrl; await img.decode();
    if (!img.naturalWidth || !img.naturalHeight || img.naturalWidth * img.naturalHeight > 24000000) throw new Error('Invalid image');
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
function safeImage(url) { if (!url) return ''; try { const u = new URL(url, location.origin); return ['http:','https:'].includes(u.protocol) ? u.href : ''; } catch { return ''; } }
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
const ratingGlass = () => '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="rating-glass-body" d="M7 3h10l1 7a6 6 0 0 1-12 0z"/><path d="M7 3h10l1 7a6 6 0 0 1-12 0zM12 16v5M8 21h8"/></svg>';
const queryParams = new URLSearchParams(location.search);
const debugMode = queryParams.get('admin') === '1' || queryParams.get('debug') === '1';
$('admin-tools').hidden = !debugMode;
$('processing-debug').hidden = !debugMode;
function formatElapsed(milliseconds) {
  const value = Number(milliseconds);
  return Number.isFinite(value) && value >= 0 ? `${(value / 1000).toFixed(1).replace('.', ',')} с` : '—';
}
function addProcessingLog(message) {
  if (!debugMode) return;
  const log = $('processing-log');
  if (!log) return;
  const item = document.createElement('li');
  item.textContent = `${formatElapsed(performance.now() - processingStartedAt)} · ${message}`;
  log.append(item);
  log.scrollTop = log.scrollHeight;
}
function updateProcessingElapsed() {
  const elapsed = $('processing-elapsed');
  if (elapsed) elapsed.textContent = `Прошло ${formatElapsed(performance.now() - processingStartedAt)}`;
}
function startProcessingLog() {
  processingStartedAt = performance.now();
  clearInterval(processingTimer);
  if (!debugMode) return;
  $('processing-log').replaceChildren();
  updateProcessingElapsed();
  processingTimer = setInterval(updateProcessingElapsed, 100);
  addProcessingLog('Фото принято');
  addProcessingLog('Запрос отправлен в распознавание');
  addProcessingLog('Сервер выполняет автоматический crop и поиск по изображению');
}
function stopProcessingLog() {
  clearInterval(processingTimer);
  processingTimer = null;
  updateProcessingElapsed();
}
function renderScanDebug({ result, elapsed }) {
  if (!debugMode) return '';
  const metrics = result?.recognition || {};
  const timings = metrics.timings_ms || {};
  const similarity = Number.isFinite(metrics.similarity) ? metrics.similarity : Number.isFinite(result?.confidence) ? result.confidence : null;
  const score = similarity === null ? '—' : `${(similarity * 100).toFixed(1).replace('.', ',')}%`;
  const serverTotal = Number(timings.total);
  const timingValues = {
    decode: Number(timings.decode),
    labelDetection: Number(timings.label_detection),
    labelCrop: Number(timings.label_crop),
    enhancement: Number(timings.enhancement),
    visual: Number(timings.visual),
    ocr: Number(timings.ocr),
  };
  const timing = value => Number.isFinite(value) && value >= 0 ? formatElapsed(value) : '—';
  const measured = Object.values(timingValues).filter(value => Number.isFinite(value) && value >= 0);
  const measuredTotal = measured.reduce((sum, value) => sum + value, 0);
  const backendOther = Number.isFinite(serverTotal) ? Math.max(0, serverTotal - measuredTotal) : null;
  const browserOther = Number.isFinite(serverTotal) ? Math.max(0, elapsed - serverTotal) : null;
  const status = result?.status === 'matched' ? 'Найдено' : result?.status === 'uncertain' ? 'Нужно проверить' : 'Не найдено';
  const scoreNote = similarity === null ? 'метрика недоступна' : 'сходство, не вероятность';
  const detailRows = [
    ['Декодирование фото', timingValues.decode, 'открытие, EXIF и приведение к RGB'],
    ['Поиск этикетки', timingValues.labelDetection, 'автоматический поиск области для crop'],
    ['Вырезание crop', timingValues.labelCrop, 'вырезание найденной области'],
    ['Подготовка изображения', timingValues.enhancement, 'контраст/резкость для enhanced-режима'],
    ['Изображение', timingValues.visual, 'SigLIP-вектор + поиск ближайших в каталоге'],
    ...(metrics.ocr && metrics.ocr !== 'skipped' ? [['OCR', timingValues.ocr, 'Tesseract и сопоставление распознанного текста']] : []),
    ['Остальное backend', backendOther, 'сборка ответа и операции, не выделенные отдельно'],
    ['Загрузка, сеть и браузер', browserOther, 'разница между полным ожиданием и backend'],
  ];
  const detailMarkup = detailRows.map(([label, value, note]) => `<div class="scan-debug-detail-row"><div><strong>${label}</strong><small>${note}</small></div><b>${timing(value)}</b></div>`).join('');
  return `<section class="scan-debug" aria-label="Диагностика сканирования"><div class="scan-debug-heading"><div><small>DEBUG · РЕЗУЛЬТАТ СКАНИРОВАНИЯ</small><h2>Технические показатели</h2></div><span class="scan-debug-status">${escape(status)}</span></div><div class="scan-debug-metrics"><div><small>СОВПАДЕНИЕ</small><strong>${score}</strong><em>${scoreNote}</em></div><div><small>СКОРОСТЬ</small><strong>${formatElapsed(elapsed)}</strong><em>полное ожидание в браузере</em></div><div><small>СЕРВЕР</small><strong>${timing(serverTotal)}</strong><em>распознавание backend</em></div></div><div class="scan-debug-details"><h3>Разбивка времени</h3>${detailMarkup}</div><p class="scan-debug-footnote">Поиск идёт по изображению (SigLIP). OCR в обычном режиме выключен. Время «Загрузка, сеть и браузер» — расчётная разница, а не отдельный замер.</p></section>`;
}
function showDiagnostics(result, elapsed) {
  if (!debugMode) return;
  const details = document.createElement('details'); details.className = 'admin-diagnostics'; details.open = true;
  const summary = document.createElement('summary'); summary.textContent = 'Диагностика debug'; details.append(summary);
  const metrics = result.recognition || {};
  const timings = metrics.timings_ms || {};
  const serverTotal = Number(timings.total);
  const clientOverhead = Number.isFinite(serverTotal) ? Math.max(0, elapsed - serverTotal) : null;
  const score = metrics.similarity;
  const lines = [
    `Статус: ${result.status}`,
    `Сходство изображений: ${Number.isFinite(score) ? (score * 100).toFixed(1) + '%' : 'не измерено (OCR)'}`,
    'Сходство — косинусная оценка, не вероятность правильного ответа.',
    `Полное ожидание: ${(elapsed / 1000).toFixed(2)} с`,
    `Обработка на сервере: ${formatElapsed(serverTotal)}`,
    `Разница загрузки/сети/браузера: ${formatElapsed(clientOverhead)}`,
    `Время этапов: декодирование ${formatElapsed(Number(timings.decode))} · поиск этикетки ${formatElapsed(Number(timings.label_detection))} · crop ${formatElapsed(Number(timings.label_crop))} · подготовка ${formatElapsed(Number(timings.enhancement))} · изображение ${formatElapsed(Number(timings.visual))} · OCR ${formatElapsed(Number(timings.ocr))}`,
    `OCR: ${metrics.ocr || metrics.reason || '—'} · подтверждает результат: ${metrics.ocr_corroborated ? 'да' : 'нет'}`,
    `Оценка текстового поиска: ${Number.isFinite(metrics.ocr_best?.score) ? (metrics.ocr_best.score * 100).toFixed(1) + '%' : '—'}`,
    `Отрыв от следующего: ${Number.isFinite(metrics.margin) ? (metrics.margin * 100).toFixed(2) + ' п.п.' : '—'}`,
    `Автоматическое выделение: ${metrics.label_detection?.confidence ? `готово · ${(metrics.label_detection.confidence * 100).toFixed(1)}%` : '—'}`,
  ];
  const text = document.createElement('pre'); text.textContent = lines.join('\n'); details.append(text);
  document.querySelector('.admin-diagnostics')?.remove();
  if (result.wine) $('result-screen').querySelector('.result-top').after(details);
  else $('admin-tools').append(details);
}
$('recognize').onclick = async () => {
  if (!photo || controller) return;
  const endpoint = window.SCANNER_CONFIG?.recognitionEndpoint;
  if (!endpoint) { notice('Фото готово. Распознавание ещё не подключено — снимок никуда не отправлен. Пока можно открыть пример карточки ниже.'); return; }
  notice(''); controller = new AbortController(); const active = controller;
  const timer = setTimeout(() => active.abort('timeout'), 30000);
  const started = performance.now();
  $('scanning-label').src = photoUrl;
  $('processing-status').textContent = 'Идёт поиск по каталогу…';
  startProcessingLog();
  $('processing').hidden = false;
  try {
    const body = new FormData(); body.append('image', photo);
    const headers = debugMode ? { 'X-Scanner-Debug': '1' } : {};
    const response = await fetch(endpoint, { method:'POST', headers, body, signal:active.signal });
    if (response.status === 403) { notice('Сервер отклонил debug-запрос.'); return; }
    if (!response.ok) throw new Error('service');
    const result = await response.json();
    const elapsed = performance.now() - started;
    addProcessingLog(`Ответ сервера получен за ${formatElapsed(elapsed)}`);
    if (result.status === 'unknown') {
      notice('Не удалось найти достаточно похожую этикетку. Переснимите бутылку крупнее, без бликов и соседних бутылок.');
      showDiagnostics(result, elapsed);
      return;
    }
    if (!['matched', 'uncertain'].includes(result.status) || !result.wine?.name || !(result.wine.slug || result.slug)) throw new Error('contract');
    showWine({ ...result.wine, slug:result.wine.slug || result.slug, image_url:result.wine.image_url || result.wine.imageUrl || result.wine.photo_name, demo:false, uncertain:result.status === 'uncertain' }, { result, elapsed });
  } catch (err) {
    notice(active.signal.aborted ? (active.signal.reason === 'timeout' ? 'Поиск занял слишком много времени. Попробуйте ещё раз.' : 'Поиск отменён. Можно выбрать другое фото.') : 'Сервис распознавания сейчас недоступен или вернул неполную карточку. Попробуйте позже.');
  } finally { clearTimeout(timer); controller = null; stopProcessingLog(); $('processing').hidden = true; }
};
$('cancel-request').onclick = () => controller?.abort('user');
function showWine(wine, scanMeta = null) {
  stopCamera(); currentWine = wine; $('scanner-screen').hidden = true; $('result-screen').hidden = false;
  const image = resolveImageUrl(wine.image_url, wine.photo_name || wine.image_name || wine.photoName);
  const rating = formatRating(wine.public_rating);
  const dishes = Array.isArray(wine.dishes) ? wine.dishes.filter(Boolean) : [];
  const grapes = Array.isArray(wine.grapes) ? wine.grapes.filter(Boolean) : (wine.grapes ? [wine.grapes] : []);
  const wineColor = inferWineColor(wine);
  const regionImage = catalogAsset('region', wine.region) || safeImage(wine.region_image_url);
  const grapeImageUrls = Array.isArray(wine.grape_image_urls) ? wine.grape_image_urls : [];
  const grapeImages = grapes.map((grape, index) => ({ name: grape, image: catalogAsset('grape', grape) || safeImage(grapeImageUrls[index] || (index === 0 ? wine.grape_image_url : '')) })).filter(item => item.image);
  const tasteProfile = deriveTasteProfile(wine);
  const aromaNotes = extractAromaNotes(wine);
  const fitMarkup = fitMarkupFor(wine);
  const scanDebugMarkup = scanMeta ? renderScanDebug(scanMeta) : '';
  const visualFacts = [
    regionImage ? `<article class="visual-fact"><img src="${escape(regionImage)}" alt="" loading="lazy"><div><small>РЕГИОН</small><strong>${escape(wine.region)}</strong></div></article>` : '',
    ...grapeImages.map(item => `<article class="visual-fact"><img src="${escape(item.image)}" alt="" loading="lazy"><div><small>СОРТ ВИНОГРАДА</small><strong>${escape(item.name)}</strong></div></article>`),
    (wine.category || wineColor) ? `<article class="visual-fact visual-fact-color"><span class="color-dot color-dot-${colorTone(wineColor)}"></span><div><small>КАТЕГОРИЯ И ЦВЕТ</small><strong>${escape(wine.category || 'Категория не указана')}</strong>${wineColor ? `<em>${escape(wineColor)}</em>` : ''}</div></article>` : ''
  ].filter(Boolean).join('');
  const dishImageUrls = Array.isArray(wine.dish_image_urls) ? wine.dish_image_urls : [];
  const dishCards = dishes.map((dish, index) => {
    const dishImage = catalogAsset('dish', dish) || safeImage(dishImageUrls[index]);
    return `<article class="dish-card">${dishImage ? `<img src="${escape(dishImage)}" alt="" loading="lazy">` : '<span class="dish-placeholder">Вино и еда</span>'}<strong>${escape(dish)}</strong></article>`;
  }).join('');
  $('result-screen').innerHTML = `
    <div class="result-top"><button class="text-button" id="back-to-scanner">${icon('arrow-left')} К сканеру</button><span class="demo-badge">${wine.demo ? 'Пример карточки · не результат сканирования' : wine.uncertain ? 'Наиболее похожее · проверьте название' : 'Вино найдено'}</span></div>
    ${scanDebugMarkup}
    <div class="result-layout"><div class="wine-portrait">${image ? `<img src="${escape(image)}" alt="${escape(wine.name)}"/>` : '<span>Фото пока нет</span>'}<button class="portrait-save icon-action" id="save-wine" type="button" aria-label="Сохранить вино">${icon('bookmark')}</button><span class="portrait-caption">СВОЁ ВИНО · РОССИЙСКИЕ ВИНОДЕЛЬНИ</span></div>
    <div class="wine-details"><p class="eyebrow">${escape(wine.winery)}</p><h1>${escape(wine.name)}</h1><p class="wine-category">${escape(wine.category || '')}${wine.region ? ' · ' + escape(wine.region) : ''}</p>
    ${visualFacts ? `<section class="catalog-visuals" aria-label="Характеристики из каталога"><div class="visual-facts">${visualFacts}</div></section>` : ''}
    <section class="rating-card" aria-labelledby="user-rating-title"><div class="public-rating"><span class="public-rating-glass">${ratingGlass()}</span><div><small>НАРОДНЫЙ РЕЙТИНГ</small><strong>${escape(rating || '—')}</strong><em>${rating ? 'из 5 · по каталогу' : 'пока нет данных'}</em></div></div><span class="rating-divider" aria-hidden="true"></span><div class="inline-user-rating"><h2 id="user-rating-title">Ваша оценка</h2><div class="rating-options" role="radiogroup" aria-label="Оценка вина">${[1,2,3,4,5].map(value => `<button class="rating-option" type="button" role="radio" aria-checked="false" aria-label="${value} из 5" data-user-rating="${value}"><span class="rating-glass">${ratingGlass()}</span></button>`).join('')}</div></div></section><p class="rating-status" id="rating-status">Нажмите на бокал, чтобы оценить</p>
    ${fitMarkup}
    <section class="taste-profile" aria-labelledby="taste-profile-title"><div class="taste-profile-heading"><h2 id="taste-profile-title">Вкусовые свойства</h2><span>Оценка по данным каталога</span></div>${tasteDimensions.map(({ key, label }) => `<div class="taste-row"><span>${escape(label)}</span><i><em style="width:${tasteProfile[key] * 20}%"></em></i><b>${tasteProfile[key]}/5</b></div>`).join('')}</section>
    <p class="wine-summary">${escape(wine.summary || wine.description || 'Описание пока не добавлено.')}</p><div class="taste-tags">${aromaNotes.map(tag => `<span>${escape(tag)}</span>`).join('')}</div>
    <div class="wine-facts"><span><small>СОРТ ВИНОГРАДА</small>${escape(grapes.join(', ') || 'Не указан')}</span><span><small>РЕГИОН</small>${escape(wine.region || 'Не указан')}</span>${wine.temperature ? `<span><small>ПОДАВАТЬ</small>${escape(wine.temperature)}</span>` : ''}${wine.alcohol ? `<span><small>КРЕПОСТЬ</small>${escape(wine.alcohol)}</span>` : ''}</div>
    <div class="result-actions"><button class="button primary" id="compare-wine" type="button">${icon('compare')} Сравнить</button><button class="button secondary" id="scan-again">${icon('camera')} Сканировать следующее</button></div></div></div>
    ${dishes.length ? `<section class="dish-pairings"><div class="dish-pairings-heading"><div>${icon('utensils')}<h2>Сочетание с блюдами</h2></div><span>Из каталога «Своё Вино»</span></div><div class="dish-grid">${dishCards}</div></section>` : ''}
    <section class="after-search" id="after-search">
      <div class="dish-pairings-heading"><div>${icon('compare')}<h2>Российские аналоги</h2></div><span>Другие винодельни, похожий стиль</span></div>
      <p class="muted" id="alternatives-status">Подбираем вина из каталога…</p>
      <div class="alt-grid" id="alternatives-grid"></div>
      <div class="dish-pairings-heading sommelier-heading"><div>${icon('utensils')}<h2>Цифровой сомелье</h2></div><span>К чему подбираете вино?</span></div>
      <div class="food-buttons" id="sommelier-occasions">
        <button type="button" data-occasion="fish">Рыба</button>
        <button type="button" data-occasion="meat">Мясо</button>
        <button type="button" data-occasion="cheese">Сыр</button>
        <button type="button" data-occasion="dessert">Десерт</button>
        <button type="button" data-occasion="aperitif">Просто выпить</button>
      </div>
      <p class="pairing-explanation" id="sommelier-hint">Выберите ситуацию — предложим российские вина из каталога.</p>
      <div class="alt-grid" id="sommelier-grid"></div>
    </section>
    ${wine.demo ? `<section class="pairings"><div class="pairings-heading">${icon('utensils')}<h2>Что у вас на ужин?</h2></div><p class="muted">Выберите блюдо — подскажем, как оно сочетается с этим стилем вина.</p><div class="food-buttons"><button data-food="fish" aria-pressed="true">Рыба и морепродукты</button><button data-food="cheese" aria-pressed="false">Мягкий сыр</button><button data-food="salad" aria-pressed="false">Лёгкий салат</button><button data-food="steak" aria-pressed="false">Стейк</button><button data-food="dessert" aria-pressed="false">Десерт</button></div><p class="pairing-explanation" id="pairing-explanation"></p></section>` : ''}
    <details class="description" open><summary><span>О вине</span><span class="description-toggle"><span class="description-hide">Свернуть</span><span class="description-show">Читать</span></span></summary><p>${escape(wine.description || 'Описание пока не добавлено.')}</p></details>`;
  $('back-to-scanner').onclick = $('scan-again').onclick = backToScanner;
  $('save-wine').onclick = saveWine;
  $('compare-wine').onclick = () => toggleCompare(wine);
  syncSaveButton();
  syncCompareButton();
  const setUserRating = (value, persistRating = true) => {
    const normalized = Number.isInteger(value) && value >= 0 && value <= 5 ? value : 0;
    if (normalized && persistRating) {
      ratings[wine.slug] = normalized;
      try { localStorage.setItem('svoe-ratings', JSON.stringify(ratings)); } catch { toast('Оценка показана только до закрытия страницы.'); }
      syncProfile();
    }
    document.querySelectorAll('[data-user-rating]').forEach(button => {
      const buttonValue = Number(button.dataset.userRating);
      button.classList.toggle('filled', buttonValue <= normalized);
      button.setAttribute('aria-checked', String(buttonValue === normalized));
    });
    $('rating-status').textContent = normalized ? `Ваша оценка: ${normalized} из 5` : 'Нажмите на бокал, чтобы оценить';
    refreshFitCard();
  };
  document.querySelectorAll('[data-user-rating]').forEach(button => button.onclick = () => setUserRating(Number(button.dataset.userRating)));
  setUserRating(Number(ratings[wine.slug]) || 0, false);
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
  bindAfterSearch(wine, scanMeta);
}
function backToScanner() { $('result-screen').hidden = true; $('scanner-screen').hidden = false; $('show-example')?.focus({ preventScroll:true }); window.scrollTo({ top:0, behavior:'instant' }); }
function wineMiniCard(item) {
  const image = resolveImageUrl(item.image_url, item.photo_name || item.image_name);
  return `<article class="alt-card" data-open-slug="${escape(item.slug)}">${image ? `<img src="${escape(image)}" alt="" loading="lazy">` : '<span class="dish-placeholder">Нет фото</span>'}<div><small>${escape(item.winery || '')}</small><strong>${escape(item.name || '')}</strong><em>${escape([item.category, item.region].filter(Boolean).join(' · '))}</em></div></article>`;
}
function bindMiniCards(root, fallback) {
  root?.querySelectorAll('[data-open-slug]').forEach(card => {
    card.onclick = async () => {
      const slug = card.getAttribute('data-open-slug');
      try {
        const response = await fetch(`/v1/catalog/${encodeURIComponent(slug)}`);
        if (!response.ok) throw new Error('catalog');
        const wine = await response.json();
        showWine({ ...wine, demo: false });
      } catch {
        if (fallback) showWine(fallback);
      }
    };
  });
}
function bindAfterSearch(wine, scanMeta) {
  const ranking = scanMeta?.result?.ranking;
  const ready = Array.isArray(scanMeta?.result?.alternatives) ? scanMeta.result.alternatives : [];
  const renderAlts = items => {
    const status = $('alternatives-status');
    const grid = $('alternatives-grid');
    if (!status || !grid) return;
    if (!items.length) {
      status.textContent = 'Похожих вин других виноделен в каталоге пока нет.';
      grid.innerHTML = '';
      return;
    }
    status.textContent = 'Если это не то вино, посмотрите близкий стиль у других производителей.';
    grid.innerHTML = items.map(wineMiniCard).join('');
    bindMiniCards(grid);
  };
  if (ready.length) renderAlts(ready);
  else if (wine.slug) {
    fetch(`/v1/catalog/${encodeURIComponent(wine.slug)}/alternatives`).then(r => r.ok ? r.json() : { items: [] }).then(data => renderAlts(data.items || [])).catch(() => renderAlts([]));
  } else renderAlts([]);
  document.querySelectorAll('#sommelier-occasions [data-occasion]').forEach(button => {
    button.onclick = async () => {
      document.querySelectorAll('#sommelier-occasions [data-occasion]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      $('sommelier-hint').textContent = 'Подбираем…';
      try {
        const response = await fetch('/v1/sommelier', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ occasion: button.dataset.occasion, slug: wine.slug || null }),
        });
        if (!response.ok) throw new Error('sommelier');
        const data = await response.json();
        $('sommelier-hint').textContent = data.hint || '';
        $('sommelier-grid').innerHTML = (data.wines || []).map(wineMiniCard).join('');
        bindMiniCards($('sommelier-grid'));
      } catch {
        $('sommelier-hint').textContent = 'Сомелье сейчас недоступен. Можно выбрать блюдо из карточки выше.';
      }
    };
  });
  if (debugMode && ranking) {
    addProcessingLog(`F1 top-1 ${ranking.f1_top1} · F1 top-5 ${ranking.f1_top5} · отрыв ${ranking.margin}`);
  }
}
function persist() { try { localStorage.setItem('svoe-wines', JSON.stringify(saved)); syncSavedCount(); syncProfile(); return true; } catch { toast('Не удалось сохранить: хранилище браузера недоступно.'); return false; } }
function saveWine() {
  if (!currentWine) return;
  const previous = [...saved];
  const index = saved.findIndex(wine => wine.slug === currentWine.slug);
  if (index >= 0) {
    saved.splice(index, 1);
    if (persist()) toast('Вино убрано из сохранённых');
    else saved = previous;
  } else {
    saved = [{ ...currentWine }, ...saved].slice(0, 100);
    if (persist()) toast('Вино сохранено');
    else saved = previous;
  }
  syncSaveButton();
  refreshFitCard();
}
function refreshFitCard() {
  const card = document.querySelector('.fit-card');
  if (!currentWine) return;
  const markup = fitMarkupFor(currentWine);
  if (card) {
    if (markup) {
      const replacement = document.createRange().createContextualFragment(markup).firstElementChild;
      if (replacement) card.replaceWith(replacement);
    } else {
      card.remove();
    }
  } else if (markup) {
    $('rating-status')?.insertAdjacentHTML('afterend', markup);
  }
}
function renderSaved() {
  $('saved-list').innerHTML = saved.length ? saved.map((w,i) => { const image = resolveImageUrl(w.image_url, w.photo_name || w.image_name || w.photoName); return `<div class="saved-item"><span class="saved-item-icon">${icon('bookmark')}</span>${image ? `<img src="${escape(image)}" alt=""/>` : ''}<button data-saved="${i}">${escape(w.name)}<small>${escape(w.winery)}${w.demo ? ' · пример' : ''}</small></button><button class="remove-saved" data-remove="${i}" aria-label="Убрать ${escape(w.name)} из сохранённых">${icon('x')}</button></div>`; }).join('') : '<p class="muted">Здесь будут сохранённые вина. Откройте карточку и нажмите на закладку.</p>';
  document.querySelectorAll('[data-saved]').forEach(b => b.onclick = () => { $('saved-dialog').close(); showWine(saved[Number(b.dataset.saved)]); });
  document.querySelectorAll('[data-remove]').forEach(b => b.onclick = () => { const previous = [...saved]; saved.splice(Number(b.dataset.remove),1); if (!persist()) saved = previous; syncSaveButton(); renderSaved(); });
}
$('saved-open').onclick = () => { renderSaved(); $('saved-dialog').showModal(); };
$('saved-close').onclick = () => $('saved-dialog').close();
$('saved-dialog').addEventListener('click', e => { if (e.target === $('saved-dialog')) { const rect = e.target.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) e.target.close(); } });
function openCompareDialog() { renderCompareDialog(); $('compare-dialog').showModal(); }
$('compare-open').onclick = openCompareDialog;
$('compare-tray-open').onclick = openCompareDialog;
$('compare-close').onclick = () => $('compare-dialog').close();
$('compare-dialog').addEventListener('click', e => { if (e.target === $('compare-dialog')) { const rect = e.target.getBoundingClientRect(); if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) e.target.close(); } });
renderCompareTray();
window.addEventListener('pagehide', () => { stopCamera(); controller?.abort('user'); if (photoUrl) URL.revokeObjectURL(photoUrl); });
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) $('close-camera').click(); });


document.querySelector('.search-trigger').onclick = () => {
  $('search-dialog').showModal(); $('wine-query').focus();
};
$('search-close').onclick = () => $('search-dialog').close();
let searchRequest;
$('catalog-search').onsubmit = async event => {
  event.preventDefault();
  searchRequest?.abort();
  searchRequest = new AbortController();
  const active = searchRequest;
  $('search-status').textContent = 'Ищем в каталоге…';
  $('search-results').replaceChildren();
  const endpoint = window.SCANNER_CONFIG?.recognitionEndpoint;
  if (!endpoint) { $('search-status').textContent = 'Поиск доступен в подключённой версии приложения.'; return; }
  const url = new URL(endpoint, location.origin);
  url.pathname = url.pathname.replace(/recognize$/, 'search');
  url.searchParams.set('q', $('wine-query').value.trim());
  const timer = setTimeout(() => active.abort(), 10000);
  try {
    const response = await fetch(url, { signal: active.signal });
    if (!response.ok) throw new Error('search');
    const { items } = await response.json();
    $('search-status').textContent = items.length ? `Найдено: ${items.length}` : 'Ничего не найдено. Уточните название.';
    for (const wine of items) {
      const button = document.createElement('button');
      button.className = 'catalog-result';
      button.textContent = `${wine.name} · ${wine.winery}`;
      button.onclick = () => { $('search-dialog').close(); showWine(wine); };
      $('search-results').append(button);
    }
  } catch { if (active === searchRequest) $('search-status').textContent = 'Поиск недоступен. Попробуйте ещё раз.'; }
  finally { clearTimeout(timer); }
};
