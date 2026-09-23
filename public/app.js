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
  x: '<path d="m6 6 12 12M6 18 18 6"/>',
  check: '<path d="m5 12 4 4L19 6"/>'
};
const icon = (name) => name === 'compare'
  ? '<span class="icon-compare" aria-hidden="true"></span>'
  : `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.wine}</svg>`;
document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
const escape = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
const catalogAssets = {
  region: {
    'крым': '/assets/catalog/region-krym.webp',
    'кубань': '/assets/catalog/region-kuban.webp'
  },
  grape: {
    'шардоне': '/assets/catalog/grape-chardonnay.webp',
    'мерло': '/assets/catalog/grape-merlot.webp',
    'совиньон блан': '/assets/catalog/grape-sauvignon-blanc.webp',
    'пино нуар': '/assets/catalog/grape-pinot-noir.webp',
    'рислинг': '/assets/catalog/grape-riesling.webp',
    'саперави': '/assets/catalog/grape-saperavi.webp',
    'каберне совиньон': '/assets/catalog/grape-cabernet-sauvignon.webp',
    'каберне фран': '/assets/catalog/grape-cabernet-franc.webp',
    'красностоп': '/assets/catalog/grape-krasnostop.webp',
    'мальвазия': '/assets/catalog/grape-malvaziya.webp',
    'первенец магарача': '/assets/catalog/grape-pervenets-magaracha.webp',
    'мальбек': '/assets/catalog/grape-malbec.webp'
  },
  dish: {
    'азиатская кухня': '/assets/catalog/dish-asian.webp',
    'блюда из рыбы': '/assets/catalog/dish-fish.webp',
    'рыба и морепродукты': '/assets/catalog/dish-fish.webp',
    'морепродукты': '/assets/catalog/dish-fish.webp',
    'устрицы': '/assets/catalog/dish-oysters.webp',
    'фруктово-ягодные десерты': '/assets/catalog/dish-fruit-dessert.webp',
    'выпечка и десерты': '/assets/catalog/dish-pastry.webp',
    'десерты': '/assets/catalog/dish-dessert.webp',
    'мороженое': '/assets/catalog/dish-pastry.webp',
    'шоколад': '/assets/catalog/dish-dessert.webp',
    'пицца': '/assets/catalog/dish-pizza.webp',
    'паста': '/assets/catalog/dish-pasta.webp',
    'салаты': '/assets/catalog/dish-salad.webp',
    'брускетты': '/assets/catalog/dish-bruschetta.webp',
    'паштеты': '/assets/catalog/dish-pate.webp',
    'кухни народов мира': '/assets/catalog/dish-world-cuisine.webp',
    'блюда из птицы': '/assets/catalog/dish-poultry.webp',
    'птица': '/assets/catalog/dish-poultry.webp',
    'запеченные овощи': '/assets/catalog/dish-grilled-vegetables.webp',
    'овощи гриль': '/assets/catalog/dish-grilled-vegetables.webp',
    'овощи на гриле': '/assets/catalog/dish-grilled-vegetables.webp',
    'легкие закуски': '/assets/catalog/dish-light-snacks.webp',
    'закуски': '/assets/catalog/dish-snacks.webp',
    'сыры': '/assets/catalog/dish-cheese.webp',
    'сыр': '/assets/catalog/dish-cheese.webp',
    'мясо и стейки': '/assets/catalog/dish-steak.webp',
    'мясное ассорти': '/assets/catalog/dish-meat-platter.webp',
    'кавказская кухня': '/assets/catalog/dish-meat-platter.webp',
    'bbq': '/assets/catalog/dish-meat-platter.webp',
    'барбекю': '/assets/catalog/dish-meat-platter.webp',
    'средиземноморская кухня': '/assets/catalog/dish-mediterranean.webp'
  }
};
const normalizeCatalogValue = value => String(value ?? '').trim().toLocaleLowerCase('ru-RU').replace(/ё/g, 'е').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
function catalogAsset(type, value) {
  const normalized = normalizeCatalogValue(value);
  if (!normalized) return '';
  const assets = catalogAssets[type] || {};
  if (assets[normalized]) return safeImage(assets[normalized]);
  const key = Object.keys(assets).find(candidate => normalized.includes(candidate) || candidate.includes(normalized));
  return key ? safeImage(assets[key]) : '';
}
// Prefer the wine's catalog photo; only use a matching variety as fallback.
const grapeBackdropAssets = {
  'мальбек': '/assets/catalog/grape-malbec-wide.webp'
};
function grapeBackdrop(wine, grapes) {
  const supplied = safeImage(wine.grape_image_url);
  if (supplied) {
    const url = new URL(supplied);
    if (url.hostname === 'api.vino-svoe.ru') {
      url.pathname = url.pathname.replace(/\/str-api\/\d+\/\d+\/resize\//, '/str-api/740/740/resize/');
    }
    return url.href;
  }
  const wide = grapes.map(grape => grapeBackdropAssets[normalizeCatalogValue(grape)]).find(Boolean);
  if (wide) return wide;
  return grapes.map(grape => catalogAsset('grape', grape)).find(Boolean) || '';
}
function wineTitle(name) {
  return escape(name).replace(/\s+(Appassimento)$/i, '<span class="wine-title-cuvee">$1</span>');
}
function ratingGlass() {
  return '<img class="rating-public-icon" src="/assets/public-rating.svg" alt="" width="32" height="32">';
}
function userRatingIcons() {
  return '<span class="rating-icons" aria-hidden="true"><img class="rating-icon-glass" src="/assets/public-rating-empty.svg" alt="" width="40" height="40"><img class="rating-icon-wine" src="/assets/public-rating.svg" alt="" width="40" height="40"></span>';
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
function collectRatedWines() {
  const profileWines = [...saved];
  if (currentWine && !profileWines.some(wine => wine.slug === currentWine.slug)) profileWines.push(currentWine);
  return profileWines
    .map(wine => ({ wine, rating: Number(ratings[wine.slug]) }))
    .filter(item => Number.isInteger(item.rating) && item.rating >= 1 && item.rating <= 5);
}
function averageTaste(items, weightOf) {
  const weight = items.reduce((sum, item) => sum + weightOf(item.rating), 0);
  if (!weight) return null;
  return Object.fromEntries(tasteDimensions.map(({ key }) => [
    key,
    items.reduce((sum, item) => sum + deriveTasteProfile(item.wine)[key] * weightOf(item.rating), 0) / weight
  ]));
}
function tasteDistance(profile, values) {
  if (!values) return 1;
  return tasteDimensions.reduce((sum, { key }) => sum + Math.abs(profile[key] - values[key]) / 4, 0) / tasteDimensions.length;
}
function getPreferenceProfile() {
  const rated = collectRatedWines();
  const likes = rated.filter(item => item.rating >= 4);
  if (!likes.length) return null;
  const dislikes = rated.filter(item => item.rating <= 2);
  return {
    likes: averageTaste(likes, rating => rating),
    dislikes: averageTaste(dislikes, rating => 6 - rating),
    likeCount: likes.length,
    dislikeCount: dislikes.length,
    count: likes.length
  };
}
function calculateFit(wine, preference = getPreferenceProfile()) {
  if (!preference?.likes) return null;
  const profile = deriveTasteProfile(wine);
  let score = 100 - tasteDistance(profile, preference.likes) * 100;
  if (preference.dislikes) score -= (1 - tasteDistance(profile, preference.dislikes)) * 35;
  return Math.max(0, Math.min(100, Math.round(score)));
}
function fitMarkupFor(wine) {
  const preference = getPreferenceProfile();
  const fit = calculateFit(wine, preference);
  if (fit === null) return '';
  return `<section class="fit-card"><strong>${fit}%</strong><div><small>ПОДХОДИТ ВАМ</small><b>Похоже на то, что вы любите</b><p>На основе ${preference.count} ${preference.count === 1 ? 'вашей высокой оценки' : preference.count < 5 ? 'ваших высоких оценок' : 'высоких оценок в вашем профиле'} и вкусового профиля вина.</p></div></section>`;
}
const example = {
  slug: 'fanagoriya-blanc-de-blancs-shardone-beloe-bryut-12', name: 'Blanc de Blancs', winery: 'Фанагория',
  category: 'Игристое белое · Брют', color: 'Белое', region: 'Кубань', grapes: ['Шардоне'], image_url: '/assets/blanc-de-blancs.webp',
  summary: 'Свежее, изящное вино с нотами белых цветов, персика и грейпфрута. Сбалансированная кислотность и лёгкая сливочность во вкусе.',
  description: 'Игра пузырьков изящная и утонченная — непрерывный жемчужный перляж. Аромат элегантный и тонкий, завораживающий. Вначале раскрывается нотами легких белых цветов и прохладой цветущего утреннего сада. Затем проявляются сочные оттенки мякоти белых фруктов и розового персика, свежие нотки грейпфрута и легкий травяной акцент. Фоном звучит тонкий оттенок миндального пралине. Вкус изящный, свежий и ажурный. Прекрасно сбалансированная, стройная кислотность отлично гармонирует с тонкой сливочностью и ведет к чистому и стройному послевкусию. Идеальный аперитив.',
  aromas: ['Белые цветы', 'Персик', 'Грейпфрут', 'Миндаль'], demo: true
};
let stream = null, cameraGeneration = 0, photo = null, photoUrl = null, controller = null, currentWine = null, toastTimer;
let bottleChoices = [];
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
function clearBottleChoices() {
  bottleChoices = [];
  $('bottle-choice-layer').hidden = true;
  $('bottle-choice-buttons').replaceChildren();
}
function renderBottleChoices() {
  const image = $('photo-preview');
  const frame = $('dropzone').getBoundingClientRect();
  if (!image.naturalWidth || !image.naturalHeight || !frame.width || !frame.height) return;
  const scale = Math.min(frame.width / image.naturalWidth, frame.height / image.naturalHeight);
  const drawnWidth = image.naturalWidth * scale;
  const drawnHeight = image.naturalHeight * scale;
  const offsetX = (frame.width - drawnWidth) / 2;
  const offsetY = (frame.height - drawnHeight) / 2;
  const buttons = $('bottle-choice-buttons');
  buttons.replaceChildren();
  bottleChoices.forEach((candidate, index) => {
    const [left, top, right, bottom] = candidate.bbox;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'bottle-choice-box';
    button.setAttribute('aria-label', `Выбрать бутылку ${index + 1}`);
    const number = document.createElement('span');
    number.textContent = String(index + 1);
    button.append(number);
    button.style.left = `${(offsetX + left * drawnWidth) / frame.width * 100}%`;
    button.style.top = `${(offsetY + top * drawnHeight) / frame.height * 100}%`;
    button.style.width = `${(right - left) * drawnWidth / frame.width * 100}%`;
    button.style.height = `${(bottom - top) * drawnHeight / frame.height * 100}%`;
    button.onclick = () => {
      const selectedBox = candidate.bbox;
      clearBottleChoices();
      runRecognize(selectedBox);
    };
    buttons.append(button);
  });
}
function showBottleChoices(candidates) {
  if (!$('photo-preview').naturalWidth || !Array.isArray(candidates) || candidates.length < 2) return false;
  bottleChoices = candidates.filter(item => Array.isArray(item?.bbox) && item.bbox.length === 4);
  if (bottleChoices.length < 2) return false;
  $('bottle-choice-layer').hidden = false;
  renderBottleChoices();
  return true;
}
window.addEventListener('resize', () => {
  if (!$('bottle-choice-layer').hidden) renderBottleChoices();
});
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
  button.classList.toggle('primary', compared);
  button.classList.toggle('secondary', !compared);
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
  } else if (comparison.length >= 3) {
    toast('Можно выбрать максимум три вина');
    return;
  } else {
    comparison.push({ ...wine });
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
const PHOTO_TYPES = new Set(['image/jpeg','image/jpg','image/pjpeg','image/png','image/webp','image/heic','image/heif','image/heic-sequence','image/heif-sequence','image/avif']);
const PHOTO_EXT = /\.(jpe?g|png|webp|heic|heif|avif)$/i;
function isAcceptedPhoto(file) {
  const type = (file.type || '').toLowerCase();
  if (PHOTO_TYPES.has(type)) return true;
  if (PHOTO_EXT.test(file.name || '')) return true;
  return type.startsWith('image/');
}
function isHeifLike(file) {
  const type = (file.type || '').toLowerCase();
  return type.includes('heic') || type.includes('heif') || /\.(heic|heif)$/i.test(file.name || '');
}
async function selectPhoto(file) {
  if (!file || controller) return;
  clearBottleChoices();
  notice('');
  if (!isAcceptedPhoto(file)) { notice('Выберите фото с телефона: JPEG, HEIC, PNG, WebP или AVIF.'); return; }
  if (file.size > 15 * 1024 * 1024) { notice('Фотография слишком большая. Выберите файл до 15 МБ.'); return; }
  const candidateUrl = URL.createObjectURL(file);
  let canPreview = false;
  try {
    const img = new Image(); img.src = candidateUrl; await img.decode();
    if (!img.naturalWidth || !img.naturalHeight || img.naturalWidth * img.naturalHeight > 24000000) throw new Error('Invalid image');
    canPreview = true;
  } catch {
    if (!isHeifLike(file)) {
      URL.revokeObjectURL(candidateUrl);
      notice('Не удалось открыть изображение. Попробуйте другую фотографию.');
      return;
    }
  }
  stopCamera(); if (photoUrl) URL.revokeObjectURL(photoUrl);
  photo = file; photoUrl = candidateUrl;
  $('photo-preview').src = canPreview ? photoUrl : '';
  $('photo-preview').hidden = !canPreview;
  $('example-bottle').hidden = true; $('sample-label').hidden = true;
  $('viewfinder-caption').textContent = canPreview
    ? 'Ваше фото · проверьте читаемость этикетки'
    : 'HEIC с iPhone принят. В этом браузере превью нет — нажмите «Узнать вино».';
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
  if (/^https?:\/\//i.test(raw) || raw.startsWith('/')) return usableBottleUrl(safeImage(raw));
  const base = window.SCANNER_CONFIG?.imageBaseUrl;
  if (!base) return '';
  try {
    const relative = raw.replace(/^\/?(?:uploads\/)?/, '').split('/').map(encodeURIComponent).join('/');
    return usableBottleUrl(safeImage(new URL(relative, base).href));
  } catch { return ''; }
}
function usableBottleUrl(url) {
  if (!url) return '';
  if (url.startsWith('/') && !url.includes('api.vino-svoe.ru')) return url;
  const name = decodeURIComponent((url.split('/').pop() || '').split('?')[0]);
  if (/_[0-9a-f]{8,}\.(webp|jpe?g|png)$/i.test(name)) return url;
  if (url.includes('api.vino-svoe.ru') && /-/.test(name)) return '';
  return url;
}
function formatRating(value) {
  const rating = Number(value);
  if (!Number.isFinite(rating) || rating < 0 || rating > 5) return '0.00';
  return rating.toFixed(2);
}
function formatScorePct(value) {
  if (value == null || value === '') return '';
  const score = Number(value);
  if (!Number.isFinite(score)) return '';
  const pct = Math.max(0, Math.min(100, score * 100));
  return `${pct.toFixed(1).replace('.', ',')}%`;
}
function withLabelScores(items, result) {
  const ranked = [...(result?.ranking?.top5 || []), ...(result?.recognition?.candidates || [])];
  const scores = Object.fromEntries(ranked.filter(item => item?.slug).map(item => [item.slug, item.score]));
  return (Array.isArray(items) ? items : []).map(item => {
    const label_score = Number.isFinite(Number(item?.label_score)) ? Number(item.label_score) : scores[item?.slug];
    return { ...item, label_score };
  });
}
const queryParams = new URLSearchParams(location.search);
const debugMode = queryParams.get('admin') === '1' || queryParams.get('debug') === '1';
let lastCompareSlug = '';
let lastCompareQuery = '';
let probeSearchTimer = 0;
$('admin-tools').hidden = !debugMode;
$('processing-debug').hidden = !debugMode;
function debugOcrEnabled() {
  const boxes = [$('debug-ocr'), $('debug-ocr-result')].filter(Boolean);
  if (!debugMode) return true;
  if (!boxes.length) return false;
  return boxes.some(box => box.checked);
}
function setDebugOcr(enabled) {
  [$('debug-ocr'), $('debug-ocr-result')].forEach(box => {
    if (box) box.checked = enabled;
  });
}
function debugCatalogImage(url) {
  if (!url) return '';
  try {
    const parsed = new URL(url, location.origin);
    return ['http:', 'https:'].includes(parsed.protocol) ? parsed.href : '';
  } catch {
    return '';
  }
}
function cropFromPhoto(bbox) {
  const img = $('photo-preview');
  if (!img || !img.naturalWidth || !Array.isArray(bbox) || bbox.length < 4) return '';
  const left = Math.max(0, Number(bbox[0]));
  const top = Math.max(0, Number(bbox[1]));
  const right = Math.min(1, Number(bbox[2]));
  const bottom = Math.min(1, Number(bbox[3]));
  if (!(right > left) || !(bottom > top)) return '';
  const sx = Math.round(left * img.naturalWidth);
  const sy = Math.round(top * img.naturalHeight);
  const sw = Math.max(1, Math.round((right - left) * img.naturalWidth));
  const sh = Math.max(1, Math.round((bottom - top) * img.naturalHeight));
  const canvas = document.createElement('canvas');
  const maxSide = 360;
  const scale = Math.min(1, maxSide / Math.max(sw, sh));
  canvas.width = Math.max(1, Math.round(sw * scale));
  canvas.height = Math.max(1, Math.round(sh * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  try { return canvas.toDataURL('image/jpeg', 0.72); } catch { return ''; }
}
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
  addProcessingLog(debugOcrEnabled() ? 'OCR включён для этого скана' : 'OCR выключен для этого скана');
}
function stopProcessingLog() {
  clearInterval(processingTimer);
  processingTimer = null;
  updateProcessingElapsed();
}
function formatPct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1).replace('.', ',')}%` : '—';
}
function viewScoreLine(item) {
  const bottle = formatPct(item.full_score);
  const label = formatPct(item.crop_score);
  const better = item.best_view === 'crop' ? 'этикетка' : item.best_view === 'full' ? 'бутылка' : '';
  return `бутылка ${bottle} · этикетка ${label}${better ? ` · лучше ${better}` : ''}`;
}
function comparePairMarkup(item, cropSrc, place) {
  const catalogSrc = debugCatalogImage(item.image_url);
  const catalogLabel = item.label_jpeg_base64 ? `data:image/jpeg;base64,${item.label_jpeg_base64}` : '';
  const pct = Number.isFinite(Number(item.score)) ? `${(Number(item.score) * 100).toFixed(1).replace('.', ',')}%` : '';
  const title = [item.winery, item.name || item.slug].filter(Boolean).join(' · ');
  const missing = item.missing ? 'В индексе нет векторов этой карточки.' : `Сравнение раздельное: ${viewScoreLine(item)}.`;
  const catalogShots = `<div class="catalog-match-shots">${catalogSrc ? `<figure><img class="crop-sent" src="${escape(catalogSrc)}" alt="Бутылка в каталоге"><figcaption>Бутылка · ${escape(formatPct(item.full_score))}</figcaption></figure>` : ''}${catalogLabel ? `<figure><img class="crop-sent" src="${escape(catalogLabel)}" alt="Кроп этикетки в индексе"><figcaption>Этикетка · ${escape(formatPct(item.crop_score))}</figcaption></figure>` : (catalogSrc ? '' : '<span class="dish-placeholder">Нет фото каталога</span>')}</div>`;
  return `<div class="scan-debug-compare-pair"><figure>${cropSrc ? `<img class="crop-sent" src="${escape(cropSrc)}" alt="Crop запроса">` : '<span class="dish-placeholder">Crop не собран</span>'}<figcaption>Ваш crop</figcaption></figure><span class="scan-debug-compare-vs" aria-hidden="true">↔</span><div>${catalogShots}<p class="scan-debug-compare-note">${escape(place)}${pct ? ` · итого ${escape(pct)}` : ''} · ${escape(title)}. ${escape(missing)}</p></div></div>`;
}
function formatPoints(delta) {
  const value = Number(delta);
  if (!Number.isFinite(value)) return '—';
  const points = value * 100;
  const sign = points > 0 ? '+' : '';
  return `${sign}${points.toFixed(1).replace('.', ',')} п.п.`;
}
function cropBoxStyle(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 4) return '';
  const left = Math.max(0, Number(bbox[0]) * 100);
  const top = Math.max(0, Number(bbox[1]) * 100);
  const width = Math.max(0, (Number(bbox[2]) - Number(bbox[0])) * 100);
  const height = Math.max(0, (Number(bbox[3]) - Number(bbox[1])) * 100);
  if (![left, top, width, height].every(Number.isFinite)) return '';
  return `left:${left}%;top:${top}%;width:${width}%;height:${height}%`;
}
function quadOverlay(quad) {
  if (!Array.isArray(quad) || quad.length < 3) return '';
  const points = quad.map(point => `${Number(point[0]) * 100},${Number(point[1]) * 100}`).join(' ');
  if (!points.includes(',')) return '';
  return `<svg class="crop-quad" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><polygon points="${points}"></polygon></svg>`;
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
  const ocrOn = metrics.ocr_enabled === true || (metrics.ocr && metrics.ocr !== 'skipped');
  const scoreNote = similarity === null ? 'метрика недоступна' : (ocrOn ? 'сходство SigLIP + OCR, не вероятность' : 'сходство SigLIP, не вероятность');
  const cropRejected = metrics.crop_quality?.usable === false;
  const bbox = metrics.label_detection?.bbox;
  const boxStyle = cropBoxStyle(bbox);
  const quadMarkup = quadOverlay(metrics.label_detection?.contour || metrics.label_detection?.quad);
  const cropSrc = metrics.crop_jpeg_base64 ? `data:image/jpeg;base64,${metrics.crop_jpeg_base64}` : cropFromPhoto(bbox);
  const compared = Array.isArray(metrics.compared_with) && metrics.compared_with.length
    ? metrics.compared_with
    : (result?.ranking?.top5 || []).slice(0, 2);
  const comparePair = compared.map((item, index) => comparePairMarkup(item, cropSrc, index === 0 ? '1 место' : `${index + 1} место`)).join('');
  const probe = metrics.probe;
  const probePair = probe ? comparePairMarkup(probe, cropSrc, 'Выбранная бутылка') : '';
  const probeSearch = `<div class="debug-probe"><label for="debug-probe-q">Сравнить crop с другой бутылкой</label><input id="debug-probe-q" type="search" autocomplete="off" placeholder="Герцъ, Сикоры" value="${escape(lastCompareQuery)}"><div id="debug-probe-hits" class="debug-probe-hits" hidden></div><p class="scan-debug-footnote">Поиск по каталогу. Выберите вино — пересканируем это фото и покажем бутылку и этикетку рядом с вашим crop. На плохом кадре это не поднимет итоговый %, только покажет, сколько набирает выбранная карточка.</p></div>`;
  const neighborCards = (result?.ranking?.top5 || []).map((item, index) => {
    const catalogSrc = debugCatalogImage(item.image_url);
    const pct = Number.isFinite(Number(item.score)) ? `${(Number(item.score) * 100).toFixed(1).replace('.', ',')}%` : '—';
    return `<article class="scan-debug-neighbor">${catalogSrc ? `<img src="${escape(catalogSrc)}" alt="">` : '<span class="dish-placeholder">Нет фото</span>'}<div><small>${index + 1}. ${escape(item.winery || '')}</small><strong>${escape(item.name || item.slug || '')}</strong><em>итого ${escape(pct)} · ${escape(viewScoreLine(item))} · цвет ${formatPoints(item.color_delta)} · OCR ${formatPoints(item.ocr_delta)}</em>${ocrOn ? `<small>OCR сравнивается с: ${escape(item.ocr_catalog_text || `${item.name || ''} ${item.winery || ''}`)}. Текст эталона: ${escape(item.ocr_reference_text || 'нет сохранённого текста')}</small>` : ''}</div></article>`;
  }).join('');
  const cropMarkup = `<div class="scan-debug-crop">${photoUrl && (quadMarkup || boxStyle) ? `<figure><div class="crop-source"><img src="${escape(photoUrl)}" alt="Исходное фото">${quadMarkup || (boxStyle ? `<span class="crop-box" style="${boxStyle}"></span>` : '')}</div><figcaption>${cropRejected ? 'Область отклонена: ненадёжный кроп' : (quadMarkup ? 'Красный контур — предполагаемая граница этикетки' : 'Красная рамка — область кропа')}</figcaption></figure>` : ''}${cropSrc ? `<figure><img class="crop-sent" src="${escape(cropSrc)}" alt="Crop"><figcaption>${metrics.query_view === 'full_image' ? 'С каталогом сравнивали исходное фото' : metrics.query_view === 'front_design' ? 'Запасной кроп: печать на стекле. Сравнили также исходное фото' : 'Этот crop сравнивали с каталогом'}</figcaption></figure>` : '<p class="scan-debug-footnote">Crop не пришёл — проверьте, что фото ещё в превью.</p>'}</div><div class="scan-debug-details"><h3>Найти бутылку в каталоге</h3>${probeSearch}${probePair ? `<div class="scan-debug-compare">${probePair}</div>` : ''}</div>${comparePair ? `<div class="scan-debug-details"><h3>С чем сравнивали crop</h3><div class="scan-debug-compare">${comparePair}</div></div>` : ''}${neighborCards ? `<div class="scan-debug-details"><h3>Похожие в индексе</h3><div class="scan-debug-neighbors">${neighborCards}</div></div>` : ''}`;
  const color = metrics.color || {};
  const paper = color.paper === 'cream' ? 'кремовая/светлая бумага' : color.paper === 'dark' ? 'тёмная бумага' : color.paper === 'mixed' ? 'смешанный тон' : 'не определён';
  const bottle = color.bottle_tone === 'red' ? 'красное' : color.bottle_tone === 'white' ? 'белое/светлое' : color.bottle_tone === 'rose' ? 'розовое' : 'не виден';
  const contribMarkup = `<div class="scan-debug-contrib"><div><small>SIGLIP</small><strong>${Number.isFinite(metrics.siglip) ? `${(metrics.siglip * 100).toFixed(1).replace('.', ',')}%` : '—'}</strong><em>сходство картинки</em></div><div><small>ЦВЕТ</small><strong>выкл</strong><em>не влияет на порядок кандидатов</em></div><div><small>OCR</small><strong>${ocrOn ? formatPoints(metrics.ocr_delta) : 'выкл'}</strong><em>${ocrOn ? (metrics.ocr_text ? `«${escape(String(metrics.ocr_text).slice(0, 80))}»` : 'текст не прочитан') : 'на этом запросе не запускался'}</em></div></div>`;
  const detailRows = [
    ['Декодирование фото', timingValues.decode, 'открытие, EXIF и приведение к RGB'],
    ['Поиск этикетки', timingValues.labelDetection, 'автоматический поиск области для crop'],
    ['Вырезание crop', timingValues.labelCrop, 'вырезание найденной области'],
    ['Подготовка изображения', timingValues.enhancement, 'контраст/резкость для enhanced-режима'],
    ['Изображение', timingValues.visual, 'SigLIP-вектор + поиск ближайших в каталоге'],
    ...(ocrOn ? [['OCR', timingValues.ocr, 'Tesseract и сопоставление распознанного текста']] : []),
    ['Остальное backend', backendOther, 'сборка ответа и операции, не выделенные отдельно'],
    ['Загрузка, сеть и браузер', browserOther, 'разница между полным ожиданием и backend'],
  ];
  const detailMarkup = detailRows.map(([label, value, note]) => `<div class="scan-debug-detail-row"><div><strong>${label}</strong><small>${note}</small></div><b>${timing(value)}</b></div>`).join('');
  return `<section class="scan-debug" aria-label="Диагностика сканирования"><div class="scan-debug-heading"><div><small>DEBUG · РЕЗУЛЬТАТ СКАНИРОВАНИЯ</small><h2>Технические показатели</h2></div><span class="scan-debug-status">${escape(status)}</span></div><div class="debug-ocr-bar"><label class="debug-ocr-toggle"><input type="checkbox" id="debug-ocr-result" ${ocrOn ? 'checked' : ''}> OCR</label><button type="button" class="button secondary" id="debug-rescan">Пересканировать это фото</button><span>Этот скан: OCR ${ocrOn ? 'включён' : 'выключен'}. Смените галочку и нажмите пересканировать.</span></div><div class="scan-debug-metrics"><div><small>ИТОГО</small><strong>${score}</strong><em>${scoreNote}</em></div><div><small>СКОРОСТЬ</small><strong>${formatElapsed(elapsed)}</strong><em>полное ожидание в браузере</em></div><div><small>СЕРВЕР</small><strong>${timing(serverTotal)}</strong><em>распознавание backend</em></div></div>${cropMarkup}${contribMarkup}<div class="scan-debug-details"><h3>Разбивка времени</h3>${detailMarkup}</div><p class="scan-debug-footnote">Проценты — косинус картинки. «п.п.» у цвета и OCR — сколько пунктов добавили или сняли. Если OCR выключен, его вклад должен быть «выкл».</p></section>`;
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
    `Сходство итог: ${Number.isFinite(score) ? (score * 100).toFixed(1) + '%' : 'не измерено'}`,
    `SigLIP: ${Number.isFinite(metrics.siglip) ? (metrics.siglip * 100).toFixed(1) + '%' : '—'} · цвет ${formatPoints(metrics.color_delta)} · OCR ${formatPoints(metrics.ocr_delta)}`,
    `Цвет бумаги: ${metrics.color?.paper || '—'} · тон бутылки: ${metrics.color?.bottle_tone || '—'}`,
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
  const resultTop = $('result-screen')?.querySelector('.result-top');
  if (!$('result-screen')?.hidden && resultTop) resultTop.after(details);
  else $('admin-tools').append(details);
}
$('debug-ocr')?.addEventListener('change', event => setDebugOcr(event.target.checked));
function bindDebugPanel() {
  const resultBox = $('debug-ocr-result');
  if (resultBox) resultBox.onchange = () => setDebugOcr(resultBox.checked);
  const rescan = $('debug-rescan');
  if (rescan) rescan.onclick = () => runRecognize();
  const probeInput = $('debug-probe-q');
  const probeHits = $('debug-probe-hits');
  if (!probeInput || !probeHits) return;
  probeInput.oninput = () => {
    clearTimeout(probeSearchTimer);
    const q = probeInput.value.trim();
    if (q.length < 2) { probeHits.hidden = true; probeHits.replaceChildren(); return; }
    probeSearchTimer = setTimeout(async () => {
      try {
        const response = await fetch(`/v1/search?q=${encodeURIComponent(q)}`);
        const data = response.ok ? await response.json() : { items: [] };
        const items = (data.items || []).slice(0, 8);
        probeHits.hidden = !items.length;
        probeHits.innerHTML = items.map(item => `<button type="button" data-probe-slug="${escape(item.slug)}"><small>${escape(item.winery || '')}</small><strong>${escape(item.name || item.slug)}</strong></button>`).join('');
        probeHits.querySelectorAll('[data-probe-slug]').forEach(button => {
          button.onclick = () => {
            lastCompareSlug = button.getAttribute('data-probe-slug') || '';
            lastCompareQuery = button.querySelector('strong')?.textContent || lastCompareSlug;
            probeHits.hidden = true;
            runRecognize();
          };
        });
      } catch {
        probeHits.hidden = true;
      }
    }, 220);
  };
}
$('recognize').onclick = () => runRecognize();
// The server silhouette and comparison crop share the same source coordinates.
async function revealLabel(detection, signal) {
  const points = detection?.contour;
  if (!Array.isArray(points) || points.length < 3 || points.some(p =>
    !Array.isArray(p) || p.length !== 2 || p.some(v => !Number.isFinite(v) || v < 0 || v > 1))) return;
  const source = $('scanning-label');
  try { await source.decode(); } catch { return; }
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  const canvas = document.createElement('canvas');
  const scale = Math.min(1, 640 / Math.max(source.naturalWidth, source.naturalHeight));
  canvas.width = Math.round(source.naturalWidth * scale);
  canvas.height = Math.round(source.naturalHeight * scale);
  canvas.setAttribute('aria-label', 'Выделенная этикетка');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const w = canvas.width, h = canvas.height;
  const background = document.createElement("canvas");
  background.width = w; background.height = h;
  background.getContext("2d").drawImage(source, 0, 0, w, h);
  const path = new Path2D();
  points.forEach(([x, y], i) => i ? path.lineTo(x * w, y * h) : path.moveTo(x * w, y * h));
  path.closePath();
  const label = document.createElement('canvas');
  label.width = w; label.height = h;
  const lc = label.getContext('2d');
  lc.clip(path); lc.drawImage(source, 0, 0, w, h);
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) * w / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) * h / 2;
  const zoom = Math.min(1.7, .85 / Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
  const container = source.parentElement;
  container.append(canvas); container.classList.add('revealing');
  $('processing-status').textContent = 'Отделяем этикетку от фона…';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  await new Promise((resolve, reject) => {
    let frame, start;
    const cleanup = () => { cancelAnimationFrame(frame); signal.removeEventListener('abort', abort); };
    const abort = () => { cleanup(); reject(new DOMException('Aborted', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    const draw = now => {
      start ??= now;
      const t = reduced ? 1 : Math.min(1, (now - start) / 1350);
      const fade = Math.min(1, t / .7);
      ctx.clearRect(0, 0, w, h);
      ctx.save();
      const move = Math.max(0, (t - .5) * 2); const ease = move * move * (3 - 2 * move);
      ctx.translate((w / 2 - cx) * ease, (h / 2 - cy) * ease);
      ctx.translate(cx, cy); ctx.scale(1 + (zoom - 1) * ease, 1 + (zoom - 1) * ease); ctx.translate(-cx, -cy);
      ctx.globalAlpha = (1 - fade) ** 2;
      ctx.drawImage(source, 0, 0, w, h);
      // A soft, staggered dissolution leaves the actual label untouched.
      const tile = 10;
      for (let y = 0; y < h; y += tile) for (let x = 0; x < w; x += tile) {
        const delay = ((x * 17 + y * 31) % 101) / 101 * .25;
        const local = Math.max(0, Math.min(1, (fade - delay) / .75));
        ctx.globalAlpha = (1 - local) * local * .65;
        const inset = local * tile / 2;
        ctx.drawImage(background, x, y, Math.min(tile, w-x), Math.min(tile, h-y), x+inset, y+inset-local*12, tile-2*inset, tile-2*inset);
      }
      ctx.globalAlpha = 1; ctx.drawImage(label, 0, 0); ctx.restore();
      if (t < 1) frame = requestAnimationFrame(draw);
      else { cleanup(); resolve(); }
    };
    frame = requestAnimationFrame(draw);
  });
}

async function runRecognize(selectedBottleBox = null) {
  if (!photo || controller) return;
  const endpoint = window.SCANNER_CONFIG?.recognitionEndpoint;
  if (!endpoint) { notice('Фото готово. Распознавание ещё не подключено — снимок никуда не отправлен. Пока можно открыть пример карточки ниже.'); return; }
  notice(''); controller = new AbortController(); const active = controller;
  const timer = setTimeout(() => active.abort('timeout'), 90000);
  const started = performance.now();
  const cutout = $('scanning-label').parentElement;
  cutout.querySelector('canvas')?.remove(); cutout.classList.remove('revealing');
  $('scanning-label').src = photoUrl;
  $('processing-status').textContent = 'Определяем бутылки на фото…';
  $('scanner-screen').hidden = false;
  $('result-screen').hidden = true;
  startProcessingLog();
  $('processing').hidden = false;
  try {
    let targetBottleBox = selectedBottleBox;
    const bottleEndpoint = window.SCANNER_CONFIG?.bottleDetectionEndpoint;
    if (!targetBottleBox && window.SCANNER_CONFIG?.bottleDetectionEnabled && bottleEndpoint && $('photo-preview').naturalWidth) {
      try {
        const detectionBody = new FormData(); detectionBody.append('image', photo);
        const detectionResponse = await fetch(bottleEndpoint, { method:'POST', body:detectionBody, signal:active.signal });
        if (detectionResponse.ok) {
          const bottleResult = await detectionResponse.json();
          addProcessingLog(`Найдено бутылок: ${Number(bottleResult.count) || 0}`);
          if (bottleResult.needs_selection && showBottleChoices(bottleResult.candidates)) return;
          if (Array.isArray(bottleResult.primary_box)) targetBottleBox = bottleResult.primary_box;
        } else {
          addProcessingLog('Детектор бутылок недоступен; продолжим автоматический поиск');
        }
      } catch (error) {
        if (active.signal.aborted) throw error;
        addProcessingLog('Не удалось определить бутылки; продолжим автоматический поиск');
      }
    }
    $('processing-status').textContent = 'Ищем этикетку и вино в каталоге…';
    const body = new FormData(); body.append('image', photo);
    if (targetBottleBox) body.append('bottle_box', JSON.stringify(targetBottleBox));
    const ocrOn = debugOcrEnabled();
    const url = new URL(endpoint, location.origin);
    if (debugMode) url.searchParams.set('ocr', ocrOn ? '1' : '0');
    const headers = debugMode ? { 'X-Scanner-Debug': '1', 'X-Scanner-OCR': ocrOn ? '1' : '0' } : {};
    if (debugMode && lastCompareSlug) headers['X-Scanner-Compare-Slug'] = lastCompareSlug;
    const response = await fetch(url.toString(), { method:'POST', headers, body, signal:active.signal });
    if (response.status === 403) { notice('Сервер отклонил debug-запрос.'); return; }
    if (!response.ok) throw new Error('service');
    const result = await response.json();
    const elapsed = performance.now() - started;
    addProcessingLog(`Ответ сервера получен за ${formatElapsed(elapsed)}`);
    addProcessingLog(result.recognition?.ocr_enabled ? 'Сервер: OCR был включён' : 'Сервер: OCR был выключен');
    clearTimeout(timer);
    await revealLabel(result.recognition?.label_detection, active.signal);
    if (active.signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (result.status === 'matched') {
      if (!result.wine?.name || !(result.wine.slug || result.slug)) throw new Error('contract');
      showWine({ ...result.wine, slug:result.wine.slug || result.slug, image_url:result.wine.image_url || result.wine.imageUrl || result.wine.photo_name, demo:false }, { result, elapsed });
      return;
    }
    if (result.status === 'unknown' || result.status === 'uncertain') {
      showNotFound(result, elapsed);
      return;
    }
    throw new Error('contract');
  } catch (err) {
    notice(active.signal.aborted ? (active.signal.reason === 'timeout' ? 'Поиск занял слишком много времени. Попробуйте ещё раз.' : 'Поиск отменён. Можно выбрать другое фото.') : 'Сервис распознавания сейчас недоступен или вернул неполную карточку. Попробуйте позже.');
  } finally { clearTimeout(timer); controller = null; stopProcessingLog(); $('processing').hidden = true; }
}
$('cancel-request').onclick = () => controller?.abort('user');
function wineSlugFromPath() {
  const match = location.pathname.match(/^\/scanner\/([^/]+)\/?$/);
  if (!match) return '';
  try { return decodeURIComponent(match[1]); } catch { return ''; }
}
function scannerHref(slug) {
  const url = new URL(location.href);
  url.pathname = slug ? `/scanner/${encodeURIComponent(slug)}` : '/scanner';
  return `${url.pathname}${url.search}`;
}
function setScannerUrl(slug, { replace = false } = {}) {
  const next = scannerHref(slug);
  if (next === `${location.pathname}${location.search}`) return;
  history[replace ? 'replaceState' : 'pushState']({ scannerSlug: slug || '' }, '', next);
}
function showWine(wine, scanMeta = null, options = {}) {
  stopCamera(); currentWine = wine; $('scanner-screen').hidden = true; $('result-screen').hidden = false;
  if (!options.skipUrl && wine?.slug && !wine.demo) setScannerUrl(wine.slug);
  const image = resolveImageUrl(wine.image_url, wine.photo_name || wine.image_name || wine.photoName);
  const rating = formatRating(wine.public_rating);
  const dishes = Array.isArray(wine.dishes) ? wine.dishes.filter(Boolean) : [];
  const grapes = Array.isArray(wine.grapes) ? wine.grapes.filter(Boolean) : (wine.grapes ? [wine.grapes] : []);
  const wineColor = inferWineColor(wine);
  const grapeImage = grapeBackdrop(wine, grapes);
  const regionImage = catalogAsset('region', wine.region) || safeImage(wine.region_image_url);
  const tasteProfile = deriveTasteProfile(wine);
  const aromaNotes = extractAromaNotes(wine);
  const fitMarkup = fitMarkupFor(wine);
  const scanDebugMarkup = scanMeta ? renderScanDebug(scanMeta) : '';
  const grapeFactImage = grapes.map(grape => catalogAsset('grape', grape)).find(Boolean) || safeImage(wine.grape_image_url);
  const colorNote = String(wine.color || '').trim();
  const visualFacts = [
    wine.region ? `<article class="visual-fact">${regionImage ? `<img src="${escape(regionImage)}" alt="" loading="lazy">` : ''}<div><small>РЕГИОН</small><strong>${escape(wine.region)}</strong></div></article>` : '',
    grapes.length ? `<article class="visual-fact">${grapeFactImage ? `<img src="${escape(grapeFactImage)}" alt="" loading="lazy">` : ''}<div><small>СОРТ</small><strong>${escape(grapes.join(', '))}</strong></div></article>` : '',
    (wine.category || wineColor) ? `<article class="visual-fact visual-fact-color color-tone-${colorTone(wineColor)}"><div><small>КАТЕГОРИЯ И ЦВЕТ</small><strong>${escape(wine.category || wineColor || 'Категория не указана')}</strong>${colorNote && colorNote !== wine.category ? `<em>${escape(colorNote)}</em>` : ''}</div></article>` : ''
  ].filter(Boolean).join('');
  const dishImageUrls = Array.isArray(wine.dish_image_urls) ? wine.dish_image_urls : [];
  const dishCards = dishes.map((dish, index) => {
    const dishImage = catalogAsset('dish', dish) || safeImage(dishImageUrls[index]);
    return `<article class="dish-card">${dishImage ? `<img src="${escape(dishImage)}" alt="" loading="lazy">` : '<span class="dish-placeholder">Вино и еда</span>'}<strong>${escape(dish)}</strong></article>`;
  }).join('');
  $('result-screen').innerHTML = `
    <div class="result-top"><button class="text-button" id="back-to-scanner">${icon('arrow-left')} К сканеру</button>${wine.demo ? '<span class="demo-badge">Пример карточки · не результат сканирования</span>' : ''}</div>
    ${scanDebugMarkup}
    <div class="result-layout">
    <div class="wine-hero"><div class="wine-portrait">${grapeImage ? `<img class="wine-grape-backdrop" src="${escape(grapeImage)}" alt="" aria-hidden="true">` : ''}${image ? `<img class="wine-bottle" src="${escape(image)}" alt="${escape(wine.name)}" fetchpriority="high"/>` : '<span>Фото пока нет</span>'}</div>
    <div class="wine-details"><p class="eyebrow">${escape(wine.winery)}</p><h1>${wineTitle(wine.name)}</h1><p class="wine-category">${escape(wine.category || '')}${wine.region ? ' · ' + escape(wine.region) : ''}</p>
    <div class="public-rating" aria-label="Народный рейтинг ${escape(rating)} из 5"><div class="public-rating-badge">${ratingGlass()}<span>${escape(rating)}</span><small>Народный рейтинг</small></div></div>
    ${visualFacts ? `<section class="catalog-visuals" aria-label="Характеристики из каталога"><div class="visual-facts">${visualFacts}</div></section>` : ''}
    </div><div class="wine-facts">${wine.alcohol ? `<span><small>КРЕПОСТЬ</small>${escape(wine.alcohol)}</span>` : ''}${wine.volume ? `<span><small>ОБЪЁМ</small>${escape(wine.volume)}</span>` : ''}${wine.year ? `<span><small>ГОД</small>${escape(wine.year)}</span>` : ''}${wine.temperature ? `<span><small>ПОДАВАТЬ</small>${escape(wine.temperature)}</span>` : ''}${grapes.length ? `<span><small>СОРТ</small>${escape(grapes.join(', '))}</span>` : ''}</div>
    <button class="portrait-save icon-action" id="save-wine" type="button" aria-label="Сохранить вино">${icon('bookmark')}</button></div>

    <div class="result-actions"><button class="button secondary" id="compare-wine" type="button">${icon('compare')} Сравнить</button><button class="button secondary" id="scan-again">${icon('camera')} Сканировать другое</button></div>
    ${fitMarkup}
    <div class="result-split">
    <div class="result-taste"><section class="taste-profile" aria-labelledby="taste-profile-title"><div class="taste-profile-heading"><h2 id="taste-profile-title">Вкусовые свойства</h2><span>Оценка по данным каталога</span></div>${tasteDimensions.map(({ key, label }) => `<div class="taste-row"><span>${escape(label)}</span><i><em style="width:${tasteProfile[key] * 20}%"></em></i><b>${tasteProfile[key]}/5</b></div>`).join('')}</section>
    <section class="rating-combo" aria-label="Ваша оценка вина"><div class="user-rating" aria-labelledby="user-rating-title"><p class="user-rating-cta" id="user-rating-title">Поставь свою оценку</p><div class="rating-options" role="radiogroup" aria-label="Оценка вина">${[1,2,3,4,5].map(value => `<button class="rating-option" type="button" role="radio" aria-checked="false" aria-label="${value} из 5" data-user-rating="${value}">${userRatingIcons()}</button>`).join('')}</div><p class="rating-status" id="rating-status" hidden></p></div></section>
    <p class="wine-summary">${escape(wine.summary || wine.description || 'Описание пока не добавлено.')}</p><div class="taste-tags">${aromaNotes.map(tag => `<span>${escape(tag)}</span>`).join('')}</div></div>
    ${dishes.length ? `<section class="dish-pairings"><div class="dish-pairings-heading"><div>${icon('utensils')}<h2>Сочетание с блюдами</h2></div></div><div class="dish-grid">${dishCards}</div></section>` : ''}
    </div>
    </div>
    <section class="after-search" id="after-search">
      <div class="dish-pairings-heading"><div>${icon('compare')}<h2>Похожие вина</h2></div></div>
      <p class="muted" id="alternatives-status">Подбираем вина из каталога…</p>
      <div class="alt-grid" id="alternatives-grid"></div>
      <section class="sommelier-panel" aria-label="Цифровой сомелье">
        <div class="sommelier-hero">
          <div>
            <h2>Цифровой сомелье</h2>
            <p class="sommelier-sub">Подберём вино за 3 шага</p>
          </div>
        </div>
        <div class="sommelier-step">
          <p class="sommelier-step-title"><span class="sommelier-num">1</span> С чем будете пить?</p>
          <div class="food-buttons" id="sommelier-occasions">
            <button type="button" data-occasion="fish">Рыба</button>
            <button type="button" data-occasion="meat">Мясо</button>
            <button type="button" data-occasion="cheese">Сыр</button>
            <button type="button" data-occasion="dessert">Десерт</button>
            <button type="button" data-occasion="aperitif">Без еды</button>
          </div>
        </div>
        <div class="sommelier-step" id="sommelier-step-color" hidden>
          <p class="sommelier-step-title"><span class="sommelier-num">2</span> Какое вино?</p>
          <div class="food-buttons" id="sommelier-colors">
            <button type="button" data-color="any">Любое</button>
            <button type="button" data-color="white">Белое</button>
            <button type="button" data-color="red">Красное</button>
            <button type="button" data-color="rose">Розовое</button>
            <button type="button" data-color="sparkling">Игристое</button>
          </div>
        </div>
        <div class="sommelier-step" id="sommelier-step-sweet" hidden>
          <p class="sommelier-step-title"><span class="sommelier-num">3</span> По вкусу?</p>
          <div class="food-buttons" id="sommelier-sweetness">
            <button type="button" data-sweetness="any">Любое</button>
            <button type="button" data-sweetness="dry">Сухое</button>
            <button type="button" data-sweetness="semi_dry">Полусухое</button>
            <button type="button" data-sweetness="semi_sweet">Полусладкое</button>
            <button type="button" data-sweetness="sweet">Сладкое</button>
          </div>
        </div>
        <p class="sommelier-hint-card" id="sommelier-hint" hidden>Выберите, с чем будете пить — подберём вина из каталога.</p>
        <p class="sommelier-found" id="sommelier-found" hidden></p>
        <div class="alt-grid sommelier-grid" id="sommelier-grid"></div>
      </section>
    </section>
    ${wine.demo ? `<section class="pairings"><div class="pairings-heading">${icon('utensils')}<h2>Что у вас на ужин?</h2></div><p class="muted">Выберите блюдо — подскажем, как оно сочетается с этим стилем вина.</p><div class="food-buttons"><button data-food="fish" aria-pressed="true">Рыба и морепродукты</button><button data-food="cheese" aria-pressed="false">Мягкий сыр</button><button data-food="salad" aria-pressed="false">Лёгкий салат</button><button data-food="steak" aria-pressed="false">Стейк</button><button data-food="dessert" aria-pressed="false">Десерт</button></div><p class="pairing-explanation" id="pairing-explanation"></p></section>` : ''}`;
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
    $('rating-status').textContent = normalized ? `Ваша оценка: ${normalized} из 5` : '';
    const cta = $('user-rating-title');
    if (cta) cta.textContent = normalized ? `Ваша оценка: ${normalized} из 5` : 'Поставь свою оценку';
    refreshFitCard();
  };
  document.querySelectorAll('[data-user-rating]').forEach(button => button.onclick = () => setUserRating(Number(button.dataset.userRating)));
  document.querySelector('.wine-grape-backdrop')?.addEventListener('error', event => event.currentTarget.remove());
  document.querySelector('.wine-bottle')?.addEventListener('error', event => {
    const placeholder = document.createElement('span');
    placeholder.textContent = 'Фото пока нет';
    event.currentTarget.replaceWith(placeholder);
  });
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
  bindDebugPanel();
}
function showNotFound(result, elapsed) {
  stopCamera();
  currentWine = null;
  notice('');
  setScannerUrl('');
  $('scanner-screen').hidden = true;
  $('result-screen').hidden = false;
  const lookalikes = withLabelScores(result.lookalikes, result).filter(item => item?.slug);
  const alternatives = Array.isArray(result.alternatives) ? result.alternatives.filter(item => item?.slug) : [];
  const scanDebugMarkup = renderScanDebug({ result, elapsed });
  const emptyHint = 'Попробуйте снять этикетку ближе, без бликов и соседних бутылок.';
  $('result-screen').innerHTML = `
    <div class="result-top"><button class="text-button" id="back-to-scanner">${icon('arrow-left')} К сканеру</button><span class="demo-badge">В каталоге нет точного совпадения</span></div>
    ${scanDebugMarkup}
    <section class="not-found-banner">
      <p class="eyebrow">Результат сканирования</p>
      <h1>Такого вина у нас нет</h1>
      <p class="muted">Ниже — похожие вина по этикетке или по вкусу.</p>
    </section>
    <section class="after-search" id="after-search">
      <div class="dish-pairings-heading"><div>${icon('camera')}<h2>Похожие по этикетке</h2></div><span>Из каталога</span></div>
      ${lookalikes.length ? '' : `<p class="muted" id="lookalikes-status">Похожих этикеток в каталоге нет. ${emptyHint}</p>`}
      <div class="alt-grid" id="lookalikes-grid">${lookalikes.map(wineMiniCard).join('')}</div>
      <div class="dish-pairings-heading sommelier-heading"><div>${icon('compare')}<h2>Похожие по вкусу</h2></div><span>Другие винодельни, близкий стиль</span></div>
      <p class="muted" id="alternatives-status">${alternatives.length ? 'Если этикетка не совпала, посмотрите вина похожего стиля.' : 'Похожих вин по вкусу пока нет.'}</p>
      <div class="alt-grid" id="alternatives-grid">${alternatives.map(wineMiniCard).join('')}</div>
    </section>
    <div class="result-actions"><button class="button secondary" id="scan-again">${icon('camera')} Сканировать другое</button></div>`;
  $('back-to-scanner').onclick = $('scan-again').onclick = backToScanner;
  bindMiniCards($('lookalikes-grid'));
  bindMiniCards($('alternatives-grid'));
  $('result-screen').focus({ preventScroll:true });
  window.scrollTo({ top:0, behavior:'instant' });
  if (debugMode) showDiagnostics(result, elapsed);
  bindDebugPanel();
}
function backToScanner(options = {}) {
  $('result-screen').hidden = true;
  $('scanner-screen').hidden = false;
  if (!options.skipUrl) setScannerUrl('');
  $('show-example')?.focus({ preventScroll:true });
  window.scrollTo({ top:0, behavior:'instant' });
}
async function openWineFromSlug(slug, { skipUrl = true } = {}) {
  try {
    const response = await fetch(`/v1/catalog/${encodeURIComponent(slug)}`);
    if (!response.ok) throw new Error('catalog');
    const wine = await response.json();
    showWine({ ...wine, demo: false }, null, { skipUrl });
  } catch {
    setScannerUrl('', { replace: true });
    notice('Это вино не найдено в каталоге.');
  }
}
function wineMiniCard(item) {
  const slug = item.slug || '';
  const href = scannerHref(slug);
  const image = resolveImageUrl(item.image_url, item.photo_name || item.image_name);
  const score = formatScorePct(item.label_score);
  const reason = item.recommend_reason;
  const rating = Number(item.public_rating);
  const ratingMarkup = Number.isFinite(rating) && rating > 0
    ? `<span class="sommelier-rating" aria-label="Оценка ${rating.toFixed(1)}">★ ${rating.toFixed(1)}</span>`
    : '';
  return `<a class="alt-card${reason ? ' sommelier-card' : ''}" href="${escape(href)}" data-open-slug="${escape(slug)}">${image ? `<img src="${escape(image)}" alt="" loading="lazy">` : '<span class="dish-placeholder">Нет фото</span>'}<div><small>${escape(item.winery || '')}</small><strong>${escape(item.name || '')}</strong><em>${escape([item.category, item.region].filter(Boolean).join(' · '))}</em>${reason ? `<span class="sommelier-reason">${escape(reason)}</span>` : ''}${ratingMarkup}</div>${score ? `<span class="alt-card-score" aria-label="Сходство этикетки ${escape(score)}">${escape(score)}</span>` : ''}</a>`;
}
function bindMiniCards(root, fallback) {
  root?.querySelectorAll('img').forEach(img => {
    img.addEventListener('error', () => {
      const placeholder = document.createElement('span');
      placeholder.className = 'dish-placeholder';
      placeholder.textContent = 'Нет фото';
      img.replaceWith(placeholder);
    });
  });
  root?.querySelectorAll('[data-open-slug]').forEach(card => {
    card.onclick = async event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const slug = card.getAttribute('data-open-slug');
      try {
        const response = await fetch(`/v1/catalog/${encodeURIComponent(slug)}`);
        if (!response.ok) throw new Error('catalog');
        const wine = await response.json();
        showWine({ ...wine, demo: false });
      } catch {
        if (fallback) showWine(fallback);
        else if (card.getAttribute('href')) location.assign(card.getAttribute('href'));
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
  const sommelierState = { occasion: '', color: 'any', sweetness: 'any' };
  let sommelierAbort = null;
  const showSommelier = (id, on) => {
    const node = $(id);
    if (node) node.hidden = !on;
  };
  const askSommelier = () => {
    if (!sommelierState.occasion) return;
    showSommelier('sommelier-hint', true);
    showSommelier('sommelier-found', true);
    $('sommelier-hint').textContent = 'Подбираем…';
    sommelierAbort?.abort();
    sommelierAbort = new AbortController();
    fetch('/v1/sommelier', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: sommelierAbort.signal,
      body: JSON.stringify({
        occasion: sommelierState.occasion,
        color: sommelierState.color,
        sweetness: sommelierState.sweetness,
        slug: wine.slug || null,
      }),
    }).then(response => {
      if (!response.ok) throw new Error('sommelier');
      return response.json();
    }).then(data => {
      $('sommelier-hint').textContent = data.hint || '';
      const found = Number(data.found) || (data.wines || []).length;
      $('sommelier-found').textContent = found ? `Найдено ${found} вин` : 'Пока нет точных пар — ниже близкие варианты.';
      $('sommelier-grid').innerHTML = (data.wines || []).map(wineMiniCard).join('');
      bindMiniCards($('sommelier-grid'));
    }).catch(error => {
      if (error?.name === 'AbortError') return;
      $('sommelier-hint').textContent = 'Сомелье сейчас недоступен. Можно выбрать блюдо из карточки выше.';
    });
  };
  document.querySelectorAll('#sommelier-occasions [data-occasion]').forEach(button => {
    button.onclick = () => {
      sommelierState.occasion = button.dataset.occasion;
      document.querySelectorAll('#sommelier-occasions [data-occasion]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      showSommelier('sommelier-step-color', true);
      askSommelier();
    };
  });
  document.querySelectorAll('#sommelier-colors [data-color]').forEach(button => {
    button.onclick = () => {
      sommelierState.color = button.dataset.color || 'any';
      document.querySelectorAll('#sommelier-colors [data-color]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      showSommelier('sommelier-step-sweet', true);
      askSommelier();
    };
  });
  document.querySelectorAll('#sommelier-sweetness [data-sweetness]').forEach(button => {
    button.onclick = () => {
      sommelierState.sweetness = button.dataset.sweetness || 'any';
      document.querySelectorAll('#sommelier-sweetness [data-sweetness]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      askSommelier();
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
    if (!persist()) saved = previous;
  } else {
    saved = [{ ...currentWine }, ...saved].slice(0, 100);
    if (!persist()) saved = previous;
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
    document.querySelector('.result-actions')?.insertAdjacentHTML('afterend', markup);
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
window.addEventListener('popstate', () => {
  const slug = wineSlugFromPath();
  if (slug) openWineFromSlug(slug);
  else if (!$('result-screen').hidden) backToScanner({ skipUrl: true });
});
if (wineSlugFromPath()) openWineFromSlug(wineSlugFromPath());
