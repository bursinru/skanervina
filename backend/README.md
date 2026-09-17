# Своё Вино: локальный и серверный backend

## Архитектура

- **Nuxt 4 / Nitro** (`gateway/`) — публичный сервер на порту 3000: единый origin, проксирование только разрешённых маршрутов.
- **Python / FastAPI** — внутренний сервис карточек, профилей, SigLIP 2 и OCR на порту 8080. Существующий интерфейс раздаётся этим же сервисом через Nuxt; переписывать его на Vue не требуется.
- **PostgreSQL 17 + pgvector** — карточки JSONB, анонимные профили, векторы 768 измерений и HNSW-индекс.
- **SigLIP 2** `google/siglip2-base-patch16-224` — локальные image embeddings, cosine nearest neighbours. FixRes checkpoint использует класс `SiglipVisionModel`; revision закреплён в `app/vision.py`.
- **Tesseract rus+eng** — дополнительное подтверждение текста. При включённом CV отсутствие модели/индекса не маскируется OCR-результатом.
- **Strapi** — источник каталога. Есть импорт CSV и JSON; отдельная CMS/admin пока не разворачивается.

Фотографии пользователей обрабатываются локально и не сохраняются. При первой индексации скачиваются публичные веса модели; платных API/ключей нет. Бутылки и иллюстрации в интерфейсе пока загружаются с исходного CDN.

## Запуск на этой машине

Окружение Python установлено в `backend/.venv-cv`, настройки в `backend/.env.local` (не в Git). Из корня проекта:

```sh
# Если база была остановлена:
/opt/homebrew/opt/postgresql@17/bin/pg_ctl -D backend/data/postgres \
  -l backend/data/postgres.log \
  -o '-k /Users/bursinru/Sites/skanervina/backend/data/socket -p 55432 -h ""' start
npm run backend
# В другом терминале:
npm --prefix gateway run build
HOST=127.0.0.1 npm --prefix gateway start
# http://localhost:3000/scanner
```

Локальная база слушает только Unix-сокет внутри проекта, не TCP. В ней настроен trust для локального пользователя macOS; серверный вариант ниже использует пароль. Остановка локальной базы: `pg_ctl -D backend/data/postgres stop` с тем же полным путём к pg_ctl.

Для чистой установки нужны Python 3.11+, Tesseract rus+eng, PostgreSQL 17 с pgvector. Создайте venv, установите `requirements-cv.txt`, создайте базу, задайте `DATABASE_URL`, `HF_HOME`, `CV_ENABLED=true`. Пример переменных — `.env.example` в этом каталоге. Nuxt устанавливается через `npm --prefix gateway ci`; совместимый Node закреплён локальной dev-зависимостью, системный Node не заменяется.

```sh
python3.11 -m venv backend/.venv-cv
backend/.venv-cv/bin/pip install -r backend/requirements-cv.txt
# Экспортируйте DATABASE_URL и HF_HOME либо загрузите собственный .env через dotenv.
PYTHONPATH=backend backend/.venv-cv/bin/python -m app.import_catalog \
  --catalog Датасет/strapi_output0709_enriched.csv \
  --index --images Датасет/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads
```

Импорт повторяемый: карточки обновляются по slug, фотографии индексируются повторно только при изменении SHA-256 или модели. Используются только изображения каталога; папка `eval/queries` в индекс не включается. Пропущенные/повреждённые изображения выводятся в итоговом отчёте. После импорта карточек перезапустите Python-сервис: текстовый каталог кешируется в памяти.

## Обычный Linux-сервер: Docker Compose

В Docker локальная база и модели сохраняются в отдельных volumes. GPU не требуется; скорость зависит от CPU. Первый запуск скачивает веса, потребуется несколько гигабайт свободного места. Docker-сборку нужно выполнять **из корня репозитория**.

1. Скопируйте исходники и отдельно `Датасет/` (он исключён из Git).
2. Создайте корневой `.env`: `POSTGRES_PASSWORD=<случайная длинная hex-строка>`. При HTTPS добавьте `COOKIE_SECURE=true`. Hex-пароль не требует URL-кодирования в DATABASE_URL.
3. Выполните:

```sh
docker compose build
docker compose up -d db
docker compose run --rm recognition python -m app.import_catalog \
  --catalog /catalog/strapi_output0709_enriched.csv --index \
  --images /catalog/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads
docker compose up -d
curl http://127.0.0.1:3000/healthz
```

4. Настройте Nginx/Caddy с HTTPS перед `127.0.0.1:3000`. Пример Nginx: `deploy/nginx.conf.example` (поменяйте домен и добавьте TLS). HTTPS нужен для камеры телефона. Python и PostgreSQL не публикуются наружу.

Данные не удаляются при `docker compose down`; **`down -v` удаляет volumes**. Резервная копия:

```sh
docker compose exec -T db pg_dump -U scanner -d scanner -Fc > scanner.dump
# Восстановление в пустую базу с установленным pgvector:
# docker compose exec -T db pg_restore -U scanner -d scanner < scanner.dump
```

Для переноса текущих локальных карточек/профилей/векторов используйте `pg_dump` локальной базы и `pg_restore` в контейнер. Повторная индексация тогда не нужна. Веса можно заново скачать в model-cache; пользовательские фотографии там отсутствуют.

## API

- `GET /healthz` — каталог, хранилище, OCR, готовность CV.
- `POST /v1/recognize` — multipart `image`, JPG/PNG/WebP до 15 MiB. Возвращает одну карточку при `matched` или `uncertain`; при `unknown` карточки нет. Сомнительный результат явно помечается в интерфейсе.
- `POST /v1/eval/predict` — тот же поиск, ответ `{slug: string | null}`; сомнительные результаты остаются `null` для совместимости оценки.
- `GET /v1/search?q=Фанагория` — до 20 кандидатов текстового поиска.
- `GET /v1/catalog/{slug}` — карточка JSON.
- `GET /v1/profile` — сохранения/оценки текущего браузера, выдаёт HttpOnly SameSite=Strict cookie.
- `PUT /v1/profile` — `{saved: [card], ratings: {slug: 1..5}}`, заголовок `X-Scanner-Client: web`. До 100 сохранений и 1000 оценок; общий размер JSON ограничен до разбора.

Профили анонимные: это ещё не аккаунты с входом и синхронизацией между устройствами. Очистка cookie лишает доступа к прежнему профилю. Одновременные изменения из двух вкладок используют последнее сохранение всего профиля. При сетевой ошибке интерфейс сообщает о несохранённом на сервере изменении; локальная копия остаётся в браузере, автоматического offline-sync пока нет.

Без `DATABASE_URL` доступен прежний SQLite/OCR режим для лёгких тестов; он не является рекомендованным CV-стеком.

## Формат JSON / Strapi

Поддерживаются массив карточек, `{ "wines": [...] }`, `{ "data": [...] }` и элементы `{ "attributes": {...} }` с полями карточки. Обязательны `slug` и `name`:

```json
[{"slug":"wine-slug","name":"Название","winery":"Винодельня","image_name":"bottle.webp","grapes":["Шардоне"],"public_rating":4.2}]
```

Также принимаются JSON-строки с русскими названиями столбцов CSV. Произвольный внутренний дамп Strapi с отдельными таблицами media/relations требует адаптера к этому формату — не угадываем связи. Отсутствие подходящих записей вызывает ошибку импорта.

## Проверки и ограничения качества

```sh
backend/.venv-cv/bin/pip install -r backend/requirements-dev.txt
npm run test:backend
npm run build
npm run build:server
```

Визуальный similarity — мера близости векторов, **не вероятность правильного ответа**. Начальные пороги: `CV_MATCH_THRESHOLD=0.88`, `CV_MATCH_MARGIN=0.04`. При похожих кандидатах возвращается `uncertain`; для уверенного ответа нужны высокий similarity и отрыв от второго кандидата либо подтверждение OCR. Пороги ещё нужно калибровать на размеченных фотографиях полок/этикеток. В предоставленном eval есть три снимка, но правильные slug скрыты организатором: итоговую точность по ним не заявляем.

Технические первоисточники: [SigLIP 2](https://huggingface.co/docs/transformers/v4.57.1/model_doc/siglip2), [pgvector](https://github.com/pgvector/pgvector), [Nuxt server](https://nuxt.com/docs/4.x/directory-structure/server).

## Последняя проверка

Фактические результаты и ограничения: [VALIDATION.md](VALIDATION.md). Для повторной проверки запущенного стека используйте `backend/.venv-cv/bin/python scripts/check_backend.py` из корня проекта. На этой машине и в Compose выбран `CV_DEVICE=cpu`; прогон без GPU выполнен. Замеры CPU, визуального поиска и OCR: [отчёт](../reports/recognition/README.md).

## Обучение

Локальный pipeline обучения и протокол измерений: [reports/training/README.md](../reports/training/README.md). Первый завершённый эксперимент: [adapter-v1](../reports/training/adapter-v1.md). Артефакты обучения находятся в `backend/data/training/` и не входят в Docker/Git; рабочая модель автоматически не заменяется.


## Одна карточка, автоматическое выделение этикетки и диагностика

Пользователь отправляет исходный снимок. Перед OCR и визуальным поиском backend автоматически находит центральную прямоугольную панель этикетки, добавляет небольшой защитный отступ и использует её как рабочий crop. Исходные пользовательские фото не сохраняются. Анимация использует исходный снимок и учитывает reduced motion.

Для проверки pipeline доступны только администратору режимы `image_full`, `image_auto`, `image_auto_enhanced`, `ocr_full`, `ocr_auto`, `ocr_auto_enhanced` и `combined` через заголовок `X-Scanner-Mode`. Обычный запрос использует `combined`.

Для debug откройте `/scanner?debug=1`: ключ не нужен. Браузер передаёт явный заголовок `X-Scanner-Debug: 1`, по которому API возвращает технические метрики; обычные запросы без этого заголовка по-прежнему их скрывают. Для защищённого серверного доступа к диагностике и benchmark-режимам можно передавать `X-Scanner-Admin: SCANNER_ADMIN_TOKEN`. Ключ не добавляется в URL, не сохраняется в localStorage и не входит в Git.

В запрос передаётся `X-Scanner-Admin` либо явный `X-Scanner-Debug`. Без них API скрывает confidence, similarity и timings; неверный ключ даёт HTTP 403. Панель показывает сходство SigLIP в процентах, отрыв от второго соседа, подробное время этапов на сервере и полное ожидание в браузере. OCR score — оценка текстового сопоставления, а не уверенность Tesseract. Обе оценки не являются вероятностью правильного распознавания. Нет необходимости давать ключ посетителям; отдельной Strapi CMS-админки этот режим не заменяет.

На сервере используйте HTTPS. Внешние сервисы OCR не нужны: Tesseract работает локально. Для лучшего качества дальше нужны размеченные реальные фото с правильными slug, в том числе блики, разные ракурсы, похожие этикетки и вина вне каталога. Обученный экспериментальный адаптер пока не используется в публичном поиске.
