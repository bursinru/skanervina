# Своё Вино — сканер этикеток

Мобильный сканер: **Nuxt → Python (SigLIP 2 + OCR) → PostgreSQL / pgvector**. Запуск на сервере — Docker Compose из корня репозитория.

## Сборка для организаторов

Нужны Docker и отдельно каталог. В Git его нет: скопируйте в `Датасет/` таблицу `strapi_output0709_enriched.csv` и фотографии `prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads`. Дополнительные реальные фото уже лежат в репозитории, в `Датасет/extra-labels`. При копировании каталога эту папку не удаляйте.

```sh
cp .env.example .env
# В .env задайте POSTGRES_PASSWORD — длинная случайная hex-строка.
docker compose build
docker compose up -d db
docker compose run --rm recognition python -m app.import_catalog \
  --catalog /catalog/strapi_output0709_enriched.csv --index \
  --images /catalog/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads
docker compose up -d
curl http://127.0.0.1:3000/healthz
```

Сканер: `http://127.0.0.1:3000/scanner`. Первый запуск скачивает веса моделей, нужно несколько гигабайт свободного места. GPU не нужен.

Импорт сам подключает `Датасет/extra-labels`: папка называется slug вина, внутри её фотографии. Кропы этикеток нарезает алгоритм по студийным снимкам, отдельно их загружать не нужно. Если детектор уже меняли на заполненной базе, повторите импорт с `--force-crops` и перед этим остановите recognition.

Подробности API, порогов и резервной копии: [backend/README.md](backend/README.md).

## Что умеет сканер

- Камера и загрузка фото, поиск по этикетке и по целой бутылке.
- Карточка каталога, рейтинг, вкус, гастросочетания и цифровой сомелье.
- Поиск по названию. Сохранённые вина в анонимном профиле браузера.

## Проверки

```sh
npm run test:backend
npm run build:server
```
