# Backend распознавания «Своё Вино»

Это первый backend-MVP. Он не обучает нейросеть: принимает изображение, проверяет его, пытается извлечь текст через Tesseract OCR и ищет совпадение в каталоге CSV. Визуальный поиск можно добавить позже внутри `app/recognition.py`, не меняя HTTP-контракт.

## Локальный запуск

Нужны Python 3.9+ и системный Tesseract с русским языком (`rus`).

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
 export CATALOG_CSV="../Датасет/strapi_output0709_enriched.csv"
export CORS_ORIGINS="http://localhost:4173"
uvicorn app.main:app --reload --port 8080
```

Проверки:

```sh
curl http://127.0.0.1:8080/healthz
curl -F "image=@../Датасет/eval/queries/019c68d0.jpg" http://127.0.0.1:8080/v1/recognize
```

Без установленного Tesseract сервис всё равно запускается, но возвращает `unknown` с причиной `binary_missing`. Это ожидаемое состояние текущего baseline.

## API

- `GET /healthz` — состояние сервиса и размер каталога.
- `POST /v1/recognize` — production-контракт для фронтенда; поле multipart `image`.
- `POST /v1/eval/predict` — совместимость с тестовым скриптом датасета, возвращает только `slug`.
- `GET /v1/catalog/{slug}` — получить карточку вина из каталога.

Обогащённый CSV дополнительно отдаёт публичный рейтинг, цвет, температуру подачи, крепость, гастросочетания, описание и прямой URL изображения. Если эти поля заполнены, backend использует их при формировании карточки. Файлы из датасета не копируются в Docker-образ и не должны попадать в GitHub.

## Docker

Собирать образ нужно из каталога `backend`. Каталог с данными монтируется отдельно:

```sh
docker build -t svoe-vino-recognition .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/../Датасет:/data:ro" \
  -e CATALOG_CSV=/data/strapi_output0709_enriched.csv \
  -e CORS_ORIGINS=http://localhost:4173 \
  svoe-vino-recognition
```

На staging/production CSV и индекс распознавания должны приходить из object storage или подключённого volume. Vercel используется для фронтенда, а не для запуска OCR/модели.
