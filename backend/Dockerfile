FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements*.txt ./
# CPU PyTorch avoids pulling CUDA libraries onto a regular server.
RUN pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-cv.txt
COPY backend/app ./app
COPY backend/migrations ./migrations
COPY public ./public
RUN useradd --uid 10001 --create-home scanner && mkdir /models /app/data && chown scanner /models /app/data
USER scanner
ENV PYTHONUNBUFFERED=1 STATIC_DIR=/app/public HF_HOME=/models
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--limit-concurrency", "16"]
