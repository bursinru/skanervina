from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import os
import asyncio
import secrets
import shutil
from pathlib import Path
from . import storage
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StrictInt
from .limits import BodyLimitMiddleware

from .catalog import WineCatalog
from .ranking import best_slug
from .recognition import Recognizer
from .recommend import alternatives as recommend_alternatives, sommelier_reply
from .settings import settings


class WineCard(BaseModel):
    demo: bool = False
    slug: str = Field(max_length=300)
    name: str = Field(max_length=1000)
    winery: str
    category: str = ""
    region: str = ""
    grapes: list[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    description: str = ""
    public_rating: Optional[float] = None
    quality_rating: Optional[float] = None
    color: str = ""
    region_image_url: Optional[str] = None
    grape_image_url: Optional[str] = None
    temperature: str = ""
    alcohol: str = ""
    dishes: list[str] = Field(default_factory=list)
    dish_image_urls: list[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source: str = "Каталог «Своё Вино»"
    label_score: Optional[float] = None


class RecognizeResponse(BaseModel):
    status: Literal["matched", "uncertain", "unknown"]
    slug: Optional[str] = None
    wine: Optional[WineCard] = None
    confidence: Optional[float] = None
    ranking: Dict[str, Any] = Field(default_factory=dict)
    alternatives: list[WineCard] = Field(default_factory=list)
    lookalikes: list[WineCard] = Field(default_factory=list)
    recognition: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Своё Вино Recognition API", version="0.1.0")
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

catalog: Optional[WineCatalog] = None
catalog_error: Optional[str] = None
try:
    if os.getenv('DATABASE_URL'):
        from .database import migrate
        migrate()
        catalog = WineCatalog.from_database(settings.image_base_url)
    else:
        catalog = WineCatalog.from_csv(settings.catalog_csv, settings.image_base_url)
except Exception as error:
    catalog_error = str(error)

recognizer = Recognizer(catalog, settings) if catalog else None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    database_ok = True
    indexed = 0
    if os.getenv('DATABASE_URL'):
        try:
            from .database import connect
            from .vision import MODEL_ID
            with connect() as db:
                indexed = db.execute('SELECT count(*) AS n FROM wine_embeddings WHERE model = %s', (MODEL_ID,)).fetchone()['n']
        except Exception:
            database_ok = False
    cv_required = os.getenv('CV_ENABLED', 'false').lower() == 'true'
    ready = bool(catalog and catalog.size and database_ok and
                 (not cv_required or (recognizer.visual_status == 'ready' and indexed > 0)))
    return {
        "status": "ok" if ready else "degraded",
        "catalog_loaded": catalog is not None and catalog.size > 0,
        "catalog_size": catalog.size if catalog else 0,
        "catalog_error": "Catalog unavailable" if catalog_error else None,
        "ocr_available": bool(shutil.which("tesseract")),
        "ocr_in_search": os.getenv("CV_OCR_ENABLED", "false").lower() == "true",
        "storage": "postgresql" if os.getenv("DATABASE_URL") else "sqlite",
        "database_available": database_ok,
        "visual_search": recognizer.visual_status if recognizer else "unavailable",
        "cv_device": recognizer.visual.encoder.device if recognizer and recognizer.visual else None,
        "indexed_images": indexed,
    }


@app.get('/readyz')
def readyz(response: Response):
    state = healthz()
    if state['status'] != 'ok':
        response.status_code = 503
    return state


def is_debug_request(request: Request) -> bool:
    """Allow unredacted telemetry only for an explicit debug request."""
    return request.headers.get('X-Scanner-Debug') == '1'


def require_service() -> Recognizer:
    if not recognizer or not recognizer.catalog.size:
        raise HTTPException(
            status_code=503,
            detail="Catalog is not loaded. Set CATALOG_CSV to the mounted catalog CSV.",
        )
    return recognizer


inference_slots = asyncio.Semaphore(2)


async def recognize_upload(image: UploadFile, mode: str = "combined", include_candidates: bool = False) -> Dict[str, Any]:
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    declared = (image.content_type or "").split(";")[0].strip().lower()
    data = await image.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Image is empty.")
    sniffed = None
    if data[:3] == b"\xff\xd8\xff":
        sniffed = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        sniffed = "image/png"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        sniffed = "image/webp"
    if declared not in allowed_types and not (
        sniffed and declared in {"", "application/octet-stream", "application/x-www-form-urlencoded"}
    ):
        raise HTTPException(
            status_code=415,
            detail="Supported image types: JPG, PNG, WebP.",
        )
    if not sniffed and declared not in allowed_types:
        raise HTTPException(status_code=415, detail="Supported image types: JPG, PNG, WebP.")
    service = require_service()
    try:
        await asyncio.wait_for(inference_slots.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(429, 'Recognition is busy; retry shortly')
    try:
        return await run_in_threadpool(service.recognize, data, mode, include_candidates)
    except Exception:
        import logging
        logging.exception('Recognition service failed')
        raise HTTPException(503, 'Recognition temporarily unavailable')
    finally:
        inference_slots.release()


@app.post("/v1/recognize", response_model=RecognizeResponse)
async def recognize(request: Request, response: Response, image: UploadFile = File(...)) -> Dict[str, Any]:
    """One best card; explicit debug requests may expose telemetry without a token."""
    token = request.headers.get('X-Scanner-Admin', '')
    expected = os.getenv('SCANNER_ADMIN_TOKEN', '')
    debug_request = is_debug_request(request)
    if token and (not expected or not secrets.compare_digest(token.encode(), expected.encode())):
        raise HTTPException(403, 'Invalid administrator token')
    mode = request.headers.get('X-Scanner-Mode', 'combined')
    if mode not in Recognizer.MODES:
        raise HTTPException(422, 'Unsupported recognition mode')
    if mode != 'combined' and (not expected or not token) and not debug_request:
        raise HTTPException(403, 'Administrator token required for benchmark modes')
    include_candidates = request.headers.get('X-Scanner-Benchmark') == '1'
    if include_candidates and (not expected or not token) and not debug_request:
        raise HTTPException(403, 'Administrator token required for benchmark metrics')
    result = await recognize_upload(image, mode, include_candidates)
    response.headers['Cache-Control'] = 'no-store'
    if not token and not debug_request:
        result = {**result, 'confidence': None, 'recognition': {
            key: value for key, value in result.get('recognition', {}).items()
            if key in ('method', 'reason', 'similarity', 'threshold', 'min_margin')
        }}
    return result


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> Dict[str, Optional[str]]:
    """Top-1 slug for the organizer script, including uncertain visual matches."""
    result = await recognize_upload(image, include_candidates=True)
    return {"slug": best_slug(result)}


@app.get("/v1/catalog/{slug}", response_model=WineCard)
def catalog_card(slug: str) -> Dict[str, Any]:
    service = require_service()
    wine = service.catalog.get(slug)
    if not wine:
        raise HTTPException(status_code=404, detail="Wine not found.")
    return wine.to_card()


class ProfileState(BaseModel):
    saved: list[WineCard] = Field(default_factory=list, max_length=100)
    ratings: Dict[str, StrictInt] = Field(default_factory=dict)


def session(request: Request, response: Response):
    token, state = storage.profile(request.cookies.get("scanner_session"))
    response.set_cookie("scanner_session", token, httponly=True, samesite="strict",
                        secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
                        max_age=60 * 60 * 24 * 365)
    response.headers["Cache-Control"] = "no-store"
    return token, state


@app.get("/v1/profile")
def get_profile(request: Request, response: Response):
    return session(request, response)[1]


@app.put("/v1/profile")
def put_profile(state: ProfileState, request: Request, response: Response):
    # Custom header forces a CORS preflight for cross-origin browser writes.
    if request.headers.get("X-Scanner-Client") != "web":
        raise HTTPException(403, "Missing client header")
    if len(state.ratings) > 1000 or any(type(v) is not int or not 1 <= v <= 5 for v in state.ratings.values()):
        raise HTTPException(422, "Ratings must be integers between 1 and 5; maximum 1000 entries")
    token, _ = session(request, response)
    payload = state.model_dump()
    if len(str(payload)) > 500_000:
        raise HTTPException(413, "Profile too large")
    storage.save(token, payload)
    return payload


@app.get("/v1/catalog/{slug}/alternatives")
def catalog_alternatives(slug: str) -> Dict[str, Any]:
    service = require_service()
    wine = service.catalog.get(slug)
    if not wine:
        raise HTTPException(status_code=404, detail="Wine not found.")
    return {"items": recommend_alternatives(service.catalog, wine)}


class SommelierRequest(BaseModel):
    occasion: str = Field(min_length=2, max_length=40)
    slug: Optional[str] = Field(default=None, max_length=300)


@app.post("/v1/sommelier")
def sommelier(body: SommelierRequest) -> Dict[str, Any]:
    service = require_service()
    current = service.catalog.get(body.slug) if body.slug else None
    result = sommelier_reply(service.catalog, body.occasion, current)
    if result.get("error"):
        raise HTTPException(422, "Unknown occasion")
    return result


@app.get("/v1/search")
def search(q: str = ""):
    if not 2 <= len(q.strip()) <= 200:
        raise HTTPException(422, "Query must contain 2–200 characters")
    return {"items": [match.wine.to_card() for match in require_service().catalog.search(q, limit=20) if match.score >= 0.25]}


PUBLIC = Path(os.getenv("STATIC_DIR", str(Path(__file__).resolve().parents[2] / "public")))


@app.get("/config.js")
def config():
    import json
    return Response("window.SCANNER_CONFIG = " + json.dumps({
        "recognitionEndpoint": "/v1/recognize", "profileEndpoint": "/v1/profile",
        "imageBaseUrl": settings.image_base_url,
    }) + ";", media_type="text/javascript")


@app.get("/scanner")
@app.get("/scanner/")
def scanner():
    return FileResponse(PUBLIC / "index.html")


if PUBLIC.is_dir():
    app.mount("/", StaticFiles(directory=PUBLIC, html=True), name="site")
