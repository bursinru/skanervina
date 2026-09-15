from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .catalog import WineCatalog
from .recognition import Recognizer
from .settings import settings


class WineCard(BaseModel):
    slug: str
    name: str
    winery: str
    category: str = ""
    region: str = ""
    grapes: list[str] = Field(default_factory=list)
    image_url: Optional[str] = None
    description: str = ""
    public_rating: Optional[float] = None
    quality_rating: Optional[float] = None
    color: str = ""
    temperature: str = ""
    alcohol: str = ""
    dishes: list[str] = Field(default_factory=list)
    source_url: Optional[str] = None
    source: str = "Каталог «Своё Вино»"


class RecognizeResponse(BaseModel):
    status: Literal["matched", "uncertain", "unknown"]
    slug: Optional[str] = None
    wine: Optional[WineCard] = None
    confidence: Optional[float] = None
    recognition: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Своё Вино Recognition API", version="0.1.0")
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
    catalog = WineCatalog.from_csv(settings.catalog_csv, settings.image_base_url)
except Exception as error:
    catalog_error = str(error)

recognizer = Recognizer(catalog, settings) if catalog else None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "status": "ok" if catalog else "degraded",
        "catalog_loaded": catalog is not None,
        "catalog_size": catalog.size if catalog else 0,
        "catalog_error": catalog_error,
    }


def require_service() -> Recognizer:
    if not recognizer:
        raise HTTPException(
            status_code=503,
            detail="Catalog is not loaded. Set CATALOG_CSV to the mounted catalog CSV.",
        )
    return recognizer


async def recognize_upload(image: UploadFile) -> Dict[str, Any]:
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Supported image types: JPG, PNG, WebP.",
        )

    data = await image.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image is larger than 15 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Image is empty.")
    return require_service().recognize(data)


@app.post("/v1/recognize", response_model=RecognizeResponse)
async def recognize(image: UploadFile = File(...)) -> Dict[str, Any]:
    """Recognize a label and return the full wine card expected by the frontend."""
    return await recognize_upload(image)


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> Dict[str, Optional[str]]:
    """Compatibility endpoint for the supplied evaluation harness."""
    result = await recognize_upload(image)
    return {"slug": result.get("slug")}


@app.get("/v1/catalog/{slug}", response_model=WineCard)
def catalog_card(slug: str) -> Dict[str, Any]:
    service = require_service()
    wine = service.catalog.get(slug)
    if not wine:
        raise HTTPException(status_code=404, detail="Wine not found.")
    return wine.to_card()
