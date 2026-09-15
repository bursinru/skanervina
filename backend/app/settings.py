import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CATALOG_CSV = REPO_ROOT / "Датасет" / "strapi_output0709.csv"
ENRICHED_CATALOG_CSV = REPO_ROOT / "Датасет" / "strapi_output0709_enriched.csv"
DEFAULT_CATALOG_CSV = (
    ENRICHED_CATALOG_CSV if ENRICHED_CATALOG_CSV.exists() else ORIGINAL_CATALOG_CSV
)
DEFAULT_IMAGE_BASE_URL = (
    "https://api.vino-svoe.ru/v1/img/str-api/800/800/resize/uploads/"
)


def _origins(value: str) -> Tuple[str, ...]:
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    catalog_csv: Path
    image_base_url: str
    max_image_bytes: int
    cors_origins: Tuple[str, ...]
    ocr_languages: str


settings = Settings(
    catalog_csv=Path(os.getenv("CATALOG_CSV", str(DEFAULT_CATALOG_CSV))),
    image_base_url=os.getenv("IMAGE_BASE_URL", DEFAULT_IMAGE_BASE_URL).rstrip("/") + "/",
    max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024))),
    cors_origins=_origins(
        os.getenv(
            "CORS_ORIGINS",
            "http://localhost:4173,http://localhost:4174",
        )
    ),
    ocr_languages=os.getenv("OCR_LANGUAGES", "rus+eng"),
)
