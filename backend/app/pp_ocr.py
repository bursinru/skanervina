"""PP-OCRv5 for the grape line. Tesseract stays the fallback."""

import logging
from typing import Any, List

import numpy as np
from PIL import Image

# A weak Cyrillic read often comes back as Latin lookalikes: «Рубин» -> «Py6iH».
_LOOKALIKE = str.maketrans({
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "k": "к", "h": "н", "i": "и", "m": "м",
    "A": "а", "B": "в", "C": "с", "E": "е", "H": "н", "K": "к", "M": "м",
    "O": "о", "P": "р", "T": "т", "X": "х", "Y": "у",
    "6": "б", "0": "о", "3": "з",
})

_ENGINE = None
_FAILED = False


def fold_lookalikes(word: str) -> str:
    """Map a mixed or digit-stained token onto Cyrillic. Leave real Latin words."""

    has_digit = any(character.isdigit() for character in word)
    has_cyrillic = any("а" <= character.lower() <= "я" or character.lower() == "ё" for character in word)
    has_latin = any("a" <= character.lower() <= "z" for character in word)
    if has_digit or (has_cyrillic and has_latin):
        return word.translate(_LOOKALIKE)
    return word


def _engine():
    global _ENGINE, _FAILED
    if _FAILED:
        return None
    if _ENGINE is not None:
        return _ENGINE
    try:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        _ENGINE = RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.lang_type": LangRec.ESLAV,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    except Exception:
        logging.exception("PP-OCRv5 is unavailable; sibling labels stay on Tesseract")
        _FAILED = True
        return None
    return _ENGINE


def read_lines(image: Image.Image) -> List[dict]:
    """Text lines with boxes. Empty when the Cyrillic model is not installed."""

    engine = _engine()
    if engine is None or image is None:
        return []
    try:
        result = engine(np.asarray(image.convert("RGB")))
    except Exception:
        logging.exception("PP-OCRv5 failed on a label")
        return []
    texts = getattr(result, "txts", None) or ()
    scores = getattr(result, "scores", None) or ()
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    lines = []
    for text, score, box in zip(texts, scores, boxes):
        word = fold_lookalikes(str(text or "").strip())
        if float(score) < 0.45 or sum(character.isalnum() for character in word) < 3:
            continue
        points = box.tolist() if hasattr(box, "tolist") else box
        lines.append({"text": word, "score": round(float(score), 4), "box": points})
    return lines


def line_text(lines: List[dict]) -> str:
    return " ".join(str(line.get("text") or "") for line in lines if line.get("text"))
