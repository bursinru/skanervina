"""Accept camera photos from iPhone (HEIC) and Android (JPEG, sometimes HEIF/AVIF)."""

from typing import Optional

ALLOWED_MIMES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/heic-sequence",
    "image/heif-sequence",
    "image/avif",
}
GENERIC_MIMES = {
    "",
    "application/octet-stream",
    "application/x-www-form-urlencoded",
    "binary/octet-stream",
}

_HEIF_BRANDS = {
    b"heic",
    b"heix",
    b"hevc",
    b"hevx",
    b"heim",
    b"heis",
    b"mif1",
    b"msf1",
    b"heif",
}
_AVIF_BRANDS = {b"avif", b"avis", b"MA1A", b"MA1B"}

_decoders_ready = False


def register_decoders() -> None:
    global _decoders_ready
    if _decoders_ready:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        register_avif = getattr(pillow_heif, "register_avif_opener", None)
        if register_avif:
            register_avif()
    except Exception:
        pass
    _decoders_ready = True


def sniff_image_type(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brands = {data[8:12]}
        for offset in range(16, min(len(data), 64), 4):
            brands.add(data[offset : offset + 4])
        if brands & _AVIF_BRANDS:
            return "image/avif"
        if brands & _HEIF_BRANDS:
            return "image/heic"
    return None


def is_allowed_upload(declared: str, data: bytes) -> bool:
    mime = (declared or "").split(";")[0].strip().lower()
    sniffed = sniff_image_type(data)
    if sniffed:
        return True
    return mime in ALLOWED_MIMES
