import unittest
from io import BytesIO
from PIL import Image
from app.image_io import is_allowed_upload, sniff_image_type


def _ftyp(*brands: bytes) -> bytes:
    body = b"ftyp" + brands[0] + b"\x00\x00\x00\x00" + b"".join(brands[1:])
    return (4 + len(body)).to_bytes(4, "big") + body


class ImageIoTests(unittest.TestCase):
    def test_sniff_common_phone_formats(self):
        self.assertEqual(sniff_image_type(b"\xff\xd8\xff\xe0rest"), "image/jpeg")
        self.assertEqual(sniff_image_type(b"\x89PNG\r\n\x1a\nrest"), "image/png")
        self.assertEqual(sniff_image_type(b"RIFF....WEBP...."), "image/webp")
        self.assertEqual(sniff_image_type(_ftyp(b"heic", b"mif1")), "image/heic")
        self.assertEqual(sniff_image_type(_ftyp(b"mif1", b"heic")), "image/heic")
        self.assertEqual(sniff_image_type(_ftyp(b"avif", b"mif1")), "image/avif")

    def test_iphone_heic_declared_or_generic(self):
        heic = _ftyp(b"heic", b"mif1") + b"\x00" * 16
        self.assertTrue(is_allowed_upload("image/heic", heic))
        self.assertTrue(is_allowed_upload("application/octet-stream", heic))
        self.assertTrue(is_allowed_upload("", heic))
        self.assertFalse(is_allowed_upload("text/plain", b"not-an-image"))

    def test_android_jpeg_empty_mime(self):
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 8
        self.assertTrue(is_allowed_upload("", jpeg))
        self.assertTrue(is_allowed_upload("image/jpg", jpeg))

    def test_decode_heif_roundtrip(self):
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            self.skipTest("pillow-heif is not installed")
        register_heif_opener()
        raw = BytesIO()
        Image.new("RGB", (12, 10), (180, 20, 40)).save(raw, format="HEIF")
        payload = raw.getvalue()
        self.assertEqual(sniff_image_type(payload), "image/heic")
        from app.recognition import Recognizer

        image, error = Recognizer._decode_image(payload)
        self.assertIsNone(error)
        self.assertEqual(image.size, (12, 10))
        self.assertEqual(image.mode, "RGB")
