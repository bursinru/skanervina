import unittest

from app.catalog import WineCatalog, normalize


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = WineCatalog.from_rows(
            [
                {
                    "Slug": "fanagoria-blanc",
                    "Название вина": "Blanc de Blancs",
                    "Винодельня": "Фанагория",
                    "Категория": "Игристое",
                    "Регион": "Кубань",
                    "Сорт винограда": "Шардоне",
                    "Описание": "Свежий брют",
                    "Название фото": "blanc de blancs.webp",
                },
                {
                    "Slug": "fanagoria-blanc",
                    "Название вина": "Дубликат",
                    "Винодельня": "Фанагория",
                },
            ],
            "https://api.vino-svoe.ru/uploads/",
        )

    def test_normalize_handles_cyrillic_yo(self):
        self.assertEqual(normalize("Ёж и Chardonnay"), "еж и chardonnay")

    def test_duplicate_slug_is_kept_once(self):
        self.assertEqual(self.catalog.size, 1)

    def test_search_returns_catalog_card_and_remote_image(self):
        matches = self.catalog.search("Фанагория Blanc de Blancs")
        self.assertEqual(matches[0].wine.slug, "fanagoria-blanc")
        self.assertGreater(matches[0].score, 0.55)
        self.assertIn("blanc%20de%20blancs.webp", matches[0].wine.image_url)

    def test_enriched_fields_are_returned(self):
        catalog = WineCatalog.from_rows(
            [
                {
                    "Slug": "fanagoria-blanc",
                    "Название вина": "Blanc de Blancs",
                    "Винодельня": "Фанагория",
                    "svoe_vino_public_rating": "4.6",
                    "svoe_vino_temperature": "8-10°C",
                    "svoe_vino_alcohol": "12%",
                    "svoe_vino_dishes_json": '["Рыба", "Сыры"]',
                    "svoe_vino_image_url": "https://api.vino-svoe.ru/image.webp",
                    "svoe_vino_source_url": "https://vino-svoe.ru/wines/fanagoria-blanc",
                }
            ],
            "https://api.vino-svoe.ru/uploads/",
        )
        card = catalog.get("fanagoria-blanc").to_card()
        self.assertEqual(card["public_rating"], 4.6)
        self.assertEqual(card["temperature"], "8-10°C")
        self.assertEqual(card["dishes"], ["Рыба", "Сыры"])
        self.assertEqual(card["image_url"], "https://api.vino-svoe.ru/image.webp")


if __name__ == "__main__":
    unittest.main()
