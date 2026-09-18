import unittest

from app.catalog import WineCatalog
from app.ranking import best_slug, ranking_metrics
from app.recommend import alternative_score, alternatives, sommelier_reply


class RecommendTests(unittest.TestCase):
    def setUp(self):
        self.catalog = WineCatalog.from_rows(
            [
                {
                    'Slug': 'massandra-muskatel',
                    'Название вина': 'Мускатель белый',
                    'Винодельня': 'Массандра',
                    'Категория': 'Белое сладкое',
                    'Цвет': 'Белое',
                    'Регион': 'Крым',
                    'Сорт винограда': 'Мускат Белый',
                    'svoe_vino_dishes_json': '["Десерт", "Выпечка"]',
                    'svoe_vino_public_rating': '5',
                    'svoe_vino_color': 'Белое',
                    'svoe_vino_category': 'Белое сладкое',
                },
                {
                    'Slug': 'solnechnaya-dolina-muskat',
                    'Название вина': 'Мускат белый',
                    'Винодельня': 'Солнечная Долина',
                    'Категория': 'Белое сладкое',
                    'Цвет': 'Белое',
                    'Регион': 'Крым',
                    'Сорт винограда': 'Мускат Белый',
                    'svoe_vino_dishes_json': '["Десерт"]',
                    'svoe_vino_public_rating': '4.5',
                    'svoe_vino_color': 'Белое',
                    'svoe_vino_category': 'Белое сладкое',
                },
                {
                    'Slug': 'fanagoria-cabernet',
                    'Название вина': 'Каберне',
                    'Винодельня': 'Фанагория',
                    'Категория': 'Красное сухое',
                    'Цвет': 'Красное',
                    'Регион': 'Кубань',
                    'Сорт винограда': 'Каберне Совиньон',
                    'svoe_vino_dishes_json': '["Мясо", "Стейк"]',
                    'svoe_vino_public_rating': '4.2',
                    'svoe_vino_color': 'Красное',
                    'svoe_vino_category': 'Красное сухое',
                },
            ],
            'https://example.com/',
        )

    def test_alternatives_come_from_other_wineries(self):
        wine = self.catalog.get('massandra-muskatel')
        items = alternatives(self.catalog, wine)
        self.assertEqual(items[0]['slug'], 'solnechnaya-dolina-muskat')
        self.assertNotEqual(items[0]['winery'], wine.winery)
        self.assertLess(alternative_score(wine, self.catalog.get('fanagoria-cabernet')), alternative_score(wine, self.catalog.get('solnechnaya-dolina-muskat')))

    def test_sommelier_picks_red_for_meat(self):
        result = sommelier_reply(self.catalog, 'meat', self.catalog.get('massandra-muskatel'))
        self.assertEqual(result['wines'][0]['slug'], 'fanagoria-cabernet')

    def test_ranking_f1_drops_when_neighbours_are_close(self):
        tight = ranking_metrics([{'slug': 'a', 'score': 0.95}, {'slug': 'b', 'score': 0.94}])
        split = ranking_metrics([{'slug': 'a', 'score': 0.95}, {'slug': 'b', 'score': 0.70}])
        self.assertLess(tight['f1_top1'], split['f1_top1'])
        self.assertEqual(best_slug({'status': 'uncertain', 'ranking': tight}), 'a')
