from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from think.etf.models import MarketMove
from think.etf.config import INDEX_SPECS
from think.etf.data import DataProvider
from think.etf.database import parse_mysql_url
from think.etf.strategy import contribution_multiplier
from think.etf.valuation import (
    absolute_yield_score,
    classify_valuation,
    combined_valuation_score,
    metric_reference,
)


class ValuationTest(unittest.TestCase):
    def test_metric_reference_and_percentile(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=100, freq="30D"),
                "value": list(range(1, 101)),
            }
        )
        reference = metric_reference(frame, history_years=20)
        self.assertEqual(reference.current, 100)
        self.assertEqual(reference.percentile, 100)
        self.assertAlmostEqual(reference.q20, 20.8)
        self.assertEqual(reference.sample_count, 100)

    def test_classification_boundaries(self):
        self.assertEqual(classify_valuation(20), "明显低估")
        self.assertEqual(classify_valuation(35), "偏低")
        self.assertEqual(classify_valuation(65), "合理")
        self.assertEqual(classify_valuation(80), "偏高")
        self.assertEqual(classify_valuation(81), "明显高估")

    def test_large_yield_spread_offsets_high_relative_percentile(self):
        self.assertEqual(absolute_yield_score(9.0), 15.0)
        score = combined_valuation_score(relative_percentile=95.0, earnings_yield_spread=9.0)
        self.assertLess(score, 80.0)


class StrategyTest(unittest.TestCase):
    @staticmethod
    def move(return_5d: float, drawdown_60d: float) -> MarketMove:
        return MarketMove(
            latest_date=datetime.now().date().isoformat(),
            close=100.0,
            return_1d=-0.01,
            return_5d=return_5d,
            return_20d=-0.10,
            drawdown_60d=drawdown_60d,
            drawdown_250d=-0.20,
            volatility_20d=0.30,
        )

    def test_low_valuation_gets_dip_bonus(self):
        base, bonus, _ = contribution_multiplier("偏低", self.move(-0.09, -0.13))
        self.assertEqual(base, 1.20)
        self.assertEqual(bonus, 0.40)

    def test_high_valuation_does_not_chase_dip(self):
        base, bonus, reason = contribution_multiplier("偏高", self.move(-0.15, -0.20))
        self.assertEqual(base, 0.75)
        self.assertEqual(bonus, 0.0)
        self.assertIn("估值偏高", reason)

    def test_fair_valuation_caps_bonus(self):
        base, bonus, _ = contribution_multiplier("合理", self.move(-0.15, -0.20))
        self.assertEqual(base, 1.0)
        self.assertEqual(bonus, 0.25)


class ExactDataTest(unittest.TestCase):
    def test_mysql_exact_history_replaces_proxy(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=24, freq="MS"),
                "pe": [30 + i / 10 for i in range(24)],
                "pb": [4 + i / 100 for i in range(24)],
                "dividend_yield": [0.8] * 24,
                "quality": ["exact"] * 24,
            }
        )

        class FakeRepository:
            def best_exact_source(self, code, min_rows):
                return "wind"

            def load_valuations(self, code, source):
                return frame

            def latest_official_valuation(self, code):
                row = frame.iloc[-1]
                return {
                    "date": row["date"].date().isoformat(),
                    "pe": row["pe"], "pb": row["pb"],
                    "dividend_yield": row["dividend_yield"], "source": "wind",
                }

        provider = DataProvider(repository=FakeRepository(), offline=True)
        spec = INDEX_SPECS["399006"]
        pe = provider.pe_history(spec)
        pb = provider.pb_history(spec)
        snapshot = provider.official_snapshot(spec)
        self.assertEqual(len(pe), 24)
        self.assertIsNotNone(pb)
        self.assertTrue(provider.using_exact("399006"))
        self.assertAlmostEqual(snapshot.pe, 32.3)


class DatabaseConfigTest(unittest.TestCase):
    def test_parse_mysql_url(self):
        config = parse_mysql_url(
            "mysql://etf_user:p%40ss@127.0.0.1:3307/etf_data"
        )
        self.assertEqual(config.user, "etf_user")
        self.assertEqual(config.password, "p@ss")
        self.assertEqual(config.port, 3307)
        self.assertEqual(config.database, "etf_data")


if __name__ == "__main__":
    unittest.main()
