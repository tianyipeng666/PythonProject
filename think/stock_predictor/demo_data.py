from __future__ import annotations

import math
import random
from datetime import date, timedelta

from .data_loader import PriceBar


def make_demo_bars(days: int = 260, seed: int = 7) -> list[PriceBar]:
    rng = random.Random(seed)
    current = date(2025, 1, 2)
    price = 12.0
    bars: list[PriceBar] = []

    while len(bars) < days:
        if current.weekday() < 5:
            drift = 0.0006 + 0.004 * math.sin(len(bars) / 22)
            shock = rng.gauss(0, 0.018)
            close = max(1.0, price * (1 + drift + shock))
            open_price = price * (1 + rng.gauss(0, 0.006))
            high = max(open_price, close) * (1 + abs(rng.gauss(0, 0.008)))
            low = min(open_price, close) * (1 - abs(rng.gauss(0, 0.008)))
            volume = 8_000_000 * (1 + abs(rng.gauss(0, 0.35)))
            amount = volume * close
            bars.append(
                PriceBar(
                    date=current.isoformat(),
                    open=round(open_price, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=round(close, 2),
                    volume=round(volume, 0),
                    amount=round(amount, 0),
                )
            )
            price = close
        current += timedelta(days=1)
    return bars

