from __future__ import annotations

try:
    from .stock_predictor.cli import main
except ImportError:
    from stock_predictor.cli import main


if __name__ == "__main__":
    main()
