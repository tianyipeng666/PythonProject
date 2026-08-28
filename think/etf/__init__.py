"""指数估值和每周定投研究工具。"""

from think.etf.analyzer import analyze_index
from think.etf.config import INDEX_SPECS, IndexSpec

__all__ = ["INDEX_SPECS", "IndexSpec", "analyze_index"]
