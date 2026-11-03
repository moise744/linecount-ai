"""
Compatibility shim: linecount.evaluation.evaluator
Exposes CountingEvaluator used by notebooks.
"""
from linecount.counting.counter import CountingEvaluator  # noqa: F401
from linecount.evaluation.metrics import compute_counting_metrics  # noqa: F401

__all__ = ["CountingEvaluator", "compute_counting_metrics"]
