"""Stage 4 Evaluation & Ablation Suite for SocraticCoder-7B."""

from .evaluator import (
    EvaluationReport,
    MethodEvaluationResult,
    SocraticBenchmarkEvaluator,
)
from .plotting import generate_publication_plots

__all__ = [
    "SocraticBenchmarkEvaluator",
    "MethodEvaluationResult",
    "EvaluationReport",
    "generate_publication_plots",
]
