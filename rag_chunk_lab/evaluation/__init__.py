"""
Evaluation modules - Metrics, Analysis, and Domain-specific Evaluations
"""

from .evaluation import load_ground_truth, evaluate_local_proxy, try_ragas_eval, evaluate_embedding_quality
from .embedding_metrics import EmbeddingMetrics, EmbeddingQualityAnalyzer
from .embedding_analysis import run_comprehensive_embedding_analysis
from .generic_evaluation import run_generic_evaluation_suite
from .legal_evaluation import run_legal_evaluation_suite
from .azure_foundry_evaluation import integrate_with_azure_foundry
from .ground_truth_generator import GroundTruthGenerator, create_llm_client

__all__ = [
    'load_ground_truth',
    'evaluate_local_proxy',
    'try_ragas_eval',
    'evaluate_embedding_quality',
    'EmbeddingMetrics',
    'EmbeddingQualityAnalyzer',
    'run_comprehensive_embedding_analysis',
    'run_generic_evaluation_suite',
    'run_legal_evaluation_suite',
    'integrate_with_azure_foundry',
    'GroundTruthGenerator',
    'create_llm_client'
]