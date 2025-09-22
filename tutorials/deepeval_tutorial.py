"""
DeepEval Tutorial - Framework Complet d'Évaluation LLM
Documentation officielle: https://docs.confident-ai.com/
GitHub: https://github.com/confident-ai/deepeval
"""

import os
from typing import List, Dict
import json

def install_deepeval():
    """
    Installation de DeepEval
    """
    print("""
🔧 INSTALLATION DEEPEVAL

# Installation basique
pip install deepeval

# Installation avec toutes les dépendances
pip install deepeval[all]

# Pour utilisation avec OpenAI
pip install openai

# Pour utilisation avec tests unitaires
pip install pytest

# Vérification
python -c "import deepeval; print('DeepEval installé avec succès!')"

# Configuration initiale
deepeval login  # Pour utiliser Confident AI (optionnel)
""")

def setup_deepeval_basic():
    """
    Configuration de base DeepEval
    """
    setup_code = '''
# 1. Import des modules DeepEval
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# 2. Configuration des métriques de base
def setup_basic_metrics():
    """Configure les métriques DeepEval de base"""
    return [
        AnswerRelevancyMetric(threshold=0.7, model="gpt-3.5-turbo"),
        FaithfulnessMetric(threshold=0.8, model="gpt-3.5-turbo"),
        ContextualRelevancyMetric(threshold=0.7, model="gpt-3.5-turbo"),
        ContextualRecallMetric(threshold=0.7, model="gpt-3.5-turbo"),
        ContextualPrecisionMetric(threshold=0.7, model="gpt-3.5-turbo")
    ]

# 3. Configuration avancée avec métriques de sécurité
def setup_advanced_metrics():
    """Configure les métriques avancées incluant biais et toxicité"""
    return [
        # Métriques de base
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8),
        ContextualRelevancyMetric(threshold=0.7),

        # Métriques de sécurité
        HallucinationMetric(threshold=0.3),  # Plus bas = moins d'hallucinations
        BiasMetric(threshold=0.3),           # Détection de biais
        ToxicityMetric(threshold=0.2)        # Détection de toxicité
    ]

# 4. Création de test cases
def create_test_case(question: str, answer: str, ground_truth: str, context: List[str]) -> LLMTestCase:
    """Crée un cas de test DeepEval"""
    return LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=ground_truth,
        retrieval_context=context,
        # Métadonnées optionnelles
        metadata={
            "domain": "generic",
            "difficulty": "medium",
            "timestamp": "2024-12-01"
        }
    )
'''

    return setup_code

def integrate_deepeval_with_rag_chunk_lab():
    """
    Intégration DeepEval avec RAG Chunk Lab
    """
    integration_code = '''
# Integration avec RAG Chunk Lab
from deepeval import evaluate
from deepeval.metrics import *
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
import time
from pathlib import Path

class DeepEvalRAGEvaluator:
    def __init__(self, model: str = "gpt-3.5-turbo", include_safety_metrics: bool = True):
        self.model = model
        self.include_safety_metrics = include_safety_metrics
        self.metrics = self._setup_metrics()

    def _setup_metrics(self):
        """Configure les métriques DeepEval"""
        base_metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=self.model),
            FaithfulnessMetric(threshold=0.8, model=self.model),
            ContextualRelevancyMetric(threshold=0.7, model=self.model),
            ContextualRecallMetric(threshold=0.7, model=self.model),
            ContextualPrecisionMetric(threshold=0.7, model=self.model)
        ]

        if self.include_safety_metrics:
            safety_metrics = [
                HallucinationMetric(threshold=0.3, model=self.model),
                BiasMetric(threshold=0.3, model=self.model),
                ToxicityMetric(threshold=0.2, model=self.model)
            ]
            base_metrics.extend(safety_metrics)

        return base_metrics

    def evaluate_pipeline(self, doc_id: str, pipeline_name: str,
                         questions: List[str], answers: List[str],
                         ground_truth: List[str], contexts: List[List[str]]) -> Dict:
        """Évalue un pipeline avec DeepEval"""

        print(f"🔍 DeepEval evaluation for {pipeline_name}...")

        # Créer les test cases
        test_cases = []
        for i, (q, a, gt, ctx) in enumerate(zip(questions, answers, ground_truth, contexts)):
            test_case = LLMTestCase(
                input=q,
                actual_output=a,
                expected_output=gt,
                retrieval_context=ctx,
                metadata={
                    "pipeline": pipeline_name,
                    "doc_id": doc_id,
                    "question_id": i
                }
            )
            test_cases.append(test_case)

        # Créer le dataset
        dataset = EvaluationDataset(test_cases=test_cases)

        # Lancer l'évaluation
        start_time = time.time()
        results = evaluate(
            test_cases=test_cases,
            metrics=self.metrics,
            print_results=True,
            write_cache=True  # Cache les résultats pour éviter les re-évaluations
        )
        evaluation_time = time.time() - start_time

        # Analyser les résultats
        analysis = self._analyze_results(results, pipeline_name, evaluation_time)

        return analysis

    def _analyze_results(self, results, pipeline_name: str, evaluation_time: float) -> Dict:
        """Analyse les résultats DeepEval"""
        analysis = {
            "pipeline": pipeline_name,
            "evaluation_time": evaluation_time,
            "total_test_cases": len(results),
            "metrics_scores": {},
            "passed_tests": 0,
            "failed_tests": 0,
            "detailed_results": []
        }

        # Analyser chaque résultat
        for result in results:
            # Compter les succès/échecs
            if result.success:
                analysis["passed_tests"] += 1
            else:
                analysis["failed_tests"] += 1

            # Extraire les scores des métriques
            test_metrics = {}
            for metric_result in result.metrics:
                metric_name = metric_result.__class__.__name__
                score = metric_result.score
                success = metric_result.success

                test_metrics[metric_name] = {
                    "score": score,
                    "success": success,
                    "threshold": metric_result.threshold,
                    "reason": getattr(metric_result, "reason", "")
                }

                # Accumuler pour les moyennes
                if metric_name not in analysis["metrics_scores"]:
                    analysis["metrics_scores"][metric_name] = {
                        "scores": [],
                        "successes": 0,
                        "total": 0
                    }

                analysis["metrics_scores"][metric_name]["scores"].append(score)
                analysis["metrics_scores"][metric_name]["total"] += 1
                if success:
                    analysis["metrics_scores"][metric_name]["successes"] += 1

            analysis["detailed_results"].append({
                "input": result.input,
                "actual_output": result.actual_output,
                "expected_output": result.expected_output,
                "success": result.success,
                "metrics": test_metrics
            })

        # Calculer les moyennes
        for metric_name, metric_data in analysis["metrics_scores"].items():
            scores = metric_data["scores"]
            metric_data["mean_score"] = sum(scores) / len(scores) if scores else 0
            metric_data["success_rate"] = metric_data["successes"] / metric_data["total"]
            metric_data["min_score"] = min(scores) if scores else 0
            metric_data["max_score"] = max(scores) if scores else 0

        # Score global
        analysis["overall_success_rate"] = analysis["passed_tests"] / analysis["total_test_cases"]

        return analysis

    def compare_pipelines(self, results: Dict[str, Dict]) -> Dict:
        """Compare plusieurs pipelines"""
        comparison = {
            "pipelines": list(results.keys()),
            "best_pipeline": {},
            "detailed_comparison": {},
            "summary": {}
        }

        # Comparer chaque métrique
        for metric_name in ["AnswerRelevancyMetric", "FaithfulnessMetric", "ContextualRelevancyMetric"]:
            metric_comparison = {}

            for pipeline, pipeline_results in results.items():
                metric_data = pipeline_results.get("metrics_scores", {}).get(metric_name, {})
                metric_comparison[pipeline] = {
                    "mean_score": metric_data.get("mean_score", 0),
                    "success_rate": metric_data.get("success_rate", 0)
                }

            # Trouver le meilleur
            best_pipeline = max(metric_comparison.keys(),
                              key=lambda p: metric_comparison[p]["mean_score"])

            comparison["best_pipeline"][metric_name] = best_pipeline
            comparison["detailed_comparison"][metric_name] = metric_comparison

        # Meilleur pipeline global
        overall_scores = {}
        for pipeline in results.keys():
            overall_scores[pipeline] = results[pipeline]["overall_success_rate"]

        comparison["best_overall"] = max(overall_scores.keys(), key=lambda p: overall_scores[p])
        comparison["summary"] = overall_scores

        return comparison

    def generate_deepeval_report(self, results: Dict, output_dir: str = "deepeval_results"):
        """Génère un rapport DeepEval"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Rapport JSON détaillé
        report_file = f"{output_dir}/deepeval_evaluation_{results['pipeline']}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Rapport HTML (si disponible)
        try:
            self._generate_html_report(results, output_dir)
        except Exception as e:
            print(f"HTML report generation failed: {e}")

        print(f"📊 DeepEval report saved: {report_file}")
        return report_file

    def _generate_html_report(self, results: Dict, output_dir: str):
        """Génère un rapport HTML (optionnel)"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepEval Report - {results['pipeline']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .success {{ background-color: #d4edda; }}
                .failure {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>DeepEval Report - {results['pipeline']}</h1>
            <h2>Summary</h2>
            <p>Overall Success Rate: {results['overall_success_rate']:.2%}</p>
            <p>Passed Tests: {results['passed_tests']}</p>
            <p>Failed Tests: {results['failed_tests']}</p>
            <p>Evaluation Time: {results['evaluation_time']:.2f}s</p>

            <h2>Metrics</h2>
        """

        for metric_name, metric_data in results["metrics_scores"].items():
            success_class = "success" if metric_data["success_rate"] > 0.7 else "failure"
            html_content += f"""
            <div class="metric {success_class}">
                <h3>{metric_name}</h3>
                <p>Mean Score: {metric_data['mean_score']:.3f}</p>
                <p>Success Rate: {metric_data['success_rate']:.2%}</p>
                <p>Range: {metric_data['min_score']:.3f} - {metric_data['max_score']:.3f}</p>
            </div>
            """

        html_content += "</body></html>"

        html_file = f"{output_dir}/deepeval_report_{results['pipeline']}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

# Utilisation dans RAG Chunk Lab
def run_deepeval_evaluation(doc_id: str,
                           per_pipeline_answers: Dict[str, List[str]],
                           per_pipeline_contexts: Dict[str, List[List[str]]],
                           questions: List[str],
                           ground_truth: List[str],
                           include_safety_metrics: bool = True) -> Dict:
    """
    Lance l'évaluation DeepEval pour tous les pipelines
    """
    evaluator = DeepEvalRAGEvaluator(include_safety_metrics=include_safety_metrics)
    results = {}

    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        pipeline_results = evaluator.evaluate_pipeline(
            doc_id, pipeline, questions, answers, ground_truth, contexts
        )

        results[pipeline] = pipeline_results

        # Générer rapport
        evaluator.generate_deepeval_report(pipeline_results)

        # Afficher résultats
        metrics_scores = pipeline_results["metrics_scores"]
        print(f"  📊 Overall Success Rate: {pipeline_results['overall_success_rate']:.2%}")
        print(f"  🎯 Answer Relevancy: {metrics_scores.get('AnswerRelevancyMetric', {}).get('mean_score', 0):.3f}")
        print(f"  💯 Faithfulness: {metrics_scores.get('FaithfulnessMetric', {}).get('mean_score', 0):.3f}")

    # Générer comparaison
    if len(results) > 1:
        comparison = evaluator.compare_pipelines(results)
        comparison_file = "deepeval_results/pipeline_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"📊 Pipeline comparison saved: {comparison_file}")

    return results
'''

    return integration_code

def deepeval_advanced_features():
    """
    Fonctionnalités avancées de DeepEval
    """
    advanced_code = '''
# FONCTIONNALITÉS AVANCÉES DEEPEVAL

# 1. Métriques Personnalisées
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

class DomainSpecificMetric(BaseMetric):
    def __init__(self, threshold: float = 0.7, domain_keywords: List[str] = None):
        self.threshold = threshold
        self.domain_keywords = domain_keywords or []

    def measure(self, test_case: LLMTestCase) -> float:
        """Implémente votre logique de métrique"""
        question = test_case.input
        answer = test_case.actual_output

        # Logique personnalisée pour votre domaine
        score = 0.0
        question_lower = question.lower()
        answer_lower = answer.lower()

        for keyword in self.domain_keywords:
            if keyword in question_lower and keyword in answer_lower:
                score += 1.0 / len(self.domain_keywords)

        return min(1.0, score)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self):
        return "Domain Specific Metric"

# Utilisation de la métrique personnalisée
domain_metric = DomainSpecificMetric(
    threshold=0.6,
    domain_keywords=["technical", "specification", "requirement"]
)

# 2. Tests Automatisés avec Pytest
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric

class TestRAGPipelines:
    @pytest.fixture
    def sample_test_case(self):
        return LLMTestCase(
            input="What is the main requirement?",
            actual_output="The main requirement is to implement authentication.",
            expected_output="Authentication implementation is required.",
            retrieval_context=["Authentication must be implemented for security."]
        )

    def test_answer_relevancy(self, sample_test_case):
        """Test de pertinence des réponses"""
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(sample_test_case, [answer_relevancy_metric])

    @pytest.mark.parametrize("pipeline", ["semantic", "azure_semantic", "fixed"])
    def test_all_pipelines(self, pipeline):
        """Test paramétrisé pour tous les pipelines"""
        # Charger les données de test pour ce pipeline
        test_cases = load_test_cases_for_pipeline(pipeline)

        metrics = [
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.8)
        ]

        for test_case in test_cases:
            assert_test(test_case, metrics)

def load_test_cases_for_pipeline(pipeline: str) -> List[LLMTestCase]:
    """Charge les cas de test pour un pipeline spécifique"""
    # Implémentation pour charger vos données de test
    pass

# 3. Intégration CI/CD
def create_ci_cd_script():
    """Script pour intégration CI/CD"""
    ci_script = '''
#!/bin/bash
# ci_deepeval.sh

echo "🔍 Running DeepEval tests..."

# Lancer les tests avec pytest
pytest tests/test_rag_pipelines.py -v --junitxml=deepeval_results.xml

# Vérifier les résultats
if [ $? -eq 0 ]; then
    echo "✅ All DeepEval tests passed"
    exit 0
else
    echo "❌ DeepEval tests failed"
    exit 1
fi
'''

    with open("ci_deepeval.sh", "w") as f:
        f.write(ci_script)

    print("📁 CI/CD script created: ci_deepeval.sh")

# 4. Monitoring Continu
class ContinuousMonitoring:
    def __init__(self, quality_thresholds: Dict[str, float]):
        self.thresholds = quality_thresholds
        self.alert_history = []

    def monitor_production_quality(self, latest_results: Dict):
        """Surveille la qualité en production"""
        alerts = []

        for metric, threshold in self.thresholds.items():
            if metric in latest_results["metrics_scores"]:
                current_score = latest_results["metrics_scores"][metric]["mean_score"]

                if current_score < threshold:
                    alert = {
                        "metric": metric,
                        "current_score": current_score,
                        "threshold": threshold,
                        "severity": "high" if current_score < threshold * 0.8 else "medium",
                        "timestamp": time.time()
                    }
                    alerts.append(alert)

        if alerts:
            self.send_alerts(alerts)

        return alerts

    def send_alerts(self, alerts: List[Dict]):
        """Envoie des alertes"""
        for alert in alerts:
            message = f"🚨 Quality Alert: {alert['metric']} = {alert['current_score']:.3f} < {alert['threshold']}"
            print(message)

            # Ici vous pourriez intégrer avec:
            # - Slack webhook
            # - Email SMTP
            # - PagerDuty
            # - Discord webhook
            # etc.

# 5. Exportation et Analyse Avancée
def export_for_analysis(results: Dict, output_dir: str = "deepeval_exports"):
    """Exporte les résultats pour analyse externe"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export CSV pour Excel/Pandas
    import pandas as pd

    # Préparer les données pour DataFrame
    rows = []
    for result in results["detailed_results"]:
        row = {
            "input": result["input"],
            "actual_output": result["actual_output"],
            "expected_output": result["expected_output"],
            "success": result["success"]
        }

        # Ajouter les scores des métriques
        for metric_name, metric_data in result["metrics"].items():
            row[f"{metric_name}_score"] = metric_data["score"]
            row[f"{metric_name}_success"] = metric_data["success"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sauvegarder
    csv_file = f"{output_dir}/deepeval_detailed_results.csv"
    df.to_csv(csv_file, index=False)

    # Statistiques agrégées
    stats_df = df.describe()
    stats_file = f"{output_dir}/deepeval_statistics.csv"
    stats_df.to_csv(stats_file)

    print(f"📊 Detailed results exported: {csv_file}")
    print(f"📈 Statistics exported: {stats_file}")

    return csv_file, stats_file

# 6. A/B Testing Automatisé
class ABTestManager:
    def __init__(self):
        self.test_results = {}

    def run_ab_test(self, pipeline_a: str, pipeline_b: str,
                   test_cases: List[LLMTestCase], metrics: List) -> Dict:
        """Lance un test A/B entre deux pipelines"""

        print(f"🧪 Running A/B test: {pipeline_a} vs {pipeline_b}")

        # Évaluer les deux pipelines
        results_a = evaluate(test_cases, metrics)
        results_b = evaluate(test_cases, metrics)

        # Analyser statistiquement
        statistical_analysis = self._statistical_comparison(results_a, results_b)

        ab_test_result = {
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "results_a": self._summarize_results(results_a),
            "results_b": self._summarize_results(results_b),
            "statistical_analysis": statistical_analysis,
            "recommendation": self._generate_recommendation(statistical_analysis)
        }

        return ab_test_result

    def _statistical_comparison(self, results_a, results_b) -> Dict:
        """Analyse statistique des résultats A/B"""
        from scipy import stats

        # Extraire les scores pour comparaison
        scores_a = [r.score for r in results_a if hasattr(r, 'score')]
        scores_b = [r.score for r in results_b if hasattr(r, 'score')]

        # Test t de Student
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "mean_a": sum(scores_a) / len(scores_a) if scores_a else 0,
            "mean_b": sum(scores_b) / len(scores_b) if scores_b else 0
        }

    def _generate_recommendation(self, analysis: Dict) -> str:
        """Génère une recommandation basée sur l'analyse"""
        if not analysis["significant"]:
            return "No statistically significant difference between pipelines"

        if analysis["mean_b"] > analysis["mean_a"]:
            return f"Pipeline B significantly outperforms Pipeline A (p={analysis['p_value']:.4f})"
        else:
            return f"Pipeline A significantly outperforms Pipeline B (p={analysis['p_value']:.4f})"
'''

    return advanced_code

def deepeval_best_practices():
    """
    Meilleures pratiques DeepEval
    """
    practices = '''
# 🏆 MEILLEURES PRATIQUES DEEPEVAL

## 1. Organisation des Tests
```python
# Structure recommandée des fichiers
tests/
├── conftest.py                 # Configuration pytest
├── test_basic_rag.py          # Tests de base
├── test_advanced_metrics.py   # Métriques avancées
├── test_safety_checks.py      # Tests de sécurité
└── datasets/
    ├── basic_qa.json          # Dataset de base
    ├── domain_specific.json   # Dataset spécialisé
    └── edge_cases.json        # Cas limites
```

## 2. Configuration des Seuils
- **Answer Relevancy**: 0.7-0.8 (usage général)
- **Faithfulness**: 0.8-0.9 (critique pour éviter hallucinations)
- **Contextual Recall**: 0.6-0.7 (récupération d'information)
- **Bias/Toxicity**: 0.1-0.3 (plus bas = mieux)

## 3. Tests en Couches
```python
# Layer 1: Tests unitaires rapides
@pytest.mark.fast
def test_basic_functionality():
    # Tests rapides avec seuils permissifs
    pass

# Layer 2: Tests d'intégration
@pytest.mark.integration
def test_pipeline_integration():
    # Tests complets avec seuils production
    pass

# Layer 3: Tests de régression
@pytest.mark.regression
def test_no_performance_degradation():
    # Comparer avec version précédente
    pass
```

## 4. Gestion des Coûts
- Utilisez des échantillons représentatifs (100-500 questions)
- Configurez des caches pour éviter les re-évaluations
- Utilisez des modèles moins chers pour les tests rapides
- Alternez entre évaluations complètes et partielles

## 5. Intégration Production
```python
# Monitoring périodique
def daily_quality_check():
    # Évaluer sur échantillon récent
    recent_data = get_recent_interactions(limit=100)
    results = evaluate_sample(recent_data)

    # Alerter si dégradation
    if results["overall_score"] < PRODUCTION_THRESHOLD:
        send_alert("Quality degradation detected")

# Tests de déploiement
def pre_deployment_gate():
    # Tests obligatoires avant mise en production
    gate_results = run_gate_tests()
    if not gate_results["passed"]:
        raise DeploymentBlocked("Quality gates failed")
```

## 6. Documentation et Traçabilité
- Documentez vos métriques personnalisées
- Versionnez vos datasets de test
- Gardez un historique des résultats
- Documentez les changements de seuils
'''

    return practices

# Fonction principale pour créer le tutoriel complet
def create_deepeval_complete_tutorial():
    """
    Crée le tutoriel complet DeepEval
    """
    tutorial_content = f"""
# 🚀 DeepEval - Tutoriel Complet pour RAG Chunk Lab

{install_deepeval()}

## 🛠️ Configuration de Base
{setup_deepeval_basic()}

## 🔌 Intégration avec RAG Chunk Lab
{integrate_deepeval_with_rag_chunk_lab()}

## 🎯 Fonctionnalités Avancées
{deepeval_advanced_features()}

## 🏆 Meilleures Pratiques
{deepeval_best_practices()}

## 📚 Ressources Additionnelles

### Documentation Officielle
- Site web: https://confident-ai.com/
- Documentation: https://docs.confident-ai.com/
- GitHub: https://github.com/confident-ai/deepeval
- Exemples: https://github.com/confident-ai/deepeval/tree/main/examples

### API Reference
- Metrics: https://docs.confident-ai.com/docs/metrics-introduction
- Test Cases: https://docs.confident-ai.com/docs/evaluation-test-cases
- Datasets: https://docs.confident-ai.com/docs/evaluation-datasets

### Intégrations
- Pytest: https://docs.confident-ai.com/docs/evaluation-pytest-integration
- LangChain: https://docs.confident-ai.com/docs/integrations-langchain
- OpenAI: https://docs.confident-ai.com/docs/integrations-openai

### Community
- GitHub Issues: https://github.com/confident-ai/deepeval/issues
- Discord: https://discord.gg/confident-ai
- Documentation Feedback: https://docs.confident-ai.com/feedback
"""

    # Créer le dossier tutorials s'il n'existe pas
    os.makedirs("tutorials", exist_ok=True)

    # Sauvegarder le tutoriel
    with open("tutorials/deepeval_complete_tutorial.md", "w", encoding="utf-8") as f:
        f.write(tutorial_content)

    return "tutorials/deepeval_complete_tutorial.md"

if __name__ == "__main__":
    tutorial_file = create_deepeval_complete_tutorial()
    print(f"📚 Tutoriel DeepEval créé: {tutorial_file}")