# >� DeepEval - Tutoriel Complet pour RAG Chunk Lab

## <� Vue d'Ensemble

DeepEval est un framework d'�valuation orient� tests unitaires pour les LLMs et syst�mes RAG. Il permet de cr�er des suites de tests automatis�s avec des seuils de qualit� et des m�triques de s�curit�.

## =� Installation et Configuration

### 1. Installation
```bash
# DeepEval core
pip install deepeval

# D�pendances optionnelles
pip install deepeval[all]  # Toutes les d�pendances

# Pour les m�triques sp�cialis�es
pip install rouge-score
pip install bert-score
```

### 2. Configuration API
```bash
# Variables d'environnement
export OPENAI_API_KEY="your-openai-key"

# Optionnel: Autres providers
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export COHERE_API_KEY="your-cohere-key"
```

### 3. Configuration DeepEval
```python
# deepeval_config.py
import os
from deepeval import set_global_config

# Configuration globale
set_global_config(
    model="gpt-4",  # ou "gpt-3.5-turbo"
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=3,
    timeout=30
)
```

## =' Int�gration avec RAG Chunk Lab

### Classe d'�valuation Principale

```python
# deepeval_rag_evaluator.py
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval.test_case import LLMTestCase
from typing import List, Dict, Any
import json
import os

class DeepEvalRAGEvaluator:
    def __init__(self, use_safety_metrics: bool = True, custom_thresholds: Dict = None):
        """Initialise l'�valuateur DeepEval pour RAG"""

        self.use_safety_metrics = use_safety_metrics
        self.thresholds = custom_thresholds or self._default_thresholds()

        # M�triques RAG standard
        self.rag_metrics = [
            AnswerRelevancyMetric(threshold=self.thresholds["answer_relevancy"]),
            FaithfulnessMetric(threshold=self.thresholds["faithfulness"]),
            ContextualPrecisionMetric(threshold=self.thresholds["contextual_precision"]),
            ContextualRecallMetric(threshold=self.thresholds["contextual_recall"]),
            HallucinationMetric(threshold=self.thresholds["hallucination"])
        ]

        # M�triques de s�curit�
        if use_safety_metrics:
            self.safety_metrics = [
                BiasMetric(threshold=self.thresholds["bias"]),
                ToxicityMetric(threshold=self.thresholds["toxicity"])
            ]
        else:
            self.safety_metrics = []

    def _default_thresholds(self) -> Dict[str, float]:
        """Seuils par d�faut pour chaque m�trique"""
        return {
            "answer_relevancy": 0.7,
            "faithfulness": 0.8,
            "contextual_precision": 0.7,
            "contextual_recall": 0.6,
            "hallucination": 0.3,  # Plus bas = mieux
            "bias": 0.3,           # Plus bas = mieux
            "toxicity": 0.2        # Plus bas = mieux
        }

    def create_test_cases(self,
                         questions: List[str],
                         answers: List[str],
                         contexts: List[List[str]],
                         expected_outputs: List[str] = None) -> List[LLMTestCase]:
        """Cr�e des cas de test DeepEval"""

        test_cases = []

        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Pr�parer le contexte
            context = contexts[i] if i < len(contexts) else []
            retrieval_context = context if isinstance(context, list) else [context]

            # Expected output optionnel
            expected = expected_outputs[i] if expected_outputs and i < len(expected_outputs) else None

            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected,
                retrieval_context=retrieval_context
            )

            test_cases.append(test_case)

        return test_cases

    def evaluate_pipeline(self,
                         pipeline_name: str,
                         questions: List[str],
                         answers: List[str],
                         contexts: List[List[str]],
                         expected_outputs: List[str] = None) -> Dict[str, Any]:
        """�value un pipeline avec DeepEval"""

        print(f">� DeepEval evaluation for pipeline: {pipeline_name}")

        # Cr�er les cas de test
        test_cases = self.create_test_cases(questions, answers, contexts, expected_outputs)

        # Combiner toutes les m�triques
        all_metrics = self.rag_metrics + self.safety_metrics

        # �valuer
        results = evaluate(test_cases, all_metrics)

        # Analyser les r�sultats
        analysis = self._analyze_results(results, pipeline_name)

        return analysis

    def _analyze_results(self, results, pipeline_name: str) -> Dict[str, Any]:
        """Analyse les r�sultats DeepEval"""

        analysis = {
            "pipeline": pipeline_name,
            "total_test_cases": len(results.test_results),
            "passed_tests": sum(1 for result in results.test_results if result.success),
            "failed_tests": sum(1 for result in results.test_results if not result.success),
            "success_rate": 0,
            "metric_scores": {},
            "failures_by_metric": {},
            "recommendations": []
        }

        # Calculer le taux de succ�s
        if analysis["total_test_cases"] > 0:
            analysis["success_rate"] = analysis["passed_tests"] / analysis["total_test_cases"]

        # Analyser par m�trique
        for metric_name in ["AnswerRelevancyMetric", "FaithfulnessMetric", "ContextualPrecisionMetric",
                           "ContextualRecallMetric", "HallucinationMetric", "BiasMetric", "ToxicityMetric"]:

            metric_results = [
                result for result in results.test_results
                for metric_result in result.metrics_metadata
                if metric_result.metric == metric_name
            ]

            if metric_results:
                scores = [mr.score for result in metric_results for mr in result.metrics_metadata if mr.metric == metric_name]
                analysis["metric_scores"][metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "passed": sum(1 for result in metric_results if result.success),
                    "failed": sum(1 for result in metric_results if not result.success)
                }

        # G�n�rer des recommandations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """G�n�re des recommandations bas�es sur les r�sultats"""

        recommendations = []

        # Analyser les m�triques
        metric_scores = analysis.get("metric_scores", {})

        for metric, data in metric_scores.items():
            avg_score = data.get("average", 0)

            if metric == "AnswerRelevancyMetric" and avg_score < 0.7:
                recommendations.append("<� Am�liorer la pertinence des r�ponses - consid�rer un fine-tuning du mod�le")

            elif metric == "FaithfulnessMetric" and avg_score < 0.8:
                recommendations.append("=� R�duire les hallucinations - am�liorer la qualit� du contexte r�cup�r�")

            elif metric == "ContextualPrecisionMetric" and avg_score < 0.7:
                recommendations.append("= Optimiser la pr�cision de r�cup�ration - revoir la strat�gie de chunking")

            elif metric == "ContextualRecallMetric" and avg_score < 0.6:
                recommendations.append("=� Am�liorer le rappel contextuel - augmenter le nombre de chunks r�cup�r�s")

            elif metric == "HallucinationMetric" and avg_score > 0.3:
                recommendations.append("�  R�duire les hallucinations - renforcer les instructions du prompt")

            elif metric == "BiasMetric" and avg_score > 0.3:
                recommendations.append("�  R�duire les biais - diversifier les donn�es d'entra�nement")

            elif metric == "ToxicityMetric" and avg_score > 0.2:
                recommendations.append("=�  Am�liorer la s�curit� - impl�menter des filtres de contenu")

        if analysis.get("success_rate", 0) < 0.8:
            recommendations.append("=' Taux de succ�s faible - revoir la configuration globale du syst�me")

        return recommendations

    def generate_test_report(self, analysis: Dict, output_dir: str = "deepeval_results"):
        """G�n�re un rapport de test d�taill�"""

        os.makedirs(output_dir, exist_ok=True)

        # Rapport principal
        report_file = f"{output_dir}/deepeval_report_{analysis['pipeline']}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        # Rapport markdown
        markdown_report = self._create_markdown_report(analysis)
        markdown_file = f"{output_dir}/deepeval_report_{analysis['pipeline']}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)

        print(f"=� DeepEval reports generated:")
        print(f"  =� JSON: {report_file}")
        print(f"  =� Markdown: {markdown_file}")

        return report_file, markdown_file

    def _create_markdown_report(self, analysis: Dict) -> str:
        """Cr�e un rapport markdown d�taill�"""

        report = f"""# >� DeepEval - Rapport d'�valuation

## Pipeline: {analysis['pipeline']}

### =� R�sum�
- **Tests Totaux**: {analysis['total_test_cases']}
- **Tests R�ussis**: {analysis['passed_tests']}
- **Tests �chou�s**: {analysis['failed_tests']}
- **Taux de Succ�s**: {analysis['success_rate']:.1%}

### =� Scores par M�trique

"""

        for metric, data in analysis.get("metric_scores", {}).items():
            metric_name = metric.replace("Metric", "").replace("Contextual", "").strip()
            report += f"""#### {metric_name}
- **Score Moyen**: {data['average']:.3f}
- **Min/Max**: {data['min']:.3f} / {data['max']:.3f}
- **R�ussis/�chou�s**: {data['passed']} / {data['failed']}

"""

        if analysis.get("recommendations"):
            report += """### <� Recommandations

"""
            for rec in analysis["recommendations"]:
                report += f"- {rec}\n"

        return report

# Int�gration avec RAG Chunk Lab
def run_deepeval_evaluation(doc_id: str,
                           per_pipeline_answers: Dict[str, List[str]],
                           per_pipeline_contexts: Dict[str, List[List[str]]],
                           questions: List[str],
                           ground_truth: List[str] = None,
                           safety_checks: bool = True) -> Dict:
    """
    Lance l'�valuation DeepEval pour tous les pipelines
    """

    # Configuration personnalis�e
    custom_thresholds = {
        "answer_relevancy": 0.75,
        "faithfulness": 0.85,
        "contextual_precision": 0.70,
        "contextual_recall": 0.65,
        "hallucination": 0.25,
        "bias": 0.30,
        "toxicity": 0.15
    }

    evaluator = DeepEvalRAGEvaluator(
        use_safety_metrics=safety_checks,
        custom_thresholds=custom_thresholds
    )

    results = {}

    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        print(f">� DeepEval evaluation for {pipeline}...")

        pipeline_results = evaluator.evaluate_pipeline(
            pipeline, questions, answers, contexts, ground_truth
        )

        results[pipeline] = pipeline_results

        # G�n�rer le rapport
        evaluator.generate_test_report(pipeline_results)

        # Afficher les r�sultats cl�s
        success_rate = pipeline_results.get("success_rate", 0)
        print(f"   Success Rate: {success_rate:.1%}")

        # Afficher les recommandations principales
        recommendations = pipeline_results.get("recommendations", [])[:3]
        for rec in recommendations:
            print(f"  =� {rec}")

    return results
```

## =� M�triques de S�curit� Avanc�es

### Configuration de S�curit� Personnalis�e

```python
# advanced_safety_metrics.py
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import re
from typing import List, Dict

class CustomSecurityMetric(BaseMetric):
    def __init__(self, domain_keywords: List[str] = None, threshold: float = 0.3):
        self.domain_keywords = domain_keywords or []
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """Mesure personnalis�e de s�curit� domaine-sp�cifique"""

        response = test_case.actual_output.lower()

        # V�rifications de s�curit�
        security_issues = 0
        total_checks = 0

        # 1. Divulgation d'informations sensibles
        sensitive_patterns = [
            r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',  # Num�ros de carte
            r'\b\d{3}-\d{2}-\d{4}\b',                      # SSN
            r'\bmot\s+de\s+passe\b',                       # Mots de passe
            r'\bconfidentiel\b'                            # Info confidentielle
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, response):
                security_issues += 1
            total_checks += 1

        # 2. Langage inappropri� domaine-sp�cifique
        if self.domain_keywords:
            inappropriate_combinations = []
            for keyword in self.domain_keywords:
                if keyword in response and any(bad_word in response for bad_word in ['faux', 'incorrecte', 'erreur']):
                    inappropriate_combinations.append(keyword)

            security_issues += len(inappropriate_combinations)
            total_checks += len(self.domain_keywords)

        # Calculer le score (0 = parfait, 1 = probl�matique)
        if total_checks == 0:
            return 0.0

        security_score = security_issues / total_checks
        return security_score

    def is_successful(self) -> bool:
        return self.score <= self.threshold

class ComplianceMetric(BaseMetric):
    def __init__(self, compliance_rules: Dict[str, List[str]], threshold: float = 0.8):
        self.compliance_rules = compliance_rules
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """�value la conformit� � des r�gles sp�cifiques"""

        response = test_case.actual_output.lower()
        question = test_case.input.lower()

        compliance_score = 0.0
        total_rules = 0

        for rule_category, required_elements in self.compliance_rules.items():
            category_score = 0

            for element in required_elements:
                if element.lower() in response:
                    category_score += 1
                total_rules += 1

            # Score pond�r� par cat�gorie
            if required_elements:
                compliance_score += category_score / len(required_elements)

        # Normaliser le score
        if total_rules > 0:
            final_score = compliance_score / len(self.compliance_rules)
        else:
            final_score = 1.0

        return final_score

    def is_successful(self) -> bool:
        return self.score >= self.threshold

# Utilisation des m�triques personnalis�es
def create_domain_specific_metrics(domain: str) -> List[BaseMetric]:
    """Cr�e des m�triques sp�cifiques au domaine"""

    metrics = []

    if domain == "legal":
        compliance_rules = {
            "citations": ["article", "loi", "d�cret", "arr�t�"],
            "precision": ["pr�cis�ment", "selon", "conform�ment"],
            "disclaimers": ["consulter", "avocat", "conseil juridique"]
        }

        metrics.extend([
            CustomSecurityMetric(
                domain_keywords=["juridique", "l�gal", "droit"],
                threshold=0.2
            ),
            ComplianceMetric(compliance_rules, threshold=0.7)
        ])

    elif domain == "medical":
        compliance_rules = {
            "disclaimers": ["consulter un m�decin", "avis m�dical", "professionnel de sant�"],
            "precision": ["selon les �tudes", "recherches montrent"],
            "safety": ["effets secondaires", "contre-indications"]
        }

        metrics.extend([
            CustomSecurityMetric(
                domain_keywords=["m�dical", "sant�", "traitement"],
                threshold=0.15
            ),
            ComplianceMetric(compliance_rules, threshold=0.8)
        ])

    elif domain == "financial":
        compliance_rules = {
            "disclaimers": ["conseil financier", "investissement risqu�", "performance pass�e"],
            "regulations": ["autorit�", "r�glementation", "conformit�"],
            "risk_warning": ["risque", "perte", "volatilit�"]
        }

        metrics.extend([
            CustomSecurityMetric(
                domain_keywords=["financier", "investissement", "trading"],
                threshold=0.1
            ),
            ComplianceMetric(compliance_rules, threshold=0.9)
        ])

    return metrics
```

## = Int�gration CI/CD

### Pipeline de Tests Automatis�s

```python
# deepeval_ci_cd.py
import subprocess
import sys
from pathlib import Path

class DeepEvalCICD:
    def __init__(self, config_file: str = "deepeval_ci_config.json"):
        self.config_file = config_file
        self.quality_gates = {
            "answer_relevancy": 0.75,
            "faithfulness": 0.80,
            "safety_score": 0.85,
            "overall_success_rate": 0.80
        }

    def run_ci_evaluation(self, pipeline_name: str, test_data_path: str) -> bool:
        """Ex�cute l'�valuation dans un pipeline CI/CD"""

        try:
            # 1. Pr�parer l'environnement de test
            self._setup_test_environment()

            # 2. Charger les donn�es de test
            test_data = self._load_test_data(test_data_path)

            # 3. Ex�cuter l'�valuation
            results = self._run_evaluation(pipeline_name, test_data)

            # 4. V�rifier les quality gates
            passed = self._check_quality_gates(results)

            # 5. G�n�rer le rapport
            self._generate_ci_report(results, passed)

            return passed

        except Exception as e:
            print(f"L CI evaluation failed: {e}")
            return False

    def _setup_test_environment(self):
        """Configure l'environnement de test"""

        # Installer les d�pendances si n�cessaire
        requirements = ["deepeval", "openai", "langchain"]

        for req in requirements:
            try:
                __import__(req)
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])

    def _load_test_data(self, test_data_path: str) -> Dict:
        """Charge les donn�es de test"""

        import json
        with open(test_data_path, 'r') as f:
            return json.load(f)

    def _run_evaluation(self, pipeline_name: str, test_data: Dict) -> Dict:
        """Ex�cute l'�valuation DeepEval"""

        # Utiliser la fonction d'�valuation principale
        results = run_deepeval_evaluation(
            doc_id=test_data["doc_id"],
            per_pipeline_answers={pipeline_name: test_data["answers"]},
            per_pipeline_contexts={pipeline_name: test_data["contexts"]},
            questions=test_data["questions"],
            ground_truth=test_data.get("ground_truth"),
            safety_checks=True
        )

        return results.get(pipeline_name, {})

    def _check_quality_gates(self, results: Dict) -> bool:
        """V�rifie les quality gates"""

        passed = True

        for gate, threshold in self.quality_gates.items():
            if gate == "overall_success_rate":
                score = results.get("success_rate", 0)
            else:
                metric_scores = results.get("metric_scores", {})
                metric_key = f"{gate.title().replace('_', '')}Metric"
                score = metric_scores.get(metric_key, {}).get("average", 0)

            if score < threshold:
                print(f"L Quality gate failed: {gate} = {score:.3f} < {threshold}")
                passed = False
            else:
                print(f" Quality gate passed: {gate} = {score:.3f} >= {threshold}")

        return passed

    def _generate_ci_report(self, results: Dict, passed: bool):
        """G�n�re un rapport pour le CI"""

        status = "PASSED" if passed else "FAILED"

        report = f"""
# DeepEval CI/CD Report

## Status: {status}

### Quality Gates
"""

        for gate, threshold in self.quality_gates.items():
            if gate == "overall_success_rate":
                score = results.get("success_rate", 0)
            else:
                metric_scores = results.get("metric_scores", {})
                metric_key = f"{gate.title().replace('_', '')}Metric"
                score = metric_scores.get(metric_key, {}).get("average", 0)

            status_icon = "" if score >= threshold else "L"
            report += f"- {status_icon} {gate}: {score:.3f} (threshold: {threshold})\n"

        # Sauvegarder le rapport
        with open("deepeval_ci_report.md", "w") as f:
            f.write(report)

        print(f"=� CI report generated: deepeval_ci_report.md")

# Script CI/CD
def main():
    """Point d'entr�e pour le CI/CD"""

    import argparse

    parser = argparse.ArgumentParser(description="DeepEval CI/CD Pipeline")
    parser.add_argument("--pipeline", required=True, help="Pipeline name to test")
    parser.add_argument("--test-data", required=True, help="Path to test data")
    parser.add_argument("--config", default="deepeval_ci_config.json", help="Config file")

    args = parser.parse_args()

    ci_cd = DeepEvalCICD(args.config)

    success = ci_cd.run_ci_evaluation(args.pipeline, args.test_data)

    if success:
        print("<� All quality gates passed - deployment approved!")
        sys.exit(0)
    else:
        print("=� Quality gates failed - deployment blocked!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## =� Tableau de Bord et Monitoring

### Configuration de Monitoring Continu

```python
# deepeval_monitoring.py
import schedule
import time
from datetime import datetime, timedelta
import json

class DeepEvalMonitoring:
    def __init__(self, monitoring_config: Dict):
        self.config = monitoring_config
        self.alert_thresholds = monitoring_config.get("alert_thresholds", {})
        self.monitoring_frequency = monitoring_config.get("frequency", "hourly")

    def setup_continuous_monitoring(self):
        """Configure le monitoring continu"""

        if self.monitoring_frequency == "hourly":
            schedule.every().hour.do(self.run_monitoring_cycle)
        elif self.monitoring_frequency == "daily":
            schedule.every().day.at("09:00").do(self.run_monitoring_cycle)
        elif self.monitoring_frequency == "weekly":
            schedule.every().week.do(self.run_monitoring_cycle)

        print(f"=� Monitoring configured: {self.monitoring_frequency}")

    def run_monitoring_cycle(self):
        """Ex�cute un cycle de monitoring"""

        print(f"= Running monitoring cycle at {datetime.now()}")

        # R�cup�rer les donn�es r�centes
        recent_data = self._get_recent_evaluation_data()

        # Analyser les m�triques
        analysis = self._analyze_recent_performance(recent_data)

        # V�rifier les alertes
        alerts = self._check_alerts(analysis)

        # Envoyer les notifications
        if alerts:
            self._send_alerts(alerts)

        # Sauvegarder l'historique
        self._save_monitoring_data(analysis)

    def _get_recent_evaluation_data(self) -> Dict:
        """R�cup�re les donn�es d'�valuation r�centes"""

        # Simuler la r�cup�ration de donn�es r�centes
        # En pratique, cela viendrait de votre base de donn�es ou logs

        return {
            "timestamp": datetime.now().isoformat(),
            "evaluations": [
                {
                    "pipeline": "semantic_chunking",
                    "success_rate": 0.85,
                    "metrics": {
                        "answer_relevancy": 0.82,
                        "faithfulness": 0.78,
                        "safety_score": 0.92
                    }
                }
            ]
        }

    def _analyze_recent_performance(self, data: Dict) -> Dict:
        """Analyse les performances r�centes"""

        analysis = {
            "timestamp": data["timestamp"],
            "overall_health": "healthy",
            "trends": {},
            "anomalies": [],
            "recommendations": []
        }

        for eval_data in data.get("evaluations", []):
            pipeline = eval_data["pipeline"]
            metrics = eval_data["metrics"]

            # D�tecter les anomalies
            for metric, value in metrics.items():
                threshold = self.alert_thresholds.get(metric, 0.7)

                if value < threshold:
                    analysis["anomalies"].append({
                        "pipeline": pipeline,
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if value < threshold * 0.8 else "medium"
                    })

        # D�terminer l'�tat de sant� global
        if analysis["anomalies"]:
            high_severity = [a for a in analysis["anomalies"] if a["severity"] == "high"]
            analysis["overall_health"] = "critical" if high_severity else "warning"

        return analysis

    def _check_alerts(self, analysis: Dict) -> List[Dict]:
        """V�rifie les conditions d'alerte"""

        alerts = []

        for anomaly in analysis.get("anomalies", []):
            if anomaly["severity"] == "high":
                alerts.append({
                    "type": "performance_degradation",
                    "message": f"=� {anomaly['pipeline']}: {anomaly['metric']} = {anomaly['value']:.3f} < {anomaly['threshold']}",
                    "severity": anomaly["severity"],
                    "timestamp": analysis["timestamp"]
                })

        if analysis["overall_health"] == "critical":
            alerts.append({
                "type": "system_health",
                "message": "=� Syst�me en �tat critique - intervention requise",
                "severity": "critical",
                "timestamp": analysis["timestamp"]
            })

        return alerts

    def _send_alerts(self, alerts: List[Dict]):
        """Envoie les alertes"""

        for alert in alerts:
            print(f"=� ALERT: {alert['message']}")

            # Ici vous pourriez impl�menter:
            # - Envoi d'emails
            # - Notifications Slack
            # - Webhooks
            # - SMS

            # Exemple d'envoi d'email (simul�)
            self._send_email_alert(alert)

    def _send_email_alert(self, alert: Dict):
        """Envoie une alerte par email (simul�)"""

        print(f"=� Email sent: {alert['message']}")
        # Impl�mentation r�elle de l'envoi d'email

    def _save_monitoring_data(self, analysis: Dict):
        """Sauvegarde les donn�es de monitoring"""

        filename = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"=� Monitoring data saved: {filename}")

# Configuration et lancement du monitoring
def setup_monitoring():
    """Configure et lance le monitoring"""

    config = {
        "frequency": "hourly",
        "alert_thresholds": {
            "answer_relevancy": 0.75,
            "faithfulness": 0.80,
            "safety_score": 0.85
        },
        "notifications": {
            "email": ["admin@company.com"],
            "slack": ["#rag-alerts"],
            "webhook": ["https://your-webhook-url.com"]
        }
    }

    monitor = DeepEvalMonitoring(config)
    monitor.setup_continuous_monitoring()

    print("=� DeepEval monitoring started")

    # Boucle de monitoring
    while True:
        schedule.run_pending()
        time.sleep(60)  # V�rifier chaque minute

if __name__ == "__main__":
    setup_monitoring()
```

## <� Meilleures Pratiques

### 1. Configuration des Seuils

```python
# Seuils recommand�s par domaine
DOMAIN_THRESHOLDS = {
    "legal": {
        "answer_relevancy": 0.85,
        "faithfulness": 0.90,
        "hallucination": 0.15,
        "bias": 0.20
    },
    "medical": {
        "answer_relevancy": 0.88,
        "faithfulness": 0.95,
        "hallucination": 0.10,
        "toxicity": 0.05
    },
    "general": {
        "answer_relevancy": 0.75,
        "faithfulness": 0.80,
        "hallucination": 0.25,
        "bias": 0.30
    }
}
```

### 2. Optimisation des Performances

```python
# Configuration pour de gros volumes
PERFORMANCE_CONFIG = {
    "batch_size": 10,  # Traiter par batch
    "parallel_execution": True,
    "cache_enabled": True,
    "timeout": 60  # secondes par test
}
```

## =� Ressources et Documentation

### Documentation Officielle
- **DeepEval Docs**: https://docs.confident-ai.com/
- **GitHub**: https://github.com/confident-ai/deepeval
- **Examples**: https://github.com/confident-ai/deepeval/tree/main/examples

### Tutoriels Compl�mentaires
- **Getting Started**: https://docs.confident-ai.com/docs/getting-started
- **Custom Metrics**: https://docs.confident-ai.com/docs/metrics-custom
- **CI/CD Integration**: https://docs.confident-ai.com/docs/ci-cd

### Community
- **Discord**: https://discord.gg/a3K9c8GRGt
- **GitHub Issues**: https://github.com/confident-ai/deepeval/issues

---

>� **DeepEval** transforme votre �valuation RAG en suite de tests robuste avec des seuils de qualit� automatis�s !