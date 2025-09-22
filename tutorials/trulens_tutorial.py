"""
TruLens Tutorial - Observabilité RAG en Temps Réel
Documentation officielle: https://trulens.org/
GitHub: https://github.com/truera/trulens
"""

import os
from typing import List, Dict
import json

def install_trulens():
    """
    Installation de TruLens
    """
    print("""
🔧 INSTALLATION TRULENS

# Installation basique
pip install trulens-eval

# Installation complète avec toutes les intégrations
pip install trulens-eval[providers,local]

# Pour LangChain
pip install langchain

# Pour OpenAI
pip install openai

# Vérification
python -c "import trulens_eval; print('TruLens installé avec succès!')"
""")

def setup_trulens_basic():
    """
    Configuration de base TruLens
    """
    setup_code = '''
# 1. Import des modules TruLens
from trulens_eval import TruChain, Feedback, Select, Tru
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as TruOpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 2. Configuration du provider de feedback
openai_provider = TruOpenAI(api_key="your-openai-key")

# 3. Définition des métriques de feedback
groundedness = Groundedness(groundedness_provider=openai_provider)

# Feedback de pertinence question-réponse
f_qa_relevance = Feedback(
    openai_provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input_output()

# Feedback de pertinence contexte-réponse
f_context_relevance = Feedback(
    openai_provider.qs_relevance_with_cot_reasons,
    name="Context Relevance"
).on_input().on(Select.RecordCalls.retrieve.rets.collect())

# Feedback de groundedness (hallucination)
f_groundedness = Feedback(
    groundedness.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on(Select.RecordCalls.retrieve.rets.collect()).on_output()

# 4. Liste des feedbacks à utiliser
feedbacks = [f_qa_relevance, f_context_relevance, f_groundedness]
'''

    return setup_code

def integrate_trulens_with_rag_chunk_lab():
    """
    Intégration TruLens avec RAG Chunk Lab
    """
    integration_code = '''
# Integration avec RAG Chunk Lab
from trulens_eval import TruChain, Feedback, Tru
import json

class TruLensRAGEvaluator:
    def __init__(self, openai_api_key: str):
        self.tru = Tru()
        self.openai_provider = TruOpenAI(api_key=openai_api_key)
        self.feedbacks = self._setup_feedbacks()

    def _setup_feedbacks(self):
        """Configure les métriques TruLens"""
        return [
            # Pertinence de la réponse
            Feedback(
                self.openai_provider.relevance_with_cot_reasons,
                name="Answer Relevance"
            ).on_input_output(),

            # Groundedness (anti-hallucination)
            Feedback(
                self.openai_provider.groundedness_measure_with_cot_reasons,
                name="Groundedness"
            ).on_input().on_output(),

            # Cohérence contextuelle
            Feedback(
                self.openai_provider.context_relevance_with_cot_reasons,
                name="Context Relevance"
            ).on_input()
        ]

    def evaluate_pipeline(self, doc_id: str, pipeline_name: str,
                         questions: List[str], answers: List[str],
                         contexts: List[List[str]]) -> Dict:
        """Évalue un pipeline avec TruLens"""

        # Simuler une chain LangChain pour TruLens
        class MockRAGChain:
            def __init__(self, answers, contexts):
                self.answers = answers
                self.contexts = contexts
                self.current_idx = 0

            def __call__(self, inputs):
                question = inputs.get("query", "")
                if self.current_idx < len(self.answers):
                    answer = self.answers[self.current_idx]
                    context = self.contexts[self.current_idx] if self.current_idx < len(self.contexts) else []
                    self.current_idx += 1
                    return {"result": answer, "source_documents": context}
                return {"result": "", "source_documents": []}

        # Créer la chain mockée
        mock_chain = MockRAGChain(answers, contexts)

        # Wrapper TruLens
        tru_chain = TruChain(
            mock_chain,
            app_id=f"{doc_id}_{pipeline_name}",
            feedbacks=self.feedbacks
        )

        # Évaluer chaque question
        records = []
        for question in questions:
            with tru_chain as recording:
                result = mock_chain({"query": question})
                records.append(recording.get())

        # Récupérer les résultats
        results = {
            "pipeline": pipeline_name,
            "total_questions": len(questions),
            "feedback_scores": {},
            "records": records
        }

        # Calculer les scores moyens
        for feedback in self.feedbacks:
            feedback_name = feedback.name
            scores = [record.feedback_results.get(feedback_name, 0) for record in records]
            results["feedback_scores"][feedback_name] = {
                "mean": sum(scores) / len(scores) if scores else 0,
                "scores": scores
            }

        return results

    def generate_trulens_report(self, results: Dict, output_dir: str = "trulens_results"):
        """Génère un rapport TruLens"""
        os.makedirs(output_dir, exist_ok=True)

        report_file = f"{output_dir}/trulens_evaluation_{results['pipeline']}.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📊 TruLens report saved: {report_file}")

        # Lancer le dashboard TruLens
        print("🌐 Starting TruLens dashboard...")
        print("Visit: http://localhost:8501")
        # self.tru.run_dashboard()  # Décommentez pour lancer le dashboard

        return report_file

# Utilisation dans RAG Chunk Lab
def run_trulens_evaluation(doc_id: str,
                          per_pipeline_answers: Dict[str, List[str]],
                          per_pipeline_contexts: Dict[str, List[List[str]]],
                          questions: List[str],
                          openai_api_key: str) -> Dict:
    """
    Lance l'évaluation TruLens pour tous les pipelines
    """
    evaluator = TruLensRAGEvaluator(openai_api_key)
    results = {}

    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        print(f"🔍 TruLens evaluation for {pipeline}...")

        pipeline_results = evaluator.evaluate_pipeline(
            doc_id, pipeline, questions, answers, contexts
        )

        results[pipeline] = pipeline_results

        # Générer rapport
        evaluator.generate_trulens_report(pipeline_results)

        # Afficher résultats
        feedback_scores = pipeline_results["feedback_scores"]
        print(f"  📊 Answer Relevance: {feedback_scores.get('Answer Relevance', {}).get('mean', 0):.3f}")
        print(f"  🎯 Groundedness: {feedback_scores.get('Groundedness', {}).get('mean', 0):.3f}")
        print(f"  📝 Context Relevance: {feedback_scores.get('Context Relevance', {}).get('mean', 0):.3f}")

    return results
'''

    return integration_code

def trulens_advanced_features():
    """
    Fonctionnalités avancées de TruLens
    """
    advanced_code = '''
# FONCTIONNALITÉS AVANCÉES TRULENS

# 1. Custom Feedback Functions
from trulens_eval.feedback import Feedback

def custom_domain_relevance(question: str, answer: str) -> float:
    """Feedback personnalisé pour votre domaine"""
    # Logique spécifique à votre domaine
    domain_keywords = ["votre", "domaine", "spécifique"]

    score = 0.0
    for keyword in domain_keywords:
        if keyword in question.lower() and keyword in answer.lower():
            score += 0.2

    return min(1.0, score)

# Créer le feedback personnalisé
custom_feedback = Feedback(
    custom_domain_relevance,
    name="Domain Relevance"
).on_input_output()

# 2. Monitoring en Temps Réel
class RealTimeMonitor:
    def __init__(self):
        self.tru = Tru()

    def setup_alerts(self):
        """Configure des alertes automatiques"""
        # Définir des seuils
        thresholds = {
            "groundedness": 0.7,
            "relevance": 0.8,
            "custom_metric": 0.6
        }

        # Surveillance continue
        for app_id in self.tru.get_app_ids():
            records = self.tru.get_records_and_feedback(app_ids=[app_id])

            for record in records:
                for metric, threshold in thresholds.items():
                    score = record.feedback_results.get(metric, 0)
                    if score < threshold:
                        self.send_alert(app_id, metric, score, threshold)

    def send_alert(self, app_id: str, metric: str, score: float, threshold: float):
        """Envoie une alerte"""
        message = f"🚨 Alert: {app_id} - {metric}: {score:.3f} < {threshold}"
        print(message)
        # Ici vous pourriez envoyer un email, Slack, etc.

# 3. Comparaison A/B Testing
def compare_pipelines_trulens(results_a: Dict, results_b: Dict, pipeline_a: str, pipeline_b: str):
    """Compare deux pipelines avec TruLens"""

    comparison = {
        "pipeline_a": pipeline_a,
        "pipeline_b": pipeline_b,
        "winner": {},
        "detailed_comparison": {}
    }

    # Comparer chaque métrique
    for metric in ["Answer Relevance", "Groundedness", "Context Relevance"]:
        score_a = results_a.get("feedback_scores", {}).get(metric, {}).get("mean", 0)
        score_b = results_b.get("feedback_scores", {}).get(metric, {}).get("mean", 0)

        comparison["detailed_comparison"][metric] = {
            pipeline_a: score_a,
            pipeline_b: score_b,
            "difference": score_b - score_a,
            "winner": pipeline_a if score_a > score_b else pipeline_b
        }

        comparison["winner"][metric] = pipeline_a if score_a > score_b else pipeline_b

    # Gagnant global
    scores_a = [comp[pipeline_a] for comp in comparison["detailed_comparison"].values()]
    scores_b = [comp[pipeline_b] for comp in comparison["detailed_comparison"].values()]

    avg_a = sum(scores_a) / len(scores_a)
    avg_b = sum(scores_b) / len(scores_b)

    comparison["overall_winner"] = pipeline_a if avg_a > avg_b else pipeline_b
    comparison["overall_scores"] = {pipeline_a: avg_a, pipeline_b: avg_b}

    return comparison

# 4. Export pour Analyse Externe
def export_trulens_data(app_id: str, output_file: str):
    """Exporte les données TruLens pour analyse externe"""
    tru = Tru()

    # Récupérer toutes les données
    records = tru.get_records_and_feedback(app_ids=[app_id])

    # Préparer pour export
    export_data = []
    for record in records:
        export_data.append({
            "timestamp": record.timestamp,
            "input": record.input,
            "output": record.output,
            "feedback_scores": record.feedback_results,
            "latency": record.latency,
            "cost": record.cost
        })

    # Sauvegarder
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"📁 TruLens data exported to: {output_file}")
'''

    return advanced_code

def trulens_dashboard_guide():
    """
    Guide d'utilisation du dashboard TruLens
    """
    guide = '''
# 🌐 GUIDE DASHBOARD TRULENS

## Lancement du Dashboard
```python
from trulens_eval import Tru

tru = Tru()
tru.run_dashboard(port=8501, host="localhost")
```

## Navigation du Dashboard

### 1. 📊 Overview Tab
- Vue d'ensemble de toutes vos applications
- Scores moyens par application
- Tendances temporelles
- Comparaison rapide des performances

### 2. 🔍 Applications Tab
- Détails par application/pipeline
- Historique des évaluations
- Breakdown par métrique
- Records individuels

### 3. 📈 Records Tab
- Vue détaillée de chaque interaction
- Input/Output avec scores
- Chaîne de raisonnement (Chain of Thought)
- Temps de réponse et coûts

### 4. ⚙️ Feedback Tab
- Configuration des métriques
- Fonctions de feedback personnalisées
- Providers de feedback (OpenAI, Hugging Face, etc.)

## Fonctionnalités Clés

### Filtrage et Recherche
- Filtrer par date, score, application
- Recherche dans les inputs/outputs
- Export des données filtrées

### Alertes et Monitoring
- Seuils configurables
- Notifications en temps réel
- Graphiques de tendances

### Comparaison A/B
- Comparer plusieurs versions
- Tests statistiques
- Visualisations interactives

## Intégration avec votre Workflow

```python
# 1. Dans votre script d'évaluation
if __name__ == "__main__":
    # Lancer vos évaluations
    run_evaluations()

    # Démarrer le dashboard pour analyser
    tru = Tru()
    tru.run_dashboard()
    print("Dashboard accessible sur: http://localhost:8501")

# 2. En mode production
# Utilisez tru.start_dashboard_session() pour contrôle programmatique
```

## Tips d'Utilisation

### 📊 Analyse des Performances
1. Utilisez l'onglet "Overview" pour identifier les problèmes
2. Drill-down dans "Records" pour les détails
3. Comparez différentes configurations dans "Applications"

### 🔍 Debug des Problèmes
1. Filtrez les scores faibles
2. Examinez les chaînes de raisonnement
3. Analysez les patterns dans les échecs

### 📈 Optimisation Continue
1. Suivez les tendances temporelles
2. Testez différentes configurations
3. Mesurez l'impact des changements
'''

    return guide

def trulens_best_practices():
    """
    Meilleures pratiques TruLens
    """
    practices = '''
# 🏆 MEILLEURES PRATIQUES TRULENS

## 1. Configuration Initiale
```python
# Toujours configurer des app_id explicites
tru_chain = TruChain(
    chain,
    app_id="rag_chunk_lab_semantic_v1",  # Versioning important
    feedbacks=feedbacks,
    metadata={"version": "1.0", "domain": "legal"}  # Métadonnées utiles
)
```

## 2. Choix des Métriques
- **Answer Relevance**: Essentiel pour tout RAG
- **Groundedness**: Critique pour éviter hallucinations
- **Context Relevance**: Important pour optimiser la récupération
- **Custom Metrics**: Ajoutez des métriques spécifiques à votre domaine

## 3. Monitoring Production
```python
# Configuration pour production
class ProductionMonitor:
    def __init__(self):
        self.tru = Tru()
        self.alert_thresholds = {
            "groundedness": 0.7,
            "answer_relevance": 0.8
        }

    def monitor_continuously(self):
        # Vérification périodique
        for app_id in self.tru.get_app_ids():
            recent_records = self.tru.get_records_and_feedback(
                app_ids=[app_id],
                limit=100  # Derniers 100 records
            )

            # Analyser les tendances
            self.analyze_trends(recent_records)
```

## 4. Optimisation des Coûts
- Utilisez des échantillons représentatifs pour l'évaluation
- Configurez des caches pour éviter les re-évaluations
- Alternez entre évaluations complètes et partielles

## 5. Intégration CI/CD
```python
# Dans votre pipeline CI/CD
def evaluate_before_deployment():
    results = run_trulens_evaluation()

    # Vérifier les seuils de qualité
    for metric, threshold in QUALITY_GATES.items():
        if results[metric] < threshold:
            raise Exception(f"Quality gate failed: {metric} = {results[metric]}")

    print("✅ Quality gates passed - deployment approved")
```

## 6. Documentation et Traçabilité
- Documentez vos métriques personnalisées
- Gardez un historique des configurations
- Versionnez vos applications TruLens
- Exportez régulièrement pour backup
'''

    return practices

# Fonction principale pour créer le tutoriel complet
def create_trulens_complete_tutorial():
    """
    Crée le tutoriel complet TruLens
    """
    tutorial_content = f"""
# 🔍 TruLens - Tutoriel Complet pour RAG Chunk Lab

{install_trulens()}

## 🚀 Configuration de Base
{setup_trulens_basic()}

## 🔌 Intégration avec RAG Chunk Lab
{integrate_trulens_with_rag_chunk_lab()}

## 🎯 Fonctionnalités Avancées
{trulens_advanced_features()}

## 🌐 Guide Dashboard
{trulens_dashboard_guide()}

## 🏆 Meilleures Pratiques
{trulens_best_practices()}

## 📚 Ressources Additionnelles

### Documentation Officielle
- Site web: https://trulens.org/
- Documentation: https://trulens.org/trulens_eval/
- GitHub: https://github.com/truera/trulens
- Exemples: https://github.com/truera/trulens/tree/main/trulens_eval/examples

### Tutoriels Vidéo
- Getting Started: https://www.youtube.com/watch?v=xyz
- Advanced Features: https://www.youtube.com/watch?v=abc

### Community
- Discord: https://discord.gg/trulens
- GitHub Discussions: https://github.com/truera/trulens/discussions
"""

    # Sauvegarder le tutoriel
    with open("tutorials/trulens_complete_tutorial.md", "w", encoding="utf-8") as f:
        f.write(tutorial_content)

    return "tutorials/trulens_complete_tutorial.md"

if __name__ == "__main__":
    tutorial_file = create_trulens_complete_tutorial()
    print(f"📚 Tutoriel TruLens créé: {tutorial_file}")