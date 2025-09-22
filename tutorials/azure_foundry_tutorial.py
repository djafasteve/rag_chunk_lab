"""
Azure AI Foundry Tutorial - Plateforme Enterprise pour Évaluation LLM
Documentation officielle: https://docs.microsoft.com/en-us/azure/machine-learning/
GitHub: https://github.com/Azure/azureml-examples
"""

import os
from typing import List, Dict
import json

def install_azure_foundry():
    """
    Installation et configuration Azure AI Foundry
    """
    print("""
🔧 INSTALLATION AZURE AI FOUNDRY

# Installation du SDK Azure ML
pip install azure-ai-ml azure-identity

# Installation des dépendances additionnelles
pip install azure-storage-blob azure-keyvault-secrets

# Pour l'intégration avec MLflow
pip install mlflow azureml-mlflow

# Pour les flows personnalisés
pip install promptflow[azure]

# Vérification
python -c "from azure.ai.ml import MLClient; print('Azure ML SDK installé avec succès!')"

# Installation Azure CLI (optionnel mais recommandé)
# Windows: https://aka.ms/installazurecliwindows
# macOS: brew install azure-cli
# Linux: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
""")

def setup_azure_foundry():
    """
    Configuration initiale Azure AI Foundry
    """
    setup_code = '''
# 1. Configuration des Variables d'Environnement
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

# Variables d'environnement requises
REQUIRED_ENV_VARS = {
    "AZURE_SUBSCRIPTION_ID": "your-subscription-id",
    "AZURE_RESOURCE_GROUP": "your-resource-group",
    "AZURE_ML_WORKSPACE": "your-workspace-name",
    "AZURE_TENANT_ID": "your-tenant-id",  # Optionnel
}

# 2. Authentification
def setup_azure_credentials():
    """Configure l'authentification Azure"""
    try:
        # Authentification automatique (production)
        credential = DefaultAzureCredential()
        print("✅ Using DefaultAzureCredential")
    except Exception:
        # Authentification interactive (développement)
        credential = InteractiveBrowserCredential()
        print("✅ Using InteractiveBrowserCredential")

    return credential

# 3. Client Azure ML
def create_ml_client():
    """Crée le client Azure ML"""
    credential = setup_azure_credentials()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE")
    )

    return ml_client

# 4. Vérification de la configuration
def verify_azure_setup():
    """Vérifie la configuration Azure"""
    try:
        ml_client = create_ml_client()
        workspace = ml_client.workspaces.get()
        print(f"✅ Connected to workspace: {workspace.name}")
        print(f"📍 Location: {workspace.location}")
        print(f"🏷️ Resource Group: {workspace.resource_group}")
        return True
    except Exception as e:
        print(f"❌ Azure setup failed: {e}")
        return False

# 5. Configuration des Compute Resources
def setup_compute_resources(ml_client: MLClient):
    """Configure les ressources de calcul"""
    from azure.ai.ml.entities import ComputeInstance, AmlCompute

    # Compute Instance pour développement interactif
    compute_instance = ComputeInstance(
        name="rag-dev-instance",
        size="Standard_DS3_v2",
        idle_time_before_shutdown_minutes=30
    )

    # Cluster de calcul pour évaluations lourdes
    compute_cluster = AmlCompute(
        name="rag-eval-cluster",
        size="Standard_D4s_v3",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=300
    )

    try:
        ml_client.compute.begin_create_or_update(compute_instance)
        print("✅ Compute instance created/updated")
    except Exception as e:
        print(f"⚠️ Compute instance setup: {e}")

    try:
        ml_client.compute.begin_create_or_update(compute_cluster)
        print("✅ Compute cluster created/updated")
    except Exception as e:
        print(f"⚠️ Compute cluster setup: {e}")
'''

    return setup_code

def create_azure_foundry_evaluation_flows():
    """
    Création de flows d'évaluation personnalisés
    """
    flows_code = '''
# Azure AI Foundry - Flows d'Évaluation Personnalisés
from azure.ai.ml.entities import Data
from azure.ai.ml import Input, Output
import yaml

class AzureFoundryRAGEvaluator:
    def __init__(self, ml_client):
        self.ml_client = ml_client
        self.workspace = ml_client.workspaces.get()

    def create_evaluation_dataset(self, questions: List[str], answers: List[str],
                                ground_truth: List[str], contexts: List[List[str]],
                                dataset_name: str) -> str:
        """Crée un dataset d'évaluation dans Azure ML"""

        # Préparer les données
        evaluation_data = []
        for i, (q, a, gt, ctx) in enumerate(zip(questions, answers, ground_truth, contexts)):
            evaluation_data.append({
                "id": f"sample_{i}",
                "question": q,
                "answer": a,
                "ground_truth": gt,
                "contexts": ctx,
                "metadata": {
                    "domain": "rag_evaluation",
                    "timestamp": "2024-12-01",
                    "version": "1.0"
                }
            })

        # Sauvegarder temporairement
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in evaluation_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\\n')
            temp_file = f.name

        # Créer le data asset Azure ML
        data_asset = Data(
            name=dataset_name,
            description=f"RAG evaluation dataset - {dataset_name}",
            path=temp_file,
            type="uri_file",
            tags={"purpose": "rag_evaluation", "framework": "rag_chunk_lab"}
        )

        created_data = self.ml_client.data.create_or_update(data_asset)
        print(f"✅ Dataset created: {created_data.name} (version {created_data.version})")

        # Nettoyer le fichier temporaire
        os.unlink(temp_file)

        return created_data.name

    def create_rag_evaluation_flow(self) -> str:
        """Crée un flow d'évaluation RAG personnalisé"""

        # Définition du flow en YAML
        flow_yaml = """
name: rag_evaluation_flow
display_name: "RAG Quality Evaluation Flow"
description: "Comprehensive evaluation flow for RAG systems"

inputs:
  question:
    type: string
    description: "Input question"
  answer:
    type: string
    description: "Generated answer"
  ground_truth:
    type: string
    description: "Expected answer"
  contexts:
    type: list
    description: "Retrieved contexts"

outputs:
  answer_relevancy:
    type: number
    description: "Answer relevancy score"
  faithfulness:
    type: number
    description: "Faithfulness score"
  context_precision:
    type: number
    description: "Context precision score"
  context_recall:
    type: number
    description: "Context recall score"
  overall_score:
    type: number
    description: "Overall quality score"

nodes:
  - name: answer_relevancy_evaluator
    type: python
    source:
      type: code
      path: ./evaluators/answer_relevancy.py
    inputs:
      question: ${inputs.question}
      answer: ${inputs.answer}
    outputs:
      score: ${outputs.answer_relevancy}

  - name: faithfulness_evaluator
    type: python
    source:
      type: code
      path: ./evaluators/faithfulness.py
    inputs:
      answer: ${inputs.answer}
      contexts: ${inputs.contexts}
    outputs:
      score: ${outputs.faithfulness}

  - name: context_evaluator
    type: python
    source:
      type: code
      path: ./evaluators/context_quality.py
    inputs:
      question: ${inputs.question}
      contexts: ${inputs.contexts}
      ground_truth: ${inputs.ground_truth}
    outputs:
      precision: ${outputs.context_precision}
      recall: ${outputs.context_recall}

  - name: score_aggregator
    type: python
    source:
      type: code
      path: ./evaluators/aggregator.py
    inputs:
      answer_relevancy: ${answer_relevancy_evaluator.outputs.score}
      faithfulness: ${faithfulness_evaluator.outputs.score}
      context_precision: ${context_evaluator.outputs.precision}
      context_recall: ${context_evaluator.outputs.recall}
    outputs:
      overall: ${outputs.overall_score}
"""

        # Créer les fichiers des évaluateurs
        self._create_evaluator_scripts()

        # Sauvegarder le flow
        flow_file = "azure_flows/rag_evaluation_flow.yaml"
        os.makedirs(os.path.dirname(flow_file), exist_ok=True)
        with open(flow_file, 'w') as f:
            f.write(flow_yaml)

        print(f"✅ Evaluation flow created: {flow_file}")
        return flow_file

    def _create_evaluator_scripts(self):
        """Crée les scripts d'évaluation Python"""

        # Script d'évaluation de la pertinence
        answer_relevancy_script = '''
import openai
import json
from typing import Dict

def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Évalue la pertinence de la réponse"""

    prompt = f"""
    Evaluate how relevant the answer is to the question on a scale of 0 to 1.

    Question: {question}
    Answer: {answer}

    Return only a number between 0 and 1.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        score = float(response.choices[0].message.content.strip())
        return max(0, min(1, score))

    except Exception as e:
        print(f"Error in answer relevancy evaluation: {e}")
        return 0.5

def main(question: str, answer: str) -> Dict[str, float]:
    """Point d'entrée principal"""
    score = evaluate_answer_relevancy(question, answer)
    return {"score": score}
'''

        # Script d'évaluation de la fidélité
        faithfulness_script = '''
import openai
from typing import List, Dict

def evaluate_faithfulness(answer: str, contexts: List[str]) -> float:
    """Évalue la fidélité de la réponse par rapport au contexte"""

    context_text = "\\n".join(contexts)

    prompt = f"""
    Evaluate if the answer is faithful to the given context (no hallucinations).
    Rate from 0 to 1, where 1 means completely faithful.

    Context: {context_text}
    Answer: {answer}

    Return only a number between 0 and 1.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        score = float(response.choices[0].message.content.strip())
        return max(0, min(1, score))

    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.5

def main(answer: str, contexts: List[str]) -> Dict[str, float]:
    """Point d'entrée principal"""
    score = evaluate_faithfulness(answer, contexts)
    return {"score": score}
'''

        # Script d'évaluation du contexte
        context_quality_script = '''
import openai
from typing import List, Dict

def evaluate_context_quality(question: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Évalue la qualité du contexte récupéré"""

    context_text = "\\n".join(contexts)

    # Évaluation de la précision
    precision_prompt = f"""
    Rate how precise/relevant the retrieved contexts are for answering the question.
    Scale: 0 to 1, where 1 means all contexts are highly relevant.

    Question: {question}
    Contexts: {context_text}

    Return only a number between 0 and 1.
    """

    # Évaluation du rappel
    recall_prompt = f"""
    Rate how much of the necessary information for the ground truth answer is covered by the contexts.
    Scale: 0 to 1, where 1 means all necessary information is present.

    Question: {question}
    Ground Truth: {ground_truth}
    Contexts: {context_text}

    Return only a number between 0 and 1.
    """

    try:
        # Précision
        precision_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": precision_prompt}],
            temperature=0
        )
        precision = float(precision_response.choices[0].message.content.strip())
        precision = max(0, min(1, precision))

        # Rappel
        recall_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": recall_prompt}],
            temperature=0
        )
        recall = float(recall_response.choices[0].message.content.strip())
        recall = max(0, min(1, recall))

        return {"precision": precision, "recall": recall}

    except Exception as e:
        print(f"Error in context evaluation: {e}")
        return {"precision": 0.5, "recall": 0.5}

def main(question: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Point d'entrée principal"""
    return evaluate_context_quality(question, contexts, ground_truth)
'''

        # Script d'agrégation
        aggregator_script = '''
from typing import Dict

def aggregate_scores(answer_relevancy: float, faithfulness: float,
                    context_precision: float, context_recall: float) -> float:
    """Agrège les scores individuels en score global"""

    # Pondération personnalisable
    weights = {
        "answer_relevancy": 0.3,
        "faithfulness": 0.3,
        "context_precision": 0.2,
        "context_recall": 0.2
    }

    overall_score = (
        answer_relevancy * weights["answer_relevancy"] +
        faithfulness * weights["faithfulness"] +
        context_precision * weights["context_precision"] +
        context_recall * weights["context_recall"]
    )

    return overall_score

def main(answer_relevancy: float, faithfulness: float,
         context_precision: float, context_recall: float) -> Dict[str, float]:
    """Point d'entrée principal"""
    overall = aggregate_scores(answer_relevancy, faithfulness, context_precision, context_recall)
    return {"overall": overall}
'''

        # Créer les fichiers
        evaluators_dir = "azure_flows/evaluators"
        os.makedirs(evaluators_dir, exist_ok=True)

        scripts = {
            "answer_relevancy.py": answer_relevancy_script,
            "faithfulness.py": faithfulness_script,
            "context_quality.py": context_quality_script,
            "aggregator.py": aggregator_script
        }

        for filename, content in scripts.items():
            with open(f"{evaluators_dir}/{filename}", 'w') as f:
                f.write(content)

        print(f"✅ Evaluator scripts created in {evaluators_dir}")

    def run_evaluation_job(self, dataset_name: str, flow_path: str,
                          pipeline_name: str) -> str:
        """Lance un job d'évaluation sur Azure"""

        from azure.ai.ml import command
        from azure.ai.ml.entities import Job

        # Configuration du job
        job = command(
            code="./azure_flows",
            command="python run_evaluation.py --dataset ${{inputs.dataset}} --output ${{outputs.results}}",
            inputs={
                "dataset": Input(type="uri_file", path=f"azureml://datastores/workspaceblobstore/paths/{dataset_name}")
            },
            outputs={
                "results": Output(type="uri_folder")
            },
            environment="azureml://registries/azureml/environments/sklearn-1.0/versions/1",
            compute="rag-eval-cluster",
            display_name=f"RAG Evaluation - {pipeline_name}",
            description=f"Comprehensive evaluation for {pipeline_name} pipeline"
        )

        # Soumettre le job
        submitted_job = self.ml_client.jobs.create_or_update(job)
        print(f"✅ Evaluation job submitted: {submitted_job.name}")
        print(f"🌐 View in Azure ML Studio: {submitted_job.studio_url}")

        return submitted_job.name

    def setup_monitoring_dashboard(self) -> str:
        """Configure un dashboard de monitoring"""

        dashboard_config = {
            "name": "RAG Quality Monitoring",
            "description": "Real-time monitoring of RAG system quality",
            "widgets": [
                {
                    "type": "metric_chart",
                    "title": "Answer Relevancy Trend",
                    "metric": "answer_relevancy",
                    "aggregation": "mean",
                    "time_range": "7d"
                },
                {
                    "type": "metric_chart",
                    "title": "Faithfulness Score",
                    "metric": "faithfulness",
                    "aggregation": "mean",
                    "time_range": "7d"
                },
                {
                    "type": "alert_status",
                    "title": "Quality Alerts",
                    "alerts": ["low_relevancy", "high_hallucination"]
                }
            ],
            "refresh_interval": "5m"
        }

        # Sauvegarder la configuration
        dashboard_file = "azure_monitoring/dashboard_config.json"
        os.makedirs(os.path.dirname(dashboard_file), exist_ok=True)
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard_config, f, indent=2)

        print(f"✅ Monitoring dashboard configured: {dashboard_file}")
        return dashboard_file
'''

    return flows_code

def azure_foundry_advanced_features():
    """
    Fonctionnalités avancées Azure AI Foundry
    """
    advanced_code = '''
# FONCTIONNALITÉS AVANCÉES AZURE AI FOUNDRY

# 1. MLflow Integration pour Tracking
import mlflow
import mlflow.azureml
from azure.ai.ml import MLClient

class MLflowRAGTracker:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client
        # Configurer MLflow pour Azure ML
        mlflow.set_tracking_uri(ml_client.workspaces.get().mlflow_tracking_uri)

    def track_experiment(self, experiment_name: str, pipeline_results: Dict):
        """Track une expérience d'évaluation avec MLflow"""

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Logger les métriques
            for metric_name, metric_value in pipeline_results.get("metrics_scores", {}).items():
                if isinstance(metric_value, dict) and "mean_score" in metric_value:
                    mlflow.log_metric(f"{metric_name}_mean", metric_value["mean_score"])
                    mlflow.log_metric(f"{metric_name}_min", metric_value.get("min_score", 0))
                    mlflow.log_metric(f"{metric_name}_max", metric_value.get("max_score", 0))

            # Logger les paramètres
            mlflow.log_param("pipeline_name", pipeline_results.get("pipeline"))
            mlflow.log_param("total_questions", pipeline_results.get("total_questions"))
            mlflow.log_param("evaluation_time", pipeline_results.get("evaluation_time"))

            # Logger les artefacts
            results_file = "evaluation_results.json"
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            mlflow.log_artifact(results_file)

            print(f"✅ Experiment tracked: {mlflow.active_run().info.run_id}")

# 2. Automated Model Registry
class RAGModelRegistry:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client

    def register_best_pipeline(self, comparison_results: Dict, threshold: float = 0.8):
        """Enregistre automatiquement le meilleur pipeline"""

        # Trouver le meilleur pipeline
        best_pipeline = None
        best_score = 0

        for pipeline, results in comparison_results.items():
            overall_score = results.get("overall_score", 0)
            if overall_score > best_score:
                best_score = overall_score
                best_pipeline = pipeline

        if best_score >= threshold:
            # Enregistrer le modèle
            from azure.ai.ml.entities import Model

            model = Model(
                name=f"rag_best_pipeline",
                version=f"v{int(time.time())}",
                description=f"Best performing RAG pipeline: {best_pipeline} (score: {best_score:.3f})",
                path="./model_artifacts",
                tags={
                    "pipeline": best_pipeline,
                    "score": str(best_score),
                    "evaluation_date": "2024-12-01",
                    "framework": "rag_chunk_lab"
                }
            )

            registered_model = self.ml_client.models.create_or_update(model)
            print(f"✅ Model registered: {registered_model.name}:{registered_model.version}")

            return registered_model
        else:
            print(f"⚠️ No pipeline meets threshold {threshold} (best: {best_score:.3f})")
            return None

# 3. Automated A/B Testing
class AzureABTestManager:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client

    def setup_ab_test(self, pipeline_a: str, pipeline_b: str,
                     traffic_split: float = 0.5) -> str:
        """Configure un test A/B automatisé"""

        from azure.ai.ml.entities import OnlineEndpoint, OnlineDeployment

        # Créer l'endpoint
        endpoint = OnlineEndpoint(
            name=f"rag-ab-test-{int(time.time())}",
            description=f"A/B test: {pipeline_a} vs {pipeline_b}",
            auth_mode="key",
            tags={"purpose": "ab_testing", "pipelines": f"{pipeline_a},{pipeline_b}"}
        )

        created_endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint)

        # Créer les déploiements
        deployment_a = OnlineDeployment(
            name=f"deployment-{pipeline_a}",
            endpoint_name=endpoint.name,
            model="rag_best_pipeline:latest",
            instance_type="Standard_DS3_v2",
            instance_count=1,
            environment_variables={"PIPELINE_TYPE": pipeline_a}
        )

        deployment_b = OnlineDeployment(
            name=f"deployment-{pipeline_b}",
            endpoint_name=endpoint.name,
            model="rag_best_pipeline:latest",
            instance_type="Standard_DS3_v2",
            instance_count=1,
            environment_variables={"PIPELINE_TYPE": pipeline_b}
        )

        # Configurer le traffic split
        created_endpoint.traffic = {
            f"deployment-{pipeline_a}": int(traffic_split * 100),
            f"deployment-{pipeline_b}": int((1 - traffic_split) * 100)
        }

        print(f"✅ A/B test configured: {created_endpoint.name}")
        return created_endpoint.name

# 4. Real-time Quality Monitoring
class RealTimeQualityMonitor:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client

    def setup_data_drift_monitor(self, dataset_name: str):
        """Configure la surveillance de dérive des données"""

        from azure.ai.ml.entities import DataDriftMonitor

        monitor = DataDriftMonitor(
            name="rag_data_drift_monitor",
            description="Monitor for data drift in RAG inputs",
            target_dataset=dataset_name,
            baseline_dataset=f"{dataset_name}_baseline",
            features=["question_length", "question_complexity", "domain_keywords"],
            threshold=0.1,
            frequency="daily",
            alert_threshold=0.05
        )

        print(f"✅ Data drift monitor configured")

    def setup_model_drift_monitor(self, model_name: str):
        """Configure la surveillance de dérive du modèle"""

        # Configuration de monitoring personnalisé
        monitoring_config = {
            "model_name": model_name,
            "metrics_to_monitor": [
                "answer_relevancy",
                "faithfulness",
                "response_time",
                "error_rate"
            ],
            "alert_rules": [
                {
                    "metric": "answer_relevancy",
                    "threshold": 0.7,
                    "comparison": "less_than",
                    "severity": "high"
                },
                {
                    "metric": "response_time",
                    "threshold": 2.0,
                    "comparison": "greater_than",
                    "severity": "medium"
                }
            ],
            "notification_channels": ["email", "webhook"]
        }

        # Sauvegarder la configuration
        config_file = f"monitoring_configs/{model_name}_monitor.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)

        print(f"✅ Model drift monitor configured: {config_file}")

# 5. Automated Reporting
class AzureReportGenerator:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client

    def generate_executive_report(self, evaluation_results: Dict) -> str:
        """Génère un rapport exécutif automatisé"""

        from datetime import datetime

        report = {
            "title": "RAG System Quality Report",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": self._generate_executive_summary(evaluation_results),
            "key_findings": self._extract_key_findings(evaluation_results),
            "recommendations": self._generate_recommendations(evaluation_results),
            "detailed_metrics": evaluation_results,
            "next_steps": self._suggest_next_steps(evaluation_results)
        }

        # Générer le rapport HTML
        html_report = self._generate_html_report(report)

        report_file = f"reports/executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"📊 Executive report generated: {report_file}")
        return report_file

    def _generate_executive_summary(self, results: Dict) -> str:
        """Génère un résumé exécutif"""
        # Analyser les résultats et générer un résumé
        return "Executive summary based on evaluation results..."

    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extrait les principales découvertes"""
        findings = []
        # Logique pour extraire les insights clés
        return findings

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Génère des recommandations automatiques"""
        recommendations = []
        # Logique pour générer des recommandations
        return recommendations

    def _suggest_next_steps(self, results: Dict) -> List[str]:
        """Suggère les prochaines étapes"""
        steps = []
        # Logique pour suggérer les actions à prendre
        return steps

    def _generate_html_report(self, report: Dict) -> str:
        """Génère le rapport HTML"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #0078d4; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ padding: 10px; margin: 5px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report['title']}</h1>
                <p>Generated: {report['generated_at']}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <p>{report['executive_summary']}</p>
            </div>

            <div class="section">
                <h2>Key Findings</h2>
                <ul>
                {''.join([f"<li>{finding}</li>" for finding in report['key_findings']])}
                </ul>
            </div>

            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                {''.join([f"<li>{rec}</li>" for rec in report['recommendations']])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html

# 6. Cost Optimization
class CostOptimizer:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client

    def optimize_compute_usage(self):
        """Optimise l'usage des ressources de calcul"""

        # Analyser l'utilisation des compute instances
        compute_instances = self.ml_client.compute.list()

        for compute in compute_instances:
            if compute.type == "ComputeInstance":
                # Vérifier l'utilisation et ajuster
                usage_stats = self._get_compute_usage(compute.name)

                if usage_stats["idle_time"] > 0.8:  # 80% idle
                    print(f"💰 Compute {compute.name} has high idle time - consider resizing")

    def _get_compute_usage(self, compute_name: str) -> Dict:
        """Récupère les statistiques d'utilisation"""
        # Simuler les statistiques d'utilisation
        return {"idle_time": 0.3, "cpu_usage": 0.4, "memory_usage": 0.5}

    def estimate_evaluation_cost(self, num_questions: int, pipelines: int) -> Dict:
        """Estime le coût d'une évaluation"""

        # Coûts approximatifs (à ajuster selon vos tarifs)
        costs = {
            "compute_hours": (num_questions * pipelines * 0.1) / 3600,  # secondes en heures
            "storage_gb": (num_questions * 0.001),  # estimation stockage
            "api_calls": num_questions * pipelines * 4,  # appels LLM pour évaluation
        }

        total_cost = (
            costs["compute_hours"] * 0.50 +  # $0.50/heure compute
            costs["storage_gb"] * 0.02 +     # $0.02/GB storage
            costs["api_calls"] * 0.002       # $0.002/appel API
        )

        costs["total_estimated_usd"] = total_cost

        return costs
'''

    return advanced_code

def azure_foundry_best_practices():
    """
    Meilleures pratiques Azure AI Foundry
    """
    practices = '''
# 🏆 MEILLEURES PRATIQUES AZURE AI FOUNDRY

## 1. Organisation et Gouvernance
- **Workspace Structure**: Séparez dev/test/prod
- **Naming Conventions**: Utilisez des noms cohérents
- **Tags**: Taggez toutes les ressources pour le tracking
- **Access Control**: Utilisez RBAC pour contrôler l'accès

## 2. Gestion des Coûts
```python
# Monitoring des coûts
def monitor_costs():
    # Configurer des budgets
    budget_config = {
        "monthly_limit": 1000,  # $1000/mois
        "alert_thresholds": [50, 80, 95],  # % du budget
        "notification_emails": ["admin@company.com"]
    }

    # Auto-scaling des ressources
    auto_scale_config = {
        "min_instances": 0,
        "max_instances": 4,
        "scale_down_delay": 300  # 5 minutes
    }
```

## 3. Sécurité et Conformité
- **Key Vault**: Stockez les secrets dans Azure Key Vault
- **Private Endpoints**: Utilisez des endpoints privés
- **Audit Logs**: Activez l'audit pour la traçabilité
- **Data Encryption**: Chiffrez les données au repos et en transit

## 4. Performance et Scalabilité
```python
# Configuration optimisée
performance_config = {
    "compute_type": "Standard_D4s_v3",  # Équilibre performance/coût
    "parallel_jobs": 4,                 # Traitement parallèle
    "batch_size": 100,                  # Optimiser le batch processing
    "cache_results": True               # Cache pour éviter recalculs
}
```

## 5. Monitoring et Alertes
- **Custom Metrics**: Définissez des métriques métier
- **Alert Rules**: Configurez des alertes proactives
- **Dashboards**: Créez des dashboards pour différents stakeholders
- **SLA Monitoring**: Surveillez les SLA de qualité

## 6. CI/CD Integration
```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
    - main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: AzureCLI@2
  displayName: 'Run RAG Evaluation'
  inputs:
    azureSubscription: 'azure-ml-connection'
    scriptType: 'bash'
    scriptLocation: 'inlineScript'
    inlineScript: |
      python run_evaluation.py --pipeline all --threshold 0.8

- task: PublishTestResults@2
  displayName: 'Publish Results'
  inputs:
    testResultsFiles: '**/evaluation_results.xml'
```

## 7. Data Management
- **Versioning**: Versionnez vos datasets
- **Lineage**: Trackez la lignée des données
- **Quality Gates**: Implémentez des gates de qualité
- **Backup Strategy**: Stratégie de sauvegarde robuste
'''

    return practices

# Fonction principale pour créer le tutoriel complet
def create_azure_foundry_complete_tutorial():
    """
    Crée le tutoriel complet Azure AI Foundry
    """
    tutorial_content = f"""
# 🌟 Azure AI Foundry - Tutoriel Complet pour RAG Chunk Lab

{install_azure_foundry()}

## ⚙️ Configuration Initiale
{setup_azure_foundry()}

## 🔄 Flows d'Évaluation Personnalisés
{create_azure_foundry_evaluation_flows()}

## 🚀 Fonctionnalités Avancées
{azure_foundry_advanced_features()}

## 🏆 Meilleures Pratiques
{azure_foundry_best_practices()}

## 📚 Ressources Additionnelles

### Documentation Officielle
- Azure ML Documentation: https://docs.microsoft.com/en-us/azure/machine-learning/
- Python SDK: https://docs.microsoft.com/en-us/python/api/azure-ai-ml/
- MLflow Integration: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow
- Prompt Flow: https://microsoft.github.io/promptflow/

### Exemples et Templates
- GitHub Examples: https://github.com/Azure/azureml-examples
- RAG Templates: https://github.com/Azure/azureml-examples/tree/main/sdk/python/generative-ai
- MLOps Examples: https://github.com/microsoft/MLOpsPython

### Formation et Certification
- Azure ML Learning Path: https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-machine-learning/
- AI Engineer Certification: https://docs.microsoft.com/en-us/learn/certifications/azure-ai-engineer/
- MLOps Best Practices: https://docs.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment

### Support et Community
- Microsoft Q&A: https://docs.microsoft.com/en-us/answers/topics/azure-machine-learning.html
- Stack Overflow: https://stackoverflow.com/questions/tagged/azure-machine-learning
- GitHub Issues: https://github.com/Azure/azure-sdk-for-python/issues
- Azure Updates: https://azure.microsoft.com/en-us/updates/?product=machine-learning

### Pricing et Calculateur
- Azure ML Pricing: https://azure.microsoft.com/en-us/pricing/details/machine-learning/
- Pricing Calculator: https://azure.microsoft.com/en-us/pricing/calculator/
- Cost Management: https://docs.microsoft.com/en-us/azure/cost-management-billing/
"""

    # Créer le dossier tutorials s'il n'existe pas
    os.makedirs("tutorials", exist_ok=True)

    # Sauvegarder le tutoriel
    with open("tutorials/azure_foundry_complete_tutorial.md", "w", encoding="utf-8") as f:
        f.write(tutorial_content)

    return "tutorials/azure_foundry_complete_tutorial.md"

if __name__ == "__main__":
    tutorial_file = create_azure_foundry_complete_tutorial()
    print(f"📚 Tutoriel Azure AI Foundry créé: {tutorial_file}")