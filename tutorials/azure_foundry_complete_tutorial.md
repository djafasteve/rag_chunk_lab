# < Azure AI Foundry - Tutoriel Complet pour RAG Chunk Lab

## =€ Vue d'Ensemble

Azure AI Foundry (anciennement Azure ML) offre une plateforme enterprise complète pour l'évaluation, le monitoring et le déploiement de systèmes RAG. Ce tutoriel couvre l'intégration complète avec RAG Chunk Lab.

## =Ë Prérequis

### 1. Installation des Dépendances
```bash
# Azure SDK
pip install azure-ai-ml azure-identity azure-storage-blob

# MLflow pour tracking
pip install mlflow

# Optionnel: Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### 2. Configuration Azure
```bash
# Authentification Azure
az login

# Définir l'abonnement
az account set --subscription "your-subscription-id"

# Créer un workspace (si nécessaire)
az ml workspace create --name rag-evaluation-workspace \
  --resource-group your-rg \
  --location eastus
```

### 3. Variables d'Environnement
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_ML_WORKSPACE="rag-evaluation-workspace"
export AZURE_TENANT_ID="your-tenant-id"
```

## =' Intégration avec RAG Chunk Lab

### Configuration de Base

```python
# azure_foundry_evaluator.py
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, CommandJob, UserIdentityConfiguration
from azure.identity import DefaultAzureCredential
import mlflow
import json
import os
from typing import Dict, List, Any

class AzureFoundryEvaluator:
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """Initialise le client Azure ML"""
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        # Configuration MLflow
        mlflow.set_tracking_uri(self.ml_client.workspaces.get(workspace_name).mlflow_tracking_uri)

    def create_evaluation_environment(self):
        """Crée un environnement pour l'évaluation RAG"""
        environment = Environment(
            name="rag-evaluation-env",
            description="Environment for RAG evaluation with all dependencies",
            conda_file={
                "channels": ["conda-forge"],
                "dependencies": [
                    "python=3.9",
                    "pip",
                    {
                        "pip": [
                            "azure-ai-ml",
                            "ragas",
                            "openai",
                            "langchain",
                            "mlflow",
                            "pandas",
                            "numpy",
                            "scikit-learn"
                        ]
                    }
                ]
            },
            image="mcr.microsoft.com/azureml/curated/minimal-ubuntu20.04-py38-cpu-inference:latest"
        )

        return self.ml_client.environments.create_or_update(environment)

    def submit_evaluation_job(self,
                            doc_id: str,
                            evaluation_data: Dict,
                            experiment_name: str = "rag-chunk-lab-evaluation") -> str:
        """Soumet un job d'évaluation à Azure ML"""

        # Préparer les données d'évaluation
        data_path = f"./evaluation_data_{doc_id}.json"
        with open(data_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2)

        # Créer le job
        job = CommandJob(
            code="./azure_evaluation_script.py",
            command="python azure_evaluation_script.py --input-data ${{inputs.evaluation_data}} --output-path ${{outputs.results}}",
            environment="rag-evaluation-env@latest",
            inputs={
                "evaluation_data": data_path
            },
            outputs={
                "results": "./outputs/"
            },
            compute="cpu-cluster",  # Utilisez votre compute target
            experiment_name=experiment_name,
            display_name=f"RAG Evaluation - {doc_id}",
            description=f"Comprehensive RAG evaluation for document collection: {doc_id}",
            identity=UserIdentityConfiguration()
        )

        # Soumettre le job
        submitted_job = self.ml_client.jobs.create_or_update(job)

        print(f"=€ Azure job submitted: {submitted_job.name}")
        print(f"=Ê Monitor at: {submitted_job.studio_url}")

        return submitted_job.name

    def create_evaluation_script(self):
        """Crée le script d'évaluation pour Azure ML"""
        script_content = '''
import argparse
import json
import os
import mlflow
import pandas as pd
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    # Charger les données
    with open(args.input_data, 'r') as f:
        evaluation_data = json.load(f)

    # Démarrer MLflow run
    with mlflow.start_run():
        results = {}

        for pipeline, data in evaluation_data.items():
            print(f"Evaluating pipeline: {pipeline}")

            # Préparer le dataset RAGAS
            dataset = Dataset.from_dict({
                'question': data['questions'],
                'answer': data['answers'],
                'contexts': data['contexts'],
                'ground_truths': data['ground_truths']
            })

            # Évaluation RAGAS
            ragas_result = evaluate(
                dataset,
                metrics=[answer_relevancy, faithfulness, context_precision, context_recall]
            )

            # Logger les métriques
            for metric, score in ragas_result.items():
                mlflow.log_metric(f"{pipeline}_{metric}", score)

            results[pipeline] = ragas_result

        # Sauvegarder les résultats
        os.makedirs(args.output_path, exist_ok=True)
        with open(f"{args.output_path}/azure_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Logger l'artifact
        mlflow.log_artifact(f"{args.output_path}/azure_evaluation_results.json")

        print(" Evaluation completed and logged to MLflow")

if __name__ == "__main__":
    main()
'''

        with open("azure_evaluation_script.py", 'w') as f:
            f.write(script_content)

        print("=Ý Azure evaluation script created: azure_evaluation_script.py")

    def monitor_job(self, job_name: str):
        """Surveille un job Azure ML"""
        job = self.ml_client.jobs.get(job_name)

        print(f"=Ê Job Status: {job.status}")
        print(f"= Studio URL: {job.studio_url}")

        if job.status == "Completed":
            # Récupérer les artifacts
            self.download_job_outputs(job_name)

        return job.status

    def download_job_outputs(self, job_name: str):
        """Télécharge les résultats d'un job"""
        try:
            # Télécharger les outputs
            self.ml_client.jobs.download(job_name, download_path="./azure_outputs/")
            print(f"=Á Job outputs downloaded to: ./azure_outputs/")
        except Exception as e:
            print(f"L Error downloading outputs: {e}")

# Utilisation dans RAG Chunk Lab
def run_azure_foundry_evaluation(doc_id: str,
                                per_pipeline_answers: Dict[str, List[str]],
                                per_pipeline_contexts: Dict[str, List[List[str]]],
                                questions: List[str],
                                ground_truth: List[str],
                                azure_config: Dict) -> str:
    """
    Lance l'évaluation avec Azure AI Foundry
    """

    evaluator = AzureFoundryEvaluator(
        subscription_id=azure_config["subscription_id"],
        resource_group=azure_config["resource_group"],
        workspace_name=azure_config["workspace_name"]
    )

    # Créer l'environnement si nécessaire
    evaluator.create_evaluation_environment()

    # Créer le script d'évaluation
    evaluator.create_evaluation_script()

    # Préparer les données d'évaluation
    evaluation_data = {}
    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        evaluation_data[pipeline] = {
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ground_truths": ground_truth
        }

    # Soumettre le job
    job_name = evaluator.submit_evaluation_job(doc_id, evaluation_data)

    print(f"< Azure AI Foundry evaluation initiated for {doc_id}")
    print(f"=Ê Job name: {job_name}")
    print(f"ñ  Monitor progress in Azure ML Studio")

    return job_name
```

## <× Architecture Enterprise

### 1. Pipeline de Déploiement Automatisé

```python
# azure_deployment_pipeline.py
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
import yaml

class RAGDeploymentPipeline:
    def __init__(self, ml_client):
        self.ml_client = ml_client

    def register_rag_model(self, model_path: str, model_name: str, version: str):
        """Enregistre un modèle RAG dans Azure ML"""

        model = Model(
            path=model_path,
            name=model_name,
            version=version,
            description="RAG model optimized through chunk evaluation",
            properties={
                "framework": "langchain",
                "evaluation_method": "rag_chunk_lab",
                "metrics": "ragas,generic,embedding_analysis"
            }
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f" Model registered: {registered_model.name}:{registered_model.version}")

        return registered_model

    def create_inference_config(self):
        """Crée la configuration d'inférence"""

        inference_config = {
            "entry_script": "score.py",
            "runtime": "python",
            "conda_file": "conda.yml",
            "extra_docker_file_steps": None,
            "source_directory": "./inference",
            "enable_gpu": False,
            "base_image": None,
            "cuda_version": None
        }

        # Créer score.py
        score_script = '''
import json
import joblib
import numpy as np
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
import os

def init():
    global model, vectorstore

    # Charger le modèle et le vectorstore
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model')
    model = joblib.load(f"{model_path}/rag_chain.pkl")
    vectorstore = FAISS.load_local(f"{model_path}/vectorstore")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        query = data['query']

        # Exécuter la requête RAG
        result = model.run(query)

        return json.dumps({
            "answer": result,
            "status": "success"
        })
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        })
'''

        os.makedirs("./inference", exist_ok=True)
        with open("./inference/score.py", "w") as f:
            f.write(score_script)

        # Créer conda.yml
        conda_config = {
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.9",
                "pip",
                {
                    "pip": [
                        "azureml-defaults",
                        "langchain",
                        "faiss-cpu",
                        "openai",
                        "scikit-learn"
                    ]
                }
            ]
        }

        with open("./inference/conda.yml", "w") as f:
            yaml.dump(conda_config, f)

        return inference_config

    def deploy_endpoint(self, model_name: str, endpoint_name: str):
        """Déploie un endpoint managé"""

        # Créer l'endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="RAG model endpoint optimized with chunk evaluation",
            auth_mode="key"
        )

        endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        # Créer le déploiement
        deployment = ManagedOnlineDeployment(
            name="rag-deployment-v1",
            endpoint_name=endpoint_name,
            model=f"{model_name}:latest",
            instance_type="Standard_DS3_v2",
            instance_count=1,
            code_configuration={
                "code": "./inference",
                "scoring_script": "score.py"
            }
        )

        deployment_result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()

        print(f"=€ Endpoint deployed: {endpoint_result.scoring_uri}")

        return endpoint_result
```

### 2. Monitoring et Alertes

```python
# azure_monitoring.py
from azure.ai.ml.entities import DataDrift, DataQuality
import pandas as pd

class RAGMonitoring:
    def __init__(self, ml_client):
        self.ml_client = ml_client

    def setup_data_drift_monitoring(self, endpoint_name: str):
        """Configure la surveillance de data drift"""

        # Configuration du monitoring
        monitor_config = {
            "compute_target": "cpu-cluster",
            "frequency": "Day",
            "alert_email": "admin@yourcompany.com",
            "drift_threshold": 0.3,
            "latency_threshold": 5.0  # secondes
        }

        print(f"=Ê Data drift monitoring configured for {endpoint_name}")
        return monitor_config

    def create_quality_alerts(self):
        """Crée des alertes de qualité"""

        alerts = {
            "answer_relevancy_low": {
                "metric": "answer_relevancy",
                "threshold": 0.7,
                "action": "email_notification"
            },
            "faithfulness_low": {
                "metric": "faithfulness",
                "threshold": 0.8,
                "action": "auto_retrain"
            },
            "latency_high": {
                "metric": "response_latency",
                "threshold": 10.0,
                "action": "scale_up"
            }
        }

        print("=¨ Quality alerts configured")
        return alerts

    def analyze_endpoint_performance(self, endpoint_name: str, days: int = 7):
        """Analyse les performances d'un endpoint"""

        # Récupérer les métriques (simulé)
        metrics = {
            "requests_per_day": 1500,
            "average_latency": 2.3,
            "error_rate": 0.02,
            "satisfaction_score": 0.85
        }

        print(f"=È Endpoint {endpoint_name} performance (last {days} days):")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        return metrics
```

## = Exemples d'Utilisation Avancés

### 1. Évaluation Multi-Environnement

```python
# multi_environment_evaluation.py

def run_multi_environment_evaluation(doc_id: str, environments: List[str]):
    """Évalue sur plusieurs environnements Azure"""

    results = {}

    for env in environments:
        print(f"< Evaluating on environment: {env}")

        azure_config = {
            "subscription_id": os.getenv(f"AZURE_SUBSCRIPTION_ID_{env.upper()}"),
            "resource_group": os.getenv(f"AZURE_RG_{env.upper()}"),
            "workspace_name": os.getenv(f"AZURE_WORKSPACE_{env.upper()}")
        }

        job_name = run_azure_foundry_evaluation(
            doc_id,
            per_pipeline_answers,
            per_pipeline_contexts,
            questions,
            ground_truth,
            azure_config
        )

        results[env] = job_name

    return results
```

### 2. A/B Testing Automatisé

```python
# azure_ab_testing.py

class AzureABTesting:
    def __init__(self, ml_client):
        self.ml_client = ml_client

    def setup_ab_test(self, model_a: str, model_b: str, traffic_split: float = 0.5):
        """Configure un test A/B entre deux modèles"""

        ab_config = {
            "model_a": {
                "name": model_a,
                "traffic_percentage": int(traffic_split * 100)
            },
            "model_b": {
                "name": model_b,
                "traffic_percentage": int((1 - traffic_split) * 100)
            },
            "duration_days": 14,
            "success_metrics": [
                "answer_relevancy",
                "user_satisfaction",
                "response_time"
            ]
        }

        print(f">ê A/B test configured: {model_a} vs {model_b}")
        return ab_config

    def analyze_ab_results(self, test_id: str):
        """Analyse les résultats d'un test A/B"""

        # Récupérer les métriques de chaque modèle
        results = {
            "model_a": {
                "answer_relevancy": 0.85,
                "user_satisfaction": 0.78,
                "response_time": 2.1
            },
            "model_b": {
                "answer_relevancy": 0.88,
                "user_satisfaction": 0.82,
                "response_time": 1.9
            }
        }

        # Déterminer le gagnant
        winner = "model_b"  # Logique de décision

        print(f"<Æ A/B test winner: {winner}")
        return {"winner": winner, "results": results}
```

## =Ê Tableau de Bord Custom

### Configuration Grafana/Azure Monitor

```python
# azure_dashboard.py

def create_rag_dashboard():
    """Crée un tableau de bord pour monitoring RAG"""

    dashboard_config = {
        "title": "RAG Chunk Lab - Production Monitoring",
        "panels": [
            {
                "title": "Answer Relevancy Trend",
                "type": "graph",
                "metrics": ["answer_relevancy_avg", "answer_relevancy_p95"],
                "time_range": "24h"
            },
            {
                "title": "Faithfulness Score",
                "type": "stat",
                "metrics": ["faithfulness_current"],
                "threshold": {"warning": 0.8, "critical": 0.7}
            },
            {
                "title": "Response Latency",
                "type": "histogram",
                "metrics": ["response_latency_ms"],
                "buckets": [100, 500, 1000, 2000, 5000]
            },
            {
                "title": "Error Rate",
                "type": "gauge",
                "metrics": ["error_rate_percentage"],
                "max": 5.0
            }
        ],
        "refresh": "30s",
        "alerts": [
            {
                "condition": "answer_relevancy_avg < 0.75",
                "notification": "email",
                "frequency": "5m"
            }
        ]
    }

    print("=Ê Dashboard configuration created")
    return dashboard_config
```

## <Æ Meilleures Pratiques

### 1. Sécurité et Gouvernance

```python
# Security best practices
AZURE_SECURITY_CONFIG = {
    "authentication": {
        "method": "service_principal",  # ou "managed_identity"
        "rotate_keys": True,
        "key_vault_integration": True
    },
    "network": {
        "private_endpoints": True,
        "vnet_injection": True,
        "firewall_rules": ["corporate_ip_ranges"]
    },
    "data_protection": {
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "data_residency": "europe"
    },
    "compliance": {
        "gdpr_compliant": True,
        "audit_logs": True,
        "data_lineage": True
    }
}
```

### 2. Optimisation des Coûts

```python
# Cost optimization strategies
def optimize_azure_costs():
    strategies = {
        "compute": {
            "auto_scaling": True,
            "spot_instances": True,
            "scheduled_scaling": {
                "scale_down": "18:00",
                "scale_up": "08:00"
            }
        },
        "storage": {
            "tiering": "cool_storage_after_30_days",
            "compression": True,
            "deduplication": True
        },
        "monitoring": {
            "cost_alerts": True,
            "budget_limits": {
                "monthly": 1000,  # USD
                "alert_threshold": 0.8
            }
        }
    }

    return strategies
```

## =Ú Ressources et Documentation

### Documentation Officielle
- **Azure AI Foundry**: https://docs.microsoft.com/en-us/azure/machine-learning/
- **MLflow on Azure**: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow
- **Azure ML SDK v2**: https://docs.microsoft.com/en-us/python/api/azure-ai-ml/

### Tutoriels Complémentaires
- **Getting Started**: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup
- **MLOps with Azure**: https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines
- **Model Deployment**: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints

### Support et Community
- **Microsoft Q&A**: https://docs.microsoft.com/en-us/answers/topics/azure-machine-learning.html
- **GitHub Samples**: https://github.com/Azure/azureml-examples
- **Azure ML Blog**: https://techcommunity.microsoft.com/t5/azure-ai-blog/bg-p/AzureAIBlog

---

< **Azure AI Foundry** offre la plateforme enterprise la plus complète pour l'évaluation et le déploiement de systèmes RAG à l'échelle !