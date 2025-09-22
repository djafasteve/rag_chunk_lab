# 🎯 Guide Complet d'Évaluation RAG - Tous Domaines

Ce guide présente un **protocole d'évaluation multi-niveaux** adaptable à tout domaine, bien au-delà de RAGAS seul.

## 🌟 Vue d'Ensemble des Protocoles

| Protocole | Complexité | Domaine | Usage Recommandé | Coût |
|-----------|------------|---------|------------------|------|
| **RAGAS** | 🟢 Simple | Général | POC, tests rapides | Gratuit |
| **Generic Evaluation** | 🟡 Modéré | Universel | Tous domaines | Gratuit |
| **TruLens** | 🟡 Modéré | Général | Debug, observabilité | OpenAI API |
| **DeepEval** | 🟡 Modéré | Général | Tests unitaires, CI/CD | OpenAI API |
| **Azure AI Foundry** | 🔴 Avancé | Enterprise | Production, gouvernance | Azure costs |

## 🚀 Installation Rapide

```bash
# Installation de base (RAGAS + Generic)
pip install -r requirements.txt

# Pour TruLens
pip install trulens-eval

# Pour DeepEval
pip install deepeval

# Pour Azure AI Foundry
pip install azure-ai-ml azure-identity

# Vérification
python -c "from rag_chunk_lab.generic_evaluation import GenericEvaluationProtocol; print('✅ Tous les protocoles prêts')"
```

## 🎯 Protocoles par Cas d'Usage

### 📊 Évaluation Standard (Recommandée)
```bash
# Protocole complet pour tout domaine
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --ragas \
  --generic-evaluation \
  --embedding-analysis \
  --use-llm
```

**Métriques obtenues :**
- ✅ RAGAS : answer_relevancy, faithfulness, context_precision, context_recall
- ✅ Generic : factual_accuracy, completeness, relevance, consistency, clarity
- ✅ Embeddings : recall@k, MRR, NDCG, diversité, cohérence sémantique

### 🔍 Évaluation avec Observabilité (TruLens)
```bash
# Configuration
export OPENAI_API_KEY="your-key"

# Évaluation avec dashboard temps réel
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --trulens \
  --use-llm
```

**Avantages TruLens :**
- 🌐 Dashboard interactif (http://localhost:8501)
- 🔍 Traçabilité complète des requêtes
- ⚡ Détection d'hallucinations en temps réel
- 📊 Métriques avec chaîne de raisonnement

### 🧪 Évaluation Tests Unitaires (DeepEval)
```bash
# Tests automatisés style pytest
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --deepeval \
  --use-llm
```

**Avantages DeepEval :**
- 🧪 Tests unitaires intégrables en CI/CD
- 🛡️ Métriques de sécurité (biais, toxicité)
- ⚡ Cache intelligent pour éviter re-évaluations
- 📈 Intégration pytest native

### 🌟 Évaluation Enterprise (Azure AI Foundry)
```bash
# Configuration Azure
export AZURE_SUBSCRIPTION_ID="your-subscription"
export AZURE_RESOURCE_GROUP="your-rg"
export AZURE_ML_WORKSPACE="your-workspace"

# Évaluation enterprise avec monitoring
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --azure-foundry \
  --use-llm
```

**Avantages Azure AI Foundry :**
- 🏢 Flows d'évaluation personnalisés
- 📊 A/B testing automatisé
- 🔄 Monitoring continu en production
- 📈 Gouvernance et traçabilité enterprise

## 🎯 Évaluation Complète (Tous Protocoles)

```bash
# Le grand chelem : tous les protocoles en une fois
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --ragas \
  --generic-evaluation \
  --embedding-analysis \
  --trulens \
  --deepeval \
  --azure-foundry \
  --use-llm
```

## 📁 Structure des Résultats

```
exports/ma_collection/
├── ragas_summary.csv              # RAGAS standard
├── ragas_per_question.csv         # Détail RAGAS
├── generic_evaluation_*.json      # Évaluation générique
├── generic_metrics_summary_*.csv  # Métriques CSV

embedding_analysis/
├── embedding_analysis_*.json      # Analyse embeddings
├── embedding_metrics_summary.csv  # Métriques embeddings

trulens_results/
├── trulens_evaluation_*.json      # Résultats TruLens
└── dashboard_data/                # Données dashboard

deepeval_results/
├── deepeval_evaluation_*.json     # Résultats DeepEval
├── deepeval_detailed_results.csv  # Export détaillé
└── pipeline_comparison.json       # Comparaison pipelines

azure_foundry/
├── evaluation_jobs/               # Jobs Azure ML
├── monitoring_configs/            # Configs monitoring
└── reports/                       # Rapports exécutifs
```

## 🔧 Configuration par Domaine

### 📚 Documents Techniques
```python
domain_config = {
    "keywords": {
        "technical": ["API", "configuration", "parameter", "function"],
        "process": ["step", "procedure", "workflow", "pipeline"],
        "quality": ["performance", "reliability", "scalability"]
    },
    "patterns": {
        "versions": r"v?\d+\.\d+\.\d+",
        "urls": r"https?://[^\s]+",
        "code": r"`[^`]+`"
    }
}
```

### 🏢 Documents Business
```python
domain_config = {
    "keywords": {
        "financial": ["budget", "cost", "revenue", "profit"],
        "temporal": ["quarter", "annual", "deadline", "milestone"],
        "metrics": ["KPI", "ROI", "growth", "target"]
    },
    "patterns": {
        "money": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
        "percentages": r"\d+(?:\.\d+)?%",
        "dates": r"\d{1,2}/\d{1,2}/\d{4}"
    }
}
```

### 🔬 Documents Scientifiques
```python
domain_config = {
    "keywords": {
        "methodology": ["hypothesis", "experiment", "analysis", "conclusion"],
        "metrics": ["correlation", "significance", "p-value", "confidence"],
        "temporal": ["duration", "period", "phase", "stage"]
    },
    "patterns": {
        "measurements": r"\d+(?:\.\d+)?\s*(?:mm|cm|m|km|mg|g|kg)",
        "scientific_notation": r"\d+(?:\.\d+)?[eE][+-]?\d+",
        "citations": r"\[\d+\]|\(\w+\s+et\s+al\.,?\s+\d{4}\)"
    }
}
```

## 🎯 Workflows Recommandés

### 🚀 Développement Initial
1. **Prototype** : RAGAS + Generic evaluation
2. **Debug** : Ajouter TruLens pour observabilité
3. **Test** : Intégrer DeepEval pour tests automatisés

### 🏢 Production Enterprise
1. **Validation** : Évaluation complète avec tous protocoles
2. **Déploiement** : Azure AI Foundry pour monitoring
3. **Maintenance** : TruLens dashboard + DeepEval CI/CD

### 📊 Recherche & Optimisation
1. **Baseline** : RAGAS + embedding analysis
2. **Exploration** : Generic evaluation avec configs domaine
3. **Publication** : Résultats complets pour reproductibilité

## 📚 Tutoriels Détaillés

- 📖 **[TruLens Tutorial](tutorials/trulens_complete_tutorial.md)** - Observabilité temps réel
- 📖 **[DeepEval Tutorial](tutorials/deepeval_complete_tutorial.md)** - Tests unitaires avancés
- 📖 **[Azure AI Foundry Tutorial](tutorials/azure_foundry_complete_tutorial.md)** - Platform enterprise

## 🏆 Meilleures Pratiques

### ✅ À Faire
- Commencer par RAGAS + Generic pour baseline
- Utiliser TruLens pour debug et compréhension
- Intégrer DeepEval dans votre CI/CD
- Azure AI Foundry pour production enterprise
- Adapter les configs de domaine à vos besoins
- Documenter vos seuils de qualité

### ❌ À Éviter
- Se limiter à RAGAS seul
- Négliger les métriques de sécurité (biais, toxicité)
- Pas de versioning des datasets de test
- Ignorer les coûts d'évaluation
- Tests uniquement en fin de développement

## 🔄 Évolution Continue

1. **Mesurer** : Baseline avec protocoles standard
2. **Analyser** : Identifier points faibles avec observabilité
3. **Optimiser** : Ajuster pipelines/modèles
4. **Valider** : Tests automatisés avec seuils qualité
5. **Déployer** : Monitoring continu en production
6. **Répéter** : Cycle d'amélioration continue

---

🎯 **Résultat** : Un système d'évaluation robuste, multi-niveaux, adaptable à tout domaine, bien au-delà de RAGAS !