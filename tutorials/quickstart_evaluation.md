# 🚀 Guide Rapide - Évaluation Multi-Protocoles

## 🎯 Installation Rapide

```bash
# Protocoles de base (inclus)
cd rag_chunk_lab
pip install -r requirements.txt

# TruLens (observabilité)
pip install trulens-eval

# DeepEval (tests unitaires)
pip install deepeval

# Azure AI Foundry (enterprise)
pip install azure-ai-ml azure-identity
```

## 🔧 Configuration Minimale

```bash
# Variables d'environnement
export OPENAI_API_KEY="your-openai-key"        # Pour TruLens/DeepEval
export AZURE_OPENAI_API_KEY="your-azure-key"   # Pour RAGAS
export AZURE_OPENAI_ENDPOINT="your-endpoint"   # Pour RAGAS

# Optionnel : Azure AI Foundry
export AZURE_SUBSCRIPTION_ID="your-subscription"
export AZURE_RESOURCE_GROUP="your-rg"
export AZURE_ML_WORKSPACE="your-workspace"
```

## 🎯 Évaluations par Niveau

### Niveau 1: Standard (Recommandé)
```bash
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --ragas \
  --generic-evaluation \
  --embedding-analysis \
  --use-llm
```

### Niveau 2: Avec Observabilité
```bash
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --ragas \
  --generic-evaluation \
  --trulens \
  --use-llm
```

### Niveau 3: Tests Automatisés
```bash
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --deepeval \
  --use-llm
```

### Niveau 4: Enterprise
```bash
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --azure-foundry \
  --use-llm
```

### 🌟 Grand Chelem (Tous)
```bash
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

## 📊 Résultats Obtenus

```
exports/ma_collection/
├── ragas_summary.csv                 # RAGAS 4 métriques
├── generic_evaluation_*.json         # 6 métriques génériques
├── embedding_analysis_*.json         # Métriques embeddings

trulens_results/
├── dashboard accessible sur localhost:8501

deepeval_results/
├── Tests unitaires avec seuils

azure_foundry/
├── Jobs enterprise + monitoring
```

## 🎯 Métriques par Protocole

| Protocole | Métriques Principales | Temps | Coût |
|-----------|----------------------|-------|------|
| **RAGAS** | answer_relevancy, faithfulness, context_precision, context_recall | 2-5 min | OpenAI API |
| **Generic** | factual_accuracy, completeness, relevance, consistency, clarity | 1-2 min | Gratuit |
| **Embeddings** | recall@k, MRR, NDCG, diversité, cohérence | 1-3 min | Gratuit |
| **TruLens** | groundedness, relevance + dashboard | 3-7 min | OpenAI API |
| **DeepEval** | Tests + sécurité (biais, toxicité) | 5-10 min | OpenAI API |
| **Azure** | Custom flows + monitoring | Variable | Azure costs |

## 🔧 Configuration par Domaine

### Documents Techniques
```python
# Dans votre script
domain_config = {
    "keywords": {
        "technical": ["API", "configuration", "function"],
        "process": ["step", "procedure", "workflow"]
    }
}
```

### Documents Business
```python
domain_config = {
    "keywords": {
        "financial": ["budget", "revenue", "cost"],
        "metrics": ["KPI", "ROI", "target"]
    }
}
```

## 🏆 Recommandations Rapides

### 🚀 Développement
1. Commencer par RAGAS + Generic
2. Ajouter TruLens pour debug
3. Intégrer DeepEval pour CI/CD

### 🏢 Production
1. Évaluation complète pour validation
2. Azure AI Foundry pour monitoring
3. Seuils de qualité automatisés

### 📊 Analyse
1. Generic evaluation pour insights généraux
2. Embedding analysis pour optimiser récupération
3. TruLens dashboard pour compréhension détaillée

## 🔍 Dépannage Rapide

```bash
# Vérifier les dépendances
python3 -c "import ragas; print('✅ RAGAS OK')"
python3 -c "import trulens_eval; print('✅ TruLens OK')"
python3 -c "import deepeval; print('✅ DeepEval OK')"

# Tester configuration Azure
az ml workspace show --name your-workspace

# Dashboard TruLens bloqué ?
killall streamlit
```

## 📚 Aller Plus Loin

- 📖 [Guide Complet](../EVALUATION_GUIDE.md)
- 🔍 [TruLens Tutorial](trulens_complete_tutorial.md)
- 🧪 [DeepEval Tutorial](deepeval_complete_tutorial.md)
- 🌟 [Azure AI Foundry Tutorial](azure_foundry_complete_tutorial.md)

---

🎯 **En 5 minutes**: Évaluation robuste multi-protocoles pour tous domaines !