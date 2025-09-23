# 🚀 Guide de Référence Rapide - Optimisation Requêtes Vagues

## 🎯 Commandes Essentielles

### ⚡ Évaluation avec Optimisation Vague

```bash
# Standard - Tous domaines
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --optimize-vague-queries \
  --generic-evaluation \
  --use-llm

# Juridique spécialisé
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --optimize-vague-queries \
  --legal-evaluation \
  --use-llm

# Complet avec monitoring
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth dataset.jsonl \
  --optimize-vague-queries \
  --ragas \
  --generic-evaluation \
  --trulens \
  --deepeval \
  --azure-foundry \
  --use-llm
```

### 🧪 Test Rapide du Système

```bash
# Démo complète
cd rag_chunk_lab
python examples/vague_query_optimization_demo.py

# Test API simple
python -c "
from rag_chunk_lab.vague_query_optimization_system import quick_vague_query_optimization
result = quick_vague_query_optimization(
    query='Comment ça marche ?',
    documents=[{'doc_id': 'test', 'text': 'Votre contenu...'}],
    domain='legal'
)
print(f'Vague: {result[\"is_vague\"]} (score: {result[\"vagueness_score\"]:.2f})')
"
```

## 🔧 Configuration par Domaine

### 🏛️ Juridique
```bash
export DOMAIN="legal"
export OPENAI_API_KEY="votre-clé"

python3 -m rag_chunk_lab.cli evaluate \
  --doc-id contrats_civils \
  --ground-truth questions_juridiques.jsonl \
  --optimize-vague-queries \
  --legal-evaluation \
  --use-llm
```

### 💻 Technique
```bash
export DOMAIN="technical"

python3 -m rag_chunk_lab.cli evaluate \
  --doc-id documentation_api \
  --ground-truth questions_tech.jsonl \
  --optimize-vague-queries \
  --generic-evaluation \
  --embedding-analysis \
  --use-llm
```

### 🏥 Médical
```bash
export DOMAIN="medical"

python3 -m rag_chunk_lab.cli evaluate \
  --doc-id protocoles_medicaux \
  --ground-truth questions_sante.jsonl \
  --optimize-vague-queries \
  --generic-evaluation \
  --trulens \
  --use-llm
```

## 📊 Résultats Générés

### 📁 Structure des Exports

```
exports/ma_collection/
├── vague_optimization_analysis.json    # Analyse des requêtes vagues
├── ragas_summary.csv                    # Métriques RAGAS
├── generic_evaluation_*.json            # Évaluation générique
├── embedding_analysis_*.json            # Analyse embeddings
├── hierarchical_chunks_stats.json       # Stats chunking
└── context_enrichment_report.json       # Rapport enrichissement

trulens_results/
├── dashboard accessible sur localhost:8501

deepeval_results/
├── test_reports_*.json                  # Rapports tests unitaires

azure_foundry/
├── evaluation_jobs_*.json               # Jobs Azure ML
```

### 📈 Métriques Clés

| Métrique | Description | Objectif |
|----------|-------------|----------|
| **Vague Detection Rate** | % requêtes vagues détectées | > 95% |
| **Expansion Quality** | Qualité des reformulations | > 0.8 |
| **Context Enrichment Score** | Richesse du contexte | > 0.7 |
| **User Satisfaction** | Satisfaction moyenne | > 4.0/5 |
| **Response Time** | Temps de réponse moyen | < 3s |

## 🎛️ Options Avancées

### 🔍 Paramètres d'Optimisation

```bash
# Niveau utilisateur
--user-level beginner|intermediate|advanced

# Seuils personnalisés
--vague-threshold 0.5
--context-quality-threshold 0.7
--response-time-limit 5.0

# Modes d'enrichissement
--enable-definitions
--enable-examples
--enable-analogies
--max-expansions 5
```

### 📊 Monitoring et Feedback

```bash
# Activation monitoring
--enable-monitoring

# Collecte feedback
--collect-feedback

# Alertes personnalisées
--alert-thresholds '{"response_time": 3.0, "relevance": 3.5}'

# Export monitoring
--export-monitoring-data monitoring_export.json
```

## 🚨 Dépannage Rapide

### ❌ Erreurs Communes

**Problem:** `ImportError: No module named 'sentence_transformers'`
```bash
pip install sentence-transformers torch
```

**Problem:** `SpaCy model not found`
```bash
python -m spacy download fr_core_news_sm
```

**Problem:** `OpenAI API key not configured`
```bash
export OPENAI_API_KEY="votre-clé"
# ou ajoutez dans votre .env
```

**Problem:** `Vague query optimization not working`
```bash
# Vérifiez les dépendances
python -c "import spacy, sentence_transformers, openai; print('✅ All dependencies OK')"

# Test isolé
python -c "
from rag_chunk_lab.vague_query_optimizer import VagueQueryOptimizer
opt = VagueQueryOptimizer()
print(opt.is_vague_query('Comment ça marche ?'))
"
```

### 🔧 Validation Système

```bash
# Test complet des composants
python -c "
from rag_chunk_lab.vague_query_optimization_system import create_vague_optimization_system
system = create_vague_optimization_system(domain='general')
status = system.get_system_status()
print('Components status:')
for comp, ok in status['components_status'].items():
    print(f'  {comp}: {\"✅\" if ok else \"❌\"}')
"

# Test performance
python -c "
import time
from rag_chunk_lab.vague_query_optimizer import VagueQueryOptimizer
opt = VagueQueryOptimizer()
start = time.time()
result = opt.is_vague_query('Test de performance')
print(f'Detection time: {time.time() - start:.3f}s')
"
```

## 📚 Liens Utiles

- 📖 [Documentation Complète](VAGUE_QUERY_OPTIMIZATION.md)
- 🔍 [TruLens Tutorial](tutorials/trulens_complete_tutorial.md)
- 🧪 [DeepEval Tutorial](tutorials/deepeval_complete_tutorial.md)
- 🌟 [Azure AI Foundry Tutorial](tutorials/azure_foundry_complete_tutorial.md)
- 🚀 [Guide de Démarrage Rapide](tutorials/quickstart_evaluation.md)

## 💡 Tips & Astuces

### 🎯 Optimisation Performance

1. **Cache aktivé** : Les résultats sont mis en cache automatiquement
2. **Batch processing** : Traitez plusieurs questions ensemble
3. **Seuils adaptatifs** : Ajustez selon votre domaine
4. **Monitoring actif** : Surveillez les métriques en temps réel

### 🧠 Amélioration Continue

1. **Collectez du feedback** : `--collect-feedback`
2. **Analysez les patterns** : Consultez les rapports d'usage
3. **Ajustez les seuils** : Optimisez selon vos métriques
4. **Fine-tuning** : Spécialisez pour votre domaine

### 🎨 Personnalisation

```python
# Configuration avancée
from rag_chunk_lab.vague_query_optimization_system import create_vague_optimization_system

system = create_vague_optimization_system(
    domain="legal",
    openai_api_key="votre-clé",
    config_overrides={
        "max_definitions": 10,
        "enable_analogies": True,
        "context_quality_threshold": 0.8,
        "response_time_warning": 2.0
    }
)
```