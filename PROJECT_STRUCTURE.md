# 📁 Structure du Projet RAG Chunk Lab

## 🏗️ Organisation Logique

Le projet a été réorganisé en sections logiques pour améliorer la maintenabilité et la navigation :

```
rag_chunk_lab/
├── 📚 docs/                           # Documentation
│   ├── guides/                        # Guides utilisateur
│   │   ├── EVALUATION_GUIDE.md
│   │   ├── QUICK_REFERENCE.md
│   │   ├── VAGUE_QUERY_OPTIMIZATION.md
│   │   └── TESTING_REPORT.md
│   └── tutorials/                     # Tutoriels détaillés
│       ├── azure_foundry_complete_tutorial.md
│       ├── deepeval_complete_tutorial.md
│       ├── trulens_complete_tutorial.md
│       └── quickstart_evaluation.md
│
├── 🧪 tests/                          # Tous les tests
│   ├── unit/                         # Tests unitaires
│   │   ├── test_vague_query_optimization.py
│   │   └── test_vague_query_optimization_fast.py
│   ├── integration/                  # Tests d'intégration
│   ├── data/                         # Données de test
│   │   ├── documents/               # Documents de test
│   │   └── multi_docs/              # Multi-documents
│   ├── run_tests.py                 # Script de test principal
│   └── [tests existants...]
│
├── 🎯 examples/                       # Exemples et démonstrations
│   └── demos/
│       └── vague_query_optimization_demo.py
│
├── 💾 data/                          # Données et cache
│   └── cache/
│       └── vague_optimization_data/
│
└── 🧠 rag_chunk_lab/                 # Code source principal
    ├── core/                         # Modules de base
    │   ├── chunkers.py              # Stratégies de chunking
    │   ├── indexing.py              # Construction d'index
    │   ├── generation.py            # Génération de réponses
    │   ├── retrieval.py             # Récupération de documents
    │   └── hierarchical_chunking.py # Chunking hiérarchique
    │
    ├── evaluation/                   # Modules d'évaluation
    │   ├── evaluation.py            # Évaluation de base (RAGAS)
    │   ├── embedding_metrics.py     # Métriques d'embeddings
    │   ├── embedding_analysis.py    # Analyse avancée
    │   ├── generic_evaluation.py    # Évaluation générique
    │   ├── legal_evaluation.py      # Évaluation juridique
    │   ├── azure_foundry_evaluation.py # Azure AI Foundry
    │   └── ground_truth_generator.py   # Génération de datasets
    │
    ├── vague_query/                  # Système requêtes vagues
    │   ├── vague_query_optimizer.py           # Détection et expansion
    │   ├── vague_query_optimization_system.py # Orchestrateur principal
    │   ├── adaptive_prompt_engineering.py     # Prompts adaptatifs
    │   ├── context_enrichment_pipeline.py     # Enrichissement contextuel
    │   ├── hybrid_embeddings.py               # Embeddings hybrides
    │   └── metadata_enricher.py               # Enrichissement métadonnées
    │
    ├── utils/                        # Utilitaires
    │   ├── utils.py                 # Fonctions utilitaires
    │   ├── config.py                # Configuration
    │   ├── monitoring.py            # Monitoring de base
    │   ├── production_monitoring.py # Monitoring production
    │   └── embedding_fine_tuning.py # Fine-tuning embeddings
    │
    ├── cli.py                        # Interface en ligne de commande
    └── api.py                        # Interface API REST
```

## 🎯 Avantages de cette Structure

### 📂 **Séparation Logique**
- **`core/`** : Fonctionnalités RAG essentielles
- **`evaluation/`** : Tous les outils d'évaluation et métriques
- **`vague_query/`** : Système d'optimisation pour requêtes ambiguës
- **`utils/`** : Outils transversaux et configuration

### 🧪 **Tests Organisés**
- **`tests/unit/`** : Tests rapides et isolés
- **`tests/integration/`** : Tests complets du système
- **`tests/data/`** : Données de test centralisées

### 📚 **Documentation Structurée**
- **`docs/guides/`** : Guides utilisateur et références
- **`docs/tutorials/`** : Tutoriels étape par étape

### 🎯 **Exemples Accessibles**
- **`examples/demos/`** : Démonstrations fonctionnelles

## 🔄 Imports Mis à Jour

Les imports ont été automatiquement mis à jour pour refléter la nouvelle structure :

### Avant
```python
from .chunkers import fixed_chunks
from .evaluation import load_ground_truth
from .vague_query_optimizer import VagueQueryOptimizer
```

### Après
```python
from .core.chunkers import fixed_chunks
from .evaluation.evaluation import load_ground_truth
from .vague_query.vague_query_optimizer import VagueQueryOptimizer
```

## 🚀 Utilisation

### Import Direct par Module
```python
# Core functionality
from rag_chunk_lab.core import fixed_chunks, build_index
from rag_chunk_lab.evaluation import load_ground_truth, try_ragas_eval
from rag_chunk_lab.vague_query import VagueQueryOptimizer, quick_vague_query_optimization
from rag_chunk_lab.utils import DEFAULTS, load_document
```

### CLI (Inchangé)
```bash
python -m rag_chunk_lab.cli evaluate --doc-id test --ground-truth data.jsonl --optimize-vague-queries
```

### Tests
```bash
# Tests unitaires rapides
python -m pytest tests/unit/ -v

# Tests d'intégration complets
python -m pytest tests/integration/ -v

# Tous les tests
python tests/run_tests.py
```

## 🔧 Rétrocompatibilité

La nouvelle structure maintient la rétrocompatibilité :
- **CLI** : Toutes les commandes fonctionnent comme avant
- **API** : Points d'entrée inchangés
- **Imports publics** : Exposés via `__init__.py`

## 📊 Scripts de Migration

- **`update_imports.py`** : Script de mise à jour automatique des imports
- **Tests automatisés** : Validation de la non-régression

Cette structure améliore significativement l'organisation du code tout en préservant toutes les fonctionnalités existantes.