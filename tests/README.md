# 🧪 Tests RAG Chunk Lab v2.0

Suite de tests complète pour valider les optimisations et fonctionnalités de RAG Chunk Lab.

## 📋 Structure des Tests

```
tests/
├── __init__.py                     # Module tests
├── conftest.py                     # Configuration pytest
├── test_optimizations.py           # Tests unitaires des optimisations
├── test_chunkers.py                # Tests du module chunkers
├── test_indexing.py                # Tests du module indexing
├── test_generation.py              # Tests du module generation
├── test_performance.py             # Tests de performance
├── test_integration_complete.py    # Tests d'intégration end-to-end
├── test_runner.py                  # Runner personnalisé avec couleurs
└── README.md                       # Cette documentation
```

## 🚀 Exécution des Tests

### Méthode Simple

```bash
# Tous les tests
python run_tests.py

# Tests par catégorie
python run_tests.py optimizations
python run_tests.py chunkers
python run_tests.py indexing
python run_tests.py generation
python run_tests.py performance
```

### Avec Pytest (si installé)

```bash
# Installation pytest
pip install pytest

# Tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_optimizations.py -v
pytest tests/test_performance.py -v

# Tests avec marqueurs
pytest tests/ -m "not slow" -v
```

### Mode Interactif

```bash
python run_tests.py
# Puis choisir la catégorie dans le menu
```

## 📊 Catégories de Tests

### 1. Tests Unitaires (`test_optimizations.py`)
- **Cache de tokenisation**: Vérifie le LRU cache
- **Singletons**: Azure client, SentenceTransformer
- **Batch processing**: Embeddings Azure par groupe
- **Optimisation mémoire**: Float32 vs Float64
- **Monitoring**: Décorateurs de performance

### 2. Tests d'Intégration Chunkers (`test_chunkers.py`)
- **Fixed chunks**: Chunking de taille fixe optimisé
- **Structure-aware**: Détection de sections/articles
- **Sliding window**: Fenêtres glissantes
- **Semantic chunks**: Pipeline sémantique local
- **Consistency**: Cohérence entre tous les chunkers

### 3. Tests d'Intégration Indexing (`test_indexing.py`)
- **Build index**: Construction TF-IDF et sémantique
- **Semantic indexing**: Embeddings locaux et Azure
- **Retrieval**: Recherche TF-IDF et sémantique
- **Cache LRU**: Performance du cache d'index
- **Error handling**: Gestion des pipelines manquants

### 4. Tests d'Intégration Generation (`test_generation.py`)
- **Extractive answer**: Génération extractive
- **LLM answer**: Génération via Azure OpenAI
- **Azure embeddings**: Embeddings individuels et batch
- **Client caching**: Cache singleton Azure
- **Error handling**: Gestion des erreurs API

### 5. Tests de Performance (`test_performance.py`)
- **Cache effectiveness**: Mesure de l'impact du cache
- **Tokenization speed**: Performance tokenisation
- **Memory optimization**: Validation float32
- **Parallel processing**: Gains parallélisation
- **Integration benchmark**: Workflow complet optimisé

### 6. Tests d'Intégration Complète (`test_integration_complete.py`)
- **End-to-end workflow**: Document → Chunks → Index → Search → Answer
- **Multiple pipelines**: Validation des 3 stratégies
- **Cache in practice**: Efficacité en conditions réelles
- **Memory usage**: Optimisation mémoire globale
- **Error scenarios**: Robustesse du système

## 🎯 Objectifs de Validation

### ✅ Optimisations Validées

1. **Cache de Tokenisation**
   - Amélioration > 50% sur appels répétés
   - Cohérence des résultats

2. **Singleton Patterns**
   - Un seul client Azure par session
   - Un seul modèle SentenceTransformer chargé

3. **Batch Processing**
   - Réduction de 8x des appels API Azure
   - Gestion des gros volumes (100+ textes)

4. **Optimisation Mémoire**
   - 50% de réduction avec float32
   - Précision maintenue

5. **Parallélisation**
   - Gain > 60% sur pipelines multiples
   - Thread safety validée

### 📈 Métriques de Performance

Les tests mesurent automatiquement :
- **Temps d'exécution** par fonction
- **Utilisation mémoire** (pics et deltas)
- **Taux de cache hit**
- **Throughput** des opérations batch
- **Gains de parallélisation**

## 🔧 Configuration et Fixtures

### Fixtures Disponibles
- `sample_pages`: Pages de test standard
- `sample_chunks`: Chunks pré-générés
- `sample_passages`: Passages pour génération
- `temp_data_dir`: Dossier temporaire par session

### Mocking Automatique
- Azure OpenAI API calls
- SentenceTransformer model loading
- File I/O operations
- Network requests

## 📋 Rapports de Test

### Sortie Colorée
- ✅ **VERT**: Tests réussis
- ❌ **ROUGE**: Échecs/erreurs
- ⚠️ **JAUNE**: Tests ignorés
- 📊 **CYAN**: Informations/statistiques

### Métriques Affichées
```
📊 RAPPORT FINAL
================
⏱️  Temps d'exécution: 12.34 secondes
📊 Tests exécutés: 45
✅ Succès: 43
❌ Échecs: 1
⚠️ Erreurs: 0
- Ignorés: 1
🎉 Taux de réussite: 95.6% - EXCELLENT
```

## 🛠️ Développement et Debug

### Ajouter des Tests

1. **Créer le fichier**: `test_nouveau_module.py`
2. **Hériter de**: `unittest.TestCase`
3. **Nommer**: Méthodes commençant par `test_`
4. **Utiliser**: Fixtures de `conftest.py`

### Debug des Échecs

```bash
# Mode verbose pour plus de détails
python run_tests.py -v

# Test spécifique avec traceback complet
python -m unittest tests.test_performance.TestPerformanceOptimizations.test_cache_effectiveness -v
```

### Profiling Performance

```python
from rag_chunk_lab.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

@monitor.monitor_function("my_test")
def my_test_function():
    # Code à tester
    pass

# Métriques automatiquement collectées
```

## ✨ Bonnes Pratiques

1. **Tests Isolés**: Chaque test nettoie après lui
2. **Mocking**: APIs externes toujours mockées
3. **Données Déterministes**: Résultats reproductibles
4. **Performance**: Validation des optimisations mesurée
5. **Coverage**: Tous les chemins critiques testés

## 🎉 Résultats Attendus

Avec toutes les optimisations, les tests devraient montrer :
- **95%+ de taux de réussite**
- **Performance 3-8x meilleure** selon la fonction
- **Utilisation mémoire optimisée**
- **Robustesse** face aux erreurs
- **Cohérence** des résultats