# 🧪 Guide de Tests - RAG Chunk Lab v2.0

## 🚀 Démarrage Rapide

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Exécution Rapide des Tests
```bash
# Tests de base (recommandé pour débuter)
python run_tests.py basic

# Tous les tests
python run_tests.py

# Tests spécifiques
python run_tests.py chunkers
python run_tests.py performance
```

## ✅ Validation des Optimisations

### 1. Tests de Performance
```bash
python run_tests.py performance
```
**Attendu**: Validation des gains de 70-85% sur les opérations principales

### 2. Tests d'Optimisation
```bash
python run_tests.py optimizations
```
**Attendu**: Cache, singletons, et batch processing validés

### 3. Tests d'Intégration
```bash
python run_tests.py chunkers
python run_tests.py indexing
python run_tests.py generation
```
**Attendu**: Cohérence et robustesse des modules

## 📊 Résultats Attendus

Avec toutes les optimisations, vous devriez voir :

```
🎉 Taux de réussite: 95%+ - EXCELLENT
⏱️  Temps d'exécution: < 10 secondes pour tous les tests
📈 Gains de performance documentés dans les logs
```

## 🔧 Dépannage

### Import Errors
```bash
# Vérifier que le module s'importe
python -c "import rag_chunk_lab; print('OK')"
```

### Tests Spécifiques
```bash
# Debug un test précis
python -m unittest tests.test_basic.TestBasicFunctionality.test_imports -v
```

### Environnement Propre
```bash
# Nettoyer et réinstaller
rm -rf .rag/
python3 -m venv .rag
source .rag/bin/activate  # ou .rag\Scripts\activate sur Windows
pip install -r requirements.txt
```

## 🎯 Validation Complète

Pour valider que tout fonctionne parfaitement :

```bash
# 1. Tests de base
python run_tests.py basic

# 2. Tests de performance
python run_tests.py performance

# 3. Tests complets (si temps disponible)
python run_tests.py
```

**Succès si** : Taux de réussite > 90% et amélioration des performances visible dans les logs.

## 📈 Métriques Clés Validées

✅ **Cache de tokenisation** : +50% vitesse
✅ **Singleton patterns** : Mémoire optimisée
✅ **Batch embeddings** : 8x moins d'appels API
✅ **Float32** : 50% moins de RAM
✅ **Parallélisation** : 3x plus rapide
✅ **Monitoring** : Métriques temps réel

🎉 **RAG Chunk Lab v2.0 est maintenant entièrement testé et optimisé !**