# Contributing to RAG Chunk Lab

Merci de votre intérêt pour contribuer à RAG Chunk Lab ! Ce guide vous aidera à démarrer.

## 🚀 Comment contribuer

### Signaler un bug
1. Vérifiez que le bug n'a pas déjà été signalé dans les [Issues](https://github.com/your-username/rag-chunk-lab/issues)
2. Créez un nouvelle issue avec le template "Bug Report"
3. Incluez des informations détaillées :
   - Version de Python et du système d'exploitation
   - Version de RAG Chunk Lab
   - Commande exacte qui a causé l'erreur
   - Message d'erreur complet
   - Étapes pour reproduire le problème

### Proposer une fonctionnalité
1. Consultez les [Issues](https://github.com/your-username/rag-chunk-lab/issues) et la [Roadmap](CHANGELOG.md#à-venir---roadmap)
2. Créez une issue avec le template "Feature Request"
3. Décrivez clairement :
   - Le problème que ça résout
   - La solution proposée
   - Des exemples d'utilisation
   - L'impact sur les utilisateurs existants

## 🛠️ Développement Local

### Configuration initiale
```bash
# Clone du repository
git clone https://github.com/your-username/rag-chunk-lab
cd rag-chunk-lab

# Configuration environnement
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Installation en mode développement
pip install -e ".[dev]"

# Configuration pre-commit (optionnel)
pre-commit install
```

### Structure du code
```
rag_chunk_lab/
├── chunkers.py        # Stratégies de chunking
├── indexing.py        # Indexation TF-IDF
├── retrieval.py       # Recherche de candidats
├── generation.py      # Génération de réponses
├── evaluation.py      # Évaluation RAGAS
├── ground_truth_generator.py  # Génération automatique
├── cli.py            # Interface ligne de commande
├── api.py            # API REST
├── config.py         # Configuration
└── utils.py          # Utilitaires
```

### Tests
```bash
# Lancer tous les tests
pytest

# Tests avec coverage
pytest --cov=rag_chunk_lab

# Tests spécifiques
pytest tests/test_chunkers.py -v
```

### Code Style
```bash
# Formatage automatique
black rag_chunk_lab/
isort rag_chunk_lab/

# Vérification du style
flake8 rag_chunk_lab/
mypy rag_chunk_lab/
```

## 📝 Conventions

### Messages de commit
Utilisez les conventions [Conventional Commits](https://conventionalcommits.org/):

```bash
feat: add semantic chunking strategy
fix: resolve RAGAS evaluation timeout
docs: update installation instructions
refactor: simplify ground truth generation
test: add comprehensive chunking tests
```

### Code Python
- **Formatage** : Black (ligne 88 caractères)
- **Import** : isort avec profil Black
- **Type hints** : Optionnels mais recommandés pour les nouvelles fonctions
- **Docstrings** : Style Google pour les fonctions publiques
- **Variables** : snake_case en anglais
- **Constantes** : UPPER_CASE

### Documentation
- **README** : Mode tutoriel, exemples concrets
- **Docstrings** : Explicatives, avec exemples si complexe
- **Comments** : Expliquer le "pourquoi", pas le "comment"
- **CHANGELOG** : Suivre [Keep a Changelog](https://keepachangelog.com/)

## 🔄 Workflow de développement

### Pour une nouvelle fonctionnalité
1. **Fork** le repository
2. **Créer une branche** : `git checkout -b feat/nouvelle-fonctionnalite`
3. **Développer** avec tests
4. **Tester** : `pytest` et vérifications style
5. **Commit** : Messages conventionnels
6. **Push** : `git push origin feat/nouvelle-fonctionnalite`
7. **Pull Request** : Description claire, screenshots si UI

### Pour un bugfix
1. **Créer une branche** : `git checkout -b fix/nom-du-bug`
2. **Reproduire** le bug avec un test
3. **Corriger** le problème
4. **Vérifier** que le test passe
5. **Pull Request** : Référencer l'issue du bug

## 🧪 Types de contributions appréciées

### Code
- **Nouvelles stratégies de chunking** (sémantique, par domaine...)
- **Support nouveaux formats** (DOCX, PPTX, HTML...)
- **Optimisations performance** (cache, parallélisation...)
- **Métriques d'évaluation custom**
- **Intégrations LLM** (OpenAI direct, Anthropic, Cohere...)

### Documentation
- **Tutoriels spécialisés** (juridique, médical, technique...)
- **Exemples pratiques** avec datasets réels
- **Traductions** (anglais, autres langues)
- **Guides d'optimisation** par cas d'usage
- **FAQ** avec problèmes courants

### Tests et Qualité
- **Tests edge cases** pour robustesse
- **Tests de performance** avec benchmarks
- **Tests d'intégration** avec vrais LLMs
- **Amélioration coverage** de code

## 🏆 Reconnaissance

Les contributeurs sont reconnus dans :
- **CHANGELOG.md** pour chaque contribution
- **README.md** section contributeurs (à venir)
- **GitHub Releases** notes importantes

## ❓ Questions

- **Discussions** : Utilisez [GitHub Discussions](https://github.com/your-username/rag-chunk-lab/discussions)
- **Issues** : Pour bugs et demandes de fonctionnalités
- **Email** : steve.moses@example.com pour questions privées

Merci de contribuer à améliorer RAG Chunk Lab ! 🚀