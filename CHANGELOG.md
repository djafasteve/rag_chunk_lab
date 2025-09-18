# Changelog

Toutes les modifications importantes de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-18

### 🎉 Première version majeure

#### Ajouté
- **Système de chunking multi-stratégies** : Fixed, Structure-aware, Sliding-window
- **Interface CLI complète** avec commandes `ingest`, `ask`, `evaluate`
- **API REST FastAPI** pour intégration dans applications
- **Générateur automatique de ground truth** avec support Ollama et Azure OpenAI
- **Évaluation RAGAS** avec 4 métriques expertes (Answer Relevancy, Faithfulness, Context Precision, Context Recall)
- **Interface visuelle avec barres de progression** pour suivre l'avancement
- **Export CSV automatique** pour analyse dans Excel
- **Support documents multiples** : PDF, TXT, MD
- **Métadonnées complètes** : page, section, source, chunk_id
- **Configuration flexible** via variables d'environnement
- **Documentation tutoriel complète** avec exemples pratiques

#### Fonctionnalités Core
- **Chunking intelligent** respectant la structure des documents juridiques
- **Indexation TF-IDF** avec boost des métadonnées
- **Retrieval hybride** avec scores de similarité
- **Génération extractive** et via LLM (Azure OpenAI)
- **Évaluation comparative** automatique des stratégies

#### Support LLM
- **Azure OpenAI** : GPT-4o-mini, text-embedding-3-small
- **Ollama local** : Mistral 7B, Llama3 8B, et autres modèles
- **Configuration automatique** des backends RAGAS

#### Exports et Analyse
- **Sources détaillées** avec page/section/snippet pour chaque réponse
- **Métriques proxy locales** (similarité Jaccard)
- **Métriques RAGAS expertes** pour évaluation professionnelle
- **Formats CSV optimisés** pour tableaux croisés dynamiques Excel

#### Interface et UX
- **Barres de progression** avec émojis et statistiques temps réel
- **Messages d'erreur clairs** avec suggestions de résolution
- **Validation automatique** des prérequis et configuration
- **Filtrage robuste** des données problématiques

### Technique
- **Architecture modulaire** : chunkers, indexing, retrieval, generation, evaluation
- **Gestion d'erreurs robuste** avec récupération automatique
- **Cache intelligent** pour optimiser les performances
- **Support multi-plateforme** : Linux, macOS, Windows
- **Dépendances optimisées** avec versions fixées

### Documentation
- **README tutoriel** avec approche step-by-step
- **Guide de dépannage** complet
- **Exemples concrets** pour chaque cas d'usage
- **Conseils d'optimisation** par type de document
- **Structure projet claire** avec rôle de chaque module

---

## [À venir] - Roadmap

### Version 1.1.0
- [ ] Support de nouveaux formats : DOCX, PPTX
- [ ] Chunking sémantique avec embeddings
- [ ] Interface web pour configuration visuelle
- [ ] Templates de prompts personnalisables
- [ ] Cache distribué pour déploiements multi-instances

### Version 1.2.0
- [ ] Support OpenAI direct (non-Azure)
- [ ] Métriques custom définies par l'utilisateur
- [ ] Intégration Weights & Biases pour tracking
- [ ] API de streaming pour réponses temps réel
- [ ] Dashboard analytics intégré

### Version 2.0.0
- [ ] Chunking adaptatif basé sur le contenu
- [ ] Support multi-langues avec détection automatique
- [ ] Système de plugins pour stratégies custom
- [ ] Interface graphique complète
- [ ] Mode cluster pour gros volumes

---

## Convention des Commits

Ce projet utilise les conventions suivantes pour les messages de commit :

- `feat:` Nouvelle fonctionnalité
- `fix:` Correction de bug
- `docs:` Mise à jour documentation
- `style:` Changements de formatage
- `refactor:` Refactoring de code
- `perf:` Amélioration performance
- `test:` Ajout/modification tests
- `chore:` Maintenance technique

Exemple : `feat: add automatic ground truth generation with Ollama support`