# RAG Chunk Lab — Laboratoire de Chunking pour RAG

Un outil complet pour **tester et comparer différentes stratégies de chunking** dans vos pipelines RAG. Idéal pour optimiser la performance sur des documents juridiques, techniques ou réglementaires.

## 🎯 Qu'est-ce que ça fait ?

**RAG Chunk Lab** teste automatiquement **3 stratégies de chunking** sur vos documents :

1. **📄 Fixed Chunks** → Découpage fixe (500 tokens + overlap 80)
2. **🗂️ Structure-Aware** → Respecte les titres et sections (Article, Chapitre...)
3. **🔄 Sliding Window** → Fenêtres glissantes (400 tokens, stride 200)

Pour chaque question, vous obtenez :
- ✅ Une réponse par stratégie (extractive ou LLM)
- 📍 Les sources exactes (page, section, snippet)
- 📊 Export CSV pour analyse dans Excel
- 🤖 Évaluation automatique avec métriques RAGAS

---

## 🚀 Installation et Premier Test

### Étape 1 : Environnement
```bash
git clone <votre-repo>
cd rag_chunk_lab
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pour le pipeline sémantique local (optionnel mais recommandé)
pip install sentence-transformers

# Pour Azure OpenAI (embeddings cloud de qualité professionnelle)
pip install openai
export AZURE_OPENAI_API_KEY="votre-clé"
export AZURE_OPENAI_ENDPOINT="https://votre-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="text-embedding-ada-002"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### Étape 2 : Ingérer vos documents

#### Option A : Un seul document
```bash
# Ingérer un document PDF/TXT/MD
python3 -m rag_chunk_lab.cli ingest --doc mon_document.pdf --doc-id test
```

#### Option B : Un dossier complet (🆕 Recommandé)
```bash
# Ingérer tous les documents d'un dossier sous un seul doc-id
python3 -m rag_chunk_lab.cli ingest --doc mes_documents/ --doc-id ma_collection

# Support automatique : .pdf, .txt, .md
# Chaque document garde son nom de fichier source dans les métadonnées
```

### Étape 3 : Interroger votre collection

#### Option A : Analyse comparative des 5 stratégies
```bash
# Voir les 5 réponses (fixed, structure, sliding, semantic, azure_semantic) pour analyser
python3 -m rag_chunk_lab.cli ask --doc-id ma_collection --question "Quel est le délai de prescription ?"

# Désactiver les pipelines sémantiques si besoin
python3 -m rag_chunk_lab.cli ask --doc-id ma_collection --question "..." --no-semantic --no-azure-semantic
```

#### Option B : Chat IA avec réponse synthétisée (🆕 Recommandé)
```bash
# Obtenir une réponse claire et contextuelle générée par l'IA (utilise Azure semantic par défaut)
python3 -m rag_chunk_lab.cli chat --doc-id ma_collection --question "Quel est le délai de prescription ?"

# Avec modèle spécialisé pour l'expertise juridique
python3 -m rag_chunk_lab.cli chat \
  --doc-id ma_collection \
  --question "Quelles sont les sanctions en cas de récidive ?" \
  --pipeline azure_semantic \
  --provider ollama \
  --model votre-modele-juridique \
  --top-k 5

# Ou avec le modèle par défaut
python3 -m rag_chunk_lab.cli chat \
  --doc-id ma_collection \
  --question "Quelles sont les sanctions en cas de récidive ?" \
  --pipeline azure_semantic
```

**🧠 Nouveauté : Pipelines Sémantiques**

**🔹 Semantic (Local)** :
- 🔍 **Comprend le sens** : Trouve "sanctions" même quand le texte dit "peines"
- 🏠 **Local** : Modèle français `dangvantuan/sentence-camembert-large`
- 🆓 **Gratuit** : Pas de coût API

**☁️ Azure Semantic (Cloud)** :
- 🎯 **Qualité professionnelle** : Embeddings Azure OpenAI de dernière génération
- 📚 **Optimisé juridique** : Excellente compréhension des textes légaux
- 🌐 **Multilingue** : Meilleure gestion français/anglais
- ⚡ **Pas de modèle lourd** : Traitement dans le cloud

**Avantages du mode chat :**
- 🎯 **Réponse synthétisée** : L'IA combine et résume les sources pertinentes
- 📚 **Citations des sources** : Références aux documents et pages consultés
- 🔍 **Contextuel** : Utilise uniquement les informations de vos documents
- ⚡ **Prêt à l'emploi** : Fonctionne avec Ollama (local) ou Azure OpenAI

**Résultat** : 3 réponses comparées + fichier `exports/test/sources_<timestamp>.csv`

| Commande   | ask                       | chat (🆕)              |
  |------------|---------------------------|------------------------|
  | Sortie     | JSON brut des 3 pipelines | Réponse IA synthétisée |
  | Usage      | Analyse comparative       | Conversation naturelle |
  | Sources    | Chunks séparés            | Citations intégrées    |
  | Lisibilité | Technique                 | Grand public           |

  🤖 Fonctionnalités clés

  - ✅ Réponse synthétisée : L'IA combine plusieurs sources et résume
  - ✅ Citations automatiques : Références aux documents et pages
  - ✅ Contextuel : Utilise uniquement vos documents (pas d'hallucination)
  - ✅ Configurable : Choix du pipeline, provider LLM, et nombre de sources
  - ✅ Fallback robuste : Affiche les chunks même si l'IA échoue
  - ✅ Support multimodal : Ollama (local/gratuit) et Azure OpenAI

---

## 🧠 Comprendre les 5 Stratégies de Recherche

### Pourquoi 5 approches différentes ?

Chaque méthode a ses forces selon le type de documents et de questions :

#### 1. **Fixed** (Chunks de taille fixe) ⚖️
- **Principe :** Découpe le texte en morceaux de taille régulière
- **Idéal pour :** Documents homogènes, recherches factuelles précises
- **Exemple :** "Quel est l'article 123 ?" dans un code juridique

#### 2. **Structure** (Conscient de la structure) 🏗️
- **Principe :** Respecte les titres, sections, paragraphes
- **Idéal pour :** Documents bien structurés, recherches par section
- **Exemple :** "Que dit le chapitre sur les contrats ?" dans un manuel

#### 3. **Sliding** (Fenêtre glissante) 🔄
- **Principe :** Fenêtres qui se chevauchent pour capturer les transitions
- **Idéal pour :** Concepts qui s'étendent sur plusieurs paragraphes
- **Exemple :** "Comment fonctionne le processus de validation ?" (description longue)

#### 4. **Semantic** (Sémantique Local) 🧠 **← Nouveauté !**
- **Principe :** Comprend le **sens** des mots avec un modèle IA local
- **Idéal pour :** Questions en langage naturel, usage gratuit
- **Modèle :** `dangvantuan/sentence-camembert-large` (français)

#### 5. **Azure Semantic** (Sémantique Cloud) ☁️ **← Premium !**
- **Principe :** Comprend le **sens** avec Azure OpenAI embeddings
- **Idéal pour :** Documents juridiques, qualité maximale
- **Exemples magiques (communs aux 2 sémantiques) :**
  - Question: "sanctions" → Trouve: "peines", "condamnations", "punitions"
  - Question: "délai" → Trouve: "durée", "terme", "période"
  - Question: "interdit" → Trouve: "prohibé", "défendu", "illégal"

### 🎯 Conseil Pratique

```bash
# 1. Commencez par tester les 5 approches
python3 -m rag_chunk_lab.cli ask --doc-id votre_doc --question "votre question"

# 2. Pour l'usage quotidien, privilégiez Azure semantic (si configuré)
python3 -m rag_chunk_lab.cli chat --doc-id votre_doc --question "votre question" --pipeline azure_semantic

# 3. Sinon, utilisez le sémantique local
python3 -m rag_chunk_lab.cli chat --doc-id votre_doc --question "votre question" --pipeline semantic

# 4. Avec un modèle spécialisé pour votre domaine d'expertise
python3 -m rag_chunk_lab.cli chat \
  --doc-id votre_doc \
  --question "votre question" \
  --model votre-modele-specialise
```

### 🤖 Modèles LLM Recommandés

#### **Pour Documents Juridiques :**
```bash
# Modèle par défaut (généraliste)
--model mistral:7b

# Modèles spécialisés juridiques (si disponibles dans votre Ollama)
--model llama3:8b  # Meilleure compréhension contextuelle
--model "hf.co/MaziyarPanahi/calme-2.3-legalkit-8b-GGUF:Q8_0"  # Modèle juridique français spécialisé
--model llama3.2:latest  # Compact et efficace
--model codellama:13b  # Si documents contiennent du code/réglementation
```

#### **Pour Documents Techniques :**
```bash
--model codellama:7b  # Spécialisé code et documentation technique
--model llama3:8b     # Bon compromis qualité/vitesse
```

#### **Configuration Permanente :**
Pour éviter de répéter `--model` à chaque fois, modifiez dans `config.py` :
```python
# Pour usage juridique quotidien
DEFAULTS.default_model = "hf.co/MaziyarPanahi/calme-2.3-legalkit-8b-GGUF:Q8_0"

# Ou pour usage généraliste rapide
DEFAULTS.default_model = "llama3.2:latest"
```

#### **Exemples Pratiques avec Modèle Juridique :**
```bash
# Question juridique avec modèle spécialisé
python3 -m rag_chunk_lab.cli chat \
  --doc-id codes_juridiques \
  --question "Quelles sont les conditions de la légitime défense ?" \
  --pipeline azure_semantic \
  --model "hf.co/MaziyarPanahi/calme-2.3-legalkit-8b-GGUF:Q8_0"

# Comparaison rapide avec modèle général
python3 -m rag_chunk_lab.cli chat \
  --doc-id codes_juridiques \
  --question "Quelles sont les conditions de la légitime défense ?" \
  --pipeline azure_semantic \
  --model mistral:7b
```

---

## 🤖 Créer un Dataset de Test Automatiquement

### Pourquoi ?
Au lieu de créer manuellement des questions/réponses, **générez automatiquement un dataset d'expert** à partir de vos documents !

### Comment faire ?

#### Option A : Avec Ollama (Local, Gratuit)
```bash
# 1. Installer et démarrer Ollama
ollama serve
ollama pull mistral:7b

# 2. Générer 10 questions expertes par document
python3 generate_ground_truth.py --folder documents/mes_docs --questions-per-doc 10
```

#### Option B : Avec Azure OpenAI (Plus performant)
```bash
# 1. Configurer Azure OpenAI
export AZURE_OPENAI_API_KEY="votre-clé"
export AZURE_OPENAI_ENDPOINT="https://votre-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# 2. Générer le dataset
python3 generate_ground_truth.py --folder documents/mes_docs --llm-provider azure
```

### Résultat
Fichier `mes_docs_ground_truth.jsonl` avec format :
```json
{
  "question": "Quel est le délai de prescription pour les contraventions de 5ème classe ?",
  "answer": "Le délai de prescription de l'action publique des contraventions de la cinquième classe est de trois ans révolus...",
  "source_document": "code_penal.pdf",
  "page": 15,
  "doc_section": "TITRE PREMIER",
  "generated_by": "ollama:mistral:7b"
}
```

---

## 📊 Évaluation et Comparaison des Stratégies

### Évaluation Automatique avec RAGAS

Une fois votre dataset créé, comparez les 3 stratégies de chunking :

```bash
# Évaluer les 3 pipelines avec métriques d'expert
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id test \
  --ground-truth mes_docs_ground_truth.jsonl \
  --ragas \
  --use-llm
```

**Pendant l'exécution, vous verrez :**
```
🔄 Collecting answers from 3 pipelines for 10 questions...
📊 Processing pipelines: 100%|████████| 3/3 [01:30<00:00]
✅ Answer collection completed!

🎯 Starting RAGAS evaluation...
🔄 Starting RAGAS evaluation for 3 pipelines with 10 questions...
📊 Metrics: answer_relevancy, faithfulness, context_precision, context_recall

🔍 Evaluating fixed pipeline...
  📝 Running RAGAS metrics for 'fixed' pipeline...
  ✅ fixed: answer_relevancy: 0.847, faithfulness: 0.923

✨ RAGAS evaluation completed!
📋 Summary:
  fixed: avg=0.856
  structure: avg=0.891  ← Meilleur !
  sliding: avg=0.834

💾 Exporting results...
```

### Fichiers d'Analyse Générés

Après évaluation, vous trouvez dans `exports/test/` :

1. **`ragas_summary.csv`** → Tableau de bord des moyennes
   ```
   pipeline,answer_relevancy,faithfulness,context_precision,context_recall
   fixed,0.847,0.923,0.756,0.834
   structure,0.891,0.945,0.812,0.867  ← La structure-aware gagne !
   sliding,0.834,0.898,0.743,0.801
   ```

2. **`ragas_per_question.csv`** → Détail par question
   - Parfait pour identifier les questions problématiques
   - Idéal pour tableaux croisés dynamiques Excel

### Analyse dans Excel

1. **Ouvrir `ragas_summary.csv`** → Vue d'ensemble rapide
2. **Ouvrir `ragas_per_question.csv`** → Analyser les détails
3. **Créer un graphique radar** comparant les 4 métriques par pipeline
4. **Identifier** quelle stratégie fonctionne le mieux sur votre type de documents

---

## 💡 Exemples Pratiques

### Cas d'Usage Typiques

#### 📚 Collection de Documentation Technique
```bash
# Dossier avec manuels PDF, guides TXT, et docs Markdown
python3 -m rag_chunk_lab.cli ingest --doc documentation_produit/ --doc-id docs_techniques

# Questions: "Comment configurer SSL?", "Quels sont les prérequis?"
python3 -m rag_chunk_lab.cli chat --doc-id docs_techniques --question "Comment configurer SSL?" --pipeline semantic
```

#### ⚖️ Corpus Juridique
```bash
# Dossier avec codes, jurisprudences, circulaires
python3 -m rag_chunk_lab.cli ingest --doc corpus_juridique/ --doc-id droit_penal

# Questions: "Quel est le délai de prescription?", "Quelles sont les circonstances aggravantes?"
python3 -m rag_chunk_lab.cli chat --doc-id droit_penal --question "Quelles sont les circonstances aggravantes?" --pipeline semantic
```

#### 🏢 Base de Connaissances Entreprise
```bash
# Procédures, politiques, manuels RH
python3 -m rag_chunk_lab.cli ingest --doc knowledge_base/ --doc-id entreprise

# Questions: "Quelle est la politique de télétravail?", "Comment demander un congé?"
python3 -m rag_chunk_lab.cli chat --doc-id entreprise --question "Quelle est la politique de télétravail?" --pipeline semantic
```

### Workflow Complet Recommandé

```bash
# 1. Ingérer votre collection de documents
python3 -m rag_chunk_lab.cli ingest --doc mes_documents/ --doc-id ma_collection

# 2. Générer automatiquement un dataset de test
python3 generate_ground_truth.py --folder mes_documents --questions-per-doc 5

# 3. Évaluer et comparer les 3 stratégies
python3 -m rag_chunk_lab.cli evaluate \
  --doc-id ma_collection \
  --ground-truth mes_documents_ground_truth.jsonl \
  --ragas --use-llm

# 4. Analyser les résultats dans Excel
# Ouvrir exports/ma_collection/ragas_summary.csv

# 5. Utiliser la stratégie sémantique pour un usage quotidien optimal
python3 -m rag_chunk_lab.cli chat --doc-id ma_collection --question "Votre question" --pipeline semantic
```

---

## 🎮 Mode API pour Intégration

### Démarrer l'API
```bash
uvicorn rag_chunk_lab.api:app --host 0.0.0.0 --port 8000
```

### Tester les 3 pipelines
```bash
# Réponse extractive (rapide)
curl "http://localhost:8000/ask?doc_id=test&question=Quel délai de prescription ?" | jq .

# Réponse LLM (plus précise)
curl "http://localhost:8000/ask?doc_id=test&question=Quel délai de prescription ?&use_llm=true" | jq .
```

---

## ⚙️ Configuration Avancée

### Options de Génération Ground Truth
```bash
python3 generate_ground_truth.py \
  --folder documents/juridique \
  --model llama3:8b \                    # Modèle plus performant
  --questions-per-doc 20 \               # Plus de questions
  --min-length 300 \                     # Textes plus longs
  --max-length 600 \                     # Limite plus haute
  --output dataset_juridique.jsonl       # Nom personnalisé
```

### Paramètres de Chunking
Modifiez dans `config.py` :
```python
DEFAULTS.fixed_size_tokens = 600        # Chunks plus grands
DEFAULTS.sliding_window = 500           # Fenêtre plus large
DEFAULTS.top_k = 10                     # Plus de contexte
```

---

## 🎯 Cas d'Usage Typiques

### 1. Documents Juridiques
- **Structure-aware** souvent meilleur (respecte articles/sections)
- Ground truth avec questions précises sur procédures

### 2. Documentation Technique
- **Fixed chunks** bon compromis vitesse/qualité
- Questions sur API, configurations, troubleshooting

### 3. Rapports d'Analyse
- **Sliding window** capture mieux les relations entre sections
- Questions sur tendances, conclusions, recommandations

---

## 🔧 Structure du Projet

```
rag_chunk_lab/
├── cli.py                 # Interface ligne de commande
├── api.py                 # API FastAPI
├── chunkers.py            # 3 stratégies de chunking
├── indexing.py            # Index TF-IDF + métadonnées
├── retrieval.py           # Recherche de candidats
├── generation.py          # Génération de réponses
├── evaluation.py          # Métriques RAGAS
├── ground_truth_generator.py  # Génération auto de datasets
└── utils.py               # Utilitaires PDF/texte

generate_ground_truth.py   # Script standalone
data/                      # Index et chunks par document
exports/                   # Résultats CSV d'évaluation
```

---

## 💡 Tips d'Optimisation

### Pour Améliorer les Performances
1. **Tester plusieurs tailles** de chunks (300, 500, 800 tokens)
2. **Ajuster l'overlap** selon le type de document (80-200 tokens)
3. **Utiliser structure-aware** sur documents bien structurés
4. **Générer plus de questions** pour une évaluation robuste (20+ par doc)

### Pour l'Analyse
1. **Créer des graphiques radar** dans Excel (4 métriques × 3 pipelines)
2. **Segmenter par type de question** (procédurale, factuelle, analytique)
3. **Comparer avec/sans LLM** pour voir l'impact de la génération
4. **Tester sur plusieurs documents** du même domaine

---

## 🆘 Résolution de Problèmes

### Ollama ne démarre pas
```bash
# Vérifier si Ollama est installé
ollama --version

# Démarrer le service
ollama serve

# Vérifier les modèles installés
ollama list
```

### Erreurs Azure OpenAI
```bash
# Vérifier les variables d'environnement
echo $AZURE_OPENAI_API_KEY
echo $AZURE_OPENAI_ENDPOINT

# Tester la connexion
curl -H "api-key: $AZURE_OPENAI_API_KEY" "$AZURE_OPENAI_ENDPOINT/openai/deployments?api-version=2024-02-15-preview"
```

### Erreurs RAGAS "IndexError"
- **Cause** : Contextes vides ou réponses manquantes
- **Solution** : Le code filtre automatiquement les entrées problématiques
- **Debug** : Vérifier les logs pour voir combien d'entrées valides restent

---

## 🎉 Prêt à Optimiser Votre RAG !

1. **Créez votre dataset** avec `generate_ground_truth.py`
2. **Comparez les stratégies** avec `evaluate --ragas`
3. **Analysez dans Excel** les fichiers CSV générés
4. **Choisissez la meilleure stratégie** pour votre cas d'usage
5. **Intégrez via l'API** dans votre application

**Résultat** : Un pipeline RAG optimisé spécifiquement pour vos documents ! 🚀