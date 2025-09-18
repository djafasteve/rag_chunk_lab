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
```

### Étape 2 : Tester avec un document
```bash
# Ingérer un document PDF/TXT/MD
python -m rag_chunk_lab.cli ingest --doc mon_document.pdf --doc-id test

# Poser une question et voir les 3 réponses
python -m rag_chunk_lab.cli ask --doc-id test --question "Quel est le délai de prescription ?"
```

**Résultat** : 3 réponses comparées + fichier `exports/test/sources_<timestamp>.csv`

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
python -m rag_chunk_lab.cli evaluate \
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