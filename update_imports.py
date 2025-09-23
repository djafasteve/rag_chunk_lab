#!/usr/bin/env python3
"""
Script pour mettre à jour automatiquement tous les imports après réorganisation
"""

import os
import re
from pathlib import Path

# Mapping des anciens chemins vers les nouveaux
IMPORT_MAPPING = {
    # Core modules
    'from .core.chunkers import': 'from .core.chunkers import',
    'from .core.indexing import': 'from .core.indexing import',
    'from .core.generation import': 'from .core.generation import',
    'from .core.retrieval import': 'from .core.retrieval import',
    'from .core.hierarchical_chunking import': 'from .core.hierarchical_chunking import',

    # Evaluation modules
    'from .evaluation.evaluation import': 'from .evaluation.evaluation import',
    'from .evaluation.embedding_metrics import': 'from .evaluation.embedding_metrics import',
    'from .evaluation.embedding_analysis import': 'from .evaluation.embedding_analysis import',
    'from .evaluation.generic_evaluation import': 'from .evaluation.generic_evaluation import',
    'from .evaluation.legal_evaluation import': 'from .evaluation.legal_evaluation import',
    'from .evaluation.azure_foundry_evaluation import': 'from .evaluation.azure_foundry_evaluation import',
    'from .evaluation.ground_truth_generator import': 'from .evaluation.ground_truth_generator import',

    # Vague query modules
    'from .vague_query.vague_query_optimizer import': 'from .vague_query.vague_query_optimizer import',
    'from .vague_query.vague_query_optimization_system import': 'from .vague_query.vague_query_optimization_system import',
    'from .vague_query.adaptive_prompt_engineering import': 'from .vague_query.adaptive_prompt_engineering import',
    'from .vague_query.context_enrichment_pipeline import': 'from .vague_query.context_enrichment_pipeline import',
    'from .vague_query.hybrid_embeddings import': 'from .vague_query.hybrid_embeddings import',
    'from .vague_query.metadata_enricher import': 'from .vague_query.metadata_enricher import',

    # Utils modules
    'from .utils.utils import': 'from .utils.utils import',
    'from .utils.config import': 'from .utils.config import',
    'from .utils.monitoring import': 'from .utils.monitoring import',
    'from .utils.production_monitoring import': 'from .utils.production_monitoring import',
    'from .utils.embedding_fine_tuning import': 'from .utils.embedding_fine_tuning import',
}

# Mapping pour les imports directs (sans from)
DIRECT_IMPORT_MAPPING = {
    'rag_chunk_lab.chunkers': 'rag_chunk_lab.core.chunkers',
    'rag_chunk_lab.indexing': 'rag_chunk_lab.core.indexing',
    'rag_chunk_lab.generation': 'rag_chunk_lab.core.generation',
    'rag_chunk_lab.retrieval': 'rag_chunk_lab.core.retrieval',
    'rag_chunk_lab.hierarchical_chunking': 'rag_chunk_lab.core.hierarchical_chunking',

    'rag_chunk_lab.evaluation': 'rag_chunk_lab.evaluation.evaluation',
    'rag_chunk_lab.embedding_metrics': 'rag_chunk_lab.evaluation.embedding_metrics',
    'rag_chunk_lab.embedding_analysis': 'rag_chunk_lab.evaluation.embedding_analysis',
    'rag_chunk_lab.generic_evaluation': 'rag_chunk_lab.evaluation.generic_evaluation',
    'rag_chunk_lab.legal_evaluation': 'rag_chunk_lab.evaluation.legal_evaluation',
    'rag_chunk_lab.azure_foundry_evaluation': 'rag_chunk_lab.evaluation.azure_foundry_evaluation',
    'rag_chunk_lab.ground_truth_generator': 'rag_chunk_lab.evaluation.ground_truth_generator',

    'rag_chunk_lab.vague_query_optimizer': 'rag_chunk_lab.vague_query.vague_query_optimizer',
    'rag_chunk_lab.vague_query_optimization_system': 'rag_chunk_lab.vague_query.vague_query_optimization_system',
    'rag_chunk_lab.adaptive_prompt_engineering': 'rag_chunk_lab.vague_query.adaptive_prompt_engineering',
    'rag_chunk_lab.context_enrichment_pipeline': 'rag_chunk_lab.vague_query.context_enrichment_pipeline',
    'rag_chunk_lab.hybrid_embeddings': 'rag_chunk_lab.vague_query.hybrid_embeddings',
    'rag_chunk_lab.metadata_enricher': 'rag_chunk_lab.vague_query.metadata_enricher',

    'rag_chunk_lab.utils': 'rag_chunk_lab.utils.utils',
    'rag_chunk_lab.config': 'rag_chunk_lab.utils.config',
    'rag_chunk_lab.monitoring': 'rag_chunk_lab.utils.monitoring',
    'rag_chunk_lab.production_monitoring': 'rag_chunk_lab.utils.production_monitoring',
    'rag_chunk_lab.embedding_fine_tuning': 'rag_chunk_lab.utils.embedding_fine_tuning',
}

def update_file_imports(file_path):
    """Met à jour les imports dans un fichier Python"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Mettre à jour les imports "from ... import"
        for old_import, new_import in IMPORT_MAPPING.items():
            content = content.replace(old_import, new_import)

        # Mettre à jour les imports directs
        for old_module, new_module in DIRECT_IMPORT_MAPPING.items():
            # Pattern pour capturer "import module" et "import module as alias"
            pattern1 = rf'\bimport {re.escape(old_module)}\b'
            replacement1 = f'import {new_module}'
            content = re.sub(pattern1, replacement1, content)

            # Pattern pour capturer "from module import ..."
            pattern2 = rf'\bfrom {re.escape(old_module)} import'
            replacement2 = f'from {new_module} import'
            content = re.sub(pattern2, replacement2, content)

        # Sauvegarder seulement si des changements ont été faits
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            print(f"⏭️  No changes: {file_path}")
            return False

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def find_python_files(directory):
    """Trouve tous les fichiers Python dans un répertoire"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Ignorer certains répertoires
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'venv', '.rag']]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files

def main():
    """Fonction principale"""
    print("🔄 Mise à jour des imports après réorganisation...")

    # Répertoire racine du projet
    project_root = Path(__file__).parent

    # Trouver tous les fichiers Python
    python_files = find_python_files(project_root)

    print(f"📁 Fichiers Python trouvés: {len(python_files)}")

    updated_files = 0

    for file_path in python_files:
        if update_file_imports(file_path):
            updated_files += 1

    print(f"\n🎉 Mise à jour terminée!")
    print(f"📊 Fichiers mis à jour: {updated_files}/{len(python_files)}")

if __name__ == "__main__":
    main()