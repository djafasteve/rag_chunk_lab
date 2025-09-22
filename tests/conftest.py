"""
Configuration pytest pour les tests RAG Chunk Lab
"""
import pytest
import sys
import os
import tempfile
import shutil

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def temp_data_dir():
    """Fixture qui crée un dossier temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_pages():
    """Fixture avec des pages de test standard"""
    return [
        {
            'page': 1,
            'text': 'Article 1\nCeci est le premier article du document de test. '
                    'Il contient plusieurs phrases pour tester les fonctionnalités de chunking '
                    'et d\'indexation du système RAG Chunk Lab.',
            'source_file': 'test_document.pdf'
        },
        {
            'page': 2,
            'text': 'Section 2\nVoici la deuxième section avec du contenu différent. '
                    'Cette section teste les capacités de traitement de texte '
                    'et de génération de chunks sur plusieurs pages.',
            'source_file': 'test_document.pdf'
        }
    ]


@pytest.fixture
def sample_chunks():
    """Fixture avec des chunks de test standard"""
    return [
        {
            'doc_id': 'test_doc',
            'page': 1,
            'start': 0,
            'end': 10,
            'section_title': 'Introduction',
            'source_file': 'test.pdf',
            'text': 'Ceci est un texte de test pour l\'indexation et la recherche.'
        },
        {
            'doc_id': 'test_doc',
            'page': 1,
            'start': 8,
            'end': 18,
            'section_title': 'Introduction',
            'source_file': 'test.pdf',
            'text': 'Un autre texte pour tester les capacités de recherche.'
        }
    ]


@pytest.fixture
def sample_passages():
    """Fixture avec des passages de test pour la génération"""
    return [
        {
            'score': 0.95,
            'text': 'Le délai de prescription est de trois ans pour les contraventions. '
                    'Cette règle s\'applique à toutes les infractions de cette catégorie.',
            'meta': {
                'doc_id': 'code_penal',
                'page': 15,
                'section_title': 'Prescription',
                'source_file': 'code_penal.pdf'
            }
        },
        {
            'score': 0.87,
            'text': 'Les sanctions applicables varient selon la gravité de l\'infraction. '
                    'Le tribunal peut prononcer des amendes ou des peines d\'emprisonnement.',
            'meta': {
                'doc_id': 'code_penal',
                'page': 23,
                'section_title': 'Sanctions',
                'source_file': 'code_penal.pdf'
            }
        }
    ]


@pytest.fixture(autouse=True)
def reset_caches():
    """Fixture qui nettoie les caches entre les tests"""
    # Nettoyer les caches LRU si nécessaire
    try:
        from rag_chunk_lab.utils import tokenize_words
        from rag_chunk_lab.indexing import load_index_data, get_sentence_transformer
        from rag_chunk_lab.generation import get_azure_client

        # Vider les caches LRU
        tokenize_words.cache_clear()
        load_index_data.cache_clear()

        # Les singletons sont plus délicats à nettoyer, on les laisse en place
        # car ils simulent le comportement réel en production

    except ImportError:
        # Si les modules ne sont pas disponibles, ignorer
        pass

    yield

    # Nettoyage après le test si nécessaire