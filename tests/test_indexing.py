"""
Tests d'intégration pour le module indexing.py
"""
import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock
from rag_chunk_lab.core.indexing import (
    build_index,
    build_semantic_index,
    build_azure_semantic_index,
    retrieve,
    semantic_retrieve,
    azure_semantic_retrieve,
    load_index_data,
    get_sentence_transformer
)


class TestIndexingIntegration(unittest.TestCase):
    """Tests d'intégration pour le système d'indexation"""

    def setUp(self):
        """Prépare les données de test"""
        self.test_chunks = [
            {
                'doc_id': 'test_doc',
                'page': 1,
                'start': 0,
                'end': 10,
                'section_title': 'Introduction',
                'source_file': 'test.pdf',
                'text': 'Ceci est un texte de test pour l\'indexation.'
            },
            {
                'doc_id': 'test_doc',
                'page': 1,
                'start': 8,
                'end': 18,
                'section_title': 'Introduction',
                'source_file': 'test.pdf',
                'text': 'Un autre texte pour tester la recherche.'
            },
            {
                'doc_id': 'test_doc',
                'page': 2,
                'start': 0,
                'end': 12,
                'section_title': 'Conclusion',
                'source_file': 'test.pdf',
                'text': 'Texte de conclusion avec des mots différents.'
            }
        ]

    def test_build_index_tfidf(self):
        """Test de construction d'index TF-IDF classique"""
        with tempfile.TemporaryDirectory() as temp_dir:
            build_index('test_doc', 'fixed', self.test_chunks, temp_dir)

            # Vérifier que les fichiers sont créés
            base_path = f"{temp_dir}/test_doc/fixed"
            self.assertTrue(os.path.exists(f"{base_path}/chunks_meta.json"))
            self.assertTrue(os.path.exists(f"{base_path}/chunks_texts.json"))
            self.assertTrue(os.path.exists(f"{base_path}/tfidf_vectorizer.joblib"))
            self.assertTrue(os.path.exists(f"{base_path}/tfidf_matrix.joblib"))

            # Vérifier le contenu des fichiers JSON
            with open(f"{base_path}/chunks_texts.json", 'r') as f:
                texts = json.load(f)
            self.assertEqual(len(texts), len(self.test_chunks))

            with open(f"{base_path}/chunks_meta.json", 'r') as f:
                meta = json.load(f)
            self.assertEqual(len(meta), len(self.test_chunks))

    def test_build_semantic_index_mock(self):
        """Test de construction d'index sémantique (mocked)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rag_chunk_lab.indexing.get_sentence_transformer') as mock_st:
                mock_model = MagicMock()
                mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
                mock_model.encode.return_value = mock_embeddings
                mock_st.return_value = mock_model

                texts = [chunk['text'] for chunk in self.test_chunks]
                build_semantic_index('test_doc', texts, temp_dir)

                # Vérifier que les fichiers sont créés
                base_path = f"{temp_dir}/test_doc/semantic"
                self.assertTrue(os.path.exists(f"{base_path}/embeddings.npy"))
                self.assertTrue(os.path.exists(f"{base_path}/model"))

                # Vérifier les embeddings
                saved_embeddings = np.load(f"{base_path}/embeddings.npy")
                self.assertEqual(saved_embeddings.shape, (3, 3))
                self.assertEqual(saved_embeddings.dtype, np.float32)  # Optimisation mémoire

    def test_build_azure_semantic_index_mock(self):
        """Test de construction d'index Azure sémantique (mocked)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rag_chunk_lab.indexing.AZURE_AVAILABLE', True):
                with patch('rag_chunk_lab.indexing.AZURE_CONFIG') as mock_config:
                    mock_config.api_key = 'test'
                    mock_config.endpoint = 'https://test.com'

                    with patch('rag_chunk_lab.indexing.get_azure_embeddings_batch') as mock_batch:
                        mock_batch.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

                        texts = [chunk['text'] for chunk in self.test_chunks]
                        build_azure_semantic_index('test_doc', texts, temp_dir)

                        # Vérifier que les fichiers sont créés
                        base_path = f"{temp_dir}/test_doc/azure_semantic"
                        self.assertTrue(os.path.exists(f"{base_path}/embeddings.npy"))

                        # Vérifier les embeddings
                        saved_embeddings = np.load(f"{base_path}/embeddings.npy")
                        self.assertEqual(saved_embeddings.shape, (3, 2))
                        self.assertEqual(saved_embeddings.dtype, np.float32)

                        # Vérifier que le batch a été appelé
                        mock_batch.assert_called_once_with(texts, batch_size=100)

    def test_retrieve_tfidf(self):
        """Test de recherche TF-IDF"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construire l'index
            build_index('test_doc', 'fixed', self.test_chunks, temp_dir)

            # Tester la recherche
            results = retrieve('test_doc', 'fixed', 'texte test', top_k=2, data_dir=temp_dir)

            self.assertIsInstance(results, list)
            self.assertTrue(len(results) > 0)
            self.assertLessEqual(len(results), 2)

            # Vérifier la structure des résultats
            for result in results:
                self.assertIn('score', result)
                self.assertIn('text', result)
                self.assertIn('meta', result)
                self.assertIsInstance(result['score'], float)

    def test_semantic_retrieve_mock(self):
        """Test de recherche sémantique (mocked)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Préparer les données
            texts = [chunk['text'] for chunk in self.test_chunks]
            meta = [{'page': chunk['page'], 'doc_id': chunk['doc_id']} for chunk in self.test_chunks]

            # Créer des embeddings factices
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
            semantic_dir = f"{temp_dir}/test_doc/semantic"
            os.makedirs(semantic_dir)
            np.save(f"{semantic_dir}/embeddings.npy", embeddings)

            with patch('rag_chunk_lab.indexing.get_sentence_transformer') as mock_st:
                mock_model = MagicMock()
                query_embedding = np.array([[0.2, 0.3]])
                mock_model.encode.return_value = query_embedding
                mock_st.return_value = mock_model

                results = semantic_retrieve('test_doc', 'texte test', 2, temp_dir, texts, meta)

                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 2)

                # Vérifier la structure
                for result in results:
                    self.assertIn('score', result)
                    self.assertIn('text', result)
                    self.assertIn('meta', result)

    def test_azure_semantic_retrieve_mock(self):
        """Test de recherche Azure sémantique (mocked)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            texts = [chunk['text'] for chunk in self.test_chunks]
            meta = [{'page': chunk['page'], 'doc_id': chunk['doc_id']} for chunk in self.test_chunks]

            # Créer des embeddings factices
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
            azure_dir = f"{temp_dir}/test_doc/azure_semantic"
            os.makedirs(azure_dir)
            np.save(f"{azure_dir}/embeddings.npy", embeddings)

            with patch('rag_chunk_lab.indexing.get_azure_embedding') as mock_embed:
                mock_embed.return_value = [0.2, 0.3]

                results = azure_semantic_retrieve('test_doc', 'texte test', 2, temp_dir, texts, meta)

                self.assertIsInstance(results, list)
                self.assertEqual(len(results), 2)

                mock_embed.assert_called_once_with('texte test')

    def test_load_index_data_cache(self):
        """Test du cache LRU pour load_index_data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Préparer les fichiers
            pipeline_dir = f"{temp_dir}/test_doc/test_pipeline"
            os.makedirs(pipeline_dir)

            texts = ['text1', 'text2']
            meta = [{'page': 1}, {'page': 2}]

            with open(f"{pipeline_dir}/chunks_texts.json", 'w') as f:
                json.dump(texts, f)
            with open(f"{pipeline_dir}/chunks_meta.json", 'w') as f:
                json.dump(meta, f)

            # Premier appel
            texts1, meta1 = load_index_data('test_doc', 'test_pipeline', temp_dir)

            # Deuxième appel (devrait utiliser le cache)
            texts2, meta2 = load_index_data('test_doc', 'test_pipeline', temp_dir)

            self.assertEqual(texts1, texts2)
            self.assertEqual(meta1, meta2)
            self.assertEqual(texts1, texts)
            self.assertEqual(meta1, meta)

    def test_pipeline_switching(self):
        """Test que build_index appelle la bonne fonction selon le pipeline"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rag_chunk_lab.indexing.build_semantic_index') as mock_semantic:
                with patch('rag_chunk_lab.indexing.build_azure_semantic_index') as mock_azure:
                    # Test pipeline sémantique
                    build_index('test_doc', 'semantic', self.test_chunks, temp_dir)
                    mock_semantic.assert_called_once()

                    # Test pipeline Azure sémantique
                    build_index('test_doc', 'azure_semantic', self.test_chunks, temp_dir)
                    mock_azure.assert_called_once()

    def test_error_handling_missing_pipeline(self):
        """Test de gestion d'erreur pour pipeline manquant"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                retrieve('nonexistent_doc', 'nonexistent_pipeline', 'query', 5, temp_dir)

    def test_singleton_sentence_transformer_cache(self):
        """Test que get_sentence_transformer utilise bien le cache"""
        with patch('rag_chunk_lab.indexing.SEMANTIC_AVAILABLE', True):
            with patch('rag_chunk_lab.indexing.SentenceTransformer') as mock_st:
                mock_instance = MagicMock()
                mock_st.return_value = mock_instance

                # Premier appel
                model1 = get_sentence_transformer()
                # Deuxième appel
                model2 = get_sentence_transformer()

                # Le constructeur ne doit être appelé qu'une fois
                mock_st.assert_called_once()
                self.assertEqual(model1, model2)


if __name__ == '__main__':
    unittest.main()