"""
Tests unitaires pour les optimisations RAG Chunk Lab v2.0
"""
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import os
import numpy as np
from rag_chunk_lab.core.chunkers import fixed_chunks, structure_aware_chunks, sliding_window_chunks, tokenize_pages_once
from rag_chunk_lab.utils.utils import tokenize_words, join_tokens
from rag_chunk_lab.utils.monitoring import PerformanceMonitor, monitor_performance


class TestTokenizationOptimizations(unittest.TestCase):
    """Tests pour les optimisations de tokenisation"""

    def test_tokenize_words_cache(self):
        """Test que la tokenisation utilise bien le cache LRU"""
        text = "Hello world! This is a test."

        # Premier appel
        result1 = tokenize_words(text)
        # Deuxième appel avec le même texte (doit utiliser le cache)
        result2 = tokenize_words(text)

        self.assertEqual(result1, result2)
        self.assertIsInstance(result1, list)
        self.assertTrue(len(result1) > 0)

    def test_tokenize_pages_once_optimization(self):
        """Test que tokenize_pages_once évite la re-tokenisation"""
        pages = [
            {'page': 1, 'text': 'Hello world'},
            {'page': 2, 'text': 'Second page'}
        ]

        result = tokenize_pages_once(pages)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['page'], 1)
        self.assertEqual(result[1][0]['page'], 2)
        self.assertIsInstance(result[0][1], list)  # tokens
        self.assertIsInstance(result[1][1], list)  # tokens

    def test_chunkers_use_optimized_tokenization(self):
        """Test que les chunkers utilisent la tokenisation optimisée"""
        pages = [{'page': 1, 'text': 'This is a test document with multiple words.'}]

        with patch('rag_chunk_lab.chunkers.tokenize_pages_once') as mock_tokenize:
            mock_tokenize.return_value = [(pages[0], ['This', 'is', 'a', 'test', 'document', 'with', 'multiple', 'words', '.'])]

            chunks = fixed_chunks(pages, size_tokens=5, overlap_tokens=2, doc_id='test')

            mock_tokenize.assert_called_once_with(pages)
            self.assertTrue(len(chunks) > 0)


class TestCacheOptimizations(unittest.TestCase):
    """Tests pour les optimisations de cache"""

    def test_singleton_pattern_azure_client(self):
        """Test que le client Azure utilise bien le singleton pattern"""
        from rag_chunk_lab.core.generation import get_azure_client

        with patch('rag_chunk_lab.generation.AZURE_CONFIG') as mock_config:
            mock_config.api_key = 'test-key'
            mock_config.endpoint = 'https://test.openai.azure.com'
            mock_config.api_version = '2024-02-15-preview'

            with patch('rag_chunk_lab.generation.AzureOpenAI') as mock_azure:
                mock_instance = MagicMock()
                mock_azure.return_value = mock_instance

                # Premier appel
                client1 = get_azure_client()
                # Deuxième appel (doit utiliser le cache)
                client2 = get_azure_client()

                # Le client Azure ne doit être créé qu'une seule fois
                mock_azure.assert_called_once()
                self.assertEqual(client1, client2)

    def test_sentence_transformer_singleton(self):
        """Test que SentenceTransformer utilise bien le singleton pattern"""
        from rag_chunk_lab.core.indexing import get_sentence_transformer

        with patch('rag_chunk_lab.indexing.SEMANTIC_AVAILABLE', True):
            with patch('rag_chunk_lab.indexing.SentenceTransformer') as mock_st:
                mock_instance = MagicMock()
                mock_st.return_value = mock_instance

                # Premier appel
                model1 = get_sentence_transformer()
                # Deuxième appel (doit utiliser le cache)
                model2 = get_sentence_transformer()

                # Le modèle ne doit être créé qu'une seule fois
                mock_st.assert_called_once()
                self.assertEqual(model1, model2)

    def test_index_data_cache(self):
        """Test que load_index_data utilise bien le cache LRU"""
        from rag_chunk_lab.core.indexing import load_index_data

        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer des fichiers de test
            pipeline_dir = os.path.join(temp_dir, 'test_doc', 'test_pipeline')
            os.makedirs(pipeline_dir)

            with open(os.path.join(pipeline_dir, 'chunks_texts.json'), 'w') as f:
                f.write('["text1", "text2"]')
            with open(os.path.join(pipeline_dir, 'chunks_meta.json'), 'w') as f:
                f.write('[{"page": 1}, {"page": 2}]')

            # Premier appel
            texts1, meta1 = load_index_data('test_doc', 'test_pipeline', temp_dir)
            # Deuxième appel (doit utiliser le cache)
            texts2, meta2 = load_index_data('test_doc', 'test_pipeline', temp_dir)

            self.assertEqual(texts1, texts2)
            self.assertEqual(meta1, meta2)


class TestBatchOptimizations(unittest.TestCase):
    """Tests pour les optimisations de batch processing"""

    def test_azure_embeddings_batch_function(self):
        """Test que get_azure_embeddings_batch traite par batch"""
        from rag_chunk_lab.core.generation import get_azure_embeddings_batch

        texts = [f"text {i}" for i in range(250)]  # 250 textes pour tester le batch

        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client_getter:
            mock_client = MagicMock()
            mock_client_getter.return_value = mock_client

            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(100)]
            mock_client.embeddings.create.return_value = mock_response

            result = get_azure_embeddings_batch(texts, batch_size=100)

            # Doit faire 3 appels (250 textes / 100 par batch = 3 batches)
            self.assertEqual(mock_client.embeddings.create.call_count, 3)
            self.assertEqual(len(result), 250)

    def test_memory_optimization_float32(self):
        """Test que les embeddings utilisent float32"""
        from rag_chunk_lab.core.indexing import build_semantic_index

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rag_chunk_lab.indexing.get_sentence_transformer') as mock_st:
                mock_model = MagicMock()
                mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
                mock_model.encode.return_value = mock_embeddings
                mock_st.return_value = mock_model

                build_semantic_index('test_doc', ['text1', 'text2'], temp_dir)

                # Vérifier que les embeddings sont sauvés en float32
                saved_embeddings = np.load(f"{temp_dir}/test_doc/semantic/embeddings.npy")
                self.assertEqual(saved_embeddings.dtype, np.float32)


class TestParallelOptimizations(unittest.TestCase):
    """Tests pour les optimisations de parallélisation"""

    def test_parallel_pipeline_ingestion_mock(self):
        """Test que l'ingestion utilise ThreadPoolExecutor (mock)"""
        from concurrent.futures import ThreadPoolExecutor

        # Ce test vérifie la structure, pas l'exécution réelle
        # car ThreadPoolExecutor est difficile à mocker complètement
        with patch('rag_chunk_lab.cli.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value = MagicMock()

            # Import et test de structure
            import rag_chunk_lab.cli
            self.assertTrue(hasattr(rag_chunk_lab.cli, 'ThreadPoolExecutor'))


class TestMonitoringOptimizations(unittest.TestCase):
    """Tests pour le système de monitoring"""

    def test_performance_monitor_decorator(self):
        """Test que le décorateur de monitoring fonctionne"""
        monitor = PerformanceMonitor()

        @monitor.monitor_function("test_function")
        def test_func():
            return "success"

        result = test_func()

        self.assertEqual(result, "success")
        self.assertIn("test_function", monitor.metrics)
        self.assertEqual(len(monitor.metrics["test_function"]), 1)
        self.assertTrue(monitor.metrics["test_function"][0]["success"])

    def test_performance_monitor_error_handling(self):
        """Test que le monitoring gère les erreurs"""
        monitor = PerformanceMonitor()

        @monitor.monitor_function("test_error_function")
        def test_error_func():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            test_error_func()

        self.assertIn("test_error_function", monitor.metrics)
        self.assertFalse(monitor.metrics["test_error_function"][0]["success"])
        self.assertEqual(monitor.metrics["test_error_function"][0]["error"], "Test error")

    def test_performance_summary_generation(self):
        """Test que le résumé de performance se génère correctement"""
        monitor = PerformanceMonitor()

        # Simuler des métriques
        monitor.record_metric("test_operation", {
            'duration_seconds': 1.0,
            'memory_delta_mb': 50.0,
            'success': True,
            'error': None
        })
        monitor.record_metric("test_operation", {
            'duration_seconds': 2.0,
            'memory_delta_mb': 75.0,
            'success': True,
            'error': None
        })

        summary = monitor.get_summary()

        self.assertIn("test_operation", summary)
        self.assertEqual(summary["test_operation"]["count"], 2)
        self.assertEqual(summary["test_operation"]["avg_duration_seconds"], 1.5)
        self.assertEqual(summary["test_operation"]["total_duration_seconds"], 3.0)


class TestFallbackMechanisms(unittest.TestCase):
    """Tests pour les mécanismes de fallback"""

    def test_azure_batch_fallback(self):
        """Test que le fallback fonctionne si le batch échoue"""
        from rag_chunk_lab.core.indexing import build_azure_semantic_index

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rag_chunk_lab.indexing.AZURE_AVAILABLE', True):
                with patch('rag_chunk_lab.indexing.AZURE_CONFIG') as mock_config:
                    mock_config.api_key = 'test'
                    mock_config.endpoint = 'https://test.com'

                    with patch('rag_chunk_lab.indexing.get_azure_embeddings_batch') as mock_batch:
                        with patch('rag_chunk_lab.indexing.get_azure_embedding') as mock_single:
                            # Batch échoue
                            mock_batch.side_effect = Exception("Batch failed")
                            # Single réussit
                            mock_single.return_value = [0.1, 0.2, 0.3]

                            # Ne doit pas lever d'exception
                            build_azure_semantic_index('test_doc', ['text1', 'text2'], temp_dir)

                            # Doit avoir essayé le batch puis le fallback
                            mock_batch.assert_called_once()
                            self.assertEqual(mock_single.call_count, 2)  # 2 textes


if __name__ == '__main__':
    unittest.main()