"""
Tests de base pour valider l'infrastructure
"""
import unittest
import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicFunctionality(unittest.TestCase):
    """Tests de base pour valider que tout fonctionne"""

    def test_imports(self):
        """Test que tous les modules s'importent correctement"""
        # Test des imports principaux
        from rag_chunk_lab import chunkers
        from rag_chunk_lab import utils
        from rag_chunk_lab import indexing
        from rag_chunk_lab import generation
        from rag_chunk_lab import monitoring

        self.assertTrue(hasattr(chunkers, 'fixed_chunks'))
        self.assertTrue(hasattr(utils, 'tokenize_words'))
        self.assertTrue(hasattr(indexing, 'build_index'))
        self.assertTrue(hasattr(generation, 'extractive_answer'))
        self.assertTrue(hasattr(monitoring, 'PerformanceMonitor'))

    def test_tokenization_basic(self):
        """Test de base de la tokenisation"""
        from rag_chunk_lab.utils import tokenize_words, join_tokens

        text = "Hello world! This is a test."
        tokens = tokenize_words(text)

        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        self.assertIn('Hello', tokens)
        self.assertIn('world', tokens)

        # Test de rejoining
        rejoined = join_tokens(tokens)
        self.assertIsInstance(rejoined, str)
        self.assertIn('Hello', rejoined)

    def test_chunking_basic(self):
        """Test de base du chunking"""
        from rag_chunk_lab.chunkers import fixed_chunks

        pages = [
            {'page': 1, 'text': 'This is a test document with some text.', 'source_file': 'test.txt'}
        ]

        chunks = fixed_chunks(pages, size_tokens=5, overlap_tokens=2, doc_id='test')

        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

        # Vérifier la structure d'un chunk
        chunk = chunks[0]
        self.assertIn('doc_id', chunk)
        self.assertIn('text', chunk)
        self.assertIn('page', chunk)
        self.assertEqual(chunk['doc_id'], 'test')

    def test_monitoring_basic(self):
        """Test de base du monitoring"""
        from rag_chunk_lab.monitoring import PerformanceMonitor

        monitor = PerformanceMonitor()

        @monitor.monitor_function("test_function")
        def test_func():
            return "success"

        result = test_func()

        self.assertEqual(result, "success")
        self.assertIn("test_function", monitor.metrics)

    def test_optimizations_working(self):
        """Test que les optimisations de base fonctionnent"""
        from rag_chunk_lab.chunkers import tokenize_pages_once

        pages = [
            {'page': 1, 'text': 'Test page one'},
            {'page': 2, 'text': 'Test page two'}
        ]

        result = tokenize_pages_once(pages)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0]['page'], 1)
        self.assertEqual(result[1][0]['page'], 2)
        self.assertIsInstance(result[0][1], list)  # tokens


if __name__ == '__main__':
    unittest.main(verbosity=2)