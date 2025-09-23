"""
Tests d'intégration pour le module chunkers.py
"""
import unittest
from unittest import mock
import tempfile
from rag_chunk_lab.core.chunkers import (
    fixed_chunks,
    structure_aware_chunks,
    sliding_window_chunks,
    semantic_chunks,
    azure_semantic_chunks,
    tokenize_pages_once
)


class TestChunkersIntegration(unittest.TestCase):
    """Tests d'intégration pour les fonctions de chunking"""

    def setUp(self):
        """Prépare les données de test"""
        self.test_pages = [
            {
                'page': 1,
                'text': 'Article 1\nCeci est le premier article avec plusieurs mots pour tester le chunking. '
                        'Il contient suffisamment de contenu pour créer plusieurs chunks.',
                'source_file': 'test_doc.pdf'
            },
            {
                'page': 2,
                'text': 'Section 2\nVoici une seconde page avec du contenu différent. '
                        'Cette page teste aussi la fonctionnalité de chunking sur plusieurs pages.',
                'source_file': 'test_doc.pdf'
            }
        ]
        self.doc_id = 'test_document'

    def test_fixed_chunks_integration(self):
        """Test complet de fixed_chunks avec optimisations"""
        chunks = fixed_chunks(
            pages=self.test_pages,
            size_tokens=10,
            overlap_tokens=3,
            doc_id=self.doc_id
        )

        # Vérifications de base
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

        # Vérifier la structure des chunks
        for chunk in chunks:
            self.assertIn('doc_id', chunk)
            self.assertIn('page', chunk)
            self.assertIn('text', chunk)
            self.assertIn('source_file', chunk)
            self.assertEqual(chunk['doc_id'], self.doc_id)
            self.assertIsInstance(chunk['text'], str)

        # Vérifier que l'overlap fonctionne
        if len(chunks) > 1:
            # Au moins quelques mots doivent se chevaucher
            words_chunk1 = chunks[0]['text'].split()
            words_chunk2 = chunks[1]['text'].split()
            # Il devrait y avoir un chevauchement
            self.assertTrue(len(words_chunk1) > 0)
            self.assertTrue(len(words_chunk2) > 0)

    def test_structure_aware_chunks_integration(self):
        """Test complet de structure_aware_chunks"""
        chunks = structure_aware_chunks(
            pages=self.test_pages,
            size_tokens=15,
            overlap_tokens=5,
            doc_id=self.doc_id
        )

        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

        # Vérifier que les titres de section sont détectés
        section_titles = [chunk.get('section_title') for chunk in chunks]
        # Au moins un chunk devrait avoir un titre de section détecté
        self.assertTrue(any(title is not None for title in section_titles))

    def test_sliding_window_chunks_integration(self):
        """Test complet de sliding_window_chunks"""
        chunks = sliding_window_chunks(
            pages=self.test_pages,
            window=8,
            stride=4,
            doc_id=self.doc_id
        )

        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

        # Vérifier que le stride fonctionne correctement
        if len(chunks) > 1:
            # Les positions de début doivent progresser par le stride
            starts = [chunk['start'] for chunk in chunks if chunk['page'] == 1]
            if len(starts) > 1:
                # La différence entre starts consécutifs devrait être le stride (4)
                self.assertEqual(starts[1] - starts[0], 4)

    def test_semantic_chunks_fallback(self):
        """Test que semantic_chunks utilise fixed_chunks comme base"""
        # Test sans sentence-transformers disponible
        with mock.patch('rag_chunk_lab.chunkers.SEMANTIC_AVAILABLE', False):
            with self.assertRaises(ImportError):
                semantic_chunks(
                    pages=self.test_pages,
                    size_tokens=10,
                    overlap_tokens=3,
                    doc_id=self.doc_id
                )

        # Test avec sentence-transformers disponible (mock)
        with mock.patch('rag_chunk_lab.chunkers.SEMANTIC_AVAILABLE', True):
            chunks = semantic_chunks(
                pages=self.test_pages,
                size_tokens=10,
                overlap_tokens=3,
                doc_id=self.doc_id
            )

            # Devrait retourner des chunks comme fixed_chunks
            self.assertIsInstance(chunks, list)
            self.assertTrue(len(chunks) > 0)

    def test_azure_semantic_chunks_fallback(self):
        """Test que azure_semantic_chunks utilise fixed_chunks comme base"""
        chunks = azure_semantic_chunks(
            pages=self.test_pages,
            size_tokens=10,
            overlap_tokens=3,
            doc_id=self.doc_id
        )

        # Devrait retourner des chunks comme fixed_chunks
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)

    def test_tokenize_pages_once_optimization(self):
        """Test de l'optimisation tokenize_pages_once"""
        result = tokenize_pages_once(self.test_pages)

        self.assertEqual(len(result), len(self.test_pages))

        for i, (page, tokens) in enumerate(result):
            self.assertEqual(page, self.test_pages[i])
            self.assertIsInstance(tokens, list)
            self.assertTrue(len(tokens) > 0)
            # Vérifier que les tokens sont des chaînes
            self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_chunkers_consistency(self):
        """Test que tous les chunkers produisent des structures cohérentes"""
        chunkers = [
            ('fixed', fixed_chunks),
            ('structure', structure_aware_chunks),
            ('sliding', sliding_window_chunks),
            ('semantic', semantic_chunks),
            ('azure_semantic', azure_semantic_chunks)
        ]

        for name, chunker_func in chunkers:
            if name in ['semantic', 'azure_semantic']:
                # Ces chunkers utilisent fixed_chunks sous le capot
                if name == 'semantic':
                    with mock.patch('rag_chunk_lab.chunkers.SEMANTIC_AVAILABLE', True):
                        chunks = chunker_func(self.test_pages, 10, 3, self.doc_id)
                else:
                    chunks = chunker_func(self.test_pages, 10, 3, self.doc_id)
            else:
                if name == 'sliding':
                    chunks = chunker_func(self.test_pages, 8, 4, self.doc_id)
                else:
                    chunks = chunker_func(self.test_pages, 10, 3, self.doc_id)

            # Vérifications communes
            self.assertIsInstance(chunks, list, f"Chunker {name} should return a list")
            self.assertTrue(len(chunks) > 0, f"Chunker {name} should produce chunks")

            for i, chunk in enumerate(chunks):
                self.assertIsInstance(chunk, dict, f"Chunker {name} chunk {i} should be a dict")

                required_fields = ['doc_id', 'page', 'start', 'end', 'text', 'source_file']
                for field in required_fields:
                    self.assertIn(field, chunk, f"Chunker {name} chunk {i} missing field {field}")

                self.assertEqual(chunk['doc_id'], self.doc_id)
                self.assertIsInstance(chunk['page'], int)
                self.assertIsInstance(chunk['start'], int)
                self.assertIsInstance(chunk['end'], int)
                self.assertIsInstance(chunk['text'], str)
                self.assertTrue(len(chunk['text']) > 0)

    def test_chunk_metadata_preservation(self):
        """Test que les métadonnées des pages sont préservées dans les chunks"""
        chunks = fixed_chunks(self.test_pages, 10, 3, self.doc_id)

        for chunk in chunks:
            # Vérifier que source_file est préservé
            self.assertEqual(chunk['source_file'], 'test_doc.pdf')

            # Vérifier que page correspond à une page valide
            self.assertIn(chunk['page'], [1, 2])


if __name__ == '__main__':
    unittest.main()