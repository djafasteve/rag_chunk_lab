#!/usr/bin/env python3
"""
Tests unitaires pour le système d'optimisation des requêtes vagues
"""

import unittest
import sys
from pathlib import Path

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent))

from rag_chunk_lab.vague_query_optimizer import VagueQueryOptimizer
from rag_chunk_lab.hierarchical_chunking import HierarchicalChunker
from rag_chunk_lab.vague_query_optimization_system import (
    create_vague_optimization_system,
    quick_vague_query_optimization
)


class TestVagueQueryOptimizer(unittest.TestCase):
    """Tests pour VagueQueryOptimizer"""

    def setUp(self):
        self.optimizer = VagueQueryOptimizer()

    def test_vague_detection(self):
        """Test de détection des requêtes vagues"""
        # Cas vagues
        vague_queries = [
            "Comment ça marche ?",
            "Qu'est-ce que c'est ?",
            "Peux-tu m'expliquer ?",
            "C'est quoi ça ?",
            "Comment faire ?"
        ]

        for query in vague_queries:
            is_vague, score = self.optimizer.is_vague_query(query)
            self.assertTrue(is_vague, f"'{query}' devrait être détectée comme vague")
            self.assertGreater(score, 0.4, f"Score trop bas pour '{query}': {score}")

    def test_precise_detection(self):
        """Test de détection des requêtes précises"""
        precise_queries = [
            "Article 1134 du Code Civil",
            "Comment calculer la TVA sur une facture",
            "Procédure de divorce par consentement mutuel",
            "Délai de prescription en droit commercial"
        ]

        for query in precise_queries:
            is_vague, score = self.optimizer.is_vague_query(query)
            self.assertFalse(is_vague, f"'{query}' ne devrait pas être vague")
            self.assertLess(score, 0.5, f"Score trop élevé pour '{query}': {score}")

    def test_query_expansion(self):
        """Test d'expansion des requêtes"""
        vague_query = "Comment ça marche ?"
        expanded = self.optimizer.expand_vague_query(vague_query)

        self.assertIsInstance(expanded, list)
        self.assertGreater(len(expanded), 1)
        self.assertIn(vague_query, expanded)  # La requête originale doit être incluse


class TestHierarchicalChunker(unittest.TestCase):
    """Tests pour HierarchicalChunker"""

    def setUp(self):
        self.chunker = HierarchicalChunker()

    def test_hierarchical_chunking(self):
        """Test de création de chunks hiérarchiques"""
        text = """
        Le droit civil français.

        Il régit les relations entre particuliers. Les contrats sont des accords
        entre parties. Ils doivent respecter certaines conditions.

        Les obligations contractuelles sont importantes.
        """

        hierarchy = self.chunker.create_hierarchical_chunks(text, "test_doc")

        # Vérifier la structure hiérarchique
        self.assertIsInstance(hierarchy, dict)
        self.assertGreater(len(hierarchy), 0)

        # Vérifier qu'il y a plusieurs niveaux
        expected_levels = ['document', 'section', 'paragraph', 'sentence', 'concept', 'summary']
        for level in expected_levels:
            self.assertIn(level, hierarchy)

        # Vérifier qu'il y a des chunks
        total_chunks = sum(len(chunks) for chunks in hierarchy.values())
        self.assertGreater(total_chunks, 0)


class TestVagueQueryOptimizationSystem(unittest.TestCase):
    """Tests pour le système complet"""

    def setUp(self):
        self.documents = [
            {"doc_id": "test1", "text": "Le droit civil français régit les relations entre particuliers."},
            {"doc_id": "test2", "text": "La procédure civile organise le déroulement des procès."}
        ]

    def test_quick_optimization(self):
        """Test de l'API rapide d'optimisation"""
        result = quick_vague_query_optimization(
            query="Comment ça marche ?",
            documents=self.documents,
            domain="legal"
        )

        # Vérifier la structure du résultat
        required_keys = ['is_vague', 'vagueness_score', 'expanded_queries',
                        'retrieved_chunks', 'enriched_context', 'optimized_prompt',
                        'context_quality', 'performance']

        for key in required_keys:
            self.assertIn(key, result)

        # Vérifier les types
        self.assertIsInstance(result['is_vague'], bool)
        self.assertIsInstance(result['vagueness_score'], (int, float))
        self.assertIsInstance(result['expanded_queries'], list)
        self.assertIsInstance(result['retrieved_chunks'], list)
        self.assertIsInstance(result['enriched_context'], str)
        self.assertIsInstance(result['optimized_prompt'], str)

    def test_full_system(self):
        """Test du système complet"""
        system = create_vague_optimization_system(domain="legal")

        # Test d'indexation
        stats = system.index_documents(self.documents)
        self.assertIsInstance(stats, dict)
        self.assertGreater(stats['processed_documents'], 0)

        # Test d'optimisation
        result = system.optimize_vague_query("Procédure ?")
        self.assertIsInstance(result, dict)
        self.assertIn('is_vague', result)

    def test_performance(self):
        """Test de performance"""
        start_time = time.time()

        result = quick_vague_query_optimization(
            query="Comment ça marche ?",
            documents=self.documents,
            domain="legal"
        )

        response_time = time.time() - start_time

        # Vérifier que la réponse est rapide (< 5 secondes)
        self.assertLess(response_time, 5.0)

        # Vérifier que le temps reporté est cohérent
        reported_time = result['performance']['response_time']
        self.assertLess(reported_time, response_time + 1.0)  # Marge d'erreur


class TestIntegration(unittest.TestCase):
    """Tests d'intégration"""

    def test_domain_adaptation(self):
        """Test d'adaptation aux différents domaines"""
        domains = ['legal', 'technical', 'medical', 'general']
        documents = [{"doc_id": "test", "text": "Test content for domain adaptation."}]

        for domain in domains:
            with self.subTest(domain=domain):
                result = quick_vague_query_optimization(
                    query="Comment ça marche ?",
                    documents=documents,
                    domain=domain
                )

                self.assertIsInstance(result, dict)
                self.assertIn('is_vague', result)

    def test_multiple_documents(self):
        """Test avec plusieurs documents"""
        large_docs = []
        for i in range(10):
            large_docs.append({
                "doc_id": f"doc_{i}",
                "text": f"Document {i} contient des informations importantes sur le sujet {i}."
            })

        result = quick_vague_query_optimization(
            query="Informations importantes ?",
            documents=large_docs,
            domain="general"
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result['retrieved_chunks']), 0)


if __name__ == '__main__':
    import time

    print("🧪 Lancement des tests unitaires pour l'optimisation des requêtes vagues")
    print("=" * 70)

    # Configuration des tests
    unittest.TestLoader.sortTestMethodsUsing = None

    # Lancer les tests
    unittest.main(verbosity=2, buffer=True)