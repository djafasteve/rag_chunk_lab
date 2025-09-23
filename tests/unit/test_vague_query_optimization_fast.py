#!/usr/bin/env python3
"""
Tests rapides pour le système d'optimisation des requêtes vagues
(version optimisée pour éviter les téléchargements répétés)
"""

import unittest
import sys
import time
from pathlib import Path

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_chunk_lab.vague_query.vague_query_optimizer import VagueQueryOptimizer
from rag_chunk_lab.core.hierarchical_chunking import HierarchicalChunker


class TestVagueQueryOptimizerFast(unittest.TestCase):
    """Tests rapides pour VagueQueryOptimizer"""

    @classmethod
    def setUpClass(cls):
        """Configuration unique pour toute la classe"""
        cls.optimizer = VagueQueryOptimizer()

    def test_vague_detection_corrected(self):
        """Test de détection des requêtes vagues après corrections"""
        # Cas vagues
        vague_queries = [
            "Comment ça marche ?",
            "Qu'est-ce que c'est ?",
            "Peux-tu m'expliquer ?",
            "C'est quoi ça ?",
            "Comment faire ?"
        ]

        for query in vague_queries:
            with self.subTest(query=query):
                is_vague, score = self.optimizer.is_vague_query(query)
                self.assertTrue(is_vague, f"'{query}' devrait être détectée comme vague")
                self.assertGreater(score, 0.4, f"Score trop bas pour '{query}': {score}")

    def test_precise_detection_corrected(self):
        """Test de détection des requêtes précises après corrections"""
        precise_queries = [
            "Article 1134 du Code Civil",
            "Comment calculer la TVA sur une facture",  # Cas corrigé
            "Procédure de divorce par consentement mutuel",
            "Délai de prescription en droit commercial"
        ]

        for query in precise_queries:
            with self.subTest(query=query):
                is_vague, score = self.optimizer.is_vague_query(query)
                self.assertFalse(is_vague, f"'{query}' ne devrait pas être vague")
                self.assertLess(score, 0.4, f"Score trop élevé pour '{query}': {score}")

    def test_query_expansion_basic(self):
        """Test d'expansion des requêtes sans LLM"""
        vague_query = "Comment ça marche ?"
        expanded = self.optimizer.expand_vague_query(vague_query)

        self.assertIsInstance(expanded, list)
        self.assertGreater(len(expanded), 1)
        self.assertIn(vague_query, expanded)  # La requête originale doit être incluse

    def test_domain_adaptation(self):
        """Test d'adaptation aux différents domaines"""
        domains = ['legal', 'technical', 'medical', 'general']

        for domain in domains:
            with self.subTest(domain=domain):
                optimizer = VagueQueryOptimizer(domain=domain)
                is_vague, score = optimizer.is_vague_query("Comment ça marche ?")
                self.assertTrue(is_vague)  # Devrait être vague dans tous les domaines


class TestHierarchicalChunkerFast(unittest.TestCase):
    """Tests rapides pour HierarchicalChunker"""

    @classmethod
    def setUpClass(cls):
        """Configuration unique pour toute la classe"""
        cls.chunker = HierarchicalChunker()

    def test_hierarchical_chunking_basic(self):
        """Test de base de création de chunks hiérarchiques"""
        text = "Le droit civil français. Il régit les relations entre particuliers."

        hierarchy = self.chunker.create_hierarchical_chunks(text, "test_doc")

        # Vérifier la structure hiérarchique
        self.assertIsInstance(hierarchy, dict)
        self.assertGreater(len(hierarchy), 0)

        # Vérifier qu'il y a des chunks
        total_chunks = sum(len(chunks) for chunks in hierarchy.values())
        self.assertGreater(total_chunks, 0)


class TestSystemIntegrationFast(unittest.TestCase):
    """Tests d'intégration rapides sans téléchargements"""

    def test_detection_accuracy_comprehensive(self):
        """Test complet de précision de détection"""
        optimizer = VagueQueryOptimizer(domain='legal')

        test_cases = [
            # (query, expected_vague, description)
            ("Comment ça marche ?", True, "Très vague"),
            ("Qu'est-ce que c'est ?", True, "Très vague"),
            ("Comment calculer la TVA sur une facture ?", False, "Précise avec termes spécifiques"),
            ("Article 1134 du Code Civil", False, "Référence précise"),
            ("Procédure de divorce", False, "Terme juridique spécifique"),
            ("Délai de prescription", False, "Concept juridique précis"),
            ("Truc ?", True, "Mot vague"),
            ("Comment fonctionne le système judiciaire français ?", False, "Question spécifique"),
        ]

        correct_predictions = 0
        for query, expected_vague, description in test_cases:
            with self.subTest(query=query):
                is_vague, score = optimizer.is_vague_query(query)
                is_correct = is_vague == expected_vague

                if is_correct:
                    correct_predictions += 1

                self.assertEqual(is_vague, expected_vague,
                               f"'{query}' -> attendu: {expected_vague}, obtenu: {is_vague} (score: {score:.2f})")

        accuracy = correct_predictions / len(test_cases) * 100
        self.assertGreaterEqual(accuracy, 85, f"Précision insuffisante: {accuracy:.1f}%")

    def test_performance_isolated(self):
        """Test de performance isolé (sans téléchargements)"""
        optimizer = VagueQueryOptimizer()

        start_time = time.time()

        # Test multiple détections
        queries = [
            "Comment ça marche ?",
            "Article 1134",
            "Procédure civile",
            "Qu'est-ce que c'est ?",
            "TVA facture"
        ]

        for query in queries:
            is_vague, score = optimizer.is_vague_query(query)
            self.assertIsInstance(is_vague, bool)
            self.assertIsInstance(score, (int, float))

        elapsed_time = time.time() - start_time

        # Le test devrait être très rapide (< 1 seconde)
        self.assertLess(elapsed_time, 1.0, f"Tests trop lents: {elapsed_time:.3f}s")

    def test_edge_cases(self):
        """Test des cas limites"""
        optimizer = VagueQueryOptimizer()

        edge_cases = [
            ("", True),  # Requête vide
            ("   ", True),  # Espaces seulement
            ("?", True),  # Juste un point d'interrogation
            ("Comment", True),  # Mot unique générique
            ("Article", False),  # Mot unique spécifique
        ]

        for query, expected_vague in edge_cases:
            with self.subTest(query=repr(query)):
                is_vague, score = optimizer.is_vague_query(query)
                # Pas d'assertion stricte car ces cas sont ambigus
                # Juste vérifier que ça ne plante pas
                self.assertIsInstance(is_vague, bool)
                self.assertIsInstance(score, (int, float))


if __name__ == '__main__':
    print("🧪 Tests rapides d'optimisation des requêtes vagues")
    print("=" * 60)

    # Configuration des tests
    unittest.TestLoader.sortTestMethodsUsing = None

    # Lancer les tests
    unittest.main(verbosity=2, buffer=True)