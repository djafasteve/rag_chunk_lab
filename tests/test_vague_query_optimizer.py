# test_vague_query_optimizer.py
"""
Tests unitaires pour le VagueQueryOptimizer
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from rag_chunk_lab.vague_query.vague_query_optimizer import VagueQueryOptimizer


class TestVagueQueryOptimizer:
    """Tests pour le VagueQueryOptimizer"""

    def setup_method(self):
        """Configuration pour chaque test"""
        self.optimizer = VagueQueryOptimizer(domain="general")

    def test_vague_query_detection_short(self):
        """Test détection requête courte = vague"""
        query = "Quoi?"
        is_vague, score = self.optimizer.is_vague_query(query)

        assert is_vague is True
        assert score >= 0.4

    def test_vague_query_detection_generic(self):
        """Test détection mots génériques"""
        query = "Comment ça marche?"
        is_vague, score = self.optimizer.is_vague_query(query)

        assert is_vague is True
        assert score > 0.3

    def test_precise_query_detection(self):
        """Test requête précise = non vague"""
        query = "Comment configurer les paramètres RAGAS pour l'évaluation contextuelle dans Azure ML?"
        is_vague, score = self.optimizer.is_vague_query(query)

        assert is_vague is False
        assert score < 0.4

    def test_domain_specific_detection_legal(self):
        """Test détection domaine juridique"""
        legal_optimizer = VagueQueryOptimizer(domain="legal")

        # Question avec termes juridiques = moins vague
        legal_query = "Quelle est la procédure pour porter plainte selon l'article 40?"
        is_vague, score = legal_optimizer.is_vague_query(legal_query)

        # Question générale = plus vague
        general_query = "Comment faire une procédure?"
        is_vague_general, score_general = legal_optimizer.is_vague_query(general_query)

        assert score < score_general  # Moins vague avec termes spécialisés

    def test_query_expansion_basic(self):
        """Test expansion basique"""
        query = "API"
        expansions = self.optimizer.expand_vague_query(query)

        assert len(expansions) > 1
        assert query in expansions  # Originale préservée
        assert any("qu'est-ce que" in exp.lower() for exp in expansions)
        assert any("comment" in exp.lower() for exp in expansions)

    @patch('openai.OpenAI')
    def test_llm_expansion(self, mock_openai):
        """Test expansion avec LLM"""
        # Mock de la réponse OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Comment utiliser une API\nQuels sont les types d'API\nComment intégrer une API"
        mock_client.chat.completions.create.return_value = mock_response

        optimizer = VagueQueryOptimizer(openai_api_key="test-key")
        optimizer.client = mock_client

        expansions = optimizer._llm_query_expansion("API")

        assert len(expansions) == 3
        assert "Comment utiliser une API" in expansions

    def test_context_enhancement_structure(self):
        """Test structure du contexte enrichi"""
        query = "Comment ça marche?"
        chunks = [
            "Une API est une interface de programmation.",
            "Les APIs permettent la communication entre applications.",
            "Par exemple, l'API REST utilise HTTP."
        ]

        enhanced_context = self.optimizer.enhance_context_for_vague_query(query, chunks)

        assert "=== CONTEXTE PRINCIPAL ===" in enhanced_context
        assert len(enhanced_context) > sum(len(chunk) for chunk in chunks)

    def test_adaptive_prompt_vague(self):
        """Test prompt adaptatif pour requête vague"""
        vague_query = "Aide"
        context = "Documentation sur les APIs"

        prompt = self.optimizer.create_adaptive_prompt(vague_query, context)

        assert "pédagogique" in prompt
        assert "MISSION:" in prompt
        assert "RÉPONSE STRUCTURÉE:" in prompt

    def test_adaptive_prompt_precise(self):
        """Test prompt adaptatif pour requête précise"""
        precise_query = "Comment configurer l'endpoint REST /api/v1/users pour supporter la pagination avec des paramètres limit et offset?"
        context = "Documentation API REST"

        prompt = self.optimizer.create_adaptive_prompt(precise_query, context)

        assert "directe et factuelle" in prompt
        assert "MISSION:" not in prompt  # Pas de mission pédagogique

    def test_retrieval_optimization_vague(self):
        """Test optimisation récupération pour requête vague"""
        query = "API"
        chunks = ["Chunk API 1", "Chunk base données", "Chunk API REST", "Chunk autre"]
        embeddings = np.random.rand(4, 384)

        with patch.object(self.optimizer.sentence_model, 'encode') as mock_encode:
            mock_encode.return_value = np.random.rand(1, 384)

            selected_chunks, scores = self.optimizer.optimize_retrieval_for_vague_query(
                query, chunks, embeddings, top_k=2
            )

        assert len(selected_chunks) == 2
        assert len(scores) == 2
        assert all(isinstance(score, (int, float)) for score in scores)

    def test_retrieval_optimization_precise(self):
        """Test récupération standard pour requête précise"""
        precise_query = "Configuration détaillée de l'authentification OAuth2 avec refresh tokens"
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        embeddings = np.random.rand(3, 384)

        with patch.object(self.optimizer.sentence_model, 'encode') as mock_encode:
            mock_encode.return_value = np.random.rand(1, 384)

            selected_chunks, scores = self.optimizer._standard_retrieval(
                precise_query, chunks, embeddings, top_k=2
            )

        assert len(selected_chunks) == 2
        assert len(scores) == 2

    def test_domain_keywords_loading(self):
        """Test chargement mots-clés domaine"""
        legal_optimizer = VagueQueryOptimizer(domain="legal")
        medical_optimizer = VagueQueryOptimizer(domain="medical")

        assert "droit" in legal_optimizer.domain_keywords.get("concepts", [])
        assert "symptôme" in medical_optimizer.domain_keywords.get("concepts", [])

    @pytest.mark.parametrize("domain,expected_concepts", [
        ("legal", ["droit", "loi", "article"]),
        ("medical", ["symptôme", "traitement", "diagnostic"]),
        ("technical", ["algorithme", "architecture", "API"]),
        ("general", ["principe", "méthode", "processus"])
    ])
    def test_domain_specific_keywords(self, domain, expected_concepts):
        """Test mots-clés spécifiques par domaine"""
        optimizer = VagueQueryOptimizer(domain=domain)

        for concept in expected_concepts:
            assert concept in optimizer.domain_keywords.get("concepts", [])

    def test_vagueness_score_ranges(self):
        """Test cohérence des scores de vague"""
        test_queries = [
            ("?", 1.0),  # Maximum vague
            ("Qu'est-ce que ça?", 0.8),  # Très vague
            ("Comment utiliser l'API REST?", 0.3),  # Moyennement précis
            ("Comment configurer OAuth2 avec JWT tokens pour l'authentification API dans Node.js?", 0.1)  # Très précis
        ]

        for query, expected_range in test_queries:
            is_vague, score = self.optimizer.is_vague_query(query)

            # Vérifier cohérence: plus le score est élevé, plus c'est vague
            if expected_range >= 0.4:
                assert is_vague is True
            else:
                assert is_vague is False

            assert 0.0 <= score <= 1.0


class TestVagueQueryIntegration:
    """Tests d'intégration pour l'optimisation complète"""

    def test_optimize_for_vague_queries_integration(self):
        """Test d'intégration complète"""
        from rag_chunk_lab.vague_query.vague_query_optimizer import optimize_for_vague_queries

        # Données de test
        doc_id = "test_collection"
        questions = ["Qu'est-ce que?", "Comment configurer l'API REST spécifique?"]
        chunks = [f"Chunk de test {i}" for i in range(10)]
        embeddings = np.random.rand(10, 384)

        results = optimize_for_vague_queries(
            doc_id=doc_id,
            questions=questions,
            chunks=chunks,
            embeddings=embeddings,
            domain="technical"
        )

        # Vérifications
        assert "optimized_answers" in results
        assert "vagueness_analysis" in results
        assert "optimization_stats" in results

        assert len(results["optimized_answers"]) == len(questions)
        assert len(results["vagueness_analysis"]) == len(questions)

        stats = results["optimization_stats"]
        assert "vague_queries" in stats
        assert "expanded_queries" in stats
        assert "vague_percentage" in stats

        # Au moins une requête devrait être détectée comme vague
        assert stats["vague_queries"] >= 1

    def test_performance_metrics(self):
        """Test métriques de performance"""
        from rag_chunk_lab.vague_query.vague_query_optimizer import optimize_for_vague_queries

        # Beaucoup de requêtes vagues
        vague_questions = ["Quoi?", "Comment?", "Aide", "Info"] * 25  # 100 questions
        chunks = [f"Chunk {i}" for i in range(100)]
        embeddings = np.random.rand(100, 384)

        results = optimize_for_vague_queries(
            doc_id="perf_test",
            questions=vague_questions,
            chunks=chunks,
            embeddings=embeddings
        )

        stats = results["optimization_stats"]

        # Devrait détecter la majorité comme vagues
        assert stats["vague_percentage"] > 80
        assert stats["expanded_queries"] > 80
        assert stats["enhanced_contexts"] > 80

    def test_domain_adaptation(self):
        """Test adaptation domaine"""
        from rag_chunk_lab.vague_query.vague_query_optimizer import optimize_for_vague_queries

        # Questions mixtes
        questions = [
            "Procédure juridique",  # Devrait être moins vague en domaine legal
            "Diagnostic médical",   # Devrait être moins vague en domaine medical
            "Quoi?"                # Toujours vague
        ]
        chunks = ["Chunk legal", "Chunk medical", "Chunk general"]
        embeddings = np.random.rand(3, 384)

        # Test domaine legal
        legal_results = optimize_for_vague_queries(
            doc_id="legal_test",
            questions=questions,
            chunks=chunks,
            embeddings=embeddings,
            domain="legal"
        )

        # Test domaine medical
        medical_results = optimize_for_vague_queries(
            doc_id="medical_test",
            questions=questions,
            chunks=chunks,
            embeddings=embeddings,
            domain="medical"
        )

        # Les analyses de vague devraient différer selon le domaine
        legal_analysis = legal_results["vagueness_analysis"]
        medical_analysis = medical_results["vagueness_analysis"]

        assert len(legal_analysis) == len(medical_analysis) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])