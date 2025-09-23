"""
Tests d'intégration pour le module generation.py
"""
import unittest
from unittest.mock import patch, MagicMock
from rag_chunk_lab.core.generation import (
    extractive_answer,
    llm_answer,
    get_azure_embedding,
    get_azure_embeddings_batch,
    get_azure_client,
    build_answer_payload
)


class TestGenerationIntegration(unittest.TestCase):
    """Tests d'intégration pour la génération de réponses"""

    def setUp(self):
        """Prépare les données de test"""
        self.test_passages = [
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
        self.test_question = "Quel est le délai de prescription ?"

    def test_extractive_answer(self):
        """Test de génération de réponse extractive"""
        answer = extractive_answer(self.test_question, self.test_passages, max_sentences=2)

        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)

        # Devrait contenir des mots-clés de la question
        answer_lower = answer.lower()
        self.assertIn('délai', answer_lower)
        self.assertIn('prescription', answer_lower)

    def test_extractive_answer_empty_passages(self):
        """Test de réponse extractive avec passages vides"""
        answer = extractive_answer(self.test_question, [], max_sentences=2)
        self.assertEqual(answer, '')

        # Test avec passages sans texte pertinent
        empty_passages = [{'text': '', 'meta': {}}]
        answer = extractive_answer(self.test_question, empty_passages, max_sentences=2)
        self.assertEqual(answer, '')

    def test_extractive_answer_fallback(self):
        """Test du fallback de réponse extractive"""
        # Passages sans phrases correspondantes
        no_match_passages = [
            {
                'text': 'Contenu totalement différent sans rapport avec la question.',
                'meta': {'doc_id': 'test'}
            }
        ]

        answer = extractive_answer(self.test_question, no_match_passages, max_sentences=2)

        # Devrait retourner le début du premier passage comme fallback
        self.assertTrue(len(answer) > 0)
        self.assertIn('Contenu totalement différent', answer)

    def test_llm_answer_mock_success(self):
        """Test de génération LLM (mocked success)"""
        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Le délai de prescription est de trois ans."
            mock_instance.chat.completions.create.return_value = mock_response

            answer = llm_answer(self.test_question, self.test_passages)

            self.assertEqual(answer, "Le délai de prescription est de trois ans.")
            mock_instance.chat.completions.create.assert_called_once()

    def test_llm_answer_error_handling(self):
        """Test de gestion d'erreur LLM"""
        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_client.side_effect = Exception("Connection error")

            answer = llm_answer(self.test_question, self.test_passages)

            self.assertIn("Erreur lors de l'appel Azure OpenAI", answer)

    def test_get_azure_embedding_mock(self):
        """Test d'embedding Azure (mocked)"""
        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.data[0].embedding = [0.1, 0.2, 0.3]
            mock_instance.embeddings.create.return_value = mock_response

            embedding = get_azure_embedding("test text")

            self.assertEqual(embedding, [0.1, 0.2, 0.3])
            mock_instance.embeddings.create.assert_called_once()

    def test_get_azure_embeddings_batch_mock(self):
        """Test d'embeddings Azure par batch (mocked)"""
        texts = ["text1", "text2", "text3"]

        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
                MagicMock(embedding=[0.5, 0.6])
            ]
            mock_instance.embeddings.create.return_value = mock_response

            embeddings = get_azure_embeddings_batch(texts, batch_size=3)

            self.assertEqual(len(embeddings), 3)
            self.assertEqual(embeddings[0], [0.1, 0.2])
            self.assertEqual(embeddings[1], [0.3, 0.4])
            self.assertEqual(embeddings[2], [0.5, 0.6])

    def test_get_azure_embeddings_batch_multiple_batches(self):
        """Test d'embeddings Azure avec plusieurs batches"""
        texts = [f"text{i}" for i in range(5)]  # 5 textes, batch_size=2

        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            # Premier batch (2 textes)
            mock_response1 = MagicMock()
            mock_response1.data = [
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4])
            ]

            # Deuxième batch (2 textes)
            mock_response2 = MagicMock()
            mock_response2.data = [
                MagicMock(embedding=[0.5, 0.6]),
                MagicMock(embedding=[0.7, 0.8])
            ]

            # Troisième batch (1 texte)
            mock_response3 = MagicMock()
            mock_response3.data = [
                MagicMock(embedding=[0.9, 1.0])
            ]

            mock_instance.embeddings.create.side_effect = [mock_response1, mock_response2, mock_response3]

            embeddings = get_azure_embeddings_batch(texts, batch_size=2)

            # Vérifier que 3 appels ont été faits
            self.assertEqual(mock_instance.embeddings.create.call_count, 3)
            # Vérifier que tous les embeddings sont retournés
            self.assertEqual(len(embeddings), 5)

    def test_get_azure_client_singleton(self):
        """Test que get_azure_client utilise bien le singleton pattern"""
        with patch('rag_chunk_lab.generation.AZURE_CONFIG') as mock_config:
            mock_config.api_key = 'test'
            mock_config.endpoint = 'https://test.com'
            mock_config.api_version = '2024-02-15-preview'

            with patch('rag_chunk_lab.generation.AzureOpenAI') as mock_azure:
                mock_instance = MagicMock()
                mock_azure.return_value = mock_instance

                # Premier appel
                client1 = get_azure_client()
                # Deuxième appel
                client2 = get_azure_client()

                # AzureOpenAI ne doit être instancié qu'une fois
                mock_azure.assert_called_once()
                self.assertEqual(client1, client2)

    def test_build_answer_payload_extractive(self):
        """Test de construction de payload de réponse extractive"""
        candidates = [
            {
                'score': 0.95,
                'text': 'Texte de test avec information pertinente.',
                'meta': {
                    'doc_id': 'test_doc',
                    'page': 1,
                    'start': 0,
                    'end': 10,
                    'section_title': 'Section 1',
                    'source_file': 'test.pdf'
                }
            }
        ]

        payload = build_answer_payload(
            pipeline='test_pipeline',
            question=self.test_question,
            candidates=candidates,
            max_sentences=2,
            use_llm=False
        )

        # Vérifier la structure
        self.assertIn('pipeline', payload)
        self.assertIn('answer', payload)
        self.assertIn('sources', payload)

        self.assertEqual(payload['pipeline'], 'test_pipeline')
        self.assertIsInstance(payload['answer'], str)
        self.assertIsInstance(payload['sources'], list)

        # Vérifier les sources
        source = payload['sources'][0]
        self.assertIn('score', source)
        self.assertIn('doc_id', source)
        self.assertIn('page', source)
        self.assertIn('snippet', source)

    def test_build_answer_payload_llm(self):
        """Test de construction de payload de réponse LLM"""
        candidates = [
            {
                'score': 0.95,
                'text': 'Texte de test avec information pertinente.',
                'meta': {
                    'doc_id': 'test_doc',
                    'page': 1,
                    'start': 0,
                    'end': 10,
                    'section_title': 'Section 1',
                    'source_file': 'test.pdf'
                }
            }
        ]

        with patch('rag_chunk_lab.generation.llm_answer') as mock_llm:
            mock_llm.return_value = "Réponse générée par LLM"

            payload = build_answer_payload(
                pipeline='test_pipeline',
                question=self.test_question,
                candidates=candidates,
                max_sentences=2,
                use_llm=True
            )

            mock_llm.assert_called_once_with(self.test_question, candidates)
            self.assertEqual(payload['answer'], "Réponse générée par LLM")

    def test_error_handling_missing_config(self):
        """Test de gestion d'erreur pour configuration manquante"""
        with patch('rag_chunk_lab.generation.AZURE_CONFIG') as mock_config:
            mock_config.api_key = ''
            mock_config.endpoint = ''

            with self.assertRaises(ValueError):
                get_azure_client()

    def test_batch_empty_texts(self):
        """Test de batch avec liste vide"""
        result = get_azure_embeddings_batch([])
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()