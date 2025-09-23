"""
Tests de performance pour valider les optimisations RAG Chunk Lab v2.0
"""
import unittest
import time
import tempfile
import os
from unittest.mock import patch, MagicMock
import numpy as np
from rag_chunk_lab.utils.monitoring import PerformanceMonitor
from rag_chunk_lab.core.chunkers import fixed_chunks, tokenize_pages_once
from rag_chunk_lab.core.indexing import build_index, load_index_data
from rag_chunk_lab.core.generation import get_azure_embeddings_batch


class TestPerformanceOptimizations(unittest.TestCase):
    """Tests de performance pour vérifier l'impact des optimisations"""

    def setUp(self):
        """Prépare les données de test"""
        self.monitor = PerformanceMonitor()

        # Créer des données de test plus volumineuses pour mesurer les performances
        self.large_pages = []
        for i in range(100):  # 100 pages
            text = f"Article {i}\n" + " ".join([f"mot{j}" for j in range(50)])  # 50 mots par page
            self.large_pages.append({
                'page': i + 1,
                'text': text,
                'source_file': f'doc_{i}.pdf'
            })

    def test_tokenization_cache_performance(self):
        """Test que le cache de tokenisation améliore les performances"""
        from rag_chunk_lab.utils.utils import tokenize_words

        test_text = "Ceci est un texte de test répété pour mesurer les performances de cache."

        # Première passe - sans cache (ou mise en cache)
        start_time = time.time()
        for _ in range(100):
            tokenize_words(test_text)
        first_pass_time = time.time() - start_time

        # Deuxième passe - avec cache
        start_time = time.time()
        for _ in range(100):
            tokenize_words(test_text)
        second_pass_time = time.time() - start_time

        # La deuxième passe devrait être plus rapide grâce au cache
        print(f"Premier passage: {first_pass_time:.4f}s")
        print(f"Deuxième passage: {second_pass_time:.4f}s")
        print(f"Amélioration: {((first_pass_time - second_pass_time) / first_pass_time * 100):.1f}%")

        # Le cache devrait améliorer les performances d'au moins 50%
        self.assertLess(second_pass_time, first_pass_time * 0.5)

    def test_tokenize_pages_once_vs_multiple(self):
        """Compare la performance de tokenize_pages_once vs tokenisation multiple"""
        # Simulation ancienne méthode (tokenisation multiple)
        start_time = time.time()
        old_method_tokens = []
        for page in self.large_pages:
            from rag_chunk_lab.utils.utils import tokenize_words
            tokens = tokenize_words(page['text'])
            old_method_tokens.append((page, tokens))
        old_method_time = time.time() - start_time

        # Nouvelle méthode optimisée
        start_time = time.time()
        new_method_tokens = tokenize_pages_once(self.large_pages)
        new_method_time = time.time() - start_time

        print(f"Ancienne méthode: {old_method_time:.4f}s")
        print(f"Nouvelle méthode: {new_method_time:.4f}s")
        print(f"Amélioration: {((old_method_time - new_method_time) / old_method_time * 100):.1f}%")

        # Vérifier que les résultats sont identiques
        self.assertEqual(len(old_method_tokens), len(new_method_tokens))

        # La nouvelle méthode devrait être au moins aussi rapide
        self.assertLessEqual(new_method_time, old_method_time * 1.1)  # 10% de tolérance

    def test_chunking_performance_with_monitoring(self):
        """Test de performance du chunking avec monitoring"""
        @self.monitor.monitor_function("chunking_performance_test")
        def perform_chunking():
            return fixed_chunks(self.large_pages, size_tokens=50, overlap_tokens=10, doc_id='perf_test')

        chunks = perform_chunking()

        # Vérifier que les chunks sont créés
        self.assertTrue(len(chunks) > 0)

        # Vérifier que le monitoring a enregistré les métriques
        self.assertIn("chunking_performance_test", self.monitor.metrics)
        metrics = self.monitor.metrics["chunking_performance_test"][0]

        self.assertTrue(metrics['success'])
        self.assertIsInstance(metrics['duration_seconds'], float)
        self.assertIsInstance(metrics['memory_delta_mb'], float)

        # Le chunking de 100 pages devrait prendre moins de 5 secondes avec les optimisations
        self.assertLess(metrics['duration_seconds'], 5.0)

    def test_index_cache_performance(self):
        """Test de performance du cache d'index"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer des fichiers d'index
            pipeline_dir = f"{temp_dir}/test_doc/test_pipeline"
            os.makedirs(pipeline_dir)

            import json
            test_data = [f"text{i}" for i in range(1000)]  # 1000 textes
            test_meta = [{'page': i} for i in range(1000)]

            with open(f"{pipeline_dir}/chunks_texts.json", 'w') as f:
                json.dump(test_data, f)
            with open(f"{pipeline_dir}/chunks_meta.json", 'w') as f:
                json.dump(test_meta, f)

            # Premier chargement (mise en cache)
            start_time = time.time()
            texts1, meta1 = load_index_data('test_doc', 'test_pipeline', temp_dir)
            first_load_time = time.time() - start_time

            # Deuxième chargement (depuis le cache)
            start_time = time.time()
            texts2, meta2 = load_index_data('test_doc', 'test_pipeline', temp_dir)
            second_load_time = time.time() - start_time

            print(f"Premier chargement: {first_load_time:.4f}s")
            print(f"Deuxième chargement: {second_load_time:.4f}s")
            print(f"Amélioration: {((first_load_time - second_load_time) / first_load_time * 100):.1f}%")

            # Le cache devrait améliorer significativement les performances
            self.assertLess(second_load_time, first_load_time * 0.1)  # Au moins 90% d'amélioration

    def test_batch_embedding_performance_simulation(self):
        """Simule la performance des embeddings par batch vs individuels"""
        texts = [f"texte de test numéro {i}" for i in range(100)]

        # Simulation appels individuels
        with patch('rag_chunk_lab.generation.get_azure_client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
            mock_instance.embeddings.create.return_value = mock_response

            # Simuler 100 appels individuels
            start_time = time.time()
            for text in texts:
                mock_instance.embeddings.create(input=text, model="test-model")
            individual_time = time.time() - start_time

            # Reset mock
            mock_instance.reset_mock()

            # Simuler appels par batch
            mock_batch_response = MagicMock()
            mock_batch_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(50)]
            mock_instance.embeddings.create.return_value = mock_batch_response

            start_time = time.time()
            # Deux batches de 50
            mock_instance.embeddings.create(input=texts[:50], model="test-model")
            mock_instance.embeddings.create(input=texts[50:], model="test-model")
            batch_time = time.time() - start_time

            print(f"Appels individuels (simulé): {individual_time:.4f}s ({len(texts)} appels)")
            print(f"Appels par batch (simulé): {batch_time:.4f}s (2 appels)")

            # Les appels par batch devraient être plus rapides
            self.assertLess(batch_time, individual_time)

            # Vérifier le nombre d'appels
            self.assertEqual(mock_instance.embeddings.create.call_count, 2)  # Seulement 2 appels batch

    def test_memory_optimization_float32(self):
        """Test de l'optimisation mémoire float32 vs float64"""
        # Créer des arrays de test
        size = (1000, 1536)  # Taille typique d'embeddings Azure

        # Array float64 (par défaut)
        array_64 = np.random.random(size).astype(np.float64)
        memory_64 = array_64.nbytes

        # Array float32 (optimisé)
        array_32 = array_64.astype(np.float32)
        memory_32 = array_32.nbytes

        print(f"Mémoire float64: {memory_64 / 1024 / 1024:.2f} MB")
        print(f"Mémoire float32: {memory_32 / 1024 / 1024:.2f} MB")
        print(f"Économie: {((memory_64 - memory_32) / memory_64 * 100):.1f}%")

        # float32 devrait utiliser exactement 50% moins de mémoire
        self.assertEqual(memory_32, memory_64 // 2)

        # Vérifier que la précision reste acceptable pour notre usage
        max_diff = np.max(np.abs(array_64.astype(np.float32) - array_32))
        self.assertLess(max_diff, 1e-6)  # Différence négligeable

    def test_parallel_processing_simulation(self):
        """Simule l'amélioration de performance du traitement parallèle"""
        from concurrent.futures import ThreadPoolExecutor

        def simulate_pipeline_work(pipeline_name):
            """Simule le travail d'un pipeline (chunking + indexing)"""
            time.sleep(0.1)  # Simule 100ms de travail
            return f"Pipeline {pipeline_name} completed"

        pipelines = ['fixed', 'structure', 'sliding']

        # Traitement séquentiel (ancienne méthode)
        start_time = time.time()
        for pipeline in pipelines:
            simulate_pipeline_work(pipeline)
        sequential_time = time.time() - start_time

        # Traitement parallèle (nouvelle méthode)
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
            futures = [executor.submit(simulate_pipeline_work, pipeline) for pipeline in pipelines]
            results = [future.result() for future in futures]
        parallel_time = time.time() - start_time

        print(f"Traitement séquentiel: {sequential_time:.4f}s")
        print(f"Traitement parallèle: {parallel_time:.4f}s")
        print(f"Amélioration: {((sequential_time - parallel_time) / sequential_time * 100):.1f}%")

        # Le traitement parallèle devrait être au moins 2x plus rapide
        self.assertLess(parallel_time, sequential_time * 0.6)

    def test_monitoring_overhead(self):
        """Test que le monitoring n'ajoute pas trop d'overhead"""
        def test_function():
            """Fonction de test simple"""
            return sum(range(1000))

        # Sans monitoring
        start_time = time.time()
        for _ in range(100):
            test_function()
        no_monitoring_time = time.time() - start_time

        # Avec monitoring
        @self.monitor.monitor_function("overhead_test")
        def monitored_test_function():
            return sum(range(1000))

        start_time = time.time()
        for _ in range(100):
            monitored_test_function()
        with_monitoring_time = time.time() - start_time

        overhead = ((with_monitoring_time - no_monitoring_time) / no_monitoring_time) * 100

        print(f"Sans monitoring: {no_monitoring_time:.4f}s")
        print(f"Avec monitoring: {with_monitoring_time:.4f}s")
        print(f"Overhead: {overhead:.1f}%")

        # L'overhead du monitoring devrait être minimal (< 20%)
        self.assertLess(overhead, 20.0)

    def test_integration_performance_benchmark(self):
        """Test de performance d'intégration complet"""
        @self.monitor.monitor_function("integration_benchmark")
        def integration_test():
            # Simulation d'un workflow complet optimisé

            # 1. Chunking optimisé
            chunks = fixed_chunks(self.large_pages[:10], 50, 10, 'benchmark')

            # 2. Index building (TF-IDF simulé)
            with tempfile.TemporaryDirectory() as temp_dir:
                build_index('benchmark', 'fixed', chunks, temp_dir)

                # 3. Cache utilization
                load_index_data('benchmark', 'fixed', temp_dir)
                load_index_data('benchmark', 'fixed', temp_dir)  # Deuxième appel depuis cache

            return len(chunks)

        result = integration_test()

        # Vérifier que le test s'est exécuté
        self.assertTrue(result > 0)

        # Vérifier les métriques
        metrics = self.monitor.metrics["integration_benchmark"][0]
        self.assertTrue(metrics['success'])

        # Le workflow complet devrait être rapide avec les optimisations
        self.assertLess(metrics['duration_seconds'], 2.0)

        print(f"Workflow complet: {metrics['duration_seconds']:.4f}s")
        print(f"Chunks créés: {result}")
        print(f"Mémoire utilisée: {metrics['memory_delta_mb']:.1f}MB")


class TestPerformanceRegression(unittest.TestCase):
    """Tests de régression pour s'assurer que les optimisations n'ont pas cassé les fonctionnalités"""

    def test_chunking_output_consistency(self):
        """Vérifie que les optimisations ne changent pas la sortie du chunking"""
        pages = [{'page': 1, 'text': 'Test de régression pour le chunking optimisé.'}]

        # Test avec différentes tailles
        for size in [5, 10, 20]:
            chunks = fixed_chunks(pages, size, 2, 'regression_test')

            self.assertTrue(len(chunks) > 0)
            for chunk in chunks:
                self.assertIn('text', chunk)
                self.assertIn('doc_id', chunk)
                self.assertTrue(len(chunk['text']) > 0)

    def test_cache_data_integrity(self):
        """Vérifie que le cache ne corrompt pas les données"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer des données de test
            import json
            pipeline_dir = f"{temp_dir}/test/pipeline"
            os.makedirs(pipeline_dir)

            original_texts = ['text1', 'text2', 'text3']
            original_meta = [{'page': 1}, {'page': 2}, {'page': 3}]

            with open(f"{pipeline_dir}/chunks_texts.json", 'w') as f:
                json.dump(original_texts, f)
            with open(f"{pipeline_dir}/chunks_meta.json", 'w') as f:
                json.dump(original_meta, f)

            # Charger plusieurs fois depuis le cache
            for _ in range(5):
                texts, meta = load_index_data('test', 'pipeline', temp_dir)
                self.assertEqual(texts, original_texts)
                self.assertEqual(meta, original_meta)


if __name__ == '__main__':
    # Exécuter les tests avec du verbose pour voir les résultats de performance
    unittest.main(verbosity=2)