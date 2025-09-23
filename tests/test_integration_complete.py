"""
Test d'intégration complet pour valider le fonctionnement end-to-end
"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from rag_chunk_lab.cli import ingest, ask
from rag_chunk_lab.utils.utils import load_document
from rag_chunk_lab.core.chunkers import fixed_chunks
from rag_chunk_lab.core.indexing import build_index, retrieve
from rag_chunk_lab.core.generation import extractive_answer


class TestCompleteIntegration(unittest.TestCase):
    """Test d'intégration complet du workflow RAG"""

    def setUp(self):
        """Prépare un document de test complet"""
        self.test_content = """
Article 1
Ceci est le premier article du code de test. Il établit les principes fondamentaux
du système de gestion des documents juridiques automatisés.

Section 2
Les délais de prescription sont fixés à trois ans pour toutes les contraventions
de première classe. Cette durée commence à courir à partir de la date de commission
de l'infraction.

Article 3
Les sanctions applicables comprennent des amendes pouvant aller de 100 à 1500 euros
selon la gravité des faits constatés. Le tribunal compétent détermine le montant
en fonction des circonstances.

Conclusion
Le présent code entre en vigueur immédiatement et remplace toutes dispositions
antérieures contraires. Les autorités compétentes sont chargées de son application.
        """.strip()

    def test_complete_workflow_integration(self):
        """Test du workflow complet: document -> chunking -> indexing -> recherche -> réponse"""

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Créer un document de test
            doc_path = os.path.join(temp_dir, 'test_doc.txt')
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(self.test_content)

            # 2. Charger le document
            pages = load_document(doc_path)
            self.assertTrue(len(pages) > 0)
            self.assertIn('Article 1', pages[0]['text'])

            # 3. Créer des chunks
            chunks = fixed_chunks(pages, size_tokens=50, overlap_tokens=10, doc_id='test_integration')
            self.assertTrue(len(chunks) > 0)

            # Vérifier que les chunks contiennent le contenu attendu
            all_text = ' '.join([chunk['text'] for chunk in chunks])
            self.assertIn('délai', all_text.lower())
            self.assertIn('prescription', all_text.lower())

            # 4. Construire l'index
            data_dir = os.path.join(temp_dir, 'data')
            build_index('test_integration', 'fixed', chunks, data_dir)

            # Vérifier que les fichiers d'index sont créés
            index_dir = os.path.join(data_dir, 'test_integration', 'fixed')
            self.assertTrue(os.path.exists(os.path.join(index_dir, 'chunks_texts.json')))
            self.assertTrue(os.path.exists(os.path.join(index_dir, 'tfidf_vectorizer.joblib')))

            # 5. Effectuer une recherche
            results = retrieve('test_integration', 'fixed', 'délai prescription', top_k=3, data_dir=data_dir)
            self.assertTrue(len(results) > 0)

            # Vérifier que les résultats pertinents sont trouvés
            found_prescription = False
            for result in results:
                if 'prescription' in result['text'].lower():
                    found_prescription = True
                    break
            self.assertTrue(found_prescription, "Le terme 'prescription' devrait être trouvé dans les résultats")

            # 6. Générer une réponse extractive
            answer = extractive_answer('Quel est le délai de prescription ?', results, max_sentences=2)
            self.assertTrue(len(answer) > 0)
            self.assertIn('trois ans', answer.lower())

    def test_workflow_with_structure_aware_chunking(self):
        """Test du workflow avec chunking structure-aware"""

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, 'structured_doc.txt')
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(self.test_content)

            pages = load_document(doc_path)

            # Utiliser structure-aware chunking
            from rag_chunk_lab.core.chunkers import structure_aware_chunks
            chunks = structure_aware_chunks(pages, size_tokens=50, overlap_tokens=10, doc_id='structured_test')

            # Vérifier que les sections sont détectées
            section_titles = [chunk.get('section_title') for chunk in chunks]
            self.assertTrue(any(title is not None for title in section_titles))

            # Continuer le workflow
            data_dir = os.path.join(temp_dir, 'data')
            build_index('structured_test', 'structure', chunks, data_dir)

            results = retrieve('structured_test', 'structure', 'sanctions amendes', top_k=3, data_dir=data_dir)
            self.assertTrue(len(results) > 0)

    def test_workflow_error_handling(self):
        """Test de la gestion d'erreur dans le workflow"""

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = os.path.join(temp_dir, 'data')

            # Test avec document inexistant
            with self.assertRaises(FileNotFoundError):
                retrieve('nonexistent', 'fixed', 'query', top_k=3, data_dir=data_dir)

            # Test avec pipeline inexistant
            with self.assertRaises(FileNotFoundError):
                retrieve('test', 'nonexistent_pipeline', 'query', top_k=3, data_dir=data_dir)

    def test_performance_with_optimizations(self):
        """Test que les optimisations améliorent bien les performances"""

        import time
        from rag_chunk_lab.utils.monitoring import PerformanceMonitor

        monitor = PerformanceMonitor()

        @monitor.monitor_function("integration_performance_test")
        def run_workflow():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Créer un document plus volumineux
                large_content = self.test_content * 10  # 10x plus grand

                doc_path = os.path.join(temp_dir, 'large_doc.txt')
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(large_content)

                pages = load_document(doc_path)
                chunks = fixed_chunks(pages, size_tokens=30, overlap_tokens=5, doc_id='perf_test')

                data_dir = os.path.join(temp_dir, 'data')
                build_index('perf_test', 'fixed', chunks, data_dir)

                # Test multiple searches (pour tester le cache)
                for query in ['délai', 'sanctions', 'article']:
                    results = retrieve('perf_test', 'fixed', query, top_k=5, data_dir=data_dir)
                    self.assertTrue(len(results) > 0)

                return len(chunks)

        chunks_created = run_workflow()

        # Vérifier que le workflow s'est exécuté
        self.assertTrue(chunks_created > 0)

        # Vérifier les métriques de performance
        metrics = monitor.metrics["integration_performance_test"][0]
        self.assertTrue(metrics['success'])

        # Le workflow optimisé devrait être rapide
        print(f"Workflow performance test: {metrics['duration_seconds']:.3f}s for {chunks_created} chunks")
        self.assertLess(metrics['duration_seconds'], 5.0)  # Moins de 5 secondes

    def test_cache_effectiveness(self):
        """Test de l'efficacité du cache dans un workflow réel"""

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, 'cache_test.txt')
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(self.test_content)

            pages = load_document(doc_path)
            chunks = fixed_chunks(pages, size_tokens=30, overlap_tokens=5, doc_id='cache_test')

            data_dir = os.path.join(temp_dir, 'data')
            build_index('cache_test', 'fixed', chunks, data_dir)

            # Premier groupe de recherches (mise en cache)
            start_time = time.time()
            for _ in range(5):
                results = retrieve('cache_test', 'fixed', 'délai prescription', top_k=3, data_dir=data_dir)
            first_batch_time = time.time() - start_time

            # Deuxième groupe de recherches (utilisation du cache)
            start_time = time.time()
            for _ in range(5):
                results = retrieve('cache_test', 'fixed', 'délai prescription', top_k=3, data_dir=data_dir)
            second_batch_time = time.time() - start_time

            print(f"Premier batch: {first_batch_time:.3f}s")
            print(f"Deuxième batch: {second_batch_time:.3f}s")

            # Le cache devrait améliorer les performances
            self.assertLessEqual(second_batch_time, first_batch_time)

    def test_multiple_pipelines_integration(self):
        """Test d'intégration avec plusieurs pipelines"""

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_path = os.path.join(temp_dir, 'multi_pipeline_test.txt')
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(self.test_content)

            pages = load_document(doc_path)
            data_dir = os.path.join(temp_dir, 'data')

            # Créer plusieurs pipelines
            pipelines = {
                'fixed': fixed_chunks,
                'structure': 'rag_chunk_lab.chunkers.structure_aware_chunks',
                'sliding': 'rag_chunk_lab.chunkers.sliding_window_chunks'
            }

            from rag_chunk_lab.core.chunkers import structure_aware_chunks, sliding_window_chunks

            # Tester chaque pipeline
            for pipeline_name in ['fixed', 'structure', 'sliding']:
                if pipeline_name == 'fixed':
                    chunks = fixed_chunks(pages, 30, 5, f'multi_test_{pipeline_name}')
                elif pipeline_name == 'structure':
                    chunks = structure_aware_chunks(pages, 30, 5, f'multi_test_{pipeline_name}')
                elif pipeline_name == 'sliding':
                    chunks = sliding_window_chunks(pages, 25, 10, f'multi_test_{pipeline_name}')

                build_index(f'multi_test_{pipeline_name}', pipeline_name, chunks, data_dir)

                # Test de recherche pour chaque pipeline
                results = retrieve(f'multi_test_{pipeline_name}', pipeline_name, 'délai', top_k=3, data_dir=data_dir)
                self.assertTrue(len(results) > 0, f"Pipeline {pipeline_name} should return results")

    def test_memory_usage_optimization(self):
        """Test que les optimisations réduisent bien l'utilisation mémoire"""

        import psutil
        import gc

        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss

        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer plusieurs documents pour tester l'utilisation mémoire
            for i in range(10):
                doc_path = os.path.join(temp_dir, f'memory_test_{i}.txt')
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(self.test_content * 2)  # Documents plus volumineux

                pages = load_document(doc_path)
                chunks = fixed_chunks(pages, 25, 5, f'memory_test_{i}')

                data_dir = os.path.join(temp_dir, 'data')
                build_index(f'memory_test_{i}', 'fixed', chunks, data_dir)

        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        print(f"Augmentation mémoire: {memory_increase:.1f} MB pour 10 documents")

        # Avec les optimisations, l'augmentation mémoire devrait être raisonnable
        self.assertLess(memory_increase, 100)  # Moins de 100 MB pour 10 documents


if __name__ == '__main__':
    unittest.main(verbosity=2)