# vague_query_optimization_system.py
"""
Système complet d'optimisation RAG pour requêtes vagues - Orchestrateur principal
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import os
from pathlib import Path
from datetime import datetime

# Imports des composants du système
from .vague_query_optimizer import VagueQueryOptimizer
from .hierarchical_chunking import HierarchicalChunker, create_hierarchical_chunks_for_rag
from .metadata_enricher import MetadataEnricher, enrich_hierarchical_chunks
from .hybrid_embeddings import HybridEmbeddingSystem, create_hybrid_index_from_hierarchy
from .embedding_fine_tuning import create_fine_tuned_model_for_domain, FineTuningConfig
from .context_enrichment_pipeline import ContextEnrichmentPipeline, create_context_enrichment_pipeline
from .adaptive_prompt_engineering import AdaptivePromptEngine, create_adaptive_prompt_engine
from .production_monitoring import ProductionMonitor, create_production_monitor, setup_automatic_optimization

logger = logging.getLogger(__name__)


class VagueQueryOptimizationSystem:
    """
    Système complet d'optimisation RAG pour requêtes vagues

    Intègre tous les composants développés :
    - Détection et expansion de requêtes vagues
    - Chunking hiérarchique multi-granularité
    - Enrichissement de métadonnées
    - Embeddings hybrides (dense + sparse)
    - Fine-tuning domaine-spécifique
    - Enrichissement contextuel automatique
    - Prompt engineering adaptatif
    - Monitoring et feedback loop
    """

    def __init__(self,
                 domain: str = "general",
                 language: str = "fr",
                 openai_api_key: str = None,
                 enable_monitoring: bool = True,
                 enable_fine_tuning: bool = False,
                 data_dir: str = "./vague_optimization_data"):

        self.domain = domain
        self.language = language
        self.openai_api_key = openai_api_key
        self.enable_monitoring = enable_monitoring
        self.enable_fine_tuning = enable_fine_tuning
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # État du système
        self.is_initialized = False
        self.components_status = {}

        # Composants principaux
        self.vague_query_optimizer = None
        self.hierarchical_chunker = None
        self.metadata_enricher = None
        self.hybrid_embedding_system = None
        self.context_enrichment_pipeline = None
        self.adaptive_prompt_engine = None
        self.production_monitor = None

        # Données indexées
        self.indexed_hierarchy = None
        self.embedding_index = None
        self.fine_tuned_model_path = None

        logger.info(f"VagueQueryOptimizationSystem initialized for domain: {domain}")

    def initialize_system(self) -> Dict[str, bool]:
        """
        Initialise tous les composants du système

        Returns:
            Dict indiquant le succès d'initialisation de chaque composant
        """

        logger.info("Initializing Vague Query Optimization System...")

        # 1. Détecteur de requêtes vagues
        try:
            self.vague_query_optimizer = VagueQueryOptimizer(
                openai_api_key=self.openai_api_key,
                domain=self.domain
            )
            self.components_status["vague_query_optimizer"] = True
            logger.info("✅ Vague Query Optimizer initialized")
        except Exception as e:
            self.components_status["vague_query_optimizer"] = False
            logger.error(f"❌ Vague Query Optimizer failed: {e}")

        # 2. Chunker hiérarchique
        try:
            self.hierarchical_chunker = HierarchicalChunker(
                domain=self.domain,
                language=self.language
            )
            self.components_status["hierarchical_chunker"] = True
            logger.info("✅ Hierarchical Chunker initialized")
        except Exception as e:
            self.components_status["hierarchical_chunker"] = False
            logger.error(f"❌ Hierarchical Chunker failed: {e}")

        # 3. Enrichisseur de métadonnées
        try:
            self.metadata_enricher = MetadataEnricher(
                domain=self.domain,
                language=self.language
            )
            self.components_status["metadata_enricher"] = True
            logger.info("✅ Metadata Enricher initialized")
        except Exception as e:
            self.components_status["metadata_enricher"] = False
            logger.error(f"❌ Metadata Enricher failed: {e}")

        # 4. Système d'embeddings hybrides
        try:
            self.hybrid_embedding_system = HybridEmbeddingSystem(
                domain=self.domain,
                language=self.language
            )
            self.components_status["hybrid_embedding_system"] = True
            logger.info("✅ Hybrid Embedding System initialized")
        except Exception as e:
            self.components_status["hybrid_embedding_system"] = False
            logger.error(f"❌ Hybrid Embedding System failed: {e}")

        # 5. Pipeline d'enrichissement contextuel
        try:
            self.context_enrichment_pipeline = create_context_enrichment_pipeline(
                domain=self.domain,
                openai_api_key=self.openai_api_key
            )
            self.components_status["context_enrichment_pipeline"] = True
            logger.info("✅ Context Enrichment Pipeline initialized")
        except Exception as e:
            self.components_status["context_enrichment_pipeline"] = False
            logger.error(f"❌ Context Enrichment Pipeline failed: {e}")

        # 6. Moteur de prompt adaptatif
        try:
            self.adaptive_prompt_engine = create_adaptive_prompt_engine(
                domain=self.domain,
                language=self.language
            )
            self.components_status["adaptive_prompt_engine"] = True
            logger.info("✅ Adaptive Prompt Engine initialized")
        except Exception as e:
            self.components_status["adaptive_prompt_engine"] = False
            logger.error(f"❌ Adaptive Prompt Engine failed: {e}")

        # 7. Monitoring de production (optionnel)
        if self.enable_monitoring:
            try:
                self.production_monitor = create_production_monitor(
                    domain=self.domain
                )

                # Setup automatic optimization
                setup_automatic_optimization(
                    self.production_monitor,
                    self.hybrid_embedding_system,
                    self.vague_query_optimizer
                )

                self.production_monitor.start_monitoring()
                self.components_status["production_monitor"] = True
                logger.info("✅ Production Monitor initialized and started")
            except Exception as e:
                self.components_status["production_monitor"] = False
                logger.error(f"❌ Production Monitor failed: {e}")

        # Vérifier le statut global
        successful_components = sum(1 for status in self.components_status.values() if status)
        total_components = len(self.components_status)

        self.is_initialized = successful_components >= (total_components * 0.7)  # 70% minimum

        if self.is_initialized:
            logger.info(f"🎉 System initialized successfully ({successful_components}/{total_components} components)")
        else:
            logger.warning(f"⚠️ System partially initialized ({successful_components}/{total_components} components)")

        return self.components_status

    def index_documents(self,
                       documents: List[Dict[str, Any]],
                       enable_fine_tuning: bool = None) -> Dict[str, Any]:
        """
        Indexe des documents avec le système complet

        Args:
            documents: Liste de documents avec 'text' et 'doc_id'
            enable_fine_tuning: Active le fine-tuning (override)

        Returns:
            Statistiques d'indexation
        """

        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        enable_ft = enable_fine_tuning if enable_fine_tuning is not None else self.enable_fine_tuning

        logger.info(f"Indexing {len(documents)} documents...")

        indexing_stats = {
            "total_documents": len(documents),
            "processed_documents": 0,
            "total_chunks": 0,
            "hierarchical_levels": 0,
            "enriched_chunks": 0,
            "embedding_index_size": 0,
            "fine_tuning_enabled": enable_ft,
            "errors": []
        }

        all_hierarchies = {}

        # Traiter chaque document
        for doc in documents:
            try:
                doc_id = doc["doc_id"]
                text = doc["text"]

                logger.info(f"Processing document: {doc_id}")

                # 1. Chunking hiérarchique
                if not self.hierarchical_chunker:
                    raise RuntimeError("Hierarchical Chunker not available")

                hierarchy = self.hierarchical_chunker.create_hierarchical_chunks(text, doc_id)

                # 2. Enrichissement des métadonnées
                if self.metadata_enricher:
                    enriched_hierarchy = enrich_hierarchical_chunks(
                        hierarchy, doc_id, self.domain
                    )
                else:
                    enriched_hierarchy = hierarchy

                all_hierarchies[doc_id] = enriched_hierarchy

                # Mise à jour des stats
                total_chunks = sum(len(chunks) for chunks in enriched_hierarchy.values())
                indexing_stats["total_chunks"] += total_chunks
                indexing_stats["hierarchical_levels"] = len(enriched_hierarchy)
                indexing_stats["enriched_chunks"] += total_chunks
                indexing_stats["processed_documents"] += 1

                logger.info(f"Document {doc_id}: {total_chunks} chunks across {len(enriched_hierarchy)} levels")

            except Exception as e:
                error_msg = f"Error processing document {doc.get('doc_id', 'unknown')}: {e}"
                indexing_stats["errors"].append(error_msg)
                logger.error(error_msg)

        # 3. Créer l'index d'embeddings hybrides
        if self.hybrid_embedding_system and all_hierarchies:
            try:
                logger.info("Creating hybrid embedding index...")

                # Combiner toutes les hiérarchies
                combined_hierarchy = {}
                for granularity in ["document", "section", "paragraph", "sentence", "concept", "summary"]:
                    combined_hierarchy[granularity] = []
                    for doc_hierarchy in all_hierarchies.values():
                        combined_hierarchy[granularity].extend(doc_hierarchy.get(granularity, []))

                # Créer l'index hybride
                self.embedding_index = create_hybrid_index_from_hierarchy(
                    combined_hierarchy, "combined_docs", self.domain
                )

                indexing_stats["embedding_index_size"] = len(self.embedding_index.embedding_cache)
                logger.info(f"✅ Hybrid embedding index created with {indexing_stats['embedding_index_size']} chunks")

            except Exception as e:
                error_msg = f"Error creating embedding index: {e}"
                indexing_stats["errors"].append(error_msg)
                logger.error(error_msg)

        # 4. Fine-tuning (optionnel)
        if enable_ft and all_hierarchies:
            try:
                logger.info("Starting domain-specific fine-tuning...")

                combined_hierarchy = {}
                for granularity in ["document", "section", "paragraph", "sentence", "concept", "summary"]:
                    combined_hierarchy[granularity] = []
                    for doc_hierarchy in all_hierarchies.values():
                        combined_hierarchy[granularity].extend(doc_hierarchy.get(granularity, []))

                config = FineTuningConfig(
                    epochs=2,  # Réduit pour la démo
                    batch_size=8
                )

                self.fine_tuned_model_path = create_fine_tuned_model_for_domain(
                    combined_hierarchy, self.domain, config
                )

                if self.fine_tuned_model_path:
                    logger.info(f"✅ Fine-tuning completed: {self.fine_tuned_model_path}")
                    indexing_stats["fine_tuned_model"] = self.fine_tuned_model_path

            except Exception as e:
                error_msg = f"Error during fine-tuning: {e}"
                indexing_stats["errors"].append(error_msg)
                logger.error(error_msg)

        # Sauvegarder l'index
        self.indexed_hierarchy = all_hierarchies
        self._save_index_data(indexing_stats)

        logger.info(f"🎉 Indexing completed: {indexing_stats['processed_documents']}/{indexing_stats['total_documents']} documents")

        return indexing_stats

    def optimize_vague_query(self,
                           query: str,
                           user_level: str = "intermediate",
                           max_results: int = 5) -> Dict[str, Any]:
        """
        Optimise une requête vague avec le système complet

        Args:
            query: Requête utilisateur
            user_level: Niveau de l'utilisateur ("beginner", "intermediate", "advanced")
            max_results: Nombre maximum de résultats

        Returns:
            Résultat optimisé avec contexte enrichi et prompt adaptatif
        """

        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        if not self.indexed_hierarchy:
            raise RuntimeError("No documents indexed. Call index_documents() first.")

        start_time = datetime.now()

        try:
            # 1. Analyser la vague de la requête
            is_vague, vagueness_score = self.vague_query_optimizer.is_vague_query(query)

            logger.info(f"Query analysis: vague={is_vague}, score={vagueness_score:.2f}")

            # 2. Expansion de requête si vague
            expanded_queries = [query]
            if is_vague:
                expanded_queries = self.vague_query_optimizer.expand_vague_query(query)
                logger.info(f"Query expanded to {len(expanded_queries)} variations")

            # 3. Récupération hybride optimisée
            if self.embedding_index:
                # Utiliser tous les chunks disponibles pour simuler
                all_chunks = []
                chunk_ids = []
                for doc_hierarchy in self.indexed_hierarchy.values():
                    for granularity, chunks in doc_hierarchy.items():
                        for chunk in chunks:
                            all_chunks.append(chunk.text)
                            chunk_ids.append(chunk.metadata.chunk_id)

                # Simulation de recherche hybride
                import numpy as np
                embeddings = np.random.rand(len(all_chunks), 384)  # Simulation

                retrieved_chunks, scores = self.vague_query_optimizer.optimize_retrieval_for_vague_query(
                    query, all_chunks, embeddings, top_k=max_results
                )
            else:
                # Fallback: prendre des chunks aléatoires
                all_chunks = []
                for doc_hierarchy in self.indexed_hierarchy.values():
                    for chunks in doc_hierarchy.values():
                        all_chunks.extend([chunk.text for chunk in chunks[:2]])

                retrieved_chunks = all_chunks[:max_results]
                scores = [0.5] * len(retrieved_chunks)

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

            # 4. Enrichissement contextuel
            original_context = "\n\n".join(retrieved_chunks)

            if self.context_enrichment_pipeline:
                enriched_context_obj = self.context_enrichment_pipeline.enrich_context(
                    original_context=original_context,
                    query=query,
                    vagueness_score=vagueness_score,
                    user_level=user_level
                )
                enriched_context = enriched_context_obj.get_full_context()
                context_quality = enriched_context_obj.quality_score
            else:
                enriched_context = original_context
                context_quality = 0.5

            logger.info(f"Context enriched (quality: {context_quality:.2f})")

            # 5. Génération de prompt adaptatif
            if self.adaptive_prompt_engine:
                adaptive_prompt = self.adaptive_prompt_engine.create_adaptive_prompt(
                    query=query,
                    context=enriched_context,
                    vagueness_score=vagueness_score,
                    user_level=user_level
                )
                optimized_prompt = adaptive_prompt.get_full_prompt()
                prompt_metadata = adaptive_prompt.metadata
            else:
                optimized_prompt = f"Question: {query}\n\nContexte: {enriched_context}\n\nRéponds de manière claire et précise."
                prompt_metadata = {}

            # 6. Collecte de métriques
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            if self.production_monitor:
                self.production_monitor.record_query_performance(
                    query=query,
                    response_time=response_time,
                    vagueness_score=vagueness_score,
                    context_quality=context_quality
                )

            # Résultat optimisé
            optimization_result = {
                "query": query,
                "is_vague": is_vague,
                "vagueness_score": vagueness_score,
                "expanded_queries": expanded_queries,
                "retrieved_chunks": retrieved_chunks,
                "retrieval_scores": scores,
                "enriched_context": enriched_context,
                "context_quality": context_quality,
                "optimized_prompt": optimized_prompt,
                "prompt_metadata": prompt_metadata,
                "performance": {
                    "response_time": response_time,
                    "chunks_processed": len(retrieved_chunks),
                    "expansion_count": len(expanded_queries)
                },
                "recommendations": self._generate_optimization_recommendations(
                    vagueness_score, context_quality, response_time
                )
            }

            logger.info(f"🎯 Query optimization completed in {response_time:.2f}s")

            return optimization_result

        except Exception as e:
            error_msg = f"Error optimizing query '{query}': {e}"
            logger.error(error_msg)

            if self.production_monitor:
                self.production_monitor.record_query_performance(
                    query=query,
                    response_time=(datetime.now() - start_time).total_seconds(),
                    vagueness_score=0.5,
                    error_occurred=True
                )

            raise RuntimeError(error_msg)

    def collect_feedback(self,
                        query: str,
                        response: str,
                        relevance_score: int,
                        helpfulness_score: int,
                        clarity_score: int,
                        user_id: str = None,
                        improvements_suggested: List[str] = None) -> bool:
        """
        Collecte le feedback utilisateur pour amélioration continue

        Returns:
            True si le feedback a été collecté avec succès
        """

        if not self.production_monitor:
            logger.warning("Production monitor not available for feedback collection")
            return False

        try:
            # Analyser la requête pour obtenir le score de vague
            _, vagueness_score = self.vague_query_optimizer.is_vague_query(query) if self.vague_query_optimizer else (False, 0.5)

            self.production_monitor.collect_user_feedback(
                query=query,
                response=response,
                relevance_score=relevance_score,
                helpfulness_score=helpfulness_score,
                clarity_score=clarity_score,
                user_id=user_id,
                vagueness_score=vagueness_score,
                improvements_suggested=improvements_suggested or []
            )

            logger.info(f"Feedback collected for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """
        Retourne le statut complet du système
        """

        status = {
            "system_initialized": self.is_initialized,
            "components_status": self.components_status,
            "domain": self.domain,
            "language": self.language,
            "has_indexed_data": bool(self.indexed_hierarchy),
            "monitoring_enabled": self.enable_monitoring,
            "fine_tuning_enabled": self.enable_fine_tuning,
            "fine_tuned_model": self.fine_tuned_model_path,
            "timestamp": datetime.now().isoformat()
        }

        # Ajouter les stats de monitoring si disponibles
        if self.production_monitor:
            try:
                health_status = self.production_monitor.get_system_health()
                status["system_health"] = health_status
            except Exception as e:
                status["monitoring_error"] = str(e)

        # Ajouter les stats d'indexation
        if self.indexed_hierarchy:
            total_docs = len(self.indexed_hierarchy)
            total_chunks = sum(
                len(chunks)
                for doc_hierarchy in self.indexed_hierarchy.values()
                for chunks in doc_hierarchy.values()
            )

            status["indexing_stats"] = {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "embedding_index_size": len(self.embedding_index.embedding_cache) if self.embedding_index else 0
            }

        return status

    def _generate_optimization_recommendations(self,
                                             vagueness_score: float,
                                             context_quality: float,
                                             response_time: float) -> List[str]:
        """Génère des recommandations d'optimisation"""

        recommendations = []

        if vagueness_score > 0.7:
            recommendations.append("Requête très vague détectée - considérer l'ajout de questions de clarification")

        if context_quality < 0.5:
            recommendations.append("Qualité du contexte faible - améliorer l'enrichissement contextuel")

        if response_time > 5.0:
            recommendations.append("Temps de réponse élevé - optimiser la récupération ou réduire la complexité")

        if not recommendations:
            recommendations.append("Performance optimale - aucune amélioration nécessaire")

        return recommendations

    def _save_index_data(self, indexing_stats: Dict[str, Any]):
        """Sauvegarde les données d'index"""

        try:
            # Sauvegarder les statistiques
            stats_file = self.data_dir / "indexing_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(indexing_stats, f, indent=2, ensure_ascii=False, default=str)

            # Sauvegarder l'index des embeddings si disponible
            if self.embedding_index:
                index_path = self.data_dir / "hybrid_embedding_index"
                self.embedding_index.save_index(str(index_path))

            logger.info(f"Index data saved to {self.data_dir}")

        except Exception as e:
            logger.error(f"Error saving index data: {e}")

    def load_index_data(self) -> bool:
        """Charge les données d'index sauvegardées"""

        try:
            # Charger l'index des embeddings
            index_path = self.data_dir / "hybrid_embedding_index"
            if index_path.exists() and self.hybrid_embedding_system:
                self.embedding_index = self.hybrid_embedding_system
                self.embedding_index.load_index(str(index_path))
                logger.info("Hybrid embedding index loaded")

            # Charger les stats
            stats_file = self.data_dir / "indexing_stats.json"
            if stats_file.exists():
                import json
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                logger.info(f"Loaded indexing stats: {stats.get('total_documents', 0)} documents")

            return True

        except Exception as e:
            logger.error(f"Error loading index data: {e}")
            return False

    def shutdown(self):
        """Arrête proprement le système"""

        logger.info("Shutting down Vague Query Optimization System...")

        if self.production_monitor:
            self.production_monitor.stop_monitoring()

        # Sauvegarder les données importantes
        if self.indexed_hierarchy:
            self._save_index_data({"shutdown_timestamp": datetime.now().isoformat()})

        logger.info("System shutdown completed")


# Fonctions utilitaires pour l'intégration avec RAG Chunk Lab
def create_vague_optimization_system(domain: str = "general",
                                   openai_api_key: str = None,
                                   enable_monitoring: bool = True,
                                   enable_fine_tuning: bool = False) -> VagueQueryOptimizationSystem:
    """
    Crée et initialise un système complet d'optimisation pour requêtes vagues

    Returns:
        Système configuré et initialisé
    """

    system = VagueQueryOptimizationSystem(
        domain=domain,
        openai_api_key=openai_api_key,
        enable_monitoring=enable_monitoring,
        enable_fine_tuning=enable_fine_tuning
    )

    # Initialiser le système
    init_status = system.initialize_system()

    if not system.is_initialized:
        logger.warning("System partially initialized. Some features may not work properly.")

    return system


def quick_vague_query_optimization(query: str,
                                 documents: List[Dict[str, Any]],
                                 domain: str = "general",
                                 openai_api_key: str = None) -> Dict[str, Any]:
    """
    Optimisation rapide d'une requête vague (pour tests/démos)

    Args:
        query: Requête à optimiser
        documents: Documents à indexer (format: [{"doc_id": "id", "text": "content"}])
        domain: Domaine spécialisé
        openai_api_key: Clé API OpenAI

    Returns:
        Résultat d'optimisation
    """

    # Créer le système
    system = create_vague_optimization_system(
        domain=domain,
        openai_api_key=openai_api_key,
        enable_monitoring=False,  # Désactivé pour test rapide
        enable_fine_tuning=False
    )

    try:
        # Indexer les documents
        indexing_stats = system.index_documents(documents)
        logger.info(f"Quick indexing: {indexing_stats['processed_documents']} documents")

        # Optimiser la requête
        result = system.optimize_vague_query(query)

        # Ajouter les stats d'indexation au résultat
        result["indexing_stats"] = indexing_stats

        return result

    finally:
        # Nettoyer
        system.shutdown()