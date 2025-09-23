# hybrid_embeddings.py
"""
Système d'embeddings hybrides (dense + sparse) optimisé pour requêtes vagues
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import logging
from collections import defaultdict, Counter
import re
import math

# Imports conditionnels
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Résultat d'embedding avec métadonnées"""
    chunk_id: str
    dense_embedding: np.ndarray
    sparse_embedding: Dict[str, float]
    embedding_metadata: Dict[str, Any]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire sérialisable"""
        return {
            "chunk_id": self.chunk_id,
            "dense_embedding": self.dense_embedding.tolist() if self.dense_embedding is not None else None,
            "sparse_embedding": self.sparse_embedding,
            "embedding_metadata": self.embedding_metadata,
            "text": self.text
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingResult':
        """Crée depuis un dictionnaire"""
        return cls(
            chunk_id=data["chunk_id"],
            dense_embedding=np.array(data["dense_embedding"]) if data["dense_embedding"] else None,
            sparse_embedding=data["sparse_embedding"],
            embedding_metadata=data["embedding_metadata"],
            text=data["text"]
        )


@dataclass
class RetrievalResult:
    """Résultat de récupération hybride"""
    chunk_id: str
    text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "hybrid_score": self.hybrid_score,
            "metadata": self.metadata
        }


class BM25Retriever:
    """Implémentation BM25 pour embeddings sparse"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.idf_scores = {}
        self.doc_freqs = {}
        self.doc_lens = {}
        self.avg_doc_len = 0
        self.vocab = set()
        self.fitted = False

        # Configuration NLP
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("fr_core_news_sm")
            except OSError:
                self.nlp = None
        else:
            self.nlp = None

    def _preprocess_text(self, text: str) -> List[str]:
        """Préprocesse le texte pour BM25"""
        if self.nlp:
            doc = self.nlp(text.lower())
            tokens = [token.lemma_ for token in doc
                     if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2]
        else:
            # Fallback simple
            tokens = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
            # Filtrage basique des mots vides français
            stop_words = {"le", "la", "les", "de", "du", "des", "et", "ou", "à", "un", "une", "dans", "pour", "avec", "sur", "par"}
            tokens = [token for token in tokens if token not in stop_words]

        return tokens

    def fit(self, documents: List[str], doc_ids: List[str]):
        """Entraîne le modèle BM25 sur les documents"""
        self.doc_ids = doc_ids
        self.documents = documents

        # Préprocesser tous les documents
        processed_docs = [self._preprocess_text(doc) for doc in documents]

        # Calculer les statistiques
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in zip(doc_ids, processed_docs)}
        self.avg_doc_len = sum(self.doc_lens.values()) / len(self.doc_lens)

        # Compter les occurrences de chaque terme
        doc_term_freqs = {}
        all_terms = set()

        for doc_id, tokens in zip(doc_ids, processed_docs):
            term_freq = Counter(tokens)
            doc_term_freqs[doc_id] = term_freq
            all_terms.update(tokens)

        self.vocab = all_terms
        self.doc_freqs = doc_term_freqs

        # Calculer IDF pour chaque terme
        N = len(documents)
        for term in self.vocab:
            df = sum(1 for doc_freqs in doc_term_freqs.values() if term in doc_freqs)
            self.idf_scores[term] = math.log((N - df + 0.5) / (df + 0.5))

        self.fitted = True
        logger.info(f"BM25 trained on {N} documents with {len(self.vocab)} unique terms")

    def get_sparse_embedding(self, text: str) -> Dict[str, float]:
        """Génère un embedding sparse pour un texte"""
        if not self.fitted:
            raise ValueError("BM25 must be fitted before generating embeddings")

        tokens = self._preprocess_text(text)
        term_freq = Counter(tokens)
        doc_len = len(tokens)

        sparse_vector = {}

        for term, tf in term_freq.items():
            if term in self.vocab:
                # Score BM25 pour ce terme
                idf = self.idf_scores.get(term, 0)
                tf_component = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len))
                bm25_score = idf * tf_component

                if bm25_score > 0:
                    sparse_vector[term] = bm25_score

        return sparse_vector

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Recherche BM25 standard"""
        query_embedding = self.get_sparse_embedding(query)

        scores = []
        for doc_id in self.doc_ids:
            doc_embedding = self.get_sparse_embedding(self.documents[self.doc_ids.index(doc_id)])

            # Calculer la similarité (produit scalaire pour BM25)
            score = sum(query_embedding.get(term, 0) * doc_embedding.get(term, 0)
                       for term in set(query_embedding.keys()).union(doc_embedding.keys()))

            scores.append((doc_id, score))

        # Trier et retourner top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridEmbeddingSystem:
    """Système d'embeddings hybrides dense + sparse"""

    def __init__(self,
                 dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 domain: str = "general",
                 language: str = "fr"):
        self.domain = domain
        self.language = language

        # Modèle dense
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.dense_model = SentenceTransformer(dense_model_name)
                logger.info(f"Loaded dense model: {dense_model_name}")
            except Exception as e:
                logger.error(f"Failed to load dense model: {e}")
                self.dense_model = None
        else:
            self.dense_model = None

        # Modèle sparse (BM25)
        self.sparse_model = BM25Retriever()

        # Configuration TF-IDF alternative
        if SKLEARN_AVAILABLE:
            self.tfidf_model = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words=self._get_stop_words(),
                lowercase=True,
                token_pattern=r'\b[a-zA-ZÀ-ÿ]{3,}\b'
            )
        else:
            self.tfidf_model = None

        # Configuration de fusion
        self.fusion_weights = {
            "dense": 0.7,
            "sparse": 0.3
        }

        # Cache des embeddings
        self.embedding_cache = {}
        self.index_metadata = {}

    def _get_stop_words(self) -> List[str]:
        """Retourne les mots vides selon la langue"""
        if self.language == "fr":
            return ["le", "la", "les", "de", "du", "des", "et", "ou", "à", "un", "une",
                   "dans", "pour", "avec", "sur", "par", "ce", "cette", "ces", "qui",
                   "que", "dont", "où", "il", "elle", "on", "nous", "vous", "ils", "elles"]
        else:
            return ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                   "of", "with", "by", "this", "that", "these", "those", "i", "you", "he",
                   "she", "it", "we", "they"]

    def fit(self, documents: List[str], doc_ids: List[str], metadata: List[Dict] = None):
        """Entraîne le système hybride sur les documents"""

        logger.info(f"Training hybrid embedding system on {len(documents)} documents")

        # Entraîner le modèle sparse
        self.sparse_model.fit(documents, doc_ids)

        # Entraîner TF-IDF si disponible
        if self.tfidf_model:
            try:
                self.tfidf_model.fit(documents)
                logger.info("TF-IDF model trained successfully")
            except Exception as e:
                logger.warning(f"TF-IDF training failed: {e}")

        # Stocker les métadonnées
        if metadata:
            self.index_metadata = {doc_id: meta for doc_id, meta in zip(doc_ids, metadata)}

        # Pré-calculer les embeddings si le dataset n'est pas trop grand
        if len(documents) < 10000:
            logger.info("Pre-computing embeddings for fast retrieval")
            self._precompute_embeddings(documents, doc_ids)

    def _precompute_embeddings(self, documents: List[str], doc_ids: List[str]):
        """Pré-calcule les embeddings pour optimiser les performances"""

        for doc_id, document in zip(doc_ids, documents):
            embedding_result = self.embed_text(document, doc_id)
            self.embedding_cache[doc_id] = embedding_result

    def embed_text(self, text: str, chunk_id: str, metadata: Dict = None) -> EmbeddingResult:
        """Génère les embeddings hybrides pour un texte"""

        # Embedding dense
        dense_embedding = None
        if self.dense_model:
            try:
                dense_embedding = self.dense_model.encode([text])[0]
            except Exception as e:
                logger.warning(f"Dense embedding failed for {chunk_id}: {e}")

        # Embedding sparse (BM25)
        sparse_embedding = {}
        if self.sparse_model.fitted:
            try:
                sparse_embedding = self.sparse_model.get_sparse_embedding(text)
            except Exception as e:
                logger.warning(f"Sparse embedding failed for {chunk_id}: {e}")

        # Métadonnées d'embedding
        embedding_metadata = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "dense_dim": len(dense_embedding) if dense_embedding is not None else 0,
            "sparse_terms": len(sparse_embedding),
            "domain": self.domain,
            "timestamp": None  # Pourrait être ajouté
        }

        if metadata:
            embedding_metadata.update(metadata)

        return EmbeddingResult(
            chunk_id=chunk_id,
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            embedding_metadata=embedding_metadata,
            text=text
        )

    def search_hybrid(self,
                     query: str,
                     top_k: int = 10,
                     dense_weight: float = None,
                     sparse_weight: float = None,
                     rerank: bool = True) -> List[RetrievalResult]:
        """
        Recherche hybride avec fusion des scores

        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            dense_weight: Poids pour l'embedding dense (optionnel)
            sparse_weight: Poids pour l'embedding sparse (optionnel)
            rerank: Active le re-ranking intelligent

        Returns:
            Liste des résultats classés
        """

        # Utiliser les poids par défaut si non spécifiés
        if dense_weight is None:
            dense_weight = self.fusion_weights["dense"]
        if sparse_weight is None:
            sparse_weight = self.fusion_weights["sparse"]

        # Normaliser les poids
        total_weight = dense_weight + sparse_weight
        dense_weight /= total_weight
        sparse_weight /= total_weight

        # Génération de l'embedding de la requête
        query_embedding = self.embed_text(query, "query")

        # Recherche dense
        dense_results = {}
        if query_embedding.dense_embedding is not None and self.embedding_cache:
            dense_results = self._search_dense(query_embedding.dense_embedding, top_k * 2)

        # Recherche sparse
        sparse_results = {}
        if query_embedding.sparse_embedding and self.sparse_model.fitted:
            sparse_results = self._search_sparse(query_embedding.sparse_embedding, top_k * 2)

        # Fusion des résultats
        hybrid_results = self._fuse_results(
            dense_results, sparse_results,
            dense_weight, sparse_weight, top_k
        )

        # Re-ranking intelligent si activé
        if rerank and len(hybrid_results) > 1:
            hybrid_results = self._intelligent_rerank(query, hybrid_results)

        return hybrid_results[:top_k]

    def _search_dense(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, float]:
        """Recherche par similarité dense (cosine)"""

        similarities = {}

        for chunk_id, cached_embedding in self.embedding_cache.items():
            if cached_embedding.dense_embedding is not None:
                # Similarité cosine
                similarity = np.dot(query_embedding, cached_embedding.dense_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding.dense_embedding)
                )
                similarities[chunk_id] = float(similarity)

        # Trier et retourner top-k
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_results[:top_k])

    def _search_sparse(self, query_sparse: Dict[str, float], top_k: int) -> Dict[str, float]:
        """Recherche par similarité sparse (BM25)"""

        similarities = {}

        for chunk_id, cached_embedding in self.embedding_cache.items():
            if cached_embedding.sparse_embedding:
                # Similarité sparse (produit scalaire normalisé)
                score = 0.0
                query_norm = math.sqrt(sum(v**2 for v in query_sparse.values()))
                doc_norm = math.sqrt(sum(v**2 for v in cached_embedding.sparse_embedding.values()))

                if query_norm > 0 and doc_norm > 0:
                    for term, query_weight in query_sparse.items():
                        doc_weight = cached_embedding.sparse_embedding.get(term, 0)
                        score += query_weight * doc_weight

                    score = score / (query_norm * doc_norm)

                similarities[chunk_id] = score

        # Trier et retourner top-k
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_results[:top_k])

    def _fuse_results(self,
                     dense_results: Dict[str, float],
                     sparse_results: Dict[str, float],
                     dense_weight: float,
                     sparse_weight: float,
                     top_k: int) -> List[RetrievalResult]:
        """Fusionne les résultats dense et sparse"""

        # Collecter tous les chunk_ids
        all_chunk_ids = set(dense_results.keys()).union(set(sparse_results.keys()))

        # Normaliser les scores dans chaque modalité
        if dense_results:
            max_dense = max(dense_results.values())
            min_dense = min(dense_results.values())
            dense_range = max_dense - min_dense
            if dense_range > 0:
                dense_results = {k: (v - min_dense) / dense_range for k, v in dense_results.items()}

        if sparse_results:
            max_sparse = max(sparse_results.values())
            min_sparse = min(sparse_results.values())
            sparse_range = max_sparse - min_sparse
            if sparse_range > 0:
                sparse_results = {k: (v - min_sparse) / sparse_range for k, v in sparse_results.items()}

        # Fusionner avec pondération
        fused_results = []

        for chunk_id in all_chunk_ids:
            dense_score = dense_results.get(chunk_id, 0.0)
            sparse_score = sparse_results.get(chunk_id, 0.0)
            hybrid_score = dense_weight * dense_score + sparse_weight * sparse_score

            # Récupérer les informations du chunk
            if chunk_id in self.embedding_cache:
                cached_embedding = self.embedding_cache[chunk_id]
                text = cached_embedding.text
                metadata = cached_embedding.embedding_metadata
            else:
                text = f"Chunk {chunk_id}"
                metadata = {}

            result = RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                dense_score=dense_score,
                sparse_score=sparse_score,
                hybrid_score=hybrid_score,
                metadata=metadata
            )

            fused_results.append(result)

        # Trier par score hybride
        fused_results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return fused_results

    def _intelligent_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-ranking intelligent basé sur des critères additionnels"""

        query_lower = query.lower()

        for result in results:
            # Bonus pour correspondance exacte de termes
            exact_match_bonus = 0.0
            query_terms = set(query_lower.split())
            text_terms = set(result.text.lower().split())
            exact_matches = query_terms.intersection(text_terms)
            if exact_matches:
                exact_match_bonus = len(exact_matches) / len(query_terms) * 0.1

            # Bonus pour diversité (éviter les doublons sémantiques)
            diversity_bonus = 0.0
            # TODO: Implémenter la logique de diversité

            # Bonus basé sur les métadonnées (longueur optimale, etc.)
            metadata_bonus = 0.0
            word_count = result.metadata.get("word_count", 0)
            if 50 <= word_count <= 300:  # Longueur optimale
                metadata_bonus = 0.05

            # Ajuster le score hybride
            result.hybrid_score += exact_match_bonus + diversity_bonus + metadata_bonus

        # Re-trier avec les nouveaux scores
        results.sort(key=lambda x: x.hybrid_score, reverse=True)

        return results

    def adaptive_weights_for_query(self, query: str) -> Dict[str, float]:
        """
        Détermine des poids adaptatifs basés sur les caractéristiques de la requête

        Returns:
            Dict avec 'dense' et 'sparse' weights
        """

        query_lower = query.lower()

        # Analyser les caractéristiques de la requête
        word_count = len(query.split())
        has_technical_terms = any(term in query_lower for term in ["API", "config", "paramètre", "fonction"])
        has_exact_terms = bool(re.search(r'"[^"]+"', query))  # Termes entre guillemets

        # Logique adaptative
        if word_count <= 3:
            # Requête courte: privilégier dense pour la sémantique
            return {"dense": 0.8, "sparse": 0.2}
        elif has_exact_terms or has_technical_terms:
            # Termes exacts ou techniques: privilégier sparse
            return {"dense": 0.4, "sparse": 0.6}
        elif word_count > 10:
            # Requête longue: équilibrer
            return {"dense": 0.6, "sparse": 0.4}
        else:
            # Cas standard
            return {"dense": 0.7, "sparse": 0.3}

    def search_with_adaptive_weights(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Recherche avec poids adaptatifs automatiques"""

        weights = self.adaptive_weights_for_query(query)

        return self.search_hybrid(
            query=query,
            top_k=top_k,
            dense_weight=weights["dense"],
            sparse_weight=weights["sparse"],
            rerank=True
        )

    def save_index(self, output_path: str):
        """Sauvegarde l'index hybride"""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder le cache d'embeddings
        embeddings_file = output_dir / "embeddings_cache.pkl"
        with open(embeddings_file, 'wb') as f:
            # Convertir en format sérialisable
            serializable_cache = {
                chunk_id: embedding.to_dict()
                for chunk_id, embedding in self.embedding_cache.items()
            }
            pickle.dump(serializable_cache, f)

        # Sauvegarder les métadonnées
        metadata_file = output_dir / "index_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                "domain": self.domain,
                "language": self.language,
                "fusion_weights": self.fusion_weights,
                "index_size": len(self.embedding_cache),
                "sparse_vocab_size": len(self.sparse_model.vocab) if self.sparse_model.fitted else 0
            }, f, indent=2)

        # Sauvegarder le modèle BM25
        bm25_file = output_dir / "bm25_model.pkl"
        with open(bm25_file, 'wb') as f:
            pickle.dump({
                "idf_scores": self.sparse_model.idf_scores,
                "doc_freqs": self.sparse_model.doc_freqs,
                "doc_lens": self.sparse_model.doc_lens,
                "avg_doc_len": self.sparse_model.avg_doc_len,
                "vocab": self.sparse_model.vocab,
                "fitted": self.sparse_model.fitted
            }, f)

        logger.info(f"Hybrid index saved to {output_path}")

    def load_index(self, input_path: str):
        """Charge un index hybride sauvegardé"""

        input_dir = Path(input_path)

        # Charger le cache d'embeddings
        embeddings_file = input_dir / "embeddings_cache.pkl"
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                serializable_cache = pickle.load(f)
                self.embedding_cache = {
                    chunk_id: EmbeddingResult.from_dict(data)
                    for chunk_id, data in serializable_cache.items()
                }

        # Charger les métadonnées
        metadata_file = input_dir / "index_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.domain = metadata.get("domain", self.domain)
                self.language = metadata.get("language", self.language)
                self.fusion_weights = metadata.get("fusion_weights", self.fusion_weights)

        # Charger le modèle BM25
        bm25_file = input_dir / "bm25_model.pkl"
        if bm25_file.exists():
            with open(bm25_file, 'rb') as f:
                bm25_data = pickle.load(f)
                self.sparse_model.idf_scores = bm25_data["idf_scores"]
                self.sparse_model.doc_freqs = bm25_data["doc_freqs"]
                self.sparse_model.doc_lens = bm25_data["doc_lens"]
                self.sparse_model.avg_doc_len = bm25_data["avg_doc_len"]
                self.sparse_model.vocab = bm25_data["vocab"]
                self.sparse_model.fitted = bm25_data["fitted"]

        logger.info(f"Hybrid index loaded from {input_path}")


# Fonctions d'intégration pour RAG Chunk Lab
def create_hybrid_index_from_hierarchy(hierarchy: Dict[str, List[Dict]],
                                      document_id: str,
                                      domain: str = "general") -> HybridEmbeddingSystem:
    """
    Crée un index hybride depuis une hiérarchie de chunks

    Args:
        hierarchy: Hiérarchie de chunks enrichis
        document_id: ID du document
        domain: Domaine spécialisé

    Returns:
        Système d'embeddings hybrides entraîné
    """

    # Extraire tous les chunks
    all_chunks = []
    chunk_ids = []
    metadata_list = []

    for granularity, chunks in hierarchy.items():
        for chunk in chunks:
            # Gérer à la fois les dictionnaires et les HierarchicalChunk dataclasses
            if hasattr(chunk, 'text'):  # HierarchicalChunk dataclass
                all_chunks.append(chunk.text)
                chunk_ids.append(chunk.metadata.chunk_id)

                # Préparer métadonnées pour l'embedding
                chunk_metadata = {
                    "granularity": granularity,
                    "document_id": document_id,
                    "domain": domain
                }

                # Ajouter métadonnées enrichies si disponibles
                if hasattr(chunk.metadata, 'enriched_metadata') and chunk.metadata.enriched_metadata:
                    enriched = chunk.metadata.enriched_metadata
                    if hasattr(enriched, 'to_dict'):
                        enriched_dict = enriched.to_dict()
                    else:
                        enriched_dict = enriched

                    chunk_metadata.update({
                        "complexity_level": enriched_dict.get("complexity_level"),
                        "content_type": enriched_dict.get("content_type"),
                        "semantic_density": enriched_dict.get("semantic_density"),
                        "domain_relevance": enriched_dict.get("domain_relevance")
                    })

            else:  # Format dictionnaire (backward compatibility)
                all_chunks.append(chunk["text"])
                chunk_ids.append(chunk["metadata"]["chunk_id"])

                # Préparer métadonnées pour l'embedding
                chunk_metadata = {
                    "granularity": granularity,
                    "document_id": document_id,
                    "domain": domain
                }

                # Ajouter métadonnées enrichies si disponibles
                if "enriched_metadata" in chunk:
                    enriched = chunk["enriched_metadata"]
                    chunk_metadata.update({
                        "complexity_level": enriched.get("complexity_level"),
                        "content_type": enriched.get("content_type"),
                        "semantic_density": enriched.get("semantic_density"),
                        "domain_relevance": enriched.get("domain_relevance")
                    })

            metadata_list.append(chunk_metadata)

    # Créer et entraîner le système hybride
    hybrid_system = HybridEmbeddingSystem(domain=domain)
    hybrid_system.fit(all_chunks, chunk_ids, metadata_list)

    logger.info(f"Created hybrid index with {len(all_chunks)} chunks from {len(hierarchy)} granularity levels")

    return hybrid_system


def search_with_vague_query_optimization(hybrid_system: HybridEmbeddingSystem,
                                        query: str,
                                        vagueness_score: float,
                                        top_k: int = 10) -> List[RetrievalResult]:
    """
    Recherche optimisée pour requêtes vagues

    Args:
        hybrid_system: Système d'embeddings hybrides
        query: Requête utilisateur
        vagueness_score: Score de vague (0-1)
        top_k: Nombre de résultats

    Returns:
        Résultats optimisés pour requêtes vagues
    """

    # Adapter les poids selon la vague de la requête
    if vagueness_score >= 0.7:
        # Très vague: privilégier dense pour capturer l'intention sémantique
        weights = {"dense": 0.8, "sparse": 0.2}
    elif vagueness_score >= 0.4:
        # Moyennement vague: équilibrer
        weights = {"dense": 0.6, "sparse": 0.4}
    else:
        # Précise: privilégier sparse pour correspondance exacte
        weights = {"dense": 0.4, "sparse": 0.6}

    # Recherche avec poids adaptatifs
    results = hybrid_system.search_hybrid(
        query=query,
        top_k=top_k * 2,  # Récupérer plus pour diversifier
        dense_weight=weights["dense"],
        sparse_weight=weights["sparse"],
        rerank=True
    )

    # Post-traitement pour requêtes vagues
    if vagueness_score >= 0.5:
        results = _diversify_results_for_vague_query(results, top_k)

    return results[:top_k]


def _diversify_results_for_vague_query(results: List[RetrievalResult], target_count: int) -> List[RetrievalResult]:
    """Diversifie les résultats pour requêtes vagues"""

    if len(results) <= target_count:
        return results

    diversified = [results[0]]  # Prendre le meilleur

    for candidate in results[1:]:
        if len(diversified) >= target_count:
            break

        # Vérifier la diversité avec les résultats déjà sélectionnés
        is_diverse = True
        for selected in diversified:
            # Similarité simple basée sur les mots communs
            candidate_words = set(candidate.text.lower().split())
            selected_words = set(selected.text.lower().split())
            overlap = len(candidate_words.intersection(selected_words))
            similarity = overlap / max(1, min(len(candidate_words), len(selected_words)))

            if similarity > 0.7:  # Trop similaire
                is_diverse = False
                break

        if is_diverse:
            diversified.append(candidate)

    return diversified