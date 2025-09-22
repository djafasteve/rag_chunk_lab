"""
Métriques avancées pour l'évaluation des embeddings et de la récupération
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingMetrics:
    """
    Classe pour calculer des métriques avancées d'évaluation des embeddings
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def calculate_recall_at_k(self,
                             retrieved_chunks: List[List[Dict]],
                             ground_truth_chunks: List[List[str]],
                             k: int = 5) -> float:
        """
        Calcule Recall@K : proportion de chunks pertinents récupérés dans les K premiers

        Args:
            retrieved_chunks: Liste de listes de chunks récupérés pour chaque question
            ground_truth_chunks: Liste de listes de chunks pertinents pour chaque question
            k: Nombre de chunks top à considérer

        Returns:
            Score Recall@K moyen
        """
        if not retrieved_chunks or not ground_truth_chunks:
            return 0.0

        recall_scores = []

        for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
            if not ground_truth:
                continue

            # Prendre les K premiers chunks récupérés
            top_k_retrieved = [chunk.get('text', '') for chunk in retrieved[:k]]

            # Compter combien de chunks pertinents sont dans les K premiers
            relevant_found = 0
            for gt_chunk in ground_truth:
                for retrieved_chunk in top_k_retrieved:
                    # Simple similarité textuelle (peut être améliorée)
                    if self._text_similarity(retrieved_chunk, gt_chunk) > 0.5:
                        relevant_found += 1
                        break

            # Recall = chunks pertinents trouvés / total chunks pertinents
            recall = relevant_found / len(ground_truth) if ground_truth else 0.0
            recall_scores.append(recall)

        return np.mean(recall_scores) if recall_scores else 0.0

    def calculate_mrr(self,
                     retrieved_chunks: List[List[Dict]],
                     ground_truth_chunks: List[List[str]]) -> float:
        """
        Calcule Mean Reciprocal Rank (MRR)

        Args:
            retrieved_chunks: Liste de listes de chunks récupérés pour chaque question
            ground_truth_chunks: Liste de listes de chunks pertinents pour chaque question

        Returns:
            Score MRR moyen
        """
        if not retrieved_chunks or not ground_truth_chunks:
            return 0.0

        reciprocal_ranks = []

        for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
            if not ground_truth:
                continue

            # Trouver le rang du premier chunk pertinent
            first_relevant_rank = None
            for rank, chunk in enumerate(retrieved, 1):
                chunk_text = chunk.get('text', '')
                for gt_chunk in ground_truth:
                    if self._text_similarity(chunk_text, gt_chunk) > 0.5:
                        first_relevant_rank = rank
                        break
                if first_relevant_rank:
                    break

            # MRR = 1/rang du premier pertinent (0 si aucun trouvé)
            rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            reciprocal_ranks.append(rr)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_ndcg(self,
                      retrieved_chunks: List[List[Dict]],
                      ground_truth_chunks: List[List[str]],
                      k: int = 10) -> float:
        """
        Calcule Normalized Discounted Cumulative Gain (NDCG@K)

        Args:
            retrieved_chunks: Liste de listes de chunks récupérés pour chaque question
            ground_truth_chunks: Liste de listes de chunks pertinents pour chaque question
            k: Nombre de chunks top à considérer

        Returns:
            Score NDCG@K moyen
        """
        if not retrieved_chunks or not ground_truth_chunks:
            return 0.0

        ndcg_scores = []

        for retrieved, ground_truth in zip(retrieved_chunks, ground_truth_chunks):
            if not ground_truth:
                continue

            # Calculer les scores de pertinence pour les K premiers
            relevance_scores = []
            for chunk in retrieved[:k]:
                chunk_text = chunk.get('text', '')
                max_similarity = 0.0
                for gt_chunk in ground_truth:
                    similarity = self._text_similarity(chunk_text, gt_chunk)
                    max_similarity = max(max_similarity, similarity)
                relevance_scores.append(max_similarity)

            # Calculer DCG
            dcg = self._calculate_dcg(relevance_scores)

            # Calculer IDCG (DCG idéal)
            ideal_scores = [1.0] * min(len(ground_truth), k) + [0.0] * max(0, k - len(ground_truth))
            idcg = self._calculate_dcg(ideal_scores)

            # NDCG = DCG / IDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _calculate_dcg(self, relevance_scores: List[float]) -> float:
        """Calcule Discounted Cumulative Gain"""
        dcg = 0.0
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)  # +2 car log2(1) = 0
        return dcg

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes using TF-IDF + cosine"""
        if not text1 or not text2:
            return 0.0

        try:
            # Vectorisation TF-IDF
            vectors = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Fallback : similarité basique par mots
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

class EmbeddingQualityAnalyzer:
    """
    Analyse la qualité technique des embeddings
    """

    def calculate_embedding_diversity(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Mesure la diversité des embeddings

        Args:
            embeddings: Matrice d'embeddings (n_docs x embedding_dim)

        Returns:
            Dictionnaire de métriques de diversité
        """
        if embeddings.shape[0] < 2:
            return {"diversity_score": 0.0, "mean_pairwise_distance": 0.0}

        # Calculer toutes les distances par paires
        similarity_matrix = cosine_similarity(embeddings)

        # Extraire les similarités (hors diagonale)
        n = similarity_matrix.shape[0]
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(similarity_matrix[i, j])

        similarities = np.array(similarities)

        # Métriques de diversité
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        mean_distance = 1.0 - mean_similarity  # Distance = 1 - cosine_similarity

        # Score de diversité : plus la variance est haute, plus c'est diversifié
        diversity_score = std_similarity

        return {
            "diversity_score": float(diversity_score),
            "mean_pairwise_distance": float(mean_distance),
            "mean_pairwise_similarity": float(mean_similarity),
            "std_similarity": float(std_similarity)
        }

    def analyze_embedding_distribution(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyse la distribution des embeddings

        Args:
            embeddings: Matrice d'embeddings (n_docs x embedding_dim)

        Returns:
            Dictionnaire de métriques de distribution
        """
        # Statistiques par dimension
        mean_per_dim = np.mean(embeddings, axis=0)
        std_per_dim = np.std(embeddings, axis=0)

        # Métriques globales
        global_mean = np.mean(mean_per_dim)
        global_std = np.mean(std_per_dim)

        # Norme moyenne des vecteurs
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        return {
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "mean_vector_norm": float(mean_norm),
            "std_vector_norm": float(std_norm),
            "dimension_variance": float(np.var(std_per_dim))
        }

    def measure_semantic_coherence(self,
                                 embeddings: np.ndarray,
                                 texts: List[str],
                                 sample_size: int = 100) -> Dict[str, float]:
        """
        Mesure la cohérence sémantique des embeddings

        Args:
            embeddings: Matrice d'embeddings
            texts: Textes correspondants
            sample_size: Taille de l'échantillon pour l'analyse

        Returns:
            Métriques de cohérence sémantique
        """
        if len(embeddings) != len(texts) or len(embeddings) < 10:
            return {"semantic_coherence": 0.0}

        # Échantillonner si trop de données
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            texts = [texts[i] for i in indices]

        # Calculer la similarité textuelle vs similarité d'embedding
        text_similarities = []
        embedding_similarities = []

        tfidf = TfidfVectorizer(stop_words='english', max_features=500)

        try:
            text_vectors = tfidf.fit_transform(texts)
            text_cosine = cosine_similarity(text_vectors)
            embedding_cosine = cosine_similarity(embeddings)

            # Échantillonner des paires pour l'analyse
            n = len(texts)
            sample_pairs = min(500, n * (n - 1) // 2)  # Limiter le nombre de paires

            for _ in range(sample_pairs):
                i, j = np.random.choice(n, 2, replace=False)
                text_similarities.append(text_cosine[i, j])
                embedding_similarities.append(embedding_cosine[i, j])

            # Corrélation entre similarités textuelles et d'embedding
            correlation = np.corrcoef(text_similarities, embedding_similarities)[0, 1]

            return {
                "semantic_coherence": float(correlation) if not np.isnan(correlation) else 0.0,
                "mean_text_similarity": float(np.mean(text_similarities)),
                "mean_embedding_similarity": float(np.mean(embedding_similarities))
            }

        except Exception as e:
            logger.warning(f"Erreur dans l'analyse de cohérence sémantique: {e}")
            return {"semantic_coherence": 0.0}

def evaluate_retrieval_performance(questions: List[str],
                                 retrieved_results: Dict[str, List[List[Dict]]],
                                 ground_truth: List[Dict],
                                 k_values: List[int] = [3, 5, 10, 15]) -> Dict[str, Dict]:
    """
    Évalue les performances de récupération pour tous les pipelines

    Args:
        questions: Liste des questions
        retrieved_results: Résultats par pipeline {pipeline: [chunks_per_question]}
        ground_truth: Ground truth avec chunks pertinents
        k_values: Valeurs de K à tester

    Returns:
        Métriques par pipeline
    """
    metrics_calculator = EmbeddingMetrics()
    results = {}

    # Extraire les chunks pertinents du ground truth
    ground_truth_chunks = []
    for gt in ground_truth:
        # Supposer que le ground truth contient des chunks pertinents
        relevant_chunks = gt.get('relevant_chunks', [gt.get('answer', '')])
        ground_truth_chunks.append(relevant_chunks)

    print(f"\n📊 Évaluation des performances de récupération...")
    print(f"   Questions: {len(questions)}")
    print(f"   Pipelines: {list(retrieved_results.keys())}")
    print(f"   K values: {k_values}")

    for pipeline, retrieved_chunks in retrieved_results.items():
        print(f"\n🔍 Évaluation pipeline: {pipeline}")

        pipeline_metrics = {}

        # Métriques pour différentes valeurs de K
        for k in k_values:
            recall_k = metrics_calculator.calculate_recall_at_k(
                retrieved_chunks, ground_truth_chunks, k
            )
            pipeline_metrics[f'recall@{k}'] = recall_k

        # MRR et NDCG
        mrr = metrics_calculator.calculate_mrr(retrieved_chunks, ground_truth_chunks)
        ndcg_10 = metrics_calculator.calculate_ndcg(retrieved_chunks, ground_truth_chunks, k=10)

        pipeline_metrics.update({
            'mrr': mrr,
            'ndcg@10': ndcg_10
        })

        results[pipeline] = pipeline_metrics

        # Affichage des résultats pour ce pipeline
        print(f"   Recall@5: {pipeline_metrics.get('recall@5', 0):.3f}")
        print(f"   MRR: {mrr:.3f}")
        print(f"   NDCG@10: {ndcg_10:.3f}")

    return results

def export_embedding_analysis(doc_id: str,
                            analysis_results: Dict,
                            output_dir: str = "exports") -> str:
    """
    Exporte les résultats d'analyse des embeddings

    Args:
        doc_id: Identifiant du document
        analysis_results: Résultats de l'analyse
        output_dir: Dossier de sortie

    Returns:
        Chemin du fichier exporté
    """
    from pathlib import Path

    export_path = Path(output_dir) / doc_id
    export_path.mkdir(parents=True, exist_ok=True)

    # Fichier JSON pour les métriques détaillées
    json_file = export_path / "embedding_analysis.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    # Fichier CSV pour un aperçu rapide
    csv_file = export_path / "embedding_metrics_summary.csv"

    # Préparer les données pour le CSV
    csv_data = []
    for pipeline, metrics in analysis_results.items():
        if isinstance(metrics, dict):
            row = {"pipeline": pipeline}
            row.update(metrics)
            csv_data.append(row)

    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)

    print(f"📁 Analyse des embeddings exportée vers: {export_path}")
    return str(json_file)