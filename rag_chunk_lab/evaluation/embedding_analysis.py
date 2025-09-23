"""
Module d'analyse technique des embeddings
"""

from typing import List, Dict, Optional, Tuple
import os, json
import numpy as np
import time
from pathlib import Path
from ..core.indexing import load_index_data
from .embedding_metrics import EmbeddingQualityAnalyzer

def get_embeddings_for_analysis(doc_id: str, pipeline_name: str, data_dir: str) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Récupère les embeddings et textes correspondants pour l'analyse technique

    Args:
        doc_id: Identifiant du document
        pipeline_name: Nom du pipeline (semantic ou azure_semantic)
        data_dir: Répertoire de données

    Returns:
        Tuple (embeddings, texts) ou None si non disponible
    """
    if pipeline_name not in ['semantic', 'azure_semantic']:
        return None

    try:
        # Charger les textes et métadonnées
        texts, meta = load_index_data(doc_id, pipeline_name, data_dir)

        # Charger les embeddings
        base = f"{data_dir}/{doc_id}/{pipeline_name}"
        embeddings_path = f"{base}/embeddings.npy"

        if not os.path.exists(embeddings_path):
            print(f"⚠️ Embeddings non trouvés pour {pipeline_name}: {embeddings_path}")
            return None

        embeddings = np.load(embeddings_path)
        print(f"✅ Embeddings chargés pour {pipeline_name}: {embeddings.shape}")

        return embeddings, texts

    except Exception as e:
        print(f"❌ Erreur lors du chargement des embeddings pour {pipeline_name}: {e}")
        return None

def analyze_pipeline_embeddings(doc_id: str, data_dir: str, pipelines: List[str] = None) -> Dict:
    """
    Analyse les embeddings de tous les pipelines disponibles

    Args:
        doc_id: Identifiant du document
        data_dir: Répertoire de données
        pipelines: Liste des pipelines à analyser (par défaut: tous les sémantiques)

    Returns:
        Dictionnaire des analyses par pipeline
    """
    if pipelines is None:
        pipelines = ['semantic', 'azure_semantic']

    analyzer = EmbeddingQualityAnalyzer()
    results = {}

    print(f"\n🔬 Analyse technique des embeddings pour: {doc_id}")

    for pipeline in pipelines:
        print(f"\n📊 Analyse du pipeline: {pipeline}")

        embedding_data = get_embeddings_for_analysis(doc_id, pipeline, data_dir)
        if embedding_data is None:
            results[pipeline] = {"error": "Embeddings non disponibles"}
            continue

        embeddings, texts = embedding_data

        try:
            # Analyse de diversité
            diversity_metrics = analyzer.calculate_embedding_diversity(embeddings)

            # Analyse de distribution
            distribution_metrics = analyzer.analyze_embedding_distribution(embeddings)

            # Analyse de cohérence sémantique
            coherence_metrics = analyzer.measure_semantic_coherence(embeddings, texts)

            # Statistiques de base
            basic_stats = {
                "embedding_dimension": embeddings.shape[1],
                "num_chunks": embeddings.shape[0],
                "total_texts_length": sum(len(text) for text in texts),
                "avg_text_length": sum(len(text) for text in texts) / len(texts) if texts else 0
            }

            # Assembler tous les résultats
            pipeline_analysis = {
                "basic_stats": basic_stats,
                "diversity": diversity_metrics,
                "distribution": distribution_metrics,
                "semantic_coherence": coherence_metrics,
                "analysis_timestamp": time.time()
            }

            results[pipeline] = pipeline_analysis

            # Afficher un résumé
            print(f"  ✅ Dimension: {basic_stats['embedding_dimension']}")
            print(f"  📊 Chunks: {basic_stats['num_chunks']}")
            print(f"  🎯 Diversité: {diversity_metrics.get('diversity_score', 0):.3f}")
            print(f"  🧠 Cohérence sémantique: {coherence_metrics.get('semantic_coherence', 0):.3f}")

        except Exception as e:
            print(f"  ❌ Erreur d'analyse pour {pipeline}: {e}")
            results[pipeline] = {"error": str(e)}

    return results

def export_embeddings_for_external_analysis(doc_id: str,
                                           pipeline_name: str,
                                           data_dir: str,
                                           output_dir: str = "embeddings_export") -> Optional[str]:
    """
    Exporte les embeddings et métadonnées pour analyse externe

    Args:
        doc_id: Identifiant du document
        pipeline_name: Nom du pipeline
        data_dir: Répertoire de données
        output_dir: Répertoire de sortie

    Returns:
        Chemin du fichier exporté ou None
    """
    embedding_data = get_embeddings_for_analysis(doc_id, pipeline_name, data_dir)
    if embedding_data is None:
        return None

    embeddings, texts = embedding_data

    # Créer le répertoire de sortie
    export_path = Path(output_dir) / doc_id
    export_path.mkdir(parents=True, exist_ok=True)

    # Exporter les embeddings en format numpy
    embeddings_file = export_path / f"{pipeline_name}_embeddings.npy"
    np.save(embeddings_file, embeddings)

    # Exporter les textes et métadonnées
    texts_file = export_path / f"{pipeline_name}_texts.json"
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    # Créer un fichier de métadonnées d'export
    metadata_file = export_path / f"{pipeline_name}_export_metadata.json"
    export_metadata = {
        "doc_id": doc_id,
        "pipeline_name": pipeline_name,
        "export_timestamp": time.time(),
        "embedding_shape": embeddings.shape,
        "num_texts": len(texts),
        "files": {
            "embeddings": str(embeddings_file.name),
            "texts": str(texts_file.name)
        }
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(export_metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Embeddings exportés vers: {export_path}")
    print(f"   - Embeddings: {embeddings_file}")
    print(f"   - Textes: {texts_file}")
    print(f"   - Métadonnées: {metadata_file}")

    return str(metadata_file)

def run_comprehensive_embedding_analysis(doc_id: str,
                                        data_dir: str = "data",
                                        export: bool = True) -> Dict:
    """
    Lance une analyse complète des embeddings pour un document

    Args:
        doc_id: Identifiant du document
        data_dir: Répertoire de données
        export: Exporter les résultats vers des fichiers

    Returns:
        Dictionnaire complet des analyses
    """
    print(f"🚀 Lancement de l'analyse complète des embeddings pour: {doc_id}")

    # 1. Analyse des métriques techniques
    technical_analysis = analyze_pipeline_embeddings(doc_id, data_dir)

    # 2. Exportation si demandée
    export_paths = {}
    if export:
        print(f"\n💾 Exportation des embeddings...")
        for pipeline in ['semantic', 'azure_semantic']:
            export_path = export_embeddings_for_external_analysis(doc_id, pipeline, data_dir)
            if export_path:
                export_paths[pipeline] = export_path

    # 3. Résumé de l'analyse
    analysis_summary = generate_analysis_summary(technical_analysis)

    comprehensive_results = {
        "doc_id": doc_id,
        "timestamp": time.time(),
        "technical_analysis": technical_analysis,
        "analysis_summary": analysis_summary,
        "export_paths": export_paths,
        "recommendations": generate_recommendations(technical_analysis)
    }

    # Afficher le résumé
    display_analysis_summary(analysis_summary)

    return comprehensive_results

def generate_analysis_summary(technical_analysis: Dict) -> Dict:
    """
    Génère un résumé de l'analyse technique
    """
    summary = {
        "pipelines_analyzed": list(technical_analysis.keys()),
        "comparison": {}
    }

    # Comparer les pipelines
    available_pipelines = [p for p in technical_analysis.keys() if "error" not in technical_analysis[p]]

    if len(available_pipelines) >= 2:
        # Comparaison entre pipelines
        comparison = {}
        for metric_category in ["diversity", "semantic_coherence"]:
            comparison[metric_category] = {}
            for pipeline in available_pipelines:
                if metric_category in technical_analysis[pipeline]:
                    if metric_category == "diversity":
                        score = technical_analysis[pipeline][metric_category].get("diversity_score", 0)
                    elif metric_category == "semantic_coherence":
                        score = technical_analysis[pipeline][metric_category].get("semantic_coherence", 0)
                    comparison[metric_category][pipeline] = score

        summary["comparison"] = comparison

    return summary

def display_analysis_summary(summary: Dict):
    """
    Affiche un résumé de l'analyse
    """
    print(f"\n📊 RÉSUMÉ DE L'ANALYSE")
    print(f"=" * 50)

    pipelines = summary.get("pipelines_analyzed", [])
    print(f"📋 Pipelines analysés: {', '.join(pipelines)}")

    comparison = summary.get("comparison", {})
    if comparison:
        print(f"\n🔍 Comparaison des performances:")

        for metric_category, scores in comparison.items():
            if scores:
                print(f"\n  {metric_category.title()}:")
                best_pipeline = max(scores, key=scores.get)
                for pipeline, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    icon = "🥇" if pipeline == best_pipeline else "📊"
                    print(f"    {icon} {pipeline}: {score:.3f}")

def generate_recommendations(technical_analysis: Dict) -> List[str]:
    """
    Génère des recommandations basées sur l'analyse technique
    """
    recommendations = []

    available_pipelines = [p for p in technical_analysis.keys() if "error" not in technical_analysis[p]]

    if not available_pipelines:
        recommendations.append("❌ Aucun pipeline d'embedding disponible pour l'analyse")
        return recommendations

    # Analyse de la diversité
    diversity_scores = {}
    coherence_scores = {}

    for pipeline in available_pipelines:
        analysis = technical_analysis[pipeline]
        if "diversity" in analysis:
            diversity_scores[pipeline] = analysis["diversity"].get("diversity_score", 0)
        if "semantic_coherence" in analysis:
            coherence_scores[pipeline] = analysis["semantic_coherence"].get("semantic_coherence", 0)

    # Recommandations basées sur la diversité
    if diversity_scores:
        best_diversity = max(diversity_scores, key=diversity_scores.get)
        if diversity_scores[best_diversity] > 0.3:
            recommendations.append(f"✅ {best_diversity} montre une bonne diversité d'embeddings ({diversity_scores[best_diversity]:.3f})")
        else:
            recommendations.append(f"⚠️ Diversité d'embeddings faible. Considérez augmenter la variété des chunks")

    # Recommandations basées sur la cohérence
    if coherence_scores:
        best_coherence = max(coherence_scores, key=coherence_scores.get)
        if coherence_scores[best_coherence] > 0.5:
            recommendations.append(f"✅ {best_coherence} montre une bonne cohérence sémantique ({coherence_scores[best_coherence]:.3f})")
        else:
            recommendations.append(f"⚠️ Cohérence sémantique faible. Le modèle d'embedding pourrait être amélioré")

    # Recommandation générale
    if len(available_pipelines) == 1:
        recommendations.append("💡 Considérez tester plusieurs pipelines d'embedding pour comparison")

    return recommendations