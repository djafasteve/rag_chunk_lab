
from typing import List, Dict, Optional
import json, re, os
import pandas as pd
from tqdm import tqdm
from .embedding_metrics import (
    EmbeddingMetrics,
    EmbeddingQualityAnalyzer,
    evaluate_retrieval_performance,
    export_embedding_analysis
)

def load_ground_truth(path: str) -> List[Dict]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def local_proxy_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb)/len(wa | wb)

def evaluate_local_proxy(answers: List[str], truths: List[str]) -> Dict[str, float]:
    scores = [local_proxy_overlap(a, t) for a,t in zip(answers, truths)]
    return {'proxy_similarity_mean': sum(scores)/len(scores) if scores else 0.0}

def try_ragas_eval(questions: List[str],
                   truths: List[str],
                   per_pipeline_answers: Dict[str, List[str]],
                   per_pipeline_contexts: Dict[str, List[List[str]]]) -> Optional[Dict]:
    """
    Attempts to run RAGAS with standard metrics. Returns a dict with:
      - 'summary': metric means per pipeline
      - 'per_question': list of rows per pipeline
      - 'metrics_order': display order of metrics
    Returns None if RAGAS or LLM backend is not available.
    """
    try:
        from datasets import Dataset
        from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
        from ragas import evaluate as ragas_evaluate
        import os

        # Check if Azure OpenAI is configured
        required_env_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_DEPLOYMENT',
            'AZURE_OPENAI_API_VERSION'
        ]

        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Missing Azure OpenAI environment variables: {missing_vars}")
            return {"error": f"Missing environment variables: {', '.join(missing_vars)}"}

        # Configure Azure OpenAI for RAGAS 0.3.4
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # Initialize Azure OpenAI components
        azure_llm = AzureChatOpenAI(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            temperature=0
        )

        azure_embeddings = AzureOpenAIEmbeddings(
            model=os.getenv('AZURE_OPENAI_EMBEDDING', 'text-embedding-3-small'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION')
        )

        # Wrap for RAGAS
        ragas_llm = LangchainLLMWrapper(azure_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)

        # Initialize metrics with custom LLM/embeddings
        from copy import deepcopy
        metrics = []

        # Create instances of metrics with custom LLM/embeddings
        ar = deepcopy(answer_relevancy)
        ar.llm = ragas_llm
        ar.embeddings = ragas_embeddings
        metrics.append(ar)

        f = deepcopy(faithfulness)
        f.llm = ragas_llm
        metrics.append(f)

        cp = deepcopy(context_precision)
        cp.llm = ragas_llm
        metrics.append(cp)

        cr = deepcopy(context_recall)
        cr.llm = ragas_llm
        cr.embeddings = ragas_embeddings
        metrics.append(cr)

        pipelines = list(per_pipeline_answers.keys())
        summary = {}
        per_q = {}
        metrics_order = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]

        print(f"\n🔄 Starting RAGAS evaluation for {len(pipelines)} pipelines with {len(questions)} questions...")
        print(f"📊 Metrics: {', '.join(metrics_order)}")

        # Progress bar for pipelines
        with tqdm(pipelines, desc="🚀 Evaluating pipelines", unit="pipeline",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for p in pbar:
                pbar.set_description(f"🔍 Evaluating {p}")

                # Debug: Check data consistency
                answers = per_pipeline_answers[p]
                contexts = per_pipeline_contexts[p]

                print(f"\n  🔍 Debug {p} pipeline:")
                print(f"    Questions: {len(questions)}")
                print(f"    Answers: {len(answers)}")
                print(f"    Contexts: {len(contexts)}")
                print(f"    Truths: {len(truths)}")

                # Check for empty contexts or answers
                empty_contexts = sum(1 for ctx in contexts if not ctx or len(ctx) == 0)
                empty_answers = sum(1 for ans in answers if not ans or len(ans.strip()) == 0)

                print(f"    Empty contexts: {empty_contexts}")
                print(f"    Empty answers: {empty_answers}")

                # Filter out problematic entries
                valid_data = []
                for i in range(len(questions)):
                    if (i < len(answers) and i < len(contexts) and
                        answers[i] and answers[i].strip() and
                        contexts[i] and len(contexts[i]) > 0 and
                        all(ctx.strip() for ctx in contexts[i])):
                        valid_data.append({
                            "question": questions[i],
                            "answer": answers[i],
                            "contexts": contexts[i],
                            "ground_truth": truths[i]
                        })

                print(f"    Valid entries: {len(valid_data)}/{len(questions)}")

                if len(valid_data) == 0:
                    print(f"  ❌ No valid data for {p} pipeline - skipping")
                    summary[p] = {m: None for m in metrics_order}
                    per_q[p] = []
                    continue

                # Create dataset with valid data only
                data = {
                    "question": [item["question"] for item in valid_data],
                    "answer": [item["answer"] for item in valid_data],
                    "contexts": [item["contexts"] for item in valid_data],
                    "ground_truth": [item["ground_truth"] for item in valid_data]
                }
                ds = Dataset.from_dict(data)

                # Show progress for RAGAS evaluation
                print(f"\n  📝 Running RAGAS metrics for '{p}' pipeline...")
                res = ragas_evaluate(ds, metrics)

                # Convert to pandas DataFrame for easier access
                df = res.to_pandas()

                # Calculate summary scores
                s = {}
                for m in metrics_order:
                    if m in df.columns:
                        scores = df[m].dropna()  # Remove NaN values
                        if len(scores) > 0:
                            s[m] = float(scores.mean())
                        else:
                            s[m] = None
                    else:
                        s[m] = None
                summary[p] = s

                # Show results for current pipeline
                results_str = ", ".join([f"{m}: {s.get(m, 'N/A'):.3f}" if s.get(m) is not None else f"{m}: N/A" for m in metrics_order[:2]])
                print(f"  ✅ {p}: {results_str}")

                # Create detailed rows for per-question analysis
                rows = []
                n = len(questions)
                for i in range(n):
                    for m in metrics_order:
                        v = None
                        if m in df.columns and i < len(df):
                            v = df.iloc[i][m]
                            try:
                                v = float(v) if v is not None and not pd.isna(v) else None
                            except Exception:
                                v = None
                        rows.append({
                            "i": i,
                            "question": questions[i],
                            "answer": per_pipeline_answers[p][i],
                            "truth": truths[i],
                            "metric": m,
                            "value": v
                        })
                per_q[p] = rows

        print(f"\n✨ RAGAS evaluation completed!")
        print(f"📋 Summary:")
        for pipe, scores in summary.items():
            avg_score = sum(v for v in scores.values() if v is not None) / len([v for v in scores.values() if v is not None])
            print(f"  {pipe}: avg={avg_score:.3f}")

        return {"summary": summary, "per_question": per_q, "metrics_order": metrics_order}
    except Exception as e:
        print(f"RAGAS evaluation error: {str(e)}")
        return {"error": str(e)}

def evaluate_embedding_quality(doc_id: str,
                              questions: List[str],
                              per_pipeline_answers: Dict[str, List[str]],
                              per_pipeline_contexts: Dict[str, List[List[str]]],
                              ground_truth: List[Dict],
                              include_retrieval_metrics: bool = True,
                              include_technical_analysis: bool = True,
                              k_values: List[int] = [3, 5, 10, 15]) -> Dict:
    """
    Évaluation complète de la qualité des embeddings et de la récupération

    Args:
        doc_id: Identifiant du document
        questions: Liste des questions
        per_pipeline_answers: Réponses par pipeline
        per_pipeline_contexts: Contextes récupérés par pipeline
        ground_truth: Ground truth
        include_retrieval_metrics: Inclure les métriques de récupération
        include_technical_analysis: Inclure l'analyse technique des embeddings
        k_values: Valeurs de K pour Recall@K

    Returns:
        Dictionnaire complet des métriques
    """
    print(f"\n🎯 Évaluation complète des embeddings pour: {doc_id}")
    results = {}

    # 1. Métriques de récupération (Recall@K, MRR, NDCG)
    if include_retrieval_metrics:
        print(f"\n📊 Calcul des métriques de récupération...")

        # Convertir les contextes en format attendu pour les métriques
        retrieved_results = {}
        for pipeline, contexts in per_pipeline_contexts.items():
            pipeline_chunks = []
            for context_list in contexts:
                chunks = [{"text": text, "score": 1.0} for text in context_list]
                pipeline_chunks.append(chunks)
            retrieved_results[pipeline] = pipeline_chunks

        retrieval_metrics = evaluate_retrieval_performance(
            questions=questions,
            retrieved_results=retrieved_results,
            ground_truth=ground_truth,
            k_values=k_values
        )
        results["retrieval_metrics"] = retrieval_metrics

    # 2. Analyse technique des embeddings (si disponible)
    if include_technical_analysis:
        print(f"\n🔬 Analyse technique des embeddings...")
        try:
            # Import de l'analyse des embeddings
            from .embedding_analysis import analyze_pipeline_embeddings
            import os

            # Utiliser le répertoire de données par défaut
            data_dir = os.environ.get('RAG_LAB_DATA', 'data')

            # Analyser les pipelines sémantiques
            semantic_pipelines = [p for p in per_pipeline_contexts.keys() if p in ['semantic', 'azure_semantic']]

            if semantic_pipelines:
                technical_analysis_results = analyze_pipeline_embeddings(
                    doc_id=doc_id,
                    data_dir=data_dir,
                    pipelines=semantic_pipelines
                )
                results["technical_analysis"] = technical_analysis_results
            else:
                results["technical_analysis"] = {
                    "note": "Aucun pipeline sémantique disponible pour l'analyse technique",
                    "available_pipelines": list(per_pipeline_contexts.keys())
                }

        except Exception as e:
            print(f"⚠️ Analyse technique non disponible: {e}")
            results["technical_analysis"] = {"error": str(e)}

    # 3. Métriques de comparaison entre pipelines
    print(f"\n📈 Analyse comparative des pipelines...")
    pipeline_comparison = compare_pipeline_performance(
        per_pipeline_answers,
        per_pipeline_contexts,
        questions
    )
    results["pipeline_comparison"] = pipeline_comparison

    # 4. Métriques spécifiques aux embeddings pour RAGAS
    print(f"\n🎯 Métriques RAGAS spécialisées pour embeddings...")
    embedding_focused_metrics = calculate_embedding_focused_ragas_metrics(
        questions,
        per_pipeline_answers,
        per_pipeline_contexts
    )
    results["embedding_focused_metrics"] = embedding_focused_metrics

    return results

def compare_pipeline_performance(per_pipeline_answers: Dict[str, List[str]],
                               per_pipeline_contexts: Dict[str, List[List[str]]],
                               questions: List[str]) -> Dict:
    """
    Compare les performances entre pipelines
    """
    comparison = {}
    pipelines = list(per_pipeline_answers.keys())

    # Statistiques de base
    for pipeline in pipelines:
        answers = per_pipeline_answers[pipeline]
        contexts = per_pipeline_contexts[pipeline]

        # Longueur moyenne des réponses
        avg_answer_length = sum(len(ans.split()) for ans in answers) / len(answers) if answers else 0

        # Nombre moyen de contextes récupérés
        avg_context_count = sum(len(ctx) for ctx in contexts) / len(contexts) if contexts else 0

        # Longueur moyenne des contextes
        total_context_words = sum(len(text.split()) for ctx in contexts for text in ctx)
        avg_context_length = total_context_words / sum(len(ctx) for ctx in contexts) if sum(len(ctx) for ctx in contexts) > 0 else 0

        comparison[pipeline] = {
            "avg_answer_length": avg_answer_length,
            "avg_context_count": avg_context_count,
            "avg_context_length": avg_context_length,
            "total_questions": len(questions),
            "successful_retrievals": sum(1 for ctx in contexts if ctx)
        }

    return comparison

def calculate_embedding_focused_ragas_metrics(questions: List[str],
                                           per_pipeline_answers: Dict[str, List[str]],
                                           per_pipeline_contexts: Dict[str, List[List[str]]]) -> Dict:
    """
    Calcule des métriques RAGAS spécialement focalisées sur la qualité des embeddings
    """
    metrics = {}

    for pipeline in per_pipeline_answers.keys():
        answers = per_pipeline_answers[pipeline]
        contexts = per_pipeline_contexts[pipeline]

        # Métrique 1: Context Quality Score
        context_quality_scores = []
        for i, (question, context_list) in enumerate(zip(questions, contexts)):
            if context_list:
                # Calculer la pertinence du contexte récupéré
                quality_score = calculate_context_relevance_score(question, context_list)
                context_quality_scores.append(quality_score)

        avg_context_quality = sum(context_quality_scores) / len(context_quality_scores) if context_quality_scores else 0

        # Métrique 2: Retrieval Consistency
        consistency_score = calculate_retrieval_consistency(contexts)

        # Métrique 3: Embedding Coverage
        coverage_score = calculate_embedding_coverage(questions, contexts)

        metrics[pipeline] = {
            "context_quality": avg_context_quality,
            "retrieval_consistency": consistency_score,
            "embedding_coverage": coverage_score,
            "sample_size": len(questions)
        }

    return metrics

def calculate_context_relevance_score(question: str, context_list: List[str]) -> float:
    """
    Calcule un score de pertinence du contexte récupéré par rapport à la question
    """
    if not context_list or not question:
        return 0.0

    question_words = set(question.lower().split())
    relevance_scores = []

    for context in context_list:
        if not context:
            continue
        context_words = set(context.lower().split())

        # Jaccard similarity
        intersection = len(question_words & context_words)
        union = len(question_words | context_words)
        jaccard = intersection / union if union > 0 else 0.0
        relevance_scores.append(jaccard)

    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

def calculate_retrieval_consistency(contexts: List[List[str]]) -> float:
    """
    Mesure la consistance de la récupération (variance dans le nombre de chunks récupérés)
    """
    if not contexts:
        return 0.0

    context_counts = [len(ctx) for ctx in contexts]
    if not context_counts:
        return 0.0

    # Plus la variance est faible, plus la consistance est élevée
    import numpy as np
    variance = np.var(context_counts)
    mean_count = np.mean(context_counts)

    # Normaliser : consistance = 1 - (variance / mean^2)
    consistency = 1.0 - (variance / (mean_count ** 2)) if mean_count > 0 else 0.0
    return max(0.0, consistency)

def calculate_embedding_coverage(questions: List[str], contexts: List[List[str]]) -> float:
    """
    Mesure la couverture : proportion de questions pour lesquelles du contexte a été récupéré
    """
    if not questions or not contexts:
        return 0.0

    successful_retrievals = sum(1 for ctx in contexts if ctx and any(text.strip() for text in ctx))
    coverage = successful_retrievals / len(questions)
    return coverage
