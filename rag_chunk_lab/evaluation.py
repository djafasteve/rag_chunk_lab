
from typing import List, Dict, Optional
import json, re, os
import pandas as pd
from tqdm import tqdm

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
