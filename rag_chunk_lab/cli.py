import typer, os, json, csv
from pathlib import Path
from tqdm import tqdm
from .config import DEFAULTS
from .utils import load_document
from .chunkers import fixed_chunks, structure_aware_chunks, sliding_window_chunks
from .indexing import build_index
from .retrieval import get_candidates
from .generation import build_answer_payload
from .evaluation import load_ground_truth, evaluate_local_proxy, try_ragas_eval
from .ground_truth_generator import GroundTruthGenerator, create_llm_client

app = typer.Typer(help='RAG Chunk Lab — compare chunking strategies for RAG quality.')
DATA_DIR = os.environ.get('RAG_LAB_DATA', 'data')

def _export_ragas_results(doc_id: str, ragas_result: dict):
    """Export RAGAS results to CSV files"""
    export_dir = Path('exports') / doc_id
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export summary (pipeline x metrics)
    summary_file = export_dir / 'ragas_summary.csv'
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        metrics = ragas_result['metrics_order']
        writer.writerow(['pipeline'] + metrics)
        for pipe, scores in ragas_result['summary'].items():
            row = [pipe]
            for metric in metrics:
                row.append(scores.get(metric, ''))
            writer.writerow(row)

    # Export per-question details
    per_q_file = export_dir / 'ragas_per_question.csv'
    with open(per_q_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pipeline', 'question_idx', 'question', 'answer', 'ground_truth', 'metric', 'value'])
        for pipe, rows in ragas_result['per_question'].items():
            for row in rows:
                writer.writerow([
                    pipe,
                    row['i'],
                    row['question'],
                    row['answer'],
                    row['truth'],
                    row['metric'],
                    row['value']
                ])

    typer.echo(f'RAGAS results exported to {export_dir}/ragas_summary.csv and ragas_per_question.csv')

@app.command()
def ingest(doc: str = typer.Option(..., '--doc'),
           doc_id: str = typer.Option(..., '--doc-id'),
           fixed_size: int = typer.Option(DEFAULTS.fixed_size_tokens, '--fixed-size'),
           fixed_overlap: int = typer.Option(DEFAULTS.fixed_overlap_tokens, '--fixed-overlap'),
           win: int = typer.Option(DEFAULTS.sliding_window, '--window'),
           stride: int = typer.Option(DEFAULTS.sliding_stride, '--stride')):
    pages = load_document(doc)
    fx = fixed_chunks(pages, fixed_size, fixed_overlap, doc_id)
    build_index(doc_id, 'fixed', fx, DATA_DIR)
    sa = structure_aware_chunks(pages, fixed_size, fixed_overlap, doc_id)
    build_index(doc_id, 'structure', sa, DATA_DIR)
    sw = sliding_window_chunks(pages, win, stride, doc_id)
    build_index(doc_id, 'sliding', sw, DATA_DIR)
    typer.echo(f'Ingested {doc_id}. Pipelines built: fixed({len(fx)}) / structure({len(sa)}) / sliding({len(sw)})')

@app.command()
def ask(doc_id: str = typer.Option(..., '--doc-id'),
        question: str = typer.Option(..., '--question'),
        top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
        max_sentences: int = typer.Option(DEFAULTS.max_sentences, '--max-sentences'),
        use_llm: bool = typer.Option(False, '--use-llm')):
    results = []
    for pipe in ['fixed', 'structure', 'sliding']:
        cands = get_candidates(doc_id, question, pipe, top_k, DATA_DIR)
        payload = build_answer_payload(pipe, question, cands, max_sentences=max_sentences, use_llm=use_llm)
        results.append(payload)
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))

@app.command()
def evaluate(doc_id: str = typer.Option(..., '--doc-id'),
             ground_truth: str = typer.Option(..., '--ground-truth'),
             top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
             local_proxy: bool = typer.Option(True, '--local-proxy/--no-local-proxy'),
             ragas: bool = typer.Option(False, '--ragas'),
             use_llm: bool = typer.Option(False, '--use-llm')):
    gts = load_ground_truth(ground_truth)
    questions = [it['question'] for it in gts]
    truths = [it['answer'] for it in gts]

    # Collect answers and contexts for all pipelines
    per_pipeline_answers = {}
    per_pipeline_contexts = {}

    print(f"\n🔄 Collecting answers from {len(['fixed', 'structure', 'sliding'])} pipelines for {len(questions)} questions...")

    with tqdm(['fixed', 'structure', 'sliding'], desc="📊 Processing pipelines", unit="pipeline") as pipe_bar:
        for pipe in pipe_bar:
            pipe_bar.set_description(f"📝 Processing {pipe}")

            answers = []
            contexts = []

            with tqdm(questions, desc=f"  💭 {pipe} questions", leave=False, unit="q") as q_bar:
                for q in q_bar:
                    q_preview = q[:50] + "..." if len(q) > 50 else q
                    q_bar.set_description(f"  💭 {pipe}: {q_preview}")

                    cands = get_candidates(doc_id, q, pipe, top_k, os.environ.get('RAG_LAB_DATA', 'data'))
                    payload = build_answer_payload(pipe, q, cands, max_sentences=4, use_llm=use_llm)
                    answers.append(payload['answer'])
                    # Extract context from candidates
                    context_list = [cand['text'] for cand in cands]
                    contexts.append(context_list)

            per_pipeline_answers[pipe] = answers
            per_pipeline_contexts[pipe] = contexts

    print("✅ Answer collection completed!")

    report = {}

    # Local proxy evaluation
    if local_proxy:
        for pipe in ['fixed', 'structure', 'sliding']:
            rep = evaluate_local_proxy(per_pipeline_answers[pipe], truths)
            report[f'{pipe}_proxy'] = rep

    # RAGAS evaluation
    if ragas and use_llm:
        print(f"\n🎯 Starting RAGAS evaluation...")
        ragas_result = try_ragas_eval(questions, truths, per_pipeline_answers, per_pipeline_contexts)
        if ragas_result and 'error' not in ragas_result and 'summary' in ragas_result:
            report['ragas'] = ragas_result
            # Export detailed results
            print(f"\n💾 Exporting results...")
            _export_ragas_results(doc_id, ragas_result)
        else:
            error_msg = ragas_result.get('error', 'Unknown error') if ragas_result else 'RAGAS evaluation failed'
            report['ragas_error'] = error_msg
            print(f"❌ RAGAS evaluation failed: {error_msg}")
    elif ragas and not use_llm:
        report['ragas_warning'] = 'RAGAS requires --use-llm flag to access LLM for evaluation.'
        print("⚠️  RAGAS requires --use-llm flag to access LLM for evaluation.")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command(name='generate-ground-truth')
def generate_ground_truth(
    folder: str = typer.Option(..., help='Dossier contenant les documents'),
    output: str = typer.Option(None, help='Fichier JSONL de sortie'),
    llm_provider: str = typer.Option('ollama', help='Provider LLM: ollama ou azure'),
    model: str = typer.Option(None, help='Modèle à utiliser'),
    questions_per_doc: int = typer.Option(10, help='Nombre de questions par document'),
    min_length: int = typer.Option(200, help='Longueur minimale du texte'),
    max_length: int = typer.Option(800, help='Longueur maximale du texte'),
    ollama_url: str = typer.Option('http://localhost:11434', help='URL du serveur Ollama')
):
    """
    Génère automatiquement un dataset de ground truth à partir d'un dossier de documents.

    Utilise un LLM (Ollama local ou Azure OpenAI) pour créer des paires question/réponse
    expertes à partir de chunks aléatoires extraits des documents.

    Exemples:

    # Avec Ollama local (mistral:7b)
    python -m rag_chunk_lab.cli generate-ground-truth --folder documents/code_penal

    # Avec Azure OpenAI
    python -m rag_chunk_lab.cli generate-ground-truth --folder documents/code_penal --llm-provider azure

    # Avec modèle personnalisé
    python -m rag_chunk_lab.cli generate-ground-truth --folder docs --model llama3:8b --questions-per-doc 15
    """

    folder_path = Path(folder)

    # Définir le fichier de sortie
    if not output:
        output = f"{folder_path.name}_ground_truth.jsonl"

    print(f"🚀 Démarrage de la génération de ground truth")
    print(f"📁 Dossier source: {folder}")
    print(f"🤖 LLM Provider: {llm_provider}")
    print(f"📊 Questions par document: {questions_per_doc}")
    print(f"💾 Fichier de sortie: {output}")

    try:
        # Créer le client LLM
        print(f"\n🔌 Connexion au LLM...")

        if llm_provider == 'ollama':
            llm_client = create_llm_client('ollama', model, base_url=ollama_url)
            print(f"✅ Connecté à Ollama: {llm_client.model}")
        elif llm_provider == 'azure':
            # Vérifier les variables d'environnement
            required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                typer.echo(f"❌ Variables d'environnement manquantes: {', '.join(missing_vars)}")
                raise typer.Exit(1)

            llm_client = create_llm_client('azure', model)
            print(f"✅ Connecté à Azure OpenAI: {llm_client.model}")
        else:
            typer.echo(f"❌ Provider LLM non supporté: {llm_provider}")
            raise typer.Exit(1)

        # Créer le générateur
        generator = GroundTruthGenerator(llm_client)

        # Générer le dataset
        output_path = generator.generate_from_folder(
            folder_path=folder,
            output_path=output,
            questions_per_doc=questions_per_doc,
            min_text_length=min_length,
            max_text_length=max_length
        )

        print(f"\n🎉 Génération terminée avec succès!")
        print(f"📄 Dataset sauvegardé: {output_path}")
        print(f"\n💡 Utilisez maintenant:")
        print(f"   python -m rag_chunk_lab.cli evaluate --doc-id <votre-doc> --ground-truth {output_path} --ragas --use-llm")

    except ConnectionError as e:
        typer.echo(f"❌ Erreur de connexion LLM: {e}")
        if llm_provider == 'ollama':
            typer.echo("💡 Assurez-vous qu'Ollama est démarré: ollama serve")
            typer.echo(f"💡 Et que le modèle est installé: ollama pull {model or 'mistral:7b'}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Erreur lors de la génération: {e}")
        raise typer.Exit(1)

if __name__ == '__main__':
    app()
