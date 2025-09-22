import typer, os, json, csv
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import DEFAULTS
from .utils import load_document
from .chunkers import fixed_chunks, structure_aware_chunks, sliding_window_chunks, semantic_chunks, azure_semantic_chunks
from .indexing import build_index
from .retrieval import get_candidates
from .generation import build_answer_payload
from .evaluation import load_ground_truth, evaluate_local_proxy, try_ragas_eval, evaluate_embedding_quality
from .ground_truth_generator import GroundTruthGenerator, create_llm_client
from .monitoring import print_performance_summary

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
    """
    Ingest a document or folder of documents.

    If --doc points to a file: processes that single document
    If --doc points to a folder: processes all supported documents (.pdf, .txt, .md) in that folder
    """
    from pathlib import Path

    doc_path = Path(doc)
    all_pages = []

    if doc_path.is_file():
        # Single document
        typer.echo(f"📄 Processing single document: {doc_path.name}")
        all_pages = load_document(str(doc_path))

    elif doc_path.is_dir():
        # Folder of documents
        doc_files = []
        for ext in ['*.pdf', '*.txt', '*.md']:
            doc_files.extend(doc_path.glob(ext))

        if not doc_files:
            typer.echo(f"❌ No supported documents (.pdf, .txt, .md) found in {doc_path}")
            raise typer.Exit(1)

        typer.echo(f"📁 Processing {len(doc_files)} documents from folder: {doc_path}")

        for doc_file in tqdm(doc_files, desc="Loading documents"):
            try:
                pages = load_document(str(doc_file))
                # Add source document info to each page
                for page in pages:
                    page['source_file'] = doc_file.name
                all_pages.extend(pages)
                typer.echo(f"  ✅ Loaded {doc_file.name} ({len(pages)} pages)")
            except Exception as e:
                typer.echo(f"  ❌ Failed to load {doc_file.name}: {e}")
                continue

        if not all_pages:
            typer.echo("❌ No pages could be loaded from any document")
            raise typer.Exit(1)

    else:
        typer.echo(f"❌ Path not found: {doc_path}")
        raise typer.Exit(1)

    # Process all pages together under the same doc_id
    typer.echo(f"🔧 Building indexes for {len(all_pages)} total pages...")

    # OPTIMISATION: Parallélisation des pipelines d'ingestion
    def build_pipeline(pipeline_name, chunker_func, *args):
        """Helper function pour construire un pipeline en parallèle"""
        try:
            chunks = chunker_func(all_pages, *args, doc_id)
            build_index(doc_id, pipeline_name, chunks, DATA_DIR)
            return pipeline_name, len(chunks), None
        except Exception as e:
            return pipeline_name, 0, str(e)

    # Définition des pipelines à traiter en parallèle
    pipeline_tasks = [
        ('fixed', fixed_chunks, fixed_size, fixed_overlap),
        ('structure', structure_aware_chunks, fixed_size, fixed_overlap),
        ('sliding', sliding_window_chunks, win, stride)
    ]

    # Pipelines optionnels (peuvent échouer)
    optional_tasks = [
        ('semantic', semantic_chunks, fixed_size, fixed_overlap),
        ('azure_semantic', azure_semantic_chunks, fixed_size, fixed_overlap)
    ]

    results = {}

    # Traitement parallèle des pipelines de base
    print("🚀 Traitement parallèle des pipelines de base...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_pipeline = {
            executor.submit(build_pipeline, name, func, *args): name
            for name, func, *args in pipeline_tasks
        }

        for future in as_completed(future_to_pipeline):
            pipeline_name = future_to_pipeline[future]
            name, count, error = future.result()
            if error:
                typer.echo(f"❌ Échec pipeline {name}: {error}")
            else:
                results[name] = count
                typer.echo(f"✅ Pipeline {name}: {count} chunks")

    # Pipelines sémantiques (séquentiels car peuvent partager des ressources)
    typer.echo("🧠 Traitement des pipelines sémantiques...")
    for name, func, *args in optional_tasks:
        pipeline_name, count, error = build_pipeline(name, func, *args)
        if error:
            if 'ImportError' in str(error) or 'sentence-transformers' in str(error):
                typer.echo(f'⚠️  Pipeline {name} ignoré: dépendance manquante')
            else:
                typer.echo(f'⚠️  Pipeline {name} ignoré: {error}')
        else:
            results[name] = count
            icon = '🧠' if name == 'semantic' else '☁️'
            typer.echo(f"✅ Pipeline {name}: {count} chunks {icon}")

    # Message de statut final
    status_parts = [f'{name}({count})' for name, count in results.items()]
    typer.echo(f'✅ Ingested {doc_id}. Pipelines built: {" / ".join(status_parts)}')

    # Affichage des métriques de performance
    print_performance_summary()

@app.command()
def ask(doc_id: str = typer.Option(..., '--doc-id'),
        question: str = typer.Option(..., '--question'),
        top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
        max_sentences: int = typer.Option(DEFAULTS.max_sentences, '--max-sentences'),
        use_llm: bool = typer.Option(False, '--use-llm'),
        include_semantic: bool = typer.Option(True, '--semantic/--no-semantic', help='Inclure le pipeline sémantique local'),
        include_azure_semantic: bool = typer.Option(True, '--azure-semantic/--no-azure-semantic', help='Inclure le pipeline sémantique Azure')):
    """
    Compare les réponses des différentes stratégies de chunking.

    Maintenant avec 5 pipelines :
    - fixed: Chunks de taille fixe
    - structure: Chunks conscients de la structure
    - sliding: Fenêtre glissante
    - semantic: Recherche sémantique locale (🆕)
    - azure_semantic: Recherche sémantique Azure (🆕)
    """
    results = []
    pipelines = ['fixed', 'structure', 'sliding']

    if include_semantic:
        pipelines.append('semantic')

    if include_azure_semantic:
        pipelines.append('azure_semantic')

    for pipe in pipelines:
        try:
            cands = get_candidates(doc_id, question, pipe, top_k, DATA_DIR)
            payload = build_answer_payload(pipe, question, cands, max_sentences=max_sentences, use_llm=use_llm)
            results.append(payload)
        except Exception as e:
            if pipe in ['semantic', 'azure_semantic']:
                typer.echo(f"⚠️  Pipeline {pipe} ignoré: {e}", err=True)
            else:
                raise e

    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))

@app.command()
def chat(doc_id: str = typer.Option(..., '--doc-id'),
         question: str = typer.Option(..., '--question'),
         top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
         provider: str = typer.Option(DEFAULTS.default_provider, '--provider', help='LLM provider: ollama or azure'),
         model: str = typer.Option(DEFAULTS.default_model, '--model', help='Model name (ex: mistral:7b, llama3:8b, votre-modele-juridique)'),
         pipeline: str = typer.Option('azure_semantic', '--pipeline', help='Chunking strategy: fixed, structure, sliding, semantic, azure_semantic')):
    """
    Chat with your documents using AI.

    Gets relevant chunks and asks an LLM to synthesize a comprehensive answer.
    Perfect for natural conversations with your document collection.
    """
    from .ground_truth_generator import create_llm_client

    # Get relevant chunks using the specified pipeline
    try:
        cands = get_candidates(doc_id, question, pipeline, top_k, DATA_DIR)
    except Exception as e:
        if pipeline in ['semantic', 'azure_semantic']:
            typer.echo(f"❌ Pipeline {pipeline} non disponible: {e}")
            typer.echo(f"💡 Conseil: Utilisez --pipeline fixed pour une alternative")
            raise typer.Exit(1)
        else:
            raise

    if not cands:
        typer.echo(f"❌ No relevant information found for: {question}")
        raise typer.Exit(1)

    # Prepare context from chunks
    context_parts = []
    for i, cand in enumerate(cands, 1):
        meta = cand.get('meta', {})
        source_file = meta.get('source_file', 'Document')
        page = meta.get('page', 'N/A')
        source_info = f"[Source {i}: {source_file} - Page {page}]"
        context_parts.append(f"{source_info}\n{cand['text']}")

    context = "\n\n".join(context_parts)

    # Create prompt for LLM
    prompt = f"""Tu es un assistant expert qui répond à des questions en te basant sur des documents fournis.

QUESTION: {question}

CONTEXTE DOCUMENTAIRE:
{context}

INSTRUCTIONS:
1. Réponds à la question en utilisant UNIQUEMENT les informations du contexte fourni
2. Sois précis et factuel
3. Si plusieurs sources sont pertinentes, synthétise-les
4. Si l'information n'est pas dans le contexte, dis-le clairement
5. Cite les sources quand c'est pertinent (ex: "Selon le document X, page Y...")

RÉPONSE:"""

    try:
        # Initialize LLM client
        typer.echo(f"🤖 Génération de la réponse avec {provider}:{model}...")
        llm_client = create_llm_client(provider, model)

        # Generate response
        response = llm_client.generate(prompt)

        # Display results
        typer.echo(f"\n💬 Question: {question}")
        typer.echo(f"📚 Sources consultées: {len(cands)} chunks (pipeline: {pipeline})")
        typer.echo(f"🤖 Réponse ({provider}:{model}):\n")
        typer.echo(response)

        # Show sources
        typer.echo(f"\n📖 Sources détaillées:")
        for i, cand in enumerate(cands, 1):
            score = cand.get('score', 0)
            meta = cand.get('meta', {})
            source_file = meta.get('source_file', 'Document')
            doc_id = meta.get('doc_id', 'N/A')
            page = meta.get('page', 'N/A')
            snippet = cand.get('text', '')[:150] + '...' if len(cand.get('text', '')) > 150 else cand.get('text', '')
            typer.echo(f"  {i}. {source_file} (collection: {doc_id}, page {page}) - Score: {score:.3f}")
            typer.echo(f"     \"{snippet}\"")

    except Exception as e:
        typer.echo(f"❌ Erreur lors de la génération: {e}")
        typer.echo(f"\n📋 Chunks trouvés sans IA:")
        for i, cand in enumerate(cands, 1):
            meta = cand.get('meta', {})
            source_file = meta.get('source_file', 'Document')
            doc_id = meta.get('doc_id', 'N/A')
            page = meta.get('page', 'N/A')
            typer.echo(f"  {i}. {source_file} (collection: {doc_id}, page {page})")
            typer.echo(f"     {cand.get('text', '')[:200]}...")
        raise typer.Exit(1)

@app.command()
def evaluate(doc_id: str = typer.Option(..., '--doc-id'),
             ground_truth: str = typer.Option(..., '--ground-truth'),
             top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
             local_proxy: bool = typer.Option(True, '--local-proxy/--no-local-proxy'),
             ragas: bool = typer.Option(False, '--ragas'),
             use_llm: bool = typer.Option(False, '--use-llm'),
             embedding_analysis: bool = typer.Option(False, '--embedding-analysis', help='Inclure l\'analyse avancée des embeddings'),
             legal_evaluation: bool = typer.Option(False, '--legal-evaluation', help='Inclure l\'évaluation juridique spécialisée'),
             generic_evaluation: bool = typer.Option(False, '--generic-evaluation', help='Inclure l\'évaluation générique complète'),
             azure_foundry: bool = typer.Option(False, '--azure-foundry', help='Utiliser Azure AI Foundry pour l\'évaluation'),
             trulens: bool = typer.Option(False, '--trulens', help='Utiliser TruLens pour l\'observabilité'),
             deepeval: bool = typer.Option(False, '--deepeval', help='Utiliser DeepEval pour les tests unitaires'),
             optimize_vague_queries: bool = typer.Option(False, '--optimize-vague-queries', help='Optimiser les performances pour les requêtes vagues')):
    gts = load_ground_truth(ground_truth)
    questions = [it['question'] for it in gts]
    truths = [it['answer'] for it in gts]

    # OPTIMISATION: Parallélisation de l'évaluation des pipelines
    per_pipeline_answers = {}
    per_pipeline_contexts = {}

    print(f"\n🔄 Collecting answers from {len(['fixed', 'structure', 'sliding'])} pipelines for {len(questions)} questions...")

    def evaluate_pipeline(pipe):
        """Traite un pipeline en parallèle avec toutes ses questions"""
        answers = []
        contexts = []

        print(f"🚀 Démarrage pipeline {pipe} ({len(questions)} questions)")

        for i, q in enumerate(questions):
            if i % 20 == 0:  # Progress tous les 20 questions
                print(f"  {pipe}: {i+1}/{len(questions)} questions...")

            try:
                cands = get_candidates(doc_id, q, pipe, top_k, os.environ.get('RAG_LAB_DATA', 'data'))
                payload = build_answer_payload(pipe, q, cands, max_sentences=4, use_llm=use_llm)
                answers.append(payload['answer'])
                # Extract context from candidates
                context_list = [cand['text'] for cand in cands]
                contexts.append(context_list)
            except Exception as e:
                print(f"  ⚠️ Erreur {pipe} question {i}: {e}")
                answers.append("")
                contexts.append([])

        print(f"✅ Pipeline {pipe} terminé: {len(answers)} réponses")
        return pipe, answers, contexts

    # Traitement parallèle des pipelines
    pipelines = ['fixed', 'structure', 'sliding']
    with ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
        future_to_pipeline = {
            executor.submit(evaluate_pipeline, pipe): pipe
            for pipe in pipelines
        }

        for future in as_completed(future_to_pipeline):
            pipe, answers, contexts = future.result()
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

    # Embedding analysis
    if embedding_analysis:
        print(f"\n🔬 Starting advanced embedding analysis...")
        try:
            embedding_results = evaluate_embedding_quality(
                doc_id=doc_id,
                questions=questions,
                per_pipeline_answers=per_pipeline_answers,
                per_pipeline_contexts=per_pipeline_contexts,
                ground_truth=gts,
                include_retrieval_metrics=True,
                include_technical_analysis=True,
                k_values=[3, 5, 10, 15]
            )
            report['embedding_analysis'] = embedding_results

            # Export detailed embedding analysis
            from .embedding_metrics import export_embedding_analysis
            export_path = export_embedding_analysis(doc_id, embedding_results)
            print(f"💾 Embedding analysis exported to: {export_path}")

        except Exception as e:
            report['embedding_analysis_error'] = str(e)
            print(f"❌ Embedding analysis failed: {e}")

    # Legal evaluation
    if legal_evaluation:
        print(f"\n⚖️ Starting legal evaluation...")
        try:
            from .legal_evaluation import run_legal_evaluation_suite

            legal_results = run_legal_evaluation_suite(
                doc_id=doc_id,
                questions=questions,
                per_pipeline_answers=per_pipeline_answers,
                per_pipeline_contexts=per_pipeline_contexts,
                ground_truth=truths
            )
            report['legal_evaluation'] = legal_results
            print(f"📋 Legal evaluation completed for {len(legal_results)} pipelines")

        except Exception as e:
            report['legal_evaluation_error'] = str(e)
            print(f"❌ Legal evaluation failed: {e}")

    # Generic evaluation
    if generic_evaluation:
        print(f"\n🔍 Starting generic evaluation...")
        try:
            from .generic_evaluation import run_generic_evaluation_suite

            generic_results = run_generic_evaluation_suite(
                doc_id=doc_id,
                questions=questions,
                per_pipeline_answers=per_pipeline_answers,
                per_pipeline_contexts=per_pipeline_contexts,
                ground_truth=truths
            )
            report['generic_evaluation'] = generic_results
            print(f"📋 Generic evaluation completed for {len(generic_results)} pipelines")

        except Exception as e:
            report['generic_evaluation_error'] = str(e)
            print(f"❌ Generic evaluation failed: {e}")

    # TruLens evaluation
    if trulens:
        print(f"\n🔍 Starting TruLens evaluation...")
        try:
            # Import du tutoriel TruLens
            import sys
            sys.path.append('tutorials')
            from trulens_tutorial import run_trulens_evaluation

            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("⚠️ OPENAI_API_KEY required for TruLens")
                report['trulens_error'] = "Missing OPENAI_API_KEY"
            else:
                trulens_results = run_trulens_evaluation(
                    doc_id=doc_id,
                    per_pipeline_answers=per_pipeline_answers,
                    per_pipeline_contexts=per_pipeline_contexts,
                    questions=questions,
                    openai_api_key=openai_api_key
                )
                report['trulens'] = trulens_results
                print(f"🔍 TruLens evaluation completed")

        except Exception as e:
            report['trulens_error'] = str(e)
            print(f"❌ TruLens evaluation failed: {e}")
            print(f"💡 Solution: pip install trulens-eval")

    # DeepEval evaluation
    if deepeval:
        print(f"\n🚀 Starting DeepEval evaluation...")
        try:
            # Import du tutoriel DeepEval
            import sys
            sys.path.append('tutorials')
            from deepeval_tutorial import run_deepeval_evaluation

            deepeval_results = run_deepeval_evaluation(
                doc_id=doc_id,
                per_pipeline_answers=per_pipeline_answers,
                per_pipeline_contexts=per_pipeline_contexts,
                questions=questions,
                ground_truth=truths,
                include_safety_metrics=True
            )
            report['deepeval'] = deepeval_results
            print(f"🚀 DeepEval evaluation completed")

        except Exception as e:
            report['deepeval_error'] = str(e)
            print(f"❌ DeepEval evaluation failed: {e}")
            print(f"💡 Solution: pip install deepeval")

    # Azure AI Foundry evaluation
    if azure_foundry:
        print(f"\n🌟 Starting Azure AI Foundry evaluation...")
        try:
            from .azure_foundry_evaluation import integrate_with_azure_foundry

            foundry_results = integrate_with_azure_foundry(
                doc_id=doc_id,
                questions=questions,
                per_pipeline_answers=per_pipeline_answers,
                per_pipeline_contexts=per_pipeline_contexts,
                ground_truth=truths
            )
            report['azure_foundry'] = foundry_results

            if 'error' not in foundry_results:
                print(f"🌟 Azure Foundry evaluation completed")
            else:
                print(f"⚠️ Azure Foundry evaluation issue: {foundry_results.get('error')}")

        except Exception as e:
            report['azure_foundry_error'] = str(e)
            print(f"❌ Azure Foundry evaluation failed: {e}")
            print(f"💡 Solution: pip install azure-ai-ml azure-identity")

    # Optimisation pour requêtes vagues
    if optimize_vague_queries:
        print(f"\n🎯 Starting vague query optimization...")
        try:
            from .vague_query_optimizer import optimize_for_vague_queries

            openai_api_key = os.getenv('OPENAI_API_KEY')
            domain = "legal" if legal_evaluation else "general"

            # Charger les embeddings (simulé ici)
            import numpy as np
            embeddings = np.random.rand(1000, 384)  # Placeholder

            # Charger les chunks (simulé)
            chunks = [f"Chunk {i}" for i in range(1000)]  # Placeholder

            vague_optimization_results = optimize_for_vague_queries(
                doc_id=doc_id,
                questions=questions,
                chunks=chunks,
                embeddings=embeddings,
                openai_api_key=openai_api_key,
                domain=domain
            )

            report['vague_query_optimization'] = vague_optimization_results
            print(f"🎯 Vague query optimization completed")

            # Afficher les statistiques
            stats = vague_optimization_results['optimization_stats']
            print(f"  📊 Requêtes vagues détectées: {stats['vague_queries']}/{len(questions)} ({stats['vague_percentage']:.1f}%)")
            print(f"  🔄 Requêtes expandées: {stats['expanded_queries']}")
            print(f"  📝 Contextes enrichis: {stats['enhanced_contexts']}")

        except Exception as e:
            report['vague_optimization_error'] = str(e)
            print(f"❌ Vague query optimization failed: {e}")

    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

    # Affichage des métriques de performance pour l'évaluation
    print_performance_summary()

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

@app.command()
def benchmark_embeddings(
    doc_id: str = typer.Option(..., '--doc-id', help='Document ID to benchmark against'),
    ground_truth: str = typer.Option(..., '--ground-truth', help='Ground truth file path'),
    models: str = typer.Option(
        'dangvantuan/sentence-camembert-large,intfloat/multilingual-e5-large,BAAI/bge-m3',
        '--models',
        help='Comma-separated list of embedding models to test'
    ),
    top_k: int = typer.Option(DEFAULTS.top_k, '--top-k'),
    use_llm: bool = typer.Option(False, '--use-llm'),
    output_dir: str = typer.Option('benchmark_results', '--output-dir', help='Output directory for results')
):
    """
    Benchmark multiple embedding models for retrieval quality.

    Compare different embedding models on the same dataset to find the best
    performing model for your specific use case.

    Example:
    python -m rag_chunk_lab.cli benchmark-embeddings --doc-id legal_docs --ground-truth legal_ground_truth.jsonl
    """
    from pathlib import Path
    import time

    models_list = [model.strip() for model in models.split(',')]

    print(f"🏁 Starting embedding benchmark")
    print(f"📊 Models to test: {len(models_list)}")
    print(f"🎯 Document ID: {doc_id}")
    print(f"📋 Ground truth: {ground_truth}")

    # Load ground truth
    try:
        gts = load_ground_truth(ground_truth)
        questions = [it['question'] for it in gts]
        truths = [it['answer'] for it in gts]
        print(f"✅ Loaded {len(questions)} test questions")
    except Exception as e:
        typer.echo(f"❌ Failed to load ground truth: {e}")
        raise typer.Exit(1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmark_results = {}

    for i, model in enumerate(models_list, 1):
        print(f"\n🔍 Benchmark {i}/{len(models_list)}: {model}")

        model_start_time = time.time()

        try:
            # Temporarily modify the semantic chunking to use this model
            # This would require implementing model switching in the semantic chunkers
            # For now, we'll note this as a future enhancement

            # Test the current semantic pipeline (assuming it's configured for the model)
            model_results = {
                "model_name": model,
                "status": "planned_enhancement",
                "note": "Dynamic model switching requires chunker modification",
                "timestamp": time.time()
            }

            print(f"⚠️  Dynamic model switching not yet implemented")
            print(f"💡 Current implementation tests with pre-configured model")

        except Exception as e:
            model_results = {
                "model_name": model,
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
            print(f"❌ Error testing {model}: {e}")

        model_end_time = time.time()
        model_results["duration_seconds"] = model_end_time - model_start_time

        benchmark_results[model] = model_results

    # Export results
    benchmark_file = output_path / f"embedding_benchmark_{doc_id}.json"
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 Benchmark completed!")
    print(f"💾 Results saved to: {benchmark_file}")

    # Display summary
    print(f"\n📈 Summary:")
    for model, results in benchmark_results.items():
        status = results.get('status', 'unknown')
        duration = results.get('duration_seconds', 0)
        print(f"  {model}: {status} ({duration:.1f}s)")

    print(f"\n💡 Note: Pour un benchmark complet, implémentez le changement dynamique de modèles")
    print(f"   dans les chunkers sémantiques (semantic_chunks et azure_semantic_chunks)")

@app.command()
def analyze_embeddings(
    doc_id: str = typer.Option(..., '--doc-id', help='Document ID to analyze'),
    data_dir: str = typer.Option('data', '--data-dir', help='Data directory'),
    pipelines: str = typer.Option('semantic,azure_semantic', '--pipelines',
                                 help='Comma-separated list of pipelines to analyze'),
    export: bool = typer.Option(True, '--export/--no-export', help='Export embeddings for external analysis'),
    output_dir: str = typer.Option('embedding_analysis', '--output-dir', help='Output directory for analysis results')
):
    """
    Analyze embedding quality and technical metrics.

    Performs comprehensive analysis of embedding diversity, distribution,
    and semantic coherence for specified pipelines.

    Example:
    python -m rag_chunk_lab.cli analyze-embeddings --doc-id legal_docs --pipelines semantic,azure_semantic
    """
    from .embedding_analysis import run_comprehensive_embedding_analysis
    from pathlib import Path

    pipelines_list = [p.strip() for p in pipelines.split(',')]

    print(f"🔬 Starting embedding analysis")
    print(f"📊 Document ID: {doc_id}")
    print(f"🎯 Pipelines: {pipelines_list}")
    print(f"📁 Data directory: {data_dir}")

    try:
        # Run comprehensive analysis
        results = run_comprehensive_embedding_analysis(
            doc_id=doc_id,
            data_dir=data_dir,
            export=export
        )

        # Export detailed results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / f"embedding_analysis_{doc_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Analysis results saved to: {results_file}")

        # Display recommendations
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMANDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")

        # Quick summary in JSON for programmatic use
        summary_output = {
            "doc_id": doc_id,
            "pipelines_analyzed": results.get("analysis_summary", {}).get("pipelines_analyzed", []),
            "recommendations_count": len(recommendations),
            "analysis_file": str(results_file)
        }

        typer.echo(json.dumps(summary_output, ensure_ascii=False, indent=2))

    except Exception as e:
        typer.echo(f"❌ Embedding analysis failed: {e}")
        raise typer.Exit(1)

if __name__ == '__main__':
    app()
