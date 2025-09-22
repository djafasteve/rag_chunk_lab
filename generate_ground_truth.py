#!/usr/bin/env python3
"""
Script standalone pour générer un dataset de ground truth
Utilisation: python generate_ground_truth.py --folder documents/code_penal
"""

import argparse
import sys
from pathlib import Path

# Ajouter le module au chemin
sys.path.insert(0, str(Path(__file__).parent))

from rag_chunk_lab.ground_truth_generator import GroundTruthGenerator, create_llm_client


def main():
    parser = argparse.ArgumentParser(
        description='Génère automatiquement un dataset de ground truth à partir de documents'
    )

    parser.add_argument('--folder', required=True,
                       help='Dossier contenant les documents à traiter')
    parser.add_argument('--output',
                       help='Fichier JSONL de sortie (défaut: {nom_dossier}_ground_truth.jsonl)')
    parser.add_argument('--llm-provider', default='ollama', choices=['ollama', 'azure'],
                       help='Provider LLM: ollama ou azure (défaut: ollama)')
    parser.add_argument('--model',
                       help='Modèle à utiliser (défaut: mistral:7b pour Ollama)')
    parser.add_argument('--questions-per-doc', type=int, default=10,
                       help='Nombre de questions à générer pour CHAQUE document supporté (défaut: 10)')
    parser.add_argument('--min-length', type=int, default=200,
                       help='Longueur minimale du texte source (défaut: 200)')
    parser.add_argument('--max-length', type=int, default=3000,
                       help='Longueur maximale du texte source (défaut: 3000)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='URL du serveur Ollama (défaut: http://localhost:11434)')
    parser.add_argument('--language', default='fr', choices=['fr', 'en', 'es'],
                       help='Langue pour la génération des questions (défaut: fr)')
    parser.add_argument('--question-style', default='standard', choices=['standard', 'minimal-keywords'],
                       help='Style de questions: standard (avec mots-clés) ou minimal-keywords (sans indices) (défaut: standard)')
    parser.add_argument('--allow-reuse', action='store_true',
                       help='Permettre de réutiliser les chunks pour générer plus de questions que de chunks disponibles')

    args = parser.parse_args()

    folder_path = Path(args.folder)

    # Définir le fichier de sortie
    if not args.output:
        args.output = f"{folder_path.name}_ground_truth.jsonl"

    print(f"🚀 Démarrage de la génération de ground truth")
    print(f"📁 Dossier source: {args.folder}")
    print(f"🤖 LLM Provider: {args.llm_provider}")
    print(f"📊 Questions par document supporté: {args.questions_per_doc}")
    print(f"💾 Fichier de sortie: {args.output}")

    try:
        # Créer le client LLM
        print(f"\n🔌 Connexion au LLM...")

        if args.llm_provider == 'ollama':
            llm_client = create_llm_client('ollama', args.model, base_url=args.ollama_url)
            print(f"✅ Connecté à Ollama: {llm_client.model}")
        elif args.llm_provider == 'azure':
            import os
            # Vérifier les variables d'environnement
            required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                print(f"❌ Variables d'environnement manquantes: {', '.join(missing_vars)}")
                return 1

            llm_client = create_llm_client('azure', args.model)
            print(f"✅ Connecté à Azure OpenAI: {llm_client.model}")

        # Créer le générateur
        generator = GroundTruthGenerator(llm_client, language=args.language, question_style=args.question_style)

        # Générer le dataset
        output_path = generator.generate_from_folder(
            folder_path=args.folder,
            output_path=args.output,
            questions_per_doc=args.questions_per_doc,
            min_text_length=args.min_length,
            max_text_length=args.max_length,
            allow_reuse=args.allow_reuse
        )

        print(f"\n🎉 Génération terminée avec succès!")
        print(f"📄 Dataset sauvegardé: {output_path}")
        print(f"\n💡 Utilisez maintenant:")
        print(f"   python -m rag_chunk_lab.cli evaluate --doc-id <votre-doc> --ground-truth {output_path} --ragas --use-llm")

        return 0

    except ConnectionError as e:
        print(f"❌ Erreur de connexion LLM: {e}")
        if args.llm_provider == 'ollama':
            print("💡 Assurez-vous qu'Ollama est démarré: ollama serve")
            print(f"💡 Et que le modèle est installé: ollama pull {args.model or 'mistral:7b'}")
        return 1
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())