#!/usr/bin/env python3
# vague_query_optimization_demo.py
"""
Démonstration complète du système d'optimisation pour requêtes vagues
"""

import os
import sys
from pathlib import Path

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag_chunk_lab.vague_query.vague_query_optimization_system import (
    create_vague_optimization_system,
    quick_vague_query_optimization
)
import logging

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_complete_system():
    """Démonstration du système complet"""

    print("🚀 Démonstration du Système d'Optimisation pour Requêtes Vagues")
    print("=" * 70)

    # Documents d'exemple (domaine juridique)
    sample_documents = [
        {
            "doc_id": "code_civil_art1",
            "text": """
            Article 1er du Code Civil

            Les lois ne sont obligatoires qu'à partir de leur publication et dans les conditions prévues par la Constitution.

            La loi ne dispose que pour l'avenir ; elle n'a point d'effet rétroactif, sauf disposition contraire expressément prévue.

            Chacun est tenu de respecter les règles de droit qui lui sont applicables. Nul n'est censé ignorer la loi.

            Le principe de non-rétroactivité des lois est un principe fondamental du droit français. Il garantit la sécurité juridique en empêchant qu'une loi nouvelle puisse remettre en cause des situations déjà constituées.

            Exemples d'application :
            - Une loi pénale nouvelle ne peut pas sanctionner des faits commis avant sa publication
            - Une modification du code du travail ne peut pas affecter des contrats déjà signés
            - Les nouvelles règles fiscales s'appliquent généralement à compter de l'année suivante
            """
        },
        {
            "doc_id": "procedure_civile",
            "text": """
            Procédure Civile - Principes Généraux

            La procédure civile est l'ensemble des règles qui organisent le déroulement d'un procès devant les juridictions civiles.

            Principes directeurs :
            1. Le principe du contradictoire : chaque partie doit pouvoir présenter ses arguments
            2. Le principe de l'égalité des armes : équilibre entre les parties
            3. Le principe de la publicité des débats : transparence de la justice
            4. Le principe de l'impartialité du juge : neutralité et objectivité

            Étapes d'une procédure civile :
            - Assignation en justice par voie d'huissier
            - Constitution d'avocat (obligatoire devant certaines juridictions)
            - Échange de conclusions entre les parties
            - Mise en état du dossier par le juge
            - Audience de plaidoirie
            - Délibéré et prononcé du jugement
            - Voies de recours possibles (appel, cassation)

            Le délai moyen d'une procédure civile varie de 6 mois à 2 ans selon la complexité et la juridiction.
            """
        },
        {
            "doc_id": "contrat_obligations",
            "text": """
            Droit des Contrats et des Obligations

            Un contrat est une convention par laquelle une ou plusieurs personnes s'obligent envers une autre ou plusieurs autres, à donner, à faire ou à ne pas faire quelque chose.

            Conditions de validité d'un contrat :
            1. Consentement libre et éclairé des parties
            2. Capacité juridique des contractants
            3. Objet licite et déterminé
            4. Cause licite

            Types de contrats :
            - Contrats synallagmatiques : obligations réciproques
            - Contrats unilatéraux : obligations d'une seule partie
            - Contrats à titre gratuit : libéralités
            - Contrats à titre onéreux : échange d'avantages

            Exécution des obligations :
            - Exécution volontaire : respect spontané du contrat
            - Exécution forcée : mise en œuvre par voie judiciaire
            - Dommages-intérêts en cas d'inexécution

            La force obligatoire du contrat est exprimée par l'adage "pacta sunt servanda" : les conventions doivent être respectées.
            """
        }
    ]

    # Configuration du système
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY non configurée - certaines fonctionnalités seront limitées")

    print(f"\n📋 Configuration:")
    print(f"   - Domaine: juridique")
    print(f"   - Documents: {len(sample_documents)}")
    print(f"   - OpenAI disponible: {'✅' if openai_api_key else '❌'}")

    # Créer le système
    print(f"\n🔧 Initialisation du système...")

    system = create_vague_optimization_system(
        domain="legal",
        openai_api_key=openai_api_key,
        enable_monitoring=True,
        enable_fine_tuning=False  # Désactivé pour la démo (trop long)
    )

    # Vérifier l'état d'initialisation
    status = system.get_system_status()
    print(f"\n📊 État du système:")
    for component, status_ok in status["components_status"].items():
        icon = "✅" if status_ok else "❌"
        print(f"   {icon} {component}")

    if not system.is_initialized:
        print("❌ Système non initialisé complètement - arrêt de la démo")
        return

    # Indexer les documents
    print(f"\n📚 Indexation des documents...")

    indexing_stats = system.index_documents(sample_documents)

    print(f"   📄 Documents traités: {indexing_stats['processed_documents']}")
    print(f"   🧩 Chunks créés: {indexing_stats['total_chunks']}")
    print(f"   📊 Niveaux hiérarchiques: {indexing_stats['hierarchical_levels']}")
    print(f"   🔍 Index embeddings: {indexing_stats['embedding_index_size']}")

    if indexing_stats["errors"]:
        print(f"   ⚠️ Erreurs: {len(indexing_stats['errors'])}")

    # Test avec différents types de requêtes
    test_queries = [
        {
            "query": "Droit ?",
            "type": "Très vague",
            "user_level": "beginner"
        },
        {
            "query": "Comment ça marche la justice ?",
            "type": "Vague générale",
            "user_level": "beginner"
        },
        {
            "query": "Procédure",
            "type": "Mot-clé simple",
            "user_level": "intermediate"
        },
        {
            "query": "Quelles sont les étapes d'une procédure civile et combien de temps ça prend ?",
            "type": "Précise et structurée",
            "user_level": "intermediate"
        },
        {
            "query": "Analyse comparative des principes directeurs de la procédure civile et leur application en droit européen",
            "type": "Complexe et experte",
            "user_level": "advanced"
        }
    ]

    print(f"\n🎯 Test d'optimisation de requêtes:")
    print("=" * 50)

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        query_type = test_case["type"]
        user_level = test_case["user_level"]

        print(f"\n{i}. Requête {query_type} (niveau {user_level}):")
        print(f"   💬 \"{query}\"")

        try:
            # Optimiser la requête
            result = system.optimize_vague_query(
                query=query,
                user_level=user_level,
                max_results=3
            )

            # Afficher les résultats d'analyse
            print(f"   📊 Analyse:")
            print(f"      - Vague: {'✅' if result['is_vague'] else '❌'} (score: {result['vagueness_score']:.2f})")
            print(f"      - Expansions: {len(result['expanded_queries'])}")
            print(f"      - Chunks récupérés: {len(result['retrieved_chunks'])}")
            print(f"      - Qualité contexte: {result['context_quality']:.2f}")
            print(f"      - Temps de réponse: {result['performance']['response_time']:.2f}s")

            # Montrer les expansions si disponibles
            if len(result['expanded_queries']) > 1:
                print(f"   🔄 Expansions générées:")
                for expansion in result['expanded_queries'][1:3]:  # Montrer 2 expansions
                    print(f"      - \"{expansion}\"")

            # Montrer les recommandations
            if result['recommendations']:
                print(f"   💡 Recommandations:")
                for rec in result['recommendations']:
                    print(f"      - {rec}")

            # Simuler du feedback pour les premières requêtes
            if i <= 2:
                relevance_score = 4 if result['is_vague'] else 5
                helpfulness_score = 3 if result['is_vague'] else 4
                clarity_score = 4

                system.collect_feedback(
                    query=query,
                    response="[Réponse simulée optimisée]",
                    relevance_score=relevance_score,
                    helpfulness_score=helpfulness_score,
                    clarity_score=clarity_score,
                    user_id=f"demo_user_{i}",
                    improvements_suggested=["Ajouter plus d'exemples"] if result['is_vague'] else []
                )

                print(f"   ✅ Feedback simulé collecté (pertinence: {relevance_score}/5)")

        except Exception as e:
            print(f"   ❌ Erreur: {e}")

    # Afficher les métriques de monitoring
    if system.production_monitor:
        print(f"\n📈 Métriques de Performance:")
        print("=" * 30)

        try:
            health_status = system.production_monitor.get_system_health()

            print(f"   🏥 Santé globale: {health_status['status']} ({health_status['overall_health_score']:.0f}/100)")

            metrics = health_status.get('metrics_summary', {}).get('metrics', {})

            if 'response_time' in metrics:
                rt = metrics['response_time']
                print(f"   ⏱️  Temps de réponse moyen: {rt.get('mean', 0):.2f}s")

            if 'vague_query_rate' in metrics:
                vqr = metrics['vague_query_rate']
                print(f"   🎯 Taux de requêtes vagues: {vqr.get('mean', 0)*100:.1f}%")

            feedback_analytics = health_status.get('feedback_analytics', {})
            if 'total_feedback' in feedback_analytics:
                print(f"   💬 Feedback collecté: {feedback_analytics['total_feedback']}")

                satisfaction = feedback_analytics.get('overall_satisfaction', {})
                if 'mean' in satisfaction:
                    print(f"   😊 Satisfaction moyenne: {satisfaction['mean']:.1f}/5")

        except Exception as e:
            print(f"   ⚠️ Erreur récupération métriques: {e}")

    # Afficher le statut final
    print(f"\n🏁 Statut Final du Système:")
    print("=" * 30)

    final_status = system.get_system_status()

    print(f"   💾 Documents indexés: {final_status.get('indexing_stats', {}).get('total_documents', 0)}")
    print(f"   🧩 Chunks totaux: {final_status.get('indexing_stats', {}).get('total_chunks', 0)}")
    print(f"   🔍 Index embeddings: {final_status.get('indexing_stats', {}).get('embedding_index_size', 0)}")
    print(f"   📊 Monitoring actif: {'✅' if final_status['monitoring_enabled'] else '❌'}")

    # Nettoyer
    print(f"\n🧹 Nettoyage...")
    system.shutdown()

    print(f"\n🎉 Démonstration terminée avec succès !")
    print(f"🔗 Le système est maintenant prêt pour l'intégration en production.")


def demo_quick_optimization():
    """Démonstration rapide avec l'API simplifiée"""

    print("\n" + "="*70)
    print("🚀 DÉMONSTRATION RAPIDE - API Simplifiée")
    print("="*70)

    # Documents simplifiés
    documents = [
        {
            "doc_id": "legal_basics",
            "text": "Le droit civil français régit les relations entre particuliers. Les contrats sont régis par le principe de liberté contractuelle et la force obligatoire des conventions."
        },
        {
            "doc_id": "procedure_basics",
            "text": "La procédure civile organise le déroulement des procès. Elle respecte le principe du contradictoire et garantit l'égalité des parties."
        }
    ]

    # Test avec requête vague
    vague_query = "Comment ça marche ?"

    print(f"📋 Test d'optimisation rapide:")
    print(f"   💬 Requête: \"{vague_query}\"")
    print(f"   📚 Documents: {len(documents)}")

    try:
        result = quick_vague_query_optimization(
            query=vague_query,
            documents=documents,
            domain="legal",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        print(f"\n✅ Optimisation réussie:")
        print(f"   🎯 Vague détectée: {'✅' if result['is_vague'] else '❌'} (score: {result['vagueness_score']:.2f})")
        print(f"   🔄 Expansions: {len(result['expanded_queries'])}")
        print(f"   📊 Qualité contexte: {result['context_quality']:.2f}")
        print(f"   ⏱️  Performance: {result['performance']['response_time']:.2f}s")

        # Afficher un extrait du prompt optimisé
        prompt_preview = result['optimized_prompt'][:200] + "..." if len(result['optimized_prompt']) > 200 else result['optimized_prompt']
        print(f"\n💬 Aperçu du prompt optimisé:")
        print(f"   \"{prompt_preview}\"")

    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    print("🎯 Système d'Optimisation RAG pour Requêtes Vagues")
    print("   Développé pour RAG Chunk Lab")
    print()

    # Vérifier les dépendances
    try:
        import openai
        print("✅ OpenAI disponible")
    except ImportError:
        print("⚠️ OpenAI non disponible - fonctionnalités LLM limitées")

    try:
        import spacy
        print("✅ SpaCy disponible")
    except ImportError:
        print("⚠️ SpaCy non disponible - analyse linguistique limitée")

    try:
        import sentence_transformers
        print("✅ Sentence Transformers disponible")
    except ImportError:
        print("⚠️ Sentence Transformers non disponible - embeddings denses limités")

    print()

    # Lancer la démonstration rapide d'abord
    demo_quick_optimization()

    # Puis la démonstration complète
    response = input("\n🤔 Lancer la démonstration complète ? (y/N): ").strip().lower()

    if response in ['y', 'yes', 'oui']:
        demo_complete_system()
    else:
        print("✨ Démonstration rapide terminée. Merci !")