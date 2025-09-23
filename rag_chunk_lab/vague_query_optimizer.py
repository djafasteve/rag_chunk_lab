# vague_query_optimizer.py
"""
Optimiseur pour améliorer les performances RAG sur les requêtes vagues
"""

from typing import List, Dict, Tuple, Any
import re
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class VagueQueryOptimizer:
    """Optimise les performances RAG pour les requêtes vagues"""

    def __init__(self, openai_api_key: str = None, domain: str = "general"):
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.domain = domain

        # Modèles d'embedding
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Patterns de requêtes vagues
        self.vague_patterns = [
            r'\b(qu\'est-ce que|que|quoi|comment|pourquoi)\b.*\?',
            r'\b(expliquer|expliquez|dire|parler de)\b',
            r'\b(aide|aidez|information|info)\b',
            r'^\w{1,3}\s*\?*$',  # Très court
            r'\b(tout|tous|toute|général|générale)\b'
        ]

        # Mots-clés de domaine pour enrichissement
        self.domain_keywords = self._load_domain_keywords(domain)

        # Configuration NLP
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("SpaCy French model not found. Install with: python -m spacy download fr_core_news_sm")
            self.nlp = None

    def _load_domain_keywords(self, domain: str) -> Dict[str, List[str]]:
        """Charge les mots-clés spécifiques au domaine"""

        domain_keywords = {
            "legal": {
                "concepts": ["droit", "loi", "article", "jurisprudence", "tribunal", "procédure", "TVA", "facture", "contrat", "obligation", "prescription", "délai"],
                "actions": ["porter plainte", "intenter", "assigner", "comparaître", "calculer", "facturer", "contracter"],
                "entities": ["avocat", "juge", "procureur", "défendeur", "demandeur", "entreprise", "client"]
            },
            "technical": {
                "concepts": ["algorithme", "architecture", "framework", "API", "base de données"],
                "actions": ["configurer", "implémenter", "optimiser", "déboguer"],
                "entities": ["serveur", "client", "utilisateur", "système"]
            },
            "medical": {
                "concepts": ["symptôme", "traitement", "diagnostic", "pathologie", "thérapie"],
                "actions": ["traiter", "diagnostiquer", "prévenir", "soigner"],
                "entities": ["patient", "médecin", "spécialiste", "hôpital"]
            },
            "general": {
                "concepts": ["principe", "méthode", "processus", "système", "concept"],
                "actions": ["faire", "utiliser", "appliquer", "comprendre"],
                "entities": ["personne", "organisation", "service", "produit"]
            }
        }

        return domain_keywords.get(domain, domain_keywords["general"])

    def is_vague_query(self, query: str) -> Tuple[bool, float]:
        """
        Détermine si une requête est vague

        Returns:
            Tuple[bool, float]: (is_vague, vagueness_score)
        """

        query_lower = query.lower().strip()
        vagueness_score = 0.0

        # 1. Longueur de la requête
        if len(query_lower.split()) <= 3:
            vagueness_score += 0.3

        # 2. Patterns de vague
        for pattern in self.vague_patterns:
            if re.search(pattern, query_lower):
                vagueness_score += 0.2

        # 3. Absence de mots-clés spécifiques au domaine
        domain_matches = 0
        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    domain_matches += 1

        if domain_matches == 0:
            vagueness_score += 0.3

        # 4. Mots interrogatifs génériques (mais pas s'il y a des termes spécifiques)
        generic_words = ["quoi", "comment", "pourquoi", "ça", "truc", "chose"]
        generic_count = sum(1 for word in generic_words if word in query_lower)

        # Vérifier s'il y a des termes spécifiques qui compensent les mots génériques
        specific_terms = ["calculer", "TVA", "facture", "article", "procédure", "délai", "prescription",
                         "système", "judiciaire", "français", "divorce", "contrat", "tribunal", "droit"]
        specific_count = sum(1 for term in specific_terms if term.lower() in query_lower)

        # Bonus pour les combinaisons de termes spécifiques
        if len(query_lower.split()) >= 4 and specific_count >= 2:
            specific_count += 1  # Bonus pour question détaillée

        # Réduire l'impact des mots génériques s'il y a des termes spécifiques
        if specific_count > 0:
            vagueness_score += max(0, min(generic_count * 0.05, 0.1))  # Impact réduit
        else:
            vagueness_score += min(generic_count * 0.1, 0.2)

        # 5. Ajustement final basé sur la richesse en termes spécifiques
        if specific_count >= 3:  # Question très spécifique
            vagueness_score *= 0.5  # Réduction de 50%
        elif specific_count >= 2:  # Question assez spécifique
            vagueness_score *= 0.7  # Réduction de 30%

        is_vague = vagueness_score >= 0.4

        return is_vague, min(vagueness_score, 1.0)

    def expand_vague_query(self, query: str) -> List[str]:
        """Expand une requête vague en plusieurs reformulations"""

        expanded_queries = [query]  # Garder l'originale

        # Expansion basique par templates
        base_expansions = [
            f"Qu'est-ce que {query}",
            f"Comment fonctionne {query}",
            f"Définition de {query}",
            f"Exemples de {query}",
            f"Applications de {query}",
            f"Principe de {query}"
        ]

        # Ajouter les expansions qui ne sont pas redondantes
        for expansion in base_expansions:
            if expansion.lower() != query.lower():
                expanded_queries.append(expansion)

        # Expansion avec LLM si disponible
        if self.client:
            try:
                llm_expansions = self._llm_query_expansion(query)
                expanded_queries.extend(llm_expansions)
            except Exception as e:
                logger.warning(f"LLM expansion failed: {e}")

        # Expansion avec analyse NLP
        if self.nlp:
            nlp_expansions = self._nlp_query_expansion(query)
            expanded_queries.extend(nlp_expansions)

        # Supprimer les doublons et limiter
        unique_queries = list(dict.fromkeys(expanded_queries))
        return unique_queries[:8]  # Limiter à 8 variations

    def _llm_query_expansion(self, query: str) -> List[str]:
        """Utilise un LLM pour expand la requête"""

        prompt = f"""
        La requête suivante semble vague: "{query}"

        Génère 3 reformulations plus précises de cette question dans le domaine {self.domain}.
        Chaque reformulation doit être:
        1. Plus spécifique que l'originale
        2. Orientée vers des aspects pratiques
        3. Adaptée au domaine {self.domain}

        Format: une reformulation par ligne, sans numérotation.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )

            expansions = response.choices[0].message.content.strip().split('\n')
            return [exp.strip() for exp in expansions if exp.strip()]

        except Exception as e:
            logger.error(f"LLM expansion error: {e}")
            return []

    def _nlp_query_expansion(self, query: str) -> List[str]:
        """Utilise NLP pour expand la requête"""

        if not self.nlp:
            return []

        doc = self.nlp(query)
        expansions = []

        # Extraire les entités et concepts
        entities = [ent.text for ent in doc.ents]
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]

        # Créer des expansions basées sur les entités/concepts
        for entity in entities:
            if entity.lower() != query.lower():
                expansions.extend([
                    f"Comment utiliser {entity}",
                    f"Avantages de {entity}",
                    f"Processus de {entity}"
                ])

        for noun in nouns:
            if noun.lower() not in query.lower():
                expansions.append(f"{query} et {noun}")

        return expansions[:5]

    def enhance_context_for_vague_query(self,
                                       query: str,
                                       retrieved_chunks: List[str],
                                       chunk_scores: List[float] = None) -> str:
        """Enrichit le contexte pour une requête vague"""

        if not retrieved_chunks:
            return ""

        # Analyser le contenu récupéré
        context_analysis = self._analyze_retrieved_content(retrieved_chunks)

        # Créer un contexte structuré
        enhanced_context = self._build_structured_context(
            query, retrieved_chunks, context_analysis, chunk_scores
        )

        return enhanced_context

    def _analyze_retrieved_content(self, chunks: List[str]) -> Dict[str, Any]:
        """Analyse le contenu récupéré pour identifier les concepts clés"""

        all_text = " ".join(chunks)

        analysis = {
            "key_concepts": [],
            "definitions": [],
            "examples": [],
            "procedures": []
        }

        if self.nlp:
            doc = self.nlp(all_text)

            # Extraire les concepts clés (entités nommées)
            entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
            analysis["key_concepts"] = list(set(entities))[:10]

            # Identifier les définitions (patterns simples)
            definition_patterns = [
                r'([A-Z][a-z]+)\s+est\s+([^.]+\.)',
                r'([A-Z][a-z]+)\s+désigne\s+([^.]+\.)',
                r'On\s+appelle\s+([A-Z][a-z]+)\s+([^.]+\.)'
            ]

            for pattern in definition_patterns:
                matches = re.findall(pattern, all_text)
                for match in matches:
                    analysis["definitions"].append(f"{match[0]}: {match[1]}")

            # Identifier les exemples
            example_indicators = ["par exemple", "notamment", "comme", "tel que"]
            sentences = [sent.text for sent in doc.sents]

            for sentence in sentences:
                for indicator in example_indicators:
                    if indicator in sentence.lower():
                        analysis["examples"].append(sentence.strip())
                        break

        return analysis

    def _build_structured_context(self,
                                 query: str,
                                 chunks: List[str],
                                 analysis: Dict[str, Any],
                                 scores: List[float] = None) -> str:
        """Construit un contexte structuré pour requête vague"""

        # Trier les chunks par pertinence si scores disponibles
        if scores:
            chunk_score_pairs = list(zip(chunks, scores))
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_chunks = [chunk for chunk, _ in chunk_score_pairs]
        else:
            sorted_chunks = chunks

        # Construire le contexte enrichi
        context_parts = []

        # Section principale avec les meilleurs chunks
        context_parts.append("=== CONTEXTE PRINCIPAL ===")
        context_parts.extend(sorted_chunks[:3])  # Top 3 chunks

        # Section concepts clés si disponibles
        if analysis["key_concepts"]:
            context_parts.append("\n=== CONCEPTS CLÉS ===")
            context_parts.append(", ".join(analysis["key_concepts"][:5]))

        # Section définitions si disponibles
        if analysis["definitions"]:
            context_parts.append("\n=== DÉFINITIONS ===")
            context_parts.extend(analysis["definitions"][:3])

        # Section exemples si disponibles
        if analysis["examples"]:
            context_parts.append("\n=== EXEMPLES ===")
            context_parts.extend(analysis["examples"][:2])

        return "\n".join(context_parts)

    def create_adaptive_prompt(self, query: str, enhanced_context: str) -> str:
        """Crée un prompt adaptatif pour requête vague"""

        is_vague, vagueness_score = self.is_vague_query(query)

        if is_vague:
            prompt = f"""
Tu es un assistant expert dans le domaine {self.domain}. L'utilisateur a posé une question générale qui nécessite une réponse pédagogique et structurée.

MISSION:
1. Identifier les concepts clés dans la question
2. Fournir une explication progressive (du simple au détaillé)
3. Donner des exemples concrets et pratiques
4. Proposer des questions de clarification si pertinent

QUESTION: {query}

CONTEXTE DISPONIBLE:
{enhanced_context}

INSTRUCTIONS DE RÉPONSE:
- Commencer par une définition simple et claire
- Structurer la réponse avec des sections (Définition, Fonctionnement, Exemples, Applications)
- Utiliser un langage accessible mais précis
- Conclure par des questions pour approfondir

RÉPONSE STRUCTURÉE:
"""
        else:
            prompt = f"""
Tu es un assistant expert dans le domaine {self.domain}. L'utilisateur a posé une question précise.

QUESTION: {query}

CONTEXTE:
{enhanced_context}

INSTRUCTIONS:
- Fournir une réponse directe et factuelle
- Utiliser le contexte fourni comme source principale
- Être précis et concis

RÉPONSE:
"""

        return prompt

    def optimize_retrieval_for_vague_query(self,
                                          query: str,
                                          chunks: List[str],
                                          embeddings: np.ndarray,
                                          top_k: int = 5) -> Tuple[List[str], List[float]]:
        """Optimise la récupération pour les requêtes vagues"""

        is_vague, vagueness_score = self.is_vague_query(query)

        if not is_vague:
            # Récupération standard pour requêtes précises
            return self._standard_retrieval(query, chunks, embeddings, top_k)

        # Stratégie optimisée pour requêtes vagues
        expanded_queries = self.expand_vague_query(query)

        # Récupération multi-requêtes avec fusion
        all_results = []

        for expanded_query in expanded_queries:
            query_embedding = self.sentence_model.encode([expanded_query])
            similarities = np.dot(embeddings, query_embedding.T).flatten()

            # Ajouter les résultats avec score pondéré
            for i, score in enumerate(similarities):
                all_results.append((i, score, expanded_query))

        # Fusionner et re-ranker les résultats
        chunk_scores = {}
        for chunk_idx, score, source_query in all_results:
            if chunk_idx not in chunk_scores:
                chunk_scores[chunk_idx] = []
            chunk_scores[chunk_idx].append(score)

        # Calculer le score final (moyenne pondérée)
        final_scores = []
        for chunk_idx in range(len(chunks)):
            if chunk_idx in chunk_scores:
                scores = chunk_scores[chunk_idx]
                final_score = max(scores)  # Prendre le meilleur score
            else:
                final_score = 0.0
            final_scores.append((chunk_idx, final_score))

        # Trier et retourner top_k
        final_scores.sort(key=lambda x: x[1], reverse=True)

        selected_indices = [idx for idx, _ in final_scores[:top_k]]
        selected_chunks = [chunks[idx] for idx in selected_indices]
        selected_scores = [score for _, score in final_scores[:top_k]]

        return selected_chunks, selected_scores

    def _standard_retrieval(self,
                           query: str,
                           chunks: List[str],
                           embeddings: np.ndarray,
                           top_k: int) -> Tuple[List[str], List[float]]:
        """Récupération standard pour requêtes précises"""

        query_embedding = self.sentence_model.encode([query])
        similarities = np.dot(embeddings, query_embedding.T).flatten()

        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        return top_chunks, top_scores

# Fonction d'intégration pour RAG Chunk Lab
def optimize_for_vague_queries(doc_id: str,
                              questions: List[str],
                              chunks: List[str],
                              embeddings: np.ndarray,
                              openai_api_key: str = None,
                              domain: str = "general") -> Dict[str, Any]:
    """
    Optimise les réponses pour les requêtes vagues

    Returns:
        Dict contenant les résultats optimisés et les métriques
    """

    optimizer = VagueQueryOptimizer(openai_api_key, domain)

    results = {
        "optimized_answers": [],
        "vagueness_analysis": [],
        "optimization_stats": {
            "vague_queries": 0,
            "expanded_queries": 0,
            "enhanced_contexts": 0
        }
    }

    for question in questions:
        # Analyser la vague
        is_vague, vagueness_score = optimizer.is_vague_query(question)

        vagueness_info = {
            "question": question,
            "is_vague": is_vague,
            "vagueness_score": vagueness_score
        }

        if is_vague:
            results["optimization_stats"]["vague_queries"] += 1

            # Optimiser la récupération
            retrieved_chunks, scores = optimizer.optimize_retrieval_for_vague_query(
                question, chunks, embeddings, top_k=5
            )

            # Enrichir le contexte
            enhanced_context = optimizer.enhance_context_for_vague_query(
                question, retrieved_chunks, scores
            )

            # Créer un prompt adaptatif
            adaptive_prompt = optimizer.create_adaptive_prompt(question, enhanced_context)

            vagueness_info.update({
                "expanded_queries": optimizer.expand_vague_query(question),
                "enhanced_context_length": len(enhanced_context),
                "adaptive_prompt_used": True
            })

            results["optimization_stats"]["expanded_queries"] += 1
            results["optimization_stats"]["enhanced_contexts"] += 1

        else:
            # Traitement standard pour requêtes précises
            retrieved_chunks, scores = optimizer._standard_retrieval(
                question, chunks, embeddings, top_k=5
            )
            enhanced_context = "\n".join(retrieved_chunks)
            adaptive_prompt = optimizer.create_adaptive_prompt(question, enhanced_context)

            vagueness_info.update({
                "adaptive_prompt_used": False
            })

        results["vagueness_analysis"].append(vagueness_info)

        # Simuler une réponse optimisée (en pratique, vous utiliseriez votre LLM)
        optimized_answer = f"[Réponse optimisée pour: {question}]"
        results["optimized_answers"].append(optimized_answer)

    # Calculer les statistiques finales
    total_questions = len(questions)
    results["optimization_stats"]["vague_percentage"] = (
        results["optimization_stats"]["vague_queries"] / total_questions * 100
        if total_questions > 0 else 0
    )

    return results