# adaptive_prompt_engineering.py
"""
Système de prompt engineering adaptatif pour optimiser les réponses RAG selon le contexte
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types de requêtes identifiées"""
    DEFINITION = "definition"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"
    EXAMPLE = "example"
    EXPLANATION = "explanation"
    FACTUAL = "factual"
    OPINION = "opinion"
    VAGUE = "vague"


class ResponseStyle(Enum):
    """Styles de réponse"""
    EDUCATIONAL = "educational"
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    STEP_BY_STEP = "step_by_step"
    STRUCTURED = "structured"


@dataclass
class PromptContext:
    """Contexte pour la génération de prompt"""
    query: str
    query_type: QueryType
    vagueness_score: float
    user_level: str  # "beginner", "intermediate", "advanced"
    domain: str
    context_length: int
    concepts_count: int
    has_examples: bool = False
    has_definitions: bool = False
    response_style: ResponseStyle = ResponseStyle.EDUCATIONAL
    custom_instructions: List[str] = field(default_factory=list)


@dataclass
class AdaptivePrompt:
    """Prompt adaptatif généré"""
    system_prompt: str
    user_prompt: str
    context_section: str
    instructions_section: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_full_prompt(self) -> str:
        """Retourne le prompt complet formaté"""
        sections = [
            self.system_prompt,
            self.context_section,
            self.user_prompt,
            self.instructions_section
        ]

        return "\n\n".join(section for section in sections if section.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context_section": self.context_section,
            "instructions_section": self.instructions_section,
            "metadata": self.metadata
        }


class QueryTypeClassifier:
    """Classificateur de types de requêtes"""

    def __init__(self, domain: str = "general", language: str = "fr"):
        self.domain = domain
        self.language = language

        # Patterns par type de requête
        self.query_patterns = self._load_query_patterns(language)

        # Mots-clés par domaine
        self.domain_keywords = self._load_domain_keywords(domain)

    def _load_query_patterns(self, language: str) -> Dict[QueryType, List[str]]:
        """Charge les patterns de classification par langue"""

        if language == "fr":
            patterns = {
                QueryType.DEFINITION: [
                    r"\b(?:qu'est-ce que|que signifie|définition|définir|c'est quoi)\b",
                    r"\b(?:que veut dire|ça veut dire quoi|signifie quoi)\b",
                    r"\b(?:expliquez-moi|explique|explication de)\b.*\b(?:qu'est|ce que)\b"
                ],

                QueryType.HOW_TO: [
                    r"\b(?:comment|comment faire|comment puis-je)\b",
                    r"\b(?:étapes pour|procédure pour|méthode pour)\b",
                    r"\b(?:guide pour|tutoriel|instructions)\b",
                    r"\b(?:comment réaliser|comment effectuer)\b"
                ],

                QueryType.COMPARISON: [
                    r"\b(?:différence entre|différences entre|comparer)\b",
                    r"\b(?:versus|vs|par rapport à|comparaison)\b",
                    r"\b(?:quel est mieux|lequel choisir|avantages)\b",
                    r"\b(?:plutôt que|au lieu de|contrairement à)\b"
                ],

                QueryType.TROUBLESHOOTING: [
                    r"\b(?:problème|erreur|bug|dysfonctionnement)\b",
                    r"\b(?:ne fonctionne pas|ne marche pas|panne)\b",
                    r"\b(?:résoudre|réparer|corriger|dépanner)\b",
                    r"\b(?:pourquoi.*ne.*pas|pourquoi ça ne)\b"
                ],

                QueryType.EXAMPLE: [
                    r"\b(?:exemple|exemples|par exemple|illustration)\b",
                    r"\b(?:montrez-moi|donnez-moi un exemple)\b",
                    r"\b(?:cas concret|cas pratique|application)\b",
                    r"\b(?:pour illustrer|comme exemple)\b"
                ],

                QueryType.EXPLANATION: [
                    r"\b(?:pourquoi|comment ça marche|fonctionnement)\b",
                    r"\b(?:principe|mécanisme|logique)\b",
                    r"\b(?:expliquer|clarifier|détailler)\b",
                    r"\b(?:raison pour|motif|cause)\b"
                ],

                QueryType.FACTUAL: [
                    r"\b(?:quand|où|qui|combien|quelle est)\b",
                    r"\b(?:date|lieu|nombre|quantité|statistique)\b",
                    r"\b(?:liste des|énumérer|répertorier)\b",
                    r"^\s*(?:quel|quelle|quels|quelles)\b"
                ],

                QueryType.OPINION: [
                    r"\b(?:pensez-vous|opinion|avis|recommandation)\b",
                    r"\b(?:suggérez|conseillez|recommandez)\b",
                    r"\b(?:que choisir|quel est le mieux)\b",
                    r"\b(?:votre avis|point de vue)\b"
                ],

                QueryType.VAGUE: [
                    r"^.{1,10}[?]*$",  # Très court
                    r"\b(?:aide|info|information|renseignement)\b",
                    r"\b(?:tout|tous|toute|général|générale)\b",
                    r"\b(?:quelque chose|n'importe quoi|truc)\b"
                ]
            }
        else:
            # Patterns anglais
            patterns = {
                QueryType.DEFINITION: [
                    r"\b(?:what is|what are|define|definition|meaning)\b",
                    r"\b(?:what does.*mean|explain what)\b"
                ],
                QueryType.HOW_TO: [
                    r"\b(?:how to|how do|how can|steps to)\b",
                    r"\b(?:guide|tutorial|instructions|procedure)\b"
                ],
                # ... autres patterns anglais
            }

        return patterns

    def _load_domain_keywords(self, domain: str) -> Dict[str, List[str]]:
        """Charge les mots-clés par domaine"""

        keywords = {
            "legal": {
                "technical": ["article", "loi", "code", "jurisprudence", "procédure"],
                "entities": ["tribunal", "avocat", "juge", "partie"],
                "concepts": ["droit", "obligation", "responsabilité", "contrat"]
            },
            "technical": {
                "technical": ["API", "configuration", "installation", "architecture"],
                "entities": ["serveur", "base", "système", "application"],
                "concepts": ["sécurité", "performance", "scalabilité", "framework"]
            },
            "medical": {
                "technical": ["diagnostic", "traitement", "symptôme", "pathologie"],
                "entities": ["patient", "médecin", "hôpital", "médicament"],
                "concepts": ["santé", "prévention", "thérapie", "soin"]
            }
        }

        return keywords.get(domain, {"technical": [], "entities": [], "concepts": []})

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Classifie une requête et retourne le type avec un score de confiance

        Args:
            query: Requête à classifier

        Returns:
            Tuple (type_de_requête, score_de_confiance)
        """

        query_lower = query.lower().strip()
        scores = {}

        # Analyser chaque type de requête
        for query_type, patterns in self.query_patterns.items():
            score = 0.0

            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                if matches > 0:
                    score += matches * 0.3

            # Bonus pour mots-clés du domaine
            if query_type in [QueryType.DEFINITION, QueryType.EXPLANATION, QueryType.FACTUAL]:
                domain_matches = 0
                for category, keywords in self.domain_keywords.items():
                    domain_matches += sum(1 for keyword in keywords if keyword.lower() in query_lower)

                score += domain_matches * 0.1

            scores[query_type] = score

        # Déterminer le type principal
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]

            # Seuil minimum pour valider le type
            if best_score >= 0.2:
                confidence = min(best_score, 1.0)
                return best_type, confidence

        # Fallback: classifier par longueur et complexité
        word_count = len(query.split())

        if word_count <= 3:
            return QueryType.VAGUE, 0.8
        elif any(word in query_lower for word in ["comment", "how"]):
            return QueryType.HOW_TO, 0.6
        elif any(word in query_lower for word in ["qu'est", "what is", "définition"]):
            return QueryType.DEFINITION, 0.6
        else:
            return QueryType.EXPLANATION, 0.5


class AdaptivePromptEngine:
    """Moteur de génération de prompts adaptatifs"""

    def __init__(self, domain: str = "general", language: str = "fr"):
        self.domain = domain
        self.language = language

        # Classificateur de requêtes
        self.query_classifier = QueryTypeClassifier(domain, language)

        # Templates de prompts
        self.prompt_templates = self._load_prompt_templates(domain, language)

        # Styles de réponse par contexte
        self.style_mappings = self._load_style_mappings()

    def _load_prompt_templates(self, domain: str, language: str) -> Dict[str, Dict[str, str]]:
        """Charge les templates de prompts par domaine et langue"""

        if language == "fr":
            templates = {
                "system_prompts": {
                    "educational": "Tu es un assistant expert dans le domaine {domain}. Ta mission est d'expliquer les concepts de manière claire et pédagogique, en adaptant ton niveau d'explication à l'utilisateur.",

                    "professional": "Tu es un consultant expert en {domain}. Fournis des réponses précises, professionnelles et directement exploitables dans un contexte professionnel.",

                    "technical": "Tu es un expert technique en {domain}. Fournis des informations précises, détaillées et techniquement rigoureuses.",

                    "conversational": "Tu es un assistant amical et accessible spécialisé en {domain}. Explique les concepts de manière naturelle et engageante."
                },

                "instructions_by_type": {
                    QueryType.DEFINITION: {
                        "beginner": "Fournis une définition claire et simple. Utilise des exemples concrets et évite le jargon technique.",
                        "intermediate": "Donne une définition précise avec des exemples pertinents. Inclus le contexte d'usage.",
                        "advanced": "Fournis une définition complète avec les nuances et les implications techniques."
                    },

                    QueryType.HOW_TO: {
                        "beginner": "Explique étape par étape de manière très détaillée. Anticipe les questions et difficultés potentielles.",
                        "intermediate": "Fournis un guide structuré avec les étapes principales et les points d'attention.",
                        "advanced": "Donne les étapes avec les alternatives et les considérations avancées."
                    },

                    QueryType.COMPARISON: {
                        "beginner": "Compare de manière simple avec un tableau clair des avantages/inconvénients.",
                        "intermediate": "Analyse comparative structurée avec critères de choix et recommandations.",
                        "advanced": "Comparaison approfondie avec analyse des cas d'usage et implications."
                    },

                    QueryType.TROUBLESHOOTING: {
                        "beginner": "Guide de dépannage étape par étape avec vérifications simples en premier.",
                        "intermediate": "Approche diagnostique structurée avec solutions alternatives.",
                        "advanced": "Analyse technique approfondie avec diagnostic avancé et solutions complexes."
                    },

                    QueryType.EXAMPLE: {
                        "beginner": "Fournis des exemples très concrets et faciles à comprendre avec explications détaillées.",
                        "intermediate": "Donne des exemples pratiques pertinents avec contexte d'application.",
                        "advanced": "Présente des exemples sophistiqués avec variations et cas edge."
                    },

                    QueryType.EXPLANATION: {
                        "beginner": "Explique le 'pourquoi' de manière simple avec analogies si nécessaire.",
                        "intermediate": "Explication complète des mécanismes avec logique sous-jacente.",
                        "advanced": "Analyse détaillée des principes avec implications et cas complexes."
                    },

                    QueryType.FACTUAL: {
                        "beginner": "Présente les faits de manière organisée et accessible.",
                        "intermediate": "Informations factuelles avec contexte et sources fiables.",
                        "advanced": "Données complètes avec analyses et implications."
                    },

                    QueryType.VAGUE: {
                        "beginner": "Identifie les concepts clés et fournis une introduction progressive avec des questions pour clarifier.",
                        "intermediate": "Présente une vue d'ensemble structurée avec les aspects principaux et des pistes d'approfondissement.",
                        "advanced": "Analyse les différentes interprétations possibles avec recommandations de focus."
                    }
                },

                "context_instructions": {
                    "with_definitions": "Le contexte inclut des définitions. Utilise-les comme base mais enrichis avec tes explications.",
                    "with_examples": "Le contexte contient des exemples. Référence-les et ajoute tes propres exemples si pertinent.",
                    "with_prerequisites": "Des prérequis sont mentionnés. Assure-toi que l'utilisateur a les bases nécessaires.",
                    "enriched": "Le contexte est enrichi avec plusieurs sections. Utilise toutes les informations disponibles de manière cohérente."
                },

                "response_formatting": {
                    "structured": "Structure ta réponse avec des sections claires, des titres et des listes à puces quand approprié.",
                    "step_by_step": "Organise ta réponse en étapes numérotées claires et logiques.",
                    "conversational": "Adopte un ton naturel et engageant, comme dans une conversation.",
                    "technical": "Utilise la terminologie technique appropriée et sois précis dans les détails."
                }
            }

        else:
            # Templates anglais (version simplifiée)
            templates = {
                "system_prompts": {
                    "educational": "You are an expert assistant in {domain}. Your mission is to explain concepts clearly and pedagogically, adapting your explanation level to the user.",
                    # ... autres templates anglais
                },
                # ... autres sections en anglais
            }

        return templates

    def _load_style_mappings(self) -> Dict[str, ResponseStyle]:
        """Charge les mappings type de requête -> style de réponse"""

        return {
            QueryType.DEFINITION: ResponseStyle.EDUCATIONAL,
            QueryType.HOW_TO: ResponseStyle.STEP_BY_STEP,
            QueryType.COMPARISON: ResponseStyle.STRUCTURED,
            QueryType.TROUBLESHOOTING: ResponseStyle.STEP_BY_STEP,
            QueryType.EXAMPLE: ResponseStyle.CONVERSATIONAL,
            QueryType.EXPLANATION: ResponseStyle.EDUCATIONAL,
            QueryType.FACTUAL: ResponseStyle.PROFESSIONAL,
            QueryType.OPINION: ResponseStyle.CONVERSATIONAL,
            QueryType.VAGUE: ResponseStyle.EDUCATIONAL
        }

    def create_adaptive_prompt(self,
                             query: str,
                             context: str,
                             vagueness_score: float = 0.5,
                             user_level: str = "intermediate",
                             custom_style: Optional[ResponseStyle] = None) -> AdaptivePrompt:
        """
        Crée un prompt adaptatif complet

        Args:
            query: Requête utilisateur
            context: Contexte enrichi
            vagueness_score: Score de vague (0-1)
            user_level: Niveau utilisateur
            custom_style: Style de réponse forcé (optionnel)

        Returns:
            Prompt adaptatif généré
        """

        # Classifier la requête
        query_type, confidence = self.query_classifier.classify_query(query)

        # Analyser le contexte
        context_analysis = self._analyze_context(context)

        # Déterminer le style de réponse
        response_style = custom_style if custom_style else self.style_mappings.get(query_type, ResponseStyle.EDUCATIONAL)

        # Créer le contexte de prompt
        prompt_context = PromptContext(
            query=query,
            query_type=query_type,
            vagueness_score=vagueness_score,
            user_level=user_level,
            domain=self.domain,
            context_length=len(context),
            concepts_count=context_analysis["concepts_count"],
            has_examples=context_analysis["has_examples"],
            has_definitions=context_analysis["has_definitions"],
            response_style=response_style
        )

        # Générer les sections du prompt
        system_prompt = self._generate_system_prompt(prompt_context)
        user_prompt = self._generate_user_prompt(prompt_context)
        context_section = self._generate_context_section(context, prompt_context)
        instructions_section = self._generate_instructions_section(prompt_context)

        # Métadonnées
        metadata = {
            "query_type": query_type.value,
            "confidence": confidence,
            "response_style": response_style.value,
            "user_level": user_level,
            "vagueness_score": vagueness_score,
            "domain": self.domain,
            "context_analysis": context_analysis,
            "timestamp": datetime.now().isoformat()
        }

        return AdaptivePrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_section=context_section,
            instructions_section=instructions_section,
            metadata=metadata
        )

    def _analyze_context(self, context: str) -> Dict[str, Any]:
        """Analyse le contexte pour adapter le prompt"""

        analysis = {
            "length": len(context),
            "word_count": len(context.split()),
            "has_definitions": "=== DÉFINITIONS" in context,
            "has_examples": "=== EXEMPLES" in context,
            "has_prerequisites": "=== PRÉREQUIS" in context,
            "has_analogies": "=== ANALOGIES" in context,
            "has_qa": "=== QUESTIONS" in context,
            "concepts_count": 0,
            "structure_sections": []
        }

        # Compter les sections structurées
        section_patterns = [
            "=== CONTEXTE PRINCIPAL ===",
            "=== DÉFINITIONS",
            "=== EXEMPLES",
            "=== ANALOGIES",
            "=== PRÉREQUIS",
            "=== CONCEPTS LIÉS",
            "=== QUESTIONS"
        ]

        for pattern in section_patterns:
            if pattern in context:
                analysis["structure_sections"].append(pattern)

        # Estimer le nombre de concepts (heuristique simple)
        if analysis["has_definitions"]:
            definition_section = context.split("=== DÉFINITIONS")[1].split("===")[0] if "=== DÉFINITIONS" in context else ""
            analysis["concepts_count"] = len([line for line in definition_section.split('\n') if line.strip().startswith('-')])

        return analysis

    def _generate_system_prompt(self, prompt_context: PromptContext) -> str:
        """Génère le prompt système adaptatif"""

        # Sélectionner le template de base
        style_key = prompt_context.response_style.value
        if style_key not in self.prompt_templates["system_prompts"]:
            style_key = "educational"

        base_system = self.prompt_templates["system_prompts"][style_key].format(
            domain=prompt_context.domain
        )

        # Adaptations selon le contexte
        adaptations = []

        if prompt_context.vagueness_score >= 0.7:
            adaptations.append("La requête est vague, donc adopte une approche pédagogique progressive.")

        if prompt_context.user_level == "beginner":
            adaptations.append("L'utilisateur est débutant, utilise un vocabulaire simple et beaucoup d'exemples.")
        elif prompt_context.user_level == "advanced":
            adaptations.append("L'utilisateur est avancé, tu peux utiliser la terminologie technique appropriée.")

        if prompt_context.concepts_count > 5:
            adaptations.append("Le contexte contient de nombreux concepts, structure bien ta réponse.")

        # Assembler le prompt système
        if adaptations:
            adaptation_text = " ".join(adaptations)
            return f"{base_system}\n\nInstructions spécifiques: {adaptation_text}"
        else:
            return base_system

    def _generate_user_prompt(self, prompt_context: PromptContext) -> str:
        """Génère la section requête utilisateur"""

        user_prompt = f"Question de l'utilisateur: {prompt_context.query}"

        # Ajouter des clarifications pour requêtes vagues
        if prompt_context.vagueness_score >= 0.6:
            if prompt_context.user_level == "beginner":
                user_prompt += "\n\nNote: Cette question semble générale. Commence par identifier les concepts clés et propose des clarifications si nécessaire."
            else:
                user_prompt += "\n\nNote: Question générale. Présente une vue d'ensemble puis approfondis les aspects principaux."

        return user_prompt

    def _generate_context_section(self, context: str, prompt_context: PromptContext) -> str:
        """Génère la section contexte avec instructions d'usage"""

        context_section = f"CONTEXTE DISPONIBLE:\n{context}"

        # Instructions d'usage du contexte
        context_instructions = []

        if prompt_context.has_definitions:
            context_instructions.append(self.prompt_templates["context_instructions"]["with_definitions"])

        if prompt_context.has_examples:
            context_instructions.append(self.prompt_templates["context_instructions"]["with_examples"])

        if "=== PRÉREQUIS" in context:
            context_instructions.append(self.prompt_templates["context_instructions"]["with_prerequisites"])

        if len(prompt_context.custom_instructions) > 3:
            context_instructions.append(self.prompt_templates["context_instructions"]["enriched"])

        if context_instructions:
            instructions_text = " ".join(context_instructions)
            context_section += f"\n\nInstructions d'usage du contexte: {instructions_text}"

        return context_section

    def _generate_instructions_section(self, prompt_context: PromptContext) -> str:
        """Génère les instructions de réponse spécifiques"""

        # Instructions principales par type de requête
        query_type_key = prompt_context.query_type
        if query_type_key in self.prompt_templates["instructions_by_type"]:
            type_instructions = self.prompt_templates["instructions_by_type"][query_type_key]
            main_instruction = type_instructions.get(prompt_context.user_level, type_instructions.get("intermediate", ""))
        else:
            main_instruction = "Fournis une réponse claire et bien structurée."

        # Instructions de formatage
        style_key = prompt_context.response_style.value
        if style_key in self.prompt_templates["response_formatting"]:
            format_instruction = self.prompt_templates["response_formatting"][style_key]
        else:
            format_instruction = "Structure ta réponse de manière claire et logique."

        # Instructions finales
        final_instructions = []

        # Instruction principale
        final_instructions.append(f"INSTRUCTIONS DE RÉPONSE: {main_instruction}")

        # Formatage
        final_instructions.append(f"FORMATAGE: {format_instruction}")

        # Instructions spéciales pour requêtes vagues
        if prompt_context.vagueness_score >= 0.7:
            final_instructions.append("STRATÉGIE REQUÊTE VAGUE: Identifie d'abord les concepts clés, puis fournis une explication progressive du simple au complexe. Termine par des questions pour aider l'utilisateur à préciser sa demande.")

        # Longueur adaptée
        if prompt_context.context_length > 3000:
            final_instructions.append("LONGUEUR: Le contexte est riche, utilise-le pleinement mais reste concis.")
        elif prompt_context.context_length < 500:
            final_instructions.append("LONGUEUR: Le contexte est limité, enrichis avec tes connaissances tout en restant fidèle aux informations fournies.")

        return "\n\n".join(final_instructions)

    def optimize_prompt_for_llm(self, adaptive_prompt: AdaptivePrompt, llm_model: str = "gpt-3.5-turbo") -> AdaptivePrompt:
        """Optimise le prompt pour un modèle LLM spécifique"""

        # Optimisations par modèle
        if "gpt-4" in llm_model.lower():
            # GPT-4: peut gérer des prompts plus complexes
            pass
        elif "gpt-3.5" in llm_model.lower():
            # GPT-3.5: prompts plus concis
            adaptive_prompt.instructions_section = self._condense_instructions(adaptive_prompt.instructions_section)
        elif "claude" in llm_model.lower():
            # Claude: aime les instructions très structurées
            adaptive_prompt.instructions_section = self._structure_for_claude(adaptive_prompt.instructions_section)

        # Limiter la longueur totale si nécessaire
        total_length = len(adaptive_prompt.get_full_prompt())
        if total_length > 8000:  # Limite pour la plupart des modèles
            adaptive_prompt = self._truncate_prompt(adaptive_prompt, 8000)

        # Métadonnées d'optimisation
        adaptive_prompt.metadata["optimized_for"] = llm_model
        adaptive_prompt.metadata["total_length"] = len(adaptive_prompt.get_full_prompt())

        return adaptive_prompt

    def _condense_instructions(self, instructions: str) -> str:
        """Condense les instructions pour modèles avec limite de contexte"""

        # Simplifier et raccourcir
        condensed = instructions.replace("INSTRUCTIONS DE RÉPONSE: ", "")
        condensed = condensed.replace("FORMATAGE: ", "Format: ")
        condensed = condensed.replace("STRATÉGIE REQUÊTE VAGUE: ", "Si vague: ")

        return condensed

    def _structure_for_claude(self, instructions: str) -> str:
        """Structure les instructions pour Claude"""

        # Claude préfère des listes à puces claires
        lines = instructions.split('\n')
        structured_lines = []

        for line in lines:
            if line.startswith("INSTRUCTIONS"):
                structured_lines.append("Instructions principales:")
            elif line.startswith("FORMATAGE"):
                structured_lines.append("• Format de réponse:")
            elif line.startswith("STRATÉGIE"):
                structured_lines.append("• Stratégie spéciale:")
            else:
                structured_lines.append(f"  {line}")

        return '\n'.join(structured_lines)

    def _truncate_prompt(self, adaptive_prompt: AdaptivePrompt, max_length: int) -> AdaptivePrompt:
        """Tronque le prompt en préservant les parties essentielles"""

        # Ordre de priorité: system > instructions > user > context
        essential_parts = [
            adaptive_prompt.system_prompt,
            adaptive_prompt.instructions_section,
            adaptive_prompt.user_prompt
        ]

        essential_length = sum(len(part) for part in essential_parts)

        if essential_length < max_length:
            # Tronquer le contexte
            remaining_length = max_length - essential_length - 100  # Marge
            if len(adaptive_prompt.context_section) > remaining_length:
                truncated_context = adaptive_prompt.context_section[:remaining_length] + "\n\n[... contexte tronqué ...]"
                adaptive_prompt.context_section = truncated_context

        adaptive_prompt.metadata["truncated"] = True
        return adaptive_prompt


# Fonctions d'intégration pour RAG Chunk Lab
def create_adaptive_prompt_engine(domain: str = "general",
                                 language: str = "fr") -> AdaptivePromptEngine:
    """
    Crée un moteur de prompt adaptatif configuré

    Args:
        domain: Domaine spécialisé
        language: Langue

    Returns:
        Moteur de prompt adaptatif
    """

    return AdaptivePromptEngine(domain=domain, language=language)


def generate_optimized_prompt_for_vague_query(engine: AdaptivePromptEngine,
                                            query: str,
                                            enriched_context: str,
                                            vagueness_score: float,
                                            user_level: str = "intermediate",
                                            llm_model: str = "gpt-3.5-turbo") -> str:
    """
    Génère un prompt optimisé pour une requête vague

    Args:
        engine: Moteur de prompt adaptatif
        query: Requête utilisateur
        enriched_context: Contexte enrichi
        vagueness_score: Score de vague
        user_level: Niveau utilisateur
        llm_model: Modèle LLM cible

    Returns:
        Prompt optimisé complet
    """

    # Générer le prompt adaptatif
    adaptive_prompt = engine.create_adaptive_prompt(
        query=query,
        context=enriched_context,
        vagueness_score=vagueness_score,
        user_level=user_level
    )

    # Optimiser pour le modèle LLM
    optimized_prompt = engine.optimize_prompt_for_llm(adaptive_prompt, llm_model)

    return optimized_prompt.get_full_prompt()


def batch_generate_adaptive_prompts(engine: AdaptivePromptEngine,
                                   queries_contexts: List[Tuple[str, str, float]],
                                   user_level: str = "intermediate") -> List[AdaptivePrompt]:
    """
    Génère des prompts adaptatifs en lot

    Args:
        engine: Moteur de prompt adaptatif
        queries_contexts: Liste de (query, context, vagueness_score)
        user_level: Niveau utilisateur

    Returns:
        Liste de prompts adaptatifs
    """

    adaptive_prompts = []

    for query, context, vagueness_score in queries_contexts:
        try:
            adaptive_prompt = engine.create_adaptive_prompt(
                query=query,
                context=context,
                vagueness_score=vagueness_score,
                user_level=user_level
            )
            adaptive_prompts.append(adaptive_prompt)

        except Exception as e:
            logger.error(f"Failed to generate adaptive prompt for query '{query[:50]}...': {e}")
            # Prompt de fallback
            fallback_prompt = AdaptivePrompt(
                system_prompt=f"Tu es un assistant expert en {engine.domain}.",
                user_prompt=f"Question: {query}",
                context_section=f"Contexte: {context}",
                instructions_section="Fournis une réponse claire et utile.",
                metadata={"error": str(e), "fallback": True}
            )
            adaptive_prompts.append(fallback_prompt)

    return adaptive_prompts