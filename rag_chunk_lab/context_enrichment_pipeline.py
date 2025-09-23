# context_enrichment_pipeline.py
"""
Pipeline d'enrichissement contextuel automatique pour optimiser les réponses RAG
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import re
import asyncio
from datetime import datetime

# Imports conditionnels
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ContextEnrichmentConfig:
    """Configuration pour l'enrichissement contextuel"""
    # LLM Configuration
    llm_provider: str = "openai"  # "openai", "azure", "ollama"
    llm_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None

    # Enrichment Settings
    enable_definitions: bool = True
    enable_examples: bool = True
    enable_analogies: bool = True
    enable_prerequisites: bool = True
    enable_related_concepts: bool = True
    enable_qa_pairs: bool = True

    # Context Limits
    max_context_length: int = 4000
    max_definitions: int = 5
    max_examples: int = 3
    max_analogies: int = 2
    max_prerequisites: int = 4
    max_related_concepts: int = 6
    max_qa_pairs: int = 3

    # Domain Configuration
    domain: str = "general"
    language: str = "fr"

    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = 24


@dataclass
class EnrichedContext:
    """Contexte enrichi avec métadonnées"""
    original_context: str
    enriched_sections: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    def get_full_context(self) -> str:
        """Retourne le contexte complet enrichi"""
        sections = []

        # Contexte original
        sections.append("=== CONTEXTE PRINCIPAL ===")
        sections.append(self.original_context)

        # Définitions
        if "definitions" in self.enriched_sections and self.enriched_sections["definitions"]:
            sections.append("\n=== DÉFINITIONS CLÉS ===")
            sections.extend(self.enriched_sections["definitions"])

        # Exemples
        if "examples" in self.enriched_sections and self.enriched_sections["examples"]:
            sections.append("\n=== EXEMPLES PRATIQUES ===")
            sections.extend(self.enriched_sections["examples"])

        # Prérequis
        if "prerequisites" in self.enriched_sections and self.enriched_sections["prerequisites"]:
            sections.append("\n=== CONCEPTS PRÉREQUIS ===")
            sections.extend(self.enriched_sections["prerequisites"])

        # Analogies
        if "analogies" in self.enriched_sections and self.enriched_sections["analogies"]:
            sections.append("\n=== ANALOGIES ===")
            sections.extend(self.enriched_sections["analogies"])

        # Concepts liés
        if "related_concepts" in self.enriched_sections and self.enriched_sections["related_concepts"]:
            sections.append("\n=== CONCEPTS LIÉS ===")
            sections.extend(self.enriched_sections["related_concepts"])

        # Q&A
        if "qa_pairs" in self.enriched_sections and self.enriched_sections["qa_pairs"]:
            sections.append("\n=== QUESTIONS FRÉQUENTES ===")
            sections.extend(self.enriched_sections["qa_pairs"])

        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "original_context": self.original_context,
            "enriched_sections": self.enriched_sections,
            "metadata": self.metadata,
            "quality_score": self.quality_score
        }


class ContextEnrichmentPipeline:
    """Pipeline d'enrichissement contextuel automatique"""

    def __init__(self, config: ContextEnrichmentConfig):
        self.config = config

        # Configuration LLM
        if OPENAI_AVAILABLE and config.openai_api_key:
            self.llm_client = OpenAI(api_key=config.openai_api_key)
        else:
            self.llm_client = None
            logger.warning("OpenAI client not available. Some enrichment features will be limited.")

        # Configuration NLP
        if SPACY_AVAILABLE:
            try:
                model_name = "fr_core_news_sm" if config.language == "fr" else "en_core_web_sm"
                self.nlp = spacy.load(model_name)
            except OSError:
                self.nlp = None
                logger.warning(f"SpaCy model not found. Install with: python -m spacy download {model_name}")
        else:
            self.nlp = None

        # Cache pour optimisation
        self._enrichment_cache = {}

        # Patterns et règles par domaine
        self.domain_patterns = self._load_domain_patterns(config.domain)

        # Templates de prompts
        self.prompt_templates = self._load_prompt_templates(config.domain, config.language)

    def _load_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """Charge les patterns spécifiques au domaine"""

        patterns = {
            "legal": {
                "key_concepts": [
                    "droit", "loi", "article", "jurisprudence", "tribunal", "procédure",
                    "contrat", "obligation", "responsabilité", "préjudice", "avocat"
                ],
                "definition_indicators": [
                    r"(?:art\.|article)\s*\d+",
                    r"selon\s+(?:la\s+)?loi",
                    r"aux\s+termes\s+de",
                    r"en\s+application\s+de"
                ],
                "example_indicators": [
                    r"par\s+exemple",
                    r"notamment\s+dans\s+le\s+cas",
                    r"comme\s+l'illustre",
                    r"tel\s+que\s+prévu"
                ],
                "prerequisite_concepts": [
                    "droit civil", "procédure", "juridiction", "compétence"
                ]
            },

            "technical": {
                "key_concepts": [
                    "API", "architecture", "configuration", "implémentation", "sécurité",
                    "performance", "base de données", "framework", "algorithme"
                ],
                "definition_indicators": [
                    r"API\s+(?:est|désigne)",
                    r"(?:une|la)\s+architecture",
                    r"(?:le|un)\s+framework",
                    r"algorithme\s+de"
                ],
                "example_indicators": [
                    r"exemple\s+d'implémentation",
                    r"cas\s+d'usage",
                    r"pour\s+illustrer",
                    r"dans\s+la\s+pratique"
                ],
                "prerequisite_concepts": [
                    "HTTP", "JSON", "base de données", "programmation"
                ]
            },

            "medical": {
                "key_concepts": [
                    "symptôme", "diagnostic", "traitement", "pathologie", "thérapie",
                    "prévention", "patient", "médecin", "médicament"
                ],
                "definition_indicators": [
                    r"symptôme\s+de",
                    r"diagnostic\s+de",
                    r"traitement\s+par",
                    r"pathologie\s+caractérisée"
                ],
                "example_indicators": [
                    r"cas\s+clinique",
                    r"patient\s+présentant",
                    r"exemple\s+de\s+traitement",
                    r"étude\s+de\s+cas"
                ],
                "prerequisite_concepts": [
                    "anatomie", "physiologie", "pharmacologie", "sémiologie"
                ]
            }
        }

        return patterns.get(domain, {
            "key_concepts": [],
            "definition_indicators": [],
            "example_indicators": [],
            "prerequisite_concepts": []
        })

    def _load_prompt_templates(self, domain: str, language: str) -> Dict[str, str]:
        """Charge les templates de prompts pour l'enrichissement"""

        if language == "fr":
            templates = {
                "definitions": """
Analysez le contexte suivant et identifiez les {max_items} concepts clés qui nécessitent une définition claire pour le domaine {domain}.

Contexte: {context}

Pour chaque concept identifié, fournissez une définition concise et précise (1-2 phrases maximum).

Format de réponse:
- Concept 1: Définition claire et concise
- Concept 2: Définition claire et concise
...
""",

                "examples": """
Basé sur le contexte suivant dans le domaine {domain}, générez {max_items} exemples pratiques et concrets pour illustrer les concepts principaux.

Contexte: {context}

Chaque exemple doit être:
- Concret et pratique
- Adapté au niveau de l'utilisateur
- Directement lié aux concepts du contexte

Format de réponse:
1. Exemple pratique 1
2. Exemple pratique 2
...
""",

                "analogies": """
Créez {max_items} analogies simples et compréhensibles pour expliquer les concepts complexes du contexte suivant (domaine {domain}).

Contexte: {context}

Les analogies doivent:
- Utiliser des concepts familiers
- Être facilement compréhensibles
- Clarifier les concepts difficiles

Format de réponse:
• Analogie 1: Concept = Comparaison simple
• Analogie 2: Concept = Comparaison simple
...
""",

                "prerequisites": """
Identifiez les {max_items} concepts prérequis essentiels qu'un utilisateur devrait comprendre avant d'aborder le contexte suivant (domaine {domain}).

Contexte: {context}

Pour chaque prérequis, expliquez brièvement pourquoi il est important.

Format de réponse:
→ Prérequis 1: Explication de son importance
→ Prérequis 2: Explication de son importance
...
""",

                "related_concepts": """
Listez {max_items} concepts liés et complémentaires au contexte suivant (domaine {domain}) qui pourraient intéresser l'utilisateur.

Contexte: {context}

Pour chaque concept lié, ajoutez une brève explication de la relation.

Format de réponse:
⊳ Concept lié 1: Relation avec le contexte
⊳ Concept lié 2: Relation avec le contexte
...
""",

                "qa_pairs": """
Générez {max_items} paires question-réponse courtes basées sur le contexte suivant (domaine {domain}).

Contexte: {context}

Les questions doivent être:
- Fréquemment posées par les utilisateurs
- Directement liées au contexte
- Avec des réponses concises (1-2 phrases)

Format de réponse:
Q: Question fréquente 1 ?
R: Réponse concise

Q: Question fréquente 2 ?
R: Réponse concise
...
"""
            }
        else:
            # Templates en anglais
            templates = {
                "definitions": """
Analyze the following context and identify the {max_items} key concepts that need clear definitions for the {domain} domain.

Context: {context}

For each identified concept, provide a concise and precise definition (1-2 sentences maximum).

Response format:
- Concept 1: Clear and concise definition
- Concept 2: Clear and concise definition
...
""",
                # ... autres templates en anglais
            }

        return templates

    def enrich_context(self,
                      original_context: str,
                      query: str = "",
                      vagueness_score: float = 0.5,
                      user_level: str = "intermediate") -> EnrichedContext:
        """
        Enrichit un contexte de manière complète

        Args:
            original_context: Contexte original à enrichir
            query: Requête utilisateur (optionnel)
            vagueness_score: Score de vague de la requête (0-1)
            user_level: Niveau utilisateur ("beginner", "intermediate", "advanced")

        Returns:
            Contexte enrichi avec toutes les sections
        """

        # Vérifier le cache
        cache_key = self._generate_cache_key(original_context, query, vagueness_score, user_level)
        if self.config.enable_cache and cache_key in self._enrichment_cache:
            cached_result = self._enrichment_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return EnrichedContext(**cached_result["data"])

        # Créer contexte enrichi
        enriched_context = EnrichedContext(
            original_context=original_context,
            metadata={
                "query": query,
                "vagueness_score": vagueness_score,
                "user_level": user_level,
                "domain": self.config.domain,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Analyse du contexte pour guider l'enrichissement
        context_analysis = self._analyze_context(original_context, query)
        enriched_context.metadata["context_analysis"] = context_analysis

        # Enrichissement adaptatif selon la vague de la requête
        enrichment_plan = self._create_enrichment_plan(context_analysis, vagueness_score, user_level)

        # Exécuter l'enrichissement selon le plan
        if enrichment_plan.get("definitions", False) and self.config.enable_definitions:
            definitions = self._generate_definitions(original_context, context_analysis)
            enriched_context.enriched_sections["definitions"] = definitions

        if enrichment_plan.get("examples", False) and self.config.enable_examples:
            examples = self._generate_examples(original_context, context_analysis, user_level)
            enriched_context.enriched_sections["examples"] = examples

        if enrichment_plan.get("analogies", False) and self.config.enable_analogies:
            analogies = self._generate_analogies(original_context, context_analysis, user_level)
            enriched_context.enriched_sections["analogies"] = analogies

        if enrichment_plan.get("prerequisites", False) and self.config.enable_prerequisites:
            prerequisites = self._generate_prerequisites(original_context, context_analysis)
            enriched_context.enriched_sections["prerequisites"] = prerequisites

        if enrichment_plan.get("related_concepts", False) and self.config.enable_related_concepts:
            related = self._generate_related_concepts(original_context, context_analysis)
            enriched_context.enriched_sections["related_concepts"] = related

        if enrichment_plan.get("qa_pairs", False) and self.config.enable_qa_pairs:
            qa_pairs = self._generate_qa_pairs(original_context, context_analysis)
            enriched_context.enriched_sections["qa_pairs"] = qa_pairs

        # Calculer score de qualité
        enriched_context.quality_score = self._calculate_quality_score(enriched_context)

        # Mettre en cache
        if self.config.enable_cache:
            self._cache_result(cache_key, enriched_context)

        return enriched_context

    def _analyze_context(self, context: str, query: str = "") -> Dict[str, Any]:
        """Analyse le contexte pour guider l'enrichissement"""

        analysis = {
            "length": len(context),
            "word_count": len(context.split()),
            "concepts": [],
            "complexity_indicators": [],
            "definition_needs": [],
            "example_opportunities": [],
            "technical_level": "intermediate"
        }

        context_lower = context.lower()

        # Identifier les concepts clés du domaine
        domain_concepts = self.domain_patterns.get("key_concepts", [])
        found_concepts = [concept for concept in domain_concepts if concept.lower() in context_lower]
        analysis["concepts"] = found_concepts

        # Analyser les indicateurs de définition
        definition_patterns = self.domain_patterns.get("definition_indicators", [])
        for pattern in definition_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                analysis["definition_needs"].append(pattern)

        # Analyser les opportunités d'exemples
        example_patterns = self.domain_patterns.get("example_indicators", [])
        for pattern in example_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                analysis["example_opportunities"].append(pattern)

        # Analyse NLP si disponible
        if self.nlp:
            doc = self.nlp(context)

            # Entités nommées
            analysis["entities"] = [ent.text for ent in doc.ents]

            # Complexité basée sur la structure
            sentences = list(doc.sents)
            if sentences:
                avg_sentence_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences)
                if avg_sentence_length > 20:
                    analysis["technical_level"] = "advanced"
                elif avg_sentence_length < 10:
                    analysis["technical_level"] = "beginner"

        return analysis

    def _create_enrichment_plan(self,
                              context_analysis: Dict[str, Any],
                              vagueness_score: float,
                              user_level: str) -> Dict[str, bool]:
        """Crée un plan d'enrichissement adaptatif"""

        plan = {
            "definitions": False,
            "examples": False,
            "analogies": False,
            "prerequisites": False,
            "related_concepts": False,
            "qa_pairs": False
        }

        # Logique adaptative selon la vague de la requête
        if vagueness_score >= 0.7:
            # Très vague: enrichissement maximal
            plan.update({
                "definitions": True,
                "examples": True,
                "analogies": True,
                "prerequisites": True,
                "related_concepts": True,
                "qa_pairs": True
            })

        elif vagueness_score >= 0.4:
            # Moyennement vague: enrichissement ciblé
            plan.update({
                "definitions": len(context_analysis.get("concepts", [])) > 2,
                "examples": True,
                "analogies": user_level == "beginner",
                "prerequisites": user_level in ["beginner", "intermediate"],
                "related_concepts": True,
                "qa_pairs": True
            })

        else:
            # Précise: enrichissement minimal et ciblé
            plan.update({
                "definitions": len(context_analysis.get("definition_needs", [])) > 0,
                "examples": len(context_analysis.get("example_opportunities", [])) > 0,
                "analogies": False,
                "prerequisites": user_level == "beginner",
                "related_concepts": False,
                "qa_pairs": len(context_analysis.get("concepts", [])) > 3
            })

        # Ajustements selon le niveau utilisateur
        if user_level == "advanced":
            plan["analogies"] = False
            plan["prerequisites"] = False
        elif user_level == "beginner":
            plan["analogies"] = True
            plan["prerequisites"] = True

        return plan

    def _generate_definitions(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des définitions pour les concepts clés"""

        if not self.llm_client:
            return self._generate_definitions_fallback(context, analysis)

        try:
            prompt = self.prompt_templates["definitions"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_definitions
            )

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )

            definitions_text = response.choices[0].message.content
            definitions = self._parse_list_response(definitions_text, "- ")

            return definitions[:self.config.max_definitions]

        except Exception as e:
            logger.warning(f"LLM definition generation failed: {e}")
            return self._generate_definitions_fallback(context, analysis)

    def _generate_definitions_fallback(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des définitions sans LLM (fallback)"""

        definitions = []
        concepts = analysis.get("concepts", [])

        # Définitions basiques pour les concepts trouvés
        for concept in concepts[:self.config.max_definitions]:
            definition = f"{concept.title()}: Concept important dans le domaine {self.config.domain}"
            definitions.append(definition)

        return definitions

    def _generate_examples(self, context: str, analysis: Dict[str, Any], user_level: str) -> List[str]:
        """Génère des exemples pratiques"""

        if not self.llm_client:
            return self._generate_examples_fallback(context, analysis)

        try:
            prompt = self.prompt_templates["examples"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_examples
            )

            # Adapter selon le niveau utilisateur
            if user_level == "beginner":
                prompt += "\nPrivilégier des exemples très simples et concrets."
            elif user_level == "advanced":
                prompt += "\nInclure des exemples plus sophistiqués et des cas d'usage avancés."

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.5
            )

            examples_text = response.choices[0].message.content
            examples = self._parse_numbered_response(examples_text)

            return examples[:self.config.max_examples]

        except Exception as e:
            logger.warning(f"LLM example generation failed: {e}")
            return self._generate_examples_fallback(context, analysis)

    def _generate_examples_fallback(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des exemples sans LLM (fallback)"""

        examples = []
        concepts = analysis.get("concepts", [])

        for concept in concepts[:self.config.max_examples]:
            example = f"Exemple d'application de {concept} dans un contexte pratique"
            examples.append(example)

        return examples

    def _generate_analogies(self, context: str, analysis: Dict[str, Any], user_level: str) -> List[str]:
        """Génère des analogies pour simplifier les concepts"""

        if not self.llm_client or user_level == "advanced":
            return []

        try:
            prompt = self.prompt_templates["analogies"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_analogies
            )

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )

            analogies_text = response.choices[0].message.content
            analogies = self._parse_list_response(analogies_text, "• ")

            return analogies[:self.config.max_analogies]

        except Exception as e:
            logger.warning(f"LLM analogy generation failed: {e}")
            return []

    def _generate_prerequisites(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère les prérequis conceptuels"""

        if not self.llm_client:
            return self._generate_prerequisites_fallback(context, analysis)

        try:
            prompt = self.prompt_templates["prerequisites"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_prerequisites
            )

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4
            )

            prerequisites_text = response.choices[0].message.content
            prerequisites = self._parse_list_response(prerequisites_text, "→ ")

            return prerequisites[:self.config.max_prerequisites]

        except Exception as e:
            logger.warning(f"LLM prerequisite generation failed: {e}")
            return self._generate_prerequisites_fallback(context, analysis)

    def _generate_prerequisites_fallback(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des prérequis sans LLM (fallback)"""

        domain_prereqs = self.domain_patterns.get("prerequisite_concepts", [])
        return domain_prereqs[:self.config.max_prerequisites]

    def _generate_related_concepts(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des concepts liés"""

        if not self.llm_client:
            return []

        try:
            prompt = self.prompt_templates["related_concepts"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_related_concepts
            )

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.6
            )

            related_text = response.choices[0].message.content
            related = self._parse_list_response(related_text, "⊳ ")

            return related[:self.config.max_related_concepts]

        except Exception as e:
            logger.warning(f"LLM related concepts generation failed: {e}")
            return []

    def _generate_qa_pairs(self, context: str, analysis: Dict[str, Any]) -> List[str]:
        """Génère des paires question-réponse"""

        if not self.llm_client:
            return []

        try:
            prompt = self.prompt_templates["qa_pairs"].format(
                context=context,
                domain=self.config.domain,
                max_items=self.config.max_qa_pairs
            )

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.4
            )

            qa_text = response.choices[0].message.content
            qa_pairs = self._parse_qa_response(qa_text)

            return qa_pairs[:self.config.max_qa_pairs]

        except Exception as e:
            logger.warning(f"LLM QA generation failed: {e}")
            return []

    def _parse_list_response(self, text: str, prefix: str) -> List[str]:
        """Parse une réponse en liste avec préfixe"""

        lines = text.strip().split('\n')
        items = []

        for line in lines:
            line = line.strip()
            if line.startswith(prefix):
                item = line[len(prefix):].strip()
                if item:
                    items.append(item)

        return items

    def _parse_numbered_response(self, text: str) -> List[str]:
        """Parse une réponse numérotée"""

        lines = text.strip().split('\n')
        items = []

        for line in lines:
            line = line.strip()
            # Matcher les formats: "1.", "1)", "1 -", etc.
            if re.match(r'^\d+[\.\)\-]\s+', line):
                item = re.sub(r'^\d+[\.\)\-]\s+', '', line).strip()
                if item:
                    items.append(item)

        return items

    def _parse_qa_response(self, text: str) -> List[str]:
        """Parse les paires Q&A"""

        lines = text.strip().split('\n')
        qa_pairs = []
        current_q = ""
        current_r = ""

        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_q and current_r:
                    qa_pairs.append(f"{current_q}\n{current_r}")
                current_q = line
                current_r = ""
            elif line.startswith('R:'):
                current_r = line

        # Ajouter la dernière paire
        if current_q and current_r:
            qa_pairs.append(f"{current_q}\n{current_r}")

        return qa_pairs

    def _calculate_quality_score(self, enriched_context: EnrichedContext) -> float:
        """Calcule un score de qualité pour le contexte enrichi"""

        score = 0.0

        # Score basé sur le nombre de sections enrichies
        sections_count = len([s for s in enriched_context.enriched_sections.values() if s])
        score += sections_count * 0.15

        # Score basé sur la densité de contenu
        total_enriched_length = sum(
            len(' '.join(items)) for items in enriched_context.enriched_sections.values()
        )
        original_length = len(enriched_context.original_context)

        if original_length > 0:
            enrichment_ratio = total_enriched_length / original_length
            score += min(enrichment_ratio * 0.3, 0.3)  # Maximum 0.3 pour ce critère

        # Score basé sur l'analyse du contexte
        context_analysis = enriched_context.metadata.get("context_analysis", {})
        concepts_count = len(context_analysis.get("concepts", []))
        score += min(concepts_count * 0.05, 0.2)  # Maximum 0.2

        # Bonus pour cohérence (heuristique simple)
        if len(enriched_context.enriched_sections) > 2:
            score += 0.1

        return min(score, 1.0)  # Limiter à 1.0

    def _generate_cache_key(self, context: str, query: str, vagueness_score: float, user_level: str) -> str:
        """Génère une clé de cache"""

        import hashlib

        content = f"{context}|{query}|{vagueness_score}|{user_level}|{self.config.domain}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Vérifie si le cache est encore valide"""

        if not self.config.enable_cache:
            return False

        cache_timestamp = cached_result.get("timestamp")
        if not cache_timestamp:
            return False

        from datetime import datetime, timedelta

        cache_time = datetime.fromisoformat(cache_timestamp)
        expiry_time = cache_time + timedelta(hours=self.config.cache_ttl_hours)

        return datetime.now() < expiry_time

    def _cache_result(self, cache_key: str, enriched_context: EnrichedContext):
        """Met en cache un résultat d'enrichissement"""

        self._enrichment_cache[cache_key] = {
            "data": enriched_context.to_dict(),
            "timestamp": datetime.now().isoformat()
        }

    def batch_enrich_contexts(self,
                            contexts: List[Tuple[str, str]],
                            vagueness_scores: List[float],
                            user_level: str = "intermediate") -> List[EnrichedContext]:
        """
        Enrichit plusieurs contextes en lot

        Args:
            contexts: Liste de tuples (contexte, requête)
            vagueness_scores: Scores de vague correspondants
            user_level: Niveau utilisateur

        Returns:
            Liste de contextes enrichis
        """

        enriched_contexts = []

        for i, (context, query) in enumerate(contexts):
            vagueness_score = vagueness_scores[i] if i < len(vagueness_scores) else 0.5

            try:
                enriched = self.enrich_context(
                    original_context=context,
                    query=query,
                    vagueness_score=vagueness_score,
                    user_level=user_level
                )
                enriched_contexts.append(enriched)

            except Exception as e:
                logger.error(f"Failed to enrich context {i}: {e}")
                # Contexte de fallback
                fallback = EnrichedContext(
                    original_context=context,
                    metadata={"error": str(e), "query": query}
                )
                enriched_contexts.append(fallback)

        return enriched_contexts


# Fonctions d'intégration pour RAG Chunk Lab
def create_context_enrichment_pipeline(domain: str = "general",
                                     openai_api_key: Optional[str] = None,
                                     config_overrides: Dict[str, Any] = None) -> ContextEnrichmentPipeline:
    """
    Crée un pipeline d'enrichissement contextuel configuré

    Args:
        domain: Domaine spécialisé
        openai_api_key: Clé API OpenAI
        config_overrides: Overrides de configuration

    Returns:
        Pipeline d'enrichissement configuré
    """

    config = ContextEnrichmentConfig(
        domain=domain,
        openai_api_key=openai_api_key
    )

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ContextEnrichmentPipeline(config)


def enrich_context_for_vague_query(pipeline: ContextEnrichmentPipeline,
                                  retrieved_chunks: List[str],
                                  query: str,
                                  vagueness_score: float,
                                  user_level: str = "intermediate") -> str:
    """
    Enrichit le contexte spécifiquement pour une requête vague

    Args:
        pipeline: Pipeline d'enrichissement
        retrieved_chunks: Chunks récupérés
        query: Requête utilisateur
        vagueness_score: Score de vague
        user_level: Niveau utilisateur

    Returns:
        Contexte enrichi formaté
    """

    # Combiner les chunks en contexte principal
    original_context = "\n\n".join(retrieved_chunks)

    # Enrichir le contexte
    enriched = pipeline.enrich_context(
        original_context=original_context,
        query=query,
        vagueness_score=vagueness_score,
        user_level=user_level
    )

    return enriched.get_full_context()