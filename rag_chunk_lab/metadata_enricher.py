# metadata_enricher.py
"""
Système d'enrichissement de métadonnées pour chunks avec intelligence contextuelle
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from collections import Counter, defaultdict
import spacy
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnrichedMetadata:
    """Métadonnées enrichies complètes pour un chunk"""

    # Identifiants et structure
    chunk_id: str
    document_id: str
    granularity: str
    parent_id: Optional[str] = None
    children_ids: List[str] = None

    # Contenu sémantique
    entities: List[Dict[str, Any]] = None  # {"text": "API", "label": "PRODUCT", "confidence": 0.9}
    concepts: List[Dict[str, Any]] = None  # {"concept": "authentication", "relevance": 0.8, "context": "..."}
    keywords: List[Dict[str, Any]] = None  # {"keyword": "OAuth", "frequency": 3, "importance": 0.7}

    # Classification de contenu
    content_type: str = "content"  # "definition", "example", "procedure", "summary", "content"
    domain_tags: List[str] = None  # ["security", "api", "authentication"]
    complexity_level: str = "medium"  # "basic", "medium", "advanced", "expert"

    # Métriques quantitatives
    semantic_density: float = 0.0  # Densité d'information sémantique
    domain_relevance: float = 0.0  # Pertinence pour le domaine
    readability_score: float = 0.0  # Score de lisibilité
    information_density: float = 0.0  # Densité d'information factuelle

    # Relations contextuelles
    related_chunks: List[str] = None  # IDs de chunks similaires
    prerequisite_concepts: List[str] = None  # Concepts prérequis
    follow_up_concepts: List[str] = None  # Concepts qui suivent logiquement

    # Annotations linguistiques
    language: str = "fr"
    named_entities_count: int = 0
    technical_terms_count: int = 0
    definition_indicators: List[str] = None  # Mots indiquant une définition

    # Métriques d'usage
    query_relevance_history: List[float] = None  # Historique de pertinence
    usage_frequency: int = 0  # Fréquence d'utilisation
    user_feedback_score: float = 0.0  # Score de feedback utilisateur

    def __post_init__(self):
        """Initialisation des listes vides"""
        for field in ['children_ids', 'entities', 'concepts', 'keywords',
                     'domain_tags', 'related_chunks', 'prerequisite_concepts',
                     'follow_up_concepts', 'definition_indicators', 'query_relevance_history']:
            if getattr(self, field) is None:
                setattr(self, field, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnrichedMetadata':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class MetadataEnricher:
    """Enrichisseur de métadonnées intelligent"""

    def __init__(self, domain: str = "general", language: str = "fr"):
        self.domain = domain
        self.language = language

        # Configuration NLP
        try:
            model_name = "fr_core_news_sm" if language == "fr" else "en_core_web_sm"
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"SpaCy model not found. Install with: python -m spacy download {model_name}")
            self.nlp = None

        # Chargement des ressources domaine
        self.domain_knowledge = self._load_domain_knowledge(domain)
        self.complexity_indicators = self._load_complexity_indicators()
        self.definition_patterns = self._load_definition_patterns()

        # Cache pour optimisation
        self._concept_cache = {}
        self._entity_cache = {}

    def _load_domain_knowledge(self, domain: str) -> Dict[str, Any]:
        """Charge la base de connaissances du domaine"""

        knowledge_bases = {
            "legal": {
                "key_concepts": {
                    "droit": {"category": "fundamental", "complexity": "basic"},
                    "jurisprudence": {"category": "case_law", "complexity": "advanced"},
                    "procédure": {"category": "process", "complexity": "medium"},
                    "obligation": {"category": "fundamental", "complexity": "medium"},
                    "responsabilité": {"category": "fundamental", "complexity": "medium"},
                    "contrat": {"category": "civil_law", "complexity": "medium"},
                    "tribunal": {"category": "institution", "complexity": "basic"},
                    "avocat": {"category": "profession", "complexity": "basic"}
                },
                "technical_terms": [
                    "article", "alinéa", "paragraphe", "sous-section",
                    "code civil", "code pénal", "code de commerce",
                    "arrêt", "jugement", "ordonnance", "décret"
                ],
                "definition_indicators": [
                    "est défini comme", "s'entend de", "désigne",
                    "on entend par", "constitue", "comprend"
                ],
                "complexity_markers": {
                    "basic": ["droit", "loi", "règle", "obligation"],
                    "medium": ["procédure", "jurisprudence", "contrat"],
                    "advanced": ["doctrine", "casuistique", "herméneutique"],
                    "expert": ["exégèse", "syllogisme juridique", "ratio decidendi"]
                }
            },

            "technical": {
                "key_concepts": {
                    "API": {"category": "interface", "complexity": "medium"},
                    "authentification": {"category": "security", "complexity": "medium"},
                    "architecture": {"category": "design", "complexity": "advanced"},
                    "algorithme": {"category": "computation", "complexity": "advanced"},
                    "base de données": {"category": "storage", "complexity": "medium"},
                    "framework": {"category": "tools", "complexity": "medium"},
                    "microservice": {"category": "architecture", "complexity": "advanced"}
                },
                "technical_terms": [
                    "endpoint", "payload", "token", "header",
                    "GET", "POST", "PUT", "DELETE",
                    "JSON", "XML", "HTTP", "HTTPS",
                    "SQL", "NoSQL", "REST", "GraphQL"
                ],
                "definition_indicators": [
                    "est une", "consiste à", "permet de",
                    "se définit comme", "représente", "constitue"
                ],
                "complexity_markers": {
                    "basic": ["API", "base", "données", "serveur"],
                    "medium": ["architecture", "sécurité", "performance"],
                    "advanced": ["microservice", "orchestration", "scalabilité"],
                    "expert": ["consensus distribué", "CAP theorem", "sharding"]
                }
            },

            "medical": {
                "key_concepts": {
                    "symptôme": {"category": "clinical", "complexity": "basic"},
                    "diagnostic": {"category": "clinical", "complexity": "medium"},
                    "pathologie": {"category": "disease", "complexity": "advanced"},
                    "thérapie": {"category": "treatment", "complexity": "medium"},
                    "prévention": {"category": "health", "complexity": "basic"},
                    "épidémiologie": {"category": "public_health", "complexity": "advanced"}
                },
                "technical_terms": [
                    "anamnèse", "examen clinique", "biopsie",
                    "pharmacocinétique", "posologie", "contre-indication",
                    "effet secondaire", "placebo", "randomisé"
                ],
                "definition_indicators": [
                    "est caractérisé par", "se manifeste par", "consiste en",
                    "est défini comme", "présente", "comprend"
                ],
                "complexity_markers": {
                    "basic": ["symptôme", "traitement", "médicament"],
                    "medium": ["diagnostic", "thérapie", "prévention"],
                    "advanced": ["pathophysiologie", "pharmacologie"],
                    "expert": ["génomique", "protéomique", "immunomodulation"]
                }
            }
        }

        return knowledge_bases.get(domain, {
            "key_concepts": {},
            "technical_terms": [],
            "definition_indicators": ["est", "désigne", "constitue"],
            "complexity_markers": {
                "basic": [], "medium": [], "advanced": [], "expert": []
            }
        })

    def _load_complexity_indicators(self) -> Dict[str, List[str]]:
        """Charge les indicateurs de complexité"""

        return {
            "basic": [
                "simple", "facile", "base", "élémentaire", "fondamental",
                "introduction", "début", "commencer", "premier"
            ],
            "medium": [
                "intermédiaire", "moyen", "standard", "normal", "typique",
                "courant", "habituel", "principal", "général"
            ],
            "advanced": [
                "avancé", "complexe", "sophistiqué", "approfondi", "détaillé",
                "spécialisé", "technique", "expert", "professionnel"
            ],
            "expert": [
                "expert", "spécialiste", "maître", "pointu", "recherche",
                "innovation", "cutting-edge", "state-of-the-art"
            ]
        }

    def _load_definition_patterns(self) -> List[str]:
        """Charge les patterns de définition"""

        base_patterns = [
            r'(.+?)\s+est\s+(.+?)[\.\n]',
            r'(.+?)\s+désigne\s+(.+?)[\.\n]',
            r'(.+?)\s+constitue\s+(.+?)[\.\n]',
            r'(.+?)\s+représente\s+(.+?)[\.\n]',
            r'On\s+appelle\s+(.+?)\s+(.+?)[\.\n]',
            r'(.+?)\s+se\s+définit\s+comme\s+(.+?)[\.\n]',
            r'(.+?)\s+consiste\s+en\s+(.+?)[\.\n]',
            r'(.+?)\s*:\s*(.+?)[\.\n]'  # Pattern avec deux-points
        ]

        # Ajouter patterns spécifiques au domaine
        domain_patterns = self.domain_knowledge.get("definition_indicators", [])
        for indicator in domain_patterns:
            pattern = f'(.+?)\\s+{re.escape(indicator)}\\s+(.+?)[\\\.\\n]'
            base_patterns.append(pattern)

        return base_patterns

    def enrich_chunk_metadata(self, text: str, chunk_id: str,
                            document_id: str, granularity: str,
                            parent_id: Optional[str] = None) -> EnrichedMetadata:
        """
        Enrichit les métadonnées d'un chunk de manière complète

        Args:
            text: Texte du chunk
            chunk_id: Identifiant unique du chunk
            document_id: Identifiant du document
            granularity: Niveau de granularité
            parent_id: ID du chunk parent

        Returns:
            Métadonnées enrichies
        """

        metadata = EnrichedMetadata(
            chunk_id=chunk_id,
            document_id=document_id,
            granularity=granularity,
            parent_id=parent_id,
            language=self.language
        )

        # Enrichissement progressif
        self._extract_entities(text, metadata)
        self._extract_concepts(text, metadata)
        self._extract_keywords(text, metadata)
        self._classify_content_type(text, metadata)
        self._analyze_complexity(text, metadata)
        self._calculate_metrics(text, metadata)
        self._identify_domain_tags(text, metadata)

        return metadata

    def _extract_entities(self, text: str, metadata: EnrichedMetadata):
        """Extrait les entités nommées avec confiance"""

        if not self.nlp:
            return

        # Cache check
        text_hash = hash(text)
        if text_hash in self._entity_cache:
            metadata.entities = self._entity_cache[text_hash]
            return

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Calcul de confiance basé sur le contexte
            confidence = self._calculate_entity_confidence(ent, doc)

            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": confidence,
                "description": spacy.explain(ent.label_) if hasattr(spacy, 'explain') else ent.label_
            }

            entities.append(entity_info)

        metadata.entities = entities
        metadata.named_entities_count = len(entities)

        # Cache result
        self._entity_cache[text_hash] = entities

    def _calculate_entity_confidence(self, entity, doc) -> float:
        """Calcule la confiance d'une entité basée sur le contexte"""

        base_confidence = 0.7  # Confiance de base spaCy

        # Bonus si l'entité est répétée
        entity_count = sum(1 for ent in doc.ents if ent.text.lower() == entity.text.lower())
        repetition_bonus = min(entity_count * 0.1, 0.3)

        # Bonus si l'entité est dans le vocabulaire du domaine
        domain_bonus = 0.0
        if entity.text.lower() in [term.lower() for term in self.domain_knowledge.get("technical_terms", [])]:
            domain_bonus = 0.2

        # Pénalité si l'entité est trop courte
        length_penalty = 0.0
        if len(entity.text) < 3:
            length_penalty = -0.2

        final_confidence = min(1.0, max(0.0, base_confidence + repetition_bonus + domain_bonus + length_penalty))
        return round(final_confidence, 2)

    def _extract_concepts(self, text: str, metadata: EnrichedMetadata):
        """Extrait les concepts avec pertinence contextuelle"""

        text_lower = text.lower()
        concepts = []

        # Concepts du domaine
        domain_concepts = self.domain_knowledge.get("key_concepts", {})

        for concept, info in domain_concepts.items():
            if concept.lower() in text_lower:
                # Calcul de pertinence contextuelle
                relevance = self._calculate_concept_relevance(concept, text, info)

                # Extraction du contexte autour du concept
                context = self._extract_concept_context(concept, text)

                concept_info = {
                    "concept": concept,
                    "category": info.get("category", "general"),
                    "complexity": info.get("complexity", "medium"),
                    "relevance": relevance,
                    "context": context,
                    "occurrences": text_lower.count(concept.lower())
                }

                concepts.append(concept_info)

        # Tri par pertinence
        concepts.sort(key=lambda x: x["relevance"], reverse=True)
        metadata.concepts = concepts[:10]  # Limiter à 10 concepts principaux

    def _calculate_concept_relevance(self, concept: str, text: str, concept_info: Dict) -> float:
        """Calcule la pertinence d'un concept dans le texte"""

        text_lower = text.lower()
        concept_lower = concept.lower()

        # Fréquence du concept
        frequency = text_lower.count(concept_lower)
        frequency_score = min(frequency / 10, 1.0)  # Normaliser sur 10 occurrences max

        # Position du concept (début = plus important)
        first_position = text_lower.find(concept_lower)
        position_score = 1 - (first_position / len(text)) if first_position != -1 else 0

        # Complexité du concept (plus complexe = plus spécifique)
        complexity_levels = {"basic": 0.3, "medium": 0.6, "advanced": 0.8, "expert": 1.0}
        complexity_score = complexity_levels.get(concept_info.get("complexity", "medium"), 0.5)

        # Score final pondéré
        relevance = (frequency_score * 0.4 + position_score * 0.3 + complexity_score * 0.3)

        return round(relevance, 3)

    def _extract_concept_context(self, concept: str, text: str, window: int = 100) -> str:
        """Extrait le contexte autour d'un concept"""

        concept_lower = concept.lower()
        text_lower = text.lower()

        position = text_lower.find(concept_lower)
        if position == -1:
            return ""

        start = max(0, position - window)
        end = min(len(text), position + len(concept) + window)

        context = text[start:end].strip()

        # Nettoyer le contexte (éviter les coupures de mots)
        if start > 0:
            space_pos = context.find(' ')
            if space_pos != -1:
                context = context[space_pos+1:]

        if end < len(text):
            space_pos = context.rfind(' ')
            if space_pos != -1:
                context = context[:space_pos]

        return context

    def _extract_keywords(self, text: str, metadata: EnrichedMetadata):
        """Extrait les mots-clés avec importance"""

        if not self.nlp:
            # Fallback simple
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
            word_freq = Counter(words)

            keywords = []
            for word, freq in word_freq.most_common(10):
                keywords.append({
                    "keyword": word,
                    "frequency": freq,
                    "importance": min(freq / len(words) * 10, 1.0),
                    "pos": "UNKNOWN"
                })

            metadata.keywords = keywords
            return

        doc = self.nlp(text)

        # Extraire tokens significatifs
        significant_tokens = []
        for token in doc:
            if (token.pos_ in ["NOUN", "ADJ", "VERB"] and
                not token.is_stop and
                not token.is_punct and
                len(token.text) > 3 and
                token.is_alpha):

                significant_tokens.append({
                    "text": token.lemma_.lower(),
                    "pos": token.pos_,
                    "is_technical": self._is_technical_term(token.text)
                })

        # Compter fréquences
        token_freq = Counter(token["text"] for token in significant_tokens)

        # Calculer importance
        keywords = []
        for token in significant_tokens:
            if token["text"] in token_freq:
                freq = token_freq[token["text"]]

                # Score d'importance
                importance = freq / len(significant_tokens)

                # Bonus pour termes techniques
                if token["is_technical"]:
                    importance *= 1.5

                # Bonus pour noms (plus informatifs)
                if token["pos"] == "NOUN":
                    importance *= 1.2

                keywords.append({
                    "keyword": token["text"],
                    "frequency": freq,
                    "importance": min(importance, 1.0),
                    "pos": token["pos"],
                    "is_technical": token["is_technical"]
                })

                # Supprimer pour éviter doublons
                del token_freq[token["text"]]

        # Trier par importance et limiter
        keywords.sort(key=lambda x: x["importance"], reverse=True)
        metadata.keywords = keywords[:15]

        # Compter termes techniques
        metadata.technical_terms_count = sum(1 for kw in keywords if kw["is_technical"])

    def _is_technical_term(self, term: str) -> bool:
        """Détermine si un terme est technique pour le domaine"""

        technical_terms = self.domain_knowledge.get("technical_terms", [])
        return term.lower() in [t.lower() for t in technical_terms]

    def _classify_content_type(self, text: str, metadata: EnrichedMetadata):
        """Classifie le type de contenu du chunk"""

        text_lower = text.lower()

        # Patterns de classification
        classification_patterns = {
            "definition": [
                r'\b(?:est|désigne|constitue|représente)\b',
                r'\bdéfinition\b',
                r'\bon appelle\b',
                r'\bse définit comme\b'
            ],
            "example": [
                r'\bpar exemple\b',
                r'\bnotamment\b',
                r'\btelle? que\b',
                r'\bcomme\b.*\b(?:par exemple|notamment)\b'
            ],
            "procedure": [
                r'\bétapes?\b',
                r'\bprocédure\b',
                r'\bméthode\b',
                r'\bdémarche\b',
                r'\bprocessus\b',
                r'\bpremièrement\b|\bdeuxièmement\b|\btroisièmement\b'
            ],
            "summary": [
                r'\ben résumé\b',
                r'\ben conclusion\b',
                r'\bpour conclure\b',
                r'\bainsi\b.*\bpour résumer\b'
            ]
        }

        # Score par type
        type_scores = {}

        for content_type, patterns in classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches

            # Normaliser par longueur du texte
            normalized_score = score / max(1, len(text.split()) / 100)
            type_scores[content_type] = normalized_score

        # Déterminer le type principal
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0.1:  # Seuil minimum
                metadata.content_type = best_type
            else:
                metadata.content_type = "content"
        else:
            metadata.content_type = "content"

    def _analyze_complexity(self, text: str, metadata: EnrichedMetadata):
        """Analyse le niveau de complexité du contenu"""

        text_lower = text.lower()
        complexity_scores = {"basic": 0, "medium": 0, "advanced": 0, "expert": 0}

        # Analyse basée sur les indicateurs de complexité
        for level, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if indicator.lower() in text_lower:
                    complexity_scores[level] += 1

        # Analyse basée sur les concepts du domaine
        domain_complexity = self.domain_knowledge.get("complexity_markers", {})
        for level, terms in domain_complexity.items():
            for term in terms:
                if term.lower() in text_lower:
                    complexity_scores[level] += 2  # Poids plus élevé pour les termes domaine

        # Analyse structurelle
        if self.nlp:
            doc = self.nlp(text)

            # Longueur moyenne des phrases
            sentences = list(doc.sents)
            if sentences:
                avg_sentence_length = sum(len(sent.text.split()) for sent in sentences) / len(sentences)

                if avg_sentence_length > 25:
                    complexity_scores["advanced"] += 1
                elif avg_sentence_length > 15:
                    complexity_scores["medium"] += 1
                else:
                    complexity_scores["basic"] += 1

            # Complexité vocabulaire
            complex_tokens = sum(1 for token in doc if len(token.text) > 10 and token.is_alpha)
            vocab_complexity = complex_tokens / max(1, len([t for t in doc if t.is_alpha]))

            if vocab_complexity > 0.3:
                complexity_scores["expert"] += 1
            elif vocab_complexity > 0.2:
                complexity_scores["advanced"] += 1
            elif vocab_complexity > 0.1:
                complexity_scores["medium"] += 1
            else:
                complexity_scores["basic"] += 1

        # Déterminer le niveau final
        if complexity_scores:
            # Pondération progressive
            weighted_scores = {
                "basic": complexity_scores["basic"] * 1,
                "medium": complexity_scores["medium"] * 2,
                "advanced": complexity_scores["advanced"] * 3,
                "expert": complexity_scores["expert"] * 4
            }

            total_score = sum(weighted_scores.values())
            if total_score == 0:
                metadata.complexity_level = "medium"
            else:
                # Calculer le niveau moyen pondéré
                if weighted_scores["expert"] / total_score > 0.3:
                    metadata.complexity_level = "expert"
                elif weighted_scores["advanced"] / total_score > 0.4:
                    metadata.complexity_level = "advanced"
                elif weighted_scores["basic"] / total_score > 0.6:
                    metadata.complexity_level = "basic"
                else:
                    metadata.complexity_level = "medium"
        else:
            metadata.complexity_level = "medium"

    def _calculate_metrics(self, text: str, metadata: EnrichedMetadata):
        """Calcule les métriques quantitatives"""

        # Densité sémantique
        if self.nlp:
            doc = self.nlp(text)
            total_tokens = len([t for t in doc if t.is_alpha])
            significant_tokens = len([t for t in doc if t.pos_ in ["NOUN", "VERB", "ADJ"] and not t.is_stop])

            metadata.semantic_density = significant_tokens / max(1, total_tokens)
        else:
            # Approximation sans NLP
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text)
            stop_words = ["le", "la", "les", "de", "du", "des", "et", "ou", "à", "un", "une"]
            significant_words = [w for w in words if w.lower() not in stop_words and len(w) > 3]

            metadata.semantic_density = len(significant_words) / max(1, len(words))

        # Pertinence domaine (basée sur les concepts identifiés)
        domain_relevance = 0.0
        if metadata.concepts:
            relevance_sum = sum(concept["relevance"] for concept in metadata.concepts)
            domain_relevance = relevance_sum / len(metadata.concepts)

        metadata.domain_relevance = domain_relevance

        # Score de lisibilité (approximation Flesch-Kincaid)
        metadata.readability_score = self._calculate_readability(text)

        # Densité d'information (basée sur entités + concepts)
        info_elements = len(metadata.entities) + len(metadata.concepts)
        text_length = len(text.split())
        metadata.information_density = info_elements / max(1, text_length / 100)  # Par 100 mots

    def _calculate_readability(self, text: str) -> float:
        """Calcule un score de lisibilité approximatif"""

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text)

        if not words:
            return 0.5

        # Statistiques de base
        avg_sentence_length = len(words) / len(sentences)

        # Approximation syllables (pour le français)
        syllable_count = 0
        for word in words:
            # Estimation simplifiée: voyelles
            vowels = sum(1 for char in word.lower() if char in 'aeiouyàéèêëîïôöùûüÿç')
            syllable_count += max(1, vowels)

        avg_syllables_per_word = syllable_count / len(words)

        # Score Flesch adapté (inversé pour que 1.0 = très lisible)
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Normaliser entre 0 et 1
        normalized_score = max(0, min(1, flesch_score / 100))

        return round(normalized_score, 3)

    def _identify_domain_tags(self, text: str, metadata: EnrichedMetadata):
        """Identifie les tags de domaine pour le chunk"""

        text_lower = text.lower()
        tags = set()

        # Tags basés sur les concepts
        for concept in metadata.concepts:
            category = concept.get("category", "")
            if category:
                tags.add(category)

        # Tags basés sur les entités
        for entity in metadata.entities:
            label = entity.get("label", "")
            if label in ["ORG", "PRODUCT", "EVENT"]:
                tags.add(label.lower())

        # Tags spécifiques au domaine
        domain_tags_mapping = {
            "legal": {
                "droit civil": ["civil_law"],
                "droit pénal": ["criminal_law"],
                "procédure": ["procedure"],
                "contrat": ["contract_law"],
                "tribunal": ["judiciary"]
            },
            "technical": {
                "API": ["api", "interface"],
                "base de données": ["database", "storage"],
                "sécurité": ["security"],
                "architecture": ["architecture", "design"],
                "performance": ["performance", "optimization"]
            },
            "medical": {
                "diagnostic": ["diagnosis", "clinical"],
                "traitement": ["treatment", "therapy"],
                "prévention": ["prevention", "health"],
                "symptôme": ["symptoms", "clinical"]
            }
        }

        domain_mapping = domain_tags_mapping.get(self.domain, {})
        for term, term_tags in domain_mapping.items():
            if term.lower() in text_lower:
                tags.update(term_tags)

        metadata.domain_tags = list(tags)

    def batch_enrich_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[EnrichedMetadata]:
        """
        Enrichit un lot de chunks de manière optimisée

        Args:
            chunks_data: Liste de dict avec keys: text, chunk_id, document_id, granularity, parent_id

        Returns:
            Liste de métadonnées enrichies
        """

        enriched_metadata = []

        for chunk_data in chunks_data:
            metadata = self.enrich_chunk_metadata(
                text=chunk_data["text"],
                chunk_id=chunk_data["chunk_id"],
                document_id=chunk_data["document_id"],
                granularity=chunk_data["granularity"],
                parent_id=chunk_data.get("parent_id")
            )

            enriched_metadata.append(metadata)

        # Post-traitement pour relations entre chunks
        self._identify_chunk_relationships(enriched_metadata)

        return enriched_metadata

    def _identify_chunk_relationships(self, metadata_list: List[EnrichedMetadata]):
        """Identifie les relations entre chunks"""

        # Calculer similarité conceptuelle entre chunks
        for i, metadata1 in enumerate(metadata_list):
            for j, metadata2 in enumerate(metadata_list[i+1:], i+1):
                similarity = self._calculate_conceptual_similarity(metadata1, metadata2)

                if similarity > 0.3:  # Seuil de similarité
                    metadata1.related_chunks.append(metadata2.chunk_id)
                    metadata2.related_chunks.append(metadata1.chunk_id)

        # Identifier prérequis/suites logiques basés sur la complexité
        for metadata in metadata_list:
            if metadata.complexity_level in ["advanced", "expert"]:
                # Chercher des chunks plus simples qui pourraient être des prérequis
                for other_metadata in metadata_list:
                    if (other_metadata.complexity_level in ["basic", "medium"] and
                        self._shares_concepts(metadata, other_metadata)):
                        metadata.prerequisite_concepts.extend([
                            concept["concept"] for concept in other_metadata.concepts[:3]
                        ])

    def _calculate_conceptual_similarity(self, metadata1: EnrichedMetadata, metadata2: EnrichedMetadata) -> float:
        """Calcule la similarité conceptuelle entre deux chunks"""

        # Similarité basée sur les concepts partagés
        concepts1 = set(concept["concept"] for concept in metadata1.concepts)
        concepts2 = set(concept["concept"] for concept in metadata2.concepts)

        if not concepts1 or not concepts2:
            return 0.0

        shared_concepts = concepts1.intersection(concepts2)
        total_concepts = concepts1.union(concepts2)

        concept_similarity = len(shared_concepts) / len(total_concepts)

        # Similarité basée sur les mots-clés
        keywords1 = set(kw["keyword"] for kw in metadata1.keywords)
        keywords2 = set(kw["keyword"] for kw in metadata2.keywords)

        if keywords1 and keywords2:
            shared_keywords = keywords1.intersection(keywords2)
            total_keywords = keywords1.union(keywords2)
            keyword_similarity = len(shared_keywords) / len(total_keywords)
        else:
            keyword_similarity = 0.0

        # Score final pondéré
        return (concept_similarity * 0.7 + keyword_similarity * 0.3)

    def _shares_concepts(self, metadata1: EnrichedMetadata, metadata2: EnrichedMetadata) -> bool:
        """Vérifie si deux chunks partagent des concepts"""

        concepts1 = set(concept["concept"] for concept in metadata1.concepts)
        concepts2 = set(concept["concept"] for concept in metadata2.concepts)

        return bool(concepts1.intersection(concepts2))

    def save_enriched_metadata(self, metadata_list: List[EnrichedMetadata], output_path: str):
        """Sauvegarde les métadonnées enrichies"""

        data = [metadata.to_dict() for metadata in metadata_list]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Enriched metadata saved to {output_path}")

    def load_enriched_metadata(self, input_path: str) -> List[EnrichedMetadata]:
        """Charge les métadonnées enrichies"""

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata_list = [EnrichedMetadata.from_dict(item) for item in data]

        logger.info(f"Loaded {len(metadata_list)} enriched metadata from {input_path}")

        return metadata_list


# Fonctions d'intégration pour RAG Chunk Lab
def enrich_hierarchical_chunks(hierarchy: Dict[str, List[Any]],
                              document_id: str,
                              domain: str = "general") -> Dict[str, List[Any]]:
    """
    Enrichit une hiérarchie de chunks avec des métadonnées avancées

    Args:
        hierarchy: Hiérarchie de chunks du HierarchicalChunker
        document_id: ID du document
        domain: Domaine spécialisé

    Returns:
        Hiérarchie enrichie avec métadonnées avancées
    """

    enricher = MetadataEnricher(domain=domain)
    enriched_hierarchy = {}

    for granularity, chunks in hierarchy.items():
        enriched_chunks = []

        for chunk in chunks:
            # Gérer à la fois les dictionnaires et les HierarchicalChunk dataclasses
            if hasattr(chunk, 'text'):  # HierarchicalChunk dataclass
                text = chunk.text
                chunk_id = chunk.metadata.chunk_id
                parent_id = chunk.metadata.parent_id

                # Enrichir les métadonnées
                enriched_metadata = enricher.enrich_chunk_metadata(
                    text=text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    granularity=granularity,
                    parent_id=parent_id
                )

                # Ajouter les métadonnées enrichies au chunk
                chunk.metadata.enriched_metadata = enriched_metadata
                enriched_chunks.append(chunk)

            else:  # Format dictionnaire (backward compatibility)
                # Enrichir les métadonnées
                enriched_metadata = enricher.enrich_chunk_metadata(
                    text=chunk["text"],
                    chunk_id=chunk["metadata"]["chunk_id"],
                    document_id=document_id,
                    granularity=granularity,
                    parent_id=chunk["metadata"].get("parent_id")
                )

                # Fusionner avec les métadonnées existantes
                chunk["enriched_metadata"] = enriched_metadata.to_dict()
                enriched_chunks.append(chunk)

        enriched_hierarchy[granularity] = enriched_chunks

    return enriched_hierarchy


def create_metadata_index(enriched_hierarchy: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Crée un index de métadonnées pour recherche optimisée

    Returns:
        Index structuré pour recherche rapide
    """

    index = {
        "concepts": defaultdict(list),
        "entities": defaultdict(list),
        "keywords": defaultdict(list),
        "content_types": defaultdict(list),
        "complexity_levels": defaultdict(list),
        "domain_tags": defaultdict(list)
    }

    for granularity, chunks in enriched_hierarchy.items():
        for chunk in chunks:
            chunk_id = chunk["metadata"]["chunk_id"]
            enriched_meta = chunk.get("enriched_metadata", {})

            # Indexer concepts
            for concept in enriched_meta.get("concepts", []):
                index["concepts"][concept["concept"]].append(chunk_id)

            # Indexer entités
            for entity in enriched_meta.get("entities", []):
                index["entities"][entity["text"]].append(chunk_id)

            # Indexer mots-clés
            for keyword in enriched_meta.get("keywords", []):
                index["keywords"][keyword["keyword"]].append(chunk_id)

            # Indexer types de contenu
            content_type = enriched_meta.get("content_type", "content")
            index["content_types"][content_type].append(chunk_id)

            # Indexer niveaux de complexité
            complexity = enriched_meta.get("complexity_level", "medium")
            index["complexity_levels"][complexity].append(chunk_id)

            # Indexer tags domaine
            for tag in enriched_meta.get("domain_tags", []):
                index["domain_tags"][tag].append(chunk_id)

    return dict(index)  # Convertir defaultdict en dict normal