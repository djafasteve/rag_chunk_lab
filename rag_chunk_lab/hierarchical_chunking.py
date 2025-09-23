# hierarchical_chunking.py
"""
Chunking hiérarchique multi-granularité pour optimiser les réponses aux requêtes vagues
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import spacy
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Métadonnées enrichies pour un chunk"""
    chunk_id: str
    granularity: str  # "paragraph", "sentence", "concept", "summary"
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    concepts: List[str] = None
    entities: List[str] = None
    keywords: List[str] = None
    semantic_density: float = 0.0
    domain_relevance: float = 0.0
    chunk_type: str = "content"  # "content", "definition", "example", "procedure"
    enriched_metadata: Optional[Any] = None

    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.concepts is None:
            self.concepts = []
        if self.entities is None:
            self.entities = []
        if self.keywords is None:
            self.keywords = []


@dataclass
class HierarchicalChunk:
    """Chunk avec structure hiérarchique"""
    text: str
    metadata: ChunkMetadata
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "text": self.text,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "granularity": self.metadata.granularity,
                "parent_id": self.metadata.parent_id,
                "children_ids": self.metadata.children_ids,
                "concepts": self.metadata.concepts,
                "entities": self.metadata.entities,
                "keywords": self.metadata.keywords,
                "semantic_density": self.metadata.semantic_density,
                "domain_relevance": self.metadata.domain_relevance,
                "chunk_type": self.metadata.chunk_type
            },
            "start_char": self.start_char,
            "end_char": self.end_char
        }


class HierarchicalChunker:
    """Chunker hiérarchique multi-granularité"""

    def __init__(self, domain: str = "general", language: str = "fr"):
        self.domain = domain
        self.language = language

        # Configuration NLP
        try:
            self.nlp = spacy.load("fr_core_news_sm") if language == "fr" else spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(f"SpaCy model not found for {language}. Install with: python -m spacy download fr_core_news_sm")
            self.nlp = None

        # Patterns domaine-spécifiques
        self.domain_patterns = self._load_domain_patterns(domain)

        # Configuration granularité
        self.granularity_config = {
            "document": {"max_tokens": 2000, "overlap": 0},
            "section": {"max_tokens": 800, "overlap": 100},
            "paragraph": {"max_tokens": 300, "overlap": 50},
            "sentence": {"max_tokens": 100, "overlap": 10},
            "concept": {"max_tokens": 150, "overlap": 25}
        }

    def _load_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """Charge les patterns spécifiques au domaine"""

        patterns = {
            "legal": {
                "definitions": [
                    r"(?:art\.|article)\s*(\d+)",
                    r"selon\s+(?:la\s+)?loi\s+.*",
                    r"(?:le\s+)?(?:code\s+)?(?:civil|pénal|commercial)",
                    r"jurisprudence\s+.*"
                ],
                "procedures": [
                    r"procédure\s+de\s+.*",
                    r"étapes?\s+(?:pour|de).*",
                    r"délai\s+de\s+.*"
                ],
                "entities": ["tribunal", "avocat", "juge", "procureur", "partie"],
                "concepts": ["droit", "obligation", "responsabilité", "contrat"]
            },
            "technical": {
                "definitions": [
                    r"(?:API|interface|protocole)\s+.*",
                    r"(?:fonction|méthode|classe)\s+.*",
                    r"algorithme\s+.*"
                ],
                "procedures": [
                    r"configuration\s+.*",
                    r"installation\s+.*",
                    r"étapes?\s+d'implémentation"
                ],
                "entities": ["serveur", "client", "base", "système"],
                "concepts": ["architecture", "sécurité", "performance", "scalabilité"]
            },
            "medical": {
                "definitions": [
                    r"symptôme\s+.*",
                    r"pathologie\s+.*",
                    r"traitement\s+.*"
                ],
                "procedures": [
                    r"diagnostic\s+.*",
                    r"thérapie\s+.*",
                    r"protocole\s+.*"
                ],
                "entities": ["patient", "médecin", "hôpital", "médicament"],
                "concepts": ["santé", "prévention", "soin", "guérison"]
            }
        }

        return patterns.get(domain, {
            "definitions": [r"définition\s+.*", r"(?:est|désigne)\s+.*"],
            "procedures": [r"procédure\s+.*", r"étapes?\s+.*"],
            "entities": [],
            "concepts": []
        })

    def create_hierarchical_chunks(self, document: str, doc_id: str) -> Dict[str, List[HierarchicalChunk]]:
        """
        Crée des chunks hiérarchiques multi-granularité

        Returns:
            Dict avec les chunks par niveau de granularité
        """

        hierarchy = {
            "document": [],
            "section": [],
            "paragraph": [],
            "sentence": [],
            "concept": [],
            "summary": []
        }

        # 1. Niveau document (vue d'ensemble)
        doc_chunk = self._create_document_chunk(document, doc_id)
        hierarchy["document"].append(doc_chunk)

        # 2. Niveau sections (structure majeure)
        sections = self._extract_sections(document, doc_id, doc_chunk.metadata.chunk_id)
        hierarchy["section"].extend(sections)

        # 3. Niveau paragraphes (contexte moyen)
        paragraphs = []
        for section in sections:
            para_chunks = self._extract_paragraphs(section.text, doc_id, section.metadata.chunk_id)
            paragraphs.extend(para_chunks)
        hierarchy["paragraph"].extend(paragraphs)

        # 4. Niveau phrases (précision fine)
        sentences = []
        for paragraph in paragraphs:
            sent_chunks = self._extract_sentences(paragraph.text, doc_id, paragraph.metadata.chunk_id)
            sentences.extend(sent_chunks)
        hierarchy["sentence"].extend(sentences)

        # 5. Niveau concepts (entités et définitions)
        concepts = self._extract_concepts(document, doc_id)
        hierarchy["concept"].extend(concepts)

        # 6. Niveau résumés (synthèse automatique)
        summaries = self._create_summaries(hierarchy, doc_id)
        hierarchy["summary"].extend(summaries)

        # Enrichir les métadonnées
        self._enrich_metadata(hierarchy)

        return hierarchy

    def _create_document_chunk(self, document: str, doc_id: str) -> HierarchicalChunk:
        """Crée le chunk document complet"""

        chunk_id = f"{doc_id}_doc_0"

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            granularity="document",
            chunk_type="content"
        )

        return HierarchicalChunk(
            text=document,
            metadata=metadata,
            start_char=0,
            end_char=len(document)
        )

    def _extract_sections(self, document: str, doc_id: str, parent_id: str) -> List[HierarchicalChunk]:
        """Extrait les sections du document"""

        sections = []

        # Patterns de sections
        section_patterns = [
            r'^\s*#{1,3}\s+.*$',  # Markdown headers
            r'^\s*\d+\.\s+.*$',   # Numérotation
            r'^\s*[IVX]+\.\s+.*$',  # Chiffres romains
            r'^\s*[A-Z][^\n]*:?\s*$'  # Titres en majuscules
        ]

        lines = document.split('\n')
        current_section = ""
        section_start = 0
        section_count = 0

        for i, line in enumerate(lines):
            is_section_header = any(re.match(pattern, line, re.MULTILINE) for pattern in section_patterns)

            if is_section_header and current_section.strip():
                # Sauvegarder la section précédente
                chunk_id = f"{doc_id}_sec_{section_count}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    granularity="section",
                    parent_id=parent_id,
                    chunk_type="content"
                )

                sections.append(HierarchicalChunk(
                    text=current_section.strip(),
                    metadata=metadata,
                    start_char=section_start,
                    end_char=section_start + len(current_section)
                ))

                section_count += 1
                current_section = line + '\n'
                section_start = sum(len(l) + 1 for l in lines[:i])

            else:
                current_section += line + '\n'

        # Dernière section
        if current_section.strip():
            chunk_id = f"{doc_id}_sec_{section_count}"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                granularity="section",
                parent_id=parent_id,
                chunk_type="content"
            )

            sections.append(HierarchicalChunk(
                text=current_section.strip(),
                metadata=metadata,
                start_char=section_start,
                end_char=len(document)
            ))

        return sections

    def _extract_paragraphs(self, text: str, doc_id: str, parent_id: str) -> List[HierarchicalChunk]:
        """Extrait les paragraphes"""

        paragraphs = []
        para_texts = [p.strip() for p in text.split('\n\n') if p.strip()]

        start_pos = 0
        for i, para_text in enumerate(para_texts):
            if len(para_text) > 50:  # Filtrer les paragraphes trop courts
                chunk_id = f"{parent_id}_para_{i}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    granularity="paragraph",
                    parent_id=parent_id,
                    chunk_type="content"
                )

                paragraphs.append(HierarchicalChunk(
                    text=para_text,
                    metadata=metadata,
                    start_char=start_pos,
                    end_char=start_pos + len(para_text)
                ))

            start_pos += len(para_text) + 2  # +2 pour \n\n

        return paragraphs

    def _extract_sentences(self, text: str, doc_id: str, parent_id: str) -> List[HierarchicalChunk]:
        """Extrait les phrases"""

        sentences = []

        if self.nlp:
            doc = self.nlp(text)

            for i, sent in enumerate(doc.sents):
                if len(sent.text.strip()) > 20:  # Filtrer phrases trop courtes
                    chunk_id = f"{parent_id}_sent_{i}"

                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        granularity="sentence",
                        parent_id=parent_id,
                        chunk_type="content"
                    )

                    sentences.append(HierarchicalChunk(
                        text=sent.text.strip(),
                        metadata=metadata,
                        start_char=sent.start_char,
                        end_char=sent.end_char
                    ))
        else:
            # Fallback sans spaCy
            sent_texts = re.split(r'[.!?]+', text)
            start_pos = 0

            for i, sent_text in enumerate(sent_texts):
                sent_text = sent_text.strip()
                if len(sent_text) > 20:
                    chunk_id = f"{parent_id}_sent_{i}"

                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        granularity="sentence",
                        parent_id=parent_id,
                        chunk_type="content"
                    )

                    sentences.append(HierarchicalChunk(
                        text=sent_text,
                        metadata=metadata,
                        start_char=start_pos,
                        end_char=start_pos + len(sent_text)
                    ))

                start_pos += len(sent_text) + 1

        return sentences

    def _extract_concepts(self, document: str, doc_id: str) -> List[HierarchicalChunk]:
        """Extrait les concepts et définitions"""

        concepts = []
        concept_count = 0

        # Définitions explicites
        definition_patterns = self.domain_patterns.get("definitions", [])

        for pattern in definition_patterns:
            matches = re.finditer(pattern, document, re.IGNORECASE | re.MULTILINE)

            for match in matches:
                # Extraire le contexte autour de la définition
                start = max(0, match.start() - 100)
                end = min(len(document), match.end() + 200)
                context = document[start:end]

                chunk_id = f"{doc_id}_concept_{concept_count}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    granularity="concept",
                    chunk_type="definition"
                )

                concepts.append(HierarchicalChunk(
                    text=context.strip(),
                    metadata=metadata,
                    start_char=start,
                    end_char=end
                ))

                concept_count += 1

        # Entités nommées avec NLP
        if self.nlp:
            doc = self.nlp(document)

            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                    # Extraire contexte autour de l'entité
                    start = max(0, ent.start_char - 150)
                    end = min(len(document), ent.end_char + 150)
                    context = document[start:end]

                    chunk_id = f"{doc_id}_entity_{concept_count}"

                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        granularity="concept",
                        chunk_type="entity",
                        entities=[ent.text]
                    )

                    concepts.append(HierarchicalChunk(
                        text=context.strip(),
                        metadata=metadata,
                        start_char=start,
                        end_char=end
                    ))

                    concept_count += 1

        return concepts

    def _create_summaries(self, hierarchy: Dict[str, List[HierarchicalChunk]], doc_id: str) -> List[HierarchicalChunk]:
        """Crée des résumés automatiques"""

        summaries = []

        # Résumé par section
        for i, section in enumerate(hierarchy["section"]):
            summary_text = self._create_extractive_summary(section.text, max_sentences=3)

            chunk_id = f"{doc_id}_summary_sec_{i}"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                granularity="summary",
                parent_id=section.metadata.chunk_id,
                chunk_type="summary"
            )

            summaries.append(HierarchicalChunk(
                text=summary_text,
                metadata=metadata
            ))

        # Résumé global du document
        if hierarchy["section"]:
            all_summaries = " ".join([s.text for s in summaries])
            global_summary = self._create_extractive_summary(all_summaries, max_sentences=5)

            chunk_id = f"{doc_id}_summary_global"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                granularity="summary",
                chunk_type="summary"
            )

            summaries.append(HierarchicalChunk(
                text=global_summary,
                metadata=metadata
            ))

        return summaries

    def _create_extractive_summary(self, text: str, max_sentences: int = 3) -> str:
        """Crée un résumé extractif simple"""

        if not self.nlp:
            # Fallback: prendre les premières phrases
            sentences = re.split(r'[.!?]+', text)
            return '. '.join(sentences[:max_sentences]).strip() + '.'

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if len(sentences) <= max_sentences:
            return text

        # Score simple basé sur la position et la longueur
        scored_sentences = []

        for i, sent in enumerate(sentences):
            # Score position (début du texte = plus important)
            position_score = 1 - (i / len(sentences))

            # Score longueur (phrases moyennes privilégiées)
            length_score = min(len(sent.text) / 100, 1)

            # Score mots-clés domaine
            keyword_score = 0
            domain_keywords = self.domain_patterns.get("concepts", [])
            for keyword in domain_keywords:
                if keyword.lower() in sent.text.lower():
                    keyword_score += 0.1

            total_score = position_score + length_score + keyword_score
            scored_sentences.append((sent.text.strip(), total_score))

        # Trier et prendre les meilleures
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentences = [sent for sent, _ in scored_sentences[:max_sentences]]

        return ' '.join(best_sentences)

    def _enrich_metadata(self, hierarchy: Dict[str, List[HierarchicalChunk]]):
        """Enrichit les métadonnées de tous les chunks"""

        for granularity, chunks in hierarchy.items():
            for chunk in chunks:
                # Extraire concepts et entités
                if self.nlp:
                    doc = self.nlp(chunk.text)

                    # Entités nommées
                    chunk.metadata.entities = [ent.text for ent in doc.ents]

                    # Mots-clés (noms et adjectifs importants)
                    keywords = [token.lemma_ for token in doc
                              if token.pos_ in ["NOUN", "ADJ"] and len(token.text) > 3]
                    chunk.metadata.keywords = list(set(keywords))[:10]

                # Concepts domaine-spécifiques
                domain_concepts = self.domain_patterns.get("concepts", [])
                chunk.metadata.concepts = [concept for concept in domain_concepts
                                         if concept.lower() in chunk.text.lower()]

                # Densité sémantique (approximation)
                chunk.metadata.semantic_density = self._calculate_semantic_density(chunk.text)

                # Pertinence domaine
                chunk.metadata.domain_relevance = self._calculate_domain_relevance(chunk.text)

    def _calculate_semantic_density(self, text: str) -> float:
        """Calcule la densité sémantique approximative"""

        if not self.nlp:
            return 0.5  # Valeur par défaut

        doc = self.nlp(text)

        if len(doc) == 0:
            return 0.0

        # Ratio de mots significatifs (noms, verbes, adjectifs)
        significant_tokens = [token for token in doc
                            if token.pos_ in ["NOUN", "VERB", "ADJ"] and not token.is_stop]

        density = len(significant_tokens) / len(doc)
        return min(density, 1.0)

    def _calculate_domain_relevance(self, text: str) -> float:
        """Calcule la pertinence pour le domaine"""

        text_lower = text.lower()

        # Compter les occurrences de mots-clés domaine
        domain_keywords = []
        for category in ["concepts", "entities"]:
            domain_keywords.extend(self.domain_patterns.get(category, []))

        if not domain_keywords:
            return 0.5  # Neutre si pas de mots-clés domaine

        matches = sum(1 for keyword in domain_keywords if keyword.lower() in text_lower)

        # Normaliser par la longueur du texte
        relevance = matches / max(1, len(text.split()) / 100)  # Par 100 mots

        return min(relevance, 1.0)

    def adaptive_chunk_selection(self,
                                query: str,
                                hierarchy: Dict[str, List[HierarchicalChunk]],
                                vagueness_score: float = 0.5) -> List[HierarchicalChunk]:
        """
        Sélectionne adaptivement les chunks selon la vague de la requête

        Args:
            query: Requête utilisateur
            hierarchy: Hiérarchie de chunks
            vagueness_score: Score de vague de la requête (0-1)

        Returns:
            Liste des chunks les plus appropriés
        """

        selected_chunks = []

        if vagueness_score >= 0.7:
            # Très vague: privilégier résumés et concepts
            selected_chunks.extend(hierarchy.get("summary", []))
            selected_chunks.extend(hierarchy.get("concept", [])[:5])
            selected_chunks.extend(hierarchy.get("paragraph", [])[:3])

        elif vagueness_score >= 0.4:
            # Moyennement vague: mix paragraphes et phrases
            selected_chunks.extend(hierarchy.get("paragraph", [])[:5])
            selected_chunks.extend(hierarchy.get("sentence", [])[:10])
            selected_chunks.extend(hierarchy.get("concept", [])[:3])

        else:
            # Précise: privilégier phrases et concepts spécifiques
            selected_chunks.extend(hierarchy.get("sentence", [])[:15])
            selected_chunks.extend(hierarchy.get("concept", [])[:5])
            selected_chunks.extend(hierarchy.get("paragraph", [])[:2])

        # Filtrer par pertinence de contenu
        query_lower = query.lower()
        scored_chunks = []

        for chunk in selected_chunks:
            # Score de correspondance textuelle simple
            text_score = sum(1 for word in query_lower.split()
                           if word in chunk.text.lower()) / max(1, len(query.split()))

            # Score métadonnées
            metadata_score = 0
            if any(keyword in query_lower for keyword in chunk.metadata.keywords):
                metadata_score += 0.2
            if any(concept in query_lower for concept in chunk.metadata.concepts):
                metadata_score += 0.3

            total_score = text_score + metadata_score + chunk.metadata.domain_relevance * 0.1
            scored_chunks.append((chunk, total_score))

        # Trier et limiter
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored_chunks[:10]]


# Fonctions d'intégration
def create_hierarchical_chunks_for_rag(document: str, doc_id: str, domain: str = "general") -> Dict[str, List[Dict]]:
    """
    Fonction d'intégration pour RAG Chunk Lab

    Returns:
        Dict avec chunks formatés pour le système RAG
    """

    chunker = HierarchicalChunker(domain=domain)
    hierarchy = chunker.create_hierarchical_chunks(document, doc_id)

    # Convertir en format compatible
    formatted_hierarchy = {}

    for granularity, chunks in hierarchy.items():
        formatted_hierarchy[granularity] = [chunk.to_dict() for chunk in chunks]

    return formatted_hierarchy


def select_optimal_chunks_for_query(query: str,
                                  hierarchy: Dict[str, List[HierarchicalChunk]],
                                  vagueness_score: float) -> List[Dict]:
    """
    Sélectionne les chunks optimaux pour une requête

    Returns:
        Liste de chunks formatés pour RAG
    """

    chunker = HierarchicalChunker()
    selected_chunks = chunker.adaptive_chunk_selection(query, hierarchy, vagueness_score)

    return [chunk.to_dict() for chunk in selected_chunks]