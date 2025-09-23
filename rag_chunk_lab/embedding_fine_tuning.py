# embedding_fine_tuning.py
"""
Système de fine-tuning d'embeddings pour domaines spécifiques
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import random
import math

# Imports conditionnels
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers with PyTorch not available for fine-tuning")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Exemple d'entraînement pour fine-tuning"""
    text_a: str
    text_b: str
    label: float  # 0.0 = dissimilaire, 1.0 = similaire
    metadata: Dict[str, Any] = None

    def to_input_example(self) -> 'InputExample':
        """Convertit en InputExample pour sentence-transformers"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for training")

        return InputExample(texts=[self.text_a, self.text_b], label=self.label)


@dataclass
class FineTuningConfig:
    """Configuration pour le fine-tuning"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_path: str = "./fine_tuned_model"
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 256
    use_contrastive_loss: bool = True
    use_triplet_loss: bool = False
    evaluation_steps: int = 500
    save_best_model: bool = True


class DomainDataGenerator:
    """Générateur de données d'entraînement spécifiques au domaine"""

    def __init__(self, domain: str = "general", language: str = "fr"):
        self.domain = domain
        self.language = language

        # Patterns et règles par domaine
        self.domain_rules = self._load_domain_rules(domain)

    def _load_domain_rules(self, domain: str) -> Dict[str, Any]:
        """Charge les règles de génération par domaine"""

        rules = {
            "legal": {
                "similarity_patterns": [
                    ("article {}", "art. {}"),  # Variations d'écriture
                    ("procédure de {}", "démarche pour {}"),
                    ("loi sur {}", "législation {}"),
                    ("tribunal de {}", "juridiction {}"),
                    ("avocat spécialisé en {}", "conseil juridique en {}")
                ],
                "concepts": [
                    "droit civil", "droit pénal", "droit commercial", "droit administratif",
                    "procédure civile", "procédure pénale", "jurisprudence", "doctrine",
                    "contrat", "responsabilité", "obligation", "préjudice"
                ],
                "entities": [
                    "Cour de cassation", "Conseil d'État", "Tribunal de grande instance",
                    "Cour d'appel", "Tribunal de commerce", "Conseil de prud'hommes"
                ]
            },

            "technical": {
                "similarity_patterns": [
                    ("API {}", "interface {}"),
                    ("configuration de {}", "paramétrage de {}"),
                    ("installation de {}", "mise en place de {}"),
                    ("architecture {}", "structure {}"),
                    ("base de données {}", "BDD {}")
                ],
                "concepts": [
                    "authentification", "autorisation", "sécurité", "chiffrement",
                    "performance", "scalabilité", "disponibilité", "résilience",
                    "microservices", "architecture", "framework", "librairie"
                ],
                "entities": [
                    "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
                    "Docker", "Kubernetes", "AWS", "Azure", "REST", "GraphQL"
                ]
            },

            "medical": {
                "similarity_patterns": [
                    ("symptôme de {}", "signe de {}"),
                    ("traitement de {}", "thérapie pour {}"),
                    ("diagnostic de {}", "diagnostic {}"),
                    ("pathologie {}", "maladie {}"),
                    ("médicament pour {}", "traitement pharmacologique de {}")
                ],
                "concepts": [
                    "diagnostic", "symptôme", "traitement", "thérapie", "prévention",
                    "pathologie", "épidémiologie", "pharmacologie", "anatomie",
                    "physiologie", "immunologie", "oncologie", "cardiologie"
                ],
                "entities": [
                    "OMS", "FDA", "EMA", "ANSM", "hôpital", "clinique",
                    "médecin", "spécialiste", "patient", "urgences"
                ]
            }
        }

        return rules.get(domain, {
            "similarity_patterns": [],
            "concepts": [],
            "entities": []
        })

    def generate_training_examples(self,
                                  chunks: List[Dict[str, Any]],
                                  num_positive: int = 1000,
                                  num_negative: int = 1000) -> List[TrainingExample]:
        """
        Génère des exemples d'entraînement depuis les chunks

        Args:
            chunks: Liste de chunks avec texte et métadonnées
            num_positive: Nombre d'exemples positifs (similaires)
            num_negative: Nombre d'exemples négatifs (dissimilaires)

        Returns:
            Liste d'exemples d'entraînement
        """

        examples = []

        # Générer exemples positifs
        positive_examples = self._generate_positive_examples(chunks, num_positive)
        examples.extend(positive_examples)

        # Générer exemples négatifs
        negative_examples = self._generate_negative_examples(chunks, num_negative)
        examples.extend(negative_examples)

        # Mélanger
        random.shuffle(examples)

        logger.info(f"Generated {len(examples)} training examples ({len(positive_examples)} positive, {len(negative_examples)} negative)")

        return examples

    def _generate_positive_examples(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des exemples positifs (chunks similaires)"""

        positive_examples = []

        # Stratégie 1: Chunks du même niveau de granularité avec concepts partagés
        conceptual_examples = self._generate_conceptual_similarities(chunks, num_examples // 3)
        positive_examples.extend(conceptual_examples)

        # Stratégie 2: Variations textuelles du même contenu
        variation_examples = self._generate_textual_variations(chunks, num_examples // 3)
        positive_examples.extend(variation_examples)

        # Stratégie 3: Relations hiérarchiques (parent-enfant)
        hierarchical_examples = self._generate_hierarchical_similarities(chunks, num_examples // 3)
        positive_examples.extend(hierarchical_examples)

        return positive_examples[:num_examples]

    def _generate_conceptual_similarities(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des exemples basés sur la similarité conceptuelle"""

        examples = []

        # Grouper chunks par concepts
        concept_groups = defaultdict(list)

        for chunk in chunks:
            enriched_meta = chunk.get("enriched_metadata", {})
            concepts = enriched_meta.get("concepts", [])

            for concept in concepts:
                concept_name = concept.get("concept", "") if isinstance(concept, dict) else str(concept)
                if concept_name:
                    concept_groups[concept_name].append(chunk)

        # Créer paires de chunks partageant des concepts
        for concept, chunk_list in concept_groups.items():
            if len(chunk_list) >= 2:
                for i in range(min(5, len(chunk_list))):  # Limiter pour éviter explosion
                    for j in range(i + 1, min(i + 3, len(chunk_list))):
                        if len(examples) >= num_examples:
                            break

                        chunk_a = chunk_list[i]
                        chunk_b = chunk_list[j]

                        # Calculer similarité basée sur concepts partagés
                        similarity_score = self._calculate_concept_similarity(chunk_a, chunk_b)

                        if similarity_score >= 0.3:  # Seuil de similarité
                            example = TrainingExample(
                                text_a=chunk_a["text"],
                                text_b=chunk_b["text"],
                                label=similarity_score,
                                metadata={
                                    "type": "conceptual",
                                    "shared_concept": concept,
                                    "similarity_score": similarity_score
                                }
                            )
                            examples.append(example)

                    if len(examples) >= num_examples:
                        break

        return examples

    def _calculate_concept_similarity(self, chunk_a: Dict, chunk_b: Dict) -> float:
        """Calcule la similarité conceptuelle entre deux chunks"""

        meta_a = chunk_a.get("enriched_metadata", {})
        meta_b = chunk_b.get("enriched_metadata", {})

        # Concepts
        concepts_a = set(c.get("concept", "") if isinstance(c, dict) else str(c)
                        for c in meta_a.get("concepts", []))
        concepts_b = set(c.get("concept", "") if isinstance(c, dict) else str(c)
                        for c in meta_b.get("concepts", []))

        # Mots-clés
        keywords_a = set(k.get("keyword", "") if isinstance(k, dict) else str(k)
                        for k in meta_a.get("keywords", []))
        keywords_b = set(k.get("keyword", "") if isinstance(k, dict) else str(k)
                        for k in meta_b.get("keywords", []))

        # Calculer intersections
        shared_concepts = concepts_a.intersection(concepts_b)
        shared_keywords = keywords_a.intersection(keywords_b)

        # Score basé sur les éléments partagés
        concept_score = len(shared_concepts) / max(1, len(concepts_a.union(concepts_b)))
        keyword_score = len(shared_keywords) / max(1, len(keywords_a.union(keywords_b)))

        # Score final pondéré
        return concept_score * 0.7 + keyword_score * 0.3

    def _generate_textual_variations(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des variations textuelles basées sur les patterns du domaine"""

        examples = []
        patterns = self.domain_rules.get("similarity_patterns", [])

        if not patterns:
            return examples

        for chunk in chunks:
            if len(examples) >= num_examples:
                break

            text = chunk["text"]

            # Appliquer patterns de transformation
            for pattern_a, pattern_b in patterns:
                # Rechercher si le pattern s'applique
                if "{}" in pattern_a:
                    # Pattern avec placeholder
                    base_pattern = pattern_a.replace("{}", "([^.]+)")
                    import re
                    matches = re.findall(base_pattern, text, re.IGNORECASE)

                    for match in matches:
                        if len(examples) >= num_examples:
                            break

                        # Créer variation
                        original_phrase = pattern_a.format(match)
                        variation_phrase = pattern_b.format(match)

                        # Remplacer dans le contexte
                        varied_text = text.replace(original_phrase, variation_phrase)

                        if varied_text != text:
                            example = TrainingExample(
                                text_a=text,
                                text_b=varied_text,
                                label=0.8,  # Haute similarité pour variations
                                metadata={
                                    "type": "textual_variation",
                                    "pattern": f"{pattern_a} -> {pattern_b}",
                                    "original_phrase": original_phrase,
                                    "variation_phrase": variation_phrase
                                }
                            )
                            examples.append(example)

        return examples

    def _generate_hierarchical_similarities(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des exemples basés sur les relations hiérarchiques"""

        examples = []

        # Créer mapping parent -> enfants
        parent_child_map = defaultdict(list)

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            parent_id = metadata.get("parent_id")

            if parent_id:
                parent_child_map[parent_id].append(chunk)

        # Créer exemples parent-enfant
        for parent_id, children in parent_child_map.items():
            if len(examples) >= num_examples:
                break

            # Trouver le chunk parent
            parent_chunk = None
            for chunk in chunks:
                if chunk.get("metadata", {}).get("chunk_id") == parent_id:
                    parent_chunk = chunk
                    break

            if parent_chunk:
                for child in children:
                    if len(examples) >= num_examples:
                        break

                    # Similarité modérée pour relations hiérarchiques
                    example = TrainingExample(
                        text_a=parent_chunk["text"],
                        text_b=child["text"],
                        label=0.6,
                        metadata={
                            "type": "hierarchical",
                            "relationship": "parent_child",
                            "parent_id": parent_id,
                            "child_id": child.get("metadata", {}).get("chunk_id")
                        }
                    )
                    examples.append(example)

        return examples

    def _generate_negative_examples(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des exemples négatifs (chunks dissimilaires)"""

        examples = []

        # Stratégie 1: Chunks de granularités très différentes
        granularity_examples = self._generate_granularity_negatives(chunks, num_examples // 2)
        examples.extend(granularity_examples)

        # Stratégie 2: Chunks de domaines conceptuels différents
        conceptual_examples = self._generate_conceptual_negatives(chunks, num_examples // 2)
        examples.extend(conceptual_examples)

        return examples[:num_examples]

    def _generate_granularity_negatives(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des négatifs basés sur des granularités très différentes"""

        examples = []

        # Grouper par granularité
        granularity_groups = defaultdict(list)
        for chunk in chunks:
            granularity = chunk.get("metadata", {}).get("granularity", "unknown")
            granularity_groups[granularity].append(chunk)

        # Définir granularités opposées
        opposite_pairs = [
            ("sentence", "document"),
            ("sentence", "summary"),
            ("concept", "paragraph")
        ]

        for gran_a, gran_b in opposite_pairs:
            chunks_a = granularity_groups.get(gran_a, [])
            chunks_b = granularity_groups.get(gran_b, [])

            for chunk_a in chunks_a[:20]:  # Limiter
                for chunk_b in chunks_b[:5]:
                    if len(examples) >= num_examples:
                        break

                    example = TrainingExample(
                        text_a=chunk_a["text"],
                        text_b=chunk_b["text"],
                        label=0.1,  # Très faible similarité
                        metadata={
                            "type": "granularity_negative",
                            "granularity_a": gran_a,
                            "granularity_b": gran_b
                        }
                    )
                    examples.append(example)

                if len(examples) >= num_examples:
                    break

        return examples

    def _generate_conceptual_negatives(self, chunks: List[Dict], num_examples: int) -> List[TrainingExample]:
        """Génère des négatifs basés sur des concepts différents"""

        examples = []

        # Grouper par concepts dominants
        concept_groups = defaultdict(list)

        for chunk in chunks:
            enriched_meta = chunk.get("enriched_metadata", {})
            concepts = enriched_meta.get("concepts", [])

            if concepts:
                # Prendre le concept principal
                main_concept = concepts[0]
                concept_name = main_concept.get("concept", "") if isinstance(main_concept, dict) else str(main_concept)
                if concept_name:
                    concept_groups[concept_name].append(chunk)

        # Créer paires de concepts différents
        concept_names = list(concept_groups.keys())

        for i, concept_a in enumerate(concept_names):
            for j, concept_b in enumerate(concept_names):
                if i >= j or len(examples) >= num_examples:
                    continue

                chunks_a = concept_groups[concept_a]
                chunks_b = concept_groups[concept_b]

                # Vérifier que les concepts sont vraiment différents
                if self._are_concepts_dissimilar(concept_a, concept_b):
                    for chunk_a in chunks_a[:5]:
                        for chunk_b in chunks_b[:3]:
                            if len(examples) >= num_examples:
                                break

                            example = TrainingExample(
                                text_a=chunk_a["text"],
                                text_b=chunk_b["text"],
                                label=0.2,  # Faible similarité
                                metadata={
                                    "type": "conceptual_negative",
                                    "concept_a": concept_a,
                                    "concept_b": concept_b
                                }
                            )
                            examples.append(example)

                        if len(examples) >= num_examples:
                            break

        return examples

    def _are_concepts_dissimilar(self, concept_a: str, concept_b: str) -> bool:
        """Détermine si deux concepts sont suffisamment dissimilaires"""

        # Simple heuristique basée sur les mots
        words_a = set(concept_a.lower().split())
        words_b = set(concept_b.lower().split())

        overlap = words_a.intersection(words_b)
        total = words_a.union(words_b)

        similarity = len(overlap) / max(1, len(total))

        return similarity < 0.3  # Concepts différents si peu de mots en commun


class EmbeddingFineTuner:
    """Système de fine-tuning d'embeddings"""

    def __init__(self, config: FineTuningConfig):
        self.config = config

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers with PyTorch required for fine-tuning")

        # Charger le modèle de base
        self.model = SentenceTransformer(config.model_name)
        self.model.max_seq_length = config.max_seq_length

        # Métriques d'évaluation
        self.training_history = {
            "loss": [],
            "eval_similarity": [],
            "epochs": []
        }

    def fine_tune(self,
                  training_examples: List[TrainingExample],
                  validation_examples: List[TrainingExample] = None,
                  domain: str = "general") -> str:
        """
        Fine-tune le modèle d'embeddings

        Args:
            training_examples: Exemples d'entraînement
            validation_examples: Exemples de validation (optionnel)
            domain: Domaine spécialisé

        Returns:
            Chemin du modèle fine-tuné
        """

        logger.info(f"Starting fine-tuning with {len(training_examples)} examples")

        # Convertir en InputExamples
        train_samples = [example.to_input_example() for example in training_examples]

        # DataLoader
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.config.batch_size)

        # Configurer la loss function
        if self.config.use_contrastive_loss:
            train_loss = losses.CosineSimilarityLoss(self.model)
        elif self.config.use_triplet_loss:
            train_loss = losses.TripletLoss(self.model)
        else:
            train_loss = losses.CosineSimilarityLoss(self.model)

        # Configurar l'évaluateur si validation disponible
        evaluator = None
        if validation_examples:
            val_samples = [example.to_input_example() for example in validation_examples]
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                val_samples, name=f"{domain}_evaluation"
            )

        # Calcul warmup steps
        warmup_steps = min(self.config.warmup_steps, len(train_dataloader) * self.config.epochs // 10)

        # Entraînement
        output_path = f"{self.config.output_path}_{domain}"

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            evaluator=evaluator,
            evaluation_steps=self.config.evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=self.config.save_best_model,
            show_progress_bar=True
        )

        # Sauvegarder les métadonnées d'entraînement
        self._save_training_metadata(output_path, domain, training_examples, validation_examples)

        logger.info(f"Fine-tuning completed. Model saved to {output_path}")

        return output_path

    def _save_training_metadata(self,
                              output_path: str,
                              domain: str,
                              training_examples: List[TrainingExample],
                              validation_examples: List[TrainingExample] = None):
        """Sauvegarde les métadonnées d'entraînement"""

        metadata = {
            "domain": domain,
            "base_model": self.config.model_name,
            "training_config": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "max_seq_length": self.config.max_seq_length,
                "use_contrastive_loss": self.config.use_contrastive_loss,
                "use_triplet_loss": self.config.use_triplet_loss
            },
            "dataset_stats": {
                "training_examples": len(training_examples),
                "validation_examples": len(validation_examples) if validation_examples else 0,
                "positive_examples": sum(1 for ex in training_examples if ex.label >= 0.5),
                "negative_examples": sum(1 for ex in training_examples if ex.label < 0.5)
            },
            "example_types": self._analyze_example_types(training_examples)
        }

        metadata_file = Path(output_path) / "training_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _analyze_example_types(self, examples: List[TrainingExample]) -> Dict[str, int]:
        """Analyse les types d'exemples d'entraînement"""

        type_counts = Counter()

        for example in examples:
            if example.metadata:
                example_type = example.metadata.get("type", "unknown")
                type_counts[example_type] += 1

        return dict(type_counts)

    def evaluate_model(self,
                      model_path: str,
                      test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Évalue un modèle fine-tuné"""

        # Charger le modèle fine-tuné
        model = SentenceTransformer(model_path)

        # Calculer embeddings pour tous les exemples
        texts_a = [ex.text_a for ex in test_examples]
        texts_b = [ex.text_b for ex in test_examples]
        true_labels = [ex.label for ex in test_examples]

        embeddings_a = model.encode(texts_a)
        embeddings_b = model.encode(texts_b)

        # Calculer similarités prédites
        if SKLEARN_AVAILABLE:
            predicted_similarities = []
            for i in range(len(embeddings_a)):
                similarity = cosine_similarity([embeddings_a[i]], [embeddings_b[i]])[0][0]
                predicted_similarities.append(similarity)
        else:
            # Fallback sans sklearn
            predicted_similarities = []
            for i in range(len(embeddings_a)):
                emb_a = embeddings_a[i]
                emb_b = embeddings_b[i]
                similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                predicted_similarities.append(similarity)

        # Calculer métriques
        mae = np.mean(np.abs(np.array(predicted_similarities) - np.array(true_labels)))
        rmse = np.sqrt(np.mean((np.array(predicted_similarities) - np.array(true_labels)) ** 2))

        # Correlation
        correlation = np.corrcoef(predicted_similarities, true_labels)[0, 1]

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "correlation": float(correlation),
            "num_examples": len(test_examples)
        }

        logger.info(f"Model evaluation: MAE={mae:.3f}, RMSE={rmse:.3f}, Correlation={correlation:.3f}")

        return metrics


# Fonctions d'intégration
def create_fine_tuned_model_for_domain(hierarchy: Dict[str, List[Dict]],
                                      domain: str = "general",
                                      config: FineTuningConfig = None) -> str:
    """
    Crée un modèle fine-tuné pour un domaine spécifique

    Args:
        hierarchy: Hiérarchie de chunks enrichis
        domain: Domaine spécialisé
        config: Configuration de fine-tuning

    Returns:
        Chemin du modèle fine-tuné
    """

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("Fine-tuning not available. Install sentence-transformers with PyTorch")
        return None

    # Configuration par défaut
    if config is None:
        config = FineTuningConfig()

    # Extraire tous les chunks
    all_chunks = []
    for granularity, chunks in hierarchy.items():
        all_chunks.extend(chunks)

    logger.info(f"Creating fine-tuned model from {len(all_chunks)} chunks")

    # Générer données d'entraînement
    data_generator = DomainDataGenerator(domain=domain)
    training_examples = data_generator.generate_training_examples(
        chunks=all_chunks,
        num_positive=1500,
        num_negative=1500
    )

    # Diviser train/validation
    random.shuffle(training_examples)
    split_idx = int(0.8 * len(training_examples))
    train_examples = training_examples[:split_idx]
    val_examples = training_examples[split_idx:]

    # Fine-tuning
    fine_tuner = EmbeddingFineTuner(config)
    model_path = fine_tuner.fine_tune(
        training_examples=train_examples,
        validation_examples=val_examples,
        domain=domain
    )

    # Évaluation finale
    if val_examples:
        metrics = fine_tuner.evaluate_model(model_path, val_examples)
        logger.info(f"Final evaluation metrics: {metrics}")

    return model_path


def compare_models_performance(base_model_name: str,
                             fine_tuned_model_path: str,
                             test_examples: List[TrainingExample]) -> Dict[str, Dict[str, float]]:
    """
    Compare les performances entre modèle de base et modèle fine-tuné

    Returns:
        Comparaison des métriques
    """

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {"error": "sentence-transformers not available"}

    # Évaluer modèle de base
    fine_tuner_base = EmbeddingFineTuner(FineTuningConfig(model_name=base_model_name))
    base_metrics = fine_tuner_base.evaluate_model(base_model_name, test_examples)

    # Évaluer modèle fine-tuné
    fine_tuner_ft = EmbeddingFineTuner(FineTuningConfig())
    ft_metrics = fine_tuner_ft.evaluate_model(fine_tuned_model_path, test_examples)

    # Calculer améliorations
    improvements = {}
    for metric in base_metrics:
        if metric != "num_examples":
            if metric in ["mae", "rmse"]:
                # Pour MAE et RMSE, plus faible = mieux
                improvement = (base_metrics[metric] - ft_metrics[metric]) / base_metrics[metric] * 100
            else:
                # Pour correlation, plus élevé = mieux
                improvement = (ft_metrics[metric] - base_metrics[metric]) / abs(base_metrics[metric]) * 100

            improvements[metric] = improvement

    return {
        "base_model": base_metrics,
        "fine_tuned_model": ft_metrics,
        "improvements_percent": improvements
    }