"""
Protocoles d'évaluation génériques pour tout domaine
"""

from typing import List, Dict, Optional, Any
import json
import re
import time
from pathlib import Path
from datetime import datetime
import statistics

class GenericEvaluationProtocol:
    """
    Protocole d'évaluation générique adaptable à tout domaine
    """

    def __init__(self, domain_config: Optional[Dict] = None):
        """
        Args:
            domain_config: Configuration spécifique au domaine
                {
                    "keywords": {"category1": ["word1", "word2"], ...},
                    "patterns": {"dates": r"\d{1,2}/\d{1,2}/\d{4}", ...},
                    "quality_metrics": ["accuracy", "completeness", "relevance"]
                }
        """
        self.domain_config = domain_config or self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Configuration par défaut adaptable"""
        return {
            "keywords": {
                "temporal": ["date", "time", "period", "duration", "deadline"],
                "quantitative": ["number", "amount", "quantity", "percentage", "total"],
                "qualitative": ["quality", "type", "category", "kind", "nature"],
                "procedural": ["process", "procedure", "step", "method", "workflow"]
            },
            "patterns": {
                "dates": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}",
                "numbers": r"\d+(?:\.\d+)?",
                "percentages": r"\d+(?:\.\d+)?%",
                "currencies": r"[$€£¥]\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?)"
            },
            "quality_metrics": [
                "factual_accuracy",
                "completeness",
                "relevance",
                "consistency",
                "clarity"
            ]
        }

    def evaluate_comprehensive_quality(self,
                                     questions: List[str],
                                     answers: List[str],
                                     ground_truth: List[str],
                                     contexts: List[List[str]],
                                     domain: str = "generic") -> Dict:
        """
        Évaluation complète de qualité générique
        """
        results = {
            "domain": domain,
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        # 1. Précision factuelle
        results["metrics"]["factual_accuracy"] = self._evaluate_factual_accuracy(
            questions, answers, ground_truth
        )

        # 2. Complétude des réponses
        results["metrics"]["completeness"] = self._evaluate_completeness(
            questions, answers, ground_truth
        )

        # 3. Pertinence contextuelle
        results["metrics"]["relevance"] = self._evaluate_relevance(
            questions, answers, contexts
        )

        # 4. Consistance
        results["metrics"]["consistency"] = self._evaluate_consistency(
            answers, ground_truth
        )

        # 5. Clarté et lisibilité
        results["metrics"]["clarity"] = self._evaluate_clarity(answers)

        # 6. Spécificité du domaine
        results["metrics"]["domain_specificity"] = self._evaluate_domain_specificity(
            questions, answers, contexts
        )

        # Score global
        results["overall_score"] = self._calculate_overall_score(results["metrics"])

        # Recommandations
        results["recommendations"] = self._generate_generic_recommendations(results["metrics"])

        return results

    def _evaluate_factual_accuracy(self, questions: List[str],
                                 answers: List[str],
                                 ground_truth: List[str]) -> Dict:
        """Évalue la précision factuelle des réponses"""
        accuracy_scores = []

        for q, a, gt in zip(questions, answers, ground_truth):
            # Extraire les faits quantifiables
            answer_facts = self._extract_facts(a)
            gt_facts = self._extract_facts(gt)

            if answer_facts and gt_facts:
                # Calculer l'intersection des faits
                common_facts = set(answer_facts) & set(gt_facts)
                total_facts = set(answer_facts) | set(gt_facts)

                accuracy = len(common_facts) / len(total_facts) if total_facts else 0.0
            else:
                # Similarité textuelle basique si pas de faits extraits
                accuracy = self._text_similarity(a, gt)

            accuracy_scores.append(accuracy)

        return {
            "score": statistics.mean(accuracy_scores) if accuracy_scores else 0.0,
            "distribution": {
                "min": min(accuracy_scores) if accuracy_scores else 0.0,
                "max": max(accuracy_scores) if accuracy_scores else 0.0,
                "std": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
            },
            "sample_size": len(accuracy_scores)
        }

    def _evaluate_completeness(self, questions: List[str],
                             answers: List[str],
                             ground_truth: List[str]) -> Dict:
        """Évalue la complétude des réponses"""
        completeness_scores = []

        for q, a, gt in zip(questions, answers, ground_truth):
            # Analyser la complétude basée sur la longueur et le contenu
            length_ratio = min(len(a), len(gt)) / max(len(a), len(gt)) if max(len(a), len(gt)) > 0 else 0.0

            # Analyser la couverture des concepts clés
            question_concepts = self._extract_key_concepts(q)
            answer_concepts = self._extract_key_concepts(a)
            gt_concepts = self._extract_key_concepts(gt)

            if question_concepts:
                answer_coverage = len(set(answer_concepts) & set(question_concepts)) / len(question_concepts)
                gt_coverage = len(set(gt_concepts) & set(question_concepts)) / len(question_concepts)
                concept_score = answer_coverage / gt_coverage if gt_coverage > 0 else answer_coverage
            else:
                concept_score = 0.5

            completeness = (length_ratio + concept_score) / 2
            completeness_scores.append(completeness)

        return {
            "score": statistics.mean(completeness_scores) if completeness_scores else 0.0,
            "details": {
                "avg_answer_length": statistics.mean([len(a) for a in answers]),
                "avg_gt_length": statistics.mean([len(gt) for gt in ground_truth])
            }
        }

    def _evaluate_relevance(self, questions: List[str],
                          answers: List[str],
                          contexts: List[List[str]]) -> Dict:
        """Évalue la pertinence par rapport au contexte"""
        relevance_scores = []

        for q, a, ctx_list in zip(questions, answers, contexts):
            if not ctx_list:
                relevance_scores.append(0.0)
                continue

            # Calculer la pertinence question-réponse
            qa_relevance = self._text_similarity(q, a)

            # Calculer la pertinence contexte-réponse
            context_text = " ".join(ctx_list)
            ctx_relevance = self._text_similarity(context_text, a)

            # Score combiné
            overall_relevance = (qa_relevance + ctx_relevance) / 2
            relevance_scores.append(overall_relevance)

        return {
            "score": statistics.mean(relevance_scores) if relevance_scores else 0.0,
            "context_utilization": sum(1 for ctx in contexts if ctx) / len(contexts)
        }

    def _evaluate_consistency(self, answers: List[str],
                            ground_truth: List[str]) -> Dict:
        """Évalue la consistance des réponses"""
        consistency_scores = []

        for a, gt in zip(answers, ground_truth):
            # Analyser la cohérence stylistique et factuelle
            style_consistency = self._analyze_style_consistency(a, gt)
            fact_consistency = self._analyze_fact_consistency(a, gt)

            overall_consistency = (style_consistency + fact_consistency) / 2
            consistency_scores.append(overall_consistency)

        return {
            "score": statistics.mean(consistency_scores) if consistency_scores else 0.0,
            "variance": statistics.variance(consistency_scores) if len(consistency_scores) > 1 else 0.0
        }

    def _evaluate_clarity(self, answers: List[str]) -> Dict:
        """Évalue la clarté et lisibilité"""
        clarity_scores = []

        for answer in answers:
            # Métriques de lisibilité
            sentence_count = len(re.findall(r'[.!?]+', answer))
            word_count = len(answer.split())

            if sentence_count > 0:
                avg_sentence_length = word_count / sentence_count
                # Score basé sur la longueur optimale des phrases (10-20 mots)
                length_score = 1.0 - abs(avg_sentence_length - 15) / 15
                length_score = max(0, min(1, length_score))
            else:
                length_score = 0.0

            # Structure et organisation
            structure_score = self._analyze_structure_quality(answer)

            clarity = (length_score + structure_score) / 2
            clarity_scores.append(clarity)

        return {
            "score": statistics.mean(clarity_scores) if clarity_scores else 0.0,
            "readability_metrics": {
                "avg_sentence_length": statistics.mean([
                    len(a.split()) / max(1, len(re.findall(r'[.!?]+', a))) for a in answers
                ]),
                "avg_word_length": statistics.mean([
                    statistics.mean([len(word) for word in a.split()]) if a.split() else 0 for a in answers
                ])
            }
        }

    def _evaluate_domain_specificity(self, questions: List[str],
                                   answers: List[str],
                                   contexts: List[List[str]]) -> Dict:
        """Évalue la spécificité au domaine"""
        domain_scores = []

        for q, a, ctx_list in zip(questions, answers, contexts):
            # Analyser l'usage de terminologie spécialisée
            domain_terms_in_question = self._count_domain_terms(q)
            domain_terms_in_answer = self._count_domain_terms(a)

            context_text = " ".join(ctx_list) if ctx_list else ""
            domain_terms_in_context = self._count_domain_terms(context_text)

            # Score basé sur l'usage approprié de la terminologie
            if domain_terms_in_question > 0:
                terminology_appropriateness = domain_terms_in_answer / domain_terms_in_question
                terminology_appropriateness = min(1.0, terminology_appropriateness)
            else:
                terminology_appropriateness = 0.5

            # Vérifier que les termes utilisés sont supportés par le contexte
            if domain_terms_in_answer > 0 and domain_terms_in_context > 0:
                context_support = min(1.0, domain_terms_in_context / domain_terms_in_answer)
            else:
                context_support = 1.0

            domain_score = (terminology_appropriateness + context_support) / 2
            domain_scores.append(domain_score)

        return {
            "score": statistics.mean(domain_scores) if domain_scores else 0.0,
            "terminology_usage": {
                "avg_terms_per_answer": statistics.mean([self._count_domain_terms(a) for a in answers])
            }
        }

    def _extract_facts(self, text: str) -> List[str]:
        """Extrait les faits quantifiables du texte"""
        facts = []

        # Extraire les nombres, dates, pourcentages
        for pattern_name, pattern in self.domain_config["patterns"].items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            facts.extend([f"{pattern_name}:{match}" for match in matches])

        return facts

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extrait les concepts clés du texte"""
        concepts = []
        text_lower = text.lower()

        for category, keywords in self.domain_config["keywords"].items():
            for keyword in keywords:
                if keyword in text_lower:
                    concepts.append(f"{category}:{keyword}")

        return concepts

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _analyze_style_consistency(self, answer: str, ground_truth: str) -> float:
        """Analyse la cohérence stylistique"""
        # Analyser la longueur relative
        length_ratio = min(len(answer), len(ground_truth)) / max(len(answer), len(ground_truth))

        # Analyser la structure (questions, listes, etc.)
        answer_structure = len(re.findall(r'[?!\-•]', answer))
        gt_structure = len(re.findall(r'[?!\-•]', ground_truth))

        structure_similarity = 1.0 - abs(answer_structure - gt_structure) / max(1, max(answer_structure, gt_structure))

        return (length_ratio + structure_similarity) / 2

    def _analyze_fact_consistency(self, answer: str, ground_truth: str) -> float:
        """Analyse la cohérence factuelle"""
        answer_facts = self._extract_facts(answer)
        gt_facts = self._extract_facts(ground_truth)

        if not answer_facts and not gt_facts:
            return 1.0

        if not answer_facts or not gt_facts:
            return 0.5

        common_facts = set(answer_facts) & set(gt_facts)
        return len(common_facts) / max(len(answer_facts), len(gt_facts))

    def _analyze_structure_quality(self, text: str) -> float:
        """Analyse la qualité structurelle du texte"""
        # Vérifier la présence d'éléments structurels
        has_intro = len(text.split('.')[0]) < len(text) * 0.3 if '.' in text else False
        has_conclusion = text.strip().endswith(('.', '!', '?'))
        has_transitions = len(re.findall(r'\b(however|therefore|moreover|furthermore|additionally)\b', text, re.IGNORECASE)) > 0

        structure_elements = sum([has_intro, has_conclusion, has_transitions])
        return structure_elements / 3

    def _count_domain_terms(self, text: str) -> int:
        """Compte les termes spécialisés du domaine"""
        count = 0
        text_lower = text.lower()

        for category, keywords in self.domain_config["keywords"].items():
            for keyword in keywords:
                count += text_lower.count(keyword)

        return count

    def _calculate_overall_score(self, metrics: Dict) -> float:
        """Calcule le score global"""
        scores = []
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and "score" in metric_data:
                scores.append(metric_data["score"])

        return statistics.mean(scores) if scores else 0.0

    def _generate_generic_recommendations(self, metrics: Dict) -> List[str]:
        """Génère des recommandations génériques"""
        recommendations = []

        # Analyser chaque métrique
        factual_score = metrics.get("factual_accuracy", {}).get("score", 0)
        if factual_score < 0.7:
            recommendations.append("🎯 Améliorer la précision factuelle des réponses")

        completeness_score = metrics.get("completeness", {}).get("score", 0)
        if completeness_score < 0.6:
            recommendations.append("📝 Enrichir la complétude des réponses")

        relevance_score = metrics.get("relevance", {}).get("score", 0)
        if relevance_score < 0.8:
            recommendations.append("🎯 Améliorer la pertinence contextuelle")

        consistency_score = metrics.get("consistency", {}).get("score", 0)
        if consistency_score < 0.7:
            recommendations.append("🔄 Renforcer la cohérence des réponses")

        clarity_score = metrics.get("clarity", {}).get("score", 0)
        if clarity_score < 0.7:
            recommendations.append("✨ Améliorer la clarté et la lisibilité")

        domain_score = metrics.get("domain_specificity", {}).get("score", 0)
        if domain_score < 0.6:
            recommendations.append("🏷️ Utiliser davantage la terminologie spécialisée")

        if not recommendations:
            recommendations.append("✅ Excellente qualité générale - maintenir le niveau")

        return recommendations

    def generate_evaluation_report(self, evaluation_results: Dict,
                                 doc_id: str,
                                 output_dir: str = "generic_evaluation") -> str:
        """Génère un rapport d'évaluation générique"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Ajouter des métadonnées
        report = {
            "doc_id": doc_id,
            "evaluation_type": "generic_comprehensive",
            "domain": evaluation_results.get("domain", "generic"),
            **evaluation_results
        }

        report_file = Path(output_dir) / f"generic_evaluation_{doc_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Générer aussi un résumé CSV
        self._generate_csv_summary(evaluation_results, doc_id, output_dir)

        return str(report_file)

    def _generate_csv_summary(self, results: Dict, doc_id: str, output_dir: str):
        """Génère un résumé CSV"""
        import csv

        csv_file = Path(output_dir) / f"generic_metrics_summary_{doc_id}.csv"

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'score', 'details'])

            for metric_name, metric_data in results.get("metrics", {}).items():
                if isinstance(metric_data, dict):
                    score = metric_data.get("score", 0)
                    details = str(metric_data.get("details", ""))
                    writer.writerow([metric_name, f"{score:.3f}", details])

            # Score global
            overall_score = results.get("overall_score", 0)
            writer.writerow(["overall_score", f"{overall_score:.3f}", "Combined score"])

def run_generic_evaluation_suite(doc_id: str,
                               questions: List[str],
                               per_pipeline_answers: Dict[str, List[str]],
                               per_pipeline_contexts: Dict[str, List[List[str]]],
                               ground_truth: List[str],
                               domain_config: Optional[Dict] = None) -> Dict:
    """
    Lance une évaluation générique complète
    """
    evaluator = GenericEvaluationProtocol(domain_config)
    results = {}

    print(f"🔍 Starting generic evaluation for: {doc_id}")

    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        print(f"📊 Evaluating pipeline: {pipeline}")

        pipeline_results = evaluator.evaluate_comprehensive_quality(
            questions=questions,
            answers=answers,
            ground_truth=ground_truth,
            contexts=contexts,
            domain=doc_id
        )

        results[pipeline] = pipeline_results

        # Générer rapport pour ce pipeline
        report_path = evaluator.generate_evaluation_report(
            pipeline_results,
            f"{doc_id}_{pipeline}"
        )

        # Afficher résultats
        overall_score = pipeline_results.get("overall_score", 0)
        print(f"  ✅ Score global: {overall_score:.3f}")
        print(f"  📋 Rapport: {report_path}")

    return results