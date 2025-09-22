"""
Protocole d'évaluation spécialisé pour documents juridiques
"""

from typing import List, Dict, Optional
import json
import re
from datetime import datetime
from pathlib import Path

class LegalEvaluationProtocol:
    """
    Protocole d'évaluation spécialisé pour les documents juridiques
    """

    def __init__(self):
        self.legal_keywords = {
            'temporal': ['délai', 'prescription', 'durée', 'terme', 'période'],
            'procedural': ['procédure', 'recours', 'appel', 'pourvoi', 'instance'],
            'sanctions': ['amende', 'sanction', 'peine', 'condamnation', 'punition'],
            'rights': ['droit', 'obligation', 'devoir', 'faculté', 'prérogative']
        }

    def evaluate_legal_accuracy(self,
                               questions: List[str],
                               answers: List[str],
                               ground_truth: List[str],
                               contexts: List[List[str]]) -> Dict:
        """
        Évalue la précision juridique des réponses
        """
        results = {
            'temporal_accuracy': self._evaluate_temporal_consistency(questions, answers, ground_truth),
            'legal_terminology': self._evaluate_legal_terminology(answers, contexts),
            'citation_accuracy': self._evaluate_citation_accuracy(answers, contexts),
            'procedural_completeness': self._evaluate_procedural_completeness(questions, answers),
            'consistency_check': self._evaluate_consistency(answers, ground_truth)
        }

        return results

    def _evaluate_temporal_consistency(self, questions: List[str],
                                     answers: List[str],
                                     ground_truth: List[str]) -> Dict:
        """
        Vérifie la cohérence des informations temporelles (délais, prescriptions)
        """
        temporal_scores = []

        for q, a, gt in zip(questions, answers, ground_truth):
            if any(keyword in q.lower() for keyword in self.legal_keywords['temporal']):
                # Extraire les durées/nombres
                answer_numbers = re.findall(r'\d+', a)
                gt_numbers = re.findall(r'\d+', gt)

                if answer_numbers and gt_numbers:
                    # Vérifier si les nombres correspondent
                    score = 1.0 if any(num in gt_numbers for num in answer_numbers) else 0.0
                else:
                    score = 0.5  # Information partielle

                temporal_scores.append(score)

        return {
            'score': sum(temporal_scores) / len(temporal_scores) if temporal_scores else 0.0,
            'evaluated_questions': len(temporal_scores),
            'details': temporal_scores
        }

    def _evaluate_legal_terminology(self, answers: List[str],
                                  contexts: List[List[str]]) -> Dict:
        """
        Évalue l'usage correct de la terminologie juridique
        """
        terminology_scores = []

        for answer, context_list in zip(answers, contexts):
            if not context_list:
                continue

            context_text = ' '.join(context_list).lower()
            answer_lower = answer.lower()

            # Vérifier si les termes juridiques de la réponse sont présents dans le contexte
            legal_terms_in_answer = []
            for category, terms in self.legal_keywords.items():
                for term in terms:
                    if term in answer_lower:
                        legal_terms_in_answer.append(term)

            if legal_terms_in_answer:
                # Calculer le pourcentage de termes juridiques supportés par le contexte
                supported_terms = sum(1 for term in legal_terms_in_answer if term in context_text)
                score = supported_terms / len(legal_terms_in_answer)
            else:
                score = 1.0  # Pas de termes juridiques spécifiques

            terminology_scores.append(score)

        return {
            'score': sum(terminology_scores) / len(terminology_scores) if terminology_scores else 0.0,
            'evaluated_answers': len(terminology_scores)
        }

    def _evaluate_citation_accuracy(self, answers: List[str],
                                   contexts: List[List[str]]) -> Dict:
        """
        Évalue la précision des références juridiques (articles, codes, etc.)
        """
        citation_scores = []

        for answer, context_list in zip(answers, contexts):
            if not context_list:
                continue

            # Rechercher les références juridiques dans la réponse
            answer_citations = re.findall(r'article\s+\d+|art\.\s*\d+|code\s+\w+', answer.lower())

            if answer_citations:
                context_text = ' '.join(context_list).lower()

                # Vérifier si les citations sont présentes dans le contexte
                supported_citations = sum(1 for citation in answer_citations
                                        if citation in context_text)
                score = supported_citations / len(answer_citations)
            else:
                score = 1.0  # Pas de citations spécifiques

            citation_scores.append(score)

        return {
            'score': sum(citation_scores) / len(citation_scores) if citation_scores else 0.0,
            'evaluated_answers': len(citation_scores)
        }

    def _evaluate_procedural_completeness(self, questions: List[str],
                                        answers: List[str]) -> Dict:
        """
        Évalue si les réponses sur les procédures sont complètes
        """
        procedural_scores = []

        for q, a in zip(questions, answers):
            if any(keyword in q.lower() for keyword in self.legal_keywords['procedural']):
                # Éléments attendus dans une réponse procédurale
                expected_elements = ['délai', 'condition', 'document', 'autorité', 'étape']

                present_elements = sum(1 for elem in expected_elements
                                     if elem in a.lower())
                score = min(present_elements / 3, 1.0)  # Au moins 3 éléments sur 5

                procedural_scores.append(score)

        return {
            'score': sum(procedural_scores) / len(procedural_scores) if procedural_scores else 0.0,
            'evaluated_questions': len(procedural_scores)
        }

    def _evaluate_consistency(self, answers: List[str],
                            ground_truth: List[str]) -> Dict:
        """
        Évalue la cohérence des réponses avec la vérité terrain
        """
        consistency_scores = []

        for answer, gt in zip(answers, ground_truth):
            # Analyse de sentiment/tonalité (positif/négatif/neutre)
            answer_sentiment = self._extract_legal_sentiment(answer)
            gt_sentiment = self._extract_legal_sentiment(gt)

            sentiment_match = 1.0 if answer_sentiment == gt_sentiment else 0.5

            # Analyse de la complétude (longueur relative)
            length_ratio = min(len(answer), len(gt)) / max(len(answer), len(gt))
            completeness = 1.0 if length_ratio > 0.5 else length_ratio * 2

            overall_score = (sentiment_match + completeness) / 2
            consistency_scores.append(overall_score)

        return {
            'score': sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0,
            'evaluated_pairs': len(consistency_scores)
        }

    def _extract_legal_sentiment(self, text: str) -> str:
        """
        Détermine le sentiment juridique (autorisé/interdit/conditionnel)
        """
        text_lower = text.lower()

        if any(word in text_lower for word in ['interdit', 'prohibé', 'défendu', 'illégal']):
            return 'negative'
        elif any(word in text_lower for word in ['autorisé', 'permis', 'légal', 'droit']):
            return 'positive'
        else:
            return 'neutral'

    def generate_legal_report(self, evaluation_results: Dict,
                            doc_id: str,
                            output_dir: str = "legal_evaluation") -> str:
        """
        Génère un rapport d'évaluation juridique
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        report = {
            "doc_id": doc_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "legal_accuracy_metrics": evaluation_results,
            "overall_legal_score": self._calculate_overall_legal_score(evaluation_results),
            "recommendations": self._generate_legal_recommendations(evaluation_results)
        }

        report_file = Path(output_dir) / f"legal_evaluation_{doc_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(report_file)

    def _calculate_overall_legal_score(self, results: Dict) -> float:
        """
        Calcule un score global de précision juridique
        """
        scores = [
            results.get('temporal_accuracy', {}).get('score', 0),
            results.get('legal_terminology', {}).get('score', 0),
            results.get('citation_accuracy', {}).get('score', 0),
            results.get('procedural_completeness', {}).get('score', 0),
            results.get('consistency_check', {}).get('score', 0)
        ]

        return sum(scores) / len(scores)

    def _generate_legal_recommendations(self, results: Dict) -> List[str]:
        """
        Génère des recommandations spécialisées
        """
        recommendations = []

        temporal_score = results.get('temporal_accuracy', {}).get('score', 0)
        if temporal_score < 0.7:
            recommendations.append("⚠️ Améliorer la précision des informations temporelles (délais, prescriptions)")

        terminology_score = results.get('legal_terminology', {}).get('score', 0)
        if terminology_score < 0.8:
            recommendations.append("📚 Renforcer l'usage de la terminologie juridique appropriée")

        citation_score = results.get('citation_accuracy', {}).get('score', 0)
        if citation_score < 0.9:
            recommendations.append("📖 Vérifier la précision des références aux articles et codes")

        procedural_score = results.get('procedural_completeness', {}).get('score', 0)
        if procedural_score < 0.6:
            recommendations.append("⚖️ Compléter les réponses procédurales (délais, conditions, étapes)")

        if not recommendations:
            recommendations.append("✅ Excellente qualité juridique - maintenir le niveau")

        return recommendations

def run_legal_evaluation_suite(doc_id: str,
                             questions: List[str],
                             per_pipeline_answers: Dict[str, List[str]],
                             per_pipeline_contexts: Dict[str, List[List[str]]],
                             ground_truth: List[str]) -> Dict:
    """
    Lance une évaluation juridique complète
    """
    legal_protocol = LegalEvaluationProtocol()

    results = {}

    for pipeline, answers in per_pipeline_answers.items():
        contexts = per_pipeline_contexts.get(pipeline, [])

        pipeline_results = legal_protocol.evaluate_legal_accuracy(
            questions=questions,
            answers=answers,
            ground_truth=ground_truth,
            contexts=contexts
        )

        results[pipeline] = pipeline_results

        # Générer rapport pour ce pipeline
        report_path = legal_protocol.generate_legal_report(
            pipeline_results,
            f"{doc_id}_{pipeline}"
        )

        print(f"📋 Rapport juridique {pipeline}: {report_path}")

    return results