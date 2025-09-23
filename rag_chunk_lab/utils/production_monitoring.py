# production_monitoring.py
"""
Système de monitoring et feedback loop pour l'optimisation continue du RAG
"""

from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import logging
from pathlib import Path
import time
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading
import queue

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types de métriques surveillées"""
    RESPONSE_TIME = "response_time"
    RELEVANCE_SCORE = "relevance_score"
    USER_SATISFACTION = "user_satisfaction"
    VAGUE_QUERY_RATE = "vague_query_rate"
    ERROR_RATE = "error_rate"
    CONTEXT_QUALITY = "context_quality"
    EMBEDDING_PERFORMANCE = "embedding_performance"


class AlertLevel(Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Point de métrique avec timestamp"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class Alert:
    """Alerte générée par le système de monitoring"""
    level: AlertLevel
    metric_type: MetricType
    message: str
    timestamp: datetime
    current_value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "metric_type": self.metric_type.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "current_value": self.current_value,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None
        }


@dataclass
class UserFeedback:
    """Feedback utilisateur sur une réponse"""
    query: str
    response: str
    relevance_score: int  # 1-5
    helpfulness_score: int  # 1-5
    clarity_score: int  # 1-5
    timestamp: datetime
    user_id: Optional[str] = None
    response_time: Optional[float] = None
    vagueness_score: Optional[float] = None
    improvements_suggested: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_overall_score(self) -> float:
        """Calcule un score global"""
        return (self.relevance_score + self.helpfulness_score + self.clarity_score) / 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "relevance_score": self.relevance_score,
            "helpfulness_score": self.helpfulness_score,
            "clarity_score": self.clarity_score,
            "overall_score": self.get_overall_score(),
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "response_time": self.response_time,
            "vagueness_score": self.vagueness_score,
            "improvements_suggested": self.improvements_suggested,
            "metadata": self.metadata
        }


@dataclass
class MonitoringConfig:
    """Configuration du système de monitoring"""
    # Fenêtres de temps
    short_window_minutes: int = 5
    medium_window_minutes: int = 30
    long_window_hours: int = 24

    # Seuils d'alerte
    response_time_warning: float = 3.0  # secondes
    response_time_critical: float = 10.0
    relevance_score_warning: float = 3.0  # score 1-5
    relevance_score_critical: float = 2.5
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.15  # 15%
    vague_query_rate_warning: float = 0.3  # 30%

    # Collecte de données
    enable_automatic_feedback: bool = True
    enable_performance_tracking: bool = True
    enable_quality_assessment: bool = True

    # Stockage
    data_retention_days: int = 30
    alert_retention_days: int = 7

    # Notifications
    enable_email_alerts: bool = False
    enable_webhook_alerts: bool = False
    webhook_url: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collecteur de métriques en temps réel"""

    def __init__(self, config: MonitoringConfig):
        self.config = config

        # Stockage des métriques
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.alerts = deque(maxlen=1000)

        # Threading pour collecte asynchrone
        self._collection_thread = None
        self._stop_collection = threading.Event()
        self._metrics_queue = queue.Queue()

        # Callbacks d'alerte
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def start_collection(self):
        """Démarre la collecte de métriques en arrière-plan"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(target=self._collection_worker)
            self._collection_thread.start()
            logger.info("Metrics collection started")

    def stop_collection(self):
        """Arrête la collecte de métriques"""
        self._stop_collection.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")

    def record_metric(self, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """Enregistre une métrique"""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata or {}
        )

        # Ajouter à la queue pour traitement asynchrone
        self._metrics_queue.put((metric_type, metric_point))

    def _collection_worker(self):
        """Worker thread pour la collecte de métriques"""
        while not self._stop_collection.is_set():
            try:
                # Traiter les métriques en attente
                while not self._metrics_queue.empty():
                    metric_type, metric_point = self._metrics_queue.get_nowait()
                    self._process_metric(metric_type, metric_point)

                # Nettoyage périodique
                self._cleanup_old_data()

                # Vérification des seuils d'alerte
                self._check_alert_thresholds()

                time.sleep(1)  # Pause entre les cycles

            except Exception as e:
                logger.error(f"Error in metrics collection worker: {e}")
                time.sleep(5)

    def _process_metric(self, metric_type: MetricType, metric_point: MetricPoint):
        """Traite une métrique"""
        self.metrics_buffer[metric_type].append(metric_point)

        # Log des métriques importantes
        if metric_type == MetricType.ERROR_RATE and metric_point.value > 0:
            logger.warning(f"Error detected: {metric_point.metadata}")
        elif metric_type == MetricType.RESPONSE_TIME and metric_point.value > self.config.response_time_warning:
            logger.warning(f"Slow response: {metric_point.value:.2f}s")

    def _cleanup_old_data(self):
        """Nettoie les anciennes données"""
        cutoff_time = datetime.now() - timedelta(days=self.config.data_retention_days)

        for metric_type in self.metrics_buffer:
            buffer = self.metrics_buffer[metric_type]
            while buffer and buffer[0].timestamp < cutoff_time:
                buffer.popleft()

        # Nettoyer les anciennes alertes
        alert_cutoff = datetime.now() - timedelta(days=self.config.alert_retention_days)
        while self.alerts and self.alerts[0].timestamp < alert_cutoff:
            self.alerts.popleft()

    def _check_alert_thresholds(self):
        """Vérifie les seuils d'alerte"""
        now = datetime.now()

        # Fenêtre courte (5 minutes)
        short_window_start = now - timedelta(minutes=self.config.short_window_minutes)

        # Vérifier temps de réponse
        response_times = self._get_metrics_in_window(MetricType.RESPONSE_TIME, short_window_start, now)
        if response_times:
            avg_response_time = statistics.mean([m.value for m in response_times])
            self._check_response_time_threshold(avg_response_time)

        # Vérifier taux d'erreur
        errors = self._get_metrics_in_window(MetricType.ERROR_RATE, short_window_start, now)
        if errors:
            avg_error_rate = statistics.mean([m.value for m in errors])
            self._check_error_rate_threshold(avg_error_rate)

        # Vérifier score de pertinence
        relevance_scores = self._get_metrics_in_window(MetricType.RELEVANCE_SCORE, short_window_start, now)
        if relevance_scores:
            avg_relevance = statistics.mean([m.value for m in relevance_scores])
            self._check_relevance_threshold(avg_relevance)

    def _get_metrics_in_window(self, metric_type: MetricType, start_time: datetime, end_time: datetime) -> List[MetricPoint]:
        """Récupère les métriques dans une fenêtre de temps"""
        buffer = self.metrics_buffer[metric_type]
        return [m for m in buffer if start_time <= m.timestamp <= end_time]

    def _check_response_time_threshold(self, avg_response_time: float):
        """Vérifie les seuils de temps de réponse"""
        if avg_response_time >= self.config.response_time_critical:
            self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.RESPONSE_TIME,
                f"Critical response time: {avg_response_time:.2f}s (threshold: {self.config.response_time_critical}s)",
                avg_response_time,
                self.config.response_time_critical
            )
        elif avg_response_time >= self.config.response_time_warning:
            self._create_alert(
                AlertLevel.WARNING,
                MetricType.RESPONSE_TIME,
                f"High response time: {avg_response_time:.2f}s (threshold: {self.config.response_time_warning}s)",
                avg_response_time,
                self.config.response_time_warning
            )

    def _check_error_rate_threshold(self, avg_error_rate: float):
        """Vérifie les seuils de taux d'erreur"""
        if avg_error_rate >= self.config.error_rate_critical:
            self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.ERROR_RATE,
                f"Critical error rate: {avg_error_rate:.1%} (threshold: {self.config.error_rate_critical:.1%})",
                avg_error_rate,
                self.config.error_rate_critical
            )
        elif avg_error_rate >= self.config.error_rate_warning:
            self._create_alert(
                AlertLevel.WARNING,
                MetricType.ERROR_RATE,
                f"High error rate: {avg_error_rate:.1%} (threshold: {self.config.error_rate_warning:.1%})",
                avg_error_rate,
                self.config.error_rate_warning
            )

    def _check_relevance_threshold(self, avg_relevance: float):
        """Vérifie les seuils de pertinence"""
        if avg_relevance <= self.config.relevance_score_critical:
            self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.RELEVANCE_SCORE,
                f"Low relevance score: {avg_relevance:.2f} (threshold: {self.config.relevance_score_critical})",
                avg_relevance,
                self.config.relevance_score_critical
            )
        elif avg_relevance <= self.config.relevance_score_warning:
            self._create_alert(
                AlertLevel.WARNING,
                MetricType.RELEVANCE_SCORE,
                f"Below average relevance: {avg_relevance:.2f} (threshold: {self.config.relevance_score_warning})",
                avg_relevance,
                self.config.relevance_score_warning
            )

    def _create_alert(self, level: AlertLevel, metric_type: MetricType, message: str, current_value: float, threshold: float):
        """Crée une nouvelle alerte"""
        alert = Alert(
            level=level,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now(),
            current_value=current_value,
            threshold=threshold
        )

        self.alerts.append(alert)
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else logging.WARNING,
            f"ALERT [{level.value.upper()}]: {message}"
        )

        # Notifier les callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Génère un résumé des métriques"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        summary = {
            "time_window": f"{hours}h",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "metrics": {}
        }

        for metric_type in MetricType:
            metrics = self._get_metrics_in_window(metric_type, start_time, end_time)

            if metrics:
                values = [m.value for m in metrics]
                summary["metrics"][metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0
                }
            else:
                summary["metrics"][metric_type.value] = {
                    "count": 0,
                    "mean": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "stdev": 0
                }

        return summary

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Ajoute un callback d'alerte"""
        self.alert_callbacks.append(callback)


class FeedbackCollector:
    """Collecteur de feedback utilisateur"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.feedback_buffer = deque(maxlen=10000)

        # Analyse automatique des patterns
        self.pattern_analyzer = FeedbackPatternAnalyzer()

    def collect_feedback(self, feedback: UserFeedback):
        """Collecte un feedback utilisateur"""
        self.feedback_buffer.append(feedback)

        # Analyse automatique si activée
        if self.config.enable_automatic_feedback:
            self._analyze_feedback_automatically(feedback)

        logger.info(f"Feedback collected: overall score {feedback.get_overall_score():.2f}")

    def _analyze_feedback_automatically(self, feedback: UserFeedback):
        """Analyse automatique du feedback"""
        # Détecter les patterns problématiques
        if feedback.get_overall_score() < 2.5:
            self._flag_poor_performance(feedback)

        # Analyser la corrélation avec la vague des requêtes
        if feedback.vagueness_score and feedback.vagueness_score > 0.7:
            self._analyze_vague_query_performance(feedback)

    def _flag_poor_performance(self, feedback: UserFeedback):
        """Signale une performance médiocre"""
        logger.warning(f"Poor feedback received for query: '{feedback.query[:50]}...'")

        # Ici on pourrait déclencher des actions correctives
        # - Re-indexation
        # - Ajustement des poids
        # - Fine-tuning du modèle

    def _analyze_vague_query_performance(self, feedback: UserFeedback):
        """Analyse la performance sur requêtes vagues"""
        if feedback.get_overall_score() < 3.0:
            logger.info(f"Vague query with poor score: {feedback.vagueness_score:.2f} -> {feedback.get_overall_score():.2f}")

    def get_feedback_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Génère des analytics de feedback"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_feedback = [f for f in self.feedback_buffer if f.timestamp >= cutoff_time]

        if not recent_feedback:
            return {"message": "No feedback in the specified time window"}

        # Calculs statistiques
        overall_scores = [f.get_overall_score() for f in recent_feedback]
        relevance_scores = [f.relevance_score for f in recent_feedback]
        helpfulness_scores = [f.helpfulness_score for f in recent_feedback]
        clarity_scores = [f.clarity_score for f in recent_feedback]

        # Corrélation avec vague queries
        vague_feedback = [f for f in recent_feedback if f.vagueness_score and f.vagueness_score > 0.5]
        precise_feedback = [f for f in recent_feedback if f.vagueness_score and f.vagueness_score <= 0.5]

        analytics = {
            "time_window": f"{hours}h",
            "total_feedback": len(recent_feedback),
            "overall_satisfaction": {
                "mean": statistics.mean(overall_scores),
                "median": statistics.median(overall_scores),
                "stdev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
            },
            "dimension_scores": {
                "relevance": statistics.mean(relevance_scores),
                "helpfulness": statistics.mean(helpfulness_scores),
                "clarity": statistics.mean(clarity_scores)
            },
            "vague_vs_precise": {
                "vague_queries": {
                    "count": len(vague_feedback),
                    "avg_score": statistics.mean([f.get_overall_score() for f in vague_feedback]) if vague_feedback else 0
                },
                "precise_queries": {
                    "count": len(precise_feedback),
                    "avg_score": statistics.mean([f.get_overall_score() for f in precise_feedback]) if precise_feedback else 0
                }
            }
        }

        return analytics


class FeedbackPatternAnalyzer:
    """Analyseur de patterns dans le feedback"""

    def __init__(self):
        self.common_issues = defaultdict(int)
        self.improvement_suggestions = defaultdict(int)

    def analyze_feedback_trends(self, feedback_list: List[UserFeedback]) -> Dict[str, Any]:
        """Analyse les tendances dans le feedback"""

        # Analyser les améliorations suggérées
        all_suggestions = []
        for feedback in feedback_list:
            all_suggestions.extend(feedback.improvements_suggested)

        suggestion_counts = Counter(all_suggestions)

        # Identifier les patterns temporels
        hourly_scores = defaultdict(list)
        for feedback in feedback_list:
            hour = feedback.timestamp.hour
            hourly_scores[hour].append(feedback.get_overall_score())

        # Patterns par type de requête (vague vs précise)
        vague_scores = []
        precise_scores = []

        for feedback in feedback_list:
            if feedback.vagueness_score:
                if feedback.vagueness_score > 0.5:
                    vague_scores.append(feedback.get_overall_score())
                else:
                    precise_scores.append(feedback.get_overall_score())

        analysis = {
            "top_improvement_suggestions": dict(suggestion_counts.most_common(10)),
            "hourly_performance": {
                hour: statistics.mean(scores) for hour, scores in hourly_scores.items()
            },
            "query_type_performance": {
                "vague_queries": {
                    "count": len(vague_scores),
                    "avg_score": statistics.mean(vague_scores) if vague_scores else 0
                },
                "precise_queries": {
                    "count": len(precise_scores),
                    "avg_score": statistics.mean(precise_scores) if precise_scores else 0
                }
            }
        }

        return analysis


class ProductionMonitor:
    """Système complet de monitoring production"""

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()

        # Composants principaux
        self.metrics_collector = MetricsCollector(self.config)
        self.feedback_collector = FeedbackCollector(self.config)

        # État du système
        self.is_running = False

        # Callbacks pour actions automatiques
        self.optimization_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # Setup des alertes
        self._setup_alert_handlers()

    def _setup_alert_handlers(self):
        """Configure les gestionnaires d'alertes"""

        def alert_handler(alert: Alert):
            """Gestionnaire d'alerte par défaut"""
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                # Déclencher des actions correctives automatiques
                self._trigger_automatic_corrections(alert)

            # Notifier si configuré
            if self.config.enable_webhook_alerts and self.config.webhook_url:
                self._send_webhook_alert(alert)

        self.metrics_collector.add_alert_callback(alert_handler)

    def start_monitoring(self):
        """Démarre le monitoring complet"""
        if not self.is_running:
            self.metrics_collector.start_collection()
            self.is_running = True
            logger.info("Production monitoring started")

    def stop_monitoring(self):
        """Arrête le monitoring"""
        if self.is_running:
            self.metrics_collector.stop_collection()
            self.is_running = False
            logger.info("Production monitoring stopped")

    def record_query_performance(self,
                                query: str,
                                response_time: float,
                                vagueness_score: float,
                                context_quality: float = None,
                                error_occurred: bool = False):
        """Enregistre les performances d'une requête"""

        # Métriques de base
        self.metrics_collector.record_metric(
            MetricType.RESPONSE_TIME,
            response_time,
            {"query_snippet": query[:100], "vagueness_score": vagueness_score}
        )

        self.metrics_collector.record_metric(
            MetricType.VAGUE_QUERY_RATE,
            1.0 if vagueness_score > 0.5 else 0.0,
            {"vagueness_score": vagueness_score}
        )

        if context_quality is not None:
            self.metrics_collector.record_metric(
                MetricType.CONTEXT_QUALITY,
                context_quality,
                {"query_snippet": query[:100]}
            )

        if error_occurred:
            self.metrics_collector.record_metric(
                MetricType.ERROR_RATE,
                1.0,
                {"query_snippet": query[:100], "error": True}
            )
        else:
            self.metrics_collector.record_metric(
                MetricType.ERROR_RATE,
                0.0,
                {"query_snippet": query[:100], "error": False}
            )

    def collect_user_feedback(self,
                            query: str,
                            response: str,
                            relevance_score: int,
                            helpfulness_score: int,
                            clarity_score: int,
                            user_id: str = None,
                            response_time: float = None,
                            vagueness_score: float = None,
                            improvements_suggested: List[str] = None):
        """Collecte le feedback utilisateur"""

        feedback = UserFeedback(
            query=query,
            response=response,
            relevance_score=relevance_score,
            helpfulness_score=helpfulness_score,
            clarity_score=clarity_score,
            timestamp=datetime.now(),
            user_id=user_id,
            response_time=response_time,
            vagueness_score=vagueness_score,
            improvements_suggested=improvements_suggested or []
        )

        self.feedback_collector.collect_feedback(feedback)

        # Enregistrer le score comme métrique
        self.metrics_collector.record_metric(
            MetricType.USER_SATISFACTION,
            feedback.get_overall_score(),
            {"user_id": user_id, "vagueness_score": vagueness_score}
        )

        self.metrics_collector.record_metric(
            MetricType.RELEVANCE_SCORE,
            relevance_score,
            {"query_snippet": query[:100]}
        )

    def _trigger_automatic_corrections(self, alert: Alert):
        """Déclenche des corrections automatiques"""

        logger.info(f"Triggering automatic corrections for {alert.metric_type.value}")

        correction_actions = {
            MetricType.RESPONSE_TIME: self._optimize_response_time,
            MetricType.RELEVANCE_SCORE: self._optimize_relevance,
            MetricType.ERROR_RATE: self._investigate_errors,
            MetricType.VAGUE_QUERY_RATE: self._optimize_vague_handling
        }

        action = correction_actions.get(alert.metric_type)
        if action:
            try:
                action(alert)
            except Exception as e:
                logger.error(f"Failed to execute automatic correction: {e}")

    def _optimize_response_time(self, alert: Alert):
        """Optimise le temps de réponse"""
        optimizations = {
            "reduce_context_length": True,
            "enable_embedding_cache": True,
            "optimize_retrieval_k": True
        }

        for callback in self.optimization_callbacks:
            callback({"type": "response_time", "optimizations": optimizations})

    def _optimize_relevance(self, alert: Alert):
        """Optimise la pertinence"""
        optimizations = {
            "adjust_embedding_weights": True,
            "retrain_on_recent_feedback": True,
            "increase_context_diversity": True
        }

        for callback in self.optimization_callbacks:
            callback({"type": "relevance", "optimizations": optimizations})

    def _investigate_errors(self, alert: Alert):
        """Investigate les erreurs"""
        investigation = {
            "analyze_error_patterns": True,
            "check_system_resources": True,
            "validate_model_health": True
        }

        for callback in self.optimization_callbacks:
            callback({"type": "errors", "investigation": investigation})

    def _optimize_vague_handling(self, alert: Alert):
        """Optimise la gestion des requêtes vagues"""
        optimizations = {
            "improve_query_expansion": True,
            "enhance_context_enrichment": True,
            "adjust_vague_detection_threshold": True
        }

        for callback in self.optimization_callbacks:
            callback({"type": "vague_queries", "optimizations": optimizations})

    def _send_webhook_alert(self, alert: Alert):
        """Envoie une alerte via webhook"""
        try:
            import requests

            payload = {
                "alert": alert.to_dict(),
                "system": "rag_chunk_lab",
                "environment": "production"
            }

            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Webhook alert sent successfully")
            else:
                logger.warning(f"Webhook alert failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Retourne l'état de santé du système"""

        metrics_summary = self.metrics_collector.get_metrics_summary(hours=1)
        feedback_analytics = self.feedback_collector.get_feedback_analytics(hours=24)

        # Calculer un score de santé global
        health_score = self._calculate_health_score(metrics_summary, feedback_analytics)

        # Récupérer les alertes récentes
        recent_alerts = [alert for alert in self.metrics_collector.alerts
                        if alert.timestamp >= datetime.now() - timedelta(hours=1)]

        health_status = {
            "overall_health_score": health_score,
            "status": self._get_health_status_label(health_score),
            "metrics_summary": metrics_summary,
            "feedback_analytics": feedback_analytics,
            "recent_alerts": [alert.to_dict() for alert in recent_alerts],
            "monitoring_active": self.is_running,
            "last_updated": datetime.now().isoformat()
        }

        return health_status

    def _calculate_health_score(self, metrics_summary: Dict, feedback_analytics: Dict) -> float:
        """Calcule un score de santé global (0-100)"""

        score = 100.0

        # Pénalités basées sur les métriques
        metrics = metrics_summary.get("metrics", {})

        # Temps de réponse
        response_time = metrics.get("response_time", {}).get("mean", 0)
        if response_time > self.config.response_time_critical:
            score -= 30
        elif response_time > self.config.response_time_warning:
            score -= 15

        # Taux d'erreur
        error_rate = metrics.get("error_rate", {}).get("mean", 0)
        if error_rate > self.config.error_rate_critical:
            score -= 40
        elif error_rate > self.config.error_rate_warning:
            score -= 20

        # Score de pertinence
        relevance = metrics.get("relevance_score", {}).get("mean", 5)
        if relevance < self.config.relevance_score_critical:
            score -= 25
        elif relevance < self.config.relevance_score_warning:
            score -= 10

        # Satisfaction utilisateur
        satisfaction = feedback_analytics.get("overall_satisfaction", {}).get("mean", 3)
        if satisfaction < 2.5:
            score -= 20
        elif satisfaction < 3.5:
            score -= 10

        return max(0, min(100, score))

    def _get_health_status_label(self, score: float) -> str:
        """Retourne un label d'état basé sur le score"""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 50:
            return "poor"
        else:
            return "critical"

    def add_optimization_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Ajoute un callback d'optimisation automatique"""
        self.optimization_callbacks.append(callback)

    def export_monitoring_data(self, output_path: str, hours: int = 24):
        """Exporte les données de monitoring"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Collecter toutes les données
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_window_hours": hours,
            "system_health": self.get_system_health(),
            "metrics": {},
            "feedback": [],
            "alerts": []
        }

        # Exporter les métriques
        for metric_type in MetricType:
            metrics = [m for m in self.metrics_collector.metrics_buffer[metric_type]
                      if m.timestamp >= cutoff_time]
            export_data["metrics"][metric_type.value] = [m.to_dict() for m in metrics]

        # Exporter le feedback
        feedback = [f for f in self.feedback_collector.feedback_buffer
                   if f.timestamp >= cutoff_time]
        export_data["feedback"] = [f.to_dict() for f in feedback]

        # Exporter les alertes
        alerts = [alert for alert in self.metrics_collector.alerts
                 if alert.timestamp >= cutoff_time]
        export_data["alerts"] = [alert.to_dict() for alert in alerts]

        # Sauvegarder
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Monitoring data exported to {output_path}")


# Fonctions d'intégration pour RAG Chunk Lab
def create_production_monitor(domain: str = "general",
                            config_overrides: Dict[str, Any] = None) -> ProductionMonitor:
    """
    Crée un moniteur de production configuré

    Args:
        domain: Domaine spécialisé
        config_overrides: Overrides de configuration

    Returns:
        Moniteur de production configuré
    """

    config = MonitoringConfig()

    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ProductionMonitor(config)


def setup_automatic_optimization(monitor: ProductionMonitor,
                                hybrid_system=None,
                                vague_optimizer=None):
    """
    Configure l'optimisation automatique basée sur le monitoring

    Args:
        monitor: Moniteur de production
        hybrid_system: Système d'embeddings hybrides (optionnel)
        vague_optimizer: Optimiseur de requêtes vagues (optionnel)
    """

    def optimization_callback(optimization_request: Dict[str, Any]):
        """Callback d'optimisation automatique"""

        opt_type = optimization_request.get("type")
        logger.info(f"Executing automatic optimization: {opt_type}")

        if opt_type == "response_time" and hybrid_system:
            # Optimiser les poids pour privilégier la vitesse
            hybrid_system.fusion_weights = {"dense": 0.8, "sparse": 0.2}

        elif opt_type == "relevance" and hybrid_system:
            # Optimiser pour la pertinence
            hybrid_system.fusion_weights = {"dense": 0.6, "sparse": 0.4}

        elif opt_type == "vague_queries" and vague_optimizer:
            # Ajuster les seuils de détection
            # vague_optimizer.adjust_detection_threshold()
            pass

        logger.info(f"Optimization completed for {opt_type}")

    monitor.add_optimization_callback(optimization_callback)