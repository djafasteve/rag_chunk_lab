import time
import psutil
import functools
from typing import Dict, Any

class PerformanceMonitor:
    """Moniteur de performance pour mesurer l'impact des optimisations"""

    def __init__(self):
        self.metrics = {}

    def monitor_function(self, func_name: str = None):
        """Décorateur pour monitorer les performances d'une fonction"""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss

                    duration = end_time - start_time
                    memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB

                    self.record_metric(name, {
                        'duration_seconds': duration,
                        'memory_delta_mb': memory_delta,
                        'peak_memory_mb': end_memory / 1024 / 1024,
                        'success': success,
                        'error': error
                    })

                    if success:
                        print(f"⚡ {name}: {duration:.2f}s, "
                              f"Mém: {memory_delta:+.1f}MB (pic: {end_memory/1024/1024:.1f}MB)")

                return result
            return wrapper
        return decorator

    def record_metric(self, name: str, metrics: Dict[str, Any]):
        """Enregistre des métriques pour une opération"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(metrics)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Retourne un résumé des métriques collectées"""
        summary = {}

        for name, records in self.metrics.items():
            if not records:
                continue

            successful_records = [r for r in records if r['success']]
            if not successful_records:
                continue

            durations = [r['duration_seconds'] for r in successful_records]
            memory_deltas = [r['memory_delta_mb'] for r in successful_records]

            summary[name] = {
                'count': len(successful_records),
                'avg_duration_seconds': sum(durations) / len(durations),
                'total_duration_seconds': sum(durations),
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'max_duration_seconds': max(durations),
                'min_duration_seconds': min(durations)
            }

        return summary

    def print_summary(self):
        """Affiche un résumé des performances"""
        summary = self.get_summary()

        print("\n📊 RÉSUMÉ DES PERFORMANCES")
        print("=" * 60)

        for name, stats in summary.items():
            print(f"\n🔧 {name}")
            print(f"   Appels: {stats['count']}")
            print(f"   Durée moy.: {stats['avg_duration_seconds']:.2f}s")
            print(f"   Durée tot.: {stats['total_duration_seconds']:.2f}s")
            print(f"   Mémoire moy.: {stats['avg_memory_delta_mb']:+.1f}MB")
            print(f"   Range: {stats['min_duration_seconds']:.2f}s - {stats['max_duration_seconds']:.2f}s")

# Instance globale du moniteur
monitor = PerformanceMonitor()

# Décorateurs pratiques
def monitor_performance(func_name: str = None):
    """Décorateur pratique pour monitorer les performances"""
    return monitor.monitor_function(func_name)

def print_performance_summary():
    """Affiche le résumé des performances"""
    monitor.print_summary()