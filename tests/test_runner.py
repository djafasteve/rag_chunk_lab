#!/usr/bin/env python3
"""
Script pour exécuter tous les tests RAG Chunk Lab avec des rapports détaillés
"""
import unittest
import sys
import os
import time
from io import StringIO

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColoredTextTestResult(unittest.TextTestResult):
    """Résultats de test avec coloration pour une meilleure lisibilité"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_count = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.showAll:
            self.stream.write("\033[92m✓ PASS\033[0m\n")
        elif self.dots:
            self.stream.write("\033[92m.\033[0m")

    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.write("\033[91m✗ ERROR\033[0m\n")
        elif self.dots:
            self.stream.write("\033[91mE\033[0m")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.write("\033[91m✗ FAIL\033[0m\n")
        elif self.dots:
            self.stream.write("\033[91mF\033[0m")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.write(f"\033[93m- SKIP ({reason})\033[0m\n")
        elif self.dots:
            self.stream.write("\033[93ms\033[0m")


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Runner de test avec coloration"""
    resultclass = ColoredTextTestResult


def run_test_suite(test_pattern=None, verbosity=2):
    """Exécute la suite de tests avec rapport détaillé"""

    print("\033[96m" + "="*80)
    print("🧪 RAG CHUNK LAB v2.0 - SUITE DE TESTS OPTIMISÉS")
    print("="*80 + "\033[0m")

    # Découvrir tous les tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    if test_pattern:
        pattern = f"test_{test_pattern}.py"
    else:
        pattern = "test_*.py"

    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)

    # Compter les tests
    test_count = suite.countTestCases()
    print(f"\n📊 Tests découverts: {test_count}")

    # Exécuter les tests
    print("\n🚀 Exécution des tests...\n")

    start_time = time.time()
    runner = ColoredTextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    execution_time = time.time() - start_time

    # Rapport final
    print("\n" + "\033[96m" + "="*80)
    print("📋 RAPPORT FINAL")
    print("="*80 + "\033[0m")

    print(f"⏱️  Temps d'exécution: {execution_time:.2f} secondes")
    print(f"📊 Tests exécutés: {result.testsRun}")
    print(f"\033[92m✓ Succès: {result.success_count}\033[0m")

    if result.failures:
        print(f"\033[91m✗ Échecs: {len(result.failures)}\033[0m")
    if result.errors:
        print(f"\033[91m⚠ Erreurs: {len(result.errors)}\033[0m")
    if result.skipped:
        print(f"\033[93m- Ignorés: {len(result.skipped)}\033[0m")

    # Taux de réussite
    if result.testsRun > 0:
        success_rate = (result.success_count / result.testsRun) * 100
        if success_rate == 100:
            print(f"\n\033[92m🎉 Taux de réussite: {success_rate:.1f}% - PARFAIT!\033[0m")
        elif success_rate >= 90:
            print(f"\n\033[92m✅ Taux de réussite: {success_rate:.1f}% - EXCELLENT\033[0m")
        elif success_rate >= 75:
            print(f"\n\033[93m⚠️  Taux de réussite: {success_rate:.1f}% - À AMÉLIORER\033[0m")
        else:
            print(f"\n\033[91m❌ Taux de réussite: {success_rate:.1f}% - PROBLÈMES DÉTECTÉS\033[0m")

    # Détails des échecs
    if result.failures or result.errors:
        print(f"\n\033[91m📋 DÉTAILS DES PROBLÈMES:\033[0m")

        for test, traceback in result.failures:
            print(f"\n\033[91m❌ ÉCHEC: {test}\033[0m")
            print(f"   {traceback.split('AssertionError:')[-1].strip()}")

        for test, traceback in result.errors:
            print(f"\n\033[91m⚠️  ERREUR: {test}\033[0m")
            print(f"   {traceback.split('Error:')[-1].strip()}")

    print("\n" + "\033[96m" + "="*80 + "\033[0m")

    return result.wasSuccessful()


def run_specific_test_category():
    """Interface pour exécuter des catégories spécifiques de tests"""

    categories = {
        '1': ('optimizations', 'Tests des optimisations générales'),
        '2': ('chunkers', 'Tests du module chunkers'),
        '3': ('indexing', 'Tests du module indexing'),
        '4': ('generation', 'Tests du module generation'),
        '5': ('performance', 'Tests de performance'),
        '6': (None, 'Tous les tests')
    }

    print("\n🎯 Choisissez une catégorie de tests:")
    for key, (pattern, description) in categories.items():
        print(f"  {key}. {description}")

    choice = input("\nVotre choix (1-6): ").strip()

    if choice in categories:
        pattern, description = categories[choice]
        print(f"\n🚀 Exécution: {description}")
        return run_test_suite(pattern)
    else:
        print("❌ Choix invalide")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Mode ligne de commande
        test_pattern = sys.argv[1] if sys.argv[1] != 'all' else None
        success = run_test_suite(test_pattern)
    else:
        # Mode interactif
        success = run_specific_test_category()

    sys.exit(0 if success else 1)