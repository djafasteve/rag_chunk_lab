#!/usr/bin/env python3
"""
Script principal pour exécuter les tests RAG Chunk Lab
Usage: python run_tests.py [category]
"""
import sys
import os

# Ajouter le chemin des tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from tests.test_runner import run_test_suite, run_specific_test_category

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_pattern = sys.argv[1] if sys.argv[1] != 'all' else None
        success = run_test_suite(test_pattern)
    else:
        success = run_specific_test_category()

    print(f"\n{'🎉 Tous les tests ont réussi!' if success else '❌ Certains tests ont échoué.'}")
    sys.exit(0 if success else 1)