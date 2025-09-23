#!/usr/bin/env python3
"""
Script pour corriger les imports relatifs internes après réorganisation
"""

import os
import re
from pathlib import Path

def fix_relative_imports(file_path):
    """Corrige les imports relatifs dans un fichier"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Corrections spécifiques par répertoire
        if '/core/' in str(file_path):
            # Dans core/, les imports vers utils doivent être ..utils
            content = re.sub(r'from \.utils\.', 'from ..utils.', content)
            content = re.sub(r'from \.core\.', 'from ..core.', content)

        elif '/evaluation/' in str(file_path):
            # Dans evaluation/, les imports vers utils doivent être ..utils
            content = re.sub(r'from \.utils\.', 'from ..utils.', content)
            content = re.sub(r'from \.core\.', 'from ..core.', content)

        elif '/vague_query/' in str(file_path):
            # Dans vague_query/, corriger les imports vers d'autres modules
            content = re.sub(r'from \.vague_query\.', 'from .', content)
            content = re.sub(r'from \.core\.', 'from ..core.', content)
            content = re.sub(r'from \.utils\.', 'from ..utils.', content)
            content = re.sub(r'from \.evaluation\.', 'from ..evaluation.', content)

        elif '/utils/' in str(file_path):
            # Dans utils/, les imports vers core doivent être ..core
            content = re.sub(r'from \.core\.', 'from ..core.', content)

        # Sauvegarder seulement si des changements ont été faits
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⏭️  No changes: {file_path}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fonction principale"""
    print("🔧 Correction des imports relatifs...")

    project_root = Path(__file__).parent
    rag_chunk_lab_dir = project_root / "rag_chunk_lab"

    python_files = []
    for root, dirs, files in os.walk(rag_chunk_lab_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    print(f"📁 Fichiers Python trouvés: {len(python_files)}")

    fixed_files = 0
    for file_path in python_files:
        if fix_relative_imports(file_path):
            fixed_files += 1

    print(f"\n🎉 Correction terminée!")
    print(f"📊 Fichiers corrigés: {fixed_files}/{len(python_files)}")

if __name__ == "__main__":
    main()