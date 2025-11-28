# -*- coding: utf-8 -*-
"""
check_paths.py
プロジェクトのパスとファイルを確認
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR

print("="*70)
print("Path and File Checker")
print("="*70)

print(f"\nScript Directory: {SCRIPT_DIR}")
print(f"Project Directory: {PROJECT_DIR}")

# 確認するディレクトリとファイル
checks = {
    'Data': {
        'path': PROJECT_DIR / "data" / "by_eye_pattern",
        'files': "graph_data_*.pkl",
        'min_count': 1
    },
    'Models': {
        'path': PROJECT_DIR / "models" / "by_eye_pattern",
        'files': "best_model_*.pt",
        'min_count': 1
    },
    'Results': {
        'path': PROJECT_DIR / "results" / "by_eye_pattern",
        'files': "*.csv",
        'min_count': 0
    },
    'Importance Maps': {
        'path': PROJECT_DIR / "results" / "importance_maps_by_eye_pattern",
        'files': "*.pkl",
        'min_count': 0
    },
    'Visualizations': {
        'path': PROJECT_DIR / "visualizations" / "importance_maps_by_eye_pattern",
        'files': "*.png",
        'min_count': 0
    }
}

all_ok = True

for name, check in checks.items():
    print(f"\n{name}:")
    path = check['path']
    
    if not path.exists():
        print(f"  ✗ Directory not found: {path}")
        all_ok = False
        continue
    
    print(f"  ✓ Directory exists: {path}")
    
    # ファイル数を確認
    files = list(path.glob(check['files']))
    print(f"  Files ({check['files']}): {len(files)}")
    
    if len(files) < check['min_count']:
        print(f"  ⚠️  Expected at least {check['min_count']} files")
        all_ok = False
    else:
        for f in files[:5]:  # 最初の5件のみ表示
            print(f"    - {f.name}")
        if len(files) > 5:
            print(f"    ... and {len(files) - 5} more")

print("\n" + "="*70)

if all_ok:
    print("✓ All checks passed!")
else:
    print("⚠️  Some checks failed. Please verify your setup.")
    print("\nQuick fix:")
    print("  1. Make sure you're in the correct directory (gnn_project6)")
    print("  2. Run: python create_graph_by_eye_pattern_revised.py")
    print("  3. Run: python train_by_eye_pattern_revised.py")

print("="*70)