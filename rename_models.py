# -*- coding: utf-8 -*-
"""
rename_models.py
モデルファイル名を正しい形式にリネーム
"""

from pathlib import Path
import shutil

print("="*70)
print("Model File Renamer")
print("="*70)

model_dir = Path("models/by_eye_pattern")

if not model_dir.exists():
    print(f"\n❌ Directory not found: {model_dir}")
    exit(1)

print(f"\nModel directory: {model_dir}")

# リネームマッピング
rename_map = {
    # gnn_project5の形式 → gnn_project6の形式
    'best_model_Pattern10-2_LeftEye.pt': 'best_model_Left_Pattern10-2.pt',
    'best_model_Pattern10-2_RightEye.pt': 'best_model_Right_Pattern10-2.pt',
    'best_model_Pattern24-2_LeftEye.pt': 'best_model_Left_Pattern24-2.pt',
    'best_model_Pattern24-2_RightEye.pt': 'best_model_Right_Pattern24-2.pt',
    'best_model_Pattern30-2_LeftEye.pt': 'best_model_Left_Pattern30-2.pt',
    'best_model_Pattern30-2_RightEye.pt': 'best_model_Right_Pattern30-2.pt',
}

print("\nRenaming files...")
renamed_count = 0
skipped_count = 0

for old_name, new_name in rename_map.items():
    old_path = model_dir / old_name
    new_path = model_dir / new_name
    
    if not old_path.exists():
        print(f"  - {old_name} → {new_name} (source not found, skipping)")
        skipped_count += 1
        continue
    
    if new_path.exists():
        print(f"  - {old_name} → {new_name} (target already exists, skipping)")
        skipped_count += 1
        continue
    
    try:
        shutil.move(str(old_path), str(new_path))
        print(f"  ✓ {old_name} → {new_name}")
        renamed_count += 1
    except Exception as e:
        print(f"  ✗ {old_name} → {new_name} (Error: {e})")

print(f"\n{'='*70}")
print(f"Renamed: {renamed_count} files")
print(f"Skipped: {skipped_count} files")
print(f"{'='*70}")

# 最終確認
print("\nFinal check - Expected model files:")
expected_files = [
    'best_model_Left_Pattern10-2.pt',
    'best_model_Left_Pattern24-2.pt',
    'best_model_Left_Pattern30-2.pt',
    'best_model_Right_Pattern10-2.pt',
    'best_model_Right_Pattern24-2.pt',
    'best_model_Right_Pattern30-2.pt',
]

all_found = True
for filename in expected_files:
    filepath = model_dir / filename
    if filepath.exists():
        size = filepath.stat().st_size / 1024 / 1024  # MB
        print(f"  ✓ {filename} ({size:.1f} MB)")
    else:
        print(f"  ✗ {filename} (NOT FOUND)")
        all_found = False

print(f"\n{'='*70}")
if all_found:
    print("✓ All expected model files are present!")
    print("\nYou can now run:")
    print("  python compute_importance_simple.py")
else:
    print("⚠️  Some model files are missing")
    print("\nYou may need to:")
    print("  python train_by_eye_pattern_revised.py")

print(f"{'='*70}")  