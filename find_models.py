# -*- coding: utf-8 -*-
"""
find_models.py
モデルファイルの場所を探す
"""

from pathlib import Path
import shutil

print("="*70)
print("Model File Finder")
print("="*70)

# 現在のディレクトリから検索
current_dir = Path.cwd()
print(f"\nSearching from: {current_dir}")

# .ptファイルを探す
print("\nSearching for *.pt files...")
pt_files = list(current_dir.rglob("best_model_*.pt"))

if len(pt_files) == 0:
    print("\n❌ No model files (*.pt) found in current directory tree")
    
    # 親ディレクトリも検索
    parent_dir = current_dir.parent
    print(f"\nSearching in parent directory: {parent_dir}")
    pt_files = list(parent_dir.rglob("best_model_*.pt"))

if len(pt_files) == 0:
    print("\n❌ No model files found anywhere")
    print("\nPossible reasons:")
    print("  1. Training didn't complete successfully")
    print("  2. Model saving failed due to permissions")
    print("  3. Models were saved in unexpected location")
    
    print("\nSolution:")
    print("  Re-run training: python train_by_eye_pattern_revised.py")
else:
    print(f"\n✓ Found {len(pt_files)} model file(s):")
    for f in pt_files:
        print(f"  - {f}")
    
    # 正しい場所にコピー
    target_dir = current_dir / "models" / "by_eye_pattern"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget directory: {target_dir}")
    
    if not target_dir.exists():
        print("  Creating directory...")
        target_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCopying files to correct location...")
    copied = 0
    for src_file in pt_files:
        dst_file = target_dir / src_file.name
        
        if dst_file.exists():
            print(f"  - {src_file.name} (already exists, skipping)")
        else:
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ {src_file.name}")
                copied += 1
            except Exception as e:
                print(f"  ✗ {src_file.name} - Error: {e}")
    
    if copied > 0:
        print(f"\n✓ Copied {copied} file(s) to {target_dir}")
    
    # 確認
    final_files = list(target_dir.glob("best_model_*.pt"))
    print(f"\nFinal check - Files in target directory: {len(final_files)}")
    for f in final_files:
        print(f"  ✓ {f.name}")

print("\n" + "="*70)