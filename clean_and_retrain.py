# -*- coding: utf-8 -*-
"""
clean_and_retrain.py
古いモデルを削除して再訓練
"""

from pathlib import Path
import shutil

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "by_eye_pattern"

print("="*70)
print("Clean Old Models and Retrain")
print("="*70)

if not MODEL_PATH.exists():
    print(f"\nModel directory not found: {MODEL_PATH}")
    print("Nothing to clean.")
else:
    # 古いモデルファイルを削除
    old_models = list(MODEL_PATH.glob("best_model_*.pt"))
    
    print(f"\nFound {len(old_models)} model files")
    print("\nRemoving old models...")
    
    for model_file in old_models:
        try:
            model_file.unlink()
            print(f"  ✓ Removed: {model_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to remove {model_file.name}: {e}")
    
    print(f"\n✓ Cleanup completed")

print(f"\n{'='*70}")
print("Now run training:")
print(f"{'='*70}")
print("\n  python train_no_batch.py")
print(f"\n{'='*70}")