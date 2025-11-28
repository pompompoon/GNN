# -*- coding: utf-8 -*-
"""
fix_all.py
すべてのパスとファイルの問題を一括で修正
"""

from pathlib import Path
import shutil

print("="*70)
print("Fix All Issues - Comprehensive Setup")
print("="*70)

current_dir = Path.cwd()
print(f"\nWorking directory: {current_dir}")

issues_found = []
issues_fixed = []

# ===== 1. モデルファイルの確認 =====
print("\n" + "-"*70)
print("Step 1: Checking model files...")
print("-"*70)

model_dir = current_dir / "models" / "by_eye_pattern"
model_dir.mkdir(parents=True, exist_ok=True)

expected_models = [
    'best_model_Left_Pattern10-2.pt',
    'best_model_Left_Pattern24-2.pt',
    'best_model_Left_Pattern30-2.pt',
    'best_model_Right_Pattern10-2.pt',
    'best_model_Right_Pattern24-2.pt',
    'best_model_Right_Pattern30-2.pt',
]

model_count = 0
for model_name in expected_models:
    model_path = model_dir / model_name
    if model_path.exists():
        print(f"  ✓ {model_name}")
        model_count += 1
    else:
        print(f"  ✗ {model_name} (missing)")
        issues_found.append(f"Model file missing: {model_name}")

# 代替モデルファイルを探してリネーム
if model_count < 6:
    print("\nSearching for alternative model files...")
    
    # gnn_project5形式のファイルを探す
    alt_models = {
        'best_model_Pattern10-2_LeftEye.pt': 'best_model_Left_Pattern10-2.pt',
        'best_model_Pattern10-2_RightEye.pt': 'best_model_Right_Pattern10-2.pt',
        'best_model_Pattern24-2_LeftEye.pt': 'best_model_Left_Pattern24-2.pt',
        'best_model_Pattern24-2_RightEye.pt': 'best_model_Right_Pattern24-2.pt',
        'best_model_Pattern30-2_LeftEye.pt': 'best_model_Left_Pattern30-2.pt',
        'best_model_Pattern30-2_RightEye.pt': 'best_model_Right_Pattern30-2.pt',
    }
    
    for old_name, new_name in alt_models.items():
        old_path = model_dir / old_name
        new_path = model_dir / new_name
        
        if old_path.exists() and not new_path.exists():
            try:
                shutil.move(str(old_path), str(new_path))
                print(f"  ✓ Renamed: {old_name} → {new_name}")
                issues_fixed.append(f"Renamed model: {old_name}")
                model_count += 1
            except Exception as e:
                print(f"  ✗ Failed to rename: {e}")

print(f"\nModel files: {model_count} / 6")

# ===== 2. 重要度マップの確認 =====
print("\n" + "-"*70)
print("Step 2: Checking importance maps...")
print("-"*70)

importance_dir = current_dir / "results" / "importance_maps_by_eye_pattern"
importance_dir.mkdir(parents=True, exist_ok=True)

# 重要度マップファイルを探す
importance_pkl = list(current_dir.rglob("importance_map_*.pkl"))

if len(importance_pkl) == 0:
    print("  ✗ No importance map files found")
    issues_found.append("No importance maps generated")
else:
    print(f"  Found {len(importance_pkl)} importance map file(s)")
    
    # 正しい場所にコピー
    copied = 0
    for src_file in importance_pkl:
        dst_file = importance_dir / src_file.name
        
        if not dst_file.exists():
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ Copied: {src_file.name}")
                copied += 1
                issues_fixed.append(f"Copied importance map: {src_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to copy {src_file.name}: {e}")
        
        # CSVファイルもコピー
        csv_file = src_file.with_suffix('.csv')
        if csv_file.exists():
            dst_csv = importance_dir / csv_file.name
            if not dst_csv.exists():
                try:
                    shutil.copy2(csv_file, dst_csv)
                except:
                    pass
    
    if copied > 0:
        print(f"  ✓ Copied {copied} file(s) to correct location")

final_importance = list(importance_dir.glob("importance_map_*.pkl"))
print(f"\nImportance maps: {len(final_importance)} / 6")

# ===== 3. データファイルの確認 =====
print("\n" + "-"*70)
print("Step 3: Checking data files...")
print("-"*70)

data_dir = current_dir / "data" / "by_eye_pattern"
data_pkl = list(data_dir.glob("graph_data_*.pkl")) if data_dir.exists() else []

print(f"  Graph data files: {len(data_pkl)}")

if len(data_pkl) < 6:
    issues_found.append(f"Insufficient data files: {len(data_pkl)} / 6")
else:
    print(f"  ✓ All data files present")

# ===== サマリー =====
print("\n" + "="*70)
print("Summary")
print("="*70)

print(f"\nStatus:")
print(f"  Models:          {model_count} / 6")
print(f"  Importance maps: {len(final_importance)} / 6")
print(f"  Data files:      {len(data_pkl)} / 6")

if len(issues_found) > 0:
    print(f"\n⚠️  Issues found ({len(issues_found)}):")
    for issue in issues_found:
        print(f"  - {issue}")

if len(issues_fixed) > 0:
    print(f"\n✓ Issues fixed ({len(issues_fixed)}):")
    for fix in issues_fixed:
        print(f"  - {fix}")

print("\n" + "="*70)
print("Next Steps")
print("="*70)

if model_count < 6:
    print("\n1. Train models:")
    print("   python train_by_eye_pattern_revised.py")

if len(final_importance) < 6:
    print("\n2. Compute importance maps:")
    print("   python compute_importance_simple.py")

if model_count >= 6 and len(final_importance) >= 6:
    print("\n✓ All files are ready!")
    print("\nRun visualization:")
    print("   python visualize_importance_by_eye_pattern.py")

print("\n" + "="*70)