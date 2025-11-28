# -*- coding: utf-8 -*-
"""
run_importance_analysis_by_eye_pattern.py
左右眼別・パターン別の重要度マップ分析の全ステップを実行

実行ステップ:
1. グラフ構築（左右眼別・パターン別）
2. モデル訓練（眼・パターンごと）
3. 重要度計算
4. 可視化
"""

import subprocess
import sys
from pathlib import Path
import time

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR.parent

if not GNN_PROJECT_PATH.exists():
    GNN_PROJECT_PATH = Path.cwd()

print(f"Project directory: {GNN_PROJECT_PATH}")

# 実行するスクリプトのリスト
SCRIPTS = [
    {
        'name': 'Graph Construction (Eye-Pattern)',
        'script': 'create_graph_by_eye_pattern_revised.py',
        'description': 'Creating graph data by eye and pattern...',
        'critical': True,
        'details': [
            '  - Eye separation: Left (0) / Right (1)',
            '  - Patterns: 30-2, 24-2, 10-2',
            '  - Edge attributes: 2D [distance_weight, sensitivity_similarity]',
            '  - Teacher data: HFA Sensitivity'
        ]
    },
    {
        'name': 'Model Training (Eye-Pattern)',
        'script': 'train_by_eye_pattern_revised.py',
        'description': 'Training GNN models for each eye-pattern combination...',
        'critical': True,
        'details': [
            '  - Separate models for each eye-pattern',
            '  - Edge dim: 2 (distance + similarity)',
            '  - Teacher: HFA Sensitivity'
        ]
    },
    {
        'name': 'Importance Computation (Eye-Pattern)',
        'script': 'compute_importance_by_eye_pattern.py',
        'description': 'Computing importance maps...',
        'critical': False,
        'details': [
            '  - Uncertainty-based importance',
            '  - Error-based importance',
            '  - Leave-one-out importance',
            '  - Combined score'
        ]
    },
    {
        'name': 'Visualization (Eye-Pattern)',
        'script': 'visualize_importance_by_eye_pattern.py',
        'description': 'Creating visualizations...',
        'critical': False,
        'details': [
            '  - Individual importance maps',
            '  - Left vs Right comparison',
            '  - Pattern comparison'
        ]
    }
]


def print_header(text):
    """ヘッダーを表示"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def print_project_summary():
    """プロジェクトサマリーを表示"""
    print("\n" + "┏" + "━"*68 + "┓")
    print("┃" + " PROJECT OVERVIEW ".center(68) + "┃")
    print("┣" + "━"*68 + "┫")
    print("┃ Objective: Predict peripheral sensitivity using GNN               ┃")
    print("┃                                                                    ┃")
    print("┃ Key Features:                                                      ┃")
    print("┃  • Separate models for Left/Right eyes                            ┃")
    print("┃  • Pattern-specific models (30-2, 24-2, 10-2)                     ┃")
    print("┃  • 2D Edge attributes: [distance, similarity]                     ┃")
    print("┃  • Teacher data: HFA Sensitivity (Gold Standard)                  ┃")
    print("┃  • Adjacency: Grid-based (≤6.5°)                                 ┃")
    print("┃                                                                    ┃")
    print("┃ Expected Output:                                                   ┃")
    print("┃  • Importance maps for each eye-pattern combination               ┃")
    print("┃  • Optimized measurement protocols (50% reduction)                ┃")
    print("┗" + "━"*68 + "┛\n")


def check_prerequisites():
    """前提条件のチェック"""
    print_header("Checking Prerequisites")
    
    checks = []
    
    # 1. HFAデータの存在確認
    print("1. Checking HFA data availability...")
    hfa_data_path = Path(r"G:\共有ドライブ\GAP_Analysis\Data\GAP2_KyodaiClinical")
    
    if hfa_data_path.exists():
        hfa_folders = list(hfa_data_path.glob("*/HFAMatchData"))
        if len(hfa_folders) > 0:
            print(f"   ✓ HFA data found: {len(hfa_folders)} folders")
            checks.append(True)
        else:
            print(f"   ✗ No HFAMatchData folders found")
            checks.append(False)
    else:
        print(f"   ✗ Data path not found: {hfa_data_path}")
        checks.append(False)
    
    # 2. 必要なスクリプトの存在確認
    print("\n2. Checking required scripts...")
    for script_info in SCRIPTS:
        script_path = SCRIPT_DIR / script_info['script']
        if script_path.exists():
            print(f"   ✓ {script_info['script']}")
            checks.append(True)
        else:
            print(f"   ✗ {script_info['script']} not found")
            checks.append(False)
    
    # models_revised.pyの確認
    models_revised_path = SCRIPT_DIR / 'models_revised.py'
    if models_revised_path.exists():
        print(f"   ✓ models_revised.py")
        checks.append(True)
    else:
        print(f"   ✗ models_revised.py not found")
        checks.append(False)
    
    # 3. Pythonライブラリの確認
    print("\n3. Checking Python libraries...")
    try:
        import torch
        import torch_geometric
        import pandas
        import numpy
        import scipy
        print(f"   ✓ All required libraries available")
        print(f"      - PyTorch: {torch.__version__}")
        print(f"      - PyG: {torch_geometric.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Missing library: {e}")
        checks.append(False)
    
    # 4. 出力ディレクトリの確認/作成
    print("\n4. Checking output directories...")
    output_dirs = [
        GNN_PROJECT_PATH / "data" / "by_eye_pattern",
        GNN_PROJECT_PATH / "models" / "by_eye_pattern",
        GNN_PROJECT_PATH / "results" / "by_eye_pattern",
        GNN_PROJECT_PATH / "results" / "importance_maps_by_eye_pattern",
        GNN_PROJECT_PATH / "visualizations" / "importance_maps_by_eye_pattern"
    ]
    
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir.exists():
            print(f"   ✓ {out_dir.relative_to(GNN_PROJECT_PATH)}")
            checks.append(True)
        else:
            print(f"   ✗ Failed to create: {out_dir}")
            checks.append(False)
    
    print("\n" + "-"*70)
    if all(checks):
        print("✓ All prerequisites satisfied!")
        return True
    else:
        print(f"✗ {sum(not c for c in checks)} check(s) failed")
        print("\nPlease fix the issues above before continuing.")
        return False


def run_script(script_path, script_name, details=None):
    """Pythonスクリプトを実行"""
    try:
        if details:
            print("\nDetails:")
            for detail in details:
                print(detail)
        
        print(f"\nRunning: {script_path.name}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=True
        )
        
        print("-" * 70)
        print(f"✓ {script_name} completed successfully")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 70)
        print(f"✗ {script_name} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False
    
    except Exception as e:
        print("-" * 70)
        print(f"✗ {script_name} failed with exception:")
        print(f"Error: {e}")
        return False


def main():
    """メイン処理"""
    
    print_header("GNN-Based Peripheral Sensitivity Prediction")
    
    # プロジェクトサマリー表示
    print_project_summary()
    
    print("This script will run all steps of the analysis:")
    print("  1. Graph data construction (by eye and pattern)")
    print("  2. Model training (separate models for each combination)")
    print("  3. Importance score computation")
    print("  4. Visualization (individual and comparison)")
    print("\n⚠ This process may take 30-90 minutes depending on your system.")
    
    # 前提条件のチェック
    if not check_prerequisites():
        print("\n⚠ Prerequisites check failed. Exiting.")
        return
    
    # 確認
    print("\n" + "="*70)
    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nAnalysis cancelled.")
        return
    
    # 開始時刻
    start_time = time.time()
    
    # 各スクリプトを順次実行
    results = {}
    
    for idx, step in enumerate(SCRIPTS, 1):
        script_name = step['name']
        script_file = step['script']
        description = step['description']
        details = step.get('details', None)
        is_critical = step.get('critical', True)
        
        print_header(f"Step {idx}/{len(SCRIPTS)}: {script_name}")
        print(description)
        
        script_path = SCRIPT_DIR / script_file
        
        if not script_path.exists():
            print(f"\n✗ Error: Script not found: {script_path}")
            results[script_name] = False
            
            if is_critical:
                print("\n⚠ Critical step failed. Stopping analysis.")
                break
            continue
        
        success = run_script(script_path, script_name, details)
        results[script_name] = success
        
        if not success and is_critical:
            print("\n⚠ Critical step failed. Stopping analysis.")
            break
        
        if idx < len(SCRIPTS):
            print("\nWaiting 3 seconds before next step...")
            time.sleep(3)
    
    # 終了時刻と所要時間
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    # 結果サマリー
    print_header("Analysis Summary")
    
    all_success = all(results.values())
    
    print("\nResults:")
    print("-" * 70)
    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        icon = "✓" if success else "✗"
        print(f"{icon} {script_name:.<55} {status}")
    print("-" * 70)
    
    print(f"\nTotal elapsed time: {hours}h {minutes}m {seconds}s")
    
    if all_success:
        print("\n" + "="*70)
        print("🎉 All steps completed successfully!".center(70))
        print("="*70)
        print("\nResults location:")
        print(f"  📊 Graph data:      {GNN_PROJECT_PATH / 'data' / 'by_eye_pattern'}")
        print(f"  🤖 Models:          {GNN_PROJECT_PATH / 'models' / 'by_eye_pattern'}")
        print(f"  📈 Importance maps: {GNN_PROJECT_PATH / 'results' / 'importance_maps_by_eye_pattern'}")
        print(f"  📉 Visualizations:  {GNN_PROJECT_PATH / 'visualizations' / 'importance_maps_by_eye_pattern'}")
        
        print("\n" + "┏" + "━"*68 + "┓")
        print("┃" + " NEXT STEPS ".center(68) + "┃")
        print("┣" + "━"*68 + "┫")
        print("┃ 1. Review the importance map visualizations                       ┃")
        print("┃    → Individual maps for each eye-pattern                         ┃")
        print("┃    → Left vs Right comparisons                                    ┃")
        print("┃    → Pattern comparisons (30-2, 24-2, 10-2)                       ┃")
        print("┃                                                                    ┃")
        print("┃ 2. Analyze model performance                                      ┃")
        print("┃    → Check training_results_by_eye_pattern.csv                    ┃")
        print("┃    → Compare MAE across different combinations                    ┃")
        print("┃                                                                    ┃")
        print("┃ 3. Validate clinical relevance                                    ┃")
        print("┃    → Compare with known glaucoma patterns                         ┃")
        print("┃    → Verify essential points align with clinical knowledge        ┃")
        print("┃                                                                    ┃")
        print("┃ 4. Adjust reduction ratio if needed                               ┃")
        print("┃    → Edit REDUCTION_RATIO in visualize script                     ┃")
        print("┗" + "━"*68 + "┛")
        
    else:
        print("\n" + "="*70)
        print("⚠ Some steps failed".center(70))
        print("="*70)
        print("\nTroubleshooting:")
        print("  1. Check error messages above for specific issues")
        print("  2. Ensure all required data files are present")
        print("  3. Verify models_revised.py is available")
        print("  4. Check sufficient disk space and memory")
    
    print("\n" + "="*70)
    print("Analysis completed".center(70))
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠ Analysis interrupted by user".center(70))
        print("="*70)
        sys.exit(1)
    except Exception as e:
        print("\n\n" + "="*70)
        print("✗ Unexpected error occurred".center(70))
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)