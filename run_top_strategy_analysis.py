# -*- coding: utf-8 -*-
"""
run_top_strategy_analysis.py
TOP Strategy GNN分析の全ステップを実行（角度パラメータ対応版）

使用例:
  python run_top_strategy_analysis.py
  python run_top_strategy_analysis.py --angle 45
  python run_top_strategy_analysis.py --angle 30 --adj-angle 60
  python run_top_strategy_analysis.py --angle 60 -y  # 確認なしで実行

実行ステップ:
1. グラフ構築（GAP感度を入力、HFA感度を教師）
2. モデル訓練（隣接点情報を使って閾値を補正）
3. 重要度計算
4. 可視化
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time

# 設定ファイル読み込み
try:
    from config_top_strategy import (
        ANGLE_THRESHOLD_SAME_RING_DEG, ANGLE_THRESHOLD_ADJACENT_RING_DEG,
        ECCENTRICITY_TOLERANCE, RING_RELAXATION_FACTOR,
        MC_SAMPLES, get_output_suffix
    )
except ImportError:
    ANGLE_THRESHOLD_SAME_RING_DEG = 60.0
    ANGLE_THRESHOLD_ADJACENT_RING_DEG = 90.0
    ECCENTRICITY_TOLERANCE = 1
    RING_RELAXATION_FACTOR = 0.5
    MC_SAMPLES = 30
    def get_output_suffix(angle_deg):
        return f"_angle{int(angle_deg)}"

SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

print(f"Project directory: {GNN_PROJECT_PATH}")


def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def print_overview(args):
    print("\n" + "┏" + "━"*68 + "┓")
    print("┃" + " TOP STRATEGY GNN OVERVIEW ".center(68) + "┃")
    print("┣" + "━"*68 + "┫")
    print("┃                                                                    ┃")
    print("┃  Standard Grid Point Counts:                                       ┃")
    print("┃    - 10-2: 68 points                                               ┃")
    print("┃    - 24-2: 54 points (52 after Mariotte exclusion)                 ┃")
    print("┃    - 30-2: 76 points (74 after Mariotte exclusion)                 ┃")
    print("┃                                                                    ┃")
    print("┃  Angular Adjacency Parameters:                                     ┃")
    print(f"┃    - Same ring angle threshold: {args.angle}°".ljust(68) + "┃")
    print(f"┃    - Adjacent ring angle threshold: {args.adj_angle}°".ljust(68) + "┃")
    print(f"┃    - Eccentricity tolerance: {args.tolerance} rings".ljust(68) + "┃")
    print(f"┃    - Relaxation factor: {args.relaxation}".ljust(68) + "┃")
    print("┃                                                                    ┃")
    print("┃  Node Features:                                                    ┃")
    print("┃    - GAP Sensitivity (primary input)                               ┃")
    print("┃    - Eccentricity, Angle                                           ┃")
    print("┃                                                                    ┃")
    print("┃  Edge Features (4D):                                               ┃")
    print("┃    - Correlation (sensitivity correlation)                         ┃")
    print("┃    - AngleSimilarity                                               ┃")
    print("┃    - DistanceWeight                                                ┃")
    print("┃    - SensitivityRatio (outer/inner) ★NEW                          ┃")
    print("┃                                                                    ┃")
    print("┃  Goal: Use neighboring points to refine threshold estimates        ┃")
    print("┃                                                                    ┃")
    print("┗" + "━"*68 + "┛\n")


def check_prerequisites(args):
    print_header("Checking Prerequisites")
    
    checks = []
    
    # データパス
    print("1. Checking data availability...")
    data_path = Path("G:/共有ドライブ/GAP_Analysis/Data/GAP2_KyodaiClinical")
    
    if data_path.exists():
        folders = list(data_path.glob("*/HFAMatchData"))
        print(f"   ✓ Data found: {len(folders)} subjects with HFA data")
        checks.append(True)
    else:
        print(f"   ✗ Data path not found: {data_path}")
        checks.append(False)
    
    # スクリプト
    print("\n2. Checking required scripts...")
    scripts = [
        'create_graph_top_strategy.py',
        'train_top_strategy.py',
        'compute_importance_top_strategy.py',
        'visualize_importance_top_strategy.py'
    ]
    for script in scripts:
        path = SCRIPT_DIR / script
        if path.exists():
            print(f"   ✓ {script}")
            checks.append(True)
        else:
            print(f"   ✗ {script} not found")
            checks.append(False)
    
    # ライブラリ
    print("\n3. Checking Python libraries...")
    try:
        import torch
        import torch_geometric
        print(f"   ✓ PyTorch: {torch.__version__}")
        print(f"   ✓ PyG: {torch_geometric.__version__}")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Missing: {e}")
        checks.append(False)
    
    # 出力ディレクトリ
    print("\n4. Creating output directories...")
    suffix = args.output_suffix if args.output_suffix else get_output_suffix(args.angle)
    dirs = [
        GNN_PROJECT_PATH / "data" / f"by_eye_pattern_top{suffix}",
        GNN_PROJECT_PATH / "models" / f"top_strategy{suffix}",
        GNN_PROJECT_PATH / "results" / f"top_strategy{suffix}",
        GNN_PROJECT_PATH / "visualizations"
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {d.relative_to(GNN_PROJECT_PATH)}")
    
    print("\n" + "-"*70)
    if all(checks):
        print("✓ All prerequisites satisfied!")
        return True
    else:
        print(f"✗ {sum(not c for c in checks)} check(s) failed")
        return False


def run_script(script_path, script_args=None, script_name=""):
    """スクリプトを実行"""
    try:
        cmd = [sys.executable, str(script_path)]
        if script_args:
            cmd.extend(script_args)
        
        print(f"\nRunning: {script_path.name} {' '.join(script_args or [])}")
        print(f"Start: {time.strftime('%H:%M:%S')}")
        print("-" * 70)
        
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        
        print("-" * 70)
        print(f"✓ {script_name or script_path.name} completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-" * 70)
        print(f"✗ {script_name or script_path.name} failed (code {e.returncode})")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNN分析の全ステップを実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python run_top_strategy_analysis.py                    # デフォルト設定
  python run_top_strategy_analysis.py --angle 45         # 45°閾値
  python run_top_strategy_analysis.py --angle 30 -y      # 30°、確認なし
  python run_top_strategy_analysis.py --skip-train       # 訓練をスキップ
  python run_top_strategy_analysis.py --skip-viz         # 可視化をスキップ
        """
    )
    
    # 角度パラメータ
    parser.add_argument('--angle', '-a', type=float, 
                        default=ANGLE_THRESHOLD_SAME_RING_DEG,
                        help=f'同一リングの角度閾値（度）。デフォルト: {ANGLE_THRESHOLD_SAME_RING_DEG}')
    
    parser.add_argument('--adj-angle', '-A', type=float, 
                        default=ANGLE_THRESHOLD_ADJACENT_RING_DEG,
                        help=f'隣接リングの角度閾値（度）。デフォルト: {ANGLE_THRESHOLD_ADJACENT_RING_DEG}')
    
    parser.add_argument('--tolerance', '-t', type=int, 
                        default=ECCENTRICITY_TOLERANCE,
                        choices=[0, 1, 2],
                        help=f'偏心度リングの許容差。デフォルト: {ECCENTRICITY_TOLERANCE}')
    
    parser.add_argument('--relaxation', '-r', type=float, 
                        default=RING_RELAXATION_FACTOR,
                        help=f'リング緩和係数。デフォルト: {RING_RELAXATION_FACTOR}')
    
    parser.add_argument('--output-suffix', '-o', type=str, 
                        default=None,
                        help='出力ディレクトリのサフィックス。未指定時は_angle{角度}')
    
    # MC Dropout
    parser.add_argument('--mc-samples', '-m', type=int, 
                        default=MC_SAMPLES,
                        help=f'MC Dropoutのサンプル数。デフォルト: {MC_SAMPLES}')
    
    # スキップオプション
    parser.add_argument('--skip-train', action='store_true',
                        help='訓練をスキップ')
    
    parser.add_argument('--skip-viz', action='store_true',
                        help='可視化をスキップ')
    
    parser.add_argument('--skip-importance', action='store_true',
                        help='重要度計算をスキップ')
    
    # 確認スキップ
    parser.add_argument('--yes', '-y', action='store_true',
                        help='確認プロンプトをスキップ')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # サフィックス決定
    suffix = args.output_suffix if args.output_suffix else get_output_suffix(args.angle)
    
    print_header("TOP Strategy GNN Analysis")
    print_overview(args)
    
    print("This script will run:")
    steps = []
    if not args.skip_train:
        steps.append("1. Graph Construction (create_graph_top_strategy.py)")
        steps.append("2. Model Training (train_top_strategy.py)")
    else:
        print("  [SKIP] Graph Construction and Training")
    
    if not args.skip_importance:
        steps.append("3. Importance Computation (compute_importance_top_strategy.py)")
    else:
        print("  [SKIP] Importance Computation")
    
    if not args.skip_viz:
        steps.append("4. Visualization (visualize_importance_top_strategy.py)")
    else:
        print("  [SKIP] Visualization")
    
    for step in steps:
        print(f"  {step}")
    
    print(f"\n★ Output suffix: {suffix}")
    print("\n⚠ This may take 30-60 minutes.")
    
    if not check_prerequisites(args):
        print("\n⚠ Prerequisites check failed.")
        return
    
    if not args.yes:
        print("\n" + "="*70)
        response = input("\nContinue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nCancelled.")
            return
    
    start_time = time.time()
    results = {}
    
    # Step 1: グラフ構築
    if not args.skip_train:
        print_header("Step 1: Graph Construction")
        script_args = [
            '--angle', str(args.angle),
            '--adj-angle', str(args.adj_angle),
            '--tolerance', str(args.tolerance),
            '--relaxation', str(args.relaxation),
        ]
        if args.output_suffix:
            script_args.extend(['--output-suffix', args.output_suffix])
        
        success = run_script(
            SCRIPT_DIR / 'create_graph_top_strategy.py',
            script_args,
            'Graph Construction'
        )
        results['Graph Construction'] = success
        
        if not success:
            print("\n⚠ Graph construction failed. Stopping.")
            return
        
        time.sleep(2)
        
        # Step 2: モデル訓練
        print_header("Step 2: Model Training")
        success = run_script(
            SCRIPT_DIR / 'train_top_strategy.py',
            ['--data-suffix', suffix],
            'Model Training'
        )
        results['Model Training'] = success
        
        if not success:
            print("\n⚠ Training failed. Stopping.")
            return
        
        time.sleep(2)
    
    # Step 3: 重要度計算
    if not args.skip_importance:
        print_header("Step 3: Importance Computation")
        success = run_script(
            SCRIPT_DIR / 'compute_importance_top_strategy.py',
            ['--data-suffix', suffix, '--mc-samples', str(args.mc_samples)],
            'Importance Computation'
        )
        results['Importance Computation'] = success
        
        if not success:
            print("\n⚠ Importance computation failed.")
        
        time.sleep(2)
    
    # Step 4: 可視化
    if not args.skip_viz:
        print_header("Step 4: Visualization")
        success = run_script(
            SCRIPT_DIR / 'visualize_importance_top_strategy.py',
            ['--data-suffix', suffix],
            'Visualization'
        )
        results['Visualization'] = success
    
    # サマリー
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print_header("Summary")
    
    print("\nResults:")
    print("-" * 70)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{'✓' if success else '✗'} {name:.<50} {status}")
    print("-" * 70)
    
    print(f"\nTotal time: {minutes}m {seconds}s")
    
    if all(results.values()):
        print("\n" + "="*70)
        print("🎉 All steps completed!".center(70))
        print("="*70)
        
        print("\nOutput locations:")
        print(f"  📊 Graphs: {GNN_PROJECT_PATH / 'data' / f'by_eye_pattern_top{suffix}'}")
        print(f"  🤖 Models: {GNN_PROJECT_PATH / 'models' / f'top_strategy{suffix}'}")
        print(f"  📈 Results: {GNN_PROJECT_PATH / 'results' / f'top_strategy{suffix}'}")
        print(f"  📉 Visualizations: {GNN_PROJECT_PATH / 'visualizations'}")
        
        print("\n" + "┏" + "━"*68 + "┓")
        print("┃" + " KEY METRICS TO CHECK ".center(68) + "┃")
        print("┣" + "━"*68 + "┫")
        print("┃ 1. Baseline (GAP only) MAE vs GNN Model MAE                       ┃")
        print("┃    → GNN should be lower (improvement from neighbors)             ┃")
        print("┃                                                                    ┃")
        print("┃ 2. Improvement percentage                                         ┃")
        print("┃    → Positive = GNN is helping                                    ┃")
        print("┃                                                                    ┃")
        print("┃ 3. Edge feature importance                                        ┃")
        print("┃    → SensitivityRatio captures center-to-periphery gradient       ┃")
        print("┗" + "━"*68 + "┛")
    else:
        print("\n⚠ Some steps failed. Check errors above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)