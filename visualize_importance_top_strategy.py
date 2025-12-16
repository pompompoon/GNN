# -*- coding: utf-8 -*-
"""
visualize_importance_top_strategy.py
TOP Strategy GNNの重要度マップ可視化（4指標版・角度パラメータ対応）

使用例:
  python visualize_importance_top_strategy.py
  python visualize_importance_top_strategy.py --data-suffix _angle45

出力:
- combined_score_{pattern}.png: 総合スコア単体図
- importance_map_{pattern}.png: 5パネル図（Combined + 4指標）
- comparison_{pattern}.png: 左右眼比較
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# 設定ファイル読み込み
try:
    from config_top_strategy import (
        WEIGHT_PRED_STD, WEIGHT_PREDICTION_ERROR,
        WEIGHT_GNN_CORRECTION, WEIGHT_LEAVE_ONE_OUT,
        USE_CORRELATION, USE_ANGLE_SIMILARITY, USE_SENSITIVITY_RATIO,
        USE_SENSITIVITY_RATIO_INVERSE, USE_DISTANCE_WEIGHT,
    )
except ImportError:
    WEIGHT_PRED_STD = 0.30
    WEIGHT_PREDICTION_ERROR = 0.25
    WEIGHT_GNN_CORRECTION = 0.25
    WEIGHT_LEAVE_ONE_OUT = 0.20
    USE_CORRELATION = False
    USE_ANGLE_SIMILARITY = True
    USE_SENSITIVITY_RATIO = True
    USE_SENSITIVITY_RATIO_INVERSE = True
    USE_DISTANCE_WEIGHT = True

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# 標準グリッド点数
STANDARD_GRID_COUNTS = {
    'Pattern30-2': 74,
    'Pattern24-2': 52,
    'Pattern10-2': 68,
}


def get_mariotte_position(eye_side):
    """マリオット盲点の位置"""
    if eye_side == 0:
        return (-15, 0)
    else:
        return (15, 0)


def draw_mariotte_indicator(ax, pattern_name, eye_side):
    """マリオット盲点の位置を表示"""
    if 'Pattern10-2' in pattern_name:
        return
    
    mx, my = get_mariotte_position(eye_side)
    
    circle = Circle((mx, 0), radius=2, facecolor='gray', 
                    edgecolor='black', alpha=0.5, linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    
    side_name = "耳側(左)" if eye_side == 0 else "耳側(右)"
    ax.annotate(f'Mariotte\n({side_name})', xy=(mx, 0), xytext=(mx, -8),
                fontsize=7, ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))


def get_axis_range(pattern_name):
    """パターンに応じた軸範囲"""
    if 'Pattern10-2' in pattern_name:
        return 12
    else:
        return 32


def create_single_metric_map(pattern_name, positions, values, metric_name, 
                             title, cmap, output_path, eye_side=None):
    """単一指標のマップを作成"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    axis_range = get_axis_range(pattern_name)
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=values, 
                        s=200, cmap=cmap, vmin=0, vmax=10, edgecolors='black', linewidths=0.5)
    
    ax.set_xlim(-axis_range, axis_range)
    ax.set_ylim(-axis_range, axis_range)
    ax.set_xlabel('X (degrees)', fontsize=12)
    ax.set_ylabel('Y (degrees)', fontsize=12)
    ax.set_title(f'{pattern_name}\n{title}', fontsize=14)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2)
    
    if eye_side is not None:
        draw_mariotte_indicator(ax, pattern_name, eye_side)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Importance (0-10)', fontsize=11)
    
    plt.tight_layout()
    
    output_file = output_path / f"{metric_name}_{pattern_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    ✓ {output_file.name}")
    
    return output_file


def create_5panel_map(pattern_name, importance_dict, output_path, eye_side=None):
    """5パネル図を作成"""
    positions = importance_dict['positions']
    
    pred_std = importance_dict.get('pred_std_normalized', np.zeros(len(positions)))
    prediction_error = importance_dict.get('prediction_error_normalized', np.zeros(len(positions)))
    gnn_correction = importance_dict.get('gnn_correction_normalized', np.zeros(len(positions)))
    leave_one_out = importance_dict.get('leave_one_out_normalized', np.zeros(len(positions)))
    combined = importance_dict.get('combined', np.zeros(len(positions)))
    
    axis_range = get_axis_range(pattern_name)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    
    # 1. Combined Score
    ax1 = axes[0, 1]
    scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], c=combined, 
                           s=120, cmap='RdYlGn_r', vmin=0, vmax=10, 
                           edgecolors='black', linewidths=0.5)
    ax1.set_xlim(-axis_range, axis_range)
    ax1.set_ylim(-axis_range, axis_range)
    ax1.set_xlabel('X (degrees)', fontsize=10)
    ax1.set_ylabel('Y (degrees)', fontsize=10)
    weight_str = f'{WEIGHT_PRED_STD}×std + {WEIGHT_PREDICTION_ERROR}×err + {WEIGHT_GNN_CORRECTION}×corr + {WEIGHT_LEAVE_ONE_OUT}×LOO'
    ax1.set_title(f'Combined Score\n({weight_str})', fontsize=11)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.2)
    if eye_side is not None:
        draw_mariotte_indicator(ax1, pattern_name, eye_side)
    plt.colorbar(scatter1, ax=ax1, label='Importance', shrink=0.8)
    
    # 2. pred_std
    ax2 = axes[0, 0]
    scatter2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pred_std,
                           s=120, cmap='Oranges', vmin=0, vmax=10,
                           edgecolors='black', linewidths=0.5)
    ax2.set_xlim(-axis_range, axis_range)
    ax2.set_ylim(-axis_range, axis_range)
    ax2.set_xlabel('X (degrees)', fontsize=10)
    ax2.set_ylabel('Y (degrees)', fontsize=10)
    ax2.set_title(f'1. pred_std ({WEIGHT_PRED_STD*100:.0f}%)\nMC Dropout Uncertainty', fontsize=11)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.2)
    if eye_side is not None:
        draw_mariotte_indicator(ax2, pattern_name, eye_side)
    plt.colorbar(scatter2, ax=ax2, label='Uncertainty', shrink=0.8)
    
    # 3. prediction_error
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(positions[:, 0], positions[:, 1], c=prediction_error,
                           s=120, cmap='Reds', vmin=0, vmax=10,
                           edgecolors='black', linewidths=0.5)
    ax3.set_xlim(-axis_range, axis_range)
    ax3.set_ylim(-axis_range, axis_range)
    ax3.set_xlabel('X (degrees)', fontsize=10)
    ax3.set_ylabel('Y (degrees)', fontsize=10)
    ax3.set_title(f'2. prediction_error ({WEIGHT_PREDICTION_ERROR*100:.0f}%)\n|Predicted - HFA|', fontsize=11)
    ax3.set_aspect('equal')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.2)
    if eye_side is not None:
        draw_mariotte_indicator(ax3, pattern_name, eye_side)
    plt.colorbar(scatter3, ax=ax3, label='Error', shrink=0.8)
    
    # 4. gnn_correction
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(positions[:, 0], positions[:, 1], c=gnn_correction,
                           s=120, cmap='Purples', vmin=0, vmax=10,
                           edgecolors='black', linewidths=0.5)
    ax4.set_xlim(-axis_range, axis_range)
    ax4.set_ylim(-axis_range, axis_range)
    ax4.set_xlabel('X (degrees)', fontsize=10)
    ax4.set_ylabel('Y (degrees)', fontsize=10)
    ax4.set_title(f'3. gnn_correction ({WEIGHT_GNN_CORRECTION*100:.0f}%)\n|Predicted - GAP|', fontsize=11)
    ax4.set_aspect('equal')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.2)
    if eye_side is not None:
        draw_mariotte_indicator(ax4, pattern_name, eye_side)
    plt.colorbar(scatter4, ax=ax4, label='Correction', shrink=0.8)
    
    # 5. leave_one_out
    ax5 = axes[1, 1]
    scatter5 = ax5.scatter(positions[:, 0], positions[:, 1], c=leave_one_out,
                           s=120, cmap='Blues', vmin=0, vmax=10,
                           edgecolors='black', linewidths=0.5)
    ax5.set_xlim(-axis_range, axis_range)
    ax5.set_ylim(-axis_range, axis_range)
    ax5.set_xlabel('X (degrees)', fontsize=10)
    ax5.set_ylabel('Y (degrees)', fontsize=10)
    ax5.set_title(f'4. leave_one_out ({WEIGHT_LEAVE_ONE_OUT*100:.0f}%)\nPoint Exclusion Influence', fontsize=11)
    ax5.set_aspect('equal')
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax5.grid(True, alpha=0.2)
    if eye_side is not None:
        draw_mariotte_indicator(ax5, pattern_name, eye_side)
    plt.colorbar(scatter5, ax=ax5, label='Influence', shrink=0.8)
    
    # 6. 空白
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    fig.suptitle(f'{pattern_name} - Importance Analysis (TOP Strategy)', 
                 fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    output_file = output_path / f"importance_map_{pattern_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ 5-panel: {output_file.name}")
    
    return output_file


def create_comparison_by_eye(input_path, output_path):
    """左右眼の比較図"""
    patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    
    for pattern in patterns:
        left_file = input_path / f"importance_map_Left_{pattern}.pkl"
        right_file = input_path / f"importance_map_Right_{pattern}.pkl"
        
        if not left_file.exists() or not right_file.exists():
            continue
        
        with open(left_file, 'rb') as f:
            left_data = pickle.load(f)
        with open(right_file, 'rb') as f:
            right_data = pickle.load(f)
        
        axis_range = get_axis_range(pattern)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left Eye
        ax1 = axes[0]
        left_pos = left_data['positions']
        left_combined = left_data['combined']
        scatter1 = ax1.scatter(left_pos[:, 0], left_pos[:, 1], c=left_combined,
                              s=200, cmap='RdYlGn_r', vmin=0, vmax=10,
                              edgecolors='black', linewidths=0.5)
        ax1.set_xlim(-axis_range, axis_range)
        ax1.set_ylim(-axis_range, axis_range)
        ax1.set_xlabel('X (degrees)', fontsize=12)
        ax1.set_ylabel('Y (degrees)', fontsize=12)
        ax1.set_title(f'Left Eye - {pattern}\n({len(left_pos)} points)', fontsize=14)
        ax1.set_aspect('equal')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax1.grid(True, alpha=0.2)
        draw_mariotte_indicator(ax1, pattern, eye_side=0)
        plt.colorbar(scatter1, ax=ax1, label='Importance', shrink=0.8)
        
        # Right Eye
        ax2 = axes[1]
        right_pos = right_data['positions']
        right_combined = right_data['combined']
        scatter2 = ax2.scatter(right_pos[:, 0], right_pos[:, 1], c=right_combined,
                              s=200, cmap='RdYlGn_r', vmin=0, vmax=10,
                              edgecolors='black', linewidths=0.5)
        ax2.set_xlim(-axis_range, axis_range)
        ax2.set_ylim(-axis_range, axis_range)
        ax2.set_xlabel('X (degrees)', fontsize=12)
        ax2.set_ylabel('Y (degrees)', fontsize=12)
        ax2.set_title(f'Right Eye - {pattern}\n({len(right_pos)} points)', fontsize=14)
        ax2.set_aspect('equal')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax2.grid(True, alpha=0.2)
        draw_mariotte_indicator(ax2, pattern, eye_side=1)
        plt.colorbar(scatter2, ax=ax2, label='Importance', shrink=0.8)
        
        plt.tight_layout()
        
        output_file = output_path / f"comparison_{pattern}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Comparison: {output_file.name}")


def create_summary_report(input_path, output_path, data_suffix):
    """分析サマリーレポートを生成"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # エッジ特徴量名を構築
    edge_features = []
    if USE_CORRELATION:
        edge_features.append('Correlation')
    if USE_ANGLE_SIMILARITY:
        edge_features.append('AngleSimilarity')
    if USE_SENSITIVITY_RATIO:
        edge_features.append('SensRatio_OuterInner')
    if USE_SENSITIVITY_RATIO_INVERSE:
        edge_features.append('SensRatio_InnerOuter')
    if USE_DISTANCE_WEIGHT:
        edge_features.append('DistanceWeight')
    edge_features_str = ', '.join(edge_features) if edge_features else 'None'
    
    lines = []
    lines.append("=" * 70)
    lines.append("IMPORTANCE MAP ANALYSIS SUMMARY REPORT (TOP Strategy)")
    lines.append(f"Generated: {timestamp}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model Features:")
    lines.append("  Node: [NormX, NormY, GAP_Sensitivity, Eccentricity, Angle]")
    lines.append(f"  Edge: [{edge_features_str}]")
    lines.append("  Adjacency: Angular-based (eccentricity rings)")
    lines.append("")
    lines.append("Standard Grid Point Counts:")
    lines.append("  Pattern30-2: 74 points (Mariotte excluded)")
    lines.append("  Pattern24-2: 52 points (Mariotte excluded)")
    lines.append("  Pattern10-2: 68 points")
    lines.append("")
    lines.append("Reduction Ratio: 50%")
    lines.append("")
    
    # 訓練結果を読み込み
    lines.append("=" * 70)
    lines.append("Training Summary")
    lines.append("=" * 70)
    
    results_path = input_path  # results/top_strategy{suffix}
    csv_file = results_path / "training_results_top_strategy.csv"
    
    if csv_file.exists():
        df_results = pd.read_csv(csv_file)
        for _, row in df_results.iterrows():
            lines.append("")
            lines.append(f"{row['eye_pattern_name']}:")
            lines.append(f"  Best Val MAE: {row['best_val_mae']:.2f} dB")
            lines.append(f"  Test MAE: {row['test_mae']:.2f} dB")
            lines.append(f"  Test RMSE: {row['test_rmse']:.2f} dB")
            lines.append(f"  Data: Train={int(row['n_train'])}, Val={int(row['n_val'])}, Test={int(row['n_test'])}")
    else:
        lines.append("")
        lines.append("  (Training results not found)")
    
    # 重要度マップ結果
    lines.append("")
    lines.append("=" * 70)
    lines.append("IMPORTANCE MAP RESULTS")
    lines.append("=" * 70)
    
    importance_files = sorted(input_path.glob("importance_map_*.pkl"))
    
    for pkl_file in importance_files:
        pattern_name = pkl_file.stem.replace("importance_map_", "")
        
        with open(pkl_file, 'rb') as f:
            importance_dict = pickle.load(f)
        
        positions = importance_dict['positions']
        combined = importance_dict.get('combined', np.zeros(len(positions)))
        
        total_points = len(positions)
        essential_points = total_points // 2  # 50% reduction
        
        # パターン名から期待される点数を取得
        for pattern_key in STANDARD_GRID_COUNTS:
            if pattern_key in pattern_name:
                expected_points = STANDARD_GRID_COUNTS[pattern_key]
                break
        else:
            expected_points = total_points
        
        lines.append("")
        lines.append(f"{pattern_name}:")
        lines.append(f"  Total Points: {total_points} (expected: {expected_points})")
        lines.append(f"  Essential Points: {essential_points}")
        lines.append(f"  Reduction: 50%")
        lines.append(f"  Score Range: {combined.min():.2f} - {combined.max():.2f}")
    
    # 出力ファイルリスト
    lines.append("")
    lines.append("=" * 70)
    lines.append("OUTPUT FILES")
    lines.append("=" * 70)
    
    output_files = sorted(output_path.glob("*.png"))
    for f in output_files:
        lines.append(f"  - {f.name}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    # ファイルに保存
    summary_file = output_path / "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ Analysis summary saved: {summary_file.name}")


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNNの重要度マップ可視化'
    )
    
    parser.add_argument('--data-suffix', '-d', type=str, default='',
                        help='データディレクトリのサフィックス (例: _angle45)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # パス設定
    INPUT_PATH = GNN_PROJECT_PATH / "results" / f"top_strategy{args.data_suffix}"
    OUTPUT_PATH = GNN_PROJECT_PATH / "visualizations" / f"importance_top{args.data_suffix}_{TIMESTAMP}"
    
    print("="*70)
    print("Visualizing Importance Maps (TOP Strategy - 4 Metrics)")
    print("="*70)
    print(f"\nInput path: {INPUT_PATH}")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {OUTPUT_PATH}")
    
    # 重要度マップファイル検索
    importance_files = list(INPUT_PATH.glob("importance_map_*.pkl"))
    
    if len(importance_files) == 0:
        print(f"\nNo data found in {INPUT_PATH}")
        print("Please run compute_importance_top_strategy.py first")
        exit(1)
    
    print(f"\nFound {len(importance_files)} importance maps")
    
    # 各パターンの可視化
    for pkl_file in sorted(importance_files):
        pattern_name = pkl_file.stem.replace("importance_map_", "")
        print(f"\n--- {pattern_name} ---")
        
        with open(pkl_file, 'rb') as f:
            importance_dict = pickle.load(f)
        
        positions = importance_dict['positions']
        eye_side = 0 if 'Left' in pattern_name else 1
        combined = importance_dict.get('combined', np.zeros(len(positions)))
        
        # 総合スコア単体図
        weight_str = f'{WEIGHT_PRED_STD}×std + {WEIGHT_PREDICTION_ERROR}×err + {WEIGHT_GNN_CORRECTION}×corr + {WEIGHT_LEAVE_ONE_OUT}×LOO'
        create_single_metric_map(
            pattern_name, positions, combined,
            'combined_score', f'Combined Importance Score\n({weight_str})',
            'RdYlGn_r', OUTPUT_PATH, eye_side
        )
        
        # 5パネル図
        create_5panel_map(pattern_name, importance_dict, OUTPUT_PATH, eye_side)
    
    # 左右眼比較
    print("\n--- Creating comparisons ---")
    create_comparison_by_eye(INPUT_PATH, OUTPUT_PATH)
    
    # サマリーレポート
    print("\n--- Creating report ---")
    create_summary_report(INPUT_PATH, OUTPUT_PATH, args.data_suffix)
    
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"\nOutput saved to: {OUTPUT_PATH}")