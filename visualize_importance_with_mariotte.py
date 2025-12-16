# -*- coding: utf-8 -*-
"""
visualize_importance_with_mariotte.py
重要度マップ可視化（マリオット盲点位置を正しく表示）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import japanize_matplotlib

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# TOP Strategy用
INPUT_PATH_TOP = GNN_PROJECT_PATH / "results" / "top_strategy"
OUTPUT_PATH = GNN_PROJECT_PATH / "visualizations" / f"importance_{TIMESTAMP}"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

REDUCTION_RATIO = 0.5


def get_mariotte_position(eye_side):
    """
    マリオット盲点の位置（正しい解剖学的位置）
    
    解剖学的事実:
    - 視神経乳頭は網膜の鼻側にある
    - 視野検査では像が反転するため、視野上では耳側に現れる
    - 左目: 耳側 = 左側 (-15°付近)
    - 右目: 耳側 = 右側 (+15°付近)
    """
    if eye_side == 0:  # Left eye - 盲点は耳側（左）
        return (-15, 0)
    else:  # Right eye - 盲点は耳側（右）
        return (15, 0)


def draw_mariotte_indicator(ax, eye_side, pattern_name):
    """マリオット盲点の位置を示すインジケーターを描画"""
    if 'Pattern10-2' in pattern_name:
        return  # 10-2にはマリオット盲点なし
    
    mx, my = get_mariotte_position(eye_side)
    
    # マリオット盲点の位置にマーカーを追加
    # 灰色の円で表示
    circle = Circle((mx, 0), radius=2, facecolor='gray', edgecolor='black', 
                    alpha=0.5, linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    
    # ラベル
    eye_name = "Left" if eye_side == 0 else "Right"
    side_name = "右側(鼻側)" if eye_side == 0 else "左側(鼻側)"
    ax.annotate(f'Mariotte\n({side_name})', xy=(mx, 0), xytext=(mx, -8),
                fontsize=8, ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))


def create_importance_visualization(pattern_name, importance_dict, output_path):
    """重要度マップの可視化（マリオット盲点位置を正しく表示）"""
    positions = importance_dict['positions']
    combined = importance_dict.get('combined', np.zeros(len(positions)))
    
    # 新しい4指標を取得（あれば）
    pred_std = importance_dict.get('pred_std_normalized', np.zeros(len(positions)))
    pred_error = importance_dict.get('prediction_error_normalized', np.zeros(len(positions)))
    gnn_correction = importance_dict.get('gnn_correction_normalized', np.zeros(len(positions)))
    leave_one_out = importance_dict.get('leave_one_out_normalized', np.zeros(len(positions)))
    
    # 旧形式対応
    if pred_std.sum() == 0:
        pred_error = importance_dict.get('error', np.zeros(len(positions)))
        gnn_correction = importance_dict.get('correction', np.zeros(len(positions)))
    
    n_total = len(positions)
    
    # eye_side判定
    if 'Left' in pattern_name:
        eye_side = 0
    elif 'Right' in pattern_name:
        eye_side = 1
    else:
        eye_side = 0  # デフォルト
    
    eye_name = "Left Eye" if eye_side == 0 else "Right Eye"
    pattern_type = pattern_name.replace('Left_', '').replace('Right_', '')
    
    # 軸範囲
    if 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        axis_range = 12
    else:
        axis_range = 32
    
    # 単体図（Combined Importanceのみ、マリオット盲点表示付き）
    fig, ax = plt.subplots(figsize=(10, 10))
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=combined, 
                        s=200, cmap='RdYlGn_r', vmin=0, vmax=10, edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(-axis_range, axis_range)
    ax.set_ylim(-axis_range, axis_range)
    ax.set_xlabel('X (degrees)', fontsize=14)
    ax.set_ylabel('Y (degrees)', fontsize=14)
    ax.set_title(f'{eye_name} - {pattern_type}\n({n_total} points)', fontsize=16)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # マリオット盲点インジケーター
    draw_mariotte_indicator(ax, eye_side, pattern_name)
    
    plt.colorbar(scatter, ax=ax, label='Combined Importance', shrink=0.8)
    
    plt.tight_layout()
    output_file = output_path / f"importance_{pattern_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file.name}")
    
    # 4パネル図
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. pred_std (不確実性)
    ax1 = axes[0, 0]
    sc1 = ax1.scatter(positions[:, 0], positions[:, 1], c=pred_std, 
                      s=150, cmap='Purples', vmin=0, vmax=10, edgecolors='black', linewidth=0.3)
    ax1.set_xlim(-axis_range, axis_range)
    ax1.set_ylim(-axis_range, axis_range)
    ax1.set_xlabel('X (degrees)')
    ax1.set_ylabel('Y (degrees)')
    ax1.set_title('1. Prediction Uncertainty (pred_std)\nHigher = needs more measurements', fontsize=11)
    ax1.set_aspect('equal')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    draw_mariotte_indicator(ax1, eye_side, pattern_name)
    plt.colorbar(sc1, ax=ax1)
    
    # 2. prediction_error (予測誤差)
    ax2 = axes[0, 1]
    sc2 = ax2.scatter(positions[:, 0], positions[:, 1], c=pred_error,
                      s=150, cmap='Reds', vmin=0, vmax=10, edgecolors='black', linewidth=0.3)
    ax2.set_xlim(-axis_range, axis_range)
    ax2.set_ylim(-axis_range, axis_range)
    ax2.set_xlabel('X (degrees)')
    ax2.set_ylabel('Y (degrees)')
    ax2.set_title('2. Prediction Error\nHigher = harder to estimate alone', fontsize=11)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    draw_mariotte_indicator(ax2, eye_side, pattern_name)
    plt.colorbar(sc2, ax=ax2)
    
    # 3. gnn_correction (GNN補正量)
    ax3 = axes[1, 0]
    sc3 = ax3.scatter(positions[:, 0], positions[:, 1], c=gnn_correction,
                      s=150, cmap='Blues', vmin=0, vmax=10, edgecolors='black', linewidth=0.3)
    ax3.set_xlim(-axis_range, axis_range)
    ax3.set_ylim(-axis_range, axis_range)
    ax3.set_xlabel('X (degrees)')
    ax3.set_ylabel('Y (degrees)')
    ax3.set_title('3. GNN Correction Amount\nHigher = neighbor info is useful', fontsize=11)
    ax3.set_aspect('equal')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    draw_mariotte_indicator(ax3, eye_side, pattern_name)
    plt.colorbar(sc3, ax=ax3)
    
    # 4. leave_one_out (除外影響度)
    ax4 = axes[1, 1]
    sc4 = ax4.scatter(positions[:, 0], positions[:, 1], c=leave_one_out,
                      s=150, cmap='Greens', vmin=0, vmax=10, edgecolors='black', linewidth=0.3)
    ax4.set_xlim(-axis_range, axis_range)
    ax4.set_ylim(-axis_range, axis_range)
    ax4.set_xlabel('X (degrees)')
    ax4.set_ylabel('Y (degrees)')
    ax4.set_title('4. Leave-One-Out Influence\nHigher = important info source', fontsize=11)
    ax4.set_aspect('equal')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    draw_mariotte_indicator(ax4, eye_side, pattern_name)
    plt.colorbar(sc4, ax=ax4)
    
    fig.suptitle(f'{eye_name} - {pattern_type} (4 Metrics)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_file2 = output_path / f"importance_4metrics_{pattern_name}.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file2.name}")


def verify_mariotte_exclusion(pattern_name, positions):
    """マリオット盲点が正しく除外されているか確認"""
    if 'Left' in pattern_name:
        eye_side = 0
        mariotte_x = 15
    else:
        eye_side = 1
        mariotte_x = -15
    
    # y=±3でマリオット盲点位置の点があるか確認
    for y in [3, -3]:
        points_at_y = positions[np.abs(positions[:, 1] - y) < 0.5]
        has_mariotte = any(np.abs(points_at_y[:, 0] - mariotte_x) < 0.5)
        
        expected = "should NOT exist" if 'Pattern10-2' not in pattern_name else "N/A"
        status = "EXCLUDED" if not has_mariotte else "EXISTS"
        
        if 'Pattern10-2' not in pattern_name:
            print(f"    ({mariotte_x}, {y}): {status} ({expected})")


if __name__ == "__main__":
    print("="*70)
    print("Visualizing Importance Maps with Correct Mariotte Position")
    print("="*70)
    
    print("\nMariotte Blind Spot Positions (Anatomical Facts):")
    print("  Left Eye:  (+15, ±3) - Right side (nasal)")
    print("  Right Eye: (-15, ±3) - Left side (nasal)")
    
    # TOP Strategy結果を探す
    importance_files = list(INPUT_PATH_TOP.glob("importance_map_*.pkl"))
    
    if len(importance_files) == 0:
        print(f"\nNo data found in {INPUT_PATH_TOP}")
        print("Trying alternative paths...")
        
        # 代替パス
        alt_paths = [
            GNN_PROJECT_PATH / "results" / "importance_maps_top",
            GNN_PROJECT_PATH / "results" / "by_eye_pattern_angular",
        ]
        for alt_path in alt_paths:
            importance_files = list(alt_path.glob("importance_map_*.pkl"))
            if importance_files:
                print(f"Found data in {alt_path}")
                break
    
    if len(importance_files) == 0:
        print("No importance map files found. Please run compute_importance first.")
        exit(1)
    
    print(f"\nFound {len(importance_files)} importance maps")
    
    for f in sorted(importance_files):
        pattern_name = f.stem.replace('importance_map_', '')
        
        print(f"\n{pattern_name}:")
        
        with open(f, 'rb') as fp:
            importance_dict = pickle.load(fp)
        
        positions = importance_dict['positions']
        print(f"  Points: {len(positions)}")
        
        # マリオット盲点除外の確認
        if 'Pattern10-2' not in pattern_name:
            print("  Mariotte exclusion check:")
            verify_mariotte_exclusion(pattern_name, positions)
        
        # 可視化
        create_importance_visualization(pattern_name, importance_dict, OUTPUT_PATH)
    
    print(f"\n{'='*70}")
    print("Completed!")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_PATH}")