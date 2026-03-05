# -*- coding: utf-8 -*-
"""
visualize_importance_with_mariotte.py
重要度マップ可視化（マリオット盲点位置を正しく表示）

使用例:
  python visualize_importance_with_mariotte.py
  python visualize_importance_with_mariotte.py --data-suffix _angle45
  python visualize_importance_with_mariotte.py --data-suffix _angle60
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
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

# 日本語フォント設定（japanize_matplotlibがなくても動作する）
try:
    import japanize_matplotlib
except ImportError:
    pass

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

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
    
    circle = Circle((mx, 0), radius=2, facecolor='gray', edgecolor='black', 
                    alpha=0.5, linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    
    eye_name = "Left" if eye_side == 0 else "Right"
    side_name = "耳側(左)" if eye_side == 0 else "耳側(右)"
    ax.annotate(f'Mariotte\n({side_name})', xy=(mx, 0), xytext=(mx, -8),
                fontsize=8, ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))


def create_importance_visualization(pattern_name, importance_dict, output_path):
    """重要度マップの可視化（Combined + 4指標を1枚に統合、マリオット盲点付き）"""
    positions = importance_dict['positions']
    combined = importance_dict.get('combined', np.zeros(len(positions)))
    
    pred_std = importance_dict.get('pred_std_normalized', np.zeros(len(positions)))
    pred_error = importance_dict.get('prediction_error_normalized', np.zeros(len(positions)))
    gnn_correction = importance_dict.get('gnn_correction_normalized', np.zeros(len(positions)))
    leave_one_out = importance_dict.get('leave_one_out_normalized', np.zeros(len(positions)))
    
    # 旧形式対応
    if pred_std.sum() == 0:
        pred_error = importance_dict.get('error', np.zeros(len(positions)))
        gnn_correction = importance_dict.get('correction', np.zeros(len(positions)))
    
    n_total = len(positions)
    
    if 'Left' in pattern_name:
        eye_side = 0
    elif 'Right' in pattern_name:
        eye_side = 1
    else:
        eye_side = 0
    
    eye_name = "Left Eye" if eye_side == 0 else "Right Eye"
    pattern_type = pattern_name.replace('Left_', '').replace('Right_', '')
    
    if 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        axis_range = 12
    else:
        axis_range = 32
    
    # === 統合図: Combined（上段中央・大） + 4指標（下段2x2） ===
    fig = plt.figure(figsize=(20, 22))
    
    # --- 上段: Combined Score（中央に大きく配置） ---
    ax_combined = fig.add_axes([0.2, 0.55, 0.6, 0.38])  # [left, bottom, width, height]
    
    scatter = ax_combined.scatter(positions[:, 0], positions[:, 1], c=combined,
                                  s=250, cmap='RdYlGn_r', vmin=0, vmax=10,
                                  edgecolors='black', linewidth=0.5)
    ax_combined.set_xlim(-axis_range, axis_range)
    ax_combined.set_ylim(-axis_range, axis_range)
    ax_combined.set_xlabel('X (degrees)', fontsize=13)
    ax_combined.set_ylabel('Y (degrees)', fontsize=13)
    ax_combined.set_title(f'Combined Importance Score\n({n_total} points)', fontsize=15, fontweight='bold')
    ax_combined.set_aspect('equal')
    ax_combined.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax_combined.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax_combined.grid(True, alpha=0.15)
    draw_mariotte_indicator(ax_combined, eye_side, pattern_name)
    cbar = plt.colorbar(scatter, ax=ax_combined, shrink=0.8, pad=0.02)
    cbar.set_label('Importance (0-10)', fontsize=11)
    
    # --- 下段: 4指標（2x2） ---
    metrics = [
        (pred_std, 'Purples', '1. Prediction Uncertainty (pred_std)',
         'Higher = needs more measurements', 'Uncertainty'),
        (pred_error, 'Reds', '2. Prediction Error',
         'Higher = harder to estimate alone', 'Error'),
        (gnn_correction, 'Blues', '3. GNN Correction Amount',
         'Higher = neighbor info is useful', 'Correction'),
        (leave_one_out, 'Greens', '4. Leave-One-Out Influence',
         'Higher = important info source', 'Influence'),
    ]
    
    panel_positions = [
        [0.03, 0.03, 0.23, 0.42],   # 左下
        [0.27, 0.03, 0.23, 0.42],   # 中左下
        [0.51, 0.03, 0.23, 0.42],   # 中右下
        [0.75, 0.03, 0.23, 0.42],   # 右下
    ]
    
    for (values, cmap, title, subtitle, cbar_label), pos in zip(metrics, panel_positions):
        ax = fig.add_axes(pos)
        sc = ax.scatter(positions[:, 0], positions[:, 1], c=values,
                       s=80, cmap=cmap, vmin=0, vmax=10,
                       edgecolors='black', linewidth=0.3)
        ax.set_xlim(-axis_range, axis_range)
        ax.set_ylim(-axis_range, axis_range)
        ax.set_xlabel('X (degrees)', fontsize=9)
        ax.set_ylabel('Y (degrees)', fontsize=9)
        ax.set_title(f'{title}\n{subtitle}', fontsize=10)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=8)
        draw_mariotte_indicator(ax, eye_side, pattern_name)
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    
    fig.suptitle(f'{eye_name} - {pattern_type} (Combined + 4 Metrics + Mariotte)',
                 fontsize=17, fontweight='bold', y=0.97)
    
    output_file = output_path / f"importance_mariotte_{pattern_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_file.name}")


def verify_mariotte_exclusion(pattern_name, positions):
    """マリオット盲点が正しく除外されているか確認"""
    if 'Left' in pattern_name:
        eye_side = 0
        mariotte_positions = [(-15, 3), (-15, -3)]
    else:
        eye_side = 1
        mariotte_positions = [(15, 3), (15, -3)]
    
    for mx, my in mariotte_positions:
        points_near = positions[
            (np.abs(positions[:, 0] - mx) < 1.0) & 
            (np.abs(positions[:, 1] - my) < 1.0)
        ]
        status = "EXCLUDED ✓" if len(points_near) == 0 else "EXISTS ✗"
        
        if 'Pattern10-2' not in pattern_name:
            print(f"    ({mx}, {my}): {status}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='重要度マップ可視化（マリオット盲点位置を正しく表示）'
    )
    parser.add_argument('--data-suffix', '-d', type=str, default='',
                        help='データディレクトリのサフィックス (例: _angle45)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # パス設定（★サフィックス対応）
    INPUT_PATH = GNN_PROJECT_PATH / "results" / f"top_strategy{args.data_suffix}"
    OUTPUT_PATH = GNN_PROJECT_PATH / "visualizations" / f"importance_mariotte{args.data_suffix}_{TIMESTAMP}"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Visualizing Importance Maps with Correct Mariotte Position")
    print("=" * 70)
    print(f"\nInput path: {INPUT_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    
    print("\nMariotte Blind Spot Positions (Anatomical Facts):")
    print("  Left Eye:  (-15, ±3) - Left side (temporal)")
    print("  Right Eye: (+15, ±3) - Right side (temporal)")
    
    # 重要度マップファイル検索
    importance_files = list(INPUT_PATH.glob("importance_map_*.pkl"))
    
    if len(importance_files) == 0:
        print(f"\nNo data found in {INPUT_PATH}")
        
        # サフィックスなしのパスも試す
        alt_paths = [
            GNN_PROJECT_PATH / "results" / "top_strategy",
            GNN_PROJECT_PATH / "results" / "importance_maps_top",
        ]
        for alt_path in alt_paths:
            importance_files = list(alt_path.glob("importance_map_*.pkl"))
            if importance_files:
                INPUT_PATH = alt_path
                print(f"Found data in {alt_path}")
                break
    
    if len(importance_files) == 0:
        print("No importance map files found. Please run compute_importance_top_strategy.py first.")
        exit(1)
    
    print(f"\nFound {len(importance_files)} importance maps")
    
    for f in sorted(importance_files):
        pattern_name = f.stem.replace('importance_map_', '')
        
        print(f"\n{pattern_name}:")
        
        with open(f, 'rb') as fp:
            importance_dict = pickle.load(fp)
        
        positions = importance_dict['positions']
        print(f"  Points: {len(positions)}")
        
        if 'Pattern10-2' not in pattern_name:
            print("  Mariotte exclusion check:")
            verify_mariotte_exclusion(pattern_name, positions)
        
        create_importance_visualization(pattern_name, importance_dict, OUTPUT_PATH)
    
    print(f"\n{'=' * 70}")
    print("Completed!")
    print(f"{'=' * 70}")
    print(f"Output: {OUTPUT_PATH}")
