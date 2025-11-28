# -*- coding: utf-8 -*-
"""
visualize_importance_by_eye_pattern.py
左右眼別・パターン別の重要度マップ可視化（タイムスタンプ付きファイル名版）

修正内容:
- ファイル名にタイムスタンプを付与して上書きを防止
- フォーマット: importance_map_{eye_pattern}_{timestamp}.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# タイムスタンプ生成（実行時に1回のみ生成）
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

INPUT_PATH = GNN_PROJECT_PATH / "results" / "importance_maps_by_eye_pattern"
# ★フォルダ名にもタイムスタンプを付与
OUTPUT_PATH = GNN_PROJECT_PATH / "visualizations" / f"importance_maps_by_eye_pattern_{TIMESTAMP}"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nPaths:")
print(f"  Project: {GNN_PROJECT_PATH}")
print(f"  Input: {INPUT_PATH}")
print(f"  Output: {OUTPUT_PATH}")
print(f"  Timestamp: {TIMESTAMP}")

# 可視化設定
REDUCTION_RATIO = 0.8  # 削減率（50%の点を残す）
FIGSIZE = (20, 10)
DPI = 300

# 日本語フォント設定
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


def load_importance_data(eye_pattern_name, input_path):
    """重要度データを読み込み"""
    pkl_file = input_path / f"importance_map_{eye_pattern_name}.pkl"
    
    if not pkl_file.exists():
        print(f"Warning: File not found: {pkl_file}")
        return None
    
    with open(pkl_file, 'rb') as f:
        importance_dict = pickle.load(f)
    
    return importance_dict


def create_importance_visualization(eye_pattern_name, importance_dict, output_path, timestamp):
    """
    重要度マップの4パネル可視化（タイムスタンプ付きファイル名）
    
    パネル構成：
    1. 全点の重要度マップ
    2. 削減プロトコル（重要な点のみ）
    3. 全点 vs 削減版の比較
    4. 重要度スコアの分布
    """
    
    if importance_dict['combined']['positions'] is None:
        print(f"Warning: No combined importance scores for {eye_pattern_name}")
        return
    
    positions = importance_dict['combined']['positions']
    scores = importance_dict['combined']['scores']
    
    # スコアで降順ソート
    sort_idx = np.argsort(scores)[::-1]
    positions_sorted = positions[sort_idx]
    scores_sorted = scores[sort_idx]
    
    # 削減後の点数を計算
    n_total = len(scores)
    n_essential = int(n_total * REDUCTION_RATIO)
    reduction_pct = (1 - REDUCTION_RATIO) * 100
    
    # Essential pointsのインデックス
    essential_indices = sort_idx[:n_essential]
    essential_positions = positions[essential_indices]
    essential_scores = scores[essential_indices]
    
    # カットオフ閾値
    threshold_score = scores_sorted[n_essential-1] if n_essential > 0 else scores_sorted[-1]
    
    # 図の作成
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ===== Panel 1: 全点の重要度マップ =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    scatter1 = ax1.scatter(
        positions[:, 0], 
        positions[:, 1],
        c=scores,
        cmap='RdYlGn_r',
        s=150,
        edgecolors='black',
        linewidths=1,
        vmin=0,
        vmax=10,
        alpha=0.8
    )
    
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Importance Score', fontsize=12)
    
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title('Importance Map - All Points', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # ===== Panel 2: 削減プロトコル =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.scatter(
        essential_positions[:, 0],
        essential_positions[:, 1],
        c='darkred',
        s=250,
        marker='s',
        edgecolors='black',
        linewidths=2,
        alpha=0.9,
        label='Essential'
    )
    
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Reduced Protocol ({n_essential} points, {reduction_pct:.0f}% reduction)', 
        fontsize=14, 
        fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='upper right', fontsize=11)
    
    # ===== Panel 3: 比較 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.scatter(
        positions[:, 0],
        positions[:, 1],
        c='lightgray',
        s=150,
        marker='o',
        edgecolors='gray',
        linewidths=0.5,
        alpha=0.5,
        label='All points'
    )
    
    ax3.scatter(
        essential_positions[:, 0],
        essential_positions[:, 1],
        c='darkred',
        s=250,
        marker='s',
        edgecolors='black',
        linewidths=2,
        alpha=0.9,
        label='Essential points'
    )
    
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title('Comparison: All vs Reduced', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend(loc='upper right', fontsize=11)
    
    # ===== Panel 4: スコア分布 =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    counts, bins, patches = ax4.hist(
        scores,
        bins=30,
        color='steelblue',
        edgecolor='black',
        alpha=0.7
    )
    
    ax4.axvline(
        threshold_score,
        color='red',
        linestyle='--',
        linewidth=2.5,
        label=f'Threshold (top {n_essential} points)'
    )
    
    ax4.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Importance Scores', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    stats_text = (
        f'Total points: {n_total}\n'
        f'Essential: {n_essential}\n'
        f'Reduction: {reduction_pct:.0f}%\n'
        f'Threshold: {threshold_score:.2f}\n'
        f'Note: Mariotte excluded'
    )
    ax4.text(
        0.98, 0.98,
        stats_text,
        transform=ax4.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10,
        family='monospace'
    )
    
    fig.suptitle(
        f'Visual Field Test Point Importance Analysis - {eye_pattern_name}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # ★シンプルなファイル名で保存（フォルダ名にタイムスタンプあり）
    output_file = output_path / f"importance_map_{eye_pattern_name}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: {output_file.name}")
    print(f"{eye_pattern_name} Summary:")
    print(f"  Total: {n_total}, Essential: {n_essential}, Reduction: {reduction_pct:.0f}%")


def create_comparison_by_eye(timestamp):
    """左右眼の比較（同じパターン）- タイムスタンプ付き"""
    patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    
    for pattern in patterns:
        left_file = INPUT_PATH / f"importance_map_Left_{pattern}.pkl"
        right_file = INPUT_PATH / f"importance_map_Right_{pattern}.pkl"
        
        if not (left_file.exists() and right_file.exists()):
            continue
        
        with open(left_file, 'rb') as f:
            left_data = pickle.load(f)
        
        with open(right_file, 'rb') as f:
            right_data = pickle.load(f)
        
        if left_data['combined']['positions'] is None or right_data['combined']['positions'] is None:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左眼
        ax1 = axes[0]
        positions_l = left_data['combined']['positions']
        scores_l = left_data['combined']['scores']
        
        scatter1 = ax1.scatter(
            positions_l[:, 0],
            positions_l[:, 1],
            c=scores_l,
            cmap='RdYlGn_r',
            s=150,
            edgecolors='black',
            linewidths=1,
            vmin=0,
            vmax=10,
            alpha=0.8
        )
        
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('X Position (degrees)', fontsize=11)
        ax1.set_ylabel('Y Position (degrees)', fontsize=11)
        ax1.set_title(f'Left Eye - {pattern}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 右眼
        ax2 = axes[1]
        positions_r = right_data['combined']['positions']
        scores_r = right_data['combined']['scores']
        
        scatter2 = ax2.scatter(
            positions_r[:, 0],
            positions_r[:, 1],
            c=scores_r,
            cmap='RdYlGn_r',
            s=150,
            edgecolors='black',
            linewidths=1,
            vmin=0,
            vmax=10,
            alpha=0.8
        )
        
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('X Position (degrees)', fontsize=11)
        ax2.set_ylabel('Y Position (degrees)', fontsize=11)
        ax2.set_title(f'Right Eye - {pattern}', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # カラーバー
        cbar = plt.colorbar(scatter2, ax=axes, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Importance Score', fontsize=12)
        
        fig.suptitle(f'Left vs Right Eye Comparison - {pattern}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # ★シンプルなファイル名で保存
        output_file = OUTPUT_PATH / f"comparison_LeftRight_{pattern}.png"
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison saved: {output_file.name}")


def create_comparison_by_pattern(timestamp):
    """パターン間の比較（同じ眼）- タイムスタンプ付き"""
    eyes = ['Left', 'Right']
    patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    
    for eye in eyes:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        valid_plots = []
        
        for idx, pattern in enumerate(patterns):
            pkl_file = INPUT_PATH / f"importance_map_{eye}_{pattern}.pkl"
            
            if not pkl_file.exists():
                continue
            
            with open(pkl_file, 'rb') as f:
                importance_dict = pickle.load(f)
            
            if importance_dict['combined']['positions'] is None:
                continue
            
            ax = axes[idx]
            positions = importance_dict['combined']['positions']
            scores = importance_dict['combined']['scores']
            
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                c=scores,
                cmap='RdYlGn_r',
                s=150,
                edgecolors='black',
                linewidths=1,
                vmin=0,
                vmax=10,
                alpha=0.8
            )
            
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('X Position (degrees)', fontsize=11)
            ax.set_ylabel('Y Position (degrees)', fontsize=11)
            ax.set_title(pattern, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            valid_plots.append(scatter)
        
        if len(valid_plots) > 0:
            cbar = plt.colorbar(valid_plots[-1], ax=axes, orientation='vertical', pad=0.05, shrink=0.8)
            cbar.set_label('Importance Score', fontsize=12)
            
            fig.suptitle(f'{eye} Eye - Pattern Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # ★シンプルなファイル名で保存
            output_file = OUTPUT_PATH / f"comparison_Patterns_{eye}.png"
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            plt.close()
            
            print(f"\nComparison saved: {output_file.name}")


def create_summary_report(timestamp):
    """サマリーレポートをテキストファイルで保存"""
    
    importance_files = list(INPUT_PATH.glob("importance_map_*.pkl"))
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("IMPORTANCE MAP ANALYSIS SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Timestamp: {timestamp}")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("Settings:")
    report_lines.append(f"  Reduction Ratio: {REDUCTION_RATIO*100:.0f}%")
    report_lines.append(f"  Mariotte Blind Spot: EXCLUDED")
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("RESULTS BY EYE AND PATTERN")
    report_lines.append("="*70)
    report_lines.append("")
    
    for importance_file in sorted(importance_files):
        eye_pattern_name = importance_file.stem.replace('importance_map_', '')
        
        with open(importance_file, 'rb') as f:
            importance_dict = pickle.load(f)
        
        if importance_dict['combined']['positions'] is None:
            continue
        
        positions = importance_dict['combined']['positions']
        scores = importance_dict['combined']['scores']
        
        n_total = len(scores)
        n_essential = int(n_total * REDUCTION_RATIO)
        reduction_pct = (1 - REDUCTION_RATIO) * 100
        
        sort_idx = np.argsort(scores)[::-1]
        scores_sorted = scores[sort_idx]
        threshold_score = scores_sorted[n_essential-1] if n_essential > 0 else scores_sorted[-1]
        
        report_lines.append(f"{eye_pattern_name}:")
        report_lines.append(f"  Total Points: {n_total}")
        report_lines.append(f"  Essential Points: {n_essential}")
        report_lines.append(f"  Reduction: {reduction_pct:.0f}%")
        report_lines.append(f"  Threshold Score: {threshold_score:.2f}")
        report_lines.append(f"  Score Range: {scores.min():.2f} - {scores.max():.2f}")
        report_lines.append(f"  Mean Score: {scores.mean():.2f}")
        report_lines.append(f"  Std Score: {scores.std():.2f}")
        report_lines.append("")
    
    report_lines.append("="*70)
    report_lines.append("OUTPUT FILES")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append("Individual importance maps:")
    for importance_file in sorted(importance_files):
        eye_pattern_name = importance_file.stem.replace('importance_map_', '')
        report_lines.append(f"  - importance_map_{eye_pattern_name}.png")
    
    report_lines.append("")
    report_lines.append("Comparison figures:")
    report_lines.append(f"  - comparison_LeftRight_Pattern30-2.png")
    report_lines.append(f"  - comparison_LeftRight_Pattern24-2.png")
    report_lines.append(f"  - comparison_LeftRight_Pattern10-2.png")
    report_lines.append(f"  - comparison_Patterns_Left.png")
    report_lines.append(f"  - comparison_Patterns_Right.png")
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    # レポートファイルを保存
    report_file = OUTPUT_PATH / f"analysis_summary.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nSummary report saved: {report_file.name}")


# メイン処理
if __name__ == "__main__":
    print("="*60)
    print("Visualizing Importance Maps by Eye and Pattern")
    print("★ Mariotte blind spot points are already EXCLUDED")
    print(f"★ Files will be saved with timestamp: {TIMESTAMP}")
    print("="*60)
    
    importance_files = list(INPUT_PATH.glob("importance_map_*.pkl"))
    
    if len(importance_files) == 0:
        print(f"\nError: No importance map data found in {INPUT_PATH}")
        print("Please run compute_importance_by_eye_pattern.py first")
        exit(1)
    
    print(f"\nFound {len(importance_files)} importance maps")
    
    # 個別可視化
    for importance_file in sorted(importance_files):
        eye_pattern_name = importance_file.stem.replace('importance_map_', '')
        
        print(f"\n{'='*60}")
        print(f"Visualizing: {eye_pattern_name}")
        print(f"{'='*60}")
        
        importance_dict = load_importance_data(eye_pattern_name, INPUT_PATH)
        
        if importance_dict is not None:
            create_importance_visualization(eye_pattern_name, importance_dict, OUTPUT_PATH, TIMESTAMP)
    
    # 比較図作成
    print(f"\n{'='*60}")
    print("Creating Comparison Figures")
    print(f"{'='*60}")
    
    print("\n1. Left vs Right Eye Comparison...")
    create_comparison_by_eye(TIMESTAMP)
    
    print("\n2. Pattern Comparison...")
    create_comparison_by_pattern(TIMESTAMP)
    
    # サマリーレポート作成
    print(f"\n{'='*60}")
    print("Creating Summary Report")
    print(f"{'='*60}")
    create_summary_report(TIMESTAMP)
    
    print(f"\n{'='*60}")
    print("Visualization Completed!")
    print(f"{'='*60}")
    print(f"Timestamp: {TIMESTAMP}")
    print(f"Output folder: {OUTPUT_PATH}")
    print(f"\nAll files have been saved in folder:")
    print(f"  {OUTPUT_PATH.relative_to(GNN_PROJECT_PATH)}")
    print(f"\nThis timestamped folder prevents overwriting previous results.")
