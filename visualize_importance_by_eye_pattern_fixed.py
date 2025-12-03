# -*- coding: utf-8 -*-
"""
visualize_importance_by_eye_pattern_fixed.py
標準グリッド座標を使用した重要度マップ可視化

修正内容:
- Humphrey視野検査の標準グリッドのみを使用
- 10-2: 68点
- 24-2: 54点（マリオット除外後52点）
- 30-2: 76点（マリオット除外後74点）
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

# タイムスタンプ
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# 入力パス（compute_importance_by_eye_pattern_fixed.pyの出力）
INPUT_PATH = GNN_PROJECT_PATH / "results" / "importance_maps_standard"

# 出力パス
OUTPUT_PATH = GNN_PROJECT_PATH / "visualizations" / f"importance_maps_by_eye_pattern_{TIMESTAMP}"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 可視化設定
REDUCTION_RATIO = 0.5  # 削減率（50%の点を残す）
FIGSIZE = (20, 10)
DPI = 300

print(f"\nPaths:")
print(f"  Input: {INPUT_PATH}")
print(f"  Output: {OUTPUT_PATH}")
print(f"  Timestamp: {TIMESTAMP}")


# ==================== 標準グリッド定義 ====================

def get_30_2_grid():
    """30-2パターン（76点）"""
    points = []
    for x in [-9, -3, 3, 9]:
        points.append((x, 21))
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, 15))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 9))
    for x in [-27, -21, -15, -9, -3, 3, 9, 15, 21, 27]:
        points.append((x, 3))
    for x in [-27, -21, -15, -9, -3, 3, 9, 15, 21, 27]:
        points.append((x, -3))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -9))
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, -15))
    for x in [-9, -3, 3, 9]:
        points.append((x, -21))
    for x in [-6, 0, 6]:
        points.append((x, 24))
    for x in [-12, -6, 0, 6, 12]:
        points.append((x, 18))
    for x in [-18, 18]:
        points.append((x, 12))
    for x in [-18, 18]:
        points.append((x, -12))
    for x in [-12, -6, 0, 6, 12]:
        points.append((x, -18))
    for x in [-6, 0, 6]:
        points.append((x, -24))
    return np.array(points, dtype=np.float32)


def get_24_2_grid():
    """24-2パターン（54点）"""
    points = []
    for x in [-9, -3, 3, 9]:
        points.append((x, 21))
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, 15))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 9))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 3))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -3))
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -9))
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, -15))
    for x in [-9, -3, 3, 9]:
        points.append((x, -21))
    points.append((0, 27))
    points.append((0, -27))
    return np.array(points, dtype=np.float32)


def get_10_2_grid():
    """10-2パターン（68点）"""
    points = []
    for x in [-5, -3, -1, 1, 3, 5]:
        points.append((x, 7))
    for x in [-7, -5, -3, -1, 1, 3, 5, 7]:
        points.append((x, 5))
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, 3))
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, 1))
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, -1))
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, -3))
    for x in [-7, -5, -3, -1, 1, 3, 5, 7]:
        points.append((x, -5))
    for x in [-5, -3, -1, 1, 3, 5]:
        points.append((x, -7))
    return np.array(points, dtype=np.float32)


def get_mariotte_blind_spot(eye_side):
    """マリオット盲点の座標"""
    if eye_side == 0:
        return np.array([[15, 3], [15, -3]], dtype=np.float32)
    else:
        return np.array([[-15, 3], [-15, -3]], dtype=np.float32)


def get_standard_grid(pattern_name, eye_side=None, exclude_mariotte=True):
    """標準グリッドを取得"""
    if 'Pattern30-2' in pattern_name or '30-2' in pattern_name:
        grid = get_30_2_grid()
    elif 'Pattern24-2' in pattern_name or '24-2' in pattern_name:
        grid = get_24_2_grid()
    elif 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        grid = get_10_2_grid()
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    
    if exclude_mariotte and eye_side is not None and '10-2' not in pattern_name:
        mariotte = get_mariotte_blind_spot(eye_side)
        mask = np.ones(len(grid), dtype=bool)
        for m_point in mariotte:
            distances = np.linalg.norm(grid - m_point, axis=1)
            mask &= (distances > 0.5)
        grid = grid[mask]
    
    return grid


def get_expected_point_count(pattern_name, exclude_mariotte=True):
    """期待される点数"""
    if 'Pattern30-2' in pattern_name:
        return 74 if exclude_mariotte else 76
    elif 'Pattern24-2' in pattern_name:
        return 52 if exclude_mariotte else 54
    elif 'Pattern10-2' in pattern_name:
        return 68
    return 0


def get_eye_side_from_name(pattern_name):
    """パターン名から眼のサイドを取得"""
    if 'Left' in pattern_name:
        return 0
    elif 'Right' in pattern_name:
        return 1
    return None


# ==================== 可視化関数 ====================

def load_importance_data(eye_pattern_name, input_path):
    """重要度データを読み込み"""
    pkl_file = input_path / f"importance_map_{eye_pattern_name}.pkl"
    
    if not pkl_file.exists():
        print(f"Warning: File not found: {pkl_file}")
        return None
    
    with open(pkl_file, 'rb') as f:
        importance_dict = pickle.load(f)
    
    return importance_dict


def create_importance_visualization(eye_pattern_name, importance_dict, output_path):
    """
    重要度マップの4パネル可視化
    """
    # データ取得（2つの形式に対応）
    if 'combined_dict' in importance_dict:
        positions = importance_dict['combined_dict']['positions']
        scores = importance_dict['combined_dict']['scores']
    elif 'combined' in importance_dict and isinstance(importance_dict['combined'], dict):
        positions = importance_dict['combined']['positions']
        scores = importance_dict['combined']['scores']
    else:
        positions = importance_dict['positions']
        scores = importance_dict['combined']
    
    if positions is None:
        print(f"Warning: No positions data for {eye_pattern_name}")
        return
    
    n_total = len(scores)
    n_essential = int(n_total * REDUCTION_RATIO)
    reduction_pct = (1 - REDUCTION_RATIO) * 100
    
    # スコアでソート
    sort_idx = np.argsort(scores)[::-1]
    positions_sorted = positions[sort_idx]
    scores_sorted = scores[sort_idx]
    
    # Essential points
    essential_indices = sort_idx[:n_essential]
    essential_positions = positions[essential_indices]
    threshold_score = scores_sorted[n_essential-1] if n_essential > 0 else 0
    
    # 図の作成
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # === Panel 1: 全点の重要度マップ ===
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(
        positions[:, 0], positions[:, 1],
        c=scores, cmap='RdYlGn_r', s=150,
        edgecolors='black', linewidths=1,
        vmin=0, vmax=10, alpha=0.8
    )
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Importance Score', fontsize=12)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Importance Map - All {n_total} Points', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # === Panel 2: 削減プロトコル ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(
        essential_positions[:, 0], essential_positions[:, 1],
        c='darkred', s=250, marker='s',
        edgecolors='black', linewidths=2, alpha=0.9,
        label='Essential'
    )
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax2.set_title(
        f'Reduced Protocol ({n_essential} points, {reduction_pct:.0f}% reduction)',
        fontsize=14, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(loc='upper right', fontsize=11)
    
    # 軸範囲設定
    if 'Pattern10-2' in eye_pattern_name:
        for ax in [ax1, ax2]:
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
    elif 'Pattern24-2' in eye_pattern_name:
        for ax in [ax1, ax2]:
            ax.set_xlim(-30, 30)
            ax.set_ylim(-32, 32)
    else:
        for ax in [ax1, ax2]:
            ax.set_xlim(-32, 32)
            ax.set_ylim(-30, 30)
    
    # === Panel 3: 比較 ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        positions[:, 0], positions[:, 1],
        c='lightgray', s=150, marker='o',
        edgecolors='gray', linewidths=0.5, alpha=0.5, label='All points'
    )
    ax3.scatter(
        essential_positions[:, 0], essential_positions[:, 1],
        c='darkred', s=250, marker='s',
        edgecolors='black', linewidths=2, alpha=0.9, label='Essential points'
    )
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.set_xlabel('X Position (degrees)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Y Position (degrees)', fontsize=12, fontweight='bold')
    ax3.set_title('Comparison: All vs Reduced', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    ax3.legend(loc='upper right', fontsize=11)
    
    # === Panel 4: スコア分布 ===
    ax4 = fig.add_subplot(gs[1, 1])
    counts, bins, patches = ax4.hist(
        scores, bins=20, color='steelblue', edgecolor='black', alpha=0.7
    )
    ax4.axvline(
        threshold_score, color='red', linestyle='--', linewidth=2.5,
        label=f'Threshold (top {n_essential} points)'
    )
    ax4.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Importance Scores', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 統計情報
    expected = get_expected_point_count(eye_pattern_name, exclude_mariotte=True)
    stats_text = (
        f'Total points: {n_total}\n'
        f'Expected: {expected}\n'
        f'Essential: {n_essential}\n'
        f'Reduction: {reduction_pct:.0f}%\n'
        f'Threshold: {threshold_score:.2f}\n'
        f'Note: Mariotte excluded'
    )
    ax4.text(
        0.98, 0.98, stats_text, transform=ax4.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, family='monospace'
    )
    
    fig.suptitle(
        f'Visual Field Test Point Importance Analysis - {eye_pattern_name}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    output_file = output_path / f"importance_map_{eye_pattern_name}.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_file.name}")
    print(f"  {eye_pattern_name}: Total={n_total}, Essential={n_essential}, Reduction={reduction_pct:.0f}%")


def create_comparison_by_eye(input_path, output_path):
    """左右眼の比較（同じパターン）"""
    patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    
    for pattern in patterns:
        left_data = load_importance_data(f"Left_{pattern}", input_path)
        right_data = load_importance_data(f"Right_{pattern}", input_path)
        
        if left_data is None or right_data is None:
            continue
        
        # データ取得
        if 'combined_dict' in left_data:
            left_positions = left_data['combined_dict']['positions']
            left_scores = left_data['combined_dict']['scores']
        else:
            left_positions = left_data['positions']
            left_scores = left_data['combined']
        
        if 'combined_dict' in right_data:
            right_positions = right_data['combined_dict']['positions']
            right_scores = right_data['combined_dict']['scores']
        else:
            right_positions = right_data['positions']
            right_scores = right_data['combined']
        
        if left_positions is None or right_positions is None:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 左眼
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            left_positions[:, 0], left_positions[:, 1],
            c=left_scores, cmap='RdYlGn_r', s=150,
            edgecolors='black', linewidths=1,
            vmin=0, vmax=10, alpha=0.8
        )
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('X Position (degrees)', fontsize=11)
        ax1.set_ylabel('Y Position (degrees)', fontsize=11)
        ax1.set_title(f'Left Eye - {pattern}\n({len(left_positions)} points)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 右眼
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            right_positions[:, 0], right_positions[:, 1],
            c=right_scores, cmap='RdYlGn_r', s=150,
            edgecolors='black', linewidths=1,
            vmin=0, vmax=10, alpha=0.8
        )
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax2.set_xlabel('X Position (degrees)', fontsize=11)
        ax2.set_ylabel('Y Position (degrees)', fontsize=11)
        ax2.set_title(f'Right Eye - {pattern}\n({len(right_positions)} points)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # 軸範囲
        if 'Pattern10-2' in pattern:
            for ax in axes:
                ax.set_xlim(-12, 12)
                ax.set_ylim(-12, 12)
        else:
            for ax in axes:
                ax.set_xlim(-32, 32)
                ax.set_ylim(-32, 32)
        
        cbar = plt.colorbar(scatter2, ax=axes, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Importance Score', fontsize=12)
        
        fig.suptitle(f'Left vs Right Eye Comparison - {pattern}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_path / f"comparison_LeftRight_{pattern}.png"
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file.name}")


def create_comparison_by_pattern(input_path, output_path):
    """パターン間の比較（同じ眼）"""
    eyes = ['Left', 'Right']
    patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    
    for eye in eyes:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        valid_plots = []
        
        for idx, pattern in enumerate(patterns):
            importance_dict = load_importance_data(f"{eye}_{pattern}", input_path)
            
            if importance_dict is None:
                continue
            
            # データ取得
            if 'combined_dict' in importance_dict:
                positions = importance_dict['combined_dict']['positions']
                scores = importance_dict['combined_dict']['scores']
            else:
                positions = importance_dict['positions']
                scores = importance_dict['combined']
            
            if positions is None:
                continue
            
            ax = axes[idx]
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1],
                c=scores, cmap='RdYlGn_r', s=150,
                edgecolors='black', linewidths=1,
                vmin=0, vmax=10, alpha=0.8
            )
            
            ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.set_xlabel('X Position (degrees)', fontsize=11)
            ax.set_ylabel('Y Position (degrees)', fontsize=11)
            ax.set_title(f'{pattern}\n({len(positions)} points)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # 軸範囲
            if 'Pattern10-2' in pattern:
                ax.set_xlim(-12, 12)
                ax.set_ylim(-12, 12)
            else:
                ax.set_xlim(-32, 32)
                ax.set_ylim(-32, 32)
            
            valid_plots.append(scatter)
        
        if len(valid_plots) > 0:
            cbar = plt.colorbar(valid_plots[-1], ax=axes, orientation='vertical', pad=0.05, shrink=0.8)
            cbar.set_label('Importance Score', fontsize=12)
            
            fig.suptitle(f'{eye} Eye - Pattern Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_file = output_path / f"comparison_Patterns_{eye}.png"
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved: {output_file.name}")


def create_summary_report(input_path, output_path):
    """サマリーレポート作成（トレーニング結果を含む）"""
    importance_files = list(input_path.glob("importance_map_*.pkl"))
    
    # トレーニング結果を読み込み
    training_results_path = GNN_PROJECT_PATH / "results" / "by_eye_pattern" / "training_results_by_eye_pattern.csv"
    training_results = {}
    
    if training_results_path.exists():
        try:
            df_training = pd.read_csv(training_results_path)
            for _, row in df_training.iterrows():
                name = row['eye_pattern_name']
                training_results[name] = {
                    'best_val_mae': row['best_val_mae'],
                    'test_mae': row['test_mae'],
                    'test_rmse': row['test_rmse'],
                    'n_train': row.get('n_train', 'N/A'),
                    'n_val': row.get('n_val', 'N/A'),
                    'n_test': row.get('n_test', 'N/A')
                }
        except Exception as e:
            print(f"Warning: Could not load training results: {e}")
    
    report_lines = [
        "=" * 70,
        "IMPORTANCE MAP ANALYSIS SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timestamp: {TIMESTAMP}",
        "=" * 70,
        "",
        "Standard Grid Point Counts:",
        "  Pattern30-2: 76 points (74 excluding Mariotte)",
        "  Pattern24-2: 54 points (52 excluding Mariotte)",
        "  Pattern10-2: 68 points",
        "",
        f"Reduction Ratio: {REDUCTION_RATIO * 100:.0f}%",
        ""
    ]
    
    # ★ トレーニング結果セクションを追加
    if training_results:
        report_lines.extend([
            "=" * 70,
            "Training Summary",
            "=" * 70,
        ])
        
        for name in sorted(training_results.keys()):
            result = training_results[name]
            report_lines.extend([
                f"{name}:",
                f"  Best Val MAE: {result['best_val_mae']:.2f} dB",
                f"  Test MAE: {result['test_mae']:.2f} dB",
                f"  Test RMSE: {result['test_rmse']:.2f} dB",
                f"  Data: Train={result['n_train']}, Val={result['n_val']}, Test={result['n_test']}"
            ])
        
        report_lines.append("")
    
    report_lines.extend([
        "=" * 70,
        "RESULTS BY EYE AND PATTERN",
        "=" * 70,
        ""
    ])
    
    for importance_file in sorted(importance_files):
        eye_pattern_name = importance_file.stem.replace('importance_map_', '')
        
        with open(importance_file, 'rb') as f:
            importance_dict = pickle.load(f)
        
        # データ取得
        if 'combined_dict' in importance_dict:
            positions = importance_dict['combined_dict']['positions']
            scores = importance_dict['combined_dict']['scores']
        else:
            positions = importance_dict['positions']
            scores = importance_dict['combined']
        
        if positions is None:
            continue
        
        n_total = len(scores)
        n_essential = int(n_total * REDUCTION_RATIO)
        reduction_pct = (1 - REDUCTION_RATIO) * 100
        expected = get_expected_point_count(eye_pattern_name, exclude_mariotte=True)
        
        sort_idx = np.argsort(scores)[::-1]
        scores_sorted = scores[sort_idx]
        threshold_score = scores_sorted[n_essential-1] if n_essential > 0 else 0
        
        report_lines.extend([
            f"{eye_pattern_name}:",
            f"  Total Points: {n_total} (expected: {expected})",
            f"  Essential Points: {n_essential}",
            f"  Reduction: {reduction_pct:.0f}%",
            f"  Threshold Score: {threshold_score:.2f}",
            f"  Score Range: {scores.min():.2f} - {scores.max():.2f}",
            f"  Mean Score: {scores.mean():.2f}",
            ""
        ])
    
    report_lines.extend([
        "=" * 70,
        "OUTPUT FILES",
        "=" * 70,
        ""
    ])
    
    for f in sorted(OUTPUT_PATH.glob("*.png")):
        report_lines.append(f"  - {f.name}")
    
    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])
    
    report_file = output_path / "analysis_summary.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Summary saved: {report_file.name}")


# ==================== メイン処理 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Visualizing Importance Maps (Standard Grid)")
    print("=" * 70)
    print(f"Standard Point Counts:")
    print(f"  30-2: 76 points (74 excluding Mariotte)")
    print(f"  24-2: 54 points (52 excluding Mariotte)")
    print(f"  10-2: 68 points")
    
    # 入力ファイル確認
    importance_files = list(INPUT_PATH.glob("importance_map_*.pkl"))
    
    if len(importance_files) == 0:
        print(f"\nError: No importance map data found in {INPUT_PATH}")
        print("Please run compute_importance_by_eye_pattern_fixed.py first")
        exit(1)
    
    print(f"\nFound {len(importance_files)} importance maps")
    
    # 個別可視化
    print(f"\n{'='*70}")
    print("Creating Individual Visualizations")
    print(f"{'='*70}")
    
    for importance_file in sorted(importance_files):
        eye_pattern_name = importance_file.stem.replace('importance_map_', '')
        importance_dict = load_importance_data(eye_pattern_name, INPUT_PATH)
        
        if importance_dict is not None:
            create_importance_visualization(eye_pattern_name, importance_dict, OUTPUT_PATH)
    
    # 比較図作成
    print(f"\n{'='*70}")
    print("Creating Comparison Figures")
    print(f"{'='*70}")
    
    print("\n1. Left vs Right Eye Comparison...")
    create_comparison_by_eye(INPUT_PATH, OUTPUT_PATH)
    
    print("\n2. Pattern Comparison...")
    create_comparison_by_pattern(INPUT_PATH, OUTPUT_PATH)
    
    # サマリーレポート
    print(f"\n{'='*70}")
    print("Creating Summary Report")
    print(f"{'='*70}")
    create_summary_report(INPUT_PATH, OUTPUT_PATH)
    
    print(f"\n{'='*70}")
    print("Visualization Completed!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_PATH}")