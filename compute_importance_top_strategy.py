# -*- coding: utf-8 -*-
"""
compute_importance_top_strategy.py
TOP Strategy GNNの重要度スコア計算（4指標版・角度パラメータ対応）

使用例:
  python compute_importance_top_strategy.py
  python compute_importance_top_strategy.py --data-suffix _angle45
  python compute_importance_top_strategy.py --data-suffix _angle30 --mc-samples 50

重要度指標:
1. pred_std: Monte Carlo Dropoutによる予測の不確実性（標準偏差）
   - 高い = 追加測定が有効
2. prediction_error: |predicted - ground_truth| 実際の予測誤差
   - 高い = 単独では推定困難
3. gnn_correction: |predicted - gap_sensitivity| GNNによる補正量
   - 高い = 隣接情報が有効
4. leave_one_out: 各点を除外した時の他点への影響度
   - 高い = 情報源として重要

出力: analysis_summary_{pattern}.csv
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime

# 設定ファイル読み込み
try:
    from config_top_strategy import (
        SENSITIVITY_MIN, SENSITIVITY_MAX,
        MC_SAMPLES,
        WEIGHT_PRED_STD, WEIGHT_PREDICTION_ERROR,
        WEIGHT_GNN_CORRECTION, WEIGHT_LEAVE_ONE_OUT
    )
except ImportError:
    SENSITIVITY_MIN = 0.0
    SENSITIVITY_MAX = 40.0
    MC_SAMPLES = 30
    WEIGHT_PRED_STD = 0.30
    WEIGHT_PREDICTION_ERROR = 0.25
    WEIGHT_GNN_CORRECTION = 0.25
    WEIGHT_LEAVE_ONE_OUT = 0.20

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 標準グリッド定義 ====================

def get_10_2_grid(eye_side=0):
    """10-2パターンの標準グリッド（68点）"""
    points = [
        [-1, 9], [1, 9],
        [-5, 7], [-3, 7], [-1, 7], [1, 7], [3, 7], [5, 7],
        [-7, 5], [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5], [7, 5],
        [-7, 3], [-5, 3], [-3, 3], [-1, 3], [1, 3], [3, 3], [5, 3], [7, 3],
        [-9, 1], [-7, 1], [-5, 1], [-3, 1], [-1, 1], [1, 1], [3, 1], [5, 1], [7, 1], [9, 1],
        [-9, -1], [-7, -1], [-5, -1], [-3, -1], [-1, -1], [1, -1], [3, -1], [5, -1], [7, -1], [9, -1],
        [-7, -3], [-5, -3], [-3, -3], [-1, -3], [1, -3], [3, -3], [5, -3], [7, -3],
        [-7, -5], [-5, -5], [-3, -5], [-1, -5], [1, -5], [3, -5], [5, -5], [7, -5],
        [-5, -7], [-3, -7], [-1, -7], [1, -7], [3, -7], [5, -7],
        [-1, -9], [1, -9],
    ]
    return np.array(points, dtype=np.float32)


def get_24_2_grid(eye_side=0):
    """24-2パターンの標準グリッド（54点）"""
    if eye_side == 0:
        points = [
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
            [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    else:
        points = [
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3],
            [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3],
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    return np.array(points, dtype=np.float32)


def get_30_2_grid(eye_side=0):
    """30-2パターンの標準グリッド（76点）"""
    points = [
        [-9, 27], [-3, 27], [3, 27], [9, 27],
        [-15, 21], [-9, 21], [-3, 21], [3, 21], [9, 21], [15, 21],
        [-21, 15], [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15], [21, 15],
        [-27, 9], [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9], [27, 9],
        [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
        [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
        [-27, -9], [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9], [27, -9],
        [-21, -15], [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15], [21, -15],
        [-15, -21], [-9, -21], [-3, -21], [3, -21], [9, -21], [15, -21],
        [-9, -27], [-3, -27], [3, -27], [9, -27],
    ]
    return np.array(points, dtype=np.float32)


def get_mariotte_blind_spot(eye_side):
    """マリオット盲点の位置"""
    if eye_side == 0:
        return np.array([[-15, 3], [-15, -3]], dtype=np.float32)
    else:
        return np.array([[15, 3], [15, -3]], dtype=np.float32)


def get_standard_grid(pattern_name, eye_side=0, exclude_mariotte=True):
    """標準グリッドを取得"""
    if 'Pattern30-2' in pattern_name:
        grid = get_30_2_grid(eye_side)
    elif 'Pattern24-2' in pattern_name:
        grid = get_24_2_grid(eye_side)
    elif 'Pattern10-2' in pattern_name:
        grid = get_10_2_grid(eye_side)
    else:
        grid = get_30_2_grid(eye_side)
    
    if exclude_mariotte and 'Pattern10-2' not in pattern_name:
        mariotte = get_mariotte_blind_spot(eye_side)
        mask = np.ones(len(grid), dtype=bool)
        for mp in mariotte:
            distances = np.sqrt(np.sum((grid - mp) ** 2, axis=1))
            mask &= (distances > 1.0)
        grid = grid[mask]
    
    return grid


# ==================== モデル定義 ====================

class TOPStrategyGNN(nn.Module):
    """TOP Strategy GNN"""
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.3, edge_dim=4):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.input_lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=4, concat=False, 
                        dropout=dropout, edge_dim=edge_dim)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        
        self.output_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        self.gap_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, edge_index, edge_attr=None):
        gap_sensitivity = x[:, 0:1] * SENSITIVITY_MAX
        
        h = self.input_lin(x)
        
        for i in range(self.num_layers):
            h_res = h
            h = self.gnn_layers[i](h, edge_index, edge_attr=edge_attr)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res
        
        pred = self.output_lin(h)
        
        gap_w = torch.sigmoid(self.gap_weight)
        final_pred = gap_w * gap_sensitivity + (1 - gap_w) * pred * SENSITIVITY_MAX
        
        final_pred = torch.clamp(final_pred.squeeze(-1), SENSITIVITY_MIN, SENSITIVITY_MAX)
        
        return final_pred


# ==================== 重要度計算（4指標） ====================

def compute_pred_std(model, graph_list, device, n_samples):
    """指標1: pred_std - Monte Carlo Dropoutによる予測の不確実性"""
    print(f"\n  Computing pred_std (MC Dropout, {n_samples} samples)...")
    
    position_std = {}
    
    for graph in tqdm(graph_list, desc="  pred_std"):
        graph = graph.to(device)
        n_nodes = graph.num_nodes
        positions = graph.pos.cpu().numpy()
        
        predictions = []
        model.train()
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(graph.x, graph.edge_index, graph.edge_attr)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        node_std = np.std(predictions, axis=0)
        
        for i in range(n_nodes):
            pos = tuple(np.round(positions[i]).astype(int))
            if pos not in position_std:
                position_std[pos] = []
            position_std[pos].append(node_std[i])
    
    model.eval()
    return position_std


def compute_prediction_error(model, graph_list, device):
    """指標2: prediction_error - |predicted - ground_truth|"""
    print("\n  Computing prediction_error...")
    
    position_error = {}
    
    model.eval()
    with torch.no_grad():
        for graph in tqdm(graph_list, desc="  prediction_error"):
            graph = graph.to(device)
            positions = graph.pos.cpu().numpy()
            
            pred = model(graph.x, graph.edge_index, graph.edge_attr)
            target = graph.y
            
            errors = torch.abs(pred - target).cpu().numpy()
            
            for i in range(len(positions)):
                pos = tuple(np.round(positions[i]).astype(int))
                if pos not in position_error:
                    position_error[pos] = []
                position_error[pos].append(errors[i])
    
    return position_error


def compute_gnn_correction(model, graph_list, device):
    """指標3: gnn_correction - |predicted - gap_sensitivity|"""
    print("\n  Computing gnn_correction...")
    
    position_correction = {}
    
    model.eval()
    with torch.no_grad():
        for graph in tqdm(graph_list, desc="  gnn_correction"):
            graph = graph.to(device)
            positions = graph.pos.cpu().numpy()
            
            pred = model(graph.x, graph.edge_index, graph.edge_attr)
            gap_sens = graph.x[:, 0] * SENSITIVITY_MAX
            
            corrections = torch.abs(pred - gap_sens).cpu().numpy()
            
            for i in range(len(positions)):
                pos = tuple(np.round(positions[i]).astype(int))
                if pos not in position_correction:
                    position_correction[pos] = []
                position_correction[pos].append(corrections[i])
    
    return position_correction


def compute_leave_one_out(model, graph_list, device, max_graphs=100):
    """指標4: leave_one_out - 各点を除外した時の影響度"""
    print(f"\n  Computing leave_one_out (max {max_graphs} graphs)...")
    
    position_influence = {}
    
    model.eval()
    
    if len(graph_list) > max_graphs:
        indices = np.random.choice(len(graph_list), max_graphs, replace=False)
        selected_graphs = [graph_list[i] for i in indices]
    else:
        selected_graphs = graph_list
    
    with torch.no_grad():
        for graph in tqdm(selected_graphs, desc="  leave_one_out"):
            graph = graph.to(device)
            n_nodes = graph.num_nodes
            positions = graph.pos.cpu().numpy()
            
            baseline_pred = model(graph.x, graph.edge_index, graph.edge_attr).cpu().numpy()
            
            for i in range(n_nodes):
                x_masked = graph.x.clone()
                x_masked[i, :] = 0
                
                masked_pred = model(x_masked, graph.edge_index, graph.edge_attr).cpu().numpy()
                
                influence = np.abs(baseline_pred - masked_pred)
                influence[i] = 0
                total_influence = np.mean(influence)
                
                pos = tuple(np.round(positions[i]).astype(int))
                if pos not in position_influence:
                    position_influence[pos] = []
                position_influence[pos].append(total_influence)
    
    return position_influence


def aggregate_scores(position_scores_dict, standard_grid):
    """位置ごとのスコアを標準グリッドに集約"""
    n_points = len(standard_grid)
    aggregated = np.zeros(n_points)
    counts = np.zeros(n_points)
    
    for i, pos in enumerate(standard_grid):
        pos_key = tuple(np.round(pos).astype(int))
        
        matched = None
        for key in position_scores_dict:
            if abs(key[0] - pos_key[0]) <= 1 and abs(key[1] - pos_key[1]) <= 1:
                matched = key
                break
        
        if matched and len(position_scores_dict[matched]) > 0:
            aggregated[i] = np.mean(position_scores_dict[matched])
            counts[i] = len(position_scores_dict[matched])
    
    return aggregated, counts


def normalize_scores(scores):
    """スコアを0-10に正規化"""
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min()) * 10
    return scores


# ==================== 結果保存 ====================

def save_analysis_summary(pattern_name, standard_grid, 
                          pred_std, prediction_error, gnn_correction, leave_one_out,
                          counts, output_path):
    """analysis_summary CSVを保存"""
    
    pred_std_norm = normalize_scores(pred_std)
    error_norm = normalize_scores(prediction_error)
    correction_norm = normalize_scores(gnn_correction)
    loo_norm = normalize_scores(leave_one_out)
    
    combined = (
        WEIGHT_PRED_STD * pred_std_norm +
        WEIGHT_PREDICTION_ERROR * error_norm +
        WEIGHT_GNN_CORRECTION * correction_norm +
        WEIGHT_LEAVE_ONE_OUT * loo_norm
    )
    
    df = pd.DataFrame({
        'X': standard_grid[:, 0].astype(int),
        'Y': standard_grid[:, 1].astype(int),
        'pred_std': pred_std,
        'pred_std_normalized': pred_std_norm,
        'prediction_error': prediction_error,
        'prediction_error_normalized': error_norm,
        'gnn_correction': gnn_correction,
        'gnn_correction_normalized': correction_norm,
        'leave_one_out': leave_one_out,
        'leave_one_out_normalized': loo_norm,
        'combined_score': combined,
        'sample_count': counts.astype(int)
    })
    
    df = df.sort_values('combined_score', ascending=False)
    
    csv_file = output_path / f"analysis_summary_{pattern_name}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"  ✓ Saved: {csv_file.name}")
    
    print(f"\n  Top 5 important positions for {pattern_name}:")
    print(df[['X', 'Y', 'pred_std_normalized', 'prediction_error_normalized', 
              'gnn_correction_normalized', 'leave_one_out_normalized', 'combined_score']].head().to_string(index=False))
    
    return df


def save_importance_pickle(pattern_name, importance_data, output_path):
    """Pickle形式で保存"""
    pkl_file = output_path / f"importance_map_{pattern_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(importance_data, f)
    print(f"  ✓ Saved: {pkl_file.name}")


# ==================== メイン ====================

def process_pattern(pattern_name, model, graph_list, device, output_path, mc_samples):
    """パターンごとの処理"""
    print(f"\n{'='*60}")
    print(f"Processing: {pattern_name}")
    print(f"  Graphs: {len(graph_list)}")
    
    eye_side = 0 if 'Left' in pattern_name else 1
    pattern_type = pattern_name.split('_')[1] if '_' in pattern_name else pattern_name
    
    standard_grid = get_standard_grid(pattern_type, eye_side, exclude_mariotte=True)
    print(f"  Standard grid: {len(standard_grid)} points")
    
    pred_std_raw = compute_pred_std(model, graph_list, device, mc_samples)
    prediction_error_raw = compute_prediction_error(model, graph_list, device)
    gnn_correction_raw = compute_gnn_correction(model, graph_list, device)
    leave_one_out_raw = compute_leave_one_out(model, graph_list, device)
    
    pred_std, counts1 = aggregate_scores(pred_std_raw, standard_grid)
    prediction_error, counts2 = aggregate_scores(prediction_error_raw, standard_grid)
    gnn_correction, counts3 = aggregate_scores(gnn_correction_raw, standard_grid)
    leave_one_out, counts4 = aggregate_scores(leave_one_out_raw, standard_grid)
    
    counts = np.maximum(np.maximum(np.maximum(counts1, counts2), counts3), counts4)
    
    df = save_analysis_summary(
        pattern_name, standard_grid,
        pred_std, prediction_error, gnn_correction, leave_one_out,
        counts, output_path
    )
    
    importance_data = {
        'positions': standard_grid,
        'pred_std': pred_std,
        'pred_std_normalized': normalize_scores(pred_std),
        'prediction_error': prediction_error,
        'prediction_error_normalized': normalize_scores(prediction_error),
        'gnn_correction': gnn_correction,
        'gnn_correction_normalized': normalize_scores(gnn_correction),
        'leave_one_out': leave_one_out,
        'leave_one_out_normalized': normalize_scores(leave_one_out),
        'combined': df['combined_score'].values,
        'count': counts
    }
    save_importance_pickle(pattern_name, importance_data, output_path)
    
    return df


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNNの重要度スコア計算',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-suffix', '-d', type=str, default='',
                        help='データディレクトリのサフィックス (例: _angle45)')
    
    parser.add_argument('--mc-samples', '-m', type=int, default=MC_SAMPLES,
                        help=f'Monte Carlo Dropoutのサンプル数。デフォルト: {MC_SAMPLES}')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # パス設定
    DATA_PATH = GNN_PROJECT_PATH / "data" / f"by_eye_pattern_top{args.data_suffix}"
    MODEL_PATH = GNN_PROJECT_PATH / "models" / f"top_strategy{args.data_suffix}"
    OUTPUT_PATH = GNN_PROJECT_PATH / "results" / f"top_strategy{args.data_suffix}"
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Computing Importance Maps (TOP Strategy - 4 Metrics)")
    print("="*70)
    print(f"\nData path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"\nMetrics:")
    print(f"  1. pred_std ({WEIGHT_PRED_STD*100:.0f}%): Monte Carlo Dropout uncertainty")
    print(f"  2. prediction_error ({WEIGHT_PREDICTION_ERROR*100:.0f}%): |predicted - ground_truth|")
    print(f"  3. gnn_correction ({WEIGHT_GNN_CORRECTION*100:.0f}%): |predicted - gap_sensitivity|")
    print(f"  4. leave_one_out ({WEIGHT_LEAVE_ONE_OUT*100:.0f}%): Point exclusion influence")
    print(f"\nDevice: {DEVICE}")
    print(f"MC Samples: {args.mc_samples}")
    
    # angular_params.txt があれば読み込んで表示
    params_file = DATA_PATH / "angular_params.txt"
    if params_file.exists():
        print(f"\n{'─'*50}")
        print("Angular Parameters:")
        print(f"{'─'*50}")
        with open(params_file, 'r', encoding='utf-8') as f:
            print(f.read())
    
    # モデルファイル検索
    model_files = sorted(MODEL_PATH.glob("best_model_*.pt"))
    
    if len(model_files) == 0:
        print(f"\nNo models found in {MODEL_PATH}")
        print("Please run training first.")
        sys.exit(1)
    
    print(f"\nFound {len(model_files)} model files")
    
    all_results = []
    
    for model_file in model_files:
        pattern_name = model_file.stem.replace("best_model_", "")
        
        graph_file = DATA_PATH / f"graph_data_{pattern_name}.pkl"
        if not graph_file.exists():
            print(f"\n  ⚠ Graph file not found: {graph_file}")
            continue
        
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        
        if isinstance(graph_data, dict):
            graph_list = graph_data['graph_list']
        else:
            graph_list = graph_data
        
        if len(graph_list) == 0:
            print(f"\n  ⚠ No graphs for {pattern_name}")
            continue
        
        checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
        
        in_channels = graph_list[0].x.shape[1]
        edge_dim = graph_list[0].edge_attr.shape[1] if graph_list[0].edge_attr is not None else 4
        
        model = TOPStrategyGNN(
            in_channels=in_channels,
            hidden_channels=64,
            num_layers=3,
            dropout=0.3,
            edge_dim=edge_dim
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        df = process_pattern(pattern_name, model, graph_list, DEVICE, OUTPUT_PATH, args.mc_samples)
        all_results.append((pattern_name, df))
    
    # 全体サマリー保存
    summary_file = OUTPUT_PATH / "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Importance Analysis Summary (TOP Strategy - 4 Metrics)\n")
        f.write("="*70 + "\n\n")
        
        f.write("【重要度指標】\n")
        f.write("-"*50 + "\n")
        f.write(f"1. pred_std ({WEIGHT_PRED_STD*100:.0f}%): Monte Carlo Dropout不確実性\n")
        f.write(f"2. prediction_error ({WEIGHT_PREDICTION_ERROR*100:.0f}%): |predicted - HFA|\n")
        f.write(f"3. gnn_correction ({WEIGHT_GNN_CORRECTION*100:.0f}%): |predicted - GAP|\n")
        f.write(f"4. leave_one_out ({WEIGHT_LEAVE_ONE_OUT*100:.0f}%): 除外影響度\n\n")
        
        f.write("【統合スコア】\n")
        f.write(f"combined = {WEIGHT_PRED_STD}×pred_std + {WEIGHT_PREDICTION_ERROR}×error + {WEIGHT_GNN_CORRECTION}×correction + {WEIGHT_LEAVE_ONE_OUT}×LOO\n\n")
        
        for pattern_name, df in all_results:
            f.write(f"\n{'─'*50}\n")
            f.write(f"Pattern: {pattern_name}\n")
            f.write(f"Points: {len(df)}\n")
            f.write(f"{'─'*50}\n\n")
            
            f.write("Top 10 Important Positions:\n")
            top10 = df.head(10)
            for _, row in top10.iterrows():
                f.write(f"  ({int(row['X']):3d}, {int(row['Y']):3d}): combined={row['combined_score']:.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n✓ Saved: {summary_file}")
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nOutput: {OUTPUT_PATH}")