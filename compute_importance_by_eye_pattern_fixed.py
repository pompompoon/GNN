# -*- coding: utf-8 -*-
"""
compute_importance_by_eye_pattern_fixed.py
標準グリッド座標を使用した重要度マップ計算

修正内容:
- Humphrey視野検査の標準グリッドのみを使用
- 10-2: 68点
- 24-2: 54点（マリオット除外後52点）
- 30-2: 76点（マリオット除外後74点）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR  # ★スクリプトのあるディレクトリ

# 入出力パス
MODEL_PATH = GNN_PROJECT_PATH / "models" / "by_eye_pattern_final"
DATA_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern_correct"
OUTPUT_PATH = GNN_PROJECT_PATH / "results" / "importance_maps_standard"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("Computing Importance Maps with Standard Grid Points")
print("=" * 70)
print(f"Device: {DEVICE}")


# ==================== モデル定義 ====================

class SimpleGATModel(nn.Module):
    """
    シンプルなGATモデル（感度予測用）
    train_by_eye_pattern_fixed.pyと同じ定義
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.3, edge_dim=2):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 入力層
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        
        # GAT層
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        
        # 出力層
        self.output_linear = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 入力変換
        x = self.input_linear(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT層
        for i in range(self.num_layers):
            x_res = x
            x = self.gat_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res
        
        # 出力
        pred = self.output_linear(x).squeeze(-1)
        pred = torch.clamp(pred, 0.0, 40.0)
        
        return pred


# ==================== 標準グリッド定義 ====================

def get_30_2_grid():
    """30-2パターン（76点）"""
    points = []
    
    # 奇数座標系
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
    
    # 偶数座標系（周辺部）
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
    
    # 上下端
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
    if eye_side == 0:  # 左眼
        return np.array([[15, 3], [15, -3]], dtype=np.float32)
    else:  # 右眼
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


# ==================== 重要度計算 ====================

@torch.no_grad()
def predict_with_model(model, graph_data, device=DEVICE):
    """グラフデータから予測（SimpleGATModel用）"""
    model.eval()
    data = graph_data.to(device)
    
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    
    # SimpleGATModelは感度のみを返す
    pred = model(data.x, data.edge_index, edge_attr)
    
    # 不確実性は予測値の分散で近似（複数予測の場合）または固定値
    # ここでは簡易的に予測値から推定
    pred_np = pred.cpu().numpy()
    
    return {
        'sensitivity': pred_np,
        'uncertainty': np.abs(pred_np - pred_np.mean()) / 10.0 + 0.1  # 簡易的な不確実性
    }


def compute_importance_for_pattern(pattern_name, model, graph_list, standard_grid, device=DEVICE):
    """
    標準グリッド点での重要度を計算
    """
    print(f"\n{'='*70}")
    print(f"Computing Importance: {pattern_name}")
    print(f"{'='*70}")
    print(f"Standard grid points: {len(standard_grid)}")
    
    # 重要度スコアを初期化
    importance_scores = {
        'uncertainty': np.zeros(len(standard_grid)),
        'error': np.zeros(len(standard_grid)),
        'count': np.zeros(len(standard_grid))
    }
    
    # 各グラフで標準グリッド点の重要度を集計
    n_samples = min(100, len(graph_list))
    
    for graph in tqdm(graph_list[:n_samples], desc="Processing graphs"):
        if not hasattr(graph, 'y') or graph.y is None:
            continue
        
        # 予測
        pred = predict_with_model(model, graph, device)
        
        # グラフの座標（整数に丸める）
        graph_positions = np.round(graph.pos.numpy(), 0)
        graph_targets = graph.y.numpy()
        
        # 標準グリッド点とマッチング
        for i, std_pos in enumerate(standard_grid):
            # 座標が一致する点を探す（許容誤差1度）
            matches = np.all(np.abs(graph_positions - std_pos) <= 1.0, axis=1)
            
            if matches.sum() > 0:
                idx = np.where(matches)[0][0]
                
                # 不確実性
                importance_scores['uncertainty'][i] += pred['uncertainty'][idx]
                
                # 誤差
                error = abs(pred['sensitivity'][idx] - graph_targets[idx])
                importance_scores['error'][i] += error
                
                importance_scores['count'][i] += 1
    
    # 平均を計算
    for key in ['uncertainty', 'error']:
        mask = importance_scores['count'] > 0
        if mask.sum() > 0:
            importance_scores[key][mask] /= importance_scores['count'][mask]
    
    # 正規化
    def normalize_scores(scores):
        if scores.max() - scores.min() > 1e-6:
            return (scores - scores.min()) / (scores.max() - scores.min()) * 10
        return np.zeros_like(scores)
    
    uncertainty_norm = normalize_scores(importance_scores['uncertainty'])
    error_norm = normalize_scores(importance_scores['error'])
    
    # 臨床的重要度（中心部ほど重要）
    clinical_scores = np.zeros(len(standard_grid))
    for i, pos in enumerate(standard_grid):
        distance = np.sqrt(pos[0]**2 + pos[1]**2)
        clinical_scores[i] = max(0, 10 - (distance / 3))
    
    # 統合スコア（重み: 臨床40%, Error 30%, Uncertainty 30%）
    combined_scores = (
        0.4 * clinical_scores +
        0.3 * error_norm +
        0.3 * uncertainty_norm
    )
    
    # データがない点は臨床的重要度のみ使用
    no_data_mask = importance_scores['count'] == 0
    combined_scores[no_data_mask] = clinical_scores[no_data_mask]
    
    # 結果を辞書に格納
    importance_dict = {
        'positions': standard_grid,
        'uncertainty': importance_scores['uncertainty'],
        'error': importance_scores['error'],
        'clinical': clinical_scores,
        'combined': combined_scores,
        'count': importance_scores['count'],
        # 可視化用の形式も追加
        'combined_dict': {
            'positions': standard_grid,
            'scores': combined_scores
        }
    }
    
    print(f"\nResults:")
    print(f"  Points with data: {(importance_scores['count'] > 0).sum()}/{len(standard_grid)}")
    print(f"  Combined score range: [{combined_scores.min():.2f}, {combined_scores.max():.2f}]")
    
    return importance_dict


def save_importance_results(pattern_name, importance_dict, output_path):
    """重要度結果を保存"""
    
    # Pickle形式
    pkl_file = output_path / f"importance_map_{pattern_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(importance_dict, f)
    print(f"✓ Saved (pickle): {pkl_file.name}")
    
    # CSV形式
    df = pd.DataFrame({
        'Position_X': importance_dict['positions'][:, 0],
        'Position_Y': importance_dict['positions'][:, 1],
        'Uncertainty': importance_dict['uncertainty'],
        'Error': importance_dict['error'],
        'Clinical': importance_dict['clinical'],
        'Combined': importance_dict['combined'],
        'SampleCount': importance_dict['count']
    })
    
    df = df.sort_values('Combined', ascending=False)
    
    csv_file = output_path / f"importance_map_{pattern_name}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved (CSV): {csv_file.name}")
    
    # 上位10点を表示
    print(f"\nTop 10 Most Important Points:")
    print(df.head(10)[['Position_X', 'Position_Y', 'Combined']].to_string(index=False))


# ==================== メイン処理 ====================

if __name__ == "__main__":
    
    # モデルファイルを検索
    model_files = sorted(MODEL_PATH.glob("best_model_*.pt"))
    
    if len(model_files) == 0:
        print(f"\nWarning: No trained models found in {MODEL_PATH}")
        print("Creating importance maps with clinical importance only...")
        
        # モデルなしで臨床的重要度のみ計算
        patterns = ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
        eyes = [('Left', 0), ('Right', 1)]
        
        for eye_name, eye_side in eyes:
            for pattern in patterns:
                pattern_name = f"{eye_name}_{pattern}"
                
                standard_grid = get_standard_grid(pattern, eye_side=eye_side, exclude_mariotte=True)
                expected = get_expected_point_count(pattern, exclude_mariotte=True)
                
                print(f"\n{pattern_name}: {len(standard_grid)} points (expected: {expected})")
                
                # 臨床的重要度のみ計算
                clinical_scores = np.zeros(len(standard_grid))
                for i, pos in enumerate(standard_grid):
                    distance = np.sqrt(pos[0]**2 + pos[1]**2)
                    clinical_scores[i] = max(0, 10 - (distance / 3))
                
                importance_dict = {
                    'positions': standard_grid,
                    'uncertainty': np.zeros(len(standard_grid)),
                    'error': np.zeros(len(standard_grid)),
                    'clinical': clinical_scores,
                    'combined': clinical_scores,
                    'count': np.zeros(len(standard_grid)),
                    'combined_dict': {
                        'positions': standard_grid,
                        'scores': clinical_scores
                    }
                }
                
                save_importance_results(pattern_name, importance_dict, OUTPUT_PATH)
        
        print(f"\n{'='*70}")
        print("Importance Maps Created (Clinical Only)")
        print(f"{'='*70}")
        exit(0)
    
    print(f"\nFound {len(model_files)} trained models")
    
    for model_file in model_files:
        pattern_name = model_file.stem.replace('best_model_', '')
        
        print(f"\n{'='*70}")
        print(f"Processing: {pattern_name}")
        print(f"{'='*70}")
        
        try:
            # 眼のサイドを取得
            eye_side = get_eye_side_from_name(pattern_name)
            
            # 標準グリッド点を取得
            standard_grid = get_standard_grid(pattern_name, eye_side=eye_side, exclude_mariotte=True)
            expected = get_expected_point_count(pattern_name, exclude_mariotte=True)
            
            print(f"Standard grid: {len(standard_grid)} points (expected: {expected})")
            
            if len(standard_grid) != expected:
                print(f"Warning: Point count mismatch!")
            
            # データ読み込み
            data_file = DATA_PATH / f"graph_data_{pattern_name}.pkl"
            
            if not data_file.exists():
                print(f"Warning: Data file not found: {data_file}")
                continue
            
            with open(data_file, 'rb') as f:
                pattern_data = pickle.load(f)
            
            graph_list = pattern_data['graph_list']
            n_features = pattern_data['n_features']
            edge_dim = pattern_data.get('edge_attr_dim', 2)
            
            print(f"Loaded {len(graph_list)} graphs")
            
            # モデル読み込み（SimpleGATModelを使用）
            checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
            config = checkpoint.get('config', {})
            
            model = SimpleGATModel(
                in_channels=n_features,
                hidden_channels=config.get('hidden_channels', 64),
                num_layers=config.get('num_layers', 3),
                dropout=config.get('dropout', 0.3),
                edge_dim=config.get('edge_dim', 2)
            ).to(DEVICE)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 重要度計算
            importance_dict = compute_importance_for_pattern(
                pattern_name, model, graph_list, standard_grid, DEVICE
            )
            
            # 保存
            save_importance_results(pattern_name, importance_dict, OUTPUT_PATH)
            
        except Exception as e:
            print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("All Importance Maps Completed!")
    print(f"{'='*70}")
    print(f"Results saved in: {OUTPUT_PATH}")