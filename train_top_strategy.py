# -*- coding: utf-8 -*-
"""
train_top_strategy.py
TOP Strategy GNNモデルの訓練（角度パラメータ対応版）

使用例:
  python train_top_strategy.py
  python train_top_strategy.py --data-suffix _angle45
  python train_top_strategy.py --data-suffix _angle30

目的:
- GAP感度（1回測定）から最終閾値（HFA感度）を推定
- 隣接点の測定結果をメッセージパッシングで集約

モデル設計:
- 入力: [GAP感度, 偏心度, 角度]
- エッジ: [相関係数, 角度類似度, 距離重み, 感度比率]
- 出力: HFA感度
"""

# OpenMP競合を回避
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 設定ファイル読み込み
try:
    from config_top_strategy import (
        BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, 
        EARLY_STOPPING_PATIENCE, WEIGHT_DECAY,
        SENSITIVITY_MIN, SENSITIVITY_MAX
    )
except ImportError:
    BATCH_SIZE = 1
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 30
    WEIGHT_DECAY = 1e-4
    SENSITIVITY_MIN = 0.0
    SENSITIVITY_MAX = 40.0

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")


class TOPStrategyGNN(nn.Module):
    """
    TOP Strategy用GNNモデル
    
    GAP感度 + 隣接点情報 → HFA感度（最終閾値）
    """
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.3, edge_dim=4):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 入力層
        self.input_lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN層（edge_attrを使用）
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=4, concat=False, 
                        dropout=dropout, edge_dim=edge_dim)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        
        # 出力層（感度予測）
        self.output_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
        # GAP感度からの直接パス（残差接続的）
        self.gap_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x, edge_index, edge_attr=None):
        # GAP感度を保存（最初の特徴量）
        gap_sensitivity = x[:, 0:1] * SENSITIVITY_MAX  # 元のスケールに戻す
        
        # 入力変換
        h = self.input_lin(x)
        
        # GNN層（edge_attrを使用）
        for i in range(self.num_layers):
            h_res = h
            h = self.gnn_layers[i](h, edge_index, edge_attr=edge_attr)
            h = self.bn_layers[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res  # 残差接続
        
        # 出力
        pred = self.output_lin(h)
        
        # GAP感度との加重平均（GNNが補正を学習）
        gap_w = torch.sigmoid(self.gap_weight)
        final_pred = gap_w * gap_sensitivity + (1 - gap_w) * pred * SENSITIVITY_MAX
        
        # クリッピング
        final_pred = torch.clamp(final_pred.squeeze(-1), SENSITIVITY_MIN, SENSITIVITY_MAX)
        
        return final_pred


def train_epoch(model, loader, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        pred = model(data.x, data.edge_index, data.edge_attr)
        target = data.y
        
        # 損失
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        loss = mse_loss + mae_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(target)
        total_mae += mae_loss.item() * len(target)
        total_samples += len(target)
    
    return total_loss / total_samples, total_mae / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """評価"""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_gaps = []
    
    for data in loader:
        data = data.to(device)
        
        pred = model(data.x, data.edge_index, data.edge_attr)
        target = data.y
        
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        loss = mse_loss + mae_loss
        
        total_loss += loss.item() * len(target)
        total_mae += mae_loss.item() * len(target)
        total_samples += len(target)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        # GAP感度も収集
        if hasattr(data, 'gap_sensitivity'):
            all_gaps.extend(data.gap_sensitivity.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    # GAP単独のMAE（ベースライン）
    gap_mae = None
    if len(all_gaps) > 0:
        all_gaps = np.array(all_gaps)
        gap_mae = np.mean(np.abs(all_gaps - all_targets))
    
    return avg_loss, avg_mae, rmse, gap_mae


def train_model(eye_pattern_name, graph_list, device, model_path):
    """モデル訓練"""
    print(f"\n{'='*60}")
    print(f"Training: {eye_pattern_name}")
    print(f"{'='*60}")
    
    # グラフのバリデーション（不正なグラフを除外）
    valid_graphs = []
    for g in graph_list:
        if g.edge_index.numel() > 0 and g.edge_index.max() < g.num_nodes:
            valid_graphs.append(g)
    
    if len(valid_graphs) < len(graph_list):
        print(f"  Warning: Filtered out {len(graph_list) - len(valid_graphs)} invalid graphs")
    
    graph_list = valid_graphs
    
    if len(graph_list) < 10:
        print(f"  Error: Not enough valid graphs ({len(graph_list)})")
        return None
    
    # データ検証
    print("\nData validation:")
    all_gap = []
    all_hfa = []
    for g in graph_list:
        all_gap.extend(g.gap_sensitivity.numpy().tolist())
        all_hfa.extend(g.y.numpy().tolist())
    
    all_gap = np.array(all_gap)
    all_hfa = np.array(all_hfa)
    
    # ベースライン（GAP単独）
    gap_only_mae = np.mean(np.abs(all_gap - all_hfa))
    gap_hfa_corr = np.corrcoef(all_gap, all_hfa)[0, 1]
    
    print(f"  GAP range: [{all_gap.min():.1f}, {all_gap.max():.1f}] dB")
    print(f"  HFA range: [{all_hfa.min():.1f}, {all_hfa.max():.1f}] dB")
    print(f"  GAP-HFA correlation: {gap_hfa_corr:.3f}")
    print(f"  ★ Baseline (GAP only) MAE: {gap_only_mae:.2f} dB")
    
    # データ分割
    n_graphs = len(graph_list)
    n_train = int(0.7 * n_graphs)
    n_val = int(0.15 * n_graphs)
    
    np.random.seed(42)
    indices = np.random.permutation(n_graphs)
    
    train_graphs = [graph_list[i] for i in indices[:n_train]]
    val_graphs = [graph_list[i] for i in indices[n_train:n_train+n_val]]
    test_graphs = [graph_list[i] for i in indices[n_train+n_val:]]
    
    print(f"\nSplit: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデル
    n_features = graph_list[0].x.shape[1]
    edge_dim = graph_list[0].edge_attr.shape[1]
    
    model = TOPStrategyGNN(
        in_channels=n_features,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        edge_dim=edge_dim
    ).to(device)
    
    print(f"\nModel: TOPStrategyGNN")
    print(f"  Node features: {n_features}")
    print(f"  Edge features: {edge_dim}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 訓練
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\n{'Epoch':>6} {'Train MAE':>10} {'Val MAE':>10} {'Gap MAE':>10} {'LR':>10}")
    print("-" * 55)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, val_rmse, val_gap_mae = evaluate(model, val_loader, device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0 or epoch <= 5:
            gap_str = f"{val_gap_mae:.2f}" if val_gap_mae else "N/A"
            print(f"{epoch:>6} {train_mae:>10.2f} {val_mae:>10.2f} {gap_str:>10} {current_lr:>10.6f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'n_features': n_features,
                'edge_dim': edge_dim,
                'config': {
                    'hidden_channels': 64,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }, model_path / f"best_model_{eye_pattern_name}.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # テスト評価
    checkpoint = torch.load(model_path / f"best_model_{eye_pattern_name}.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse, test_gap_mae = evaluate(model, test_loader, device)
    
    # 改善率
    improvement = (test_gap_mae - test_mae) / test_gap_mae * 100 if test_gap_mae else 0
    
    print(f"\n{'='*60}")
    print(f"Results: {eye_pattern_name}")
    print(f"{'='*60}")
    print(f"  ★ Baseline (GAP only) MAE: {test_gap_mae:.2f} dB")
    print(f"  ★ GNN Model MAE: {test_mae:.2f} dB")
    print(f"  ★ Improvement: {improvement:.1f}%")
    print(f"  Test RMSE: {test_rmse:.2f} dB")
    
    return {
        'eye_pattern_name': eye_pattern_name,
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'baseline_gap_mae': test_gap_mae,
        'improvement_pct': improvement,
        'n_train': len(train_graphs),
        'n_val': len(val_graphs),
        'n_test': len(test_graphs),
        'gap_hfa_corr': gap_hfa_corr,
        'n_features': n_features,
        'edge_dim': edge_dim
    }


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNNモデルの訓練',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python train_top_strategy.py
  python train_top_strategy.py --data-suffix _angle45
  python train_top_strategy.py --data-suffix _angle30
        """
    )
    
    parser.add_argument('--data-suffix', '-d', type=str, default='',
                        help='データディレクトリのサフィックス (例: _angle45)')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("="*70)
    print("Training TOP Strategy GNN")
    print("="*70)
    
    args = parse_args()
    
    # パス設定
    DATA_PATH = GNN_PROJECT_PATH / "data" / f"by_eye_pattern_top{args.data_suffix}"
    MODEL_PATH = GNN_PROJECT_PATH / "models" / f"top_strategy{args.data_suffix}"
    RESULT_PATH = GNN_PROJECT_PATH / "results" / f"top_strategy{args.data_suffix}"
    
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData path: {DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Result path: {RESULT_PATH}")
    
    # angular_params.txt があれば読み込んで表示
    params_file = DATA_PATH / "angular_params.txt"
    if params_file.exists():
        print(f"\n{'─'*50}")
        print("Angular Parameters:")
        print(f"{'─'*50}")
        with open(params_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print(f"{'─'*50}")
    
    print("\n★ Goal: Improve threshold estimation using neighboring points")
    print("  Input: GAP sensitivity + spatial context")
    print("  Output: HFA sensitivity (final threshold)")
    print("  Baseline: GAP sensitivity alone")
    
    # データ読み込み
    data_files = sorted(DATA_PATH.glob("graph_data_*.pkl"))
    
    if len(data_files) == 0:
        print(f"\nError: No data found in {DATA_PATH}")
        print("Please run create_graph_top_strategy.py first")
        exit(1)
    
    print(f"\nFound {len(data_files)} eye-pattern data files")
    
    results = []
    
    for data_file in data_files:
        eye_pattern_name = data_file.stem.replace('graph_data_', '')
        
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        
        # エッジ特徴量の情報を表示
        if len(graph_list) > 0:
            edge_dim = graph_list[0].edge_attr.shape[1]
            edge_names = pattern_data.get('edge_feature_names', [])
            print(f"\n{eye_pattern_name}: Edge features ({edge_dim}D): {edge_names}")
        
        if len(graph_list) < 10:
            print(f"\nSkipping {eye_pattern_name}: insufficient data ({len(graph_list)} graphs)")
            continue
        
        result = train_model(eye_pattern_name, graph_list, DEVICE, MODEL_PATH)
        if result is not None:
            results.append(result)
    
    # サマリー
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    
    if len(results) == 0:
        print("\nNo models were trained successfully.")
    else:
        for r in results:
            print(f"\n{r['eye_pattern_name']}:")
            print(f"  Baseline (GAP) MAE: {r['baseline_gap_mae']:.2f} dB")
            print(f"  GNN Model MAE: {r['test_mae']:.2f} dB")
            print(f"  Improvement: {r['improvement_pct']:.1f}%")
            print(f"  Edge features: {r['edge_dim']}D")
    
    # CSV保存
    df_results = pd.DataFrame(results)
    csv_file = RESULT_PATH / "training_results_top_strategy.csv"
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print("Completed!")
    print(f"{'='*70}")
    print(f"Models: {MODEL_PATH}")
    print(f"Results: {csv_file}")