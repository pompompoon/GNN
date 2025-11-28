# -*- coding: utf-8 -*-
"""
train_by_pattern_revised.py
パターンごとにGNNモデルを訓練（修正版）

修正内容:
- エッジ属性: 2次元 [距離の逆数, 感度相関]
- 教師データ: HFA_Sensitivity（ノードレベル回帰）
- 目的: 周辺感度の予測（HFA感度を教師として）
- models_revised.pyを使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
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
GNN_PROJECT_PATH = SCRIPT_DIR.parent

if not GNN_PROJECT_PATH.exists():
    GNN_PROJECT_PATH = Path.cwd()

DATA_PATH = GNN_PROJECT_PATH / "data" / "by_pattern"
MODEL_PATH = GNN_PROJECT_PATH / "models" / "by_pattern"
RESULT_PATH = GNN_PROJECT_PATH / "results" / "by_pattern"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

# ★models_revised.pyをインポート
from models_revised import create_model

# 訓練設定
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20

print(f"Device: {DEVICE}")


def train_epoch(model, loader, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # ★yがHFA_Sensitivityのノードレベル値
        pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # ガウス負対数尤度損失（回帰）
        nll_loss = torch.mean(
            0.5 * torch.log(pred_std**2 + 1e-6) + 
            0.5 * ((data.y - pred_mean)**2) / (pred_std**2 + 1e-6)
        )
        
        loss = nll_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
        all_preds.extend(pred_mean.detach().cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # MAE計算
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return avg_loss, mae


@torch.no_grad()
def evaluate(model, loader, device):
    """評価"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_stds = []
    
    for data in loader:
        data = data.to(device)
        
        pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        nll_loss = torch.mean(
            0.5 * torch.log(pred_std**2 + 1e-6) + 
            0.5 * ((data.y - pred_mean)**2) / (pred_std**2 + 1e-6)
        )
        
        total_loss += nll_loss.item() * data.num_graphs
        
        all_preds.extend(pred_mean.cpu().numpy())
        all_targets.extend(data.y.cpu().numpy())
        all_stds.extend(pred_std.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # MAE, RMSE計算
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
    
    return avg_loss, mae, rmse


def train_pattern_model(pattern_name, graph_list, device=DEVICE):
    """パターン専用モデルの訓練"""
    print(f"\n{'='*60}")
    print(f"Training Model: {pattern_name}")
    print(f"{'='*60}")
    
    # データ分割
    n_graphs = len(graph_list)
    n_train = int(0.7 * n_graphs)
    n_val = int(0.15 * n_graphs)
    
    np.random.seed(42)
    indices = np.random.permutation(n_graphs)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_graphs = [graph_list[i] for i in train_idx]
    val_graphs = [graph_list[i] for i in val_idx]
    test_graphs = [graph_list[i] for i in test_idx]
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデル初期化
    n_features = graph_list[0].x.shape[1]
    
    # ★edge_dim=2に設定（距離 + 相関）
    model = create_model(
        model_name='gat',
        in_channels=n_features,
        hidden_channels=64,
        out_channels=2,
        num_layers=3,
        dropout=0.3,
        edge_dim=2  # ★2次元エッジ属性
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input features: {n_features}")
    print(f"  Hidden channels: 64")
    print(f"  Edge dim: 2 [distance_weight, correlation]")
    print(f"  Num layers: 3")
    print(f"  Model type: GAT")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 訓練ループ
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device)
        
        # 学習率の更新
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 学習率が変化した場合にログ出力
        if new_lr != old_lr:
            print(f"  Learning rate adjusted: {old_lr:.6f} -> {new_lr:.6f}")
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: "
                  f"Train Loss={train_loss:.4f}, Train MAE={train_mae:.2f} | "
                  f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.2f}")
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            
            # ベストモデル保存
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'config': {
                    'MODEL_NAME': 'gat',
                    'HIDDEN_CHANNELS': 64,
                    'NUM_LAYERS': 3,
                    'DROPOUT': 0.3,
                    'EDGE_DIM': 2
                }
            }, MODEL_PATH / f"best_model_{pattern_name}.pt")
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # テストデータで評価
    checkpoint = torch.load(MODEL_PATH / f"best_model_{pattern_name}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device)
    
    print(f"\n✓ Best Val Loss: {best_val_loss:.4f}")
    print(f"✓ Best Val MAE: {best_val_mae:.2f} dB")
    print(f"✓ Test MAE: {test_mae:.2f} dB")
    print(f"✓ Test RMSE: {test_rmse:.2f} dB")
    
    return {
        'pattern_name': pattern_name,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'n_train': len(train_graphs),
        'n_val': len(val_graphs),
        'n_test': len(test_graphs)
    }


# メイン処理
if __name__ == "__main__":
    print("="*60)
    print("GNN Training by Pattern (Revised)")
    print("Edge attributes: [distance_weight, sensitivity_correlation]")
    print("Teacher data: HFA_Sensitivity")
    print("Goal: Predict peripheral sensitivity using HFA as ground truth")
    print("="*60)
    
    # データ読み込み
    data_files = sorted(DATA_PATH.glob("graph_data_Pattern*.pkl"))
    
    if len(data_files) == 0:
        print(f"\nError: No graph data found in {DATA_PATH}")
        print("Please run create_graph_by_pattern_revised.py first")
        exit(1)
    
    print(f"\nFound {len(data_files)} pattern data files")
    
    results = []
    
    for data_file in data_files:
        pattern_name = data_file.stem.replace('graph_data_', '')
        
        # データ読み込み
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        
        if len(graph_list) < 10:
            print(f"\n⚠ Skipping {pattern_name}: insufficient data ({len(graph_list)} graphs)")
            continue
        
        print(f"\n{pattern_name}: {len(graph_list)} graphs")
        print(f"  Features: {pattern_data['n_features']}")
        print(f"  Edge attr dim: {pattern_data.get('edge_attr_dim', 'N/A')}")
        print(f"  Adjacency threshold: {pattern_data.get('adjacency_threshold', 'N/A')}")
        print(f"  Teacher data: {pattern_data.get('teacher_data', 'N/A')}")
        
        # 訓練実行
        result = train_pattern_model(pattern_name, graph_list, DEVICE)
        results.append(result)
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['pattern_name']}:")
        print(f"  Best Val MAE: {result['best_val_mae']:.2f} dB")
        print(f"  Test MAE: {result['test_mae']:.2f} dB")
        print(f"  Test RMSE: {result['test_rmse']:.2f} dB")
        print(f"  Data: Train={result['n_train']}, Val={result['n_val']}, Test={result['n_test']}")
    
    # CSV保存
    df_results = pd.DataFrame(results)
    csv_file = RESULT_PATH / "training_results_revised.csv"
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Models saved in: {MODEL_PATH}")
    print(f"Results saved: {csv_file}")