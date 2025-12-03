# -*- coding: utf-8 -*-
"""
train_by_eye_pattern_fixed.py
左右眼別・パターン別にGNNモデルを訓練（修正版）

修正内容:
- 損失関数をシンプルなMSE+MAE損失に変更
- モデル出力を0-40dBの範囲にクリッピング
- 学習率とバッチサイズの調整
- 詳細なログ出力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
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
GNN_PROJECT_PATH = SCRIPT_DIR  # ★修正: .parentを削除

DATA_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern_correct"
MODEL_PATH = GNN_PROJECT_PATH / "models" / "by_eye_pattern_final"
RESULT_PATH = GNN_PROJECT_PATH / "results" / "by_eye_pattern"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
RESULT_PATH.mkdir(parents=True, exist_ok=True)

# 訓練設定
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1  # グラフ単位なので1
NUM_EPOCHS = 150
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30
WEIGHT_DECAY = 1e-4

# ★ 視野感度の範囲
SENSITIVITY_MIN = 0.0
SENSITIVITY_MAX = 40.0

print(f"Device: {DEVICE}")
print(f"Sensitivity range: [{SENSITIVITY_MIN}, {SENSITIVITY_MAX}] dB")


class SimpleGATModel(nn.Module):
    """
    シンプルなGATモデル（感度予測用）
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
            # edge_dimは最初の層のみ使用（シンプル化）
            self.gat_layers.append(
                GATConv(hidden_channels, hidden_channels, heads=4, concat=False, dropout=dropout)
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        
        # 出力層（ノードレベルの感度予測）
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
            x_res = x  # 残差接続用
            x = self.gat_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # 残差接続
        
        # 出力（ノードごとの感度予測）
        pred = self.output_linear(x).squeeze(-1)
        
        # ★ 出力を0-40dBの範囲にクリッピング
        pred = torch.clamp(pred, SENSITIVITY_MIN, SENSITIVITY_MAX)
        
        return pred


def train_epoch(model, loader, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # 予測
        pred = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
        target = data.y
        
        # ★ シンプルな損失関数（MSE + MAE）
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        loss = mse_loss + mae_loss  # 両方を使用
        
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * len(target)
        total_mae += mae_loss.item() * len(target)
        total_samples += len(target)
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_loss, avg_mae


@torch.no_grad()
def evaluate(model, loader, device):
    """評価"""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        
        pred = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
        target = data.y
        
        mse_loss = F.mse_loss(pred, target)
        mae_loss = F.l1_loss(pred, target)
        loss = mse_loss + mae_loss
        
        total_loss += loss.item() * len(target)
        total_mae += mae_loss.item() * len(target)
        total_samples += len(target)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return avg_loss, avg_mae, rmse


def train_eye_pattern_model(eye_pattern_name, graph_list, device=DEVICE):
    """眼・パターン専用モデルの訓練"""
    print(f"\n{'='*60}")
    print(f"Training Model: {eye_pattern_name}")
    print(f"{'='*60}")
    
    # ★ データの検証
    print("\nData validation:")
    all_sensitivities = []
    all_features = []
    for g in graph_list:
        all_sensitivities.extend(g.y.numpy().tolist())
        all_features.append(g.x.numpy())
    
    sens_arr = np.array(all_sensitivities)
    print(f"  Sensitivity range: [{sens_arr.min():.2f}, {sens_arr.max():.2f}] dB")
    print(f"  Sensitivity mean: {sens_arr.mean():.2f} dB")
    print(f"  Sensitivity std: {sens_arr.std():.2f} dB")
    
    # 特徴量の確認
    all_features = np.concatenate(all_features, axis=0)
    print(f"  Feature ranges:")
    for i in range(all_features.shape[1]):
        print(f"    Feature {i}: [{all_features[:, i].min():.3f}, {all_features[:, i].max():.3f}]")
    
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
    
    print(f"\nData split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    
    # モデル初期化
    n_features = graph_list[0].x.shape[1]
    
    model = SimpleGATModel(
        in_channels=n_features,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        edge_dim=2
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input features: {n_features}")
    print(f"  Hidden channels: 64")
    print(f"  Num GAT layers: 3")
    print(f"  Output: Sensitivity [0, 40] dB")
    
    # パラメータ数
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # 訓練ループ
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\nTraining started...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train MAE':>10} {'Val Loss':>12} {'Val MAE':>10} {'LR':>10}")
    print("-" * 70)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mae = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # ログ出力
        if epoch % 10 == 0 or epoch == 1 or epoch <= 5:
            print(f"{epoch:>6} {train_loss:>12.4f} {train_mae:>10.2f} {val_loss:>12.4f} {val_mae:>10.2f} {current_lr:>10.6f}")
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            
            # ベストモデル保存
            save_path = MODEL_PATH / f"best_model_{eye_pattern_name}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'n_features': n_features,
                'config': {
                    'hidden_channels': 64,
                    'num_layers': 3,
                    'dropout': 0.3,
                    'edge_dim': 2
                }
            }, save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # テストデータで評価
    checkpoint_path = MODEL_PATH / f"best_model_{eye_pattern_name}.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Training Completed: {eye_pattern_name}")
    print(f"{'='*60}")
    print(f"✓ Best Val MAE: {best_val_mae:.2f} dB")
    print(f"✓ Test MAE: {test_mae:.2f} dB")
    print(f"✓ Test RMSE: {test_rmse:.2f} dB")
    
    # ★ 予測範囲の確認
    model.eval()
    with torch.no_grad():
        sample_preds = []
        for data in test_loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr if hasattr(data, 'edge_attr') else None)
            sample_preds.extend(pred.cpu().numpy())
        
        sample_preds = np.array(sample_preds)
        print(f"✓ Prediction range: [{sample_preds.min():.2f}, {sample_preds.max():.2f}] dB")
    
    return {
        'eye_pattern_name': eye_pattern_name,
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
    print("="*70)
    print("GNN Training by Eye and Pattern (FIXED VERSION)")
    print("★ Simple MSE+MAE loss")
    print("★ Output clamped to [0, 40] dB")
    print("="*70)
    
    # データ読み込み
    data_files = sorted(DATA_PATH.glob("graph_data_*.pkl"))
    
    if len(data_files) == 0:
        print(f"\nError: No graph data found in {DATA_PATH}")
        print("Please run create_graph_by_eye_pattern_fixed.py first")
        exit(1)
    
    print(f"\nFound {len(data_files)} eye-pattern data files")
    
    results = []
    
    for data_file in data_files:
        eye_pattern_name = data_file.stem.replace('graph_data_', '')
        
        # データ読み込み
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        
        if len(graph_list) < 10:
            print(f"\n⚠ Skipping {eye_pattern_name}: insufficient data ({len(graph_list)} graphs)")
            continue
        
        print(f"\n{eye_pattern_name}: {len(graph_list)} graphs")
        print(f"  Features: {pattern_data['n_features']}")
        print(f"  Feature names: {pattern_data.get('feature_names', 'N/A')}")
        
        # 訓練実行
        result = train_eye_pattern_model(eye_pattern_name, graph_list, DEVICE)
        results.append(result)
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    
    for result in results:
        print(f"\n{result['eye_pattern_name']}:")
        print(f"  Best Val MAE: {result['best_val_mae']:.2f} dB")
        print(f"  Test MAE: {result['test_mae']:.2f} dB")
        print(f"  Test RMSE: {result['test_rmse']:.2f} dB")
    
    # CSV保存
    df_results = pd.DataFrame(results)
    csv_file = RESULT_PATH / "training_results_by_eye_pattern.csv"
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"Models saved in: {MODEL_PATH}")
    print(f"Results saved: {csv_file}")
    
    # ★ MAEの妥当性チェック
    print(f"\n★ MAE Validation:")
    all_maes = [r['test_mae'] for r in results]
    if all(mae <= 10 for mae in all_maes):
        print("  ✓ All MAEs are within reasonable range (≤10 dB)")
    else:
        print("  ⚠ Some MAEs are high. Consider:")
        print("    - Check data preprocessing")
        print("    - Increase training epochs")
        print("    - Adjust model architecture")