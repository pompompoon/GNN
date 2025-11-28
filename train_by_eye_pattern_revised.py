# -*- coding: utf-8 -*-
"""
train_by_eye_pattern_revised.py
左右眼別・パターン別にGNNモデルを訓練（修正版）

修正内容:
- 左右眼別・パターン別にモデル構築
- エッジ属性: 2次元 [距離の逆数, 感度類似性]
- 教師データ: HFA_Sensitivity（ノードレベル回帰）
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

DATA_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern"
MODEL_PATH = GNN_PROJECT_PATH / "models" / "by_eye_pattern"
RESULT_PATH = GNN_PROJECT_PATH / "results" / "by_eye_pattern"

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
    """1エポックの訓練（バッチサイズ1用に最適化）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # エッジ属性の確認（安全に取得）
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # バッチサイズ1なので、基本的に問題は発生しないはず
        # しかし念のため検証
        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0:
            max_edge_idx = data.edge_index.max().item()
            
            if max_edge_idx >= num_nodes:
                print(f"  Warning: Skipping batch with invalid edges (max_idx={max_edge_idx}, nodes={num_nodes})")
                continue
        
        try:
            pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, edge_attr, batch)
            
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
            
        except Exception as e:
            print(f"  Error during forward pass: {e}")
            print(f"  Skipping this batch...")
            continue
    
    if len(all_preds) == 0:
        return float('inf'), float('inf')
    
    avg_loss = total_loss / len(all_preds)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    return avg_loss, mae


@torch.no_grad()
def evaluate(model, loader, device):
    """評価（バッチサイズ1用に最適化）"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_stds = []
    
    for data in loader:
        data = data.to(device)
        
        # エッジ属性の確認
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 検証
        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0:
            max_edge_idx = data.edge_index.max().item()
            
            if max_edge_idx >= num_nodes:
                print(f"  Warning: Skipping batch with invalid edges (max_idx={max_edge_idx}, nodes={num_nodes})")
                continue
        
        try:
            pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, edge_attr, batch)
            
            nll_loss = torch.mean(
                0.5 * torch.log(pred_std**2 + 1e-6) + 
                0.5 * ((data.y - pred_mean)**2) / (pred_std**2 + 1e-6)
            )
            
            total_loss += nll_loss.item() * data.num_graphs
            
            all_preds.extend(pred_mean.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
            all_stds.extend(pred_std.cpu().numpy())
            
        except Exception as e:
            print(f"  Error during evaluation: {e}")
            continue
    
    if len(all_preds) == 0:
        return float('inf'), float('inf'), float('inf')
    
    avg_loss = total_loss / len(all_preds)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
    
    return avg_loss, mae, rmse


def train_eye_pattern_model(eye_pattern_name, graph_list, device=DEVICE):
    """眼・パターン専用モデルの訓練"""
    print(f"\n{'='*60}")
    print(f"Training Model: {eye_pattern_name}")
    print(f"{'='*60}")
    
    # モデル保存ディレクトリの確認
    if not MODEL_PATH.exists():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {MODEL_PATH}")
    
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
    
    # ⚠️ バッチサイズを1に強制（バッチ処理の問題を回避）
    print(f"\n⚠️  Using BATCH_SIZE=1 to avoid batching issues")
    BATCH_SIZE_SAFE = 1
    
    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE_SAFE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE_SAFE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE_SAFE, shuffle=False)
    
    # モデル初期化
    n_features = graph_list[0].x.shape[1]
    
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
    print(f"  Edge dim: 2 [distance_weight, similarity]")
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
    current_lr = LEARNING_RATE
    
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
            try:
                save_path = MODEL_PATH / f"best_model_{eye_pattern_name}.pt"
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
                }, save_path)
                
                if epoch % 10 == 0:
                    print(f"  ✓ Best model saved: {save_path.name}")
                    
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to save model: {e}")
                
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # テストデータで評価
    checkpoint_path = MODEL_PATH / f"best_model_{eye_pattern_name}.pt"
    
    if not checkpoint_path.exists():
        print(f"\n⚠️  Warning: Best model checkpoint not found: {checkpoint_path}")
        print(f"Using current model for evaluation...")
        test_loss, test_mae, test_rmse = evaluate(model, test_loader, device)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_mae, test_rmse = evaluate(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Training Completed: {eye_pattern_name}")
    print(f"{'='*60}")
    print(f"✓ Best Val Loss: {best_val_loss:.4f}")
    print(f"✓ Best Val MAE: {best_val_mae:.2f} dB")
    print(f"✓ Test MAE: {test_mae:.2f} dB")
    print(f"✓ Test RMSE: {test_rmse:.2f} dB")
    
    checkpoint_path = MODEL_PATH / f"best_model_{eye_pattern_name}.pt"
    if checkpoint_path.exists():
        print(f"✓ Model saved: {checkpoint_path}")
    else:
        print(f"⚠️  Model not saved (check permissions or disk space)")
    
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
    print("="*60)
    print("GNN Training by Eye and Pattern")
    print("Edge attributes: [distance_weight, sensitivity_similarity]")
    print("Teacher data: HFA_Sensitivity")
    print("="*60)
    
    # データ読み込み
    data_files = sorted(DATA_PATH.glob("graph_data_*.pkl"))
    
    if len(data_files) == 0:
        print(f"\nError: No graph data found in {DATA_PATH}")
        print("Please run create_graph_by_eye_pattern_revised.py first")
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
        print(f"  Edge attr dim: {pattern_data.get('edge_attr_dim', 'N/A')}")
        print(f"  Eye side: {pattern_data.get('eye_side', 'N/A')}")
        print(f"  Pattern ID: {pattern_data.get('pattern_id', 'N/A')}")
        
        # 訓練実行
        result = train_eye_pattern_model(eye_pattern_name, graph_list, DEVICE)
        results.append(result)
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\n{result['eye_pattern_name']}:")
        print(f"  Best Val MAE: {result['best_val_mae']:.2f} dB")
        print(f"  Test MAE: {result['test_mae']:.2f} dB")
        print(f"  Test RMSE: {result['test_rmse']:.2f} dB")
        print(f"  Data: Train={result['n_train']}, Val={result['n_val']}, Test={result['n_test']}")
    
    # CSV保存
    df_results = pd.DataFrame(results)
    csv_file = RESULT_PATH / "training_results_by_eye_pattern.csv"
    df_results.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Models saved in: {MODEL_PATH}")
    print(f"Results saved: {csv_file}")