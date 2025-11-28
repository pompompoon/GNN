# -*- coding: utf-8 -*-
"""
train_no_batch.py
バッチ処理なしの訓練（グラフごとに個別処理）
"""

import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np

# パス設定
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "by_eye_pattern"
MODEL_PATH = SCRIPT_DIR / "models" / "by_eye_pattern"

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# models_revised.pyをインポート
sys.path.insert(0, str(SCRIPT_DIR))
from models_revised import create_model

# ハイパーパラメータ
DEVICE = 'cpu'
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 7

print("="*70)
print("Training Without Batching (Graph-by-Graph)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")

# テスト用: Left_Pattern30-2のみ
test_pattern = "Left_Pattern30-2"
data_file = DATA_PATH / f"graph_data_{test_pattern}.pkl"

if not data_file.exists():
    print(f"\n❌ Data file not found: {data_file}")
    sys.exit(1)

print(f"\n{'='*70}")
print(f"Training: {test_pattern}")
print(f"{'='*70}")

# データ読み込み
with open(data_file, 'rb') as f:
    pattern_data = pickle.load(f)

graph_list = pattern_data['graph_list']
n_features = pattern_data['n_features']
edge_dim = pattern_data.get('edge_attr_dim', 2)

print(f"Graphs: {len(graph_list)}")
print(f"Features: {n_features}")
print(f"Edge attributes: {edge_dim}")

# データ分割
n_graphs = len(graph_list)
train_size = int(0.7 * n_graphs)
val_size = int(0.15 * n_graphs)

train_graphs = graph_list[:train_size]
val_graphs = graph_list[train_size:train_size+val_size]
test_graphs = graph_list[train_size+val_size:]

print(f"\nTrain: {len(train_graphs)}")
print(f"Val: {len(val_graphs)}")
print(f"Test: {len(test_graphs)}")

# モデル作成
model = create_model(
    model_name='gat',
    in_channels=n_features,
    hidden_channels=64,
    out_channels=2,
    num_layers=3,
    dropout=0.3,
    edge_dim=edge_dim
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")


def train_epoch_no_batch(model, graphs, optimizer, device):
    """グラフごとに個別処理する訓練"""
    model.train()
    total_loss = 0
    
    for graph in graphs:
        try:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
            
            # batch=Noneで単一グラフとして処理
            pred_mean, pred_std, vis_logits = model(
                graph.x, graph.edge_index, edge_attr, batch=None
            )
            
            loss = torch.nn.functional.mse_loss(pred_mean, graph.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        except Exception as e:
            print(f"\n  Error in graph: {e}")
            continue
    
    return total_loss / len(graphs)


def evaluate_no_batch(model, graphs, device, debug=False):
    """グラフごとに個別評価"""
    model.eval()
    total_loss = 0
    total_mae = 0
    n_valid = 0
    
    with torch.no_grad():
        for i, graph in enumerate(graphs):
            # デバッグ情報を先に出力
            if debug and i < 3:
                print(f"\n  Debug Graph {i} (BEFORE .to(device)):")
                print(f"    x.shape: {graph.x.shape}")
                print(f"    edge_index.shape: {graph.edge_index.shape}")
                print(f"    edge_index.max(): {graph.edge_index.max().item()}")
                print(f"    y.shape: {graph.y.shape}")
            
            try:
                graph = graph.to(device)
                
                if debug and i < 3:
                    print(f"  Debug Graph {i} (AFTER .to(device)):")
                    print(f"    x.shape: {graph.x.shape}")
                    print(f"    edge_index.shape: {graph.edge_index.shape}")
                    print(f"    edge_index.max(): {graph.edge_index.max().item()}")
                
                edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
                
                if debug and i < 3:
                    print(f"    Calling model.forward...")
                
                pred_mean, pred_std, vis_logits = model(
                    graph.x, graph.edge_index, edge_attr, batch=None
                )
                
                if debug and i < 3:
                    print(f"    pred_mean.shape: {pred_mean.shape}")
                    print(f"    ✓ Success")
                
                loss = torch.nn.functional.mse_loss(pred_mean, graph.y)
                mae = torch.abs(pred_mean - graph.y).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                n_valid += 1
                
            except Exception as e:
                if debug or i < 5:
                    print(f"\n  ✗ Error in graph {i}: {e}")
                    if debug and i < 3:
                        import traceback
                        traceback.print_exc()
                continue
    
    if n_valid == 0:
        return float('inf'), float('inf')
    
    return total_loss / n_valid, total_mae / n_valid


# 訓練ループ
print(f"\n{'='*70}")
print("Training...")
print(f"{'='*70}")

best_val_mae = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch_no_batch(model, train_graphs, optimizer, DEVICE)
    val_loss, val_mae = evaluate_no_batch(model, val_graphs, DEVICE)
    
    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val MAE: {val_mae:.2f} dB")
    
    # Early stopping & モデル保存
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        
        # 保存
        save_path = MODEL_PATH / f"best_model_{test_pattern}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': val_mae,
            'config': {
                'MODEL_NAME': 'gat',
                'HIDDEN_CHANNELS': 64,
                'NUM_LAYERS': 3,
                'DROPOUT': 0.3,
                'EDGE_DIM': edge_dim,
                'IN_CHANNELS': n_features,
                'OUT_CHANNELS': 2
            }
        }, save_path)
        
        print(f"  ✓ Best model saved (MAE: {val_mae:.2f} dB)")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# テスト評価
print(f"\n{'='*70}")
print("Evaluating on test set (with debug)...")
print(f"{'='*70}")

# 訓練中に保存された最新のモデルを使用（既にロード済み）
# チェックポイントの再読み込みは不要
test_loss, test_mae = evaluate_no_batch(model, test_graphs, DEVICE, debug=True)

print(f"\n✓ Training Completed!")
print(f"  Best Val MAE: {best_val_mae:.2f} dB" if best_val_mae < float('inf') else "  Best Val MAE: N/A (no valid epochs)")
print(f"  Test MAE: {test_mae:.2f} dB" if test_mae < float('inf') else "  Test MAE: N/A")

if best_val_mae < float('inf'):
    model_file = MODEL_PATH / f"best_model_{test_pattern}.pt"
    if model_file.exists():
        print(f"  Model saved: {model_file}")
    else:
        print(f"  ⚠️  Model file not found: {model_file}")
else:
    print(f"  ⚠️  No model was saved (training failed)")

print(f"\n{'='*70}")
if best_val_mae < float('inf'):
    print("Success!")
else:
    print("Training failed - all epochs had errors")
print(f"{'='*70}")
print("\nThe model works! Now you can:")
print("  1. Train all patterns: python train_by_eye_pattern_revised.py")
print("     (Make sure to use the no-batch approach)")
print("  2. Or compute importance for this single model:")
print("     python compute_importance_simple.py")
print(f"\n{'='*70}")