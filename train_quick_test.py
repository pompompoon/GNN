# -*- coding: utf-8 -*-
"""
train_quick_test.py
高速テスト訓練（1パターンのみ）
"""

import torch
import pickle
from pathlib import Path
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import sys

# パス設定
SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "by_eye_pattern"
MODEL_PATH = SCRIPT_DIR / "models" / "by_eye_pattern"

MODEL_PATH.mkdir(parents=True, exist_ok=True)

# models_revised.pyをインポート
sys.path.insert(0, str(SCRIPT_DIR))
from models_revised import create_model

# ハイパーパラメータ（高速化）
DEVICE = 'cpu'
BATCH_SIZE = 32  # 1 → 32 に変更（DataLoaderのバッチ処理を活用）
NUM_EPOCHS = 20  # 短縮
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5  # 短縮

print("="*70)
print("Quick Test Training (1 Pattern Only)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Early stopping: {EARLY_STOPPING_PATIENCE}")

# テスト用: Left_Pattern30-2のみ訓練（最も小さいデータセット）
test_pattern = "Left_Pattern30-2"
data_file = DATA_PATH / f"graph_data_{test_pattern}.pkl"

if not data_file.exists():
    print(f"\n❌ Data file not found: {data_file}")
    print("\nPlease run: python create_graph_by_eye_pattern_revised.py")
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

# DataLoader
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

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

# 訓練関数
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        pred_mean, pred_std, vis_logits = model(
            batch.x, batch.edge_index, edge_attr, batch.batch
        )
        
        loss = torch.nn.functional.mse_loss(pred_mean, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# 評価関数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            pred_mean, pred_std, vis_logits = model(
                batch.x, batch.edge_index, edge_attr, batch.batch
            )
            
            loss = torch.nn.functional.mse_loss(pred_mean, batch.y)
            mae = torch.abs(pred_mean - batch.y).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
    
    return total_loss / len(loader), total_mae / len(loader)

# 訓練ループ
print(f"\n{'='*70}")
print("Training...")
print(f"{'='*70}")

best_val_mae = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_mae = evaluate(model, val_loader, DEVICE)
    
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
        
        print(f"  ✓ Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

# テスト評価
print(f"\n{'='*70}")
print("Evaluating on test set...")
print(f"{'='*70}")

checkpoint = torch.load(MODEL_PATH / f"best_model_{test_pattern}.pt", 
                       map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_mae = evaluate(model, test_loader, DEVICE)

print(f"\n✓ Training Completed!")
print(f"  Best Val MAE: {best_val_mae:.2f} dB")
print(f"  Test MAE: {test_mae:.2f} dB")
print(f"  Model saved: {save_path}")

print(f"\n{'='*70}")
print("Next step:")
print(f"{'='*70}")
print("\nIf this works well, train all patterns:")
print("  python train_by_eye_pattern_revised.py")
print("\nOr test importance computation on this single model:")
print("  python compute_importance_simple.py")
print(f"\n{'='*70}")