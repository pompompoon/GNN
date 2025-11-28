# -*- coding: utf-8 -*-
"""
train_all_patterns.py
全パターンの自動訓練（バッチなし版）
"""

import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
import time

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
print("Training All Patterns (No Batching)")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")

# 全パターンのリスト
ALL_PATTERNS = [
    "Left_Pattern30-2",
    "Left_Pattern24-2",
    "Left_Pattern10-2",
    "Right_Pattern30-2",
    "Right_Pattern24-2",
    "Right_Pattern10-2"
]


def train_epoch_no_batch(model, graphs, optimizer, device):
    """グラフごとに個別処理する訓練"""
    model.train()
    total_loss = 0
    n_valid = 0
    
    for graph in graphs:
        try:
            graph = graph.to(device)
            optimizer.zero_grad()
            
            edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
            
            pred_mean, pred_std, vis_logits = model(
                graph.x, graph.edge_index, edge_attr, batch=None
            )
            
            loss = torch.nn.functional.mse_loss(pred_mean, graph.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_valid += 1
            
        except Exception as e:
            continue
    
    if n_valid == 0:
        return float('inf')
    
    return total_loss / n_valid


def evaluate_no_batch(model, graphs, device):
    """グラフごとに個別評価"""
    model.eval()
    total_loss = 0
    total_mae = 0
    n_valid = 0
    
    with torch.no_grad():
        for graph in graphs:
            try:
                graph = graph.to(device)
                
                edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
                
                pred_mean, pred_std, vis_logits = model(
                    graph.x, graph.edge_index, edge_attr, batch=None
                )
                
                loss = torch.nn.functional.mse_loss(pred_mean, graph.y)
                mae = torch.abs(pred_mean - graph.y).mean()
                
                total_loss += loss.item()
                total_mae += mae.item()
                n_valid += 1
                
            except Exception as e:
                continue
    
    if n_valid == 0:
        return float('inf'), float('inf')
    
    return total_loss / n_valid, total_mae / n_valid


def train_single_pattern(pattern_name):
    """1つのパターンを訓練"""
    
    print(f"\n{'='*70}")
    print(f"Pattern: {pattern_name}")
    print(f"{'='*70}")
    
    data_file = DATA_PATH / f"graph_data_{pattern_name}.pkl"
    
    if not data_file.exists():
        print(f"  ✗ Data file not found: {data_file.name}")
        return None
    
    # データ読み込み
    try:
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        n_features = pattern_data['n_features']
        edge_dim = pattern_data.get('edge_attr_dim', 2)
        
        print(f"  Graphs: {len(graph_list)}")
        print(f"  Features: {n_features}")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return None
    
    # データ分割
    n_graphs = len(graph_list)
    train_size = int(0.7 * n_graphs)
    val_size = int(0.15 * n_graphs)
    
    train_graphs = graph_list[:train_size]
    val_graphs = graph_list[train_size:train_size+val_size]
    test_graphs = graph_list[train_size+val_size:]
    
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
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
    
    # 訓練ループ
    best_val_mae = float('inf')
    patience_counter = 0
    
    print(f"\n  Training...")
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch_no_batch(model, train_graphs, optimizer, DEVICE)
        val_loss, val_mae = evaluate_no_batch(model, val_graphs, DEVICE)
        
        # 10エポックごとに表示
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} dB")
        
        # Early stopping & モデル保存
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            
            # 保存
            save_path = MODEL_PATH / f"best_model_{pattern_name}.pt"
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
            
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # テスト評価
    test_loss, test_mae = evaluate_no_batch(model, test_graphs, DEVICE)
    
    print(f"\n  ✓ Completed!")
    print(f"    Best Val MAE: {best_val_mae:.2f} dB")
    print(f"    Test MAE: {test_mae:.2f} dB")
    
    return {
        'pattern': pattern_name,
        'best_val_mae': best_val_mae,
        'test_mae': test_mae,
        'n_train': len(train_graphs),
        'n_val': len(val_graphs),
        'n_test': len(test_graphs)
    }


# メイン処理
print(f"\n{'='*70}")
print(f"Training {len(ALL_PATTERNS)} patterns")
print(f"{'='*70}")

results = []
start_time = time.time()

for i, pattern_name in enumerate(ALL_PATTERNS, 1):
    print(f"\n[{i}/{len(ALL_PATTERNS)}]")
    
    result = train_single_pattern(pattern_name)
    
    if result is not None:
        results.append(result)
    
    # 進捗表示
    elapsed = time.time() - start_time
    if i < len(ALL_PATTERNS):
        avg_time = elapsed / i
        remaining = avg_time * (len(ALL_PATTERNS) - i)
        print(f"\n  Elapsed: {elapsed/60:.1f} min | "
              f"Estimated remaining: {remaining/60:.1f} min")

# 最終サマリー
total_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"All Training Completed!")
print(f"{'='*70}")
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Successfully trained: {len(results)} / {len(ALL_PATTERNS)} patterns\n")

if len(results) > 0:
    print("Results Summary:")
    print("-" * 70)
    print(f"{'Pattern':<25} {'Val MAE':>10} {'Test MAE':>10} {'Graphs':>10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['pattern']:<25} "
              f"{result['best_val_mae']:>9.2f}  "
              f"{result['test_mae']:>9.2f}  "
              f"{result['n_train']+result['n_val']+result['n_test']:>9}")
    
    print("-" * 70)
    
    avg_val_mae = sum(r['best_val_mae'] for r in results) / len(results)
    avg_test_mae = sum(r['test_mae'] for r in results) / len(results)
    
    print(f"{'Average':<25} {avg_val_mae:>9.2f}  {avg_test_mae:>9.2f}")
    print("=" * 70)

print(f"\n✓ Models saved in: {MODEL_PATH}")
print(f"\nNext steps:")
print(f"  1. Compute importance maps:")
print(f"     python compute_importance_simple.py")
print(f"  2. Visualize:")
print(f"     python visualize_importance_by_eye_pattern.py")
print(f"\n{'='*70}")