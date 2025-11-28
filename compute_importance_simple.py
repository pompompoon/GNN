# -*- coding: utf-8 -*-
"""
compute_importance_simple.py
簡易版の重要度計算（デバッグ用）
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import sys

# パス設定
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "by_eye_pattern"
DATA_PATH = SCRIPT_DIR / "data" / "by_eye_pattern"
OUTPUT_PATH = SCRIPT_DIR / "results" / "importance_maps_by_eye_pattern"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = 'cpu'

print("="*70)
print("Simple Importance Map Computation (Debug Version)")
print("="*70)

print(f"\nPaths:")
print(f"  Script: {SCRIPT_DIR}")
print(f"  Models: {MODEL_PATH}")
print(f"  Data: {DATA_PATH}")
print(f"  Output: {OUTPUT_PATH}")

# モデルファイルの確認
model_files = list(MODEL_PATH.glob("best_model_*.pt"))
print(f"\nModel files found: {len(model_files)}")
for mf in model_files:
    print(f"  - {mf.name}")

if len(model_files) == 0:
    print("\n❌ ERROR: No model files found!")
    print(f"Please check: {MODEL_PATH}")
    sys.exit(1)

# データファイルの確認
data_files = list(DATA_PATH.glob("graph_data_*.pkl"))
print(f"\nData files found: {len(data_files)}")
for df in data_files:
    print(f"  - {df.name}")

if len(data_files) == 0:
    print("\n❌ ERROR: No data files found!")
    print(f"Please check: {DATA_PATH}")
    sys.exit(1)

# models_revised.pyをインポート
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from models_revised import create_model
    print("\n✓ models_revised.py imported successfully")
except Exception as e:
    print(f"\n❌ ERROR: Failed to import models_revised.py")
    print(f"   {e}")
    sys.exit(1)

# 簡易的な重要度計算（不確実性のみ）
def compute_simple_importance(model, graph_list, device=DEVICE):
    """簡易版の重要度計算（不確実性のみ）"""
    model.eval()
    all_uncertainties = []
    all_positions = []
    
    print("\nComputing importance (uncertainty-based)...")
    for graph in tqdm(graph_list[:10], desc="Processing"):  # 最初の10グラフのみ
        try:
            graph = graph.to(device)
            
            with torch.no_grad():
                edge_attr = graph.edge_attr if hasattr(graph, 'edge_attr') else None
                pred_mean, pred_std, vis_logits = model(
                    graph.x, graph.edge_index, edge_attr, None
                )
            
            uncertainty = pred_std.cpu().numpy()
            positions = graph.pos.cpu().numpy()
            
            all_uncertainties.append(uncertainty)
            all_positions.append(positions)
            
        except Exception as e:
            print(f"\n  Warning: Skipping graph due to error: {e}")
            continue
    
    if len(all_uncertainties) == 0:
        return None, None
    
    # 集約
    all_uncertainties = np.concatenate(all_uncertainties)
    all_positions = np.concatenate(all_positions, axis=0)
    
    # 位置でグループ化
    positions_rounded = np.round(all_positions, 1)
    unique_positions = np.unique(positions_rounded, axis=0)
    
    importance_scores = []
    importance_positions = []
    
    for pos in unique_positions:
        mask = np.all(np.isclose(positions_rounded, pos, atol=0.15), axis=1)
        if mask.sum() > 0:
            avg_uncertainty = all_uncertainties[mask].mean()
            importance_scores.append(avg_uncertainty)
            importance_positions.append(pos)
    
    return np.array(importance_positions), np.array(importance_scores)


# メイン処理
print("\n" + "="*70)
print("Processing each model...")
print("="*70)

results = []

for model_file in sorted(model_files):
    eye_pattern_name = model_file.stem.replace('best_model_', '')
    
    print(f"\n{'='*70}")
    print(f"Processing: {eye_pattern_name}")
    print(f"{'='*70}")
    
    try:
        # データ読み込み
        data_file = DATA_PATH / f"graph_data_{eye_pattern_name}.pkl"
        if not data_file.exists():
            print(f"  ❌ Data file not found: {data_file.name}")
            continue
        
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        n_features = pattern_data['n_features']
        edge_dim = pattern_data.get('edge_attr_dim', 2)
        
        print(f"  Data loaded: {len(graph_list)} graphs, {n_features} features")
        
        # モデル読み込み
        checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
        config = checkpoint.get('config', {})
        
        model = create_model(
            model_name='gat',
            in_channels=n_features,
            hidden_channels=config.get('HIDDEN_CHANNELS', 64),
            out_channels=2,
            num_layers=config.get('NUM_LAYERS', 3),
            dropout=config.get('DROPOUT', 0.3),
            edge_dim=edge_dim
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Model loaded")
        
        # 重要度計算
        positions, scores = compute_simple_importance(model, graph_list, DEVICE)
        
        if positions is None:
            print(f"  ❌ Importance calculation failed")
            continue
        
        # 正規化
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6) * 10.0
        
        # 保存（簡易版）
        importance_dict = {
            'combined': {
                'positions': positions,
                'scores': scores_norm,
                'scores_normalized': scores_norm
            }
        }
        
        pkl_file = OUTPUT_PATH / f"importance_map_{eye_pattern_name}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(importance_dict, f)
        
        print(f"  ✓ Saved: {pkl_file.name}")
        
        # CSV保存
        df = pd.DataFrame({
            'Position_X': positions[:, 0],
            'Position_Y': positions[:, 1],
            'ImportanceScore': scores_norm
        })
        df = df.sort_values('ImportanceScore', ascending=False)
        
        csv_file = OUTPUT_PATH / f"importance_map_{eye_pattern_name}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"  ✓ Saved: {csv_file.name}")
        print(f"\n  Top 5 Important Points:")
        print(df.head(5).to_string(index=False))
        
        results.append(eye_pattern_name)
        
    except Exception as e:
        print(f"\n  ❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*70}")
print(f"Completed! Processed {len(results)} patterns")
print(f"{'='*70}")

if len(results) > 0:
    print("\nSuccessfully processed:")
    for r in results:
        print(f"  ✓ {r}")
    print(f"\nOutput directory: {OUTPUT_PATH}")
else:
    print("\n❌ No patterns were successfully processed")
    print("Please check the errors above")