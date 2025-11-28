# -*- coding: utf-8 -*-
"""
compute_importance_by_eye_pattern.py
左右眼別・パターン別の重要度マップを計算（簡略化版）

修正内容:
- マリオット盲点は既にグラフから除外済み
- 全ての点が機能点として扱われる
"""

import torch
import torch.nn.functional as F
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
GNN_PROJECT_PATH = SCRIPT_DIR

MODEL_PATH = GNN_PROJECT_PATH / "models" / "by_eye_pattern"
DATA_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern"
OUTPUT_PATH = GNN_PROJECT_PATH / "results" / "importance_maps_by_eye_pattern"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nPaths:")
print(f"  Project: {GNN_PROJECT_PATH}")
print(f"  Models: {MODEL_PATH}")
print(f"  Data: {DATA_PATH}")
print(f"  Output: {OUTPUT_PATH}")

sys.path.insert(0, str(SCRIPT_DIR))

from models_revised import create_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# 重要度計算の設定
N_SAMPLES_FOR_IMPORTANCE = 20
IMPORTANCE_METHODS = ['uncertainty', 'error', 'leave_one_out', 'combined']


@torch.no_grad()
def predict_with_model(model, graph_data, device=DEVICE):
    """グラフデータから予測を実行"""
    model.eval()
    data = graph_data.to(device)
    
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    batch = data.batch if hasattr(data, 'batch') else None
    
    pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, edge_attr, batch)
    
    return {
        'sensitivity': pred_mean.cpu().numpy(),
        'uncertainty': pred_std.cpu().numpy(),
        'vis_logits': vis_logits.cpu().numpy()
    }


@torch.no_grad()
def predict_without_point(model, graph_data, exclude_idx, device=DEVICE):
    """特定の点を除外して予測"""
    model.eval()
    
    data = graph_data.clone()
    data.x = data.x.clone()
    data.x[exclude_idx] = 0.0
    
    data = data.to(device)
    
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
    batch = data.batch if hasattr(data, 'batch') else None
    
    pred_mean, pred_std, vis_logits = model(data.x, data.edge_index, edge_attr, batch)
    
    return {
        'sensitivity': pred_mean.cpu().numpy(),
        'uncertainty': pred_std.cpu().numpy()
    }


def compute_uncertainty_importance(model, graph_list, device=DEVICE):
    """不確実性ベースの重要度"""
    print("\nComputing uncertainty-based importance...")
    
    all_uncertainties = []
    all_positions = []
    
    for graph in tqdm(graph_list, desc="Uncertainty"):
        pred = predict_with_model(model, graph, device)
        
        uncertainty = pred['uncertainty']
        positions = graph.pos.cpu().numpy()
        
        all_uncertainties.append(uncertainty)
        all_positions.append(positions)
    
    all_uncertainties = np.concatenate(all_uncertainties)
    all_positions = np.concatenate(all_positions, axis=0)
    
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


def compute_error_importance(model, graph_list, device=DEVICE):
    """誤差ベースの重要度"""
    print("\nComputing error-based importance...")
    
    all_errors = []
    all_positions = []
    
    for graph in tqdm(graph_list, desc="Error"):
        if not hasattr(graph, 'y') or graph.y is None:
            continue
        
        pred = predict_with_model(model, graph, device)
        
        ground_truth = graph.y.cpu().numpy()
        predicted = pred['sensitivity']
        
        abs_error = np.abs(predicted - ground_truth)
        positions = graph.pos.cpu().numpy()
        
        all_errors.append(abs_error)
        all_positions.append(positions)
    
    if len(all_errors) == 0:
        return None, None
    
    all_errors = np.concatenate(all_errors)
    all_positions = np.concatenate(all_positions, axis=0)
    
    positions_rounded = np.round(all_positions, 1)
    unique_positions = np.unique(positions_rounded, axis=0)
    
    importance_scores = []
    importance_positions = []
    
    for pos in unique_positions:
        mask = np.all(np.isclose(positions_rounded, pos, atol=0.15), axis=1)
        if mask.sum() > 0:
            avg_error = all_errors[mask].mean()
            importance_scores.append(avg_error)
            importance_positions.append(pos)
    
    return np.array(importance_positions), np.array(importance_scores)


def compute_leave_one_out_importance(model, graph_list, device=DEVICE, max_graphs=5):
    """Leave-one-out方式の重要度"""
    print("\nComputing leave-one-out importance...")
    
    sample_graphs = graph_list[:max_graphs]
    point_importance_scores = {}
    
    for graph_idx, graph in enumerate(tqdm(sample_graphs, desc="Leave-one-out")):
        if not hasattr(graph, 'y') or graph.y is None:
            continue
        
        pred_base = predict_with_model(model, graph, device)
        ground_truth = graph.y.cpu().numpy()
        base_mae = np.mean(np.abs(pred_base['sensitivity'] - ground_truth))
        
        positions = graph.pos.cpu().numpy()
        
        for point_idx in range(graph.num_nodes):
            pred_without = predict_without_point(model, graph, point_idx, device)
            
            mae_without = np.mean(np.abs(pred_without['sensitivity'] - ground_truth))
            mae_increase = mae_without - base_mae
            
            pos = tuple(np.round(positions[point_idx], 1))
            
            if pos not in point_importance_scores:
                point_importance_scores[pos] = []
            
            point_importance_scores[pos].append(mae_increase)
    
    importance_positions = []
    importance_scores = []
    
    for pos, scores in point_importance_scores.items():
        importance_positions.append(list(pos))
        importance_scores.append(np.mean(scores))
    
    return np.array(importance_positions), np.array(importance_scores)


def normalize_scores(scores):
    """スコアを0-10の範囲に正規化"""
    if scores is None or len(scores) == 0:
        return scores
    
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score - min_score < 1e-6:
        return np.ones_like(scores) * 5.0
    
    normalized = (scores - min_score) / (max_score - min_score) * 10.0
    return normalized


def compute_combined_importance(importance_dict):
    """複数の重要度指標を統合"""
    weights = {
        'uncertainty': 0.4,
        'error': 0.4,
        'leave_one_out': 0.2
    }
    
    all_positions = []
    for method in importance_dict.keys():
        if importance_dict[method]['positions'] is not None:
            all_positions.append(importance_dict[method]['positions'])
    
    if len(all_positions) == 0:
        return None, None
    
    all_positions = np.concatenate(all_positions, axis=0)
    unique_positions = np.unique(np.round(all_positions, 1), axis=0)
    
    combined_scores = []
    combined_positions = []
    
    for pos in unique_positions:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, weight in weights.items():
            if method not in importance_dict or importance_dict[method]['positions'] is None:
                continue
            
            method_positions = importance_dict[method]['positions']
            method_scores = importance_dict[method]['scores_normalized']
            
            mask = np.all(np.isclose(method_positions, pos, atol=0.15), axis=1)
            
            if mask.sum() > 0:
                score = method_scores[mask].mean()
                weighted_sum += weight * score
                total_weight += weight
        
        if total_weight > 0:
            combined_scores.append(weighted_sum / total_weight)
            combined_positions.append(pos)
    
    return np.array(combined_positions), np.array(combined_scores)


def compute_importance_for_eye_pattern(eye_pattern_name, model, graph_list, device=DEVICE):
    """眼・パターン専用の重要度マップを計算"""
    print(f"\n{'='*60}")
    print(f"Computing Importance Map: {eye_pattern_name}")
    print(f"★ All points are functional (Mariotte excluded)")
    print(f"{'='*60}")
    
    n_samples = min(N_SAMPLES_FOR_IMPORTANCE, len(graph_list))
    sample_graphs = graph_list[:n_samples]
    
    print(f"Using {n_samples} graphs for importance computation")
    
    importance_dict = {}
    
    # 1. 不確実性ベース
    positions_unc, scores_unc = compute_uncertainty_importance(model, sample_graphs, device)
    importance_dict['uncertainty'] = {
        'positions': positions_unc,
        'scores': scores_unc,
        'scores_normalized': normalize_scores(scores_unc)
    }
    
    # 2. 誤差ベース
    positions_err, scores_err = compute_error_importance(model, sample_graphs, device)
    importance_dict['error'] = {
        'positions': positions_err,
        'scores': scores_err,
        'scores_normalized': normalize_scores(scores_err) if scores_err is not None else None
    }
    
    # 3. Leave-one-out
    positions_loo, scores_loo = compute_leave_one_out_importance(
        model, sample_graphs, device, max_graphs=5
    )
    importance_dict['leave_one_out'] = {
        'positions': positions_loo,
        'scores': scores_loo,
        'scores_normalized': normalize_scores(scores_loo)
    }
    
    # 4. 統合スコア
    positions_combined, scores_combined = compute_combined_importance(importance_dict)
    importance_dict['combined'] = {
        'positions': positions_combined,
        'scores': scores_combined,
        'scores_normalized': scores_combined
    }
    
    return importance_dict


def save_importance_results(eye_pattern_name, importance_dict, output_path):
    """重要度結果を保存"""
    
    # Pickle形式で保存
    pkl_file = output_path / f"importance_map_{eye_pattern_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(importance_dict, f)
    print(f"\nSaved (pickle): {pkl_file}")
    
    # CSV形式で保存
    if importance_dict['combined']['positions'] is not None:
        positions = importance_dict['combined']['positions']
        scores = importance_dict['combined']['scores']
        
        df = pd.DataFrame({
            'Position_X': positions[:, 0],
            'Position_Y': positions[:, 1],
            'ImportanceScore': scores
        })
        
        df = df.sort_values('ImportanceScore', ascending=False)
        
        csv_file = output_path / f"importance_map_{eye_pattern_name}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Saved (CSV): {csv_file}")
        
        print(f"\nTop 10 Most Important Points:")
        print(df.head(10).to_string(index=False))


# メイン処理
if __name__ == "__main__":
    print("="*60)
    print("Computing Importance Maps by Eye and Pattern")
    print("★ Mariotte blind spot points are already EXCLUDED")
    print("Teacher data: HFA_Sensitivity (functional points only)")
    print("Edge attributes: [distance_weight, sensitivity_similarity]")
    print("="*60)
    
    model_files = list(MODEL_PATH.glob("best_model_*.pt"))
    
    if len(model_files) == 0:
        print(f"\nError: No trained models found in {MODEL_PATH}")
        print("Please run train_by_eye_pattern_revised.py first")
        exit(1)
    
    print(f"\nFound {len(model_files)} trained models")
    
    for model_file in sorted(model_files):
        eye_pattern_name = model_file.stem.replace('best_model_', '')
        
        print(f"\n{'='*60}")
        print(f"Processing: {eye_pattern_name}")
        print(f"{'='*60}")
        
        try:
            checkpoint = torch.load(model_file, map_location=DEVICE, weights_only=False)
            config = checkpoint.get('config', {})
            
            print(f"Checkpoint info:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Val MAE: {checkpoint.get('val_mae', 'N/A'):.2f}")
            print(f"  Edge dim: {config.get('EDGE_DIM', 'N/A')}")
            
            # データ読み込み
            data_file = DATA_PATH / f"graph_data_{eye_pattern_name}.pkl"
            if not data_file.exists():
                print(f"Warning: Data file not found: {data_file}")
                continue
            
            with open(data_file, 'rb') as f:
                pattern_data = pickle.load(f)
            
            n_features = pattern_data['n_features']
            graph_list = pattern_data['graph_list']
            edge_dim = pattern_data.get('edge_attr_dim', 2)
            
            print(f"Data loaded:")
            print(f"  Features: {n_features}")
            print(f"  Graphs: {len(graph_list)}")
            print(f"  Edge dim: {edge_dim}")
            print(f"  Mariotte excluded: {pattern_data.get('mariotte_excluded', False)}")
            
            # モデル初期化
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
            model.eval()
            
            print(f"✓ Model loaded successfully")
            
            # 重要度計算
            importance_dict = compute_importance_for_eye_pattern(
                eye_pattern_name, model, graph_list, DEVICE
            )
            
            # 保存
            save_importance_results(eye_pattern_name, importance_dict, OUTPUT_PATH)
        
        except Exception as e:
            print(f"\n✗ Error processing {eye_pattern_name}:")
            print(f"  {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All Importance Maps Computed!")
    print(f"{'='*60}")
    print(f"Results saved in: {OUTPUT_PATH}")