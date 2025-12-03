# -*- coding: utf-8 -*-
"""
create_graph_by_eye_pattern_fixed.py
HFAパターン情報を正しく読み込んでグラフデータを作成（修正版）

修正内容:
- 入力特徴量（座標）の正規化
- 教師データ（感度）の範囲チェックとクリッピング
- データ品質の検証
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# データパス（Googleドライブのパス）
DATA_ROOT = Path("G:/共有ドライブ/GAP_Analysis/Data/GAP2_KyodaiClinical")
OUTPUT_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern_correct"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nPaths:")
print(f"  Project: {GNN_PROJECT_PATH}")
print(f"  Data Root: {DATA_ROOT}")
print(f"  Output: {OUTPUT_PATH}")

# パターンマッピング
PATTERN_NAME_MAP = {
    '中心30-2 閾値テスト': 'Pattern30-2',
    '中心24-2 閾値テスト': 'Pattern24-2',
    '中心10-2 閾値テスト': 'Pattern10-2',
}

PATTERN_ID_MAP = {
    'Pattern30-2': 0,
    'Pattern24-2': 1,
    'Pattern10-2': 2,
}

EYE_MAP = {
    0: 'Left',
    1: 'Right'
}

# ★ 視野感度の有効範囲
SENSITIVITY_MIN = 0.0
SENSITIVITY_MAX = 40.0

# ★ 座標の正規化パラメータ（パターンごと）
COORD_NORMALIZE_PARAMS = {
    'Pattern30-2': {'x_max': 27.0, 'y_max': 24.0},
    'Pattern24-2': {'x_max': 21.0, 'y_max': 27.0},
    'Pattern10-2': {'x_max': 9.0, 'y_max': 7.0},
}


def load_response_data(subject_folder):
    """
    response*.csvからマリオット盲点情報を読み込む
    """
    response_files = list(subject_folder.glob("response_*.csv"))
    
    if len(response_files) == 0:
        return None
    
    response_file = response_files[0]
    
    try:
        df_response = pd.read_csv(response_file)
        
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'IsMariotte']
        
        if not all(col in df_response.columns for col in required_cols):
            return None
        
        if 'IsInspectionPointer' in df_response.columns:
            df_response = df_response[df_response['IsInspectionPointer'] == True].copy()
        
        return df_response
        
    except Exception as e:
        print(f"  Error reading {response_file.name}: {e}")
        return None


def load_hfa_data(subject_folder):
    """
    HFAデータとオプション情報を読み込む
    """
    hfa_folder = subject_folder / "HFAMatchData"
    
    if not hfa_folder.exists():
        return None, None
    
    hfa_result_files = list(hfa_folder.glob("hfa_result_*.csv"))
    hfa_option_files = list(hfa_folder.glob("hfa_option_*.csv"))
    
    if len(hfa_result_files) == 0 or len(hfa_option_files) == 0:
        return None, None
    
    hfa_result_file = hfa_result_files[0]
    hfa_option_file = hfa_option_files[0]
    
    try:
        df_hfa = pd.read_csv(hfa_result_file)
        
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'Sensitivity']
        
        if not all(col in df_hfa.columns for col in required_cols):
            return None, None
        
        df_option = pd.read_csv(hfa_option_file)
        
        if 'TestPattern' not in df_option.columns:
            print(f"  Warning: TestPattern column not found in {hfa_option_file.name}")
            return None, None
        
        test_pattern = df_option['TestPattern'].iloc[0]
        
        return df_hfa, test_pattern
        
    except Exception as e:
        print(f"  Error reading HFA data: {e}")
        return None, None


def match_mariotte_flags(df_hfa, df_response):
    """
    HFAデータとresponseデータを座標でマッチングし、マリオット盲点フラグを設定
    """
    df_hfa = df_hfa.copy()
    df_hfa['IsMariotte'] = False
    
    for idx, row in df_hfa.iterrows():
        eye_side = row['EyeSide']
        x = row['InspectionAngleX']
        y = row['InspectionAngleY']
        
        match_mask = (
            (df_response['EyeSide'] == eye_side) &
            (np.abs(df_response['InspectionAngleX'] - x) <= 1) &
            (np.abs(df_response['InspectionAngleY'] - y) <= 1)
        )
        
        matched_points = df_response[match_mask]
        
        if len(matched_points) > 0:
            is_mariotte = matched_points['IsMariotte'].iloc[0]
            df_hfa.at[idx, 'IsMariotte'] = bool(is_mariotte == 1 or is_mariotte == True)
    
    return df_hfa


def validate_sensitivity(sensitivities, subject_id=""):
    """
    ★ 感度データの検証とクリッピング
    """
    original_count = len(sensitivities)
    
    # NaN/Infチェック
    valid_mask = np.isfinite(sensitivities)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Warning [{subject_id}]: {n_invalid} invalid values (NaN/Inf)")
    
    # 範囲外の値をチェック
    out_of_range = (sensitivities < SENSITIVITY_MIN) | (sensitivities > SENSITIVITY_MAX)
    if out_of_range.any():
        n_out = out_of_range.sum()
        min_val = sensitivities.min()
        max_val = sensitivities.max()
        print(f"  Warning [{subject_id}]: {n_out} values out of range [{SENSITIVITY_MIN}, {SENSITIVITY_MAX}]")
        print(f"    Actual range: [{min_val:.1f}, {max_val:.1f}]")
    
    # クリッピング
    sensitivities_clipped = np.clip(sensitivities, SENSITIVITY_MIN, SENSITIVITY_MAX)
    
    # NaNを中央値で置換
    if not valid_mask.all():
        median_val = np.nanmedian(sensitivities_clipped)
        sensitivities_clipped[~valid_mask] = median_val
    
    return sensitivities_clipped


def normalize_coordinates(positions, pattern_name):
    """
    ★ 座標を[-1, 1]の範囲に正規化
    """
    params = COORD_NORMALIZE_PARAMS.get(pattern_name, {'x_max': 30.0, 'y_max': 30.0})
    
    normalized = positions.copy()
    normalized[:, 0] = positions[:, 0] / params['x_max']  # X座標
    normalized[:, 1] = positions[:, 1] / params['y_max']  # Y座標
    
    # クリッピング（念のため）
    normalized = np.clip(normalized, -1.5, 1.5)
    
    return normalized


def compute_edge_attributes(positions, sensitivities=None):
    """
    エッジ属性を計算: [距離の逆数, 感度類似性（オプション）]
    """
    n_points = len(positions)
    
    if n_points < 2:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
        return edge_index, edge_attr
    
    k = min(6, n_points - 1)
    
    edge_list = []
    edge_attrs = []
    
    for i in range(n_points):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        distances[i] = np.inf
        
        nearest_indices = np.argsort(distances)[:k]
        
        for j in nearest_indices:
            edge_list.append([i, j])
            
            # 距離に基づく重み（正規化済み座標なので調整）
            dist = distances[j]
            dist_weight = 1.0 / (1.0 + dist * 10)  # スケール調整
            
            # 感度類似性（オプション）
            if sensitivities is not None:
                sens_diff = abs(sensitivities[i] - sensitivities[j])
                sens_similarity = 1.0 / (1.0 + sens_diff / 10.0)
            else:
                # 角度に基づく類似性
                vec_i = positions[i] / (np.linalg.norm(positions[i]) + 1e-6)
                vec_j = positions[j] / (np.linalg.norm(positions[j]) + 1e-6)
                sens_similarity = (np.dot(vec_i, vec_j) + 1) / 2
            
            edge_attrs.append([dist_weight, sens_similarity])
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr


def create_graph_for_eye(df_hfa, eye_side, pattern_name, subject_id=""):
    """
    特定の眼のグラフを作成（正規化・検証付き）
    """
    # 指定の眼・マリオット盲点でない点のみ
    mask = (
        (df_hfa['EyeSide'] == eye_side) &
        (~df_hfa['IsMariotte'])
    )
    
    df_subset = df_hfa[mask].copy()
    
    if len(df_subset) < 5:
        return None
    
    # データ抽出
    positions_raw = df_subset[['InspectionAngleX', 'InspectionAngleY']].values.astype(np.float32)
    sensitivities_raw = df_subset['Sensitivity'].values.astype(np.float32)
    
    # ★ 感度の検証とクリッピング
    sensitivities = validate_sensitivity(sensitivities_raw, subject_id)
    
    # ★ 座標の正規化
    positions_normalized = normalize_coordinates(positions_raw, pattern_name)
    
    # 特徴量: 正規化座標 + 追加特徴量
    # 中心からの距離（正規化済み）
    distance_from_center = np.sqrt(positions_normalized[:, 0]**2 + positions_normalized[:, 1]**2)
    distance_from_center = distance_from_center.reshape(-1, 1)
    
    # 特徴量を結合: [norm_x, norm_y, distance]
    features = np.concatenate([positions_normalized, distance_from_center], axis=1).astype(np.float32)
    
    # エッジとエッジ属性を計算（正規化座標を使用）
    edge_index, edge_attr = compute_edge_attributes(positions_normalized, sensitivities)
    
    # PyG Dataオブジェクト作成
    graph = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(sensitivities, dtype=torch.float32),
        pos=torch.tensor(positions_raw, dtype=torch.float32),  # 元の座標も保持
        pos_normalized=torch.tensor(positions_normalized, dtype=torch.float32),
        eye_side=eye_side
    )
    
    return graph


def process_all_subjects():
    """すべての被験者データを処理"""
    
    if not DATA_ROOT.exists():
        print(f"\n❌ Data root not found: {DATA_ROOT}")
        return
    
    subject_folders = [f for f in DATA_ROOT.iterdir() if f.is_dir()]
    
    print(f"\nFound {len(subject_folders)} subject folders")
    
    # 眼・パターンごとにグラフを集約
    graphs_by_eye_pattern = {
        f"{EYE_MAP[eye]}_{pattern}": []
        for eye in [0, 1]
        for pattern in ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    }
    
    # 統計情報
    total_points = 0
    total_mariotte_excluded = 0
    pattern_counts = {}
    sensitivity_stats = {'min': float('inf'), 'max': float('-inf'), 'values': []}
    
    # 各被験者を処理
    for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
        subject_id = subject_folder.name
        
        # responseデータを読み込み
        df_response = load_response_data(subject_folder)
        
        if df_response is None:
            continue
        
        # HFAデータとパターン情報を読み込み
        df_hfa, test_pattern = load_hfa_data(subject_folder)
        
        if df_hfa is None or test_pattern is None:
            continue
        
        # TestPatternをパターン名に変換
        if test_pattern not in PATTERN_NAME_MAP:
            print(f"  Warning: Unknown test pattern: {test_pattern}")
            continue
        
        pattern_name = PATTERN_NAME_MAP[test_pattern]
        
        # 統計
        pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        # マリオット盲点フラグをマッチング
        df_hfa = match_mariotte_flags(df_hfa, df_response)
        
        total_points += len(df_hfa)
        total_mariotte_excluded += df_hfa['IsMariotte'].sum()
        
        # 感度統計を収集
        sens_values = df_hfa['Sensitivity'].values
        sensitivity_stats['min'] = min(sensitivity_stats['min'], np.nanmin(sens_values))
        sensitivity_stats['max'] = max(sensitivity_stats['max'], np.nanmax(sens_values))
        sensitivity_stats['values'].extend(sens_values[np.isfinite(sens_values)].tolist())
        
        # 各眼でグラフ作成
        for eye_side in [0, 1]:
            graph = create_graph_for_eye(df_hfa, eye_side, pattern_name, subject_id)
            
            if graph is not None:
                key = f"{EYE_MAP[eye_side]}_{pattern_name}"
                graphs_by_eye_pattern[key].append(graph)
    
    # 統計情報表示
    print(f"\n{'='*70}")
    print("Overall Statistics")
    print(f"{'='*70}")
    print(f"Total points processed: {total_points}")
    print(f"Mariotte blind spot points excluded: {total_mariotte_excluded}")
    print(f"Functional points used: {total_points - total_mariotte_excluded}")
    if total_points > 0:
        print(f"Exclusion rate: {100 * total_mariotte_excluded / total_points:.1f}%")
    
    print(f"\n★ Sensitivity Statistics:")
    print(f"  Min: {sensitivity_stats['min']:.1f} dB")
    print(f"  Max: {sensitivity_stats['max']:.1f} dB")
    if sensitivity_stats['values']:
        print(f"  Mean: {np.mean(sensitivity_stats['values']):.1f} dB")
        print(f"  Std: {np.std(sensitivity_stats['values']):.1f} dB")
    
    print(f"\nPattern distribution:")
    for pattern_name, count in sorted(pattern_counts.items()):
        print(f"  {pattern_name}: {count} subjects")
    
    # 結果を保存
    print(f"\n{'='*70}")
    print("Saving graph data by eye and pattern")
    print(f"{'='*70}")
    
    for key, graph_list in graphs_by_eye_pattern.items():
        if len(graph_list) == 0:
            print(f"\n{key}: No data (skipped)")
            continue
        
        print(f"\n{key}: {len(graph_list)} graphs")
        
        # 統計情報
        n_features = graph_list[0].x.shape[1]
        avg_nodes = np.mean([g.num_nodes for g in graph_list])
        avg_edges = np.mean([g.edge_index.shape[1] for g in graph_list])
        
        # ★ 感度の範囲確認
        all_sensitivities = []
        for g in graph_list:
            all_sensitivities.extend(g.y.numpy().tolist())
        sens_min = np.min(all_sensitivities)
        sens_max = np.max(all_sensitivities)
        sens_mean = np.mean(all_sensitivities)
        
        print(f"  Features: {n_features} (normalized coords + distance)")
        print(f"  Avg nodes per graph: {avg_nodes:.1f}")
        print(f"  Avg edges per graph: {avg_edges:.1f}")
        print(f"  ★ Sensitivity range: [{sens_min:.1f}, {sens_max:.1f}] dB")
        print(f"  ★ Sensitivity mean: {sens_mean:.1f} dB")
        
        # Pickle形式で保存
        eye_name, pattern_name = key.split('_', 1)
        pattern_id = PATTERN_ID_MAP[pattern_name]
        
        pattern_data = {
            'graph_list': graph_list,
            'n_features': n_features,
            'eye_side': int(eye_name == 'Right'),
            'pattern_id': pattern_id,
            'pattern_name': pattern_name,
            'edge_attr_dim': 2,
            'feature_names': ['NormX', 'NormY', 'Distance'],
            'mariotte_excluded': True,
            'sensitivity_in_features': False,
            'coord_normalized': True,
            'sensitivity_range': [SENSITIVITY_MIN, SENSITIVITY_MAX]
        }
        
        output_file = OUTPUT_PATH / f"graph_data_{key}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(pattern_data, f)
        
        print(f"  ✓ Saved: {output_file.name}")
    
    print(f"\n{'='*70}")
    print("Graph Data Creation Completed!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"\n★ Key improvements:")
    print(f"  - Coordinates normalized to [-1, 1]")
    print(f"  - Sensitivity clipped to [{SENSITIVITY_MIN}, {SENSITIVITY_MAX}] dB")
    print(f"  - Features: [NormX, NormY, Distance from center]")


# メイン処理
if __name__ == "__main__":
    print("="*70)
    print("Creating Graph Data by Eye and Pattern (FIXED VERSION)")
    print("★ Coordinates NORMALIZED")
    print("★ Sensitivity VALIDATED and CLIPPED")
    print("="*70)
    
    process_all_subjects()